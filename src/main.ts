import * as THREE from 'three/webgpu';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { bloom } from 'three/addons/tsl/display/BloomNode.js';
import {
	Fn, If, vec3, float, uint, uv, uniform, instancedArray, instanceIndex,
	atan, smoothstep, fwidth, wgslFn, pass,
} from 'three/tsl';

// ---------------------------------------------------------------------------
// Parameters (ported from ParticleLogic.js)
// ---------------------------------------------------------------------------

const PARTICLE_COUNT = 1_000_000;
const LIFESPAN = 1000;        // frames
const NOISE_SCALE = 10.3;      // spatial frequency of the curl field
const SPEED = 0.01;           // world units per frame
const CURL_EPS = 0.01;        // central-difference step for the curl
const PARTICLE_SIZE = 0.006;  // sprite diameter in world units

// Post-processing, ported from the reference App.js.
const EXPOSURE = 0.1;         // additive blending blows way past 1, so pull it back hard
const BLOOM_STRENGTH = 0.2;
const BLOOM_RADIUS = 0.04;
const BLOOM_THRESHOLD = 3.0;  // only genuinely dense clumps glow

/** JS number -> WGSL f32 literal (`1` is an integer literal in WGSL, `1.0` is not). */
const f32 = (n: number) => (Number.isInteger(n) ? `${n}.0` : `${n}`);

/** wgslFn() is typed as returning a bare Node, which blocks method chaining. */
const wgsl = (code: string, includes: any[] = []) =>
	wgslFn(code, includes) as unknown as (args: Record<string, any>) => any;

// ---------------------------------------------------------------------------
// Shader math, in plain WGSL. Each helper is a normal WGSL function; the second
// argument to wgsl() lists the functions it calls so they get emitted too.
// ---------------------------------------------------------------------------

/** Lattice hash: vec3 -> [0,1) */
const hash31 = wgsl(`
fn hash31( p: vec3f ) -> f32 {
	return fract( sin( dot( p, vec3f( 127.1, 311.7, 74.7 ) ) ) * 43758.5453123 );
}`);

/** Trilinearly interpolated value noise, in [-1,1]. Replaces the simplex noise. */
const valueNoise = wgsl(`
fn valueNoise( p: vec3f ) -> f32 {
	let i = floor( p );
	let f = fract( p );
	let u = f * f * ( 3.0 - 2.0 * f );

	let n000 = hash31( i );
	let n100 = hash31( i + vec3f( 1, 0, 0 ) );
	let n010 = hash31( i + vec3f( 0, 1, 0 ) );
	let n110 = hash31( i + vec3f( 1, 1, 0 ) );
	let n001 = hash31( i + vec3f( 0, 0, 1 ) );
	let n101 = hash31( i + vec3f( 1, 0, 1 ) );
	let n011 = hash31( i + vec3f( 0, 1, 1 ) );
	let n111 = hash31( i + vec3f( 1, 1, 1 ) );

	let n00 = mix( n000, n100, u.x );
	let n10 = mix( n010, n110, u.x );
	let n01 = mix( n001, n101, u.x );
	let n11 = mix( n011, n111, u.x );

	return 2.0 * mix( mix( n00, n10, u.y ), mix( n01, n11, u.y ), u.z ) - 1.0;
}`, [hash31]);

/**
 * Curl of the vector field F(p) = ( N(p), N(p.yzx + 100), N(p.zxy + 200) ),
 * by central differences -- same construction as the reference implementation.
 */
const curlNoise = wgsl(`
fn curlNoise( p: vec3f ) -> vec3f {
	let e = ${f32(CURL_EPS)};
	let k = 1.0 / ( 2.0 * e );

	let x = p.x;
	let y = p.y;
	let z = p.z;

	let dFz_dy = ( valueNoise( vec3f( z + 200, x + 200, y + 200 + e ) ) - valueNoise( vec3f( z + 200, x + 200, y + 200 - e ) ) ) * k;
	let dFy_dz = ( valueNoise( vec3f( y + 100, z + 100 + e, x + 100 ) ) - valueNoise( vec3f( y + 100, z + 100 - e, x + 100 ) ) ) * k;

	let dFx_dz = ( valueNoise( vec3f( x, y, z + e ) ) - valueNoise( vec3f( x, y, z - e ) ) ) * k;
	let dFz_dx = ( valueNoise( vec3f( z + 200 + e, x + 200, y + 200 ) ) - valueNoise( vec3f( z + 200 - e, x + 200, y + 200 ) ) ) * k;

	let dFy_dx = ( valueNoise( vec3f( y + 100 + e, z + 100, x + 100 ) ) - valueNoise( vec3f( y + 100 - e, z + 100, x + 100 ) ) ) * k;
	let dFx_dy = ( valueNoise( vec3f( x, y + e, z ) ) - valueNoise( vec3f( x, y - e, z ) ) ) * k;

	return vec3f( dFz_dy - dFy_dz, dFx_dz - dFz_dx, dFy_dx - dFx_dy );
}`, [valueNoise]);

/** Fully saturated hue -> RGB (equivalent to HSL with s = 1, l = 0.5). */
const hue2rgb = wgsl(`
fn hue2rgb( h: f32 ) -> vec3f {
	let k = ( h * 6.0 + vec3f( 0, 4, 2 ) ) % 6.0;
	return saturate( abs( k - 3.0 ) - 1.0 );
}`);

/** PCG hash: u32 -> [0,1) */
const pcg = wgsl(`
fn pcg( seed: u32 ) -> f32 {
	let state = seed * 747796405u + 2891336453u;
	let word = ( ( state >> ( ( state >> 28u ) + 4u ) ) ^ state ) * 277803737u;
	return f32( ( word >> 22u ) ^ word ) / 4294967296.0;
}`);

/** A fresh particle: xyz is a point in the unit cube at the origin, w is a lifetime. */
const spawn = wgsl(`
fn spawn( seed: u32 ) -> vec4f {
	return vec4f(
		pcg( seed ) - 0.5,
		pcg( seed + 1u ) - 0.5,
		pcg( seed + 2u ) - 0.5,
		pcg( seed + 3u ) * ${f32(LIFESPAN)}
	);
}`, [pcg]);

// ---------------------------------------------------------------------------
// GPU state: lives in storage buffers, never round-trips to the CPU
// ---------------------------------------------------------------------------

const positionBuffer = instancedArray(PARTICLE_COUNT, 'vec3');
const colorBuffer = instancedArray(PARTICLE_COUNT, 'vec3');
const lifeBuffer = instancedArray(PARTICLE_COUNT, 'float');

const frame = uniform(0, 'uint');

/** Per-particle seed, decorrelated across frames. */
const seedFor = () => instanceIndex.mul(uint(4)).add(frame.mul(uint(1000003)));

const computeInit = Fn(() => {
	const fresh = spawn({ seed: seedFor() }).toVar();
	positionBuffer.element(instanceIndex).assign(fresh.xyz);
	lifeBuffer.element(instanceIndex).assign(fresh.w);
	colorBuffer.element(instanceIndex).assign(vec3(1, 1, 1));
})().compute(PARTICLE_COUNT);

const computeUpdate = Fn(() => {
	const position = positionBuffer.element(instanceIndex);
	const life = lifeBuffer.element(instanceIndex);

	const velocity = curlNoise({ p: position.mul(NOISE_SCALE) }).mul(SPEED).toVar();

	// Colour by direction of travel, exactly as in the reference.
	colorBuffer.element(instanceIndex).assign(
		hue2rgb({ h: atan(velocity.y, velocity.x).div(Math.PI).add(1).mul(0.5) })
	);

	position.addAssign(velocity);
	life.subAssign(1);

	If(life.lessThanEqual(0), () => {
		const fresh = spawn({ seed: seedFor() }).toVar();
		position.assign(fresh.xyz);
		life.assign(fresh.w);
	});
})().compute(PARTICLE_COUNT);

// ---------------------------------------------------------------------------
// Rendering: one instanced billboard per particle
// ---------------------------------------------------------------------------

// Distance from the sprite centre, 0 at the centre and 1 at the disc edge.
const r = uv().sub(0.5).length().mul(2);
// fwidth(r) is exactly one pixel expressed in r's units, so this is a 1px edge.
const disc = smoothstep(float(1).sub(fwidth(r)), 1, r).oneMinus();

const material = new THREE.SpriteNodeMaterial({
	transparent: true,
	depthWrite: false,
	blending: THREE.AdditiveBlending,
});
material.positionNode = positionBuffer.toAttribute();
// .xyz matters: storage buffers of vec3 are padded to 4 floats on the GPU, so the
// attribute arrives as a vec4 whose w is the never-written padding. Assigning the
// whole vec4 to colorNode would set the material's alpha to 0 and draw nothing.
material.colorNode = colorBuffer.toAttribute().xyz;
material.opacityNode = disc;
// sizeAttenuation stays on, so the on-screen radius is proportional to
// PARTICLE_SIZE / -z_view -- i.e. it falls off with camera-space depth.
material.scaleNode = uniform(PARTICLE_SIZE);

const particles = new THREE.Sprite(material);
particles.count = PARTICLE_COUNT;
particles.frustumCulled = false;

// ---------------------------------------------------------------------------
// Scene / renderer
// ---------------------------------------------------------------------------

const scene = new THREE.Scene();
scene.add(particles);

const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.01, 100);
camera.position.set(0, 0, 2);

const renderer = new THREE.WebGPURenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = EXPOSURE;
document.body.appendChild(renderer.domElement);

// Bloom over the HDR scene pass. RenderPipeline applies tone mapping and the sRGB
// conversion to outputNode itself, so this is the same order as RenderPass ->
// UnrealBloomPass -> OutputPass in the reference. The pass resizes with the canvas.
const scenePass = pass(scene, camera).getTextureNode();
const postProcessing = new THREE.RenderPipeline(renderer);
postProcessing.outputNode = scenePass.add(
	bloom(scenePass, BLOOM_STRENGTH, BLOOM_RADIUS, BLOOM_THRESHOLD),
);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

window.addEventListener('resize', () => {
	camera.aspect = window.innerWidth / window.innerHeight;
	camera.updateProjectionMatrix();
	renderer.setSize(window.innerWidth, window.innerHeight);
});

const info = document.getElementById('info')!;

async function main() {
	await renderer.init();

	// The shaders are WGSL, so there is no WebGL2 fallback -- say so plainly
	// rather than letting the GLSL backend fail on WGSL it cannot parse.
	if (!(renderer.backend as any).isWebGPUBackend) {
		throw new Error('this demo needs WebGPU (Chrome/Edge 113+); no adapter was available');
	}

	await renderer.computeAsync(computeInit);

	let last = performance.now();
	let fps = 0;

	renderer.setAnimationLoop(() => {
		frame.value += 1;
		renderer.compute(computeUpdate);

		controls.update();
		postProcessing.render();

		const now = performance.now();
		fps += (1000 / (now - last) - fps) * 0.05;
		last = now;
		info.textContent = `${PARTICLE_COUNT.toLocaleString()} particles   ${fps.toFixed(0)} fps`;
	});
}

main().catch((err) => {
	info.textContent = `Could not start:\n${err}`;
	console.error(err);
});
