import * as THREE from 'three/webgpu';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { bloom } from 'three/addons/tsl/display/BloomNode.js';
import {
	Fn, If, Loop, float, uint, uv, uniform, instancedArray, instanceIndex,
	atan, smoothstep, fwidth, wgslFn, pass, varying, step,
	vec3,
	normalize,
	cameraViewMatrix,
	vec4,
	positionViewDirection,
	dot,
	pow
} from 'three/tsl';

// ---------------------------------------------------------------------------
// Parameters (ported from ParticleLogic.js)
// ---------------------------------------------------------------------------

const PARTICLE_COUNT = 1_000;
const TAIL_LENGTH = 100;      // history samples per particle, as in the reference
const TRAIL_POINTS = PARTICLE_COUNT * TAIL_LENGTH; // one sprite per sample
const LIFESPAN = 100;        // frames
const NOISE_SCALE = 6.3;      // spatial frequency of the curl field
const SPEED = 0.001;           // world units per frame
const CURL_EPS = 0.01;        // central-difference step for the curl
const PARTICLE_SIZE = 0.02;  // sprite diameter in world units

// Post-processing, ported from the reference App.js.
const EXPOSURE = 1.5;         // additive blending blows way past 1, so pull it back hard
const BLOOM_STRENGTH = 0.6;
const BLOOM_RADIUS = 0.0003;
const BLOOM_THRESHOLD = 0.1;  // only genuinely dense clumps glow

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
	//return vec3f(valueNoise(p), valueNoise(p.yzx + vec3f(100.0)), valueNoise(p.zxy + vec3f(200.0))); // Placeholder for actual curl noise computation
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

// Live simulation state: one entry per particle.
const positionBuffer = instancedArray(PARTICLE_COUNT, 'vec3');
const lifeBuffer = instancedArray(PARTICLE_COUNT, 'float');

// The trails. Each particle owns TAIL_LENGTH consecutive slots used as a ring
// buffer; every frame writes exactly one slot, so a trail costs one store per
// particle no matter how long it is. This is also what gets drawn.
const trailPositions = instancedArray(TRAIL_POINTS, 'vec3');
const trailColors = instancedArray(TRAIL_POINTS, 'vec3');

const frame = uniform(0, 'uint');
const trailSlot = uniform(0, 'uint'); // ring slot written this frame

/** Per-particle seed, decorrelated across frames. */
const seedFor = () => instanceIndex.mul(uint(4)).add(frame.mul(uint(1000003)));

/** Index of this particle's slot `n` in the trail buffers. */
const trailIndex = (n: any) => instanceIndex.mul(uint(TAIL_LENGTH)).add(n);

/** One step of the flow: advance `p`, and give back the colour for that step. */
const doStep = (p: any) => {
	const velocity = curlNoise({ p: p.mul(NOISE_SCALE) }).mul(SPEED).toVar();
	p.addAssign(velocity);
	// Colour by direction of travel, exactly as in the reference.
	return hue2rgb({ h: atan(velocity.y, velocity.x).div(Math.PI).add(1).mul(0.5) }).add(0.01);
};

/**
 * Fill a whole trail in one go by running the flow forward TAIL_LENGTH times, and
 * return where the particle ended up. Used on spawn and respawn: the reference does
 * the same thing (reinit() calls update() TAIL_LENGTH times). Just parking every
 * slot on the spawn point instead would stack TAIL_LENGTH sprites in one spot and
 * speckle the scene with bright dots until each trail grew back out.
 */
const rollTrail = (position: any) => {
	const p = position.toVar();

	Loop({ type: 'uint', start: 0, end: TAIL_LENGTH }, ({ i }) => {
		const color = doStep(p);

		// Walk the ring forward from the head so the last step written is the newest
		// sample, leaving the per-frame appends below in the right phase.
		const slot = trailSlot.add(i).add(uint(1)).toVar();
		If(slot.greaterThanEqual(uint(TAIL_LENGTH)), () => {
			slot.subAssign(uint(TAIL_LENGTH));
		});

		trailPositions.element(trailIndex(slot)).assign(p);
		trailColors.element(trailIndex(slot)).assign(color);
	});

	return p;
};

const computeInit = Fn(() => {
	const fresh = spawn({ seed: seedFor() }).toVar();
	lifeBuffer.element(instanceIndex).assign(fresh.w);
	positionBuffer.element(instanceIndex).assign(rollTrail(fresh.xyz));
})().compute(PARTICLE_COUNT);

const computeUpdate = Fn(() => {
	const position = positionBuffer.element(instanceIndex);
	const life = lifeBuffer.element(instanceIndex);

	const p = position.toVar();
	const color = doStep(p);
	life.subAssign(1);

	// Append the new head of the trail.
	trailPositions.element(trailIndex(trailSlot)).assign(p);
	trailColors.element(trailIndex(trailSlot)).assign(color);
	position.assign(p);

	/*If(life.lessThanEqual(0), () => {
		const fresh = spawn({ seed: seedFor() }).toVar();
		life.assign(fresh.w);
		position.assign(rollTrail(fresh.xyz));
	});*/
})().compute(PARTICLE_COUNT);

// ---------------------------------------------------------------------------
// Rendering: one instanced billboard per particle
// ---------------------------------------------------------------------------

// Distance from the sprite centre, 0 at the centre and 1 at the disc edge.
const r = uv().sub(0.5).length().mul(2);
// fwidth(r) is exactly one pixel expressed in r's units, so this is a 1px edge.
const disc = step(float(1), r).oneMinus();

const d  = uv().sub(0.5).mul(2);                 // [-1,1] across the sprite
const normalZ = r.mul(r).oneMinus().max(0).sqrt();    // sqrt(1 - x² - y²)
const N  = vec3(d.x, d.y, normalZ);
const lightDirWorld = vec3(0.4, 0.8, 0.5);
const L = normalize(cameraViewMatrix.mul(vec4(lightDirWorld, 0)).xyz);
const V = positionViewDirection;
const H = normalize(L.add(V));
const diff = dot(N, L).max(0);
const spec = pow(dot(N, H).max(0), 40).mul(2.0);

// Trail fade. A sample's age is how far its ring slot sits behind the head, so it
// has to be worked out here at draw time -- the stored colour is written once and
// never revisited, so baking a fade into it would freeze each sample at whatever
// brightness it had when written. Adding TAIL_LENGTH before subtracting keeps the
// unsigned arithmetic from wrapping.
const sampleSlot = instanceIndex.mod(uint(TAIL_LENGTH));
const sampleAge = trailSlot.add(uint(TAIL_LENGTH)).sub(sampleSlot).mod(uint(TAIL_LENGTH));
// varying() forces this into the vertex stage: instanceIndex is a vertex-only
// builtin, so the fragment shader cannot read it directly.
const trailFade = varying(sampleAge.toFloat().div(TAIL_LENGTH).oneMinus()).pow2();

const material = new THREE.SpriteNodeMaterial({
	//transparent: true,
	depthWrite: true,
	depthTest: true,
	blending: THREE.NoBlending,
	alphaTest: 0.001,
});
// One sprite per trail sample, so the trails need no rendering code of their own.
material.positionNode = trailPositions.toAttribute();
// .xyz matters: storage buffers of vec3 are padded to 4 floats on the GPU, so the
// attribute arrives as a vec4 whose w is the never-written padding. Assigning the
// whole vec4 to colorNode would set the material's alpha to 0 and draw nothing.
const base = trailColors.toAttribute().xyz;
material.colorNode = base.mul(0.5).add(diff.mul(0.5)).add(spec);
material.opacityNode = disc.mul(trailFade);
// sizeAttenuation stays on, so the on-screen radius is proportional to
// PARTICLE_SIZE / -z_view -- i.e. it falls off with camera-space depth.
material.scaleNode = uniform(PARTICLE_SIZE);

const particles = new THREE.Sprite(material);
particles.count = TRAIL_POINTS;
particles.frustumCulled = false;

// ---------------------------------------------------------------------------
// Scene / renderer
// ---------------------------------------------------------------------------

const scene = new THREE.Scene();
scene.add(particles);

const camera = new THREE.PerspectiveCamera(90, window.innerWidth / window.innerHeight, 0.01, 100);
camera.position.set(0, 0, 2);

const renderer = new THREE.WebGPURenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
//renderer.toneMapping = THREE.ACESFilmicToneMapping;
//renderer.toneMapping = THREE.LinearToneMapping;
renderer.toneMapping = THREE.AgXToneMapping;
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
		trailSlot.value = (trailSlot.value + 1) % TAIL_LENGTH;
		renderer.compute(computeUpdate);
		renderer.compute(computeUpdate);
		renderer.compute(computeUpdate);

		controls.update();
		postProcessing.render();

		const now = performance.now();
		fps += (1000 / (now - last) - fps) * 0.05;
		last = now;
		info.textContent = `${PARTICLE_COUNT.toLocaleString()} particles`
			+ ` x ${TAIL_LENGTH} trail   ${fps.toFixed(0)} fps`;
	});
}

main().catch((err) => {
	info.textContent = `Could not start:\n${err}`;
	console.error(err);
});
