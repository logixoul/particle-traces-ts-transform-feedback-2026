import * as THREE from 'three/webgpu';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import {
	Fn, If, vec3, vec4, float, uint, uv, uniform, instancedArray, instanceIndex, hash,
	floor, fract, mix, mod, dot, sin, abs, saturate, atan, smoothstep, fwidth,
} from 'three/tsl';

// ---------------------------------------------------------------------------
// Parameters (ported from ParticleLogic.js)
// ---------------------------------------------------------------------------

const PARTICLE_COUNT = 1_000_000;
const LIFESPAN = 1000;        // frames
const NOISE_SCALE = 0.3;      // spatial frequency of the curl field
const SPEED = 0.01;           // world units per frame
const CURL_EPS = 0.01;        // central-difference step for the curl
const PARTICLE_SIZE = 0.006;  // sprite diameter in world units

// ---------------------------------------------------------------------------
// Hash-based value noise (replaces simplex noise)
// ---------------------------------------------------------------------------

/** Lattice hash: vec3 -> [0,1) */
const hash31 = Fn(([p]: [any]) => {
	return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))).mul(43758.5453123));
});
hash31.setLayout({ name: 'hash31', type: 'float', inputs: [{ name: 'p', type: 'vec3' }] });

/** Trilinearly interpolated value noise, in [-1,1]. */
const valueNoise = Fn(([p]: [any]) => {
	const i = floor(p);
	const f = fract(p);
	const u = f.mul(f).mul(float(3).sub(f.mul(2))); // smoothstep weights

	const n000 = hash31(i.add(vec3(0, 0, 0)));
	const n100 = hash31(i.add(vec3(1, 0, 0)));
	const n010 = hash31(i.add(vec3(0, 1, 0)));
	const n110 = hash31(i.add(vec3(1, 1, 0)));
	const n001 = hash31(i.add(vec3(0, 0, 1)));
	const n101 = hash31(i.add(vec3(1, 0, 1)));
	const n011 = hash31(i.add(vec3(0, 1, 1)));
	const n111 = hash31(i.add(vec3(1, 1, 1)));

	const nx00 = mix(n000, n100, u.x);
	const nx10 = mix(n010, n110, u.x);
	const nx01 = mix(n001, n101, u.x);
	const nx11 = mix(n011, n111, u.x);

	const nxy0 = mix(nx00, nx10, u.y);
	const nxy1 = mix(nx01, nx11, u.y);

	return mix(nxy0, nxy1, u.z).mul(2).sub(1);
});
valueNoise.setLayout({ name: 'valueNoise', type: 'float', inputs: [{ name: 'p', type: 'vec3' }] });

/**
 * Curl of the vector field F(p) = ( N(p), N(p.yzx + 100), N(p.zxy + 200) ),
 * by central differences -- same construction as the reference implementation.
 */
const curlNoise = Fn(([p]: [any]) => {
	const e = float(CURL_EPS);
	const inv2e = float(1 / (2 * CURL_EPS));

	const x = p.x, y = p.y, z = p.z;

	const dFz_dy = valueNoise(vec3(z.add(200), x.add(200), y.add(200).add(e)))
		.sub(valueNoise(vec3(z.add(200), x.add(200), y.add(200).sub(e)))).mul(inv2e);
	const dFy_dz = valueNoise(vec3(y.add(100), z.add(100).add(e), x.add(100)))
		.sub(valueNoise(vec3(y.add(100), z.add(100).sub(e), x.add(100)))).mul(inv2e);

	const dFx_dz = valueNoise(vec3(x, y, z.add(e)))
		.sub(valueNoise(vec3(x, y, z.sub(e)))).mul(inv2e);
	const dFz_dx = valueNoise(vec3(z.add(200).add(e), x.add(200), y.add(200)))
		.sub(valueNoise(vec3(z.add(200).sub(e), x.add(200), y.add(200)))).mul(inv2e);

	const dFy_dx = valueNoise(vec3(y.add(100).add(e), z.add(100), x.add(100)))
		.sub(valueNoise(vec3(y.add(100).sub(e), z.add(100), x.add(100)))).mul(inv2e);
	const dFx_dy = valueNoise(vec3(x, y.add(e), z))
		.sub(valueNoise(vec3(x, y.sub(e), z))).mul(inv2e);

	return vec3(dFz_dy.sub(dFy_dz), dFx_dz.sub(dFz_dx), dFy_dx.sub(dFx_dy));
});
curlNoise.setLayout({ name: 'curlNoise', type: 'vec3', inputs: [{ name: 'p', type: 'vec3' }] });

/** Fully saturated hue -> RGB (equivalent to HSL with s = 1, l = 0.5). */
const hue2rgb = Fn(([h]: [any]) => {
	const k = mod(h.mul(6).add(vec3(0, 4, 2)), 6);
	return saturate(abs(k.sub(3)).sub(1));
});
hue2rgb.setLayout({ name: 'hue2rgb', type: 'vec3', inputs: [{ name: 'h', type: 'float' }] });

// ---------------------------------------------------------------------------
// GPU state: lives in storage buffers, never round-trips to the CPU
// ---------------------------------------------------------------------------

const positionBuffer = instancedArray(PARTICLE_COUNT, 'vec3');
const colorBuffer = instancedArray(PARTICLE_COUNT, 'vec3');
const lifeBuffer = instancedArray(PARTICLE_COUNT, 'float');

const frame = uniform(0, 'uint');

/**
 * A fresh particle: xyz is a random point in the unit cube centred on the origin,
 * w is a random lifetime. Returning the state (rather than writing the buffers from
 * inside here) matters -- TSL only emits nodes whose result is used somewhere.
 */
const spawn = Fn(([seed]: [any]) => {
	return vec4(
		hash(seed).sub(0.5),
		hash(seed.add(uint(1))).sub(0.5),
		hash(seed.add(uint(2))).sub(0.5),
		hash(seed.add(uint(3))).mul(LIFESPAN),
	);
});
spawn.setLayout({ name: 'spawn', type: 'vec4', inputs: [{ name: 'seed', type: 'uint' }] });

/** Per-particle seed, decorrelated across frames. */
const seedFor = () => instanceIndex.mul(uint(4)).add(frame.mul(uint(1000003)));

const computeInit = Fn(() => {
	const fresh = spawn(seedFor());
	positionBuffer.element(instanceIndex).assign(fresh.xyz);
	lifeBuffer.element(instanceIndex).assign(fresh.w);
	colorBuffer.element(instanceIndex).assign(vec3(1, 1, 1));
})().compute(PARTICLE_COUNT);

const computeUpdate = Fn(() => {
	const position = positionBuffer.element(instanceIndex);
	const life = lifeBuffer.element(instanceIndex);

	const velocity = curlNoise(position.mul(NOISE_SCALE)).mul(SPEED).toVar();

	// Colour by direction of travel, exactly as in the reference.
	colorBuffer.element(instanceIndex).assign(
		hue2rgb(atan(velocity.y, velocity.x).div(Math.PI).add(1).mul(0.5))
	);

	position.addAssign(velocity);
	life.subAssign(1);

	If(life.lessThanEqual(0), () => {
		const fresh = spawn(seedFor());
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
material.colorNode = colorBuffer.toAttribute();
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
document.body.appendChild(renderer.domElement);

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
	await renderer.computeAsync(computeInit);

	let last = performance.now();
	let fps = 0;

	renderer.setAnimationLoop(() => {
		frame.value += 1;
		renderer.compute(computeUpdate);

		controls.update();
		renderer.render(scene, camera);

		const now = performance.now();
		fps += (1000 / (now - last) - fps) * 0.05;
		last = now;
		info.textContent = `${PARTICLE_COUNT.toLocaleString()} particles   ${fps.toFixed(0)} fps`;
	});
}

main().catch((err) => {
	info.textContent = `WebGPU unavailable:\n${err}`;
	console.error(err);
});
