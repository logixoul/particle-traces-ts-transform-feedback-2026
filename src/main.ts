import * as THREE from 'three/webgpu';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { bloom } from 'three/addons/tsl/display/BloomNode.js';
import {
	Fn, If, Loop, float, uint, uniform, instancedArray, instanceIndex,
	atan, wgslFn, pass, varying, vec3, normalize, dot, pow, inverseSqrt,
	positionGeometry, positionWorld, cameraPosition
} from 'three/tsl';

// ---------------------------------------------------------------------------
// Parameters (ported from ParticleLogic.js)
// ---------------------------------------------------------------------------

const PARTICLE_COUNT = 1_000;
const TAIL_LENGTH = 50;      // history samples per particle, as in the reference
const TRAIL_POINTS = PARTICLE_COUNT * TAIL_LENGTH; // one sprite per sample
const LIFESPAN = 1000;        // frames
const NOISE_SCALE = 6.3;      // spatial frequency of the curl field
const SPEED = 0.003;           // world units per frame
const CURL_EPS = 0.01;        // central-difference step for the curl
const PARTICLE_SIZE = 0.01;  // tube diameter in world units
const SPECULAR_F0 = 0.04;     // reflectance head-on; the Fresnel term rises to 1 at grazing
const AXIAL_MAX = 400;          // cap on the look-down-the-barrel emittance flare
const TUBE_SIDES = 6;         // radial segments per tube; 4 x this many triangles per trail sample

const EXPOSURE = 0.6;
const BLOOM_STRENGTH = 0.6;
const BLOOM_RADIUS = 0.0003;
const BLOOM_THRESHOLD = 0.1;

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
	return hue2rgb({ h: atan(velocity.y, velocity.x).div(Math.PI).add(1).mul(0.5) }).add(.01);
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

	If(life.lessThanEqual(0), () => {
		const fresh = spawn({ seed: seedFor() }).toVar();
		life.assign(fresh.w);
		position.assign(rollTrail(fresh.xyz));
	});
})().compute(PARTICLE_COUNT);

// ---------------------------------------------------------------------------
// Rendering: one instanced tube segment per trail sample
// ---------------------------------------------------------------------------

/**
 * A unit vector perpendicular to `dir`, at the angle given by `cs` = (cos, sin).
 *
 * The (u, v) frame comes from Duff et al.'s branchless orthonormal basis, which
 * matters for one specific reason: it is a pure function of `dir`. Two segments
 * that share a direction therefore land their ring vertices in the same phase, so
 * the tube stays continuous across a join instead of showing a twisted hexagon.
 * (It does flip discontinuously for dir.z very near -1; a segment that lands there
 * gets one skewed bevel band for one frame, which is not worth guarding against.)
 */
const ringOffset = wgsl(`
fn ringOffset( dir: vec3f, cs: vec2f ) -> vec3f {
	let s = select( -1.0, 1.0, dir.z >= 0.0 );
	let a = -1.0 / ( s + dir.z );
	let b = dir.x * dir.y * a;
	let u = vec3f( 1.0 + s * dir.x * dir.x * a, s * b, -s * dir.x );
	let v = vec3f( b, s + dir.y * dir.y * a, -dir.y );
	return u * cs.x + v * cs.y;
}`);

/**
 * The instance template: three rings of TUBE_SIDES vertices, skinned with two
 * bands of quads. position.xy is (cos, sin) around the ring and position.z is the
 * ring number -- packing it into the standard `position` attribute rather than
 * custom ones keeps anything in three that expects a position attribute happy.
 *
 * Ring 0 and ring 1 sit on the same point but are perpendicular to different
 * directions (the previous segment's and this one's), so band 0-1 is the bevel
 * wedge that closes the join, and band 1-2 is the cylinder body.
 */
const buildSegmentGeometry = () => {
	const position: number[] = [];
	for (let ring = 0; ring < 3; ring++) {
		for (let side = 0; side < TUBE_SIDES; side++) {
			const angle = (side / TUBE_SIDES) * Math.PI * 2;
			position.push(Math.cos(angle), Math.sin(angle), ring);
		}
	}

	// Wound counter-clockwise as seen from outside the tube, which is what the
	// backface culling wants -- (u, v, dir) is right-handed, so the ring runs
	// counter-clockwise around dir and the rings are ordered along +dir.
	const index: number[] = [];
	for (let ring = 0; ring < 2; ring++) {
		for (let side = 0; side < TUBE_SIDES; side++) {
			const next = (side + 1) % TUBE_SIDES;
			const a = ring * TUBE_SIDES + side;
			const b = ring * TUBE_SIDES + next;
			const c = (ring + 1) * TUBE_SIDES + next;
			const d = (ring + 1) * TUBE_SIDES + side;
			index.push(a, b, c, a, c, d);
		}
	}

	const geometry = new THREE.InstancedBufferGeometry();
	geometry.setAttribute('position', new THREE.Float32BufferAttribute(position, 3));
	geometry.setIndex(index);
	geometry.instanceCount = TRAIL_POINTS;
	return geometry;
};

// Instances are indexed by *age* rather than by raw ring slot: instance `age` of a
// particle covers the segment between the sample `age + 1` frames old and the one
// `age` frames old. That makes the fade a plain function of the instance index, and
// it puts the ring buffer's seam at a known place (the oldest age) instead of
// somewhere that moves every frame.
const particleIndex = instanceIndex.div(uint(TAIL_LENGTH));
const age = instanceIndex.mod(uint(TAIL_LENGTH));

/** The trail buffer index of the sample `n` frames behind this particle's head. */
const sampleIndex = (n: any) => particleIndex.mul(uint(TAIL_LENGTH))
	// n reaches TAIL_LENGTH + 1 below, so two full turns of headroom keep the
	// unsigned subtraction from wrapping.
	.add(trailSlot.add(uint(2 * TAIL_LENGTH)).sub(n).mod(uint(TAIL_LENGTH)));

const older = trailPositions.element(sampleIndex(age.add(uint(1))));
const newer = trailPositions.element(sampleIndex(age));
// The sample before `older`, i.e. where the previous segment came from. Only needed
// for the bevel's first ring.
const oldest = trailPositions.element(sampleIndex(age.add(uint(2))));

const ring = positionGeometry.z;

// One segment per particle spans the ring buffer's seam, joining the newest sample
// to the oldest one across the whole scene. Collapse it to a point: every vertex
// lands on `older`, so its triangles are zero-area and never rasterise.
const isSeam = age.equal(uint(TAIL_LENGTH - 1));
// The segment next to the seam has no previous sample to bevel against either, so
// it just uses its own direction for both rings and comes out flat-ended.
const hasPrevious = age.lessThan(uint(TAIL_LENGTH - 2));

const direction = normalize(newer.sub(older));
const previousDirection = hasPrevious.select(normalize(older.sub(oldest)), direction);

// Rings 0 and 1 sit at the older end, ring 2 at the newer end; ring 0 is the only
// one perpendicular to the previous segment, which is what makes the bevel.
const center = isSeam.select(older, ring.lessThan(1.5).select(older, newer));
const axis = ring.lessThan(0.5).select(previousDirection, direction);
const radius = isSeam.select(float(0), float(PARTICLE_SIZE * 0.5));

// A cylinder's normal is just the radial direction, so the ring offset does double
// duty. varying() moves both this and the colour into the vertex stage, which they
// need anyway: instanceIndex is a vertex-only builtin.
const offset = ringOffset({ dir: axis, cs: positionGeometry.xy });
// : any because varying() is typed as returning a bare Node, which blocks normalize().
const surfaceNormal: any = varying(offset);
// The tube's own direction, needed per-fragment for the axial term below. Ring 0 is
// perpendicular to the previous segment, so this interpolates across the bevel band
// rather than stepping -- which is what you want, and why it needs renormalising.
const tubeAxis: any = varying(axis);

// The colour is carried per ring so it interpolates along the segment instead of
// stepping at every join.
const olderColor = trailColors.element(sampleIndex(age.add(uint(1)))).xyz;
const newerColor = trailColors.element(sampleIndex(age)).xyz;
const emittance = varying(ring.lessThan(1.5).select(olderColor, newerColor));

// Trail fade by age. Written once and never revisited, so it has to be applied at
// draw time; with NoBlending it only ever reaches alphaTest, which trims the single
// oldest sample. Move it onto `emittance` instead to make the tail actually darken.
const trailFade = varying(age.toFloat().div(TAIL_LENGTH).oneMinus()).pow2();

// Blinn-Phong with no ambient and no diffuse term: the stored colour is emitted
// flat, and the only thing that reads as geometry is the specular highlight.
const lightDirection = normalize(vec3(0.4, 0.8, 0.5));
const viewDirection = normalize(cameraPosition.sub(positionWorld));
const halfway = normalize(lightDirection.add(viewDirection));
// Interpolating across the band shortens the normal, hence the renormalise.
// How much glowing gas this ray crossed. A ray that hits a cylinder of radius R
// where the surface normal makes angle theta with the view exits after 2*R*cos(theta),
// and cos(theta) is exactly dot(N, V) -- so the chord length, normalised to 1 at the
// centre of the silhouette and 0 at its edge, costs one dot product of things the
// shader already has. Averaged across the tube's width this comes to PI/4, so the
// reciprocal keeps total emitted energy where it was before rather than dimming
// everything by a fifth.
const chord = dot(normalize(surfaceNormal), viewDirection).max(0).mul(4 / Math.PI);

// A ray running along a tube rather than across it crosses correspondingly more gas:
// the cross-section chord above gets divided by sin(angle between ray and axis), so a
// tube pointing at the camera flares. That diverges when you look straight down one,
// hence AXIAL_MAX -- clamping 1/sin to it is the same as flooring sin^2 at its
// reciprocal square, which gets the whole thing down to one inverseSqrt with no
// division and no infinity to nurse. AXIAL_MAX doubles as the flare's brightness knob.
const alongAxis = dot(normalize(tubeAxis), viewDirection);
const axial = inverseSqrt(alongAxis.mul(alongAxis).oneMinus().max(1 / (AXIAL_MAX * AXIAL_MAX)));

const specular = pow(dot(normalize(surfaceNormal), halfway).max(0), 8);
// Schlick's approximation, on dot(H, V) -- the angle to the microfacet that is
// actually doing the reflecting, not to the surface, which is what keeps the rim of
// a tube lighting up rather than the whole silhouette. Applied after the step, so it
// varies the highlight's brightness and leaves its hard edge where it is; move it
// inside the greaterThan() instead and it would grow the highlight at grazing angles.
const fresnel = dot(halfway, viewDirection).max(0).oneMinus().pow(5)
	.mul(1 - SPECULAR_F0).add(SPECULAR_F0);
const specularStepped = specular.greaterThan(0.5).select(float(1), float(0)).mul(10).mul(fresnel);

//const diffuse =

const material = new THREE.MeshBasicNodeMaterial({
	depthWrite: true,
	depthTest: true,
	//blending: THREE.AdditiveBlending,
	//alphaTest: 0.001,
});
material.positionNode = center.add(offset.mul(radius));
material.colorNode = emittance.mul(chord).mul(axial).add(specularStepped);
material.opacityNode = trailFade;

const particles = new THREE.Mesh(buildSegmentGeometry(), material);
particles.frustumCulled = false;

// ---------------------------------------------------------------------------
// Scene / renderer
// ---------------------------------------------------------------------------

const scene = new THREE.Scene();
scene.add(particles);

const camera = new THREE.PerspectiveCamera(120, window.innerWidth / window.innerHeight, 0.01, 100);
camera.position.set(0, 0, 0.3);

const renderer = new THREE.WebGPURenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
//renderer.toneMapping = THREE.LinearToneMapping;
//renderer.toneMapping = THREE.AgXToneMapping;
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
