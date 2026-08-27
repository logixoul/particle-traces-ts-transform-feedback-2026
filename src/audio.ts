/**
 * Crude kick/snare detection off a looping mp3.
 *
 * The whole algorithm is: watch the energy in one narrow frequency band, and call it
 * a hit whenever that energy jumps a set distance above its own recent average. That
 * is all -- no onset detection, no spectral flux, no tempo tracking. It is wrong often
 * enough that you would not ship it in a music app, but a particle field flashing a few
 * frames late or firing on the occasional hi-hat is not something an eye notices.
 *
 * Two consequences worth knowing before you tune it. The snare band overlaps hats and
 * cymbals, so it will fire on those too -- to this detector "snare" really means "a
 * transient somewhere in the upper mids". And the bass band overlaps the bassline, so
 * a track with busy low end triggers more than one with a dry kick.
 */

const FFT_SIZE = 1024;

/**
 * The two bands, in Hz, with how hard each has to spike to count as a hit. Bass is the
 * kick's fundamental; snare is the noisy crack rather than the drum's body, which sits
 * down around 200 Hz where the bassline would drown it.
 *
 * `margin` is how far above the running average the band's energy must jump. It is an
 * *additive* margin rather than a ratio because getByteFrequencyData reports decibels
 * mapped onto 0-255, not linear energy -- so adding a constant here is already a ratio
 * in energy terms, which is exactly what a transient is. Testing a ratio directly
 * cannot work: on a loud track the bass band sits pegged near the top of the dB range,
 * so a 1.35x test would need 1.21 on a scale that stops at 1.0, firing once on the
 * first frame and then never again. The dB range is 70 wide by default, so 0.04 here
 * is a little under 3 dB.
 *
 * `floor` is the level the band must reach outright, so that near-silence cannot creep
 * over the margin on noise alone.
 *
 * The snare wants a finer margin than the kick: hi-hats keep the upper mids busy all
 * the time, so that band is much steadier and a backbeat stands out of it by less.
 */
const BASS = { hz: [10, 1500], margin: 0.1, floor: -9999.0 };
const SNARE = { hz: [1500, 5000], margin: 0.02, floor: -9999.0 };

// How fast the running average chases the signal. Low, so that the average represents
// roughly the last few hundred ms of the band and a hit stands out against it.
const AVERAGE_SMOOTHING = 0.08;
// Shortest gap between two hits of the same drum. Mostly this stops one kick from
// registering two or three times as its envelope rings.
const REFRACTORY_MS = 220;
// How quickly the output envelope falls back to 0 after a hit.
const ENVELOPE_HALF_LIFE_MS = 90;

type BandConfig = { hz: number[]; margin: number; floor: number };

/** One frequency band, tracking its own average, refractory period and envelope. */
class Band {
	private readonly firstBin: number;
	private readonly lastBin: number;
	private readonly margin: number;
	private readonly floor: number;
	private average = 0;
	private lastHitAt = -Infinity;

	/** 1 on a hit, decaying towards 0. This is what the scene reads. */
	level = 0;
	smoothedLevel = 0;

	constructor({ hz, margin, floor }: BandConfig, binHz: number) {
		// Bin 0 is DC and carries the signal's offset rather than any audible energy,
		// so the low edge is clamped past it.
		this.firstBin = Math.max(1, Math.floor(hz[0] / binHz));
		this.lastBin = Math.ceil(hz[1] / binHz);
		this.margin = margin;
		this.floor = floor;
	}

	update(spectrum: Uint8Array, now: number, dt: number) {
		let sum = 0;
		for (let bin = this.firstBin; bin <= this.lastBin; bin++) sum += spectrum[bin];
		const energy = sum / ((this.lastBin - this.firstBin + 1) * 255);

		if (energy > this.average + this.margin && energy > this.floor
			&& now - this.lastHitAt > REFRACTORY_MS) {
			this.lastHitAt = now;
			this.level = 1;
		} else {
			// Half-life rather than a per-frame constant, so the decay looks the same
			// whether the scene is running at 30fps or 144.
			this.level *= Math.pow(0.5, dt / ENVELOPE_HALF_LIFE_MS);
		}
		
		// Updated after the test, so a hit does not get to raise the bar it just cleared.
		this.average += (energy - this.average) * AVERAGE_SMOOTHING;
		this.smoothedLevel += (this.level - this.smoothedLevel) * 0.2;
	}
}

export type AudioReactor = {
	/** Kick envelope, 1 on a hit and decaying to 0. */
	readonly bass: number;
	/** Snare envelope, same shape. */
	readonly snare: number;
	/** False until the browser lets the track start. */
	readonly playing: boolean;
	/** Call once per frame with performance.now(). */
	update(now: number): void;
};

/**
 * Starts loading `url` immediately and plays it on a loop as soon as it is allowed to.
 *
 * Browsers only permit audio to start from a user gesture, so this tries once on load
 * -- which succeeds if the page already has one, e.g. on a reload after a click -- and
 * otherwise arms a listener that starts the track on the first click or keypress.
 * Until then both envelopes stay at 0 and the scene renders exactly as it would
 * without any of this.
 */
export function createAudioReactor(url: string): AudioReactor {
	const context = new AudioContext();
	const analyser = context.createAnalyser();
	analyser.fftSize = FFT_SIZE;
	// The Band class keeps its own running average, so the analyser's smoothing would
	// only blur the transients we are trying to catch.
	analyser.smoothingTimeConstant = 0;

	const spectrum = new Uint8Array(analyser.frequencyBinCount);
	const binHz = context.sampleRate / FFT_SIZE;
	const bass = new Band(BASS, binHz);
	const snare = new Band(SNARE, binHz);

	// Decoding does not need a running context, so it overlaps the wait for a gesture
	// and the track starts the instant the click lands.
	const decoded = fetch(url)
		.then((response) => response.arrayBuffer())
		.then((bytes) => context.decodeAudioData(bytes));
	decoded.catch((error) => console.error(`could not load ${url}:`, error));

	let playing = false;
	let lastUpdate = -1;

	const play = async () => {
		if (playing) return;
		await context.resume();
		if (context.state !== 'running') return; // still blocked; wait for a gesture

		const buffer = await decoded;
		if (playing) return; // the gesture may have raced the attempt made on load

		const source = context.createBufferSource();
		source.buffer = buffer;
		source.loop = true;
		source.connect(analyser).connect(context.destination);
		source.start(0, 36);
		
		playing = true;
	};

	const tryToPlay = () => { void play(); };
	tryToPlay();
	window.addEventListener('pointerdown', tryToPlay);
	window.addEventListener('keydown', tryToPlay);

	return {
		get bass() { return bass.smoothedLevel; },
		get snare() { return snare.smoothedLevel; },
		get playing() { return playing; },
		update(now: number) {
			const dt = lastUpdate < 0 ? 0 : now - lastUpdate;
			lastUpdate = now;
			if (!playing) return;
			analyser.getByteFrequencyData(spectrum);
			bass.update(spectrum, now, dt);
			snare.update(spectrum, now, dt);
		},
	};
}
