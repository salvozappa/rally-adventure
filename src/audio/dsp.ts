/**
 * ============================================================================
 *  DSP — the small, boring primitives every synth voice in the game is built
 *  from. No game knowledge lives here.
 * ============================================================================
 *
 * Two rules drive the design:
 *
 *  1. Nothing is allocated while the game is running. Buffers, filters and
 *     voice pools are built once at construction; playing a sound means
 *     scheduling automation on parameters that already exist.
 *
 *  2. Every parameter change goes through `SmoothParam`, which uses
 *     `setTargetAtTime` rather than assigning `.value`. Assigning `.value`
 *     steps the parameter at a block boundary and you hear it as a click on
 *     every single frame ("zipper noise"). An exponential approach with a
 *     20-60 ms time constant is inaudible and costs nothing.
 */

export const TWO_PI = Math.PI * 2;

export function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/** Hermite smoothstep, clamped. */
export function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = clamp((x - edge0) / (edge1 - edge0 || 1e-9), 0, 1);
  return t * t * (3 - 2 * t);
}

export function dbToGain(db: number): number {
  return Math.pow(10, db / 20);
}

/** Frame-rate independent exponential approach: `cur` toward `target`. */
export function approach(cur: number, target: number, rate: number, dt: number): number {
  const k = 1 - Math.exp(-rate * dt);
  return cur + (target - cur) * k;
}

/** Deterministic PRNG. Same seed, same texture, every run. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return function next(): number {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export type NoiseKind = 'white' | 'pink' | 'brown';

/**
 * A looping noise bed. Two channels, generated independently so the left and
 * right halves are uncorrelated — that alone is what makes a noise source
 * sound *wide* instead of like a point in the middle of your head.
 *
 * Length matters: a short loop develops an audible periodicity ("looping
 * whoosh"). Four seconds of prime-ish length is long enough that nobody hears
 * the seam, and costs ~1.5 MB.
 */
export function createNoiseBuffer(
  ctx: BaseAudioContext,
  seconds: number,
  kind: NoiseKind = 'white',
  seed = 1234,
): AudioBuffer {
  const n = Math.max(1, Math.floor(ctx.sampleRate * seconds));
  const buf = ctx.createBuffer(2, n, ctx.sampleRate);
  for (let ch = 0; ch < 2; ch++) {
    const d = buf.getChannelData(ch);
    const rnd = mulberry32(seed + ch * 7919);
    if (kind === 'white') {
      for (let i = 0; i < n; i++) d[i] = rnd() * 2 - 1;
    } else if (kind === 'pink') {
      // Paul Kellet's economy pink filter: -3 dB/octave to within 0.05 dB.
      let b0 = 0;
      let b1 = 0;
      let b2 = 0;
      let b3 = 0;
      let b4 = 0;
      let b5 = 0;
      let b6 = 0;
      for (let i = 0; i < n; i++) {
        const w = rnd() * 2 - 1;
        b0 = 0.99886 * b0 + w * 0.0555179;
        b1 = 0.99332 * b1 + w * 0.0750759;
        b2 = 0.969 * b2 + w * 0.153852;
        b3 = 0.8665 * b3 + w * 0.3104856;
        b4 = 0.55 * b4 + w * 0.5329522;
        b5 = -0.7616 * b5 - w * 0.016898;
        d[i] = (b0 + b1 + b2 + b3 + b4 + b5 + b6 + w * 0.5362) * 0.11;
        b6 = w * 0.115926;
      }
    } else {
      // Brown / red: leaky integration, -6 dB/octave.
      let last = 0;
      for (let i = 0; i < n; i++) {
        const w = rnd() * 2 - 1;
        last = (last + 0.02 * w) / 1.02;
        d[i] = last * 3.5;
      }
    }
    // Normalise so every kind arrives at roughly the same loudness.
    let peak = 1e-6;
    for (let i = 0; i < n; i++) peak = Math.max(peak, Math.abs(d[i]));
    const g = 0.9 / peak;
    for (let i = 0; i < n; i++) d[i] *= g;
  }
  return buf;
}

/**
 * A synthetic impulse response for the reverb send. Outdoors on a hillside
 * there is no room — what you hear is a handful of scattered slap-backs off
 * trees and rock faces and then nothing. So this is deliberately *sparse*:
 * exponentially decaying noise multiplied by a random gate, which gives
 * discrete reflections instead of a smooth tail, and a decaying brightness so
 * later reflections are duller than early ones.
 */
export function createImpulseResponse(
  ctx: BaseAudioContext,
  seconds = 1.1,
  decay = 3.4,
  brightness = 0.55,
  seed = 99,
): AudioBuffer {
  const n = Math.max(1, Math.floor(ctx.sampleRate * seconds));
  const buf = ctx.createBuffer(2, n, ctx.sampleRate);
  for (let ch = 0; ch < 2; ch++) {
    const d = buf.getChannelData(ch);
    const rnd = mulberry32(seed + ch * 104729);
    let lp = 0;
    for (let i = 0; i < n; i++) {
      const t = i / n;
      const env = Math.pow(1 - t, decay);
      // Sparse early reflections: only ~18% of samples carry an impulse near
      // the start, thickening as the tail fills in.
      const density = 0.12 + 0.88 * t;
      const hit = rnd() < density ? rnd() * 2 - 1 : 0;
      // Progressive damping: high frequencies die first, as they do in air.
      const a = clamp(brightness * (1 - t * 0.85), 0.02, 0.99);
      lp += a * (hit - lp);
      d[i] = (hit * 0.45 + lp * 0.75) * env;
    }
    // Kill the direct sound; the dry path already carries it.
    const pre = Math.floor(ctx.sampleRate * 0.008);
    for (let i = 0; i < pre && i < n; i++) d[i] *= i / pre;
  }
  return buf;
}

/** A soft-clip curve for the WaveShaper: gentle odd-harmonic saturation. */
export function createSaturationCurve(amount = 2.2, length = 2048): Float32Array<ArrayBuffer> {
  const c = new Float32Array(new ArrayBuffer(length * 4));
  for (let i = 0; i < length; i++) {
    const x = (i / (length - 1)) * 2 - 1;
    c[i] = Math.tanh(x * amount) / Math.tanh(amount);
  }
  return c;
}

// ---------------------------------------------------------------------------
//  Parameter smoothing
// ---------------------------------------------------------------------------

/**
 * An AudioParam plus the last value we asked for. Skipping no-op writes keeps
 * the automation timeline from filling up with thousands of identical events
 * per second, which the browser has to walk on every render quantum.
 */
export class SmoothParam {
  private last = Number.NaN;

  constructor(
    readonly param: AudioParam,
    private readonly tau = 0.05,
    private readonly epsilon = 1e-4,
  ) {}

  /** Exponential approach. The default is right for gains and cutoffs. */
  set(value: number, now: number, tau?: number): void {
    if (!Number.isFinite(value)) return;
    if (Math.abs(value - this.last) < this.epsilon) return;
    this.last = value;
    this.param.setTargetAtTime(value, now, tau ?? this.tau);
  }

  /** Immediate step — only for things that genuinely are discontinuous. */
  jump(value: number, now: number): void {
    if (!Number.isFinite(value)) return;
    this.last = value;
    this.param.cancelScheduledValues(now);
    this.param.setValueAtTime(value, now);
  }

  get value(): number {
    return this.param.value;
  }
}

// ---------------------------------------------------------------------------
//  Voice pools
// ---------------------------------------------------------------------------

export interface BurstOptions {
  /** Peak linear gain. */
  gain: number;
  /** Filter centre frequency, Hz. */
  freq: number;
  /** Filter Q. High Q on a short burst = a metallic *ring*. */
  q?: number;
  type?: BiquadFilterType;
  /** Attack, s. A few ms; 0 gives you a click on top of the click. */
  attack?: number;
  /** Decay to silence, s. */
  decay?: number;
  /** -1..1 */
  pan?: number;
  /** If given, the filter sweeps here over the decay — cheap "material". */
  sweepTo?: number;
  /** Playback-rate scatter so repeated hits are never identical. */
  rateJitter?: number;
}

/**
 * A fixed set of noise voices, cycled round-robin. Each voice is a looping
 * noise source that has been running since construction; "firing" one just
 * writes an envelope onto its gain. Nothing is created, nothing is collected,
 * and there is no start-up latency on the first hit.
 */
export class NoiseBurstPool {
  private readonly srcs: AudioBufferSourceNode[] = [];
  private readonly filters: BiquadFilterNode[] = [];
  private readonly gains: GainNode[] = [];
  private readonly pans: StereoPannerNode[] = [];
  private readonly free: number[] = [];
  private next = 0;
  private disposed = false;

  constructor(
    private readonly ctx: BaseAudioContext,
    dest: AudioNode,
    size: number,
    buffer: AudioBuffer,
    seed = 5,
  ) {
    const rnd = mulberry32(seed);
    for (let i = 0; i < size; i++) {
      const src = ctx.createBufferSource();
      src.buffer = buffer;
      src.loop = true;
      const f = ctx.createBiquadFilter();
      f.type = 'bandpass';
      f.frequency.value = 1000;
      f.Q.value = 1;
      const g = ctx.createGain();
      g.gain.value = 0;
      const p = ctx.createStereoPanner();
      p.pan.value = 0;
      src.connect(f).connect(g).connect(p).connect(dest);
      // Stagger the read positions so two voices never play the same samples.
      src.start(0, rnd() * Math.max(0.01, buffer.duration - 0.05));
      this.srcs.push(src);
      this.filters.push(f);
      this.gains.push(g);
      this.pans.push(p);
      this.free.push(0);
    }
  }

  fire(time: number, o: BurstOptions): void {
    if (this.disposed || o.gain <= 0.0001) return;
    const now = this.ctx.currentTime;
    const t = Math.max(time, now);
    const attack = Math.max(0.0005, o.attack ?? 0.002);
    const decay = Math.max(0.005, o.decay ?? 0.12);

    // Prefer a voice that has already finished; otherwise steal the oldest.
    let idx = -1;
    for (let i = 0; i < this.free.length; i++) {
      const j = (this.next + i) % this.free.length;
      if (this.free[j] <= t) {
        idx = j;
        break;
      }
    }
    if (idx < 0) idx = this.next % this.free.length;
    this.next = (idx + 1) % this.free.length;
    this.free[idx] = t + attack + decay;

    const f = this.filters[idx];
    const g = this.gains[idx].gain;
    const p = this.pans[idx].pan;
    const rate = this.srcs[idx].playbackRate;

    f.type = o.type ?? 'bandpass';
    f.Q.cancelScheduledValues(t);
    f.Q.setValueAtTime(Math.max(0.0001, o.q ?? 1), t);
    f.frequency.cancelScheduledValues(t);
    f.frequency.setValueAtTime(clamp(o.freq, 20, 20000), t);
    if (o.sweepTo !== undefined) {
      f.frequency.exponentialRampToValueAtTime(clamp(o.sweepTo, 20, 20000), t + attack + decay);
    }

    if (o.rateJitter) {
      rate.cancelScheduledValues(t);
      rate.setValueAtTime(clamp(1 + (Math.random() - 0.5) * 2 * o.rateJitter, 0.06, 4), t);
    }

    p.cancelScheduledValues(t);
    p.setValueAtTime(clamp(o.pan ?? 0, -1, 1), t);

    g.cancelScheduledValues(t);
    g.setValueAtTime(0, t);
    g.linearRampToValueAtTime(o.gain, t + attack);
    g.exponentialRampToValueAtTime(Math.max(1e-4, o.gain * 0.0008), t + attack + decay);
    g.linearRampToValueAtTime(0, t + attack + decay + 0.004);
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const s of this.srcs) {
      try {
        s.stop();
      } catch {
        /* already stopped */
      }
      s.disconnect();
    }
    for (const f of this.filters) f.disconnect();
    for (const g of this.gains) g.disconnect();
    for (const p of this.pans) p.disconnect();
  }
}

export interface ToneOptions {
  gain: number;
  /** Start frequency, Hz. */
  freq: number;
  /** End frequency, Hz — a downward sweep is what makes a thud feel heavy. */
  toFreq?: number;
  attack?: number;
  decay?: number;
  pan?: number;
  type?: OscillatorType;
  /** Bandpass/lowpass shaping on the way out. */
  filterFreq?: number;
  filterQ?: number;
  filterType?: BiquadFilterType;
}

/**
 * The tonal counterpart to `NoiseBurstPool`: permanently-running oscillators
 * whose frequency and gain are automated per hit. Used for landing thuds,
 * suspension bottom-out, gear clunks and bird calls.
 */
export class ToneBurstPool {
  private readonly oscs: OscillatorNode[] = [];
  private readonly filters: BiquadFilterNode[] = [];
  private readonly gains: GainNode[] = [];
  private readonly pans: StereoPannerNode[] = [];
  private readonly free: number[] = [];
  private next = 0;
  private disposed = false;

  constructor(
    private readonly ctx: BaseAudioContext,
    dest: AudioNode,
    size: number,
  ) {
    for (let i = 0; i < size; i++) {
      const o = ctx.createOscillator();
      o.type = 'sine';
      o.frequency.value = 100;
      const f = ctx.createBiquadFilter();
      f.type = 'lowpass';
      f.frequency.value = 20000;
      f.Q.value = 0.7;
      const g = ctx.createGain();
      g.gain.value = 0;
      const p = ctx.createStereoPanner();
      o.connect(f).connect(g).connect(p).connect(dest);
      o.start(0);
      this.oscs.push(o);
      this.filters.push(f);
      this.gains.push(g);
      this.pans.push(p);
      this.free.push(0);
    }
  }

  fire(time: number, o: ToneOptions): void {
    if (this.disposed || o.gain <= 0.0001) return;
    const t = Math.max(time, this.ctx.currentTime);
    const attack = Math.max(0.0005, o.attack ?? 0.004);
    const decay = Math.max(0.01, o.decay ?? 0.25);

    let idx = -1;
    for (let i = 0; i < this.free.length; i++) {
      const j = (this.next + i) % this.free.length;
      if (this.free[j] <= t) {
        idx = j;
        break;
      }
    }
    if (idx < 0) idx = this.next % this.free.length;
    this.next = (idx + 1) % this.free.length;
    this.free[idx] = t + attack + decay;

    const osc = this.oscs[idx];
    const filt = this.filters[idx];
    const g = this.gains[idx].gain;

    osc.type = o.type ?? 'sine';
    const f0 = clamp(o.freq, 10, 18000);
    osc.frequency.cancelScheduledValues(t);
    osc.frequency.setValueAtTime(f0, t);
    if (o.toFreq !== undefined) {
      osc.frequency.exponentialRampToValueAtTime(clamp(o.toFreq, 10, 18000), t + attack + decay);
    }

    filt.type = o.filterType ?? 'lowpass';
    filt.frequency.cancelScheduledValues(t);
    filt.frequency.setValueAtTime(clamp(o.filterFreq ?? 18000, 20, 20000), t);
    filt.Q.cancelScheduledValues(t);
    filt.Q.setValueAtTime(Math.max(0.0001, o.filterQ ?? 0.7), t);

    this.pans[idx].pan.cancelScheduledValues(t);
    this.pans[idx].pan.setValueAtTime(clamp(o.pan ?? 0, -1, 1), t);

    g.cancelScheduledValues(t);
    g.setValueAtTime(0, t);
    g.linearRampToValueAtTime(o.gain, t + attack);
    g.exponentialRampToValueAtTime(Math.max(1e-4, o.gain * 0.0008), t + attack + decay);
    g.linearRampToValueAtTime(0, t + attack + decay + 0.004);
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const o of this.oscs) {
      try {
        o.stop();
      } catch {
        /* already stopped */
      }
      o.disconnect();
    }
    for (const f of this.filters) f.disconnect();
    for (const g of this.gains) g.disconnect();
    for (const p of this.pans) p.disconnect();
  }
}

// ---------------------------------------------------------------------------
//  Graph helpers
// ---------------------------------------------------------------------------

export function biquad(
  ctx: BaseAudioContext,
  type: BiquadFilterType,
  freq: number,
  q = 0.7,
  gainDb = 0,
): BiquadFilterNode {
  const f = ctx.createBiquadFilter();
  f.type = type;
  f.frequency.value = clamp(freq, 10, 20000);
  f.Q.value = q;
  f.gain.value = gainDb;
  return f;
}

export function gainNode(ctx: BaseAudioContext, value: number): GainNode {
  const g = ctx.createGain();
  g.gain.value = value;
  return g;
}

export function panner(ctx: BaseAudioContext, pan: number): StereoPannerNode {
  const p = ctx.createStereoPanner();
  p.pan.value = clamp(pan, -1, 1);
  return p;
}

/** Chain a list of nodes head-to-tail and return the tail. */
export function chain<T extends AudioNode>(...nodes: [AudioNode, ...AudioNode[], T]): T {
  for (let i = 0; i < nodes.length - 1; i++) nodes[i].connect(nodes[i + 1]);
  return nodes[nodes.length - 1] as T;
}

export function disconnectAll(nodes: Array<AudioNode | undefined | null>): void {
  for (const n of nodes) {
    if (!n) continue;
    try {
      n.disconnect();
    } catch {
      /* already gone */
    }
  }
}

/**
 * A feedback comb — a delay line with a damped feedback path. Physically this
 * is a pipe: the delay is the round trip down the tube and back, the lowpass
 * is the loss per bounce.
 *
 * The wet path is mixed in at a controlled level rather than summed 1:1 with
 * the dry, and that detail matters more than it looks. A 1:1 sum produces
 * full nulls at every odd multiple of 1/(2*delay); with a 7.4 ms pipe the
 * first of those lands at 68 Hz, which is squarely on top of the firing
 * fundamental between about 1200 and 1600 rpm. The engine would lose its
 * bottom octave in the exact rev range you drive in most. At a wet level of
 * ~0.45 the deepest notch is about -4 dB, so the pipe colours the sound
 * without ever gating a harmonic out of existence.
 */
export class CombFilter {
  readonly input: GainNode;
  readonly output: GainNode;
  private readonly delay: DelayNode;
  private readonly fb: GainNode;
  private readonly damp: BiquadFilterNode;
  private readonly dryGain: GainNode;
  private readonly wetGain: GainNode;

  readonly time: SmoothParam;
  readonly feedback: SmoothParam;
  readonly damping: SmoothParam;
  readonly wet: SmoothParam;

  constructor(
    ctx: BaseAudioContext,
    delaySeconds: number,
    feedback: number,
    dampHz: number,
    wet = 0.45,
  ) {
    this.input = gainNode(ctx, 1);
    this.output = gainNode(ctx, 1);
    this.delay = ctx.createDelay(0.2);
    this.delay.delayTime.value = clamp(delaySeconds, 0.0001, 0.2);
    this.fb = gainNode(ctx, clamp(feedback, 0, 0.95));
    this.damp = biquad(ctx, 'lowpass', dampHz, 0.6);
    this.dryGain = gainNode(ctx, 1);
    this.wetGain = gainNode(ctx, clamp(wet, 0, 1));

    this.input.connect(this.dryGain).connect(this.output);
    this.input.connect(this.delay);
    this.delay.connect(this.damp).connect(this.fb).connect(this.delay);
    this.delay.connect(this.wetGain).connect(this.output);

    this.time = new SmoothParam(this.delay.delayTime, 0.08);
    this.feedback = new SmoothParam(this.fb.gain, 0.08);
    this.damping = new SmoothParam(this.damp.frequency, 0.08);
    this.wet = new SmoothParam(this.wetGain.gain, 0.08);
  }

  dispose(): void {
    disconnectAll([
      this.input,
      this.output,
      this.delay,
      this.fb,
      this.damp,
      this.dryGain,
      this.wetGain,
    ]);
  }
}
