/**
 * ============================================================================
 *  ENGINE SYNTH — a 3.6 L naturally-aspirated six, built from combustion up.
 * ============================================================================
 *
 * WHY IT IS BUILT THIS WAY
 *
 * The naive approach — a sawtooth oscillator whose frequency follows RPM —
 * fails for a specific reason: a sawtooth is *perfectly* periodic, and its
 * harmonics are locked in phase and amplitude forever. Real engine noise is a
 * train of discrete, individually-imperfect explosions ringing a metal box.
 * Your ear is extremely good at telling those apart, which is why the
 * sawtooth version always sounds like a hairdryer.
 *
 * So this synthesises the actual mechanism:
 *
 *   EXCITATION (AudioWorklet, per-sample)
 *     A four-stroke six fires once every 120 degrees of crank, so the firing
 *     rate is rpm/60 * 6/2 = rpm/20 Hz. 760 rpm idle -> 38 Hz; 6200 rpm
 *     redline -> 310 Hz. Each firing injects an impulse into three
 *     attack/decay envelope pairs:
 *
 *        crack  ~0.2 ms  broadband noise      the mechanical "tick"
 *        body   1-3 ms   noise + pulse        the bark; width tracks load
 *        boom   6-10 ms  unipolar pulse       the low-frequency slam
 *
 *     A decaying exponential of time constant tau has a spectrum rolling off
 *     from 1/(2*pi*tau), so making the body envelope *shorter* under load is
 *     literally the same thing as adding upper harmonics. That is the load
 *     -> timbre link, and it comes out of the physics rather than a filter
 *     sweep bolted on afterwards.
 *
 *   IMPERFECTION
 *     Each of the six cylinders carries a fixed amplitude and timing trim of
 *     a couple of percent (they are not identical castings and never will
 *     be), plus a fresh random perturbation on every firing. The fixed trims
 *     repeat once per engine cycle — two crank revolutions — which puts real
 *     energy at half-orders (firing/6, firing/3, firing/2 ...). Those
 *     half-order components ARE the lumpy, loping character of a big
 *     naturally-aspirated engine, and here they fall out of the model for
 *     free instead of being faked with a second oscillator.
 *
 *   RESONANCE (main thread, built once)
 *     The excitation is nearly flat-ish broadband. All the character comes
 *     from what it is poured through: an exhaust formant pair, a comb filter
 *     standing in for the pipe (7.4 ms round trip ~ a 1.3 m tailpipe),
 *     a body resonance, and a separate intake path that only wakes up under
 *     throttle. These track RPM sub-linearly, so the formants drift upward
 *     as revs rise without simply transposing — the same reason a shouted
 *     vowel still sounds like the same vowel.
 *
 *   TWO VOICES
 *     The worklet emits two channels: combustion (channel 0) and mechanical
 *     pumping (channel 1). On throttle you hear mostly combustion through a
 *     saturated, wide-open path. Off throttle the fuel is cut, combustion
 *     collapses, and what is left is the pumping voice through a hollow
 *     high-passed path, with unburnt-fuel pops firing in the exhaust. The
 *     crossfade is continuous, so part-throttle sits believably between them.
 */

import type { EngineTuning } from '../physics/VehicleTuning';
import {
  CombFilter,
  NoiseBurstPool,
  SmoothParam,
  ToneBurstPool,
  biquad,
  chain,
  clamp,
  createNoiseBuffer,
  createSaturationCurve,
  disconnectAll,
  gainNode,
  lerp,
  mulberry32,
  panner,
  smoothstep,
} from './dsp';

/** Cylinder count. A six fires three times per crank revolution. */
const CYLINDERS = 6;

/**
 * Final trim on the engine bus, set by measurement rather than by ear: with
 * this value a wide-open-throttle hold renders at roughly -19 dBFS RMS and
 * peaks near -10 dBFS, which leaves the master limiter with real headroom to
 * work in instead of running permanently in gain reduction.
 */
const OUTPUT_TRIM = 0.026;

// ---------------------------------------------------------------------------
//  The worklet
// ---------------------------------------------------------------------------

/**
 * Source for the per-sample excitation generator. It is kept as a string and
 * installed from a Blob URL so the whole engine stays inside one module and
 * needs no build-system support for a second entry point.
 *
 * Deliberately written in plain ES2017 with no imports: worklet global scope
 * has no DOM, no modules of its own, and a hard real-time budget of 128
 * samples per callback.
 */
const ENGINE_WORKLET_SRC = String.raw`
const CYL = 6;

class EngineProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      { name: 'firingFreq', defaultValue: 38, minValue: 0.5, maxValue: 1200, automationRate: 'a-rate' },
      { name: 'drive',      defaultValue: 0,  minValue: 0,   maxValue: 1,    automationRate: 'a-rate' },
      { name: 'cut',        defaultValue: 0,  minValue: 0,   maxValue: 1,    automationRate: 'a-rate' },
      { name: 'level',      defaultValue: 0,  minValue: 0,   maxValue: 4,    automationRate: 'a-rate' },
      { name: 'jitter',     defaultValue: 0.02, minValue: 0, maxValue: 0.4,  automationRate: 'k-rate' },
      { name: 'wobble',     defaultValue: 0,  minValue: 0,   maxValue: 0.25, automationRate: 'k-rate' },
      { name: 'harmonics',  defaultValue: 0.3, minValue: 0,  maxValue: 1,    automationRate: 'k-rate' }
    ];
  }

  constructor() {
    super();
    this.sr = sampleRate;
    this.seed = 0x9e3779b9;

    // --- firing clock -----------------------------------------------------
    this.acc = 1;            // >= 1 so the first sample fires immediately
    this.cyl = 0;
    this.rateJit = 1;

    // Permanent per-cylinder character. Compression, injector flow and cam
    // timing all vary a little between cylinders on any real engine.
    this.trimAmp = new Float32Array(CYL);
    this.trimRate = new Float32Array(CYL);
    var meanRate = 0;
    for (var i = 0; i < CYL; i++) {
      this.trimAmp[i] = 1 + (this.rnd() - 0.5) * 0.09;
      this.trimRate[i] = (this.rnd() - 0.5) * 0.028;
      meanRate += this.trimRate[i];
    }
    meanRate /= CYL;
    // Remove the mean so the trims skew the *spacing* without dragging the
    // average firing rate off the RPM we were asked for.
    for (var j = 0; j < CYL; j++) this.trimRate[j] -= meanRate;

    // --- envelope states (attack pole a, decay pole b, per band) ----------
    this.aC = 0; this.bC = 0;   // crack
    this.aB = 0; this.bB = 0;   // body
    this.aO = 0; this.bO = 0;   // boom
    this.aM = 0; this.bM = 0;   // mechanical / pumping

    // --- misc state -------------------------------------------------------
    this.hPhase = 0;
    this.wob = 0; this.wobTarget = 0; this.wobCount = 0;
    this.hissLp = 0;
    this.dc1x = 0; this.dc1y = 0;
    this.dc2x = 0; this.dc2y = 0;
    this.alive = true;

    this.port.onmessage = (e) => {
      if (e.data === 'stop') this.alive = false;
    };
  }

  rnd() {
    this.seed = (Math.imul(this.seed, 1664525) + 1013904223) >>> 0;
    return this.seed / 4294967296;
  }

  /**
   * Peak value of the attack/decay pair for a unit impulse, so each band can
   * be normalised to a known amplitude regardless of how the time constants
   * are being modulated. Closed form: the pair is a difference of two
   * geometric sequences, and the maximum is at
   *     k = ln(ln(dD)/ln(dA)) / (ln(dA) - ln(dD)).
   */
  peakOf(dA, dD) {
    var la = Math.log(dA), ld = Math.log(dD);
    if (!(la < 0) || !(ld < 0) || Math.abs(la - ld) < 1e-9) return 1;
    var k = Math.log(ld / la) / (la - ld);
    if (!(k > 0)) return 1;
    var p = ((1 - dD) / (dA - dD)) * (Math.exp(la * k) - Math.exp(ld * k));
    return Math.abs(p) > 1e-6 ? p : 1;
  }

  process(inputs, outputs, params) {
    if (!this.alive) return false;
    var out = outputs[0];
    if (!out || out.length === 0) return true;
    var L = out[0];
    var R = out.length > 1 ? out[1] : out[0];
    var n = L.length;

    var pF = params.firingFreq, pD = params.drive, pC = params.cut, pL = params.level;
    var jitter = params.jitter[0];
    var wobbleAmt = params.wobble[0];
    var harmAmt = params.harmonics[0];

    // Time constants are recomputed once per render quantum: 2.7 ms of
    // granularity on a timbre control is far below the ear's resolution and
    // it keeps 8 exp() calls out of the sample loop.
    var driveK = pD.length > 1 ? pD[0] : pD[0];

    var tauAC = 0.00007,            tauDC = 0.00024;
    var tauAB = 0.00022,            tauDB = 0.00100 + 0.00230 * (1 - driveK);
    var tauAO = 0.00070,            tauDO = 0.00560 + 0.00380 * driveK;
    var tauAM = 0.00030,            tauDM = 0.00170;

    var dAC = Math.exp(-1 / (tauAC * this.sr)), dDC = Math.exp(-1 / (tauDC * this.sr));
    var dAB = Math.exp(-1 / (tauAB * this.sr)), dDB = Math.exp(-1 / (tauDB * this.sr));
    var dAO = Math.exp(-1 / (tauAO * this.sr)), dDO = Math.exp(-1 / (tauDO * this.sr));
    var dAM = Math.exp(-1 / (tauAM * this.sr)), dDM = Math.exp(-1 / (tauDM * this.sr));

    var nC = 1 / this.peakOf(dAC, dDC);
    var nB = 1 / this.peakOf(dAB, dDB);
    var nO = 1 / this.peakOf(dAO, dDO);
    var nM = 1 / this.peakOf(dAM, dDM);

    // Band mix. Under load the bark (body) and slam (boom) dominate; off
    // load the mechanical crack is proportionally louder because there is
    // much less combustion energy to bury it.
    var gC = 0.30 + 0.16 * driveK;
    var gB = 0.42 + 0.38 * driveK;
    var gO = 0.55 + 0.55 * driveK;
    var gM = 0.34;

    // Harmonic stack amplitudes. Upper orders swell with load, which is the
    // classic "brassy under throttle" behaviour.
    var h1 = 0.55;
    var h2 = 0.30 * (0.35 + 0.65 * driveK);
    var h3 = 0.19 * (0.18 + 0.82 * driveK);
    var h4 = 0.11 * (0.05 + 0.95 * driveK);
    var h6 = 0.06 * driveK;

    for (var i = 0; i < n; i++) {
      var f = pF.length > 1 ? pF[i] : pF[0];
      var drive = pD.length > 1 ? pD[i] : pD[0];
      var cut = pC.length > 1 ? pC[i] : pC[0];
      var level = pL.length > 1 ? pL[i] : pL[0];

      // --- idle instability ---------------------------------------------
      // A cold-ish six at idle hunts by a few tens of RPM. Perfectly steady
      // idle is one of the loudest tells of synthetic engine audio.
      if (this.wobCount <= 0) { this.wobCount = 47; this.wobTarget = this.rnd() * 2 - 1; }
      this.wobCount--;
      this.wob += (this.wobTarget - this.wob) * 0.004;
      var freq = f * (1 + wobbleAmt * this.wob);
      if (freq < 0.5) freq = 0.5;

      // --- firing clock --------------------------------------------------
      this.acc += (freq * this.rateJit) / this.sr;
      if (this.acc >= 1) {
        this.acc -= 1;
        var c = this.cyl;
        this.cyl = (c + 1) % CYL;
        this.rateJit = 1 + this.trimRate[c] + (this.rnd() - 0.5) * jitter * 2;
        if (this.rateJit < 0.5) this.rateJit = 0.5;

        var amp = this.trimAmp[c] * (1 + (this.rnd() - 0.5) * jitter * 5);
        if (amp < 0) amp = 0;
        // Fuel cut kills combustion but NOT pumping: the pistons keep going
        // up and down, which is exactly why a limiter sounds like a stutter
        // rather than silence.
        var fuel = (1 - cut) * (0.30 + 0.70 * drive);
        var combAmp = amp * (0.35 + 0.65 * fuel);

        this.aC += combAmp;
        this.aB += combAmp;
        this.aO += combAmp;
        this.aM += amp;
      }

      // --- envelopes ------------------------------------------------------
      this.aC *= dAC; this.bC = this.bC * dDC + this.aC * (1 - dDC);
      this.aB *= dAB; this.bB = this.bB * dDB + this.aB * (1 - dDB);
      this.aO *= dAO; this.bO = this.bO * dDO + this.aO * (1 - dDO);
      this.aM *= dAM; this.bM = this.bM * dDM + this.aM * (1 - dDM);

      var eC = this.bC * nC;
      var eB = this.bB * nB;
      var eO = this.bO * nO;
      var eM = this.bM * nM;

      var w = this.rnd() * 2 - 1;
      var w2 = this.rnd() * 2 - 1;

      // --- harmonic stack -------------------------------------------------
      this.hPhase += freq / this.sr;
      if (this.hPhase >= 1) this.hPhase -= 1;
      var th = 6.283185307179586 * this.hPhase;
      var harm = h1 * Math.sin(th) + h2 * Math.sin(2 * th) + h3 * Math.sin(3 * th)
               + h4 * Math.sin(4 * th) + h6 * Math.sin(6 * th);
      harm *= harmAmt * (1 - cut) * (0.25 + 0.75 * drive);

      // --- combustion voice ----------------------------------------------
      // The +0.42 offsets make the body and boom bands unipolar, i.e. an
      // actual pressure pulse rather than a symmetric wiggle. That asymmetry
      // is what fills in the even harmonics.
      var comb = gC * eC * w
               + gB * eB * (0.58 * w2 + 0.42)
               + gO * eO
               + harm;

      // --- mechanical / pumping voice -------------------------------------
      // Valvetrain and induction hiss, scaled by how fast it is all moving.
      this.hissLp += (w - this.hissLp) * 0.42;
      var hiss = (w - this.hissLp) * 0.05 * Math.min(1, freq / 90);
      var mech = gM * eM * (0.80 * w2 + 0.20) + hiss;

      // --- DC blockers (the unipolar pulses carry a large DC term) --------
      var y1 = comb - this.dc1x + 0.9975 * this.dc1y;
      this.dc1x = comb; this.dc1y = y1;
      var y2 = mech - this.dc2x + 0.9975 * this.dc2y;
      this.dc2x = mech; this.dc2y = y2;

      // Safety saturation. The resonators downstream have real gain and an
      // unbounded excitation would eventually find it.
      L[i] = Math.tanh(y1 * level);
      R[i] = Math.tanh(y2 * level);
    }
    return true;
  }
}

registerProcessor('rally-engine', EngineProcessor);
`;

const workletLoaded = new WeakSet<BaseAudioContext>();
const workletPending = new WeakMap<BaseAudioContext, Promise<boolean>>();

/** Install the excitation worklet. Safe to call repeatedly on one context. */
export function loadEngineWorklet(ctx: BaseAudioContext): Promise<boolean> {
  if (workletLoaded.has(ctx)) return Promise.resolve(true);
  const pending = workletPending.get(ctx);
  if (pending) return pending;
  const p = (async () => {
    const aw = (ctx as unknown as { audioWorklet?: AudioWorklet }).audioWorklet;
    if (!aw || typeof Blob === 'undefined' || typeof URL === 'undefined') return false;
    const url = URL.createObjectURL(new Blob([ENGINE_WORKLET_SRC], { type: 'text/javascript' }));
    try {
      await aw.addModule(url);
      workletLoaded.add(ctx);
      return true;
    } catch {
      return false;
    } finally {
      URL.revokeObjectURL(url);
    }
  })();
  workletPending.set(ctx, p);
  return p;
}

// ---------------------------------------------------------------------------
//  Excitation sources
// ---------------------------------------------------------------------------

interface ExcitationSource {
  /** Two-channel: 0 = combustion, 1 = mechanical. */
  readonly out: AudioNode;
  set(
    firingFreq: number,
    drive: number,
    cut: number,
    level: number,
    jitter: number,
    wobble: number,
    now: number,
  ): void;
  dispose(): void;
}

class WorkletExcitation implements ExcitationSource {
  readonly out: AudioWorkletNode;
  private readonly pFreq: SmoothParam;
  private readonly pDrive: SmoothParam;
  private readonly pCut: SmoothParam;
  private readonly pLevel: SmoothParam;
  private readonly pJitter: SmoothParam;
  private readonly pWobble: SmoothParam;

  constructor(ctx: BaseAudioContext, harmonics: number) {
    this.out = new AudioWorkletNode(ctx, 'rally-engine', {
      numberOfInputs: 0,
      numberOfOutputs: 1,
      outputChannelCount: [2],
      channelCount: 2,
      channelCountMode: 'explicit',
      channelInterpretation: 'discrete',
    });
    const p = this.out.parameters;
    this.pFreq = new SmoothParam(p.get('firingFreq') as AudioParam, 0.03, 1e-3);
    this.pDrive = new SmoothParam(p.get('drive') as AudioParam, 0.05, 1e-3);
    this.pCut = new SmoothParam(p.get('cut') as AudioParam, 0.004, 1e-3);
    this.pLevel = new SmoothParam(p.get('level') as AudioParam, 0.04, 1e-3);
    this.pJitter = new SmoothParam(p.get('jitter') as AudioParam, 0.15, 1e-3);
    this.pWobble = new SmoothParam(p.get('wobble') as AudioParam, 0.2, 1e-3);
    (p.get('harmonics') as AudioParam).value = harmonics;
  }

  set(
    firingFreq: number,
    drive: number,
    cut: number,
    level: number,
    jitter: number,
    wobble: number,
    now: number,
  ): void {
    this.pFreq.set(firingFreq, now);
    this.pDrive.set(drive, now);
    this.pCut.set(cut, now);
    this.pLevel.set(level, now);
    this.pJitter.set(jitter, now);
    this.pWobble.set(wobble, now);
  }

  dispose(): void {
    try {
      this.out.port.postMessage('stop');
    } catch {
      /* context already closed */
    }
    this.out.disconnect();
  }
}

/**
 * Fallback for the (rare) case where AudioWorklet is unavailable or the Blob
 * module is refused by a CSP. Much cruder — a pulse-shaped PeriodicWave plus
 * noise — but it keeps the car audible instead of silent, and it feeds the
 * exact same resonator chain so the character survives.
 */
class OscillatorExcitation implements ExcitationSource {
  readonly out: ChannelMergerNode;
  private readonly osc: OscillatorNode;
  private readonly oscGain: GainNode;
  private readonly noiseSrc: AudioBufferSourceNode;
  private readonly noiseGain: GainNode;
  private readonly mechSrc: AudioBufferSourceNode;
  private readonly mechGain: GainNode;
  private readonly mechFilter: BiquadFilterNode;
  private readonly pFreq: SmoothParam;
  private readonly pOsc: SmoothParam;
  private readonly pNoise: SmoothParam;
  private readonly pMech: SmoothParam;

  constructor(ctx: BaseAudioContext, noise: AudioBuffer) {
    this.out = ctx.createChannelMerger(2);

    // A pulse-train spectrum: harmonics falling at roughly 1/n with a little
    // extra weight on the low orders, which is what a shaped impulse gives.
    const N = 24;
    const real = new Float32Array(N);
    const imag = new Float32Array(N);
    for (let k = 1; k < N; k++) imag[k] = (1 / k) * Math.exp(-k / 11);
    const wave = ctx.createPeriodicWave(real, imag, { disableNormalization: false });

    this.osc = ctx.createOscillator();
    this.osc.setPeriodicWave(wave);
    this.osc.frequency.value = 38;
    this.oscGain = gainNode(ctx, 0);
    this.osc.connect(this.oscGain).connect(this.out, 0, 0);
    this.osc.start(0);

    this.noiseSrc = ctx.createBufferSource();
    this.noiseSrc.buffer = noise;
    this.noiseSrc.loop = true;
    this.noiseGain = gainNode(ctx, 0);
    this.noiseSrc.connect(this.noiseGain).connect(this.out, 0, 0);
    this.noiseSrc.start(0);

    this.mechSrc = ctx.createBufferSource();
    this.mechSrc.buffer = noise;
    this.mechSrc.loop = true;
    this.mechFilter = biquad(ctx, 'highpass', 900, 0.7);
    this.mechGain = gainNode(ctx, 0);
    this.mechSrc.connect(this.mechFilter).connect(this.mechGain).connect(this.out, 0, 1);
    this.mechSrc.start(0);

    this.pFreq = new SmoothParam(this.osc.frequency, 0.03);
    this.pOsc = new SmoothParam(this.oscGain.gain, 0.05);
    this.pNoise = new SmoothParam(this.noiseGain.gain, 0.05);
    this.pMech = new SmoothParam(this.mechGain.gain, 0.05);
  }

  set(
    firingFreq: number,
    drive: number,
    cut: number,
    level: number,
    _jitter: number,
    _wobble: number,
    now: number,
  ): void {
    const g = level * (1 - cut);
    this.pFreq.set(firingFreq, now);
    this.pOsc.set(g * (0.20 + 0.30 * drive), now);
    this.pNoise.set(g * 0.08, now);
    this.pMech.set(level * (0.10 + 0.06 * (1 - drive)), now);
  }

  dispose(): void {
    for (const s of [this.noiseSrc, this.mechSrc]) {
      try {
        s.stop();
      } catch {
        /* fine */
      }
    }
    try {
      this.osc.stop();
    } catch {
      /* fine */
    }
    disconnectAll([
      this.osc,
      this.oscGain,
      this.noiseSrc,
      this.noiseGain,
      this.mechSrc,
      this.mechFilter,
      this.mechGain,
      this.out,
    ]);
  }
}

// ---------------------------------------------------------------------------
//  Public state passed in each frame
// ---------------------------------------------------------------------------

export interface EngineDriveState {
  rpm: number;
  /** -1 (full overrun) .. 0 (coasting) .. +1 (full load). */
  load: number;
  /** True while the gearbox is cutting torque. */
  shifting: boolean;
  /** 0..1 clutch engagement, straight from the drivetrain. */
  clutch: number;
  /** Signed forward speed, m/s — only used for the mild doppler/effort tilt. */
  speed: number;
}

// ---------------------------------------------------------------------------
//  EngineSynth
// ---------------------------------------------------------------------------

export class EngineSynth {
  readonly output: GainNode;

  private readonly ctx: BaseAudioContext;
  private readonly tuning: EngineTuning;

  private source: ExcitationSource | null = null;
  private readonly sourceSum: GainNode;
  private readonly splitter: ChannelSplitterNode;

  private readonly firingRes: BiquadFilterNode;
  private readonly dcBlock: BiquadFilterNode;

  // combustion path
  private readonly satPre: GainNode;
  private readonly sat: WaveShaperNode;
  private readonly exhaust1: BiquadFilterNode;
  private readonly exhaust2: BiquadFilterNode;
  private readonly loadLp: BiquadFilterNode;
  private readonly loadGain: GainNode;

  // overrun path
  private readonly overHp: BiquadFilterNode;
  private readonly overBp: BiquadFilterNode;
  private readonly overGain: GainNode;

  // shared body + pipe
  private readonly bodyRes: BiquadFilterNode;
  private readonly pipe: CombFilter;
  private readonly rumble: BiquadFilterNode;

  // intake
  private readonly intakeHp: BiquadFilterNode;
  private readonly intakeBp: BiquadFilterNode;
  private readonly intakeGain: GainNode;
  private readonly intakePan: StereoPannerNode;

  // pops / mechanical one-shots
  private readonly pops: NoiseBurstPool;
  private readonly clunks: NoiseBurstPool;
  private readonly thuds: ToneBurstPool;

  private readonly pExhaust1: SmoothParam;
  private readonly pExhaust2: SmoothParam;
  private readonly pLoadLp: SmoothParam;
  private readonly pLoadGain: SmoothParam;
  private readonly pOverBp: SmoothParam;
  private readonly pOverGain: SmoothParam;
  private readonly pIntakeBp: SmoothParam;
  private readonly pIntakeGain: SmoothParam;
  private readonly pSatPre: SmoothParam;
  private readonly pBodyRes: SmoothParam;
  private readonly pRumble: SmoothParam;
  private readonly pFiringRes: SmoothParam;
  private readonly pOut: SmoothParam;

  private readonly rnd = mulberry32(0x51ed);

  // running state
  private smoothLoad = 0;
  private smoothRpm: number;
  private prevRpm: number;
  private limiterPhase = 0;
  private popTimer = 0;
  private shiftTimer = 0;
  private prevShifting = false;
  private ready = false;
  private disposed = false;

  /** Excitation level currently commanded — exposed for the preview page. */
  lastFiringFreq = 0;
  lastCut = 0;

  constructor(ctx: BaseAudioContext, tuning: EngineTuning, noise: AudioBuffer, harmonics = 0.32) {
    this.ctx = ctx;
    this.tuning = tuning;
    this.smoothRpm = tuning.idleRpm;
    this.prevRpm = tuning.idleRpm;

    // Silent until the first update(). Anything that defaults to a live level
    // is a graph-startup thump waiting to happen: the context begins at t = 0
    // whenever the user finally clicks, and every resonator downstream would
    // be charged by a full-amplitude excitation before the first frame ran.
    this.output = gainNode(ctx, 0);
    this.sourceSum = gainNode(ctx, 1);
    this.sourceSum.channelCount = 2;
    this.sourceSum.channelCountMode = 'explicit';
    this.sourceSum.channelInterpretation = 'discrete';
    this.splitter = ctx.createChannelSplitter(2);
    this.sourceSum.connect(this.splitter);

    // --- combustion path --------------------------------------------------
    // Saturation first: driving a resonator with a soft-clipped signal is how
    // you get the growl. Clipping *after* the filters just sounds broken.
    this.satPre = gainNode(ctx, 0.7);
    this.sat = ctx.createWaveShaper();
    this.sat.curve = createSaturationCurve(1.9);
    this.sat.oversample = '2x';
    // Two exhaust formants. The first is the deep pipe mode, the second the
    // mid honk that gives the engine its vowel.
    this.exhaust1 = biquad(ctx, 'peaking', 150, 1.4, 5);
    this.exhaust2 = biquad(ctx, 'peaking', 430, 3.2, 6);
    this.loadLp = biquad(ctx, 'lowpass', 1400, 0.9);
    this.loadGain = gainNode(ctx, 0.6);
    this.splitter.connect(this.satPre, 0);
    chain(this.satPre, this.sat, this.exhaust1, this.exhaust2, this.loadLp, this.loadGain);

    // --- overrun path -----------------------------------------------------
    // Hollow: everything below 240 Hz stripped out, a single mid band left.
    this.overHp = biquad(ctx, 'highpass', 240, 0.8);
    this.overBp = biquad(ctx, 'bandpass', 900, 1.4);
    this.overGain = gainNode(ctx, 0.25);
    this.splitter.connect(this.overHp, 1);
    chain(this.overHp, this.overBp, this.overGain);
    // The mechanical voice also feeds a little into the loaded path so the
    // valvetrain never disappears entirely.
    this.splitter.connect(this.satPre, 1);

    // --- shared body + pipe ----------------------------------------------
    this.bodyRes = biquad(ctx, 'peaking', 235, 2.0, 3);
    // A peaking filter parked exactly on the firing frequency. Without it the
    // fixed formants can leave the fundamental 20 dB below its own second
    // harmonic at low-mid revs, which sounds hollow and thin — the engine
    // loses its bottom octave right where you spend most of your driving.
    // This is the audio equivalent of order tracking.
    this.firingRes = biquad(ctx, 'peaking', 120, 1.4, 8);
    // 7.4 ms round trip: a ~1.27 m tailpipe at 343 m/s. Its comb notches are
    // spaced 135 Hz apart, which is squarely in the range that reads as
    // "exhaust" rather than "flanger".
    this.pipe = new CombFilter(ctx, 0.0074, 0.34, 2600, 0.32);
    this.rumble = biquad(ctx, 'lowshelf', 80, 0.7, 2);
    // Soft-clipping an asymmetric pulse train rectifies it, so every
    // saturation stage in this chain manufactures DC. Strip it once, here, at
    // the end — 22 Hz is below anything the engine ever produces.
    this.dcBlock = biquad(ctx, 'highpass', 22, 0.6);

    this.loadGain.connect(this.bodyRes);
    this.overGain.connect(this.bodyRes);
    // Order matters: the firing-order support sits *after* the pipe, not
    // before it. In front of the comb its boost simply lands in whichever
    // notch the pipe happens to have parked on the fundamental, which left an
    // 8 dB hole in the rev range around 4000 rpm. Behind it, the fundamental
    // is supported no matter where the comb null falls.
    chain(this.bodyRes, this.pipe.input);
    chain(this.pipe.output, this.firingRes, this.rumble, this.dcBlock, this.output);

    // --- intake path ------------------------------------------------------
    // Induction roar is a separate, brighter, slightly off-centre thing. It
    // only matters when the throttle plate is open, which is exactly the cue
    // that tells you the driver is *asking* for something.
    this.intakeHp = biquad(ctx, 'highpass', 520, 0.7);
    this.intakeBp = biquad(ctx, 'bandpass', 1400, 1.1);
    this.intakeGain = gainNode(ctx, 0);
    this.intakePan = panner(ctx, -0.16);
    this.splitter.connect(this.intakeHp, 0);
    chain(this.intakeHp, this.intakeBp, this.intakeGain, this.intakePan, this.output);

    // --- one-shot pools ---------------------------------------------------
    this.pops = new NoiseBurstPool(ctx, this.output, 5, noise, 31);
    this.clunks = new NoiseBurstPool(ctx, this.output, 4, noise, 77);
    this.thuds = new ToneBurstPool(ctx, this.output, 3);

    // --- smoothed handles -------------------------------------------------
    this.pExhaust1 = new SmoothParam(this.exhaust1.frequency, 0.05);
    this.pExhaust2 = new SmoothParam(this.exhaust2.frequency, 0.05);
    this.pLoadLp = new SmoothParam(this.loadLp.frequency, 0.06);
    this.pLoadGain = new SmoothParam(this.loadGain.gain, 0.05);
    this.pOverBp = new SmoothParam(this.overBp.frequency, 0.06);
    this.pOverGain = new SmoothParam(this.overGain.gain, 0.06);
    this.pIntakeBp = new SmoothParam(this.intakeBp.frequency, 0.05);
    this.pIntakeGain = new SmoothParam(this.intakeGain.gain, 0.06);
    this.pSatPre = new SmoothParam(this.satPre.gain, 0.06);
    this.pBodyRes = new SmoothParam(this.bodyRes.frequency, 0.08);
    this.pRumble = new SmoothParam(this.rumble.gain, 0.1);
    this.pFiringRes = new SmoothParam(this.firingRes.frequency, 0.04);
    this.pOut = new SmoothParam(this.output.gain, 0.05);

    // Bring the excitation online. Until the worklet module resolves we run
    // the oscillator fallback, then hot-swap; the resonator chain is shared
    // so the transition is a change of texture, not a gap.
    this.source = new OscillatorExcitation(ctx, noise);
    this.source.out.connect(this.sourceSum);

    void loadEngineWorklet(ctx).then((ok) => {
      if (this.disposed || !ok) {
        this.ready = true;
        return;
      }
      try {
        const w = new WorkletExcitation(ctx, harmonics);
        w.out.connect(this.sourceSum);
        const old = this.source;
        this.source = w;
        old?.dispose();
        this.ready = true;
      } catch {
        this.ready = true;
      }
    });
  }

  /** True once the excitation source has settled (worklet or fallback). */
  get isReady(): boolean {
    return this.ready;
  }

  get usingWorklet(): boolean {
    return this.source instanceof WorkletExcitation;
  }

  /** Firing frequency in Hz for a given RPM — the number the whole synth hangs off. */
  static firingFrequency(rpm: number): number {
    return (rpm / 60) * (CYLINDERS / 2);
  }

  /** Peak crank torque available at this RPM, N.m. Used to normalise load. */
  private availableTorque(rpm: number): number {
    const c = this.tuning.torqueCurve;
    if (c.length === 0) return 300;
    if (rpm <= c[0][0]) return c[0][1];
    for (let i = 1; i < c.length; i++) {
      if (rpm <= c[i][0]) {
        const [r0, t0] = c[i - 1];
        const [r1, t1] = c[i];
        const u = (rpm - r0) / Math.max(1e-6, r1 - r0);
        return lerp(t0, t1, u);
      }
    }
    return c[c.length - 1][1];
  }

  torqueAt(rpm: number): number {
    return this.availableTorque(rpm);
  }

  // -------------------------------------------------------------------------

  update(dt: number, s: EngineDriveState): void {
    if (this.disposed) return;
    const ctx = this.ctx;
    const now = ctx.currentTime;
    const t = this.tuning;

    const rpm = clamp(Number.isFinite(s.rpm) ? s.rpm : t.idleRpm, 0, t.maxRpm);
    // Light smoothing on RPM. The sim runs at 120 Hz and the render loop can
    // be anywhere; without this you hear the frame rate as a warble.
    this.smoothRpm += (rpm - this.smoothRpm) * clamp(dt * 26, 0, 1);
    const rpmRate = (this.smoothRpm - this.prevRpm) / Math.max(dt, 1e-4);
    this.prevRpm = this.smoothRpm;

    const rNorm = clamp((this.smoothRpm - t.idleRpm) / (t.redlineRpm - t.idleRpm), 0, 1.15);
    const firing = EngineSynth.firingFrequency(this.smoothRpm);
    this.lastFiringFreq = firing;

    // --- load ---------------------------------------------------------------
    const rawLoad = clamp(Number.isFinite(s.load) ? s.load : 0, -1, 1);
    // Asymmetric slew: load arrives fast (the throttle is a step input) and
    // leaves a bit slower, which stops part-throttle chatter buzzing.
    const lr = rawLoad > this.smoothLoad ? 16 : 9;
    this.smoothLoad += (rawLoad - this.smoothLoad) * clamp(dt * lr, 0, 1);
    const drive = clamp(this.smoothLoad, 0, 1);
    const overrun = clamp(-this.smoothLoad, 0, 1);

    // --- rev limiter ---------------------------------------------------------
    // Fuel is cut hard in bursts once past the redline. `limiterCutTime` from
    // the tuning sets the cut length so the audio stutter and the physics
    // stutter are the same event.
    let cut = 0;
    const over = this.smoothRpm - t.redlineRpm;
    if (over > 0) {
      this.limiterPhase += dt;
      const period = t.limiterCutTime * 2;
      const severity = smoothstep(0, t.limiterBandRpm, over);
      cut = this.limiterPhase % period < t.limiterCutTime * (0.55 + 0.45 * severity) ? 1 : 0;
    } else {
      this.limiterPhase = 0;
    }

    // --- gearshift torque cut ------------------------------------------------
    if (s.shifting && !this.prevShifting) this.shiftTimer = 0.001;
    this.prevShifting = s.shifting;
    if (this.shiftTimer > 0) {
      this.shiftTimer += dt;
      if (this.shiftTimer > 0.26) this.shiftTimer = 0;
    }
    const shiftDip = this.shiftTimer > 0 ? smoothstep(0.26, 0.06, this.shiftTimer) : 0;
    // A slipping clutch also means the engine is not connected to anything,
    // so back the excitation off with it.
    const clutchDip = 1 - 0.25 * clamp(1 - s.clutch, 0, 1) * smoothstep(1200, 2600, this.smoothRpm);
    this.lastCut = cut;

    // --- excitation ----------------------------------------------------------
    // Louder with revs and with load, but sub-linearly: engines get louder
    // fast off idle and then plateau, they do not keep doubling.
    const effort = 0.68 + 0.32 * Math.pow(rNorm, 0.55);
    // Kept modest on purpose. The tanh inside the worklet is a safety net and
    // a touch of combustion grit, not the main tone shaper — if you drive it
    // hard the excitation squares off and every resonator downstream is then
    // filtering a square wave.
    const level = clamp(
      (0.58 + 0.42 * drive) * effort * clutchDip * (1 - 0.55 * shiftDip),
      0,
      1.4,
    );
    // Jitter is largest at idle (combustion is least repeatable when there is
    // barely any fuel going in) and tightens up as the engine loads up.
    const jitter = lerp(0.055, 0.014, clamp(rNorm * 0.8 + drive * 0.4, 0, 1));
    const wobble = lerp(0.055, 0.0, smoothstep(0, 0.16, rNorm)) * (1 - drive * 0.7);

    this.source?.set(firing, drive, cut, level, jitter, wobble, now);

    // --- resonator tracking ----------------------------------------------------
    // Sub-linear tracking: formants drift up with revs but nothing like
    // proportionally, so the engine keeps its identity across the range
    // instead of sounding like a sample being transposed.
    const rp = Math.pow(clamp(this.smoothRpm, 300, t.maxRpm) / 1000, 0.78);
    this.pExhaust1.set(clamp(68 + 40 * rp + firing * 0.35, 50, 700), now);
    this.pExhaust2.set(clamp(300 + 210 * rp + firing * 0.55, 180, 2600), now);
    this.pBodyRes.set(clamp(210 + 40 * rp, 150, 460), now);

    // The lowpass is the single most important load cue: wide open under
    // throttle (you hear all the combustion detail), shut down on a trailing
    // throttle (the sound retreats into the bodywork).
    this.pLoadLp.set(clamp(1100 + 5000 * Math.pow(drive, 0.7) + 900 * rNorm, 400, 14000), now);
    this.pSatPre.set(0.85 + 0.50 * drive, now);
    this.pLoadGain.set(0.62 + 0.32 * drive, now);
    // Track the firing frequency, but keep the support filter inside the band
    // where it is doing useful work.
    this.pFiringRes.set(clamp(firing, 28, 700), now, 0.03);

    this.pOverBp.set(clamp(620 + 520 * rp, 300, 3200), now);
    // Overrun is quiet in absolute terms — that is the point — but it must
    // never vanish, or a trailing throttle sounds like the engine died.
    this.pOverGain.set((0.16 + 0.5 * overrun) * (0.45 + 0.55 * rNorm), now);

    this.pIntakeBp.set(clamp(760 + 900 * rp + firing * 1.4, 400, 6000), now);
    this.pIntakeGain.set(0.5 * Math.pow(drive, 1.4) * (0.35 + 0.65 * rNorm), now);

    this.pRumble.set(1 + 3 * drive, now);
    this.pipe.feedback.set(clamp(0.24 + 0.18 * drive, 0, 0.7), now);
    this.pipe.damping.set(clamp(1500 + 5200 * drive, 400, 12000), now);

    this.pOut.set(clamp(OUTPUT_TRIM * (1 - 0.45 * shiftDip) * (1 - 0.35 * cut), 0, 1), now);

    // --- overrun pops and crackles ---------------------------------------------
    // Unburnt fuel igniting in a hot exhaust. Only on a real trailing
    // throttle, only above about 1800 rpm, and only while the revs are
    // genuinely falling — that last condition is what stops it firing
    // constantly while coasting at a steady speed.
    const decelerating = rpmRate < -180;
    const popping = overrun > 0.30 && this.smoothRpm > 1800 && decelerating;
    this.popTimer -= dt;
    if (popping && this.popTimer <= 0) {
      const intensity = clamp(overrun * (0.4 + 0.6 * rNorm), 0, 1);
      this.popTimer = lerp(0.16, 0.035, intensity) * (0.5 + this.rnd());
      const n = this.rnd() < 0.25 ? 2 : 1;
      for (let i = 0; i < n; i++) {
        const tt = now + i * (0.012 + this.rnd() * 0.02);
        this.pops.fire(tt, {
          gain: (0.16 + 0.34 * intensity) * (0.5 + 0.5 * this.rnd()),
          freq: 260 + this.rnd() * 900,
          q: 1.1 + this.rnd() * 2.2,
          attack: 0.0012,
          decay: 0.035 + this.rnd() * 0.07,
          sweepTo: 90 + this.rnd() * 140,
          pan: (this.rnd() - 0.5) * 0.5,
          rateJitter: 0.35,
        });
      }
      // Every so often a proper crack rather than a burble.
      if (this.rnd() < 0.18 * intensity) {
        this.thuds.fire(now, {
          gain: 0.2 * intensity,
          freq: 190,
          toFreq: 62,
          attack: 0.001,
          decay: 0.1,
          filterFreq: 900,
          filterQ: 1.2,
          pan: 0.1,
        });
      }
    }

    // --- limiter bark ------------------------------------------------------------
    if (cut > 0 && this.rnd() < 0.35) {
      this.pops.fire(now, {
        gain: 0.16,
        freq: 420 + this.rnd() * 700,
        q: 2.0,
        attack: 0.001,
        decay: 0.04,
        pan: (this.rnd() - 0.5) * 0.4,
        rateJitter: 0.3,
      });
    }
  }

  /** Mechanical gearchange: a dull clunk through the transmission tunnel. */
  playShiftClunk(strength = 1): void {
    if (this.disposed) return;
    const now = this.ctx.currentTime;
    const g = clamp(strength, 0, 1);
    this.clunks.fire(now, {
      gain: 0.30 * g,
      freq: 240,
      q: 1.6,
      attack: 0.0012,
      decay: 0.055,
      sweepTo: 120,
      pan: -0.08,
      rateJitter: 0.25,
    });
    this.thuds.fire(now + 0.004, {
      gain: 0.22 * g,
      freq: 130,
      toFreq: 52,
      attack: 0.002,
      decay: 0.11,
      filterFreq: 620,
      filterQ: 1.1,
      pan: -0.05,
    });
    // A faint metallic ring off the bellhousing.
    this.clunks.fire(now + 0.006, {
      gain: 0.10 * g,
      freq: 1900 + this.rnd() * 700,
      q: 9,
      attack: 0.0008,
      decay: 0.09,
      pan: 0.12,
      rateJitter: 0.4,
    });
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.source?.dispose();
    this.source = null;
    this.pops.dispose();
    this.clunks.dispose();
    this.thuds.dispose();
    this.pipe.dispose();
    disconnectAll([
      this.sourceSum,
      this.splitter,
      this.satPre,
      this.sat,
      this.exhaust1,
      this.exhaust2,
      this.loadLp,
      this.loadGain,
      this.overHp,
      this.overBp,
      this.overGain,
      this.bodyRes,
      this.firingRes,
      this.rumble,
      this.dcBlock,
      this.intakeHp,
      this.intakeBp,
      this.intakeGain,
      this.intakePan,
      this.output,
    ]);
  }
}
