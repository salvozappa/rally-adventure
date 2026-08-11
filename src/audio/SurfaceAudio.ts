/**
 * ============================================================================
 *  SURFACE AUDIO — everything that is not the engine: tyres, the ground under
 *  them, the suspension above them, and the air going past.
 * ============================================================================
 *
 * The organising idea is that surfaces are not switched, they are *blended*.
 * Every surface is a point in a small parameter space (roll brightness, body
 * resonance, grain rate, how readily it squeals vs scrabbles), and each frame
 * the four wheels vote — weighted by contact load — for a blended point in
 * that space. Straddling the edge of a gravel patch therefore sounds like
 * straddling the edge of a gravel patch, and the transition costs nothing
 * because it is just parameter smoothing on a graph that already exists.
 *
 * Three continuous voices and three event sources:
 *
 *   ROLL      broadband noise shaped by the blended surface, level and
 *             brightness rising with speed, with a tread thrum locked to
 *             wheel rotation.
 *   SKID      a high-Q squeal voice and a broadband scrabble voice, crossfaded
 *             against each other by surface hardness and against grip by slip.
 *   WIND      pink noise opening up with speed, wide in the stereo field,
 *             with a whistle band and a big whoosh when the car is airborne.
 *
 *   GRAIN     discrete stone strikes under the arches on loose surfaces.
 *   SUSPENSION compression thumps, bottom-out clunks, rebound.
 *   RATTLE    loose bodywork, proportional to how hard the ground is working
 *             the suspension. This is the cheapest character in the whole mix:
 *             it is what makes the car feel forty years old.
 */

import type { SurfaceKind, VehicleState, WheelState } from '../types';
import {
  NoiseBurstPool,
  SmoothParam,
  ToneBurstPool,
  biquad,
  chain,
  clamp,
  createNoiseBuffer,
  disconnectAll,
  gainNode,
  mulberry32,
  smoothstep,
} from './dsp';

/** Speed the surface table is written at, m/s (~72 km/h). */
const REF_SPEED = 20;

interface SurfaceVoice {
  /** Overall roll loudness multiplier. */
  level: number;
  /** Roll lowpass cutoff at REF_SPEED, Hz — this is "brightness". */
  lp: number;
  /** Roll highpass, Hz. Loose surfaces have less low-end body than hard ones. */
  hp: number;
  /** Resonant body of the contact patch. */
  bp: number;
  bpQ: number;
  bpGain: number;
  /** Discrete stone strikes per second at REF_SPEED. */
  grainRate: number;
  grainFreq: number;
  grainQ: number;
  grainGain: number;
  grainDecay: number;
  /** How much a slipping tyre squeals (hard) vs scrabbles (loose). */
  squeal: number;
  scrabble: number;
  /** Centre frequency of the squeal on this surface, Hz. */
  squealFreq: number;
  /** Multiplier on body rattle — how much this surface shakes the car. */
  rough: number;
}

/**
 * Surface table. Written by ear against the physics tuning in
 * VehicleTuning.SURFACES so that the surface that grips hardest (rock) is also
 * the one that rings, and the one that grips least (snow) is also the quietest.
 */
const SURFACE_VOICES: Record<SurfaceKind, SurfaceVoice> = {
  // Hard-packed dirt: the reference. Broad, dull, a bit of loose top layer.
  dirt: {
    level: 0.80, lp: 1500, hp: 110, bp: 420, bpQ: 0.9, bpGain: 4,
    grainRate: 16, grainFreq: 1100, grainQ: 2.4, grainGain: 0.10, grainDecay: 0.030,
    squeal: 0.35, scrabble: 0.75, squealFreq: 820, rough: 0.9,
  },
  // Grass: swishy. Almost no low end, all brush against the sidewall.
  grass: {
    level: 0.62, lp: 3800, hp: 420, bp: 2400, bpQ: 0.5, bpGain: 2,
    grainRate: 7, grainFreq: 3200, grainQ: 1.2, grainGain: 0.05, grainDecay: 0.022,
    squeal: 0.25, scrabble: 0.70, squealFreq: 700, rough: 0.7,
  },
  // Rock: hard and ringing. High-Q body resonance, sharp grain, real squeal.
  rock: {
    level: 0.86, lp: 5600, hp: 150, bp: 1450, bpQ: 3.0, bpGain: 7,
    grainRate: 12, grainFreq: 3600, grainQ: 6.5, grainGain: 0.13, grainDecay: 0.055,
    squeal: 1.00, scrabble: 0.20, squealFreq: 1150, rough: 1.6,
  },
  // Gravel: the loudest surface there is. Crunchy, bright, wall-to-wall stones.
  gravel: {
    level: 1.00, lp: 4400, hp: 190, bp: 1750, bpQ: 1.1, bpGain: 4,
    grainRate: 64, grainFreq: 2700, grainQ: 3.2, grainGain: 0.16, grainDecay: 0.028,
    squeal: 0.15, scrabble: 1.00, squealFreq: 640, rough: 1.25,
  },
  // Sand: soft, dark, no impacts to speak of — just a hiss and a lot of drag.
  sand: {
    level: 0.72, lp: 900, hp: 90, bp: 300, bpQ: 0.6, bpGain: 3,
    grainRate: 22, grainFreq: 620, grainQ: 0.9, grainGain: 0.05, grainDecay: 0.030,
    squeal: 0.04, scrabble: 1.00, squealFreq: 520, rough: 0.55,
  },
  // Mud: squelch. Lowest band of all with a wet, resonant thump to it.
  mud: {
    level: 0.88, lp: 700, hp: 70, bp: 250, bpQ: 1.9, bpGain: 8,
    grainRate: 26, grainFreq: 380, grainQ: 1.6, grainGain: 0.10, grainDecay: 0.070,
    squeal: 0.08, scrabble: 0.95, squealFreq: 480, rough: 0.85,
  },
  // Snow: quiet, muffled, packing under the tread.
  snow: {
    level: 0.50, lp: 1000, hp: 90, bp: 340, bpQ: 0.5, bpGain: 2,
    grainRate: 10, grainFreq: 780, grainQ: 0.8, grainGain: 0.04, grainDecay: 0.026,
    squeal: 0.05, scrabble: 0.80, squealFreq: 560, rough: 0.5,
  },
};

const SURFACE_KEYS = Object.keys(SURFACE_VOICES) as SurfaceKind[];

/** Working blend target, mutated in place each frame — never allocated. */
const BLEND: SurfaceVoice = { ...SURFACE_VOICES.dirt };

function blendSurfaces(wheels: readonly WheelState[], out: SurfaceVoice): number {
  let total = 0;
  for (const k of Object.keys(out) as (keyof SurfaceVoice)[]) out[k] = 0;
  for (const w of wheels) {
    if (!w.grounded) continue;
    // Weight by contact load: the wheel carrying the car decides what you hear.
    const wt = 0.25 + clamp(w.load / 6000, 0, 1.4);
    const v = SURFACE_VOICES[w.surface] ?? SURFACE_VOICES.dirt;
    for (const k of Object.keys(out) as (keyof SurfaceVoice)[]) out[k] += v[k] * wt;
    total += wt;
  }
  if (total <= 0) {
    const v = SURFACE_VOICES.dirt;
    for (const k of Object.keys(out) as (keyof SurfaceVoice)[]) out[k] = v[k];
    return 0;
  }
  for (const k of Object.keys(out) as (keyof SurfaceVoice)[]) out[k] /= total;
  return total;
}

/** Buses this module writes into. */
export interface SurfaceBuses {
  tyre: AudioNode;
  wind: AudioNode;
  fx: AudioNode;
}

export class SurfaceAudio {
  private readonly ctx: BaseAudioContext;
  private readonly rnd = mulberry32(0xbeef21);

  // --- roll ---------------------------------------------------------------
  private readonly rollSrc: AudioBufferSourceNode;
  private readonly rollHp: BiquadFilterNode;
  private readonly rollLp: BiquadFilterNode;
  private readonly rollBody: BiquadFilterNode;
  private readonly rollGain: GainNode;
  private readonly treadOsc: OscillatorNode;
  private readonly treadDepth: GainNode;

  // --- skid ---------------------------------------------------------------
  private readonly skidSrc: AudioBufferSourceNode;
  private readonly squealBp: BiquadFilterNode;
  private readonly squealRes: BiquadFilterNode;
  private readonly squealGain: GainNode;
  private readonly squealLfo: OscillatorNode;
  private readonly squealLfoDepth: GainNode;
  private readonly scrabbleBp: BiquadFilterNode;
  private readonly scrabbleHp: BiquadFilterNode;
  private readonly scrabbleGain: GainNode;
  private readonly scrabbleAmSrc: AudioBufferSourceNode;
  private readonly scrabbleAmLp: BiquadFilterNode;
  private readonly scrabbleAmDepth: GainNode;

  // --- wind ---------------------------------------------------------------
  private readonly windSrc: AudioBufferSourceNode;
  private readonly windLp: BiquadFilterNode;
  private readonly windHp: BiquadFilterNode;
  private readonly windGain: GainNode;
  private readonly whistleSrc: AudioBufferSourceNode;
  private readonly whistleBp: BiquadFilterNode;
  private readonly whistleGain: GainNode;

  // --- body rattle bed ----------------------------------------------------
  private readonly rattleSrc: AudioBufferSourceNode;
  private readonly rattleBp: BiquadFilterNode;
  private readonly rattleGain: GainNode;

  // --- event pools --------------------------------------------------------
  private readonly grains: NoiseBurstPool;
  private readonly clatter: NoiseBurstPool;
  private readonly impacts: NoiseBurstPool;
  private readonly tones: ToneBurstPool;

  // --- smoothed handles ---------------------------------------------------
  private readonly pRollGain: SmoothParam;
  private readonly pRollLp: SmoothParam;
  private readonly pRollHp: SmoothParam;
  private readonly pRollBody: SmoothParam;
  private readonly pRollBodyQ: SmoothParam;
  private readonly pRollBodyGain: SmoothParam;
  private readonly pTreadFreq: SmoothParam;
  private readonly pTreadDepth: SmoothParam;
  private readonly pSquealGain: SmoothParam;
  private readonly pSquealFreq: SmoothParam;
  private readonly pSquealRes: SmoothParam;
  private readonly pScrabbleGain: SmoothParam;
  private readonly pScrabbleFreq: SmoothParam;
  private readonly pWindGain: SmoothParam;
  private readonly pWindLp: SmoothParam;
  private readonly pWhistleGain: SmoothParam;
  private readonly pWhistleFreq: SmoothParam;
  private readonly pRattleGain: SmoothParam;
  private readonly pRattleFreq: SmoothParam;

  // --- running state ------------------------------------------------------
  private readonly prevCompression = [0, 0, 0, 0];
  private readonly wheelCooldown = [0, 0, 0, 0];
  private grainAccum = 0;
  private clatterAccum = 0;
  private roughness = 0;
  private landingRebound = -1;
  private disposed = false;

  /** Last blended surface parameters — exposed for the preview readout. */
  readonly blend: SurfaceVoice = { ...SURFACE_VOICES.dirt };

  constructor(ctx: BaseAudioContext, buses: SurfaceBuses) {
    this.ctx = ctx;

    // Separate noise beds so nothing shares a waveform with anything else;
    // correlated noise between two voices sounds like one voice being
    // filtered twice, which is exactly what we are trying to avoid.
    const white = createNoiseBuffer(ctx, 4.03, 'white', 8101);
    const whiteB = createNoiseBuffer(ctx, 3.71, 'white', 2237);
    const pink = createNoiseBuffer(ctx, 5.17, 'pink', 4409);
    // The tyre bed is partly correlated across the channels: fully
    // decorrelated noise puts the tyres *around* you, which is wrong — they
    // are two metres away in a known direction.
    const tyreBed = correlate(createNoiseBuffer(ctx, 4.51, 'white', 6151), 0.45);

    // ================= ROLL =================
    this.rollSrc = ctx.createBufferSource();
    this.rollSrc.buffer = tyreBed;
    this.rollSrc.loop = true;
    this.rollHp = biquad(ctx, 'highpass', 120, 0.7);
    this.rollLp = biquad(ctx, 'lowpass', 1400, 0.8);
    this.rollBody = biquad(ctx, 'peaking', 420, 1.0, 4);
    this.rollGain = gainNode(ctx, 0);
    chain(this.rollSrc, this.rollHp, this.rollLp, this.rollBody, this.rollGain);
    this.rollGain.connect(buses.tyre);
    this.rollSrc.start(0);

    // Tread thrum: a real tyre has discrete blocks, so at speed there is a
    // periodic component locked to wheel rotation. Modulating the roll gain
    // at block-passing frequency is enough to imply it.
    this.treadOsc = ctx.createOscillator();
    this.treadOsc.type = 'triangle';
    this.treadOsc.frequency.value = 1;
    this.treadDepth = gainNode(ctx, 0);
    this.treadOsc.connect(this.treadDepth).connect(this.rollGain.gain);
    this.treadOsc.start(0);

    // ================= SKID =================
    this.skidSrc = ctx.createBufferSource();
    this.skidSrc.buffer = white;
    this.skidSrc.loop = true;
    this.skidSrc.start(0);

    // Squeal: stick-slip of the tread blocks. A very narrow band on noise
    // gives a pitch without the dead purity of an oscillator, and a slow
    // frequency wobble keeps it from sitting still.
    this.squealBp = biquad(ctx, 'bandpass', 900, 17);
    this.squealRes = biquad(ctx, 'peaking', 1800, 6, 7);
    this.squealGain = gainNode(ctx, 0);
    chain(this.skidSrc, this.squealBp, this.squealRes, this.squealGain);
    this.squealGain.connect(buses.tyre);

    this.squealLfo = ctx.createOscillator();
    this.squealLfo.type = 'sine';
    this.squealLfo.frequency.value = 6.7;
    this.squealLfoDepth = gainNode(ctx, 38);
    this.squealLfo.connect(this.squealLfoDepth).connect(this.squealBp.frequency);
    this.squealLfo.start(0);

    // Scrabble: loose material being flung. Broadband, and amplitude-modulated
    // by slow noise so it churns instead of hissing.
    this.scrabbleHp = biquad(ctx, 'highpass', 400, 0.7);
    this.scrabbleBp = biquad(ctx, 'bandpass', 1900, 0.75);
    this.scrabbleGain = gainNode(ctx, 0);
    chain(this.skidSrc, this.scrabbleHp, this.scrabbleBp, this.scrabbleGain);
    this.scrabbleGain.connect(buses.tyre);

    this.scrabbleAmSrc = ctx.createBufferSource();
    this.scrabbleAmSrc.buffer = whiteB;
    this.scrabbleAmSrc.loop = true;
    this.scrabbleAmLp = biquad(ctx, 'lowpass', 11, 0.6);
    this.scrabbleAmDepth = gainNode(ctx, 0);
    chain(this.scrabbleAmSrc, this.scrabbleAmLp, this.scrabbleAmDepth);
    this.scrabbleAmDepth.connect(this.scrabbleGain.gain);
    this.scrabbleAmSrc.start(0);

    // ================= WIND =================
    this.windSrc = ctx.createBufferSource();
    this.windSrc.buffer = pink;
    this.windSrc.loop = true;
    this.windHp = biquad(ctx, 'highpass', 190, 0.6);
    this.windLp = biquad(ctx, 'lowpass', 700, 0.5);
    this.windGain = gainNode(ctx, 0);
    chain(this.windSrc, this.windHp, this.windLp, this.windGain);
    this.windGain.connect(buses.wind);
    this.windSrc.start(0);

    this.whistleSrc = ctx.createBufferSource();
    this.whistleSrc.buffer = whiteB;
    this.whistleSrc.loop = true;
    this.whistleBp = biquad(ctx, 'bandpass', 1500, 3.2);
    this.whistleGain = gainNode(ctx, 0);
    chain(this.whistleSrc, this.whistleBp, this.whistleGain);
    this.whistleGain.connect(buses.wind);
    this.whistleSrc.start(0);

    // ================= RATTLE BED =================
    this.rattleSrc = ctx.createBufferSource();
    this.rattleSrc.buffer = whiteB;
    this.rattleSrc.loop = true;
    this.rattleBp = biquad(ctx, 'bandpass', 2200, 1.6);
    this.rattleGain = gainNode(ctx, 0);
    chain(this.rattleSrc, this.rattleBp, this.rattleGain);
    this.rattleGain.connect(buses.fx);
    this.rattleSrc.start(0);

    // ================= EVENT POOLS =================
    this.grains = new NoiseBurstPool(ctx, buses.tyre, 10, white, 401);
    this.clatter = new NoiseBurstPool(ctx, buses.fx, 8, whiteB, 613);
    this.impacts = new NoiseBurstPool(ctx, buses.fx, 6, white, 887);
    this.tones = new ToneBurstPool(ctx, buses.fx, 6);

    // ================= HANDLES =================
    this.pRollGain = new SmoothParam(this.rollGain.gain, 0.06);
    this.pRollLp = new SmoothParam(this.rollLp.frequency, 0.09);
    this.pRollHp = new SmoothParam(this.rollHp.frequency, 0.12);
    this.pRollBody = new SmoothParam(this.rollBody.frequency, 0.12);
    this.pRollBodyQ = new SmoothParam(this.rollBody.Q, 0.12);
    this.pRollBodyGain = new SmoothParam(this.rollBody.gain, 0.12);
    this.pTreadFreq = new SmoothParam(this.treadOsc.frequency, 0.06);
    this.pTreadDepth = new SmoothParam(this.treadDepth.gain, 0.1);
    this.pSquealGain = new SmoothParam(this.squealGain.gain, 0.05);
    this.pSquealFreq = new SmoothParam(this.squealBp.frequency, 0.06);
    this.pSquealRes = new SmoothParam(this.squealRes.frequency, 0.08);
    this.pScrabbleGain = new SmoothParam(this.scrabbleGain.gain, 0.05);
    this.pScrabbleFreq = new SmoothParam(this.scrabbleBp.frequency, 0.08);
    this.pWindGain = new SmoothParam(this.windGain.gain, 0.07);
    this.pWindLp = new SmoothParam(this.windLp.frequency, 0.1);
    this.pWhistleGain = new SmoothParam(this.whistleGain.gain, 0.08);
    this.pWhistleFreq = new SmoothParam(this.whistleBp.frequency, 0.1);
    this.pRattleGain = new SmoothParam(this.rattleGain.gain, 0.08);
    this.pRattleFreq = new SmoothParam(this.rattleBp.frequency, 0.12);
  }

  // -------------------------------------------------------------------------

  update(dt: number, s: VehicleState, wheelRadius: number): void {
    if (this.disposed) return;
    const now = this.ctx.currentTime;
    const speed = Math.min(Math.abs(s.speed), 90);
    const sn = speed / REF_SPEED;

    const contactWeight = blendSurfaces(s.wheels, BLEND);
    // Copy out for the preview readout without allocating.
    for (const k of Object.keys(BLEND) as (keyof SurfaceVoice)[]) this.blend[k] = BLEND[k];

    let grounded = 0;
    for (const w of s.wheels) if (w.grounded) grounded++;
    const groundFrac = s.wheels.length ? grounded / s.wheels.length : 0;

    // ---------------- roll ----------------
    // Rolling noise is roughly proportional to speed^1.4 in level and rises
    // in brightness with speed as well — the contact patch flexes faster.
    const rollAmt = smoothstep(0.15, 2.2, speed) * Math.pow(clamp(sn, 0, 3), 0.7);
    this.pRollGain.set(clamp(BLEND.level * rollAmt * groundFrac * 0.5, 0, 1.2), now);
    this.pRollLp.set(clamp(BLEND.lp * (0.42 + 0.58 * clamp(sn, 0, 2.2)), 120, 16000), now);
    this.pRollHp.set(clamp(BLEND.hp, 30, 2000), now);
    this.pRollBody.set(clamp(BLEND.bp, 60, 8000), now);
    this.pRollBodyQ.set(clamp(BLEND.bpQ, 0.1, 12), now);
    this.pRollBodyGain.set(clamp(BLEND.bpGain, -20, 20), now);

    // Tread block passing frequency: wheel revs/s times an assumed 18 blocks.
    const wheelRev = speed / (2 * Math.PI * Math.max(wheelRadius, 0.05));
    this.pTreadFreq.set(clamp(wheelRev * 18, 0.05, 900), now);
    // Only hard surfaces have a defined enough tread note to bother with.
    const treadAmt = clamp((BLEND.bpQ - 0.6) / 2.4, 0, 1) * smoothstep(3, 12, speed);
    this.pTreadDepth.set(treadAmt * 0.16 * BLEND.level * groundFrac, now);

    // ---------------- slip ----------------
    // Peak slip across the wheels rather than the mean: one wheel lighting up
    // should be audible even when the other three are gripping.
    let slipLong = 0;
    let slipLat = 0;
    for (const w of s.wheels) {
      if (!w.grounded) continue;
      slipLong = Math.max(slipLong, Math.abs(w.slipRatio));
      slipLat = Math.max(slipLat, Math.abs(w.slipAngle));
    }
    // The tyre model peaks around 0.12 slip ratio / 0.15 rad; audible squeal
    // starts a little before the peak and saturates well past it.
    const slipL = smoothstep(0.07, 0.55, slipLong);
    const slipA = smoothstep(0.06, 0.42, slipLat);
    const slip = clamp(Math.max(slipL, slipA * 1.05), 0, 1);
    // Slip at a standstill is meaningless — the wheel has to be moving
    // relative to the ground for anything to be rubbed.
    const slipSpeedGate = smoothstep(0.6, 4.0, speed + slipLong * wheelRadius * 8);
    const slipAmt = slip * slipSpeedGate * groundFrac;

    const squealAmt = slipAmt * BLEND.squeal;
    const scrabbleAmt = slipAmt * BLEND.scrabble;

    // Squeal pitch climbs with how hard the tyre is being worked, which is
    // what makes a long slide sound like it is *going* somewhere.
    const squealF = BLEND.squealFreq * (0.78 + 0.55 * slip + 0.10 * clamp(sn, 0, 2));
    this.pSquealFreq.set(clamp(squealF, 120, 6000), now);
    this.pSquealRes.set(clamp(squealF * 2.02, 200, 12000), now);
    this.pSquealGain.set(clamp(Math.pow(squealAmt, 1.2) * 0.62, 0, 1), now);

    this.pScrabbleFreq.set(clamp(1100 + 1500 * clamp(sn, 0, 2) + 900 * slip, 300, 9000), now);
    const scrabbleBase = clamp(Math.pow(scrabbleAmt, 1.1) * 0.32, 0, 1);
    this.pScrabbleGain.set(scrabbleBase, now);
    this.scrabbleAmDepth.gain.setTargetAtTime(scrabbleBase * 0.85, now, 0.06);

    // ---------------- discrete stone strikes ----------------
    // Loose material thrown up into the wheel arches. Rate scales with speed
    // and with slip (a spinning wheel digs), so a wheelspin on gravel reads as
    // a shower of stones rather than a change of filter.
    const grainRate =
      BLEND.grainRate * clamp(sn, 0, 2.6) * groundFrac * (1 + 2.6 * slipAmt) * contactSc(contactWeight);
    this.grainAccum += grainRate * dt;
    let fired = 0;
    while (this.grainAccum >= 1 && fired < 6) {
      this.grainAccum -= 1;
      fired++;
      const r = this.rnd();
      this.grains.fire(now + this.rnd() * dt, {
        gain: BLEND.grainGain * (0.35 + 0.9 * r * r) * clamp(0.4 + sn * 0.5, 0, 1.6),
        freq: BLEND.grainFreq * (0.55 + 1.1 * this.rnd()),
        q: BLEND.grainQ * (0.7 + 0.6 * this.rnd()),
        attack: 0.0008,
        decay: BLEND.grainDecay * (0.6 + 0.9 * this.rnd()),
        pan: (this.rnd() - 0.5) * 1.5,
        rateJitter: 0.5,
      });
    }
    if (this.grainAccum > 8) this.grainAccum = 8;

    // ---------------- suspension ----------------
    let rough = 0;
    for (let i = 0; i < s.wheels.length && i < 4; i++) {
      const w = s.wheels[i];
      const c = clamp(w.compression, 0, 1);
      const rate = (c - this.prevCompression[i]) / Math.max(dt, 1e-4);
      this.prevCompression[i] = c;
      this.wheelCooldown[i] -= dt;
      rough += Math.abs(rate);

      if (!w.grounded) continue;
      const pan = i % 2 === 0 ? -0.5 : 0.5;

      // A hit: the wheel is being driven up its travel fast.
      if (rate > 2.2 && this.wheelCooldown[i] <= 0) {
        const g = clamp((rate - 2.2) / 9, 0, 1);
        this.wheelCooldown[i] = 0.055;
        this.tones.fire(now, {
          gain: 0.16 + 0.34 * g,
          freq: 108 + 46 * g,
          toFreq: 44,
          attack: 0.002,
          decay: 0.09 + 0.11 * g,
          filterFreq: 420 + 500 * g,
          filterQ: 1.1,
          pan,
        });
        this.impacts.fire(now, {
          gain: (0.08 + 0.22 * g) * (0.6 + 0.5 * BLEND.rough),
          freq: 300 + 900 * g,
          q: 1.3,
          attack: 0.001,
          decay: 0.05 + 0.06 * g,
          sweepTo: 140,
          pan,
          rateJitter: 0.3,
        });
      }

      // Bottom-out: the bump stop. Metal on rubber on metal — high-Q ring.
      if (c > 0.93 && rate > 0.6 && this.wheelCooldown[i] <= 0.04) {
        this.wheelCooldown[i] = 0.09;
        const g = clamp(rate / 8, 0.2, 1);
        this.impacts.fire(now, {
          gain: 0.20 * g,
          freq: 1500 + this.rnd() * 900,
          q: 11,
          attack: 0.0008,
          decay: 0.10 + 0.08 * g,
          pan,
          rateJitter: 0.4,
        });
        this.tones.fire(now + 0.002, {
          gain: 0.22 * g,
          freq: 78,
          toFreq: 40,
          attack: 0.0015,
          decay: 0.16,
          filterFreq: 380,
          pan,
        });
      }
    }

    // Roughness drives both the continuous rattle bed and the sparse clatter.
    const roughTarget = clamp((rough / 4) * BLEND.rough * groundFrac, 0, 1.4);
    this.roughness += (roughTarget - this.roughness) * clamp(dt * 7, 0, 1);
    this.pRattleGain.set(clamp(this.roughness * 0.06 * (0.4 + 0.6 * clamp(sn, 0, 2)), 0, 0.14), now);
    this.pRattleFreq.set(clamp(1600 + 1400 * this.roughness, 400, 6000), now);

    // Discrete loose-trim clatter: doors, mirrors, the roll cage, the tools in
    // the back. Sparse, random, and completely defining for a beaten-up 4x4.
    const clatterRate = this.roughness * 14 * clamp(0.25 + sn * 0.9, 0, 2.4);
    this.clatterAccum += clatterRate * dt;
    let cf = 0;
    while (this.clatterAccum >= 1 && cf < 4) {
      this.clatterAccum -= 1;
      cf++;
      const r = this.rnd();
      this.clatter.fire(now + this.rnd() * dt, {
        gain: 0.035 + 0.075 * r * this.roughness,
        freq: 700 + this.rnd() * 3400,
        q: 4 + this.rnd() * 12,
        attack: 0.0008,
        decay: 0.02 + this.rnd() * 0.075,
        pan: (this.rnd() - 0.5) * 1.6,
        rateJitter: 0.5,
      });
    }
    if (this.clatterAccum > 6) this.clatterAccum = 6;

    // Occasional suspension creak when the car is articulating slowly — the
    // sound of an old bush being twisted rather than a shock being hit.
    if (this.roughness > 0.12 && this.rnd() < dt * 1.6 * this.roughness) {
      this.tones.fire(now, {
        gain: 0.05 + 0.06 * this.rnd(),
        freq: 190 + this.rnd() * 260,
        toFreq: 150 + this.rnd() * 200,
        attack: 0.02,
        decay: 0.14 + this.rnd() * 0.2,
        type: 'sawtooth',
        filterFreq: 900,
        filterQ: 3.5,
        pan: (this.rnd() - 0.5) * 1.2,
      });
    }

    // ---------------- wind ----------------
    // Level rises with roughly v^1.6 (aero noise is somewhere between v^1 and
    // v^2), brightness with v. Airborne it swells and opens right up because
    // there is suddenly nothing between you and the air.
    const air = s.airborne ? 1 : 0;
    const windLevel = Math.pow(clamp(speed / 26, 0, 2.4), 1.6);
    this.pWindGain.set(clamp(windLevel * (0.16 + 0.20 * air), 0, 0.7), now, air ? 0.05 : 0.12);
    this.pWindLp.set(clamp(340 + 165 * speed + 1800 * air, 200, 12000), now, air ? 0.06 : 0.14);
    this.pWhistleFreq.set(clamp(900 + 38 * speed, 300, 5200), now);
    this.pWhistleGain.set(clamp(windLevel * 0.05 * (0.4 + 0.9 * air), 0, 0.18), now);

    // Landing rebound: a second, softer thump a beat after touchdown, which is
    // the springs pushing the body back up. Scheduled, not polled.
    if (this.landingRebound > 0) {
      this.landingRebound -= dt;
      if (this.landingRebound <= 0) this.landingRebound = -1;
    }
  }

  // -------------------------------------------------------------------------

  /**
   * Touchdown. `impactSpeed` is the vertical speed killed, m/s — roughly 4 for
   * dropping off a kerb, 12+ for a proper jump.
   */
  playLanding(impactSpeed: number): void {
    if (this.disposed) return;
    const now = this.ctx.currentTime;
    const g = clamp(impactSpeed / 13, 0.12, 1.25);

    // The weight: a low sine dropping an octave and a half in 300 ms.
    this.tones.fire(now, {
      gain: 0.55 * g,
      freq: 112 + 40 * g,
      toFreq: 33,
      attack: 0.003,
      decay: 0.24 + 0.16 * g,
      filterFreq: 500,
      filterQ: 0.9,
    });
    // The ground: a broadband slap, dark.
    this.impacts.fire(now, {
      gain: 0.42 * g,
      freq: 220,
      q: 0.7,
      type: 'lowpass',
      attack: 0.001,
      decay: 0.15 + 0.1 * g,
      sweepTo: 90,
    });
    // The vehicle: the suspension and everything bolted to it.
    this.impacts.fire(now + 0.004, {
      gain: 0.20 * g,
      freq: 1250 + 700 * g,
      q: 5,
      attack: 0.0008,
      decay: 0.11,
      pan: -0.2,
      rateJitter: 0.3,
    });
    this.impacts.fire(now + 0.011, {
      gain: 0.16 * g,
      freq: 2400,
      q: 8,
      attack: 0.0008,
      decay: 0.14,
      pan: 0.25,
      rateJitter: 0.35,
    });

    // Rebound, one damped half-cycle later. Front spring rate 33 kN/m on a
    // 445 kg corner gives ~1.37 Hz, so ~0.36 s to come back up.
    const rb = now + 0.34;
    this.tones.fire(rb, {
      gain: 0.20 * g,
      freq: 74,
      toFreq: 38,
      attack: 0.01,
      decay: 0.2,
      filterFreq: 340,
    });
    this.clatter.fire(rb + 0.01, {
      gain: 0.13 * g,
      freq: 1700,
      q: 7,
      attack: 0.001,
      decay: 0.13,
      pan: 0.3,
      rateJitter: 0.4,
    });
    // And a shower of everything loose in the cabin.
    for (let i = 0; i < 5; i++) {
      this.clatter.fire(now + 0.02 + this.rnd() * 0.22, {
        gain: (0.05 + 0.09 * this.rnd()) * g,
        freq: 800 + this.rnd() * 3200,
        q: 5 + this.rnd() * 14,
        attack: 0.0008,
        decay: 0.03 + this.rnd() * 0.1,
        pan: (this.rnd() - 0.5) * 1.7,
        rateJitter: 0.5,
      });
    }
    this.landingRebound = 0.34;
  }

  /** Bodywork against something solid. */
  playCollision(impactSpeed: number): void {
    if (this.disposed) return;
    const now = this.ctx.currentTime;
    const g = clamp(impactSpeed / 11, 0.1, 1.3);

    // The crunch itself: a burst of high-Q bands at inharmonic frequencies,
    // scattered over ~40 ms. Inharmonic is the whole trick — harmonically
    // related partials read as a bell, inharmonic ones as sheet metal.
    const partials = [780, 1170, 1490, 2050, 2660, 3400, 4700];
    for (let i = 0; i < partials.length; i++) {
      const jitterHz = partials[i] * (0.85 + this.rnd() * 0.3);
      this.impacts.fire(now + this.rnd() * 0.035, {
        gain: (0.20 + 0.16 * this.rnd()) * g * (1 - i / (partials.length + 2)),
        freq: jitterHz,
        q: 7 + this.rnd() * 18,
        attack: 0.0006,
        decay: 0.04 + this.rnd() * 0.22,
        pan: (this.rnd() - 0.5) * 1.2,
        rateJitter: 0.4,
      });
    }
    // Broadband smash under it all.
    this.impacts.fire(now, {
      gain: 0.34 * g,
      freq: 1500,
      q: 0.55,
      attack: 0.0006,
      decay: 0.09,
      sweepTo: 320,
    });
    // And the mass behind it.
    this.tones.fire(now, {
      gain: 0.42 * g,
      freq: 96,
      toFreq: 41,
      attack: 0.002,
      decay: 0.19,
      filterFreq: 420,
    });
    // Trailing debris rattle.
    for (let i = 0; i < 6; i++) {
      this.clatter.fire(now + 0.03 + this.rnd() * 0.35, {
        gain: (0.05 + 0.1 * this.rnd()) * g,
        freq: 900 + this.rnd() * 4200,
        q: 6 + this.rnd() * 16,
        attack: 0.0006,
        decay: 0.02 + this.rnd() * 0.13,
        pan: (this.rnd() - 0.5) * 1.8,
        rateJitter: 0.5,
      });
    }
  }

  /** 0..1 — how hard the ground is currently shaking the car. For ducking. */
  get roughnessLevel(): number {
    return clamp(this.roughness, 0, 1);
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const s of [
      this.rollSrc,
      this.skidSrc,
      this.windSrc,
      this.whistleSrc,
      this.rattleSrc,
      this.scrabbleAmSrc,
    ]) {
      try {
        s.stop();
      } catch {
        /* already stopped */
      }
    }
    for (const o of [this.treadOsc, this.squealLfo]) {
      try {
        o.stop();
      } catch {
        /* already stopped */
      }
    }
    this.grains.dispose();
    this.clatter.dispose();
    this.impacts.dispose();
    this.tones.dispose();
    disconnectAll([
      this.rollSrc, this.rollHp, this.rollLp, this.rollBody, this.rollGain,
      this.treadOsc, this.treadDepth,
      this.skidSrc, this.squealBp, this.squealRes, this.squealGain,
      this.squealLfo, this.squealLfoDepth,
      this.scrabbleBp, this.scrabbleHp, this.scrabbleGain,
      this.scrabbleAmSrc, this.scrabbleAmLp, this.scrabbleAmDepth,
      this.windSrc, this.windLp, this.windHp, this.windGain,
      this.whistleSrc, this.whistleBp, this.whistleGain,
      this.rattleSrc, this.rattleBp, this.rattleGain,
    ]);
  }
}

/** Grain rate falls off when only one corner is loaded. */
function contactSc(weight: number): number {
  return clamp(0.35 + weight * 0.5, 0, 1.2);
}

/**
 * Bleed one channel of a stereo buffer into the other, in place. `amount` 0
 * leaves it fully decorrelated (maximally wide), 1 makes it mono.
 */
function correlate(buf: AudioBuffer, amount: number): AudioBuffer {
  if (buf.numberOfChannels < 2) return buf;
  const a = buf.getChannelData(0);
  const b = buf.getChannelData(1);
  const k = clamp(amount, 0, 1);
  const norm = 1 / Math.sqrt(1 + k * k);
  for (let i = 0; i < b.length; i++) {
    b[i] = (b[i] * (1 - k) + a[i] * k) * (1 + k) * norm;
  }
  return buf;
}

export { SURFACE_VOICES, SURFACE_KEYS };
export type { SurfaceVoice };
