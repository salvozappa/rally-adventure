/**
 * ============================================================================
 *  GAME AUDIO — the mixer, the bus structure, and the bridge from physics
 *  state to synthesis parameters.
 * ============================================================================
 *
 * SIGNAL FLOW
 *
 *      engine ─┬──────────────────────────────┐
 *      tyre   ─┼──────────────────────────────┤
 *      fx     ─┼──────────────────────────────┼─▶ preMaster ─▶ limiter
 *      wind   ─┼──────────────────────────────┤        │           │
 *      ambient─┘                              │        │           ▼
 *              └─ sends ─▶ convolver ─▶ return┘        │      soft clip
 *                                                      │           │
 *                                                      └──▶ master gain ─▶ out
 *
 * The soft clipper after the compressor is not belt-and-braces, it is the
 * thing that makes "never clips" a structural guarantee rather than a hope: a
 * WaveShaper maps everything outside [-1, 1] onto the endpoints of its curve,
 * and that curve asymptotes below 1. So the output peak can never exceed the
 * master gain, whatever the compressor lets through on a transient.
 *
 * ENGINE LOAD
 *
 * `VehicleState` publishes RPM, gear and clutch but not throttle, so load is
 * reconstructed rather than read. Two independent estimates are summed:
 *
 *   1. Connected: how much crank torque it must be taking to produce the
 *      longitudinal acceleration we can see, against aero drag and rolling
 *      resistance, referred back through the gear ratio and normalised by the
 *      torque available at this RPM. Because `localAccel` is *specific force*
 *      it already contains the road gradient, so climbing a hill correctly
 *      reads as load without anyone having to tell the audio about hills.
 *
 *   2. Free: crank inertia times angular acceleration. This is what carries
 *      a standing launch and a neutral blip, where the wheels tell you
 *      nothing about what the driver is asking for.
 *
 * The result is negative on a trailing throttle, which is exactly the signal
 * the engine synth needs to cross into its overrun voice.
 */

import type { VehicleState } from '../types';
import { JEEP_TUNING } from '../physics/VehicleTuning';
import { EngineSynth, type EngineDriveState } from './EngineSynth';
import { SurfaceAudio } from './SurfaceAudio';
import {
  SmoothParam,
  ToneBurstPool,
  biquad,
  chain,
  clamp,
  createImpulseResponse,
  createNoiseBuffer,
  createSaturationCurve,
  disconnectAll,
  gainNode,
  mulberry32,
  panner,
  smoothstep,
} from './dsp';

const RAD_S_PER_RPM = (2 * Math.PI) / 60;
const G = 9.81;

/**
 * Static bus balance, measured rather than guessed: these are the multipliers
 * that put a wide-open-throttle engine at about -19 dBFS RMS with the tyres
 * 10 dB under it, wind 16 dB under, body rattle 18 dB under and the ambience
 * bed 22 dB under. Ducking and speed/load scaling all happen on top of these.
 */
const BUS_MIX = {
  engine: 1.0,
  tyre: 0.26,
  wind: 0.45,
  fx: 0.26,
  ambience: 0.4,
} as const;

export interface GameAudioOptions {
  /** 0..1, default 0.85. */
  masterVolume?: number;
  /** Supply an OfflineAudioContext to render rather than play. */
  context?: BaseAudioContext;
  /** Start muted (e.g. restoring a user preference). */
  muted?: boolean;
}

export type AudioBusName = 'master' | 'engine' | 'tyre' | 'wind' | 'ambience' | 'fx';

export class GameAudio {
  private readonly ctx: BaseAudioContext;
  private readonly ownsContext: boolean;

  // --- buses --------------------------------------------------------------
  private readonly engineBus: GainNode;
  private readonly tyreBus: GainNode;
  private readonly windBus: GainNode;
  private readonly ambienceBus: GainNode;
  private readonly fxBus: GainNode;
  private readonly preMaster: GainNode;
  private readonly masterHp: BiquadFilterNode;
  private readonly limiter: DynamicsCompressorNode;
  private readonly softClip: WaveShaperNode;
  private readonly master: GainNode;

  // --- reverb -------------------------------------------------------------
  private readonly convolver: ConvolverNode;
  private readonly reverbReturn: GainNode;
  private readonly sendEngine: GainNode;
  private readonly sendTyre: GainNode;
  private readonly sendFx: GainNode;

  // --- stereo placement ---------------------------------------------------
  private readonly enginePan: StereoPannerNode;
  private readonly tyrePan: StereoPannerNode;

  // --- voices -------------------------------------------------------------
  private readonly engine: EngineSynth;
  private readonly surface: SurfaceAudio;

  // --- ambience -----------------------------------------------------------
  private readonly ambSrc: AudioBufferSourceNode;
  private readonly ambLp: BiquadFilterNode;
  private readonly ambHp: BiquadFilterNode;
  private readonly ambGain: GainNode;
  private readonly ambLfo: OscillatorNode;
  private readonly ambLfoDepth: GainNode;
  private readonly birds: ToneBurstPool;
  private readonly pAmbGain: SmoothParam;
  private readonly pEngineBus: SmoothParam;
  private readonly pTyreBus: SmoothParam;
  private readonly pWindBus: SmoothParam;
  private readonly pSendEngine: SmoothParam;

  // --- state --------------------------------------------------------------
  private readonly driveState: EngineDriveState = {
    rpm: JEEP_TUNING.drivetrain.engine.idleRpm,
    load: 0,
    shifting: false,
    clutch: 1,
    speed: 0,
  };
  private readonly rnd = mulberry32(0x1234abc);
  /**
   * Per-bus user trim. Kept separate from the bus gains because `update`
   * rewrites those every frame for ducking; without this split any external
   * level change would be silently stamped out on the next frame.
   */
  private readonly trims: Record<AudioBusName, number> = {
    master: 1, engine: 1, tyre: 1, wind: 1, ambience: 1, fx: 1,
  };
  private volume: number;
  private muted: boolean;
  private prevGear = 1;
  private prevEngOmega = JEEP_TUNING.drivetrain.engine.idleRpm * RAD_S_PER_RPM;
  private smoothDOmega = 0;
  private shiftHold = 0;
  private birdTimer = 3;
  private readyPromise: Promise<void>;
  private disposed = false;

  /** Last reconstructed engine load, -1..1. Exposed for the preview HUD. */
  lastLoad = 0;

  constructor(opts: GameAudioOptions = {}) {
    this.volume = clamp(opts.masterVolume ?? 0.85, 0, 1);
    this.muted = opts.muted ?? false;

    if (opts.context) {
      this.ctx = opts.context;
      this.ownsContext = false;
    } else {
      const Ctor: typeof AudioContext =
        (globalThis as unknown as { AudioContext?: typeof AudioContext }).AudioContext ??
        (globalThis as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext!;
      this.ctx = new Ctor({ latencyHint: 'interactive' });
      this.ownsContext = true;
    }
    const ctx = this.ctx;

    // ===================== master chain =====================
    this.master = gainNode(ctx, this.muted ? 0 : this.volume);
    // 2:1 above -9 dBFS with a fast attack: this is doing gentle programme
    // control, catching the 12 dB gap between an idle and a full-throttle
    // landing, not smashing the mix flat.
    this.limiter = ctx.createDynamicsCompressor();
    this.limiter.threshold.value = -9;
    this.limiter.knee.value = 6;
    this.limiter.ratio.value = 12;
    this.limiter.attack.value = 0.003;
    this.limiter.release.value = 0.16;
    // Hard ceiling. tanh(x*1.6)/tanh(1.6) reaches exactly 1 at x=1 and the
    // WaveShaper clamps anything beyond, so nothing downstream can exceed it.
    this.softClip = ctx.createWaveShaper();
    this.softClip.curve = createSaturationCurve(1.6, 4096);
    this.softClip.oversample = '2x';

    this.preMaster = gainNode(ctx, 1);
    // Every soft-clipper in the chain rectifies asymmetric material a little,
    // and DC offset costs headroom without being audible. Strip it once for
    // the whole mix, below anything the game ever generates.
    this.masterHp = biquad(ctx, 'highpass', 18, 0.6);
    chain(this.preMaster, this.masterHp, this.limiter, this.softClip, this.master);
    if ((ctx as BaseAudioContext).destination) this.master.connect(ctx.destination);

    // ===================== reverb send =====================
    // Short, sparse and bright: an outdoor slap off trees and rock, not a hall.
    this.convolver = ctx.createConvolver();
    this.convolver.normalize = true;
    this.convolver.buffer = createImpulseResponse(ctx, 0.95, 3.6, 0.6, 4242);
    this.reverbReturn = gainNode(ctx, 0.5);
    // Roll the very bottom off the send: low frequencies in a reverb outdoors
    // just make everything muddy, and there is no wall to reflect them.
    const revHp = biquad(ctx, 'highpass', 380, 0.7);
    chain(this.convolver, revHp, this.reverbReturn, this.preMaster);

    this.sendEngine = gainNode(ctx, 0.1);
    this.sendTyre = gainNode(ctx, 0.09);
    this.sendFx = gainNode(ctx, 0.22);
    this.sendEngine.connect(this.convolver);
    this.sendTyre.connect(this.convolver);
    this.sendFx.connect(this.convolver);

    // ===================== buses =====================
    // Engine dead centre — it is bolted to you. Tyres slightly spread. Wind is
    // already wide because its noise bed is decorrelated across the channels.
    this.engineBus = gainNode(ctx, 1);
    this.enginePan = panner(ctx, 0);
    this.engineBus.connect(this.enginePan);
    this.enginePan.connect(this.preMaster);
    this.engineBus.connect(this.sendEngine);

    this.tyreBus = gainNode(ctx, BUS_MIX.tyre);
    this.tyrePan = panner(ctx, 0.05);
    this.tyreBus.connect(this.tyrePan);
    this.tyrePan.connect(this.preMaster);
    this.tyreBus.connect(this.sendTyre);

    this.windBus = gainNode(ctx, BUS_MIX.wind);
    this.windBus.connect(this.preMaster);

    this.fxBus = gainNode(ctx, BUS_MIX.fx);
    this.fxBus.connect(this.preMaster);
    this.fxBus.connect(this.sendFx);

    this.ambienceBus = gainNode(ctx, BUS_MIX.ambience);
    this.ambienceBus.connect(this.preMaster);

    // ===================== voices =====================
    this.engine = new EngineSynth(ctx, JEEP_TUNING.drivetrain.engine, createNoiseBuffer(ctx, 2.9, 'white', 1777));
    this.engine.output.connect(this.engineBus);

    this.surface = new SurfaceAudio(ctx, {
      tyre: this.tyreBus,
      wind: this.windBus,
      fx: this.fxBus,
    });

    // ===================== ambience =====================
    // Wind moving through trees: pink noise band, slowly breathing. Silence
    // reads as a broken build; this costs three nodes and fixes that.
    this.ambSrc = ctx.createBufferSource();
    this.ambSrc.buffer = createNoiseBuffer(ctx, 6.3, 'pink', 31337);
    this.ambSrc.loop = true;
    this.ambHp = biquad(ctx, 'highpass', 280, 0.5);
    this.ambLp = biquad(ctx, 'lowpass', 2100, 0.4);
    this.ambGain = gainNode(ctx, 0.05);
    chain(this.ambSrc, this.ambHp, this.ambLp, this.ambGain);
    this.ambGain.connect(this.ambienceBus);
    this.ambSrc.start(0);

    this.ambLfo = ctx.createOscillator();
    this.ambLfo.type = 'sine';
    this.ambLfo.frequency.value = 0.077;
    this.ambLfoDepth = gainNode(ctx, 620);
    this.ambLfo.connect(this.ambLfoDepth).connect(this.ambLp.frequency);
    this.ambLfo.start(0);

    this.birds = new ToneBurstPool(ctx, this.ambienceBus, 3);

    // ===================== handles =====================
    this.pAmbGain = new SmoothParam(this.ambGain.gain, 0.35);
    this.pEngineBus = new SmoothParam(this.engineBus.gain, 0.06);
    this.pTyreBus = new SmoothParam(this.tyreBus.gain, 0.08);
    this.pWindBus = new SmoothParam(this.windBus.gain, 0.1);
    this.pSendEngine = new SmoothParam(this.sendEngine.gain, 0.2);

    // The graph sits frozen at t = 0 until the context is resumed, and every
    // filter and the convolver charge from zero the moment it starts. Fading
    // the master in over the first 600 ms of context time turns what would be
    // an audible thump into nothing at all — and because a suspended context's
    // clock does not advance, this works however long the user takes to click.
    this.master.gain.setValueAtTime(0, 0);
    this.master.gain.linearRampToValueAtTime(this.muted ? 0 : this.volume, 0.6);

    const engine = this.engine;
    this.readyPromise = (async () => {
      // Give the worklet a chance to install before anyone renders offline.
      const t0 = Date.now();
      while (!engine.isReady && Date.now() - t0 < 4000) {
        await new Promise<void>((r) => setTimeout(r, 8));
      }
    })();
  }

  // -------------------------------------------------------------------------
  //  Public API
  // -------------------------------------------------------------------------

  /**
   * Browsers create an AudioContext in the `suspended` state and only allow it
   * to start from inside a user gesture. This is safe to call as often as you
   * like, from anywhere, including before the graph has anything to say.
   */
  async resume(): Promise<void> {
    const ctx = this.ctx as AudioContext;
    if (this.disposed) return;
    if (typeof ctx.resume !== 'function') return;
    if (ctx.state === 'running' || ctx.state === 'closed') return;
    try {
      await ctx.resume();
    } catch {
      /* Still suspended: the gesture was not trusted. Try again next time. */
    }
  }

  /** Resolves once the engine excitation source has settled. */
  whenReady(): Promise<void> {
    return this.readyPromise;
  }

  update(dt: number, s: VehicleState): void {
    if (this.disposed) return;
    const step = clamp(Number.isFinite(dt) ? dt : 1 / 60, 1 / 480, 0.1);
    const now = this.ctx.currentTime;

    // ---- gear changes ----
    if (s.gear !== this.prevGear) {
      const jump = Math.abs(s.gear - this.prevGear);
      // A gearchange is a real mechanical event with mass behind it.
      this.engine.playShiftClunk(clamp(0.55 + 0.25 * jump, 0, 1));
      this.shiftHold = JEEP_TUNING.drivetrain.transmission.shiftTime;
      this.prevGear = s.gear;
    }
    if (this.shiftHold > 0) this.shiftHold = Math.max(0, this.shiftHold - step);

    // ---- load reconstruction ----
    const load = this.estimateLoad(step, s);
    this.lastLoad = load;

    const d = this.driveState;
    d.rpm = s.engineRpm;
    d.load = load;
    d.clutch = clamp(s.clutch, 0, 1);
    d.speed = s.forwardSpeed;
    // The clutch dipping is itself evidence of a shift, and it lines the audio
    // torque-cut up with the drivetrain's rather than with the gear *number*,
    // which only changes at the end of the shift.
    d.shifting = this.shiftHold > 0 || (d.clutch < 0.55 && Math.abs(s.forwardSpeed) > 2);

    this.engine.update(step, d);
    this.surface.update(step, s, JEEP_TUNING.tire.radius);

    // ---- mix ----
    const rpmNorm = clamp(
      (s.engineRpm - JEEP_TUNING.drivetrain.engine.idleRpm) /
        (JEEP_TUNING.drivetrain.engine.redlineRpm - JEEP_TUNING.drivetrain.engine.idleRpm),
      0,
      1.1,
    );
    const activity = clamp(0.45 * rpmNorm + 0.55 * clamp(load, 0, 1), 0, 1);

    // Duck the ambience under engine load. The bed is there to stop silence,
    // not to be heard over a wide-open throttle.
    this.pAmbGain.set(
      this.trims.ambience *
        0.055 *
        (1 - 0.78 * activity) *
        (1 - 0.5 * smoothstep(4, 22, Math.abs(s.speed))),
      now,
    );

    // Tyres are pulled down a little when the engine is shouting, otherwise
    // gravel at 25 m/s and full throttle turns into one undifferentiated hiss.
    this.pTyreBus.set(this.trims.tyre * BUS_MIX.tyre * (1 - 0.3 * activity), now);
    this.pEngineBus.set(this.trims.engine * BUS_MIX.engine, now);
    this.pWindBus.set(this.trims.wind * BUS_MIX.wind, now);
    // More reverb send when the engine is loud: a shouting engine excites the
    // hillside, an idling one does not.
    this.pSendEngine.set(0.06 + 0.13 * activity, now);

    // ---- birds ----
    // Sparse, and only when the engine is not drowning them out anyway.
    this.birdTimer -= step;
    if (this.birdTimer <= 0) {
      this.birdTimer = 3.5 + this.rnd() * 8;
      if (activity < 0.45) this.chirp(now);
    }
  }

  playLanding(impactSpeed: number): void {
    if (this.disposed) return;
    const v = Number.isFinite(impactSpeed) ? Math.abs(impactSpeed) : 0;
    this.surface.playLanding(v);
  }

  playCollision(impactSpeed: number): void {
    if (this.disposed) return;
    const v = Number.isFinite(impactSpeed) ? Math.abs(impactSpeed) : 0;
    this.surface.playCollision(v);
  }

  setVolume(v: number): void {
    this.volume = clamp(Number.isFinite(v) ? v : 0, 0, 1);
    this.applyMaster();
  }

  getVolume(): number {
    return this.volume;
  }

  setMuted(m: boolean): void {
    this.muted = !!m;
    this.applyMaster();
  }

  isMuted(): boolean {
    return this.muted;
  }

  /** The context, for anyone who needs to schedule against the same clock. */
  get context(): BaseAudioContext {
    return this.ctx;
  }

  /** True if the per-sample AudioWorklet excitation is running. */
  get usingWorklet(): boolean {
    return this.engine.usingWorklet;
  }

  /**
   * An analyser tapped off a bus, for meters and the preview page. Created on
   * demand — never call this per frame.
   */
  createAnalyser(bus: AudioBusName = 'master', fftSize = 4096): AnalyserNode {
    const a = this.ctx.createAnalyser();
    a.fftSize = fftSize;
    a.smoothingTimeConstant = 0.72;
    a.minDecibels = -110;
    a.maxDecibels = -6;
    this.busNode(bus).connect(a);
    return a;
  }

  /**
   * A `MediaStream` carrying a bus, for recording alongside the canvas.
   *
   * Tapped off the bus rather than the context destination, so what gets
   * recorded is the game mix itself — unaffected by the OS output device, and
   * still captured when the player has their speakers muted.
   *
   * Created on demand; hold the result for the life of the recording.
   */
  captureStream(bus: AudioBusName = 'master'): MediaStream | null {
    const ctx = this.ctx as AudioContext;
    if (typeof ctx.createMediaStreamDestination !== 'function') return null;
    // A suspended context hands back a track that never delivers a single
    // sample, and MediaRecorder then blocks waiting for it — producing a
    // completely empty file, video included. Refusing here means a recording
    // started before the audio gesture is silent rather than lost.
    if (ctx.state !== 'running') return null;
    const dest = ctx.createMediaStreamDestination();
    this.busNode(bus).connect(dest);
    return dest.stream;
  }

  /**
   * Set a bus level directly. Intended for previews and for a future
   * accessibility mix (e.g. "engine only"); the frame loop does not use it.
   */
  setBusGain(bus: AudioBusName, gain: number): void {
    const g = clamp(gain, 0, 4);
    this.trims[bus] = g;
    const n = this.busNode(bus) as GainNode;
    if (n.gain) n.gain.setTargetAtTime(g, this.ctx.currentTime, 0.01);
  }

  /** Mute every bus except one. Pass null to restore them all. */
  soloBus(bus: AudioBusName | null): void {
    const all: AudioBusName[] = ['engine', 'tyre', 'wind', 'ambience', 'fx'];
    for (const b of all) this.setBusGain(b, bus === null || b === bus ? 1 : 0);
  }

  private busNode(bus: AudioBusName): AudioNode {
    switch (bus) {
      case 'engine':
        return this.engineBus;
      case 'tyre':
        return this.tyreBus;
      case 'wind':
        return this.windBus;
      case 'ambience':
        return this.ambienceBus;
      case 'fx':
        return this.fxBus;
      default:
        return this.master;
    }
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.engine.dispose();
    this.surface.dispose();
    this.birds.dispose();
    for (const s of [this.ambSrc]) {
      try {
        s.stop();
      } catch {
        /* already stopped */
      }
    }
    try {
      this.ambLfo.stop();
    } catch {
      /* already stopped */
    }
    disconnectAll([
      this.ambSrc, this.ambHp, this.ambLp, this.ambGain, this.ambLfo, this.ambLfoDepth,
      this.engineBus, this.tyreBus, this.windBus, this.ambienceBus, this.fxBus,
      this.enginePan, this.tyrePan,
      this.sendEngine, this.sendTyre, this.sendFx, this.convolver, this.reverbReturn,
      this.preMaster, this.masterHp, this.limiter, this.softClip, this.master,
    ]);
    if (this.ownsContext) {
      const ctx = this.ctx as AudioContext;
      if (typeof ctx.close === 'function' && ctx.state !== 'closed') {
        void ctx.close().catch(() => undefined);
      }
    }
  }

  // -------------------------------------------------------------------------
  //  Internals
  // -------------------------------------------------------------------------

  private applyMaster(): void {
    const now = this.ctx.currentTime;
    const g = this.muted ? 0 : this.volume;
    this.master.gain.setTargetAtTime(g, now, 0.02);
  }

  /**
   * Reconstruct normalised engine load from the published vehicle state.
   * Returns -1 (full engine braking) .. +1 (wide open).
   */
  private estimateLoad(dt: number, s: VehicleState): number {
    const dr = JEEP_TUNING.drivetrain;
    const tr = dr.transmission;
    const eng = dr.engine;
    const tire = JEEP_TUNING.tire;
    const aero = JEEP_TUNING.aero;
    const mass = JEEP_TUNING.chassis.mass;

    const rpm = clamp(Number.isFinite(s.engineRpm) ? s.engineRpm : eng.idleRpm, 0, eng.maxRpm);
    const engOmega = rpm * RAD_S_PER_RPM;
    const avail = Math.max(40, this.engine.torqueAt(rpm));

    // --- free-revving term: crank inertia * angular acceleration ---
    const rawD = (engOmega - this.prevEngOmega) / dt;
    this.prevEngOmega = engOmega;
    // Heavier smoothing than the rest: a derivative of a 120 Hz signal
    // sampled at render rate is noisy, and this term is worth ~0.9 of full
    // load, so noise on it would flap the timbre.
    this.smoothDOmega += (rawD - this.smoothDOmega) * clamp(dt * 9, 0, 1);
    const revTerm = clamp((eng.inertia * this.smoothDOmega) / avail, -1.4, 1.4);

    // --- connected term: torque implied by what the car is actually doing ---
    let ratio = 0;
    if (s.gear > 0 && s.gear <= tr.gearRatios.length) {
      ratio = tr.gearRatios[s.gear - 1] * tr.finalDrive;
    } else if (s.gear < 0) {
      ratio = -tr.reverseRatio * tr.finalDrive;
    }
    const v = Number.isFinite(s.forwardSpeed) ? s.forwardSpeed : 0;
    const wheelOmega = v / tire.radius;
    // Prefer the ratio the car is actually running — it picks up low range
    // and any mid-shift state for free.
    if (Math.abs(wheelOmega) > 2 && s.clutch > 0.8) {
      const measured = engOmega / wheelOmega;
      if (Number.isFinite(measured) && Math.abs(measured) > 0.8 && Math.abs(measured) < 90) {
        ratio = measured;
      }
    }

    let connTerm = 0;
    const connected = Math.abs(ratio) > 0.5 && s.clutch > 0.3 && Math.abs(v) > 0.4;
    if (connected) {
      // Specific force along the chassis +Z, in m/s^2. Contains the gradient.
      const a = clamp(s.localAccel.z, -4, 4) * G;
      const drag = 0.5 * aero.airDensity * aero.dragCoefficient * aero.frontalArea * v * Math.abs(v);
      const rollRes =
        0.03 * mass * G * Math.sign(v) * smoothstep(0, 1.5, Math.abs(v));
      const force = mass * a + drag + rollRes;
      const wheelTorque = force * tire.radius;
      const engTorque = wheelTorque / (ratio * tr.efficiency);
      connTerm = clamp(engTorque / avail, -1.6, 1.6);
    }

    // The connected term systematically under-reads: it ignores driveline
    // inertia, diff preload and the tyre's rolling losses beyond the constant
    // term. 1.25 puts a full-throttle pull at ~1.0 on the tuning's own numbers.
    let load = connTerm * 1.25 + revTerm * 0.85;

    // Engine braking is small in absolute torque — maybe 35 N.m against 350
    // available — but it is a *loud*, distinctive sound. Scale the negative
    // side up so a lift-off is unmistakable.
    if (load < 0) load *= 2.6;

    // Below the clutch lock point the drivetrain is barely attached; keep some
    // load so a crawl in low range does not sound like coasting.
    if (rpm < eng.idleRpm * 1.05 && Math.abs(load) < 0.06) load = 0;

    return clamp(load, -1, 1);
  }

  /** A short two-to-four note call, pitch-swept. Not a beep. */
  private chirp(now: number): void {
    const base = 2400 + this.rnd() * 2600;
    const pan = (this.rnd() - 0.5) * 1.7;
    const notes = 2 + Math.floor(this.rnd() * 3);
    for (let i = 0; i < notes; i++) {
      const t = now + i * (0.07 + this.rnd() * 0.09);
      const up = this.rnd() < 0.6;
      const f0 = base * (0.86 + this.rnd() * 0.3);
      this.birds.fire(t, {
        gain: 0.035 + this.rnd() * 0.03,
        freq: up ? f0 : f0 * 1.45,
        toFreq: up ? f0 * 1.5 : f0 * 0.78,
        attack: 0.006,
        decay: 0.05 + this.rnd() * 0.06,
        type: 'sine',
        filterFreq: 9000,
        pan,
      });
    }
  }
}

export default GameAudio;
