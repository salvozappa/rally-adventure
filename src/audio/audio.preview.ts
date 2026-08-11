/**
 * ============================================================================
 *  AUDIO PREVIEW — the test rig for the whole audio subsystem.
 * ============================================================================
 *
 * Two ways to drive the real `GameAudio`:
 *
 *   MANUAL   sliders write a synthetic VehicleState directly. The load slider
 *            is inverted through the same physics the load estimator uses, so
 *            "LOAD 0.8" really does arrive at the engine as 0.8 — and the
 *            readout shows what the estimator reconstructed, which makes the
 *            round trip itself a test.
 *
 *   DRIVE    a small forward vehicle model (torque curve, gear ratios, drag,
 *            clutch, auto box — all read from JEEP_TUNING) is driven by a
 *            scripted throttle/brake/handbrake track: idle, launch, shift up
 *            through the gears, lift off, downshift, handbrake slide, jump,
 *            land. Nothing is faked; the audio sees a state that a real drive
 *            would have produced.
 *
 * Plus an offline verification pass (`runOfflineAnalysis`) which renders the
 * same drive cycle through an OfflineAudioContext and asserts on the samples:
 * peak, RMS, DC, and — via an FFT of steady-RPM renders — that the strongest
 * partial really does sit at the firing frequency, rpm/20 Hz for a six.
 */

import * as THREE from 'three';
import { GameAudio } from './GameAudio';
import { JEEP_TUNING } from '../physics/VehicleTuning';
import type { SurfaceKind, VehicleState, WheelState } from '../types';

const T = JEEP_TUNING;
const ENG = T.drivetrain.engine;
const TRANS = T.drivetrain.transmission;
const CL = T.drivetrain.clutch;
const RAD_S_PER_RPM = (2 * Math.PI) / 60;
const G = 9.81;

const SURFACES: SurfaceKind[] = ['dirt', 'grass', 'rock', 'gravel', 'sand', 'mud', 'snow'];

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}
function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}
function smoothstep(e0: number, e1: number, x: number): number {
  const t = clamp((x - e0) / (e1 - e0 || 1e-9), 0, 1);
  return t * t * (3 - 2 * t);
}

export function torqueAt(rpm: number): number {
  const c = ENG.torqueCurve;
  if (rpm <= c[0][0]) return c[0][1];
  for (let i = 1; i < c.length; i++) {
    if (rpm <= c[i][0]) {
      const [r0, t0] = c[i - 1];
      const [r1, t1] = c[i];
      return lerp(t0, t1, (rpm - r0) / Math.max(1e-6, r1 - r0));
    }
  }
  return c[c.length - 1][1];
}

// ---------------------------------------------------------------------------
//  A blank VehicleState we can mutate in place
// ---------------------------------------------------------------------------

export function makeVehicleState(): VehicleState {
  const wheels: WheelState[] = [];
  for (let i = 0; i < 4; i++) {
    wheels.push({
      position: new THREE.Vector3(),
      quaternion: new THREE.Quaternion(),
      compression: 0.48,
      grounded: true,
      slipRatio: 0,
      slipAngle: 0,
      load: (T.chassis.mass * G) / 4,
      surface: 'dirt',
      spin: 0,
      steerAngle: 0,
    });
  }
  return {
    position: new THREE.Vector3(),
    quaternion: new THREE.Quaternion(),
    velocity: new THREE.Vector3(),
    angularVelocity: new THREE.Vector3(),
    forwardSpeed: 0,
    speed: 0,
    engineRpm: ENG.idleRpm,
    gear: 1,
    clutch: 1,
    wheels,
    airborne: false,
    localAccel: new THREE.Vector3(),
  };
}

// ---------------------------------------------------------------------------
//  Forward vehicle model — just enough to produce an honest VehicleState
// ---------------------------------------------------------------------------

export interface DriverInput {
  throttle: number;
  brake: number;
  handbrake: number;
  steer: number;
  /** Terrain roughness 0..1: drives suspension movement and rattle. */
  rough: number;
  surface: SurfaceKind;
  /** Force the car into the air for jumps. */
  airborne: boolean;
}

export class MiniSim {
  readonly state = makeVehicleState();
  v = 0;
  rpm = ENG.idleRpm;
  gear = 1;
  private shiftTimer = 0;
  private cooldown = 0;
  private clutch = 1;
  private accel = 0;
  private bumpPhase = 0;
  private rnd = 0.5;
  /** Set by the caller when a jump ends, so the harness can fire the landing. */
  lastVerticalSpeed = 0;
  airTime = 0;

  reset(): void {
    this.v = 0;
    this.rpm = ENG.idleRpm;
    this.gear = 1;
    this.shiftTimer = 0;
    this.cooldown = 0;
    this.clutch = 1;
    this.accel = 0;
    this.airTime = 0;
    this.state.velocity.set(0, 0, 0);
  }

  step(dt: number, u: DriverInput): VehicleState {
    const s = this.state;
    const r = T.tire.radius;
    const m = T.chassis.mass;
    const throttle = clamp(u.throttle, 0, 1);

    const ratio = this.gear > 0 ? TRANS.gearRatios[this.gear - 1] * TRANS.finalDrive : 0;
    const wheelOmega = this.v / r;
    const targetRpm = (wheelOmega * ratio * 60) / (2 * Math.PI);

    // ---- engine torque ----
    const omega = this.rpm * RAD_S_PER_RPM;
    const frictionScale = 1 - throttle * (1 - ENG.frictionThrottleRelief);
    const friction = (ENG.frictionConstant + ENG.frictionViscous * omega) * frictionScale;
    // Rev limiter: fuel cut past the redline, exactly as the drivetrain does it.
    const limited = this.rpm > ENG.redlineRpm ? 0 : 1;
    let tq = torqueAt(this.rpm) * throttle * limited - friction;

    this.shiftTimer = Math.max(0, this.shiftTimer - dt);
    this.cooldown = Math.max(0, this.cooldown - dt);
    const shifting = this.shiftTimer > 0;

    let transmitted = 0;
    if (u.airborne) {
      // Wheels off the ground: the engine only has its own inertia to fight.
      this.clutch = shifting ? 0.1 : 1;
      this.rpm += ((tq * (shifting ? 1 : 0.35)) / ENG.inertia / RAD_S_PER_RPM) * dt;
      transmitted = 0;
    } else if (shifting || this.gear === 0) {
      this.clutch = 0.1;
      this.rpm += (tq / ENG.inertia / RAD_S_PER_RPM) * dt;
      transmitted = 0;
    } else {
      const launchTarget = lerp(ENG.idleRpm, CL.launchRpm, Math.pow(throttle, 0.6));
      if (targetRpm < launchTarget - 40) {
        // Clutch slipping — a standing start. The engine flares toward the
        // launch RPM and passes on a fraction of its torque.
        this.clutch = clamp(targetRpm / Math.max(launchTarget, 1), 0.05, 1);
        this.rpm += (launchTarget - this.rpm) * clamp(dt * 5, 0, 1);
        transmitted = tq * clamp(0.25 + 0.75 * this.clutch, 0, 1);
      } else {
        this.clutch = 1;
        this.rpm += (Math.max(ENG.idleRpm, targetRpm) - this.rpm) * clamp(dt * 26, 0, 1);
        transmitted = tq;
      }
    }
    this.rpm = clamp(this.rpm, ENG.stallRpm, ENG.maxRpm);

    // ---- auto gearbox ----
    if (!shifting && this.cooldown <= 0 && !u.airborne) {
      const up = lerp(TRANS.upshiftRpmLow, TRANS.upshiftRpmHigh, throttle);
      const down = lerp(TRANS.downshiftRpmLow, TRANS.downshiftRpmHigh, throttle);
      if (this.rpm > up && this.gear < TRANS.gearRatios.length) {
        this.gear++;
        this.shiftTimer = TRANS.shiftTime;
        this.cooldown = TRANS.shiftCooldown;
      } else if (this.rpm < down && this.gear > 1) {
        this.gear--;
        this.shiftTimer = TRANS.shiftTime;
        this.cooldown = TRANS.shiftCooldown;
      }
    }

    // ---- longitudinal forces ----
    const drag =
      0.5 * T.aero.airDensity * T.aero.dragCoefficient * T.aero.frontalArea * this.v * Math.abs(this.v);
    const rollRes = 0.03 * m * G * Math.sign(this.v) * smoothstep(0, 1.5, Math.abs(this.v));
    const brakeForce =
      (clamp(u.brake, 0, 1) * T.brakes.maxTorque + clamp(u.handbrake, 0, 1) * T.brakes.handbrakeTorque) /
      r;
    const drive = u.airborne ? 0 : (transmitted * ratio * TRANS.efficiency) / r;
    const F = drive - drag - rollRes - brakeForce * Math.sign(this.v) * smoothstep(0, 0.4, Math.abs(this.v));
    const a = u.airborne ? -drag / m : F / m;
    this.v = Math.max(0, this.v + a * dt);
    this.accel += (a - this.accel) * clamp(dt * 22, 0, 1);

    // ---- slip ----
    const surf = T.surfaces[u.surface];
    const gripForce = surf.friction * m * G;
    // Longitudinal slip when the demanded tractive force exceeds available grip.
    const demand = Math.abs(drive) + brakeForce * 0.6;
    const excess = clamp((demand - gripForce * 0.62) / Math.max(gripForce, 1), 0, 2.5);
    const spinSlip = excess * (1.6 - 0.9 * smoothstep(0, 22, this.v));
    const hb = clamp(u.handbrake, 0, 1) * smoothstep(1.5, 8, this.v);
    // Handbrake locks the rear: negative slip ratio plus a big slip angle.
    const lockSlip = hb * 0.9;
    const slipAngle = clamp(Math.abs(u.steer) * hb * 0.75 + Math.abs(u.steer) * 0.06 * smoothstep(6, 26, this.v), 0, 0.9);

    // ---- suspension / roughness ----
    this.bumpPhase += dt * (2.5 + this.v * 0.9);
    this.rnd = (this.rnd * 9301 + 0.49297) % 1;

    // ---- publish ----
    s.forwardSpeed = this.v;
    s.speed = this.v;
    s.velocity.set(0, u.airborne ? this.state.velocity.y - G * dt : 0, this.v);
    if (!u.airborne) s.velocity.y = 0;
    this.lastVerticalSpeed = s.velocity.y;
    s.engineRpm = this.rpm;
    s.gear = this.gear;
    s.clutch = this.clutch;
    s.airborne = u.airborne;
    s.localAccel.set(0, 1, u.airborne ? 0 : a / G);
    if (u.airborne) this.airTime += dt;
    else this.airTime = 0;

    for (let i = 0; i < 4; i++) {
      const w = s.wheels[i];
      const rear = i >= 2;
      w.grounded = !u.airborne;
      w.surface = u.surface;
      w.load = u.airborne ? 0 : ((m * G) / 4) * (1 - this.accel * 0.35 * (rear ? -1 : 1));
      w.slipRatio = u.airborne ? 0 : (rear ? spinSlip - lockSlip : spinSlip * 0.55 - lockSlip * 0.15);
      w.slipAngle = u.airborne ? 0 : slipAngle * (rear ? 1 : 0.6);
      w.spin = (this.v / T.tire.radius) * (1 + w.slipRatio);
      const bump =
        u.rough *
        (0.5 * Math.sin(this.bumpPhase * (1.7 + i * 0.31)) + 0.5 * Math.sin(this.bumpPhase * (4.3 + i * 0.17)) + (this.rnd - 0.5) * 0.6);
      w.compression = u.airborne ? 0.05 : clamp(0.42 + this.accel * 0.06 * (rear ? 1 : -1) + bump * 0.42, 0, 1);
    }
    return s;
  }
}

// ---------------------------------------------------------------------------
//  Drive cycle script
// ---------------------------------------------------------------------------

export interface CycleEvent {
  t: number;
  label: string;
  input: Partial<DriverInput>;
}

/**
 * idle -> launch -> up through the gears -> lift -> downshift ->
 * handbrake slide -> jump -> land -> settle. 26 seconds.
 */
export const DRIVE_CYCLE: CycleEvent[] = [
  { t: 0.0, label: 'idle', input: { throttle: 0, brake: 0, handbrake: 0, steer: 0, rough: 0.05, surface: 'dirt', airborne: false } },
  { t: 2.5, label: 'blip in neutral', input: { throttle: 0.85 } },
  { t: 3.1, label: 'idle', input: { throttle: 0 } },
  { t: 4.5, label: 'LAUNCH — full throttle', input: { throttle: 1, rough: 0.12 } },
  { t: 12.5, label: 'hold, revs against the limiter', input: { throttle: 1, rough: 0.18 } },
  { t: 14.0, label: 'LIFT OFF — overrun, pops', input: { throttle: 0, rough: 0.2 } },
  { t: 16.5, label: 'part throttle cruise', input: { throttle: 0.35, rough: 0.25, surface: 'gravel' } },
  { t: 18.0, label: 'HANDBRAKE SLIDE', input: { throttle: 0.55, handbrake: 1, steer: 1, rough: 0.3 } },
  { t: 20.0, label: 'straighten, back on power', input: { throttle: 0.9, handbrake: 0, steer: 0.1, rough: 0.35, surface: 'rock' } },
  { t: 22.0, label: 'JUMP — airborne', input: { throttle: 0.7, rough: 0, airborne: true } },
  { t: 23.1, label: 'LANDING', input: { airborne: false, rough: 0.55, throttle: 0.5 } },
  { t: 24.0, label: 'settle', input: { throttle: 0.2, rough: 0.2 } },
  { t: 25.0, label: 'stop', input: { throttle: 0, brake: 0.6, rough: 0.08 } },
];

export const CYCLE_DURATION = 26.5;

export function cycleInputAt(t: number, out: DriverInput): DriverInput {
  for (const e of DRIVE_CYCLE) {
    if (t >= e.t) Object.assign(out, e.input);
  }
  return out;
}

export function cycleLabelAt(t: number): string {
  let label = '';
  for (const e of DRIVE_CYCLE) if (t >= e.t) label = e.label;
  return label;
}

export function defaultInput(): DriverInput {
  return { throttle: 0, brake: 0, handbrake: 0, steer: 0, rough: 0.05, surface: 'dirt', airborne: false };
}

// ---------------------------------------------------------------------------
//  FFT (radix-2, in place) for the offline verification
// ---------------------------------------------------------------------------

export function fftMagnitude(input: Float32Array): Float32Array {
  const n = input.length;
  const re = new Float32Array(n);
  const im = new Float32Array(n);
  // Hann window: without it the leakage from a 38 Hz fundamental smears
  // across the low bins and you cannot tell a peak from a skirt.
  for (let i = 0; i < n; i++) {
    re[i] = input[i] * (0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1)));
  }
  // bit reversal
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const tr = re[i]; re[i] = re[j]; re[j] = tr;
      const ti = im[i]; im[i] = im[j]; im[j] = ti;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wr = Math.cos(ang);
    const wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1;
      let ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const ur = re[i + k];
        const ui = im[i + k];
        const vr = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci;
        const vi = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
        re[i + k] = ur + vr;
        im[i + k] = ui + vi;
        re[i + k + len / 2] = ur - vr;
        im[i + k + len / 2] = ui - vi;
        const ncr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = ncr;
      }
    }
  }
  const mag = new Float32Array(n / 2);
  for (let i = 0; i < n / 2; i++) mag[i] = Math.hypot(re[i], im[i]) / (n / 4);
  return mag;
}

export interface BufferStats {
  peak: number;
  rms: number;
  dc: number;
  /** Fraction of samples at or beyond +/-0.999 — true clipping. */
  clipped: number;
}

export function analyseBuffer(buf: AudioBuffer, skipSeconds = 0): BufferStats {
  let peak = 0;
  let sum = 0;
  let dc = 0;
  let clipped = 0;
  let n = 0;
  const skip = Math.min(buf.length - 1, Math.floor(skipSeconds * buf.sampleRate));
  for (let ch = 0; ch < buf.numberOfChannels; ch++) {
    const d = buf.getChannelData(ch);
    for (let i = skip; i < d.length; i++) {
      const a = Math.abs(d[i]);
      if (a > peak) peak = a;
      if (a >= 0.999) clipped++;
      sum += d[i] * d[i];
      dc += d[i];
      n++;
    }
  }
  return { peak, rms: Math.sqrt(sum / Math.max(n, 1)), dc: dc / Math.max(n, 1), clipped: clipped / Math.max(n, 1) };
}

/** Quantise a time to a 128-sample render quantum boundary. */
function quantum(t: number, sampleRate: number): number {
  return (Math.round((t * sampleRate) / 128) * 128) / sampleRate;
}

/**
 * Render a scripted state sequence offline. `drive` is called at each 1/60 s
 * boundary and must return the VehicleState for that instant.
 */
async function renderOffline(
  duration: number,
  sampleRate: number,
  drive: (t: number, dt: number, audio: GameAudio) => VehicleState,
  onCreate?: (audio: GameAudio) => void,
): Promise<AudioBuffer> {
  const ctx = new OfflineAudioContext(2, Math.ceil(duration * sampleRate), sampleRate);
  const audio = new GameAudio({ context: ctx, masterVolume: 0.85 });
  await audio.whenReady();
  onCreate?.(audio);

  const dt = 1 / 60;
  const steps = Math.floor(duration / dt);
  for (let i = 1; i < steps; i++) {
    const t = quantum(i * dt, sampleRate);
    if (t >= duration) break;
    void ctx.suspend(t).then(() => {
      const s = drive(t, dt, audio);
      audio.update(dt, s);
      void ctx.resume();
    });
  }
  // Prime the very first frame before rendering starts.
  audio.update(dt, drive(0, dt, audio));
  const buf = await ctx.startRendering();
  audio.dispose();
  return buf;
}

export interface SteadyResult {
  rpm: number;
  expectedFiring: number;
  measuredPeak: number;
  errorPct: number;
  harmonics: number[];
  /** dB of the firing peak relative to the loudest bin in 20 Hz..16 kHz. */
  peakDb: number;
  noiseFloorDb: number;
  stats: BufferStats;
}

/**
 * Hold a steady RPM and load, render, and check the strongest low-frequency
 * partial lands on rpm/20 Hz — the firing rate of a four-stroke six.
 */
export async function measureSteadyRpm(
  rpm: number,
  load: number,
  sampleRate = 48000,
  solo: import('./GameAudio').AudioBusName | null = null,
): Promise<SteadyResult> {
  const dur = 2.2;
  const state = makeVehicleState();
  state.engineRpm = rpm;
  state.gear = 3;
  state.clutch = 1;
  state.forwardSpeed = 0;
  state.speed = 0;
  for (const w of state.wheels) {
    w.grounded = true;
    w.slipRatio = 0;
    w.slipAngle = 0;
    w.compression = 0.48;
  }
  // Feed load in directly through localAccel; with speed 0 the connected term
  // is disabled, so we invert the free-rev term instead: load = I*dw/dt / Tmax
  // is not usable at a steady RPM, so hold the state and set localAccel to a
  // value the estimator will read through the connected path.
  state.forwardSpeed = 18;
  state.speed = 18;
  const avail = torqueAt(rpm);
  const ratio = (rpm * RAD_S_PER_RPM) / (18 / T.tire.radius);
  const engTorque = ((load / 1.25) * avail);
  const wheelTorque = engTorque * ratio * TRANS.efficiency;
  const force = wheelTorque / T.tire.radius;
  const drag = 0.5 * T.aero.airDensity * T.aero.dragCoefficient * T.aero.frontalArea * 18 * 18;
  const rollRes = 0.03 * T.chassis.mass * G;
  state.localAccel.set(0, 1, (force - drag - rollRes) / T.chassis.mass / G);

  const buf = await renderOffline(dur, sampleRate, () => state, (audio) => {
    if (solo) audio.soloBus(solo);
  });

  // Analyse the last 32768 samples of the left channel, well past any
  // start-up transient and after every smoothed parameter has settled.
  const N = 32768;
  const d = buf.getChannelData(0);
  const slice = new Float32Array(N);
  slice.set(d.subarray(Math.max(0, d.length - N)));
  const mag = fftMagnitude(slice);
  const binHz = sampleRate / N;

  const expected = rpm / 20;
  // Search a generous window around the expected fundamental.
  const lo = Math.max(1, Math.floor((expected * 0.55) / binHz));
  const hi = Math.min(mag.length - 1, Math.ceil((expected * 1.9) / binHz));
  let bestBin = lo;
  for (let i = lo; i <= hi; i++) if (mag[i] > mag[bestBin]) bestBin = i;
  // Parabolic interpolation for sub-bin accuracy.
  const y0 = mag[bestBin - 1] ?? 0;
  const y1 = mag[bestBin];
  const y2 = mag[bestBin + 1] ?? 0;
  const denom = y0 - 2 * y1 + y2;
  const delta = Math.abs(denom) > 1e-12 ? (0.5 * (y0 - y2)) / denom : 0;
  const measured = (bestBin + clamp(delta, -1, 1)) * binHz;

  const harmonics: number[] = [];
  for (let h = 1; h <= 8; h++) {
    const b = Math.round((expected * h) / binHz);
    if (b < mag.length) harmonics.push(20 * Math.log10(Math.max(mag[b], 1e-9)));
  }

  // Noise floor: median magnitude between 6 kHz and 15 kHz.
  const nlo = Math.floor(6000 / binHz);
  const nhi = Math.min(mag.length - 1, Math.floor(15000 / binHz));
  const tail: number[] = [];
  for (let i = nlo; i <= nhi; i += 3) tail.push(mag[i]);
  tail.sort((a, b) => a - b);
  const floor = tail[Math.floor(tail.length / 2)] ?? 1e-9;

  return {
    rpm,
    expectedFiring: expected,
    measuredPeak: measured,
    errorPct: (100 * (measured - expected)) / expected,
    harmonics,
    peakDb: 20 * Math.log10(Math.max(mag[bestBin], 1e-9)),
    noiseFloorDb: 20 * Math.log10(Math.max(floor, 1e-9)),
    // Skip the first 0.6 s: every filter and the convolver are charging from
    // zero at t=0 and that transient would swamp a steady-state measurement.
    stats: analyseBuffer(buf, 0.6),
  };
}

/** Render the whole scripted drive cycle offline and report on the samples. */
export async function measureDriveCycle(sampleRate = 48000): Promise<{
  stats: BufferStats;
  windowRms: Array<{ t: number; rms: number; label: string }>;
}> {
  const sim = new MiniSim();
  const input = defaultInput();
  let landed = false;
  const buf = await renderOffline(CYCLE_DURATION, sampleRate, (t, dt, audio) => {
    cycleInputAt(t, input);
    const s = sim.step(dt, input);
    if (input.airborne) landed = false;
    else if (!landed && sim.airTime === 0 && t > 23 && t < 23.3) {
      landed = true;
      audio.playLanding(11.5);
    }
    return s;
  });

  const stats = analyseBuffer(buf);
  const d = buf.getChannelData(0);
  const windowRms: Array<{ t: number; rms: number; label: string }> = [];
  const win = Math.floor(sampleRate * 0.5);
  for (let start = 0; start + win < d.length; start += win * 2) {
    let sum = 0;
    for (let i = start; i < start + win; i++) sum += d[i] * d[i];
    const t = start / sampleRate;
    windowRms.push({ t, rms: Math.sqrt(sum / win), label: cycleLabelAt(t) });
  }
  return { stats, windowRms };
}

// ---------------------------------------------------------------------------
//  Page wiring
// ---------------------------------------------------------------------------

interface Controls {
  mode: HTMLSelectElement;
  rpm: HTMLInputElement;
  load: HTMLInputElement;
  gear: HTMLInputElement;
  speed: HTMLInputElement;
  slipRatio: HTMLInputElement;
  slipAngle: HTMLInputElement;
  compression: HTMLInputElement;
  surface: HTMLSelectElement;
  airborne: HTMLInputElement;
  volume: HTMLInputElement;
}

function el<T extends HTMLElement>(id: string): T {
  const n = document.getElementById(id);
  if (!n) throw new Error('missing element #' + id);
  return n as T;
}

function boot(): void {
  const audio = new GameAudio({ masterVolume: 0.85 });
  const analyser = audio.createAnalyser('master', 16384);
  const scopeAnalyser = audio.createAnalyser('master', 2048);
  scopeAnalyser.smoothingTimeConstant = 0;
  const engineAnalyser = audio.createAnalyser('engine', 16384);

  const c: Controls = {
    mode: el<HTMLSelectElement>('mode'),
    rpm: el<HTMLInputElement>('rpm'),
    load: el<HTMLInputElement>('load'),
    gear: el<HTMLInputElement>('gear'),
    speed: el<HTMLInputElement>('speed'),
    slipRatio: el<HTMLInputElement>('slipRatio'),
    slipAngle: el<HTMLInputElement>('slipAngle'),
    compression: el<HTMLInputElement>('compression'),
    surface: el<HTMLSelectElement>('surface'),
    airborne: el<HTMLInputElement>('airborne'),
    volume: el<HTMLInputElement>('volume'),
  };
  for (const s of SURFACES) {
    const o = document.createElement('option');
    o.value = s;
    o.textContent = s;
    c.surface.appendChild(o);
  }
  c.surface.value = 'dirt';

  const specCanvas = el<HTMLCanvasElement>('spectrum');
  const waveCanvas = el<HTMLCanvasElement>('wave');
  const readout = el<HTMLPreElement>('readout');
  const report = el<HTMLPreElement>('report');
  const statusEl = el<HTMLSpanElement>('status');

  const state = makeVehicleState();
  const sim = new MiniSim();
  const input = defaultInput();
  let cycleT = -1;
  let cycleLanded = false;

  // ---- unlock ----
  const unlock = async (): Promise<void> => {
    await audio.resume();
    statusEl.textContent = `${audio.context.state} · ${audio.context.sampleRate} Hz · ${audio.usingWorklet ? 'worklet' : 'fallback'}`;
  };
  window.addEventListener('pointerdown', () => void unlock());
  window.addEventListener('keydown', () => void unlock());
  el<HTMLButtonElement>('start').addEventListener('click', () => void unlock());

  // ---- buttons ----
  el<HTMLButtonElement>('landing').addEventListener('click', () => audio.playLanding(11));
  el<HTMLButtonElement>('landingSoft').addEventListener('click', () => audio.playLanding(3.5));
  el<HTMLButtonElement>('collision').addEventListener('click', () => audio.playCollision(9));
  el<HTMLButtonElement>('cycle').addEventListener('click', () => {
    sim.reset();
    Object.assign(input, defaultInput());
    cycleT = 0;
    cycleLanded = false;
    c.mode.value = 'cycle';
  });
  el<HTMLButtonElement>('mute').addEventListener('click', (e) => {
    const b = e.currentTarget as HTMLButtonElement;
    const m = !audio.isMuted();
    audio.setMuted(m);
    b.textContent = m ? 'UNMUTE' : 'MUTE';
  });
  c.volume.addEventListener('input', () => audio.setVolume(Number(c.volume.value)));

  const offlineBtn = el<HTMLButtonElement>('offline');
  offlineBtn.addEventListener('click', async () => {
    offlineBtn.disabled = true;
    report.textContent = 'rendering offline…';
    try {
      report.textContent = await runOfflineAnalysis();
    } catch (err) {
      report.textContent = 'FAILED: ' + String(err);
    }
    offlineBtn.disabled = false;
  });

  // ---- manual state assembly ----
  function manualState(dt: number): VehicleState {
    const rpm = Number(c.rpm.value);
    const load = Number(c.load.value);
    const gear = Number(c.gear.value);
    const speed = Number(c.speed.value);
    const surface = c.surface.value as SurfaceKind;

    state.engineRpm = rpm;
    state.gear = gear;
    state.clutch = 1;
    state.forwardSpeed = speed;
    state.speed = speed;
    state.airborne = c.airborne.checked;

    // Invert the load estimator so the slider means what it says. With a
    // measured ratio available (speed > 0.8 m/s) the connected term carries
    // all of it, so we solve that term for the acceleration it implies.
    const avail = torqueAt(rpm);
    let az = 0;
    if (speed > 0.9) {
      const ratio = (rpm * RAD_S_PER_RPM) / (speed / T.tire.radius);
      const target = load < 0 ? load / 2.6 : load;
      const engTorque = (target / 1.25) * avail;
      const wheelTorque = engTorque * ratio * TRANS.efficiency;
      const force = wheelTorque / T.tire.radius;
      const drag =
        0.5 * T.aero.airDensity * T.aero.dragCoefficient * T.aero.frontalArea * speed * speed;
      const rollRes = 0.03 * T.chassis.mass * G * smoothstep(0, 1.5, speed);
      az = (force - drag - rollRes) / T.chassis.mass / G;
    }
    state.localAccel.set(0, 1, az);

    const sr = Number(c.slipRatio.value);
    const sa = Number(c.slipAngle.value);
    const comp = Number(c.compression.value);
    for (let i = 0; i < state.wheels.length; i++) {
      const w = state.wheels[i];
      w.grounded = !state.airborne;
      w.surface = surface;
      w.slipRatio = sr;
      w.slipAngle = sa;
      w.load = state.airborne ? 0 : (T.chassis.mass * G) / 4;
      w.compression = state.airborne ? 0.05 : comp;
      w.spin = (speed / T.tire.radius) * (1 + sr);
    }
    void dt;
    return state;
  }

  // ---- frame loop ----
  let last = performance.now();
  let peakHold: Float32Array | null = null;
  let holdDecay = 0;

  function frame(now: number): void {
    requestAnimationFrame(frame);
    const dt = Math.min(0.1, Math.max(1 / 240, (now - last) / 1000));
    last = now;

    let s: VehicleState;
    let label = '';
    if (c.mode.value === 'cycle' && cycleT >= 0) {
      cycleT += dt;
      cycleInputAt(cycleT, input);
      label = cycleLabelAt(cycleT);
      s = sim.step(dt, input);
      if (input.airborne) cycleLanded = false;
      else if (!cycleLanded && cycleT > 23.05 && cycleT < 23.4) {
        cycleLanded = true;
        audio.playLanding(11.5);
      }
      // Mirror the sim onto the sliders so you can see what is happening.
      c.rpm.value = String(Math.round(s.engineRpm));
      c.gear.value = String(s.gear);
      c.speed.value = s.forwardSpeed.toFixed(1);
      c.surface.value = input.surface;
      c.airborne.checked = s.airborne;
      if (cycleT > CYCLE_DURATION) {
        cycleT = -1;
        c.mode.value = 'manual';
      }
    } else {
      s = manualState(dt);
    }

    audio.update(dt, s);
    drawSpectrum(specCanvas, analyser, engineAnalyser, s.engineRpm, audio.context.sampleRate);
    drawWave(waveCanvas, scopeAnalyser);

    if (!peakHold || peakHold.length !== analyser.frequencyBinCount) {
      peakHold = new Float32Array(analyser.frequencyBinCount);
      peakHold.fill(-140);
    }
    holdDecay += dt;

    const firing = s.engineRpm / 20;
    readout.textContent =
      `mode      ${c.mode.value}${label ? '  [' + label + ']' : ''}\n` +
      `rpm       ${s.engineRpm.toFixed(0).padStart(6)}   firing ${firing.toFixed(1)} Hz\n` +
      `gear      ${String(s.gear).padStart(6)}   clutch ${s.clutch.toFixed(2)}\n` +
      `speed     ${s.forwardSpeed.toFixed(1).padStart(6)} m/s (${(s.forwardSpeed * 3.6).toFixed(0)} km/h)\n` +
      `accel z   ${s.localAccel.z.toFixed(3).padStart(6)} g\n` +
      `LOAD est  ${audio.lastLoad.toFixed(3).padStart(6)}   (slider ${Number(c.load.value).toFixed(2)})\n` +
      `slip      ratio ${s.wheels[0].slipRatio.toFixed(2)}  angle ${s.wheels[0].slipAngle.toFixed(2)} rad\n` +
      `surface   ${s.wheels[0].surface}   airborne ${s.airborne}\n` +
      `comp      ${s.wheels[0].compression.toFixed(2)}\n` +
      `engine    ${audio.usingWorklet ? 'AudioWorklet' : 'oscillator fallback'}`;
  }
  requestAnimationFrame(frame);

  statusEl.textContent = 'click anywhere to start audio';
  void audio.whenReady().then(() => {
    statusEl.textContent = `${audio.context.state} · ${audio.usingWorklet ? 'worklet' : 'fallback'} · click to start`;
  });

  // Expose for scripted checks from the console / automation.
  (window as unknown as Record<string, unknown>).audioPreview = {
    audio,
    runOfflineAnalysis,
    measureSteadyRpm,
    measureDriveCycle,
    setSlider: (id: string, v: number | string) => {
      const n = document.getElementById(id) as HTMLInputElement | null;
      if (n) {
        n.value = String(v);
        n.dispatchEvent(new Event('input'));
      }
    },
  };
}

// ---------------------------------------------------------------------------
//  Drawing
// ---------------------------------------------------------------------------

const F_MIN = 20;
const F_MAX = 18000;

function fToX(f: number, w: number): number {
  return (Math.log(clamp(f, F_MIN, F_MAX) / F_MIN) / Math.log(F_MAX / F_MIN)) * w;
}

const specData = new WeakMap<AnalyserNode, Float32Array<ArrayBuffer>>();
function freqData(a: AnalyserNode): Float32Array<ArrayBuffer> {
  let d = specData.get(a);
  if (!d || d.length !== a.frequencyBinCount) {
    d = new Float32Array(new ArrayBuffer(a.frequencyBinCount * 4));
    specData.set(a, d);
  }
  a.getFloatFrequencyData(d);
  return d;
}

function drawSpectrum(
  canvas: HTMLCanvasElement,
  master: AnalyserNode,
  engine: AnalyserNode,
  rpm: number,
  sampleRate: number,
): void {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
    canvas.width = w * dpr;
    canvas.height = h * dpr;
  }
  const g = canvas.getContext('2d');
  if (!g) return;
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  g.fillStyle = '#07080c';
  g.fillRect(0, 0, w, h);

  const dbMin = -110;
  const dbMax = -6;
  const yOf = (db: number): number => h - ((clamp(db, dbMin, dbMax) - dbMin) / (dbMax - dbMin)) * h;

  // grid
  g.strokeStyle = 'rgba(255,255,255,0.07)';
  g.fillStyle = 'rgba(200,200,210,0.45)';
  g.font = '10px ui-monospace, monospace';
  g.lineWidth = 1;
  for (const f of [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]) {
    const x = fToX(f, w);
    g.beginPath();
    g.moveTo(x, 0);
    g.lineTo(x, h);
    g.stroke();
    g.fillText(f >= 1000 ? f / 1000 + 'k' : String(f), x + 3, h - 4);
  }
  for (let db = -100; db <= -10; db += 20) {
    const y = yOf(db);
    g.beginPath();
    g.moveTo(0, y);
    g.lineTo(w, y);
    g.stroke();
    g.fillText(db + ' dB', 4, y - 3);
  }

  // firing frequency markers — the whole point of the display
  const firing = rpm / 20;
  for (let k = 1; k <= 12; k++) {
    const f = firing * k;
    if (f > F_MAX) break;
    const x = fToX(f, w);
    g.strokeStyle = k === 1 ? 'rgba(255,140,60,0.85)' : 'rgba(255,140,60,0.22)';
    g.beginPath();
    g.moveTo(x, 0);
    g.lineTo(x, h);
    g.stroke();
  }
  g.fillStyle = 'rgba(255,160,80,0.95)';
  g.fillText(`firing ${firing.toFixed(1)} Hz`, fToX(firing, w) + 4, 12);

  const drawTrace = (a: AnalyserNode, stroke: string, fill: string | null): void => {
    const d = freqData(a);
    const binHz = sampleRate / (a.fftSize || 2048);
    g.beginPath();
    let started = false;
    for (let i = 1; i < d.length; i++) {
      const f = i * binHz;
      if (f < F_MIN) continue;
      if (f > F_MAX) break;
      const x = fToX(f, w);
      const y = yOf(d[i]);
      if (!started) {
        g.moveTo(x, y);
        started = true;
      } else g.lineTo(x, y);
    }
    g.strokeStyle = stroke;
    g.lineWidth = 1.25;
    g.stroke();
    if (fill) {
      g.lineTo(w, h);
      g.lineTo(0, h);
      g.closePath();
      g.fillStyle = fill;
      g.fill();
    }
  };

  drawTrace(master, '#6fe3ff', 'rgba(70,180,220,0.13)');
  drawTrace(engine, 'rgba(255,190,90,0.8)', null);

  g.fillStyle = 'rgba(200,200,210,0.6)';
  g.fillText('master', w - 96, 12);
  g.fillStyle = 'rgba(255,190,90,0.9)';
  g.fillText('engine bus', w - 96, 24);
}

const waveData = new WeakMap<AnalyserNode, Float32Array<ArrayBuffer>>();
function drawWave(canvas: HTMLCanvasElement, a: AnalyserNode): void {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
    canvas.width = w * dpr;
    canvas.height = h * dpr;
  }
  const g = canvas.getContext('2d');
  if (!g) return;
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.fillStyle = '#07080c';
  g.fillRect(0, 0, w, h);

  let d = waveData.get(a);
  if (!d || d.length !== a.fftSize) {
    d = new Float32Array(new ArrayBuffer(a.fftSize * 4));
    waveData.set(a, d);
  }
  a.getFloatTimeDomainData(d);

  g.strokeStyle = 'rgba(255,255,255,0.1)';
  g.beginPath();
  g.moveTo(0, h / 2);
  g.lineTo(w, h / 2);
  g.stroke();
  // full-scale guides
  g.strokeStyle = 'rgba(255,80,80,0.3)';
  for (const v of [-1, 1]) {
    const y = h / 2 - v * (h / 2) * 0.96;
    g.beginPath();
    g.moveTo(0, y);
    g.lineTo(w, y);
    g.stroke();
  }

  let peak = 0;
  g.beginPath();
  for (let i = 0; i < d.length; i++) {
    const x = (i / (d.length - 1)) * w;
    const y = h / 2 - d[i] * (h / 2) * 0.96;
    peak = Math.max(peak, Math.abs(d[i]));
    if (i === 0) g.moveTo(x, y);
    else g.lineTo(x, y);
  }
  g.strokeStyle = peak > 0.99 ? '#ff5b5b' : '#8dffb0';
  g.lineWidth = 1.1;
  g.stroke();

  g.fillStyle = 'rgba(200,200,210,0.7)';
  g.font = '10px ui-monospace, monospace';
  g.fillText(`peak ${peak.toFixed(3)}`, 6, 12);
}

// ---------------------------------------------------------------------------
//  Offline verification report
// ---------------------------------------------------------------------------

export async function runOfflineAnalysis(): Promise<string> {
  const lines: string[] = [];
  const fails: string[] = [];
  const ok = (cond: boolean, msg: string): string => {
    if (!cond) fails.push(msg);
    return cond ? 'PASS' : 'FAIL';
  };

  lines.push('=== STEADY RPM — firing frequency must be rpm/20 Hz (6-cyl 4-stroke) ===');
  lines.push('  rpm    expect    measured    err%     peak dB   floor dB   H1..H6 dB');
  for (const [rpm, load] of [
    [760, 0.0],
    [1500, 0.5],
    [2500, 0.85],
    [4000, 0.9],
    [6000, 0.6],
  ] as Array<[number, number]>) {
    const r = await measureSteadyRpm(rpm, load);
    const h = r.harmonics.slice(0, 6).map((v) => v.toFixed(0).padStart(5)).join('');
    lines.push(
      `${String(rpm).padStart(6)}${r.expectedFiring.toFixed(1).padStart(9)}` +
        `${r.measuredPeak.toFixed(2).padStart(12)}${r.errorPct.toFixed(1).padStart(8)}` +
        `${r.peakDb.toFixed(1).padStart(11)}${r.noiseFloorDb.toFixed(1).padStart(11)}   ${h}`,
    );
    lines.push(
      `        ${ok(Math.abs(r.errorPct) < 6, `firing peak off by ${r.errorPct.toFixed(1)}% at ${rpm} rpm`)}` +
        ` firing peak · ${ok(r.peakDb - r.noiseFloorDb > 12, `only ${(r.peakDb - r.noiseFloorDb).toFixed(1)} dB above floor at ${rpm}`)}` +
        ` fundamental ${(r.peakDb - r.noiseFloorDb).toFixed(1)} dB over floor` +
        ` · ${ok(r.stats.peak < 1.0 && r.stats.clipped === 0, `clipping at ${rpm} rpm`)} peak ${r.stats.peak.toFixed(3)}` +
        ` rms ${r.stats.rms.toFixed(4)}`,
    );
    // A real engine has a decaying harmonic series, not one lone tone and not
    // a flat noise bed. Check H2..H4 sit below H1 but well above the floor.
    const h1 = r.harmonics[0];
    const upper = r.harmonics.slice(1, 5);
    const aboveFloor = upper.filter((v) => v > r.noiseFloorDb + 8).length;
    lines.push(
      `        ${ok(aboveFloor >= 3, `only ${aboveFloor} strong harmonics at ${rpm} rpm`)}` +
        ` harmonic series: ${aboveFloor}/4 partials >8 dB over floor, H1 ${h1.toFixed(1)} dB`,
    );
  }

  lines.push('');
  lines.push('=== DRIVE CYCLE — 26.5 s, rendered offline ===');
  const cyc = await measureDriveCycle();
  lines.push(
    `peak ${cyc.stats.peak.toFixed(4)}   rms ${cyc.stats.rms.toFixed(4)}   ` +
      `dc ${cyc.stats.dc.toExponential(2)}   clipped samples ${(cyc.stats.clipped * 100).toFixed(4)}%`,
  );
  lines.push(`  ${ok(cyc.stats.peak < 1.0, 'drive cycle clips')} peak below 1.0`);
  lines.push(`  ${ok(cyc.stats.clipped === 0, 'samples at full scale')} no samples at full scale`);
  lines.push(
    `  ${ok(cyc.stats.rms > 0.02 && cyc.stats.rms < 0.4, `rms ${cyc.stats.rms.toFixed(4)} out of range`)} rms in a sensible range (0.02..0.40)`,
  );
  lines.push(`  ${ok(Math.abs(cyc.stats.dc) < 0.01, 'DC offset')} DC offset negligible`);
  lines.push('');
  lines.push('  t(s)   rms      section');
  for (const w of cyc.windowRms) {
    const bar = '#'.repeat(Math.min(40, Math.round(w.rms * 260)));
    lines.push(`${w.t.toFixed(1).padStart(6)} ${w.rms.toFixed(4)}  ${bar.padEnd(41)}${w.label}`);
  }

  lines.push('');
  lines.push(fails.length === 0 ? '=== ALL CHECKS PASSED ===' : `=== ${fails.length} FAILURES ===`);
  for (const f of fails) lines.push('  ! ' + f);
  return lines.join('\n');
}

if (typeof document !== 'undefined') {
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
}
