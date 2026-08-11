/**
 * ============================================================================
 *  DRIVETRAIN — engine, clutch, gearbox, transfer case, limited-slip diffs.
 * ============================================================================
 *
 * Pure numbers in, pure numbers out: the vehicle hands over the driven wheel
 * speeds and the driver's pedals, and gets back a torque for every wheel plus
 * the engine state for the HUD and audio.
 *
 * The only genuinely subtle bit is the clutch. A clutch is a stiff damper
 * between two small inertias, and at 120 Hz an explicit integration of it
 * blows up (the engine's time constant under `clutch.stiffness` is ~2 ms,
 * a quarter of the step). So the engine speed is advanced *implicitly* with
 * respect to the clutch torque:
 *
 *     w' = (w + dt*(Teng + k*w_driveline)/I) / (1 + dt*k/I)
 *
 * which is unconditionally stable, locks up in about two steps, and still
 * leaves the small steady-state slip a real clutch has. The torque is then
 * read back off the solved speed and clamped to the clutch's capacity.
 *
 * Torque conventions: positive engine torque with a positive gear ratio gives
 * positive wheel torque, which drives the car forwards. Reverse is a negative
 * ratio, so nothing else in the model needs to know reverse exists.
 */

import type { DrivetrainTuning, DifferentialTuning, WheelTuning } from './VehicleTuning';
import type { DriveInput } from '../types';

const RPM_PER_RAD_S = 60 / (2 * Math.PI);
const RAD_S_PER_RPM = 1 / RPM_PER_RAD_S;

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

/** Neutral is 0, reverse is -1, forward gears are 1..n. */
export const GEAR_REVERSE = -1;
export const GEAR_NEUTRAL = 0;

/**
 * How far below the upshift line a downshift has to land before the automatic
 * will take it. Pure anti-hunt hysteresis, rpm.
 */
const HUNT_MARGIN_RPM = 250;

export interface DrivetrainOutput {
  /** Torque delivered to each wheel by the driveline, N·m (indexed like `wheels`). */
  wheelTorque: number[];
  engineRpm: number;
  /** -1 reverse, 0 neutral, 1..n forward. */
  gear: number;
  /** 0 fully open .. 1 fully locked. */
  clutch: number;
  /** Engine torque actually being produced this step, N·m (after limiter and friction). */
  engineTorque: number;
  /** Throttle after the reverse remap — what the vehicle should treat as "go". */
  throttle: number;
  /** Brake pedal after the reverse remap — what the vehicle should feed the brakes. */
  brake: number;
  /** True while a gearchange is cutting torque. */
  shifting: boolean;
  /** True when the transfer case is in low range. */
  lowRange: boolean;
  /** Signed overall ratio from engine to wheels, 0 in neutral / mid-shift. */
  totalRatio: number;
}

export class Drivetrain {
  private readonly t: DrivetrainTuning;
  private readonly wheels: readonly WheelTuning[];

  /** Indices into `wheels` for the driven wheels of each axle. */
  private readonly frontDriven: number[] = [];
  private readonly rearDriven: number[] = [];

  /** Engine speed, rad/s. */
  private engineOmega: number;
  private gear: number;
  private clutchEngage = 0;

  private shiftTimer = 0;
  private shiftCooldown = 0;
  private targetGear = 1;
  private limiterTimer = 0;
  private lowRange = false;

  private prevShiftUp = false;
  private prevShiftDown = false;
  private reverseHold = 0;
  private forwardHold = 0;

  readonly out: DrivetrainOutput;

  constructor(tuning: DrivetrainTuning, wheels: readonly WheelTuning[]) {
    this.t = tuning;
    this.wheels = wheels;
    for (let i = 0; i < wheels.length; i++) {
      const w = wheels[i]!;
      if (!w.driven) continue;
      (w.axle === 'front' ? this.frontDriven : this.rearDriven).push(i);
    }
    this.engineOmega = tuning.engine.idleRpm * RAD_S_PER_RPM;
    this.gear = 1;
    this.targetGear = 1;
    this.out = {
      wheelTorque: new Array<number>(wheels.length).fill(0),
      engineRpm: tuning.engine.idleRpm,
      gear: 1,
      clutch: 0,
      engineTorque: 0,
      throttle: 0,
      brake: 0,
      shifting: false,
      lowRange: false,
      totalRatio: 0,
    };
  }

  /** Engage or disengage the low-range transfer case. Refused above a crawl. */
  setLowRange(on: boolean, forwardSpeed: number): boolean {
    if (Math.abs(forwardSpeed) > this.t.transmission.directionChangeSpeed * 2) return false;
    this.lowRange = on;
    return true;
  }

  isLowRange(): boolean {
    return this.lowRange;
  }

  get rpm(): number {
    return this.engineOmega * RPM_PER_RAD_S;
  }

  /** Put the driveline back to a cold start. */
  reset(): void {
    this.engineOmega = this.t.engine.idleRpm * RAD_S_PER_RPM;
    this.gear = 1;
    this.targetGear = 1;
    this.clutchEngage = 0;
    this.shiftTimer = 0;
    this.shiftCooldown = 0;
    this.limiterTimer = 0;
    this.reverseHold = 0;
    this.forwardHold = 0;
    this.out.wheelTorque.fill(0);
  }

  /** Engine torque at wide-open throttle, N·m, linearly interpolated. */
  private wotTorque(rpm: number): number {
    const curve = this.t.engine.torqueCurve;
    if (curve.length === 0) return 0;
    if (rpm <= curve[0]![0]) return curve[0]![1];
    for (let i = 1; i < curve.length; i++) {
      const [r1, t1] = curve[i]!;
      if (rpm <= r1) {
        const [r0, t0] = curve[i - 1]!;
        const f = r1 > r0 ? (rpm - r0) / (r1 - r0) : 0;
        return t0 + (t1 - t0) * f;
      }
    }
    // Past the end of the curve, fall away rather than hold flat.
    const last = curve[curve.length - 1]!;
    const drop = 1 - (rpm - last[0]) / 2000;
    return last[1] * clamp(drop, 0, 1);
  }

  /** Signed engine-to-wheel ratio for a gear index. 0 for neutral. */
  private ratioFor(gear: number): number {
    const tr = this.t.transmission;
    const transfer = this.lowRange ? tr.transferLow : tr.transferHigh;
    if (gear === GEAR_NEUTRAL) return 0;
    if (gear < 0) return -tr.reverseRatio * transfer * tr.finalDrive;
    const g = tr.gearRatios[gear - 1];
    if (g === undefined) return 0;
    return g * transfer * tr.finalDrive;
  }

  private requestGear(gear: number): void {
    const tr = this.t.transmission;
    if (gear === this.gear || this.shiftTimer > 0) return;
    if (gear > tr.gearRatios.length) return;
    if (gear < GEAR_REVERSE) return;
    this.targetGear = gear;
    this.shiftTimer = tr.shiftTime;
    this.shiftCooldown = tr.shiftTime + tr.shiftCooldown;
  }

  /**
   * Advance the driveline one fixed step.
   *
   * @param wheelSpin  spin speed of every wheel, rad/s (positive = rolling forwards)
   * @param forwardSpeed  chassis forward velocity, m/s (signed)
   */
  update(dt: number, input: DriveInput, wheelSpin: readonly number[], forwardSpeed: number): DrivetrainOutput {
    const eng = this.t.engine;
    const tr = this.t.transmission;
    const out = this.out;

    // ---- gear selection -------------------------------------------------
    this.shiftCooldown = Math.max(0, this.shiftCooldown - dt);
    if (this.shiftTimer > 0) {
      this.shiftTimer -= dt;
      if (this.shiftTimer <= 0) {
        this.shiftTimer = 0;
        this.gear = this.targetGear;
      }
    }

    const rpmNow = this.engineOmega * RPM_PER_RAD_S;

    // Manual overrides always work, in automatic or not.
    const upEdge = input.shiftUp && !this.prevShiftUp;
    const downEdge = input.shiftDown && !this.prevShiftDown;
    this.prevShiftUp = input.shiftUp;
    this.prevShiftDown = input.shiftDown;

    const nearStop = Math.abs(forwardSpeed) < tr.directionChangeSpeed;
    if (upEdge) {
      if (this.gear === GEAR_REVERSE) this.requestGear(nearStop ? GEAR_NEUTRAL : GEAR_REVERSE);
      else this.requestGear(Math.min(tr.gearRatios.length, this.gear + 1));
    } else if (downEdge) {
      if (this.gear <= 1) this.requestGear(nearStop ? this.gear - 1 : this.gear);
      else this.requestGear(this.gear - 1);
    }

    // Arcade reverse: the brake pedal, held at a standstill, selects reverse
    // and then acts as the reverse throttle.
    let throttle = input.throttle;
    let brake = input.brake;
    if (tr.autoReverse) {
      if (this.gear >= GEAR_NEUTRAL && nearStop && input.brake > 0.5 && input.throttle < 0.1) {
        this.reverseHold += dt;
        if (this.reverseHold >= tr.autoReverseDelay) {
          this.requestGear(GEAR_REVERSE);
          this.reverseHold = 0;
        }
      } else {
        this.reverseHold = 0;
      }
      if (this.gear === GEAR_REVERSE) {
        // Pedals swap: "brake" is go, "throttle" is stop.
        throttle = input.brake;
        brake = input.throttle;
        if (forwardSpeed > -tr.directionChangeSpeed && input.throttle > 0.5) {
          this.forwardHold += dt;
          if (this.forwardHold >= tr.autoReverseDelay) {
            this.requestGear(1);
            this.forwardHold = 0;
          }
        } else {
          this.forwardHold = 0;
        }
      } else {
        this.forwardHold = 0;
      }
    }
    if (this.gear === GEAR_NEUTRAL && throttle > 0.05 && this.shiftTimer <= 0) this.requestGear(1);

    // Automatic shift schedule. Thresholds slide with throttle so a lifted
    // pedal short-shifts and a floored one holds to the redline.
    if (tr.automatic && this.gear >= 1 && this.shiftTimer <= 0 && this.shiftCooldown <= 0) {
      const upRpm = eng.idleRpm + (tr.upshiftRpmLow - eng.idleRpm) + (tr.upshiftRpmHigh - tr.upshiftRpmLow) * throttle;
      const downRpm = tr.downshiftRpmLow + (tr.downshiftRpmHigh - tr.downshiftRpmLow) * throttle;
      if (rpmNow > upRpm && this.gear < tr.gearRatios.length) {
        this.requestGear(this.gear + 1);
      } else if (rpmNow < downRpm && this.gear > 1) {
        const cur = this.ratioFor(this.gear);
        const next = this.ratioFor(this.gear - 1);
        const projected = cur !== 0 ? (rpmNow * next) / cur : 0;
        // Two refusals, and the second one is what keeps the box honest: a
        // downshift that would land the engine above the upshift line puts the
        // gearbox straight back where it came from, and the pair oscillate
        // forever. Requiring a margin below `upRpm` makes hunting impossible
        // for any ratio set, not just the one that happens to be tuned in.
        const wouldHunt = projected > upRpm - HUNT_MARGIN_RPM;
        if (projected < eng.redlineRpm * tr.downshiftRpmCeilingFraction && !wouldHunt) {
          this.requestGear(this.gear - 1);
        }
      }
    }

    const shifting = this.shiftTimer > 0;
    const ratio = shifting ? 0 : this.ratioFor(this.gear);

    // ---- driveline speed reflected to the engine ------------------------
    const wFront = axleAverage(wheelSpin, this.frontDriven);
    const wRear = axleAverage(wheelSpin, this.rearDriven);
    const split = clamp(this.t.frontTorqueSplit, 0, 1);
    const haveF = this.frontDriven.length > 0;
    const haveR = this.rearDriven.length > 0;
    let wProp: number;
    if (haveF && haveR) wProp = split * wFront + (1 - split) * wRear;
    else if (haveF) wProp = wFront;
    else if (haveR) wProp = wRear;
    else wProp = 0;
    const omegaDriveline = ratio !== 0 ? wProp * ratio : this.engineOmega;

    // ---- clutch engagement ----------------------------------------------
    const cl = this.t.clutch;
    const drivelineRpm = Math.abs(omegaDriveline) * RPM_PER_RAD_S;
    let engage: number;
    if (shifting || ratio === 0) {
      engage = 0;
    } else if (drivelineRpm >= cl.lockRpm) {
      engage = 1;
    } else {
      // Below lock-up the clutch is a proportional governor holding the engine
      // at a launch speed that scales with throttle. That is what gives a
      // controllable standing start instead of a bog or a bang.
      const speedEngage = clamp((drivelineRpm - eng.stallRpm) / Math.max(1, cl.lockRpm - eng.stallRpm), 0, 1);
      const targetRpm = eng.idleRpm + (cl.launchRpm - eng.idleRpm) * clamp(throttle, 0, 1);
      const launchEngage = clamp((rpmNow - targetRpm) / cl.launchBandRpm + 0.5, 0, 1);
      engage = throttle > 0.02 ? Math.max(speedEngage, launchEngage) : speedEngage;
    }
    // Never hold the clutch in hard enough to stall the engine.
    const stallGuard = clamp((rpmNow - eng.stallRpm) / Math.max(1, eng.idleRpm - eng.stallRpm), 0, 1);
    engage *= stallGuard;
    // Move towards the target rather than snapping, so a shift feels mechanical.
    const engageRate = engage > this.clutchEngage ? dt / Math.max(1e-4, tr.shiftTime) : dt / 0.06;
    this.clutchEngage += clamp(engage - this.clutchEngage, -engageRate, engageRate);
    this.clutchEngage = clamp(this.clutchEngage, 0, 1);

    // ---- engine ----------------------------------------------------------
    const idleAssist = clamp((eng.idleRpm - rpmNow) * eng.idleGovernorGain, 0, eng.idleMaxThrottle);
    const effThrottle = clamp(Math.max(clamp(throttle, 0, 1), idleAssist), 0, 1);

    // Soft rev limiter: fuel tapers over `limiterBandRpm`, and once it is fully
    // past the band a short hard cut gives the classic stutter.
    let fuel = 1;
    if (this.limiterTimer > 0) {
      this.limiterTimer -= dt;
      fuel = 0;
    } else if (rpmNow > eng.redlineRpm) {
      fuel = clamp(1 - (rpmNow - eng.redlineRpm) / Math.max(1, eng.limiterBandRpm), 0, 1);
      if (rpmNow > eng.redlineRpm + eng.limiterBandRpm) {
        this.limiterTimer = eng.limiterCutTime;
        fuel = 0;
      }
    }

    const driveTorque = this.wotTorque(rpmNow) * effThrottle * fuel;
    // Pumping losses fall away as the throttle opens, so engine braking is a
    // lift-off phenomenon rather than a constant drag.
    const frictionScale = 1 - (1 - eng.frictionThrottleRelief) * effThrottle;
    const friction = (eng.frictionConstant + eng.frictionViscous * Math.max(0, this.engineOmega)) * frictionScale;
    const engineTorque = driveTorque - friction;

    // Implicit clutch solve (see the file header for why).
    const k = cl.stiffness * this.clutchEngage;
    const cap = cl.capacity * this.clutchEngage;
    let omegaNew: number;
    let clutchTorque = 0;
    if (k > 1e-6) {
      const a = dt / eng.inertia;
      omegaNew = (this.engineOmega + a * (engineTorque + k * omegaDriveline)) / (1 + a * k);
      clutchTorque = k * (omegaNew - omegaDriveline);
      if (Math.abs(clutchTorque) > cap) {
        clutchTorque = Math.sign(clutchTorque) * cap;
        omegaNew = this.engineOmega + a * (engineTorque - clutchTorque);
      }
    } else {
      omegaNew = this.engineOmega + (dt * engineTorque) / eng.inertia;
    }
    if (!Number.isFinite(omegaNew)) omegaNew = eng.idleRpm * RAD_S_PER_RPM;
    // The starter/idle control never lets it actually die.
    this.engineOmega = clamp(omegaNew, eng.stallRpm * RAD_S_PER_RPM * 0.5, eng.maxRpm * RAD_S_PER_RPM);
    if (!Number.isFinite(clutchTorque)) clutchTorque = 0;

    // ---- torque out to the wheels ---------------------------------------
    const wheelTorque = out.wheelTorque;
    wheelTorque.fill(0);
    if (ratio !== 0) {
      const axleTorque = clutchTorque * ratio * tr.efficiency;
      let frontTorque: number;
      let rearTorque: number;
      if (haveF && haveR) {
        frontTorque = axleTorque * split;
        rearTorque = axleTorque * (1 - split);
        // Centre LSD: bias towards whichever axle is turning slower.
        const bias = lsdBias(this.t.centreDiff, axleTorque, wFront - wRear);
        frontTorque -= bias;
        rearTorque += bias;
      } else {
        frontTorque = haveF ? axleTorque : 0;
        rearTorque = haveR ? axleTorque : 0;
      }
      applyAxle(wheelTorque, wheelSpin, this.frontDriven, frontTorque, this.t.frontDiff);
      applyAxle(wheelTorque, wheelSpin, this.rearDriven, rearTorque, this.t.rearDiff);
    }

    for (let i = 0; i < wheelTorque.length; i++) {
      if (!Number.isFinite(wheelTorque[i]!)) wheelTorque[i] = 0;
    }

    out.engineRpm = this.engineOmega * RPM_PER_RAD_S;
    out.gear = this.gear;
    out.clutch = this.clutchEngage;
    out.engineTorque = engineTorque;
    out.throttle = clamp(throttle, 0, 1);
    out.brake = clamp(brake, 0, 1);
    out.shifting = shifting;
    out.lowRange = this.lowRange;
    out.totalRatio = ratio;
    return out;
  }
}

function axleAverage(spin: readonly number[], idx: readonly number[]): number {
  if (idx.length === 0) return 0;
  let s = 0;
  for (let i = 0; i < idx.length; i++) s += spin[idx[i]!] ?? 0;
  return s / idx.length;
}

/**
 * Torque a limited-slip differential biases from the faster side to the slower.
 *
 * The `preload` term is the important one off-road: it applies with no input
 * torque at all, so a wheel dangling in the air cannot take all the drive away
 * from the wheel that still has grip. The torque-sensitive term is the ramp,
 * and `tanh` gives a smooth, bounded engagement with slip speed.
 */
function lsdBias(diff: DifferentialTuning, axleTorque: number, deltaOmega: number): number {
  const capacity = Math.min(diff.maxBias, diff.preload + diff.torqueSensitivity * Math.abs(axleTorque));
  return capacity * Math.tanh(deltaOmega / Math.max(1e-3, diff.slipReference));
}

function applyAxle(
  wheelTorque: number[],
  spin: readonly number[],
  idx: readonly number[],
  axleTorque: number,
  diff: DifferentialTuning,
): void {
  if (idx.length === 0) return;
  if (idx.length === 1) {
    wheelTorque[idx[0]!] = axleTorque;
    return;
  }
  const a = idx[0]!;
  const b = idx[1]!;
  const half = axleTorque / 2;
  const bias = lsdBias(diff, axleTorque, (spin[a] ?? 0) - (spin[b] ?? 0));
  wheelTorque[a] = half - bias;
  wheelTorque[b] = half + bias;
  // Any extra wheels on the axle (not a thing on the Jeep, but keep it sane).
  for (let i = 2; i < idx.length; i++) wheelTorque[idx[i]!] = half;
}
