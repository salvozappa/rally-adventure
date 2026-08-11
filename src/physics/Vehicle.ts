/**
 * ============================================================================
 *  VEHICLE — a custom raycast car on one Rapier rigid body.
 * ============================================================================
 *
 * Rapier's own DynamicRayCastVehicleController is a straight port of Bullet's
 * demo vehicle; its tyre model is a friction *clamp*, not a slip curve, so it
 * can never feel like it is sliding. Everything below is built from scratch on
 * top of a single dynamic body:
 *
 *   1. reset the accumulated forces (Rapier keeps user forces between steps)
 *   2. read the body pose and velocity, roll the interpolation buffers
 *   3. advance the steering rack
 *   4. shapecast a sphere down each strut to find the ground
 *   5. springs + dampers + anti-roll bars -> the vertical load on each tyre
 *   6. the drivetrain turns pedals + wheel speeds into a torque per wheel
 *   7. per wheel: relax the slip states, solve the wheel's rotational ODE
 *      semi-implicitly against the tyre curve, apply the resulting forces at
 *      the contact patch
 *   8. aero, and a light attitude assist while airborne
 *   9. publish `state`
 *
 * The caller runs this immediately before `world.step()`, at a fixed 120 Hz.
 *
 * A note on why step 7 is "semi-implicit": a wheel is a 2 kg-m^2 inertia bolted
 * to a tyre whose longitudinal stiffness is ~70 kN per unit of slip. Integrate
 * that explicitly at 120 Hz and it rings itself apart at any speed above
 * walking pace. Linearising the tyre curve about the current slip and solving
 * the wheel speed and slip ratio together makes it unconditionally stable
 * without any of the damping hacks that make a car feel numb.
 */

import * as THREE from 'three';
import type RAPIER from '@dimforge/rapier3d-compat';
import type { DriveInput, PhysicsContext, SurfaceKind, VehicleState, WheelState } from '../types';
import { GROUP, interactionGroups } from '../types';
import type { AxleSuspensionTuning, SurfaceTuning, VehicleTuning, WheelTuning } from './VehicleTuning';
import { Drivetrain } from './Drivetrain';
import { longitudinalSlope, makeTireForces, tireForces, type TireForces } from './TireModel';

const WORLD_UP = new THREE.Vector3(0, 1, 0);
const GRAVITY_Y = 9.81;

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

let warned = new Set<string>();
function warnOnce(key: string, message: string): void {
  if (warned.has(key)) return;
  warned.add(key);
  console.warn(`[Vehicle] ${message}`);
}
/** Test-harness hook: forget which warnings have already been printed. */
export function resetVehicleWarnings(): void {
  warned = new Set<string>();
}

function finite(v: number): boolean {
  return Number.isFinite(v);
}
function finiteVec(v: THREE.Vector3): boolean {
  return Number.isFinite(v.x) && Number.isFinite(v.y) && Number.isFinite(v.z);
}

/** Everything the sim keeps per wheel that is not part of the public WheelState. */
interface WheelRuntime {
  readonly tuning: WheelTuning;
  readonly susp: AxleSuspensionTuning;
  readonly attach: THREE.Vector3;
  /** Spin speed, rad/s. Positive = rolling forwards. */
  spin: number;
  /** Accumulated spin angle for the visual, rad. */
  spinAngle: number;
  /** Relaxed longitudinal slip ratio. */
  slipRatio: number;
  /** Relaxed tan(slip angle). */
  tanSlipAngle: number;
  steerAngle: number;
  /** Current strut length, attach point to hub, m. */
  length: number;
  /** Compression distance, m: restLength - length, clamped to [0, travel]. */
  compressionDist: number;
  grounded: boolean;
  load: number;
  surface: SurfaceKind;
  /** World-space contact point and outward ground normal. */
  contact: THREE.Vector3;
  normal: THREE.Vector3;
  /** World-space hub centre. */
  hub: THREE.Vector3;
  /** Ground-plane basis at the contact patch. */
  fwd: THREE.Vector3;
  right: THREE.Vector3;
  /** Interpolation buffers for the visual transform. */
  prevPos: THREE.Vector3;
  curPos: THREE.Vector3;
  prevQuat: THREE.Quaternion;
  curQuat: THREE.Quaternion;
  /**
   * Ackermann bookkeeping: true when the wheel sits at negative X. With the
   * chassis facing +Z that is the driver's RIGHT, despite the name — the name
   * predates the convention being pinned down and is kept because the wheel
   * tuning entries are named the same way.
   */
  readonly isLeft: boolean;
}

export class Vehicle {
  readonly state: VehicleState;

  private readonly ctx: PhysicsContext;
  private readonly tuning: VehicleTuning;
  private readonly rapier: typeof RAPIER;
  private readonly body: RAPIER.RigidBody;
  private readonly collider: RAPIER.Collider;
  private readonly castShape: RAPIER.Ball;
  private readonly castFilter: number;
  private readonly drivetrain: Drivetrain;

  private readonly wheels: WheelRuntime[] = [];
  private readonly frontIdx: number[] = [];
  private readonly rearIdx: number[] = [];
  private readonly wheelTorque: number[];

  /** Steering rack angle, rad (before Ackermann). */
  private steerAngle = 0;
  private airTime = 0;
  private recoverTimer = 0;
  private prevRecover = false;
  private disposed = false;

  /** Chassis interpolation buffers. */
  private readonly prevPos = new THREE.Vector3();
  private readonly curPos = new THREE.Vector3();
  private readonly prevQuat = new THREE.Quaternion();
  private readonly curQuat = new THREE.Quaternion();
  private hasPrev = false;

  private readonly prevVel = new THREE.Vector3();
  private readonly wheelbase: number;
  private readonly frontTrack: number;

  // --- scratch, allocated once. This runs 120 times a second. ---
  private readonly _q = new THREE.Quaternion();
  private readonly _qi = new THREE.Quaternion();
  private readonly _qSteer = new THREE.Quaternion();
  private readonly _qSpin = new THREE.Quaternion();
  private readonly _up = new THREE.Vector3();
  private readonly _fwd = new THREE.Vector3();
  private readonly _right = new THREE.Vector3();
  private readonly _down = new THREE.Vector3();
  private readonly _lin = new THREE.Vector3();
  private readonly _ang = new THREE.Vector3();
  private readonly _com = new THREE.Vector3();
  private readonly _p = new THREE.Vector3();
  private readonly _r = new THREE.Vector3();
  private readonly _v = new THREE.Vector3();
  private readonly _f = new THREE.Vector3();
  private readonly _t = new THREE.Vector3();
  private readonly _tmp = new THREE.Vector3();
  private readonly _tire: TireForces = makeTireForces();
  private readonly _tireScratch: TireForces = makeTireForces();
  private readonly _castOrigin = { x: 0, y: 0, z: 0 };
  private readonly _castDir = { x: 0, y: 0, z: 0 };
  private readonly _identityRot = { x: 0, y: 0, z: 0, w: 1 };
  private readonly _forceOut = { x: 0, y: 0, z: 0 };
  private readonly _pointOut = { x: 0, y: 0, z: 0 };

  constructor(ctx: PhysicsContext, tuning: VehicleTuning, spawn: { position: THREE.Vector3; heading?: number }) {
    this.ctx = ctx;
    this.tuning = tuning;
    this.rapier = ctx.rapier;
    const R = ctx.rapier;
    const ch = tuning.chassis;

    const heading = spawn.heading ?? 0;
    const q = new THREE.Quaternion().setFromAxisAngle(WORLD_UP, heading);

    const bodyDesc = R.RigidBodyDesc.dynamic()
      .setTranslation(spawn.position.x, spawn.position.y, spawn.position.z)
      .setRotation({ x: q.x, y: q.y, z: q.z, w: q.w })
      .setLinearDamping(ch.linearDamping)
      .setAngularDamping(ch.angularDamping)
      // The car is a fast, small body over big terrain triangles: CCD is the
      // difference between landing a jump and disappearing through the floor.
      .setCcdEnabled(true)
      .setCanSleep(false);
    this.body = ctx.world.createRigidBody(bodyDesc);

    // Mass, centre of mass and inertia all come from the tuning rather than
    // from the collider's geometry: a real vehicle carries its mass far lower
    // than a uniform box of the same size would.
    const colDesc = R.ColliderDesc.cuboid(ch.halfExtents.x, ch.halfExtents.y, ch.halfExtents.z)
      .setTranslation(ch.colliderOffset.x, ch.colliderOffset.y, ch.colliderOffset.z)
      .setMass(0)
      .setFriction(ch.colliderFriction)
      .setRestitution(ch.colliderRestitution)
      .setCollisionGroups(interactionGroups(GROUP.CHASSIS, GROUP.TERRAIN | GROUP.PROP));
    this.collider = ctx.world.createCollider(colDesc, this.body);
    this.body.setAdditionalMassProperties(
      ch.mass,
      ch.centreOfMass,
      ch.inertia,
      this._identityRot,
      true,
    );

    this.castShape = new R.Ball(tuning.tire.radius);
    // The wheel probe is its own membership so terrain and props can opt in or
    // out; it must never see the chassis it belongs to.
    this.castFilter = interactionGroups(GROUP.WHEEL_RAY, GROUP.TERRAIN | GROUP.PROP);

    // ---- wheels ---------------------------------------------------------
    for (let i = 0; i < tuning.wheels.length; i++) {
      const w = tuning.wheels[i]!;
      const susp = w.axle === 'front' ? tuning.suspension.front : tuning.suspension.rear;
      this.wheels.push({
        tuning: w,
        susp,
        attach: new THREE.Vector3(w.attach.x, w.attach.y, w.attach.z),
        spin: 0,
        spinAngle: 0,
        slipRatio: 0,
        tanSlipAngle: 0,
        steerAngle: 0,
        length: tuning.suspension.restLength,
        compressionDist: 0,
        grounded: false,
        load: 0,
        surface: 'dirt',
        contact: new THREE.Vector3(),
        normal: new THREE.Vector3(0, 1, 0),
        hub: new THREE.Vector3(),
        fwd: new THREE.Vector3(0, 0, 1),
        right: new THREE.Vector3(1, 0, 0),
        prevPos: new THREE.Vector3(),
        curPos: new THREE.Vector3(),
        prevQuat: new THREE.Quaternion(),
        curQuat: new THREE.Quaternion(),
        isLeft: w.attach.x < 0,
      });
      (w.axle === 'front' ? this.frontIdx : this.rearIdx).push(i);
    }
    this.wheelTorque = new Array<number>(this.wheels.length).fill(0);
    this.drivetrain = new Drivetrain(tuning.drivetrain, tuning.wheels);

    // Ackermann geometry: use the tuning if given, otherwise measure it.
    this.wheelbase = tuning.steering.wheelbase > 0 ? tuning.steering.wheelbase : this.measureWheelbase();
    this.frontTrack = tuning.steering.frontTrack > 0 ? tuning.steering.frontTrack : this.measureFrontTrack();

    // ---- published state ------------------------------------------------
    const wheelStates: WheelState[] = this.wheels.map((rt) => ({
      position: new THREE.Vector3(),
      quaternion: new THREE.Quaternion(),
      compression: 0,
      grounded: false,
      slipRatio: 0,
      slipAngle: 0,
      load: 0,
      surface: rt.surface,
      spin: 0,
      steerAngle: 0,
    }));
    this.state = {
      position: spawn.position.clone(),
      quaternion: q.clone(),
      velocity: new THREE.Vector3(),
      angularVelocity: new THREE.Vector3(),
      forwardSpeed: 0,
      speed: 0,
      engineRpm: tuning.drivetrain.engine.idleRpm,
      gear: 1,
      clutch: 0,
      wheels: wheelStates,
      airborne: true,
      localAccel: new THREE.Vector3(),
    };

    this.syncTransforms(true);
  }

  // -------------------------------------------------------------------------
  //  Public API
  // -------------------------------------------------------------------------

  /** The underlying rigid body, for anything that needs to attach to the car. */
  get rigidBody(): RAPIER.RigidBody {
    return this.body;
  }

  /** Engage or release the low-range transfer case. Only works at a crawl. */
  setLowRange(on: boolean): boolean {
    return this.drivetrain.setLowRange(on, this.state.forwardSpeed);
  }

  isLowRange(): boolean {
    return this.drivetrain.isLowRange();
  }

  fixedUpdate(dt: number, input: DriveInput, surfaceAt: (x: number, z: number) => SurfaceKind): void {
    if (this.disposed) return;
    if (!finite(dt) || dt <= 0) {
      warnOnce('dt', `non-finite dt (${dt}); skipping step`);
      return;
    }
    // A step longer than 1/30 s makes the suspension explicit terms unstable;
    // the caller is supposed to run this fixed, so just clamp and carry on.
    const h = clamp(dt, 1e-4, 1 / 30);

    // Rapier keeps user forces between steps, so they must be cleared or they
    // integrate twice.
    this.body.resetForces(false);
    this.body.resetTorques(false);

    this.recoverTimer = Math.max(0, this.recoverTimer - h);
    const recoverEdge = input.recover && !this.prevRecover;
    this.prevRecover = input.recover;
    if (recoverEdge) this.recover();

    this.readBody();
    this.rollInterpolation();

    const st = this.state;
    const forwardSpeed = this._lin.dot(this._fwd);
    const speed = this._lin.length();
    st.forwardSpeed = forwardSpeed;
    st.speed = speed;

    this.updateSteering(h, input, forwardSpeed);

    // ---- 1. find the ground under each wheel ----------------------------
    for (let i = 0; i < this.wheels.length; i++) this.castWheel(this.wheels[i]!, surfaceAt);

    // ---- 2. suspension: springs, dampers, anti-roll ---------------------
    this.applySuspension(h);

    // ---- 3. driveline ---------------------------------------------------
    for (let i = 0; i < this.wheels.length; i++) this.wheelTorque[i] = this.wheels[i]!.spin;
    const dv = this.drivetrain.update(h, input, this.wheelTorque, forwardSpeed);
    for (let i = 0; i < this.wheels.length; i++) this.wheelTorque[i] = dv.wheelTorque[i] ?? 0;

    // ---- 4. tyres -------------------------------------------------------
    const brakes = this.tuning.brakes;
    const handbrake = clamp(input.handbrake, 0, 1);
    for (let i = 0; i < this.wheels.length; i++) {
      const w = this.wheels[i]!;
      const axleBias = w.tuning.axle === 'front' ? brakes.frontBias : 1 - brakes.frontBias;
      // Bias is per axle; halve it again across the two wheels of that axle.
      const count = w.tuning.axle === 'front' ? this.frontIdx.length : this.rearIdx.length;
      const brakeTorque =
        (brakes.maxTorque * axleBias * dv.brake) / Math.max(1, count) +
        brakes.handbrakeTorque * w.tuning.handbrakeShare * handbrake;
      this.updateWheel(h, w, this.wheelTorque[i] ?? 0, brakeTorque);
    }

    // ---- 5. aero and assists --------------------------------------------
    this.applyAero(speed);
    this.applyAirborneAssist(h);

    // ---- 6. publish ------------------------------------------------------
    this.publish(h, dv);
  }

  /**
   * Blend the last two physics poses. `alpha` is the fraction of a step that
   * has elapsed since the most recent one. Wheel transforms are interpolated
   * too, so the car and its wheels never disagree by a frame.
   */
  getInterpolated(alpha: number, outPos: THREE.Vector3, outQuat: THREE.Quaternion): void {
    const a = clamp(finite(alpha) ? alpha : 0, 0, 1);
    outPos.lerpVectors(this.prevPos, this.curPos, a);
    outQuat.copy(this.prevQuat).slerp(this.curQuat, a);
    for (let i = 0; i < this.wheels.length; i++) {
      const w = this.wheels[i]!;
      const ws = this.state.wheels[i]!;
      ws.position.lerpVectors(w.prevPos, w.curPos, a);
      ws.quaternion.copy(w.prevQuat).slerp(w.curQuat, a);
    }
  }

  /**
   * Flip the car back onto its wheels, keeping its heading. Also works as an
   * unstick when the car is upright but wedged. Rate-limited by
   * `assists.recoverCooldown`.
   */
  recover(): void {
    if (this.disposed || this.recoverTimer > 0) return;
    const a = this.tuning.assists;
    this.recoverTimer = a.recoverCooldown;

    const t = this.body.translation();
    const rot = this.body.rotation();
    this._q.set(rot.x, rot.y, rot.z, rot.w);
    // Keep whatever heading the car had; if it is nose-up, fall back to the
    // roof direction so the recovery is not random.
    this._fwd.set(0, 0, 1).applyQuaternion(this._q);
    if (Math.abs(this._fwd.y) > 0.95) this._fwd.set(0, 1, 0).applyQuaternion(this._q);
    const heading = Math.atan2(this._fwd.x, this._fwd.z);
    this._q.setFromAxisAngle(WORLD_UP, finite(heading) ? heading : 0);

    this.body.setRotation({ x: this._q.x, y: this._q.y, z: this._q.z, w: this._q.w }, true);
    this.body.setTranslation({ x: t.x, y: t.y + a.recoverLift, z: t.z }, true);
    const lv = this.body.linvel();
    this.body.setLinvel({ x: lv.x * 0.2, y: 0, z: lv.z * 0.2 }, true);
    this.body.setAngvel({ x: 0, y: 0, z: 0 }, true);
    this.body.resetForces(false);
    this.body.resetTorques(false);
    this.resetWheelStates();
    this.syncTransforms(true);
  }

  /** Teleport the car, upright and stationary. */
  respawn(position: THREE.Vector3, heading = 0): void {
    if (this.disposed) return;
    this._q.setFromAxisAngle(WORLD_UP, finite(heading) ? heading : 0);
    this.body.setTranslation({ x: position.x, y: position.y, z: position.z }, true);
    this.body.setRotation({ x: this._q.x, y: this._q.y, z: this._q.z, w: this._q.w }, true);
    this.body.setLinvel({ x: 0, y: 0, z: 0 }, true);
    this.body.setAngvel({ x: 0, y: 0, z: 0 }, true);
    this.body.resetForces(false);
    this.body.resetTorques(false);
    this.steerAngle = 0;
    this.airTime = 0;
    this.recoverTimer = 0;
    this.drivetrain.reset();
    this.resetWheelStates();
    this.prevVel.set(0, 0, 0);
    this.state.localAccel.set(0, 0, 0);
    this.syncTransforms(true);
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.ctx.world.removeCollider(this.collider, false);
    this.ctx.world.removeRigidBody(this.body);
  }

  // -------------------------------------------------------------------------
  //  Internals
  // -------------------------------------------------------------------------

  private measureWheelbase(): number {
    let front = 0;
    let rear = 0;
    for (const w of this.tuning.wheels) {
      if (w.axle === 'front') front = w.attach.z;
      else rear = w.attach.z;
    }
    return Math.max(0.5, Math.abs(front - rear));
  }

  private measureFrontTrack(): number {
    let min = Infinity;
    let max = -Infinity;
    for (const w of this.tuning.wheels) {
      if (w.axle !== 'front') continue;
      min = Math.min(min, w.attach.x);
      max = Math.max(max, w.attach.x);
    }
    return Number.isFinite(min) && max > min ? max - min : 1.5;
  }

  private resetWheelStates(): void {
    for (const w of this.wheels) {
      w.spin = 0;
      w.slipRatio = 0;
      w.tanSlipAngle = 0;
      w.length = this.tuning.suspension.restLength;
      w.compressionDist = 0;
      w.grounded = false;
      w.load = 0;
    }
  }

  /** Read the body's pose and velocity into the scratch vectors + basis. */
  private readBody(): void {
    const t = this.body.translation();
    const r = this.body.rotation();
    const lv = this.body.linvel();
    const av = this.body.angvel();
    const com = this.body.worldCom();
    this._p.set(t.x, t.y, t.z);
    this._q.set(r.x, r.y, r.z, r.w);
    this._qi.copy(this._q).invert();
    this._lin.set(lv.x, lv.y, lv.z);
    this._ang.set(av.x, av.y, av.z);
    this._com.set(com.x, com.y, com.z);
    this._right.set(1, 0, 0).applyQuaternion(this._q);
    this._up.set(0, 1, 0).applyQuaternion(this._q);
    this._fwd.set(0, 0, 1).applyQuaternion(this._q);
    this._down.copy(this._up).negate();

    if (!finiteVec(this._p) || !finiteVec(this._lin) || !finiteVec(this._ang)) {
      warnOnce('nan-body', 'rigid body went non-finite; resetting velocities');
      this.body.setLinvel({ x: 0, y: 0, z: 0 }, true);
      this.body.setAngvel({ x: 0, y: 0, z: 0 }, true);
      this._lin.set(0, 0, 0);
      this._ang.set(0, 0, 0);
      if (!finiteVec(this._p)) {
        this._p.set(0, 5, 0);
        this.body.setTranslation({ x: 0, y: 5, z: 0 }, true);
      }
    }
  }

  /** Copy `cur` into `prev` and refresh `cur` from the body. */
  private rollInterpolation(): void {
    if (!this.hasPrev) {
      this.syncTransforms(true);
      return;
    }
    this.prevPos.copy(this.curPos);
    this.prevQuat.copy(this.curQuat);
    for (const w of this.wheels) {
      w.prevPos.copy(w.curPos);
      w.prevQuat.copy(w.curQuat);
    }
    this.curPos.copy(this._p);
    this.curQuat.copy(this._q);
  }

  /** Force both interpolation buffers to the body's current pose. */
  private syncTransforms(both: boolean): void {
    const t = this.body.translation();
    const r = this.body.rotation();
    this.curPos.set(t.x, t.y, t.z);
    this.curQuat.set(r.x, r.y, r.z, r.w);
    if (both) {
      this.prevPos.copy(this.curPos);
      this.prevQuat.copy(this.curQuat);
      this.hasPrev = true;
    }
    this._q.copy(this.curQuat);
    this.updateWheelVisuals(true);
  }

  // ---- steering -----------------------------------------------------------

  private updateSteering(dt: number, input: DriveInput, forwardSpeed: number): void {
    const s = this.tuning.steering;
    const v = Math.abs(forwardSpeed);

    const speedFade = 1 / (1 + (v / Math.max(0.1, s.speedFalloffRef)) ** 2);
    const maxAngle = s.maxAngleHighSpeed + (s.maxAngleLowSpeed - s.maxAngleHighSpeed) * speedFade;
    const rateFade = 1 / (1 + (v / Math.max(0.1, s.rateFalloffRef)) ** 2);
    const rate = s.rateHighSpeed + (s.rateLowSpeed - s.rateHighSpeed) * rateFade;

    // `DriveInput.steer` is +1 for a RIGHT turn (see types.ts). `steerAngle` is
    // a yaw about the chassis +Y, so the two differ by a sign, and it is worth
    // being explicit about why rather than leaving a bare minus here.
    //
    // The chassis frame is forward = +Z, up = +Y. The right hand of anything
    // facing +Z with +Y up is forward x up = -X, NOT +X — that is the one label
    // in VehicleTuning's header that does not survive contact with the rest of
    // the convention. A right turn therefore swings the nose toward -X, which is
    // a NEGATIVE rotation about +Y. Get this backwards and the car steers away
    // from the key you are holding the moment the camera sits behind it, which
    // is exactly the bug this replaces.
    const driver = -clamp(finite(input.steer) ? input.steer : 0, -1, 1);
    let target = driver * maxAngle;

    // Counter-steer assist. It reads the rear axle's own sideslip, so it only
    // wakes up when the back is actually stepping out, and it is capped hard
    // enough that the driver is always the one in charge.
    if (s.counterSteerGain > 0 && v > s.counterSteerMinSpeed) {
      this._r.set(0, 0, -this.wheelbase * 0.5).applyQuaternion(this._q).add(this._p).sub(this._com);
      this._v.copy(this._ang).cross(this._r).add(this._lin).applyQuaternion(this._qi);
      const beta = Math.atan2(this._v.x, Math.max(1, Math.abs(this._v.z)));
      if (finite(beta)) {
        const mag = Math.max(0, Math.abs(beta) - s.counterSteerDeadzone);
        const fade = clamp(
          (v - s.counterSteerMinSpeed) / Math.max(0.01, s.counterSteerFullSpeed - s.counterSteerMinSpeed),
          0,
          1,
        );
        const assist = clamp(Math.sign(beta) * mag * s.counterSteerGain * fade, -s.counterSteerMaxAngle, s.counterSteerMaxAngle);
        target = clamp(target + assist, -s.maxAngleLowSpeed, s.maxAngleLowSpeed);
      }
    }

    // The rack returns to centre faster than it is turned away from it.
    const returning = Math.abs(target) < Math.abs(this.steerAngle) || target * this.steerAngle < 0;
    const step = rate * (returning ? s.returnRateScale : 1) * dt;
    this.steerAngle += clamp(target - this.steerAngle, -step, step);
    if (!finite(this.steerAngle)) this.steerAngle = 0;

    // Ackermann: the inside wheel follows a tighter radius, so it must steer
    // more. `ackermann` blends between parallel steering and full geometry.
    const abs = Math.abs(this.steerAngle);
    let inner = abs;
    let outer = abs;
    if (abs > 1e-4 && s.ackermann > 0) {
      const radius = this.wheelbase / Math.tan(abs);
      const half = this.frontTrack * 0.5;
      const geomInner = Math.atan(this.wheelbase / Math.max(0.05, radius - half));
      const geomOuter = Math.atan(this.wheelbase / (radius + half));
      inner = abs + (geomInner - abs) * s.ackermann;
      outer = abs + (geomOuter - abs) * s.ackermann;
    }
    const sign = Math.sign(this.steerAngle);
    for (const w of this.wheels) {
      if (!w.tuning.steered) {
        w.steerAngle = 0;
        continue;
      }
      // The inner wheel is the one on the side the nose is swinging toward.
      // `isLeft` is really "sits at negative X"; with forward at +Z that is the
      // chassis RIGHT, and a negative rack angle is a right turn — so for
      // sign < 0 the negative-X wheel is the inner one.
      const isInner = sign > 0 ? !w.isLeft : w.isLeft;
      w.steerAngle = sign * (isInner ? inner : outer) * w.tuning.steerSign;
    }
  }

  // ---- ground probe -------------------------------------------------------

  /**
   * Sweep a wheel-sized sphere down the strut. A sphere rather than a ray so a
   * wheel rides over the seam between two terrain triangles instead of
   * dropping into it, and so a kerb edge lifts the wheel the way a real tyre's
   * carcass would.
   *
   * The sweep starts at the *attach point*, which is above the wheel, and the
   * time of impact is therefore the strut length directly.
   */
  private castWheel(w: WheelRuntime, surfaceAt: (x: number, z: number) => SurfaceKind): void {
    const susp = this.tuning.suspension;
    this._r.copy(w.attach).applyQuaternion(this._q).add(this._p);
    this._castOrigin.x = this._r.x;
    this._castOrigin.y = this._r.y;
    this._castOrigin.z = this._r.z;
    this._castDir.x = this._down.x;
    this._castDir.y = this._down.y;
    this._castDir.z = this._down.z;

    const maxDist = susp.restLength;
    const hit = this.ctx.world.castShape(
      this._castOrigin,
      this._identityRot,
      this._castDir,
      this.castShape,
      0,
      maxDist,
      true,
      undefined,
      this.castFilter,
      undefined,
      this.body,
    );

    if (hit === null) {
      w.grounded = false;
      w.length = susp.restLength;
      w.compressionDist = 0;
      w.load = 0;
      w.hub.copy(this._r).addScaledVector(this._down, susp.restLength);
      return;
    }

    let toi = hit.time_of_impact;
    if (!finite(toi)) {
      warnOnce('cast-toi', 'shapecast returned a non-finite time of impact');
      w.grounded = false;
      w.length = susp.restLength;
      w.compressionDist = 0;
      w.load = 0;
      return;
    }
    toi = clamp(toi, 0, maxDist);

    // `normal1` comes back as the world-space normal of the surface that was
    // hit. Flip it if it faces away and reject anything close to a wall: a
    // vertical face is for the chassis collider to deal with, not the tyres.
    this._tmp.set(hit.normal1.x, hit.normal1.y, hit.normal1.z);
    const len = this._tmp.length();
    if (!(len > 1e-4) || !finiteVec(this._tmp)) {
      this._tmp.copy(this._up);
    } else {
      this._tmp.multiplyScalar(1 / len);
      if (this._tmp.dot(this._down) > 0) this._tmp.negate();
    }
    if (this._tmp.dot(this._up) < 0.2) {
      w.grounded = false;
      w.length = susp.restLength;
      w.compressionDist = 0;
      w.load = 0;
      w.hub.copy(this._r).addScaledVector(this._down, susp.restLength);
      return;
    }

    w.grounded = true;
    w.normal.copy(this._tmp);
    w.length = toi;
    w.compressionDist = clamp(susp.restLength - toi, 0, susp.travel);
    w.hub.copy(this._r).addScaledVector(this._down, toi);
    if (finite(hit.witness1.x) && finite(hit.witness1.y) && finite(hit.witness1.z)) {
      w.contact.set(hit.witness1.x, hit.witness1.y, hit.witness1.z);
    } else {
      w.contact.copy(w.hub).addScaledVector(w.normal, -this.tuning.tire.radius);
    }
    w.surface = surfaceAt(w.contact.x, w.contact.z) ?? 'dirt';
  }

  // ---- suspension ---------------------------------------------------------

  private applySuspension(dt: number): void {
    const susp = this.tuning.suspension;
    const travel = susp.travel;
    const bumpStopStart = susp.bumpStopZone * travel;

    // Anti-roll: the bar reacts to the *difference* in compression across an
    // axle, adding load to the compressed side. Deliberately soft here — an
    // off-road truck wants to articulate, and a stiff bar lifts wheels.
    const arb = [0, 0, 0, 0];
    this.axleAntiRoll(this.frontIdx, this.tuning.suspension.front.antiRollRate, arb);
    this.axleAntiRoll(this.rearIdx, this.tuning.suspension.rear.antiRollRate, arb);

    for (let i = 0; i < this.wheels.length; i++) {
      const w = this.wheels[i]!;
      if (!w.grounded) {
        w.load = 0;
        continue;
      }
      const x = susp.restLength - w.length; // compression, m (>= 0 when in contact)

      // Damper velocity from the chassis motion at the contact patch, resolved
      // along the strut. Positive = compressing.
      const nDotUp = Math.max(0.2, w.normal.dot(this._up));
      this._r.copy(w.contact).sub(this._com);
      this._v.copy(this._ang).cross(this._r).add(this._lin);
      const vel = clamp(-this._v.dot(w.normal) / nDotUp, -12, 12);

      // Progressive rate: linear coil plus a quadratic bump stop over the last
      // (1 - bumpStopZone) of travel. Force = k*x + kbs*pen^2.
      //
      // The spring is evaluated SEMI-IMPLICITLY, at where the strut will be at
      // the end of the step rather than where it is now. On the coil this is
      // worth an extra 260 N·s/m of damping and changes nothing you can feel.
      // On the bump stop it is the difference between a car and a pogo stick:
      // the stop's local rate at full compression is ~1.8 MN/m, which at 120 Hz
      // is w*dt = 0.5 — explicit integration there returns *more* energy than
      // it absorbs (a 1 m drop measured a coefficient of restitution of 1.07)
      // and the car launches itself off every landing. Evaluating at x + v*dt
      // adds exactly the k*dt of damping that removes the phase error.
      const xEnd = clamp(x + vel * dt, 0, travel * 1.5);
      const spring = w.susp.springRate * xEnd;
      const pen = Math.max(0, xEnd - bumpStopStart);
      let bumpStop = susp.bumpStopStiffness * pen * pen;
      // Hysteresis. A urethane stop returns far less energy than it stores, and
      // that loss is what keeps a bottomed-out landing from catapulting the car
      // straight back into the air. The blend over `bumpStopHysteresisVelocity`
      // keeps the force continuous through the reversal at the bottom of the
      // stroke; a hard switch on sign(vel) makes the solver chatter there.
      if (bumpStop > 0 && susp.bumpStopHysteresis > 0) {
        const release = 0.5 - 0.5 * clamp(vel / Math.max(1e-3, susp.bumpStopHysteresisVelocity), -1, 1);
        bumpStop *= 1 - susp.bumpStopHysteresis * release;
      }

      const compressing = vel > 0;
      const c = compressing ? w.susp.bumpDamping : w.susp.reboundDamping;
      const av = Math.abs(vel);
      // Digressive past the knee: a real off-road shock has to let the wheel
      // snap up over a square edge without spiking the chassis. The two sides
      // get their own ratio — the bump side is the one that has to stay soft
      // for ride quality, the rebound side is the one that controls the car
      // after a big hit and wants to stay near linear.
      const digressive = compressing ? susp.damperDigressiveRatio : susp.reboundDigressiveRatio;
      const raw =
        av <= susp.damperKneeVelocity
          ? c * av
          : c * susp.damperKneeVelocity + c * digressive * (av - susp.damperKneeVelocity);
      // Hard cap, and a second cap at the impulse that would exactly stop the
      // corner in one step. Without that, a big damper rate at 120 Hz can push
      // harder than the motion it is resisting and oscillate.
      const cornerMass = this.tuning.chassis.mass * 0.25;
      const maxStable = (cornerMass * av) / dt;
      const damper = Math.sign(vel) * Math.min(raw, susp.maxDamperForce, maxStable);

      let force = spring + bumpStop + damper + arb[i]!;
      if (!finite(force)) {
        warnOnce('susp-nan', 'non-finite suspension force; treating as zero');
        force = 0;
      }
      // A spring can push, never pull.
      force = clamp(force, 0, susp.maxSuspensionForce);
      w.load = force;

      if (force > 0) this.addForceAtPoint(w.normal, force, w.contact);
    }
  }

  private axleAntiRoll(idx: readonly number[], rate: number, out: number[]): void {
    if (idx.length !== 2 || rate <= 0) return;
    const a = this.wheels[idx[0]!]!;
    const b = this.wheels[idx[1]!]!;
    // A wheel in the air contributes no bar force — that is exactly what lets
    // the axle droop fully instead of the bar dragging the body over.
    if (!a.grounded || !b.grounded) return;
    const delta = a.compressionDist - b.compressionDist;
    const f = clamp(rate * delta, -this.tuning.suspension.maxSuspensionForce, this.tuning.suspension.maxSuspensionForce);
    out[idx[0]!] = f;
    out[idx[1]!] = -f;
  }

  // ---- tyres and wheel rotation ------------------------------------------

  private updateWheel(dt: number, w: WheelRuntime, driveTorque: number, brakeTorque: number): void {
    const tire = this.tuning.tire;
    const I = Math.max(0.05, tire.inertia);
    const R = tire.radius;

    if (!w.grounded) {
      // Free wheel: drive torque against bearing drag, then the brakes.
      let spin = w.spin + (dt * (driveTorque - tire.freeSpinDamping * w.spin)) / I;
      spin = this.applyBrake(spin, brakeTorque, dt, I);
      w.spin = clamp(finite(spin) ? spin : 0, -tire.maxSpin, tire.maxSpin);
      // Let the contact patch unwind rather than snapping, so the slip readout
      // does not flicker on a rough surface.
      const decay = Math.exp(-dt / 0.12);
      w.slipRatio *= decay;
      w.tanSlipAngle *= decay;
      w.load = 0;
      return;
    }

    const surface: SurfaceTuning = this.tuning.surfaces[w.surface] ?? this.tuning.surfaces.dirt;

    // Ground-plane basis at the contact patch.
    this._qSteer.setFromAxisAngle(this._up, w.steerAngle);
    this._t.copy(this._fwd).applyQuaternion(this._qSteer);
    this._t.addScaledVector(w.normal, -this._t.dot(w.normal));
    if (this._t.lengthSq() < 1e-8) this._t.copy(this._fwd);
    this._t.normalize();
    w.fwd.copy(this._t);
    w.right.copy(w.normal).cross(w.fwd).normalize();

    // Velocity of the chassis at the contact patch, in the ground plane.
    this._r.copy(w.contact).sub(this._com);
    this._v.copy(this._ang).cross(this._r).add(this._lin);
    const vx = this._v.dot(w.fwd);
    const vy = this._v.dot(w.right);
    if (!finite(vx) || !finite(vy)) {
      warnOnce('contact-vel', 'non-finite contact velocity; skipping tyre forces');
      return;
    }

    const load = w.load;

    // --- lateral slip, relaxed ------------------------------------------
    // sigma_a * d(tan a)/dt + |vx| * tan a = -vy, integrated implicitly so it
    // cannot ring at any speed. `relaxMinSpeed` floors the decay term, which
    // is what keeps a parked car's contact patch behaving like a spring
    // instead of dividing by zero.
    const vRelax = Math.max(Math.abs(vx), tire.relaxMinSpeed);
    const kA = (dt * vRelax) / Math.max(1e-3, tire.relaxLengthLat);
    let tanA = (w.tanSlipAngle + (dt * -vy) / Math.max(1e-3, tire.relaxLengthLat)) / (1 + kA);
    tanA = clamp(finite(tanA) ? tanA : 0, -tire.maxTanSlipAngle, tire.maxTanSlipAngle);
    w.tanSlipAngle = tanA;

    // --- longitudinal slip + wheel speed, solved together ----------------
    const sigma = Math.max(1e-3, tire.relaxLengthLong);
    const denom = 1 + (dt * vRelax) / sigma;
    // kappa_new = a + b * spin_new
    const a0 = (w.slipRatio + (dt * -vx) / sigma) / denom;
    const b0 = (dt * R) / sigma / denom;

    const f0 = tireForces(w.slipRatio, tanA, load, surface, tire, this._tire).fx;
    const slope = longitudinalSlope(w.slipRatio, tanA, load, surface, tire, this._tireScratch);

    // I*dw/dt = T - R*Fx(kappa_new), with Fx linearised about the current slip.
    const lhs = 1 + (dt * R * slope * b0) / I;
    let spin = (w.spin + (dt * (driveTorque - R * (f0 + slope * (a0 - w.slipRatio)))) / I) / lhs;
    if (!finite(spin)) {
      warnOnce('wheel-nan', 'wheel solve went non-finite; zeroing wheel speed');
      spin = 0;
    }

    // Rolling resistance rides with the brakes so it can never drive the wheel
    // backwards: both are pure resisting torques.
    const rollingTorque = surface.rollingResistance * load * R;
    spin = this.applyBrake(spin, brakeTorque + rollingTorque, dt, I);
    spin = clamp(spin, -tire.maxSpin, tire.maxSpin);
    w.spin = spin;

    let kappa = a0 + b0 * spin;
    kappa = clamp(finite(kappa) ? kappa : 0, -tire.maxSlipRatio, tire.maxSlipRatio);
    w.slipRatio = kappa;

    // --- final forces ----------------------------------------------------
    const t = tireForces(kappa, tanA, load, surface, tire, this._tire);
    let fx = t.fx;
    let fy = t.fy;

    // Low-speed contact-patch damping. The relaxation spring above is almost
    // undamped on its own; this is what stops a parked car buzzing. It acts on
    // slip velocity only, so it costs nothing once the car is rolling.
    //
    // Signs matter more here than anywhere else in the file. The longitudinal
    // slip *velocity* of the patch is (vx - spin*R): positive means the road is
    // running forwards under a wheel that is turning too slowly, so the tyre
    // must push the chassis BACKWARDS. Getting this the wrong way round turns
    // the damper into a motor and the parked car drives itself off.
    const speed = Math.hypot(vx, vy);
    const fade = clamp(1 - speed / Math.max(0.01, tire.lowSpeedDampingFade), 0, 1);
    if (fade > 0) {
      const cd = tire.lowSpeedDamping * fade * (load / Math.max(1, tire.nominalLoad));
      fx -= cd * (vx - spin * R);
      fy -= cd * vy;
    }

    // Everything stays inside the friction circle, damping included.
    const cap = t.capacity;
    const mag = Math.hypot(fx, fy);
    if (mag > cap && mag > 1e-6) {
      const s = cap / mag;
      fx *= s;
      fy *= s;
    }
    if (!finite(fx) || !finite(fy)) {
      warnOnce('tire-nan', 'non-finite tyre force; skipping');
      return;
    }

    this._f.copy(w.fwd).multiplyScalar(fx).addScaledVector(w.right, fy);
    this.addForceAtPoint(this._f, 1, w.contact);
  }

  /**
   * Apply a purely resisting torque without letting it reverse the wheel. If
   * one step of it would take the wheel past zero, the wheel simply stops —
   * which is how lockup (and a slip ratio of -1) emerges.
   */
  private applyBrake(spin: number, torque: number, dt: number, inertia: number): number {
    if (!(torque > 0)) return spin;
    const dw = (torque * dt) / inertia;
    if (Math.abs(spin) <= dw) return 0;
    return spin - Math.sign(spin) * dw;
  }

  // ---- aero ---------------------------------------------------------------

  private applyAero(speed: number): void {
    const aero = this.tuning.aero;
    if (speed > 0.1) {
      const q = 0.5 * aero.airDensity * aero.dragCoefficient * aero.frontalArea * speed;
      this._f.copy(this._lin).multiplyScalar(-q);
      if (finiteVec(this._f)) {
        this._forceOut.x = this._f.x;
        this._forceOut.y = this._f.y;
        this._forceOut.z = this._f.z;
        this.body.addForce(this._forceOut, true);
      }
    }
    const down = aero.downforceCoefficient * speed * speed;
    if (down > 1) {
      const half = this.wheelbase * 0.5;
      this._r.set(0, 0, half).applyQuaternion(this._q).add(this._p);
      this.addForceAtPoint(this._up, -down * aero.downforceFrontBias, this._r);
      this._r.set(0, 0, -half).applyQuaternion(this._q).add(this._p);
      this.addForceAtPoint(this._up, -down * (1 - aero.downforceFrontBias), this._r);
    }
  }

  // ---- airborne attitude --------------------------------------------------

  /**
   * While airborne, nudge the car back towards level and bleed off tumble.
   * Deliberately weak and delayed: it should stop a jump ending on the roof,
   * not fly the car for the player. Yaw is damped far less so the car can
   * still be pointed mid-air.
   */
  private applyAirborneAssist(dt: number): void {
    let grounded = false;
    for (const w of this.wheels) if (w.grounded) grounded = true;
    if (grounded) {
      this.airTime = 0;
      return;
    }
    this.airTime += dt;
    const a = this.tuning.assists;
    const fade = clamp((this.airTime - a.airAssistDelay) / Math.max(1e-3, a.airAssistFade), 0, 1);
    if (fade <= 0) return;

    const inertia = this.tuning.chassis.inertia;
    // Self-righting: the cross product of the chassis up with world up already
    // has magnitude sin(tilt) and points along the axis that fixes it.
    this._t.copy(this._up).cross(WORLD_UP);
    const iTilt = 0.5 * (inertia.x + inertia.z);
    this._t.multiplyScalar(a.airUprightGain * iTilt * fade);

    // Rate damping, in the chassis frame so pitch/roll and yaw differ.
    this._v.copy(this._ang).applyQuaternion(this._qi);
    this._v.set(
      -this._v.x * a.airPitchDamping * inertia.x,
      -this._v.y * a.airYawDamping * inertia.y,
      -this._v.z * a.airPitchDamping * inertia.z,
    );
    this._v.applyQuaternion(this._q).multiplyScalar(fade);
    this._t.add(this._v);

    const m = a.maxAssistTorque;
    this._t.set(clamp(this._t.x, -m, m), clamp(this._t.y, -m, m), clamp(this._t.z, -m, m));
    if (!finiteVec(this._t)) return;
    this._forceOut.x = this._t.x;
    this._forceOut.y = this._t.y;
    this._forceOut.z = this._t.z;
    this.body.addTorque(this._forceOut, true);
  }

  // ---- output -------------------------------------------------------------

  private addForceAtPoint(dir: THREE.Vector3, magnitude: number, point: THREE.Vector3): void {
    const fx = dir.x * magnitude;
    const fy = dir.y * magnitude;
    const fz = dir.z * magnitude;
    if (!finite(fx) || !finite(fy) || !finite(fz)) {
      warnOnce('force-nan', 'non-finite force; skipping');
      return;
    }
    this._forceOut.x = fx;
    this._forceOut.y = fy;
    this._forceOut.z = fz;
    this._pointOut.x = point.x;
    this._pointOut.y = point.y;
    this._pointOut.z = point.z;
    this.body.addForceAtPoint(this._forceOut, this._pointOut, true);
  }

  private updateWheelVisuals(sync: boolean): void {
    const susp = this.tuning.suspension;
    for (const w of this.wheels) {
      this._r.copy(w.attach).applyQuaternion(this._q).add(this.curPos);
      const len = w.grounded ? w.length : susp.restLength;
      this._tmp.set(0, -1, 0).applyQuaternion(this._q);
      w.curPos.copy(this._r).addScaledVector(this._tmp, len);
      this._qSteer.setFromAxisAngle(new THREE.Vector3(0, 1, 0), w.steerAngle);
      this._qSpin.setFromAxisAngle(new THREE.Vector3(1, 0, 0), w.spinAngle);
      w.curQuat.copy(this._q).multiply(this._qSteer).multiply(this._qSpin);
      if (sync) {
        w.prevPos.copy(w.curPos);
        w.prevQuat.copy(w.curQuat);
      }
    }
  }

  private publish(dt: number, dv: ReturnType<Drivetrain['update']>): void {
    const st = this.state;
    st.velocity.copy(this._lin);
    st.angularVelocity.copy(this._ang);
    st.engineRpm = dv.engineRpm;
    st.gear = dv.gear;
    st.clutch = dv.clutch;

    let airborne = true;
    for (let i = 0; i < this.wheels.length; i++) {
      const w = this.wheels[i]!;
      const ws = st.wheels[i]!;
      if (w.grounded) airborne = false;
      w.spinAngle = (w.spinAngle + w.spin * dt) % (Math.PI * 2);
      ws.compression = clamp(w.compressionDist / Math.max(1e-3, this.tuning.suspension.travel), 0, 1);
      ws.grounded = w.grounded;
      ws.slipRatio = w.slipRatio;
      ws.slipAngle = Math.atan(w.tanSlipAngle);
      ws.load = w.load;
      ws.surface = w.surface;
      ws.spin = w.spin;
      ws.steerAngle = w.steerAngle;
    }
    st.airborne = airborne;

    // Specific force in the chassis frame, in g — what an accelerometer bolted
    // to the dash would read, which is what camera shake and audio want.
    this._v.copy(this._lin).sub(this.prevVel).multiplyScalar(1 / dt);
    this._v.y += GRAVITY_Y;
    this._v.applyQuaternion(this._qi).multiplyScalar(1 / GRAVITY_Y);
    if (finiteVec(this._v)) {
      const k = clamp(dt * 18, 0, 1);
      st.localAccel.lerp(this._v, k);
    }
    this.prevVel.copy(this._lin);

    this.updateWheelVisuals(false);
    // Keep the published pose valid even if nobody calls getInterpolated.
    st.position.copy(this.curPos);
    st.quaternion.copy(this.curQuat);
    for (let i = 0; i < this.wheels.length; i++) {
      st.wheels[i]!.position.copy(this.wheels[i]!.curPos);
      st.wheels[i]!.quaternion.copy(this.wheels[i]!.curQuat);
    }
  }
}
