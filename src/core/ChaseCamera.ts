import * as THREE from 'three';
import type { VehicleState, TerrainSampler } from '../types';

export type CameraMode = 'chase' | 'chaseFar' | 'hood' | 'bumper' | 'orbit';

export interface ChaseCameraOptions {
  /** Metres behind the car at rest. */
  distance: number;
  /** Metres above the car's origin. */
  height: number;
  /** Metres above the car origin that the camera aims at. */
  lookHeight: number;
  /** Base vertical FOV, degrees. */
  fov: number;
  /** Extra FOV at top speed, degrees. */
  fovSpeedGain: number;
  /** Speed at which fovSpeedGain is fully applied, m/s. */
  fovSpeedRef: number;
  /** Position spring stiffness. Higher = tighter follow. */
  stiffness: number;
  /** Position spring damping ratio. 1 = critically damped. */
  damping: number;
  /** How much the camera swings toward the velocity vector vs the car heading, 0..1. */
  velocityBias: number;
}

/** Half a turn about the vertical: chassis forward (+Z) -> camera forward (-Z). */
const FLIP = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.PI);

const DEFAULTS: ChaseCameraOptions = {
  distance: 6.4,
  height: 2.45,
  lookHeight: 1.05,
  fov: 62,
  fovSpeedGain: 16,
  fovSpeedRef: 38,
  stiffness: 42,
  damping: 1.0,
  velocityBias: 0.42,
};

/**
 * Spring-damped chase camera.
 *
 * The important behaviours, in rough order of how much they matter to feel:
 *  - the camera trails a *blend* of the car's heading and its velocity vector,
 *    so a drift shows you the side of the car instead of pinning you behind it;
 *  - it stays world-upright and only borrows a fraction of the chassis roll,
 *    because inheriting full roll makes players motion-sick within a minute;
 *  - FOV widens with speed, which reads as acceleration far more strongly than
 *    the actual velocity change does;
 *  - it refuses to sink below the terrain, so cresting a hill never buries the
 *    view in dirt.
 */
export class ChaseCamera {
  readonly options: ChaseCameraOptions;
  mode: CameraMode = 'chase';

  /** Smoothed camera position and its velocity, for the spring. */
  private pos = new THREE.Vector3();
  private vel = new THREE.Vector3();
  private lookAt = new THREE.Vector3();
  /** Smoothed follow direction (unit, world XZ). */
  private followDir = new THREE.Vector3(0, 0, 1);
  private currentFov: number;
  private roll = 0;
  private shake = 0;
  private shakeTime = 0;
  private orbitAngle = 0;
  private initialised = false;

  private tmpA = new THREE.Vector3();
  private tmpB = new THREE.Vector3();

  constructor(
    private camera: THREE.PerspectiveCamera,
    private terrain: TerrainSampler | null = null,
    opts: Partial<ChaseCameraOptions> = {},
  ) {
    this.options = { ...DEFAULTS, ...opts };
    this.currentFov = this.options.fov;
  }

  setTerrain(t: TerrainSampler): void {
    this.terrain = t;
  }

  cycleMode(): CameraMode {
    const order: CameraMode[] = ['chase', 'chaseFar', 'hood', 'bumper', 'orbit'];
    this.mode = order[(order.indexOf(this.mode) + 1) % order.length];
    this.initialised = false;
    return this.mode;
  }

  /** Kick the camera — call on landings and collisions. `g` is impact strength. */
  addShake(g: number): void {
    this.shake = Math.min(1.2, this.shake + g * 0.16);
  }

  /** Snap to the target with no smoothing. Use on spawn/respawn. */
  reset(): void {
    this.initialised = false;
  }

  update(dt: number, s: VehicleState): void {
    const o = this.options;
    // Guard against the huge dt of a tab-switch resume, which would otherwise
    // make the spring overshoot wildly.
    dt = Math.min(dt, 0.05);

    const speed = s.speed;

    // --- interior cameras are rigidly attached, so handle them separately ---
    if (this.mode === 'hood' || this.mode === 'bumper') {
      const local = this.mode === 'hood'
        ? this.tmpA.set(0, 1.32, 0.15)
        : this.tmpA.set(0, 0.62, 1.85);
      const world = local.applyQuaternion(s.quaternion).add(s.position);
      this.camera.position.copy(world);
      // The chassis forward axis is +Z (see VehicleTuning's header and
      // Vehicle.readBody), but a Three camera looks down its own -Z. So a
      // cockpit camera rigidly bolted to the car is the chassis rotation turned
      // through half a turn about the vertical — without this the hood camera
      // stares out of the back window.
      this.camera.quaternion.copy(s.quaternion).multiply(FLIP);
      this.applyFov(dt, speed, this.mode === 'bumper' ? 8 : 4);
      this.applyShake(dt, 0.35);
      return;
    }

    // --- car heading and velocity, flattened to the horizontal plane ---
    // Forward is chassis-local +Z, the convention the whole vehicle subsystem
    // uses (VehicleTuning's header; Vehicle.readBody builds its forward vector
    // the same way). `followDir` is therefore the direction the car is FACING,
    // and the camera is placed a distance back along it.
    const heading = this.tmpA.set(0, 0, 1).applyQuaternion(s.quaternion);
    heading.y = 0;
    if (heading.lengthSq() < 1e-6) heading.set(0, 0, 1);
    heading.normalize();

    const velDir = this.tmpB.copy(s.velocity);
    velDir.y = 0;
    const planarSpeed = velDir.length();

    // Blend heading with travel direction. Only trust the velocity vector once
    // we're actually moving, and only when going forwards — reversing should
    // not whip the camera around to the front of the car.
    let desiredDir = heading;
    if (planarSpeed > 2.5 && s.forwardSpeed > 0) {
      velDir.divideScalar(planarSpeed);
      const bias = o.velocityBias * THREE.MathUtils.smoothstep(planarSpeed, 2.5, 12);
      desiredDir = velDir.lerp(heading, 1 - bias).normalize();
    }

    if (!this.initialised) {
      this.followDir.copy(desiredDir);
    } else {
      // Rotate the follow direction toward the target at a rate that rises with
      // speed: slow, lazy swings when crawling, quick when charging.
      const turnRate = THREE.MathUtils.lerp(3.2, 7.5, THREE.MathUtils.clamp(speed / 30, 0, 1));
      this.followDir.lerp(desiredDir, 1 - Math.exp(-turnRate * dt)).normalize();
    }

    // --- desired camera placement ---
    const far = this.mode === 'chaseFar';
    const dist = far ? o.distance * 1.75 : o.distance;
    const hgt = far ? o.height * 1.5 : o.height;

    // Pull back a little more the faster we go — subtle, but it stops the car
    // filling the frame exactly when you most need to see ahead.
    const speedPull = THREE.MathUtils.clamp(speed / o.fovSpeedRef, 0, 1) * 1.1;

    const target = this.tmpA;
    if (this.mode === 'orbit') {
      this.orbitAngle += dt * 0.35;
      target.set(
        s.position.x + Math.cos(this.orbitAngle) * dist * 1.4,
        s.position.y + hgt * 1.3,
        s.position.z + Math.sin(this.orbitAngle) * dist * 1.4,
      );
    } else {
      target.set(
        s.position.x - this.followDir.x * (dist + speedPull),
        s.position.y + hgt,
        s.position.z - this.followDir.z * (dist + speedPull),
      );
    }

    if (!this.initialised) {
      this.pos.copy(target);
      this.vel.set(0, 0, 0);
      this.initialised = true;
    } else {
      // Critically-damped-ish spring, integrated semi-implicitly so it stays
      // stable at large dt.
      const k = o.stiffness;
      const c = 2 * o.damping * Math.sqrt(k);
      const ax = (target.x - this.pos.x) * k - this.vel.x * c;
      const ay = (target.y - this.pos.y) * k - this.vel.y * c;
      const az = (target.z - this.pos.z) * k - this.vel.z * c;
      this.vel.x += ax * dt;
      this.vel.y += ay * dt;
      this.vel.z += az * dt;
      this.pos.x += this.vel.x * dt;
      this.pos.y += this.vel.y * dt;
      this.pos.z += this.vel.z * dt;
    }

    // --- never let the camera go underground ---
    if (this.terrain) {
      const groundY = this.terrain.heightAt(this.pos.x, this.pos.z) + 0.85;
      if (this.pos.y < groundY) {
        this.pos.y = groundY;
        if (this.vel.y < 0) this.vel.y = 0;
      }
    }

    this.camera.position.copy(this.pos);

    // --- aim ---
    // Look slightly ahead of the car rather than at it. At speed this shifts
    // the composition so you see more road and less bonnet.
    const lead = THREE.MathUtils.clamp(speed * 0.16, 0, 5.5);
    const desiredLook = this.tmpB.set(
      s.position.x + this.followDir.x * lead,
      s.position.y + o.lookHeight,
      s.position.z + this.followDir.z * lead,
    );
    this.lookAt.lerp(desiredLook, 1 - Math.exp(-9 * dt));
    this.camera.lookAt(this.lookAt);

    // --- borrow a fraction of chassis roll, for drama without nausea ---
    // Chassis right is -X, not +X: with forward at +Z and up at +Y, forward
    // cross up lands on -X. It is also the direction that points to the right of
    // the screen for a camera trailing the car, which is what makes its world-Y
    // component the amount the horizon should tip.
    const carRight = this.tmpA.set(-1, 0, 0).applyQuaternion(s.quaternion);
    const targetRoll = Math.asin(THREE.MathUtils.clamp(carRight.y, -1, 1)) * 0.22;
    this.roll += (targetRoll - this.roll) * (1 - Math.exp(-5 * dt));
    if (Math.abs(this.roll) > 1e-4) {
      // Roll about the camera's own forward axis, so the horizon tips without
      // the aim point moving.
      this.camera.rotateZ(this.roll);
    }

    this.applyFov(dt, speed, 0);
    this.applyShake(dt, 1);
  }

  private applyFov(dt: number, speed: number, extra: number): void {
    const o = this.options;
    const t = THREE.MathUtils.clamp(speed / o.fovSpeedRef, 0, 1.35);
    // Ease the gain so the last few km/h still register visually.
    const target = o.fov + extra + o.fovSpeedGain * (t * t * (3 - 2 * Math.min(t, 1)));
    this.currentFov += (target - this.currentFov) * (1 - Math.exp(-4.5 * dt));
    if (Math.abs(this.camera.fov - this.currentFov) > 0.01) {
      this.camera.fov = this.currentFov;
      this.camera.updateProjectionMatrix();
    }
  }

  private applyShake(dt: number, scale: number): void {
    if (this.shake <= 0.0001) return;
    this.shakeTime += dt;
    const a = this.shake * scale;
    // Two incommensurate frequencies read as "impact" rather than "vibration".
    const t = this.shakeTime;
    const ox = (Math.sin(t * 47.3) * 0.6 + Math.sin(t * 91.7) * 0.4) * a * 0.09;
    const oy = (Math.sin(t * 53.1) * 0.6 + Math.sin(t * 79.3) * 0.4) * a * 0.11;
    this.camera.position.x += ox;
    this.camera.position.y += oy;
    this.camera.rotateZ(ox * 0.35);
    this.shake = Math.max(0, this.shake - dt * 2.6);
  }
}
