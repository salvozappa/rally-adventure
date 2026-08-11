/**
 * Shared contracts. Every subsystem talks through these — nothing imports
 * another subsystem's internals.
 */
import type * as THREE from 'three';
import type RAPIER from '@dimforge/rapier3d-compat';

/** Normalised driver intent, produced by Input, consumed by Vehicle. */
export interface DriveInput {
  /** -1 (full left) .. +1 (full right) */
  steer: number;
  /** 0..1 */
  throttle: number;
  /** 0..1 */
  brake: number;
  /** 0..1 — locks rear axle */
  handbrake: number;
  /** momentary: request gear change (manual mode) */
  shiftUp: boolean;
  shiftDown: boolean;
  /** momentary: flip the car back onto its wheels */
  recover: boolean;
}

/** Per-wheel state published by the vehicle for rendering, audio and FX. */
export interface WheelState {
  /** World transform of the wheel hub. */
  position: THREE.Vector3;
  quaternion: THREE.Quaternion;
  /** Suspension compression 0 (fully extended) .. 1 (bottomed out) */
  compression: number;
  /** true when the raycast found ground this step */
  grounded: boolean;
  /** Longitudinal slip ratio (negative = braking lock, positive = spin) */
  slipRatio: number;
  /** Lateral slip angle, radians */
  slipAngle: number;
  /** Magnitude of the contact force along the suspension axis, newtons */
  load: number;
  /** Surface the wheel is touching, for particles/audio */
  surface: SurfaceKind;
  /** Wheel spin speed, rad/s */
  spin: number;
  /** Steering angle applied to this wheel, radians */
  steerAngle: number;
}

export type SurfaceKind = 'dirt' | 'grass' | 'rock' | 'gravel' | 'sand' | 'mud' | 'snow';

/** Everything the rest of the game needs to know about the car each frame. */
export interface VehicleState {
  position: THREE.Vector3;
  quaternion: THREE.Quaternion;
  /** World-space linear velocity, m/s */
  velocity: THREE.Vector3;
  angularVelocity: THREE.Vector3;
  /** Forward speed along the chassis axis, m/s (signed) */
  forwardSpeed: number;
  /** Speedometer value, always positive, m/s */
  speed: number;
  engineRpm: number;
  gear: number;
  /** 0..1 clutch engagement */
  clutch: number;
  /**
   * 0..1 throttle actually being applied. Optional: consumers (exhaust FX,
   * audio) must degrade gracefully when the vehicle does not publish it.
   */
  throttle?: number;
  wheels: WheelState[];
  /** true when no wheel is grounded */
  airborne: boolean;
  /** g-force in chassis local space, for camera shake */
  localAccel: THREE.Vector3;
}

/** Terrain query interface — used by scatter, AI, camera and FX. */
export interface TerrainSampler {
  /** Height of the ground at world (x, z). */
  heightAt(x: number, z: number): number;
  /** Upward surface normal at world (x, z). */
  normalAt(x: number, z: number, out?: THREE.Vector3): THREE.Vector3;
  /** Dominant material at world (x, z). */
  surfaceAt(x: number, z: number): SurfaceKind;
  /**
   * 0..1 "this is an intended driving route". Optional: samplers that have no
   * authored routes simply omit it and consumers fall back to 0, which means
   * "scatter freely". Scatter uses it to keep the racing line clear.
   */
  routeAt?(x: number, z: number): number;
  /** Half-extent of the playable square, metres. */
  readonly halfSize: number;
}

/** Handle to the Rapier world plus the fixed timestep it runs at. */
export interface PhysicsContext {
  world: RAPIER.World;
  rapier: typeof RAPIER;
  /** seconds; the world is stepped at exactly this rate */
  readonly dt: number;
}

/** Collision groups. Rapier packs membership<<16 | filter. */
export const GROUP = {
  TERRAIN: 0x0001,
  CHASSIS: 0x0002,
  PROP: 0x0004,
  WHEEL_RAY: 0x0008,
} as const;

export function interactionGroups(memberships: number, filter: number): number {
  return ((memberships & 0xffff) << 16) | (filter & 0xffff);
}

/** A subsystem that ticks. Modules register themselves with the Game loop. */
export interface Tickable {
  /** Called at the fixed physics rate. */
  fixedUpdate?(dt: number): void;
  /** Called once per rendered frame, with interpolation alpha. */
  update?(dt: number, alpha: number): void;
}
