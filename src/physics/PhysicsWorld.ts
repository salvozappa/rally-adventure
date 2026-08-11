import RAPIER from '@dimforge/rapier3d-compat';
import type { PhysicsContext } from '../types';

/** Fixed physics rate. High enough that a raycast suspension stays stable. */
export const PHYSICS_HZ = 120;
export const PHYSICS_DT = 1 / PHYSICS_HZ;

export async function createPhysics(gravity = -9.81): Promise<PhysicsContext> {
  await RAPIER.init();
  const world = new RAPIER.World({ x: 0, y: gravity, z: 0 });
  world.timestep = PHYSICS_DT;
  // A few extra solver iterations: the chassis rides on eight simultaneous
  // impulse constraints once you count suspension and tire forces, and the
  // default two passes let it sag under hard cornering.
  world.numSolverIterations = 8;
  return { world, rapier: RAPIER, dt: PHYSICS_DT };
}

/**
 * Fixed-timestep accumulator. Returns the interpolation alpha so renderers can
 * blend between the last two physics states instead of showing 120Hz jitter at
 * a 60Hz refresh.
 */
export class FixedStepper {
  private accumulator = 0;
  /** Guard against the spiral of death after a tab-switch stall. */
  private readonly maxSteps = 5;

  constructor(private readonly dt: number = PHYSICS_DT) {}

  /** Calls `step` zero or more times, then returns alpha in [0, 1). */
  advance(frameTime: number, step: (dt: number) => void): number {
    this.accumulator += Math.min(frameTime, this.dt * this.maxSteps);
    let n = 0;
    while (this.accumulator >= this.dt && n < this.maxSteps) {
      step(this.dt);
      this.accumulator -= this.dt;
      n++;
    }
    return this.accumulator / this.dt;
  }
}
