import RAPIER from '@dimforge/rapier3d-compat';
import * as THREE from 'three';
import { Vehicle } from './Vehicle';
import { JEEP_TUNING } from './VehicleTuning';
import { GROUP, interactionGroups } from '../types';
import type { DriveInput, PhysicsContext, SurfaceKind } from '../types';

await RAPIER.init();
const DT = 1 / 120;
const world = new RAPIER.World({ x: 0, y: -9.81, z: 0 });
world.timestep = DT;
world.numSolverIterations = 8;
const ctx: PhysicsContext = { world, rapier: RAPIER, dt: DT };
const cell = 2, sx = 400, sz = 400, nx = sx / cell, nz = sz / cell;
const Z0 = 30, Z1 = 40, slope = Math.tan(10 * Math.PI / 180);
const heights = new Float32Array((nx + 1) * (nz + 1));
for (let ix = 0; ix <= nx; ix++) {
  const x = (ix / nx - 0.5) * sx;
  for (let iz = 0; iz <= nz; iz++) {
    const z = (iz / nz - 0.5) * sz;
    heights[ix * (nz + 1) + iz] = Math.abs(x) <= 8 && z >= Z0 && z <= Z1 ? slope * (z - Z0) : 0;
  }
}
const gb = world.createRigidBody(RAPIER.RigidBodyDesc.fixed());
world.createCollider(
  RAPIER.ColliderDesc.heightfield(nz, nx, heights, { x: sx, y: 1, z: sz })
    .setFriction(1).setRestitution(0)
    .setCollisionGroups(interactionGroups(GROUP.TERRAIN, GROUP.CHASSIS | GROUP.WHEEL_RAY | GROUP.PROP)),
  gb,
);
world.step();

const T = JEEP_TUNING;
const v = new Vehicle(ctx, T, { position: new THREE.Vector3(0, 0.02, -50), heading: 0 });
const surf = (): SurfaceKind => 'dirt';
const f = (n: number, d = 3) => n.toFixed(d).padStart(d + 5);
const corners: THREE.Vector3[] = [];
for (const sxx of [-1, 1]) for (const sy of [-1, 1]) for (const szz of [-1, 1])
  corners.push(new THREE.Vector3(sxx * T.chassis.halfExtents.x, 0.9 + sy * T.chassis.halfExtents.y, szz * T.chassis.halfExtents.z));
const tmp = new THREE.Vector3();
const e = new THREE.Euler();
for (let i = 0; i < 120 * 12; i++) {
  const s = v.state;
  const err = 18 - s.forwardSpeed;
  const inp: DriveInput = { steer: 0, throttle: Math.max(0, Math.min(1, err * 0.5)), brake: Math.max(0, Math.min(0.3, -err * 0.2)), handbrake: 0, shiftUp: false, shiftDown: false, recover: false };
  v.fixedUpdate(DT, inp, surf);
  world.step();
  if (s.position.z > 34 && s.position.z < 80 && i % 6 === 0) {
    let lo = Infinity;
    for (const c of corners) { tmp.copy(c).applyQuaternion(s.quaternion).add(s.position); lo = Math.min(lo, tmp.y); }
    e.setFromQuaternion(s.quaternion, 'YXZ');
    console.log(
      `t=${f(i * DT, 2)} z=${f(s.position.z, 1)} y=${f(s.position.y)} vy=${f(s.velocity.y, 2)} pitch=${f(e.x * 180 / Math.PI, 2)} roll=${f(e.z * 180 / Math.PI, 2)} ` +
        `wx=${f(s.angularVelocity.x, 3)} air=${s.airborne ? 'A' : '-'} clear=${f(lo * 1000, 1)}mm comp=${s.wheels.map((w) => (w.compression * 100).toFixed(0).padStart(4)).join(',')} ` +
        `load=${s.wheels.map((w) => (w.load / 1000).toFixed(1).padStart(6)).join(',')}`,
    );
  }
}
