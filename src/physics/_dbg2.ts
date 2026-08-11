import RAPIER from '@dimforge/rapier3d-compat';
await RAPIER.init();
const ball = new RAPIER.Ball(0.4);
const dir = { x: 0, y: -1, z: 0 };
const rot = { x: 0, y: 0, z: 0, w: 1 };

function trial(label: string, build: (w: RAPIER.World) => void, x: number, z: number): void {
  const world = new RAPIER.World({ x: 0, y: -9.81, z: 0 });
  world.timestep = 1 / 120;
  build(world);
  world.step();
  let worst = 0;
  for (const y of [0.45, 0.5, 0.6, 0.696, 0.75, 0.82]) {
    const hit = world.castShape({ x, y, z }, rot, dir, ball, 0, 5, true, undefined, undefined, undefined, undefined);
    const err = hit ? Math.abs(hit.time_of_impact - (y - 0.4)) : NaN;
    worst = Math.max(worst, err);
  }
  console.log(`  ${label.padEnd(42, '.')} worst err ${(worst * 1000).toFixed(4)} mm`);
  world.free();
}

const boxAt = (half: number) => (w: RAPIER.World) => {
  const b = w.createRigidBody(RAPIER.RigidBodyDesc.fixed().setTranslation(0, -2, 0));
  w.createCollider(RAPIER.ColliderDesc.cuboid(half, 2, half), b);
};
for (const h of [500, 200, 100, 50, 25, 10, 5]) trial(`cuboid half-extent ${h} m, probe at origin`, boxAt(h), 0.8, 1.3);
for (const h of [500, 100, 50, 25]) trial(`cuboid half-extent ${h} m, probe at x=200`, boxAt(h), 200.8, 1.3);

// thin slab
const slab = (half: number, t: number) => (w: RAPIER.World) => {
  const b = w.createRigidBody(RAPIER.RigidBodyDesc.fixed().setTranslation(0, -t, 0));
  w.createCollider(RAPIER.ColliderDesc.cuboid(half, t, half), b);
};
for (const h of [500, 50]) trial(`slab half ${h} m x 0.2 m thick`, slab(h, 0.1), 0.8, 1.3);

// heightfield
const hf = (n: number, size: number) => (w: RAPIER.World) => {
  const heights = new Float32Array((n + 1) * (n + 1));
  const b = w.createRigidBody(RAPIER.RigidBodyDesc.fixed().setTranslation(0, 0, 0));
  w.createCollider(RAPIER.ColliderDesc.heightfield(n, n, heights, { x: size, y: 1, z: size }), b);
};
trial('heightfield 64x64 over 1000 m', hf(64, 1000), 0.8, 1.3);
trial('heightfield 500x500 over 1000 m', hf(500, 1000), 0.8, 1.3);
trial('heightfield 500x500 over 1000 m @x=200', hf(500, 1000), 200.8, 1.3);

// trimesh of 2 triangles
const tri = (half: number) => (w: RAPIER.World) => {
  const v = new Float32Array([-half, 0, -half, half, 0, -half, half, 0, half, -half, 0, half]);
  const i = new Uint32Array([0, 2, 1, 0, 3, 2]);
  const b = w.createRigidBody(RAPIER.RigidBodyDesc.fixed());
  w.createCollider(RAPIER.ColliderDesc.trimesh(v, i), b);
};
trial('trimesh 2 tri over 1000 m', tri(500), 0.8, 1.3);
