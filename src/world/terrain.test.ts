/**
 * Terrain verification harness. Headless — no GL context is ever created; the
 * mesh is built purely to count triangles and check the buffers for NaN.
 *
 *   npx esbuild src/world/terrain.test.ts --bundle --platform=node \
 *     --format=esm --outfile=/tmp/tt.mjs --log-level=error && node /tmp/tt.mjs
 *
 * The load-bearing test is #2. If the Rapier heightfield disagrees with
 * `heightAt`, every wheel raycast in the game lies about where the ground is
 * and the car floats or sinks. Everything else is a quality gate; that one is a
 * correctness gate.
 */

import RAPIER from '@dimforge/rapier3d-compat';
import * as THREE from 'three';
import { Terrain } from './Terrain';
import {
  BOUNDARY_INNER,
  boundaryDistance,
  generateTerrain,
  type TerrainField,
} from './heightfield';
import { mulberry32 } from './noise';
import type { PhysicsContext } from '../types';

/* -------------------------------------------------------------------------- */
/* Harness                                                                    */
/* -------------------------------------------------------------------------- */

let failures = 0;
let checks = 0;

function check(ok: boolean, label: string, detail = ''): void {
  checks++;
  if (!ok) failures++;
  const mark = ok ? 'PASS' : 'FAIL';
  console.log(`  [${mark}] ${label}${detail ? `  ${detail}` : ''}`);
}

function section(title: string): void {
  console.log(`\n${title}`);
  console.log('-'.repeat(title.length));
}

function report(label: string, value: string | number): void {
  const v = typeof value === 'number' ? fmt(value) : value;
  console.log(`  ${label.padEnd(38, '.')} ${v}`);
}

function fmt(v: number): string {
  if (!Number.isFinite(v)) return String(v);
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e6)) return v.toExponential(3);
  return v.toFixed(Math.abs(v) >= 100 ? 1 : 4).replace(/\.?0+$/, '') || '0';
}

const DEG = 180 / Math.PI;

/* -------------------------------------------------------------------------- */
/* Setup                                                                      */
/* -------------------------------------------------------------------------- */

const SEED = 20260807;
const RESOLUTION = 1024;

await RAPIER.init();
const world = new RAPIER.World({ x: 0, y: -9.81, z: 0 });
world.timestep = 1 / 120;
const ctx: PhysicsContext = { world, rapier: RAPIER, dt: 1 / 120 };

console.log('Rally terrain verification');
console.log(`seed=${SEED} resolution=${RESOLUTION}`);

const tBuild = performance.now();
const terrain = new Terrain(ctx, { seed: SEED, resolution: RESOLUTION });
const totalMs = performance.now() - tBuild;
const field = terrain.field;

// The query pipeline is only populated after a step.
world.step();

section('0. Cost');
report('generateTerrain', `${field.timings ? fmt(terrain.generationMs) : '?'} ms`);
for (const [k, v] of Object.entries(field.timings)) report(`  stage: ${k}`, `${fmt(v)} ms`);
report('mesh + collider build', `${fmt(terrain.buildMs)} ms`);
report('total constructor', `${fmt(totalMs)} ms`);
report('samples', `${field.samples} x ${field.samples}`);
report('grid spacing', `${fmt(field.spacing)} m`);
report('control map', `${field.controlSize} x ${field.controlSize}`);
report('triangles (all chunks at LOD 0)', terrain.triangleCount.toLocaleString('en-US'));
report('chunk meshes', terrain.object3d.children.length);

/* -------------------------------------------------------------------------- */
/* 1. Determinism                                                             */
/* -------------------------------------------------------------------------- */

section('1. Determinism');
{
  const res = 512;
  const a = generateTerrain({ seed: 4242, resolution: res });
  const b = generateTerrain({ seed: 4242, resolution: res });
  const c = generateTerrain({ seed: 4243, resolution: res });

  const same = (x: ArrayLike<number>, y: ArrayLike<number>): number => {
    let worst = 0;
    for (let i = 0; i < x.length; i++) worst = Math.max(worst, Math.abs(x[i]! - y[i]!));
    return worst;
  };

  const dh = same(a.heights, b.heights);
  const doc = same(a.occlusion, b.occlusion);
  const dctl = same(a.control, b.control);
  const drm = same(a.routeMask, b.routeMask);
  const diff = same(a.heights, c.heights);

  check(dh === 0, 'heights identical across two generations', `max delta ${dh}`);
  check(doc === 0, 'occlusion identical', `max delta ${doc}`);
  check(dctl === 0, 'control map identical', `max delta ${dctl}`);
  check(drm === 0, 'route mask identical', `max delta ${drm}`);
  check(diff > 1, 'a different seed makes a different world', `max delta ${fmt(diff)} m`);
}

/* -------------------------------------------------------------------------- */
/* 2. Collider agreement  (the one that matters)                              */
/* -------------------------------------------------------------------------- */

section('2. Collider agreement — 1000 downward raycasts');
{
  const rnd = mulberry32(0xc0ffee);
  const N = 1000;
  const margin = field.spacing * 2;
  const reach = field.halfSize - margin;
  const rayDir = { x: 0, y: -1, z: 0 };
  const top = field.maxHeight + 50;

  let worst = 0;
  let worstAt = { x: 0, z: 0 };
  let sum = 0;
  let misses = 0;

  for (let i = 0; i < N; i++) {
    const x = (rnd() * 2 - 1) * reach;
    const z = (rnd() * 2 - 1) * reach;
    const hit = world.castRay(
      new RAPIER.Ray({ x, y: top, z }, rayDir),
      top - field.minHeight + 100,
      true,
    );
    if (!hit) {
      misses++;
      continue;
    }
    const colliderY = top - hit.timeOfImpact;
    const sampledY = terrain.heightAt(x, z);
    const err = Math.abs(colliderY - sampledY);
    sum += err;
    if (err > worst) {
      worst = err;
      worstAt = { x, z };
    }
  }

  report('rays cast', N);
  report('rays that missed the collider', misses);
  report('mean |error|', `${fmt(sum / (N - misses) * 1000)} mm`);
  report('max |error|', `${fmt(worst * 1000)} mm`);
  report('worst point', `x=${fmt(worstAt.x)} z=${fmt(worstAt.z)}`);

  check(misses === 0, 'every ray hit the heightfield');
  check(worst < 0.005, 'collider agrees with heightAt within 5 mm', `${fmt(worst * 1000)} mm`);
}

/* -------------------------------------------------------------------------- */
/* 3. heightAt on exact sample points                                         */
/* -------------------------------------------------------------------------- */

section('3. heightAt reproduces the raw grid');
{
  const rnd = mulberry32(0x51de);
  let worst = 0;
  for (let i = 0; i < 5000; i++) {
    const ix = Math.floor(rnd() * field.samples);
    const iz = Math.floor(rnd() * field.samples);
    const x = ix * field.spacing - field.halfSize;
    const z = iz * field.spacing - field.halfSize;
    const raw = field.heights[ix * field.samples + iz]!;
    worst = Math.max(worst, Math.abs(terrain.heightAt(x, z) - raw));
  }
  report('max |heightAt - grid|', `${fmt(worst * 1000)} mm`);
  check(worst < 1e-3, 'exact sample points reproduce grid values');
}

/* -------------------------------------------------------------------------- */
/* 4. Feature audit                                                           */
/* -------------------------------------------------------------------------- */

section('4. Feature audit');

const slopeStats = measureSlopes(field);
const interior = measureSlopes(field, true);
report('height range', `${fmt(field.minHeight)} .. ${fmt(field.maxHeight)} m`);
report('relief', `${fmt(field.maxHeight - field.minHeight)} m`);
report('mean slope', `${fmt(slopeStats.mean)} deg`);
report('median slope', `${fmt(slopeStats.median)} deg`);
report('max slope', `${fmt(slopeStats.max)} deg`);
report('ground under 15 deg', `${fmt(slopeStats.under15 * 100)} %`);
report('ground under 20 deg', `${fmt(slopeStats.under20 * 100)} %`);
report('ground over 45 deg', `${fmt(slopeStats.over45 * 100)} %`);
console.log('');
report('interior height range', `${fmt(interior.minHeight)} .. ${fmt(interior.maxHeight)} m`);
report('interior relief', `${fmt(interior.relief)} m`);
report('interior mean slope', `${fmt(interior.mean)} deg`);
report('interior median slope', `${fmt(interior.median)} deg`);
report('interior over 45 deg', `${fmt(interior.over45 * 100)} %`);

check(field.maxHeight - field.minHeight > 80, 'the world has real relief', `${fmt(field.maxHeight - field.minHeight)} m`);
check(slopeStats.max > 55, 'the boundary range is genuinely impassable', `${fmt(slopeStats.max)} deg`);

// --- drivability balance ----------------------------------------------------
// These are the gates that stop the generator drifting back into an alpine
// world. The old field sat at a 49 deg median with 56 % of the ground over
// 45 deg, which is terrain the vehicle physically cannot use; there was no
// check for it, so nothing complained.
check(
  slopeStats.median >= 12 && slopeStats.median <= 18,
  'median slope is 12-18 deg — rolling, not alpine',
  `${fmt(slopeStats.median)} deg`,
);
check(slopeStats.mean < 22, 'mean slope under 22 deg', `${fmt(slopeStats.mean)} deg`);
check(
  slopeStats.under20 >= 0.55,
  'at least 55 % of the ground is comfortably drivable (<20 deg)',
  `${fmt(slopeStats.under20 * 100)} %`,
);
check(
  slopeStats.over45 < 0.12,
  'under 12 % of the ground is over 45 deg',
  `${fmt(slopeStats.over45 * 100)} %`,
);
check(
  interior.relief >= 110 && interior.relief <= 200,
  'the drivable world has 110-200 m of relief',
  `${fmt(interior.relief)} m`,
);

// --- jump crests ------------------------------------------------------------
{
  console.log('');
  let crests = 0;
  const THRESHOLD = 2.0;
  for (const k of field.layout.kickers) {
    // Prominence of the lip over the ground either side of it, measured along
    // the direction the jump is built for.
    const at = (s: number): number => terrain.heightAt(k.x + k.dx * s, k.z + k.dz * s);
    let peak = -Infinity;
    let peakS = 0;
    for (let s = -k.lipWidth; s <= k.lipWidth; s += 0.25) {
      const h = at(s);
      if (h > peak) {
        peak = h;
        peakS = s;
      }
    }
    const base = (at(-k.approach - 6) + at(k.landing + 6)) * 0.5;
    const prominence = peak - base;

    // Launch speed: the lip is a parabola, you leave the ground when its
    // curvature beats gravity, i.e. when v^2 * |d2y/ds2| > g.
    const d = 1.5;
    const curv = (at(peakS - d) - 2 * peak + at(peakS + d)) / (d * d);
    const vLaunch = curv < 0 ? Math.sqrt(9.81 / -curv) : Infinity;

    if (prominence >= THRESHOLD) crests++;
    report(
      `crest ${k.name}`,
      `+${fmt(prominence)} m  launch ~${Number.isFinite(vLaunch) ? `${fmt(vLaunch)} m/s (${fmt(vLaunch * 3.6)} km/h)` : 'n/a'}`,
    );
  }
  report(`crests above ${THRESHOLD} m`, `${crests} / ${field.layout.kickers.length}`);
  check(crests >= 4, 'at least four launchable jump crests survived generation');
}

// --- steep climb ------------------------------------------------------------
{
  console.log('');
  const spec = field.layout.corridors.find((c) => c.name === 'steep-climb')!;
  const grade = measureCorridorGrade(terrain, spec.points);
  report('steep-climb mean grade', `${fmt(grade.mean)} deg`);
  report('steep-climb median grade', `${fmt(grade.median)} deg`);
  report('steep-climb 10-90 pct', `${fmt(grade.p10)} .. ${fmt(grade.p90)} deg`);
  report('steep-climb rise / run', `${fmt(grade.rise)} m over ${fmt(grade.run)} m`);
  check(grade.mean >= 20 && grade.mean <= 30, 'steep climb averages 20-30 deg', `${fmt(grade.mean)} deg`);

  const bail = field.layout.corridors.find((c) => c.name === 'bail-out')!;
  const bg = measureCorridorGrade(terrain, bail.points);
  report('bail-out mean grade', `${fmt(bg.mean)} deg`);
  check(bg.mean < grade.mean, 'the bail-out really is the gentler line');

  // Every corridor's authored band has to be *reachable*. `applyCorridor`
  // regulates the sampled profile by clamping each step into
  // [minSlopeDeg, maxSlopeDeg]; if the terrain's two endpoints are further
  // apart than the band can span over the corridor's own length, the profile
  // physically cannot join them and the leftover rise reappears as a step
  // where the corridor feathers back into the hillside. It is a silent
  // failure — the corridor still reads as a corridor, it just has a wall at
  // one end — so it needs its own check rather than an eyeball.
  console.log('');
  for (const spec of field.layout.corridors) {
    const g = measureCorridorGrade(terrain, spec.points);
    const need = Math.atan(Math.abs(g.rise) / g.run) * DEG;
    report(
      `${spec.name} band`,
      `needs ${fmt(need)} deg, authored ${spec.minSlopeDeg}-${spec.maxSlopeDeg} deg`,
    );
    check(
      need <= spec.maxSlopeDeg + 1.5 &&
        (spec.minSlopeDeg === 0 || need >= spec.minSlopeDeg - 1.5),
      `${spec.name} asks its regulator for a grade it can deliver`,
      `${fmt(need)} deg`,
    );
  }
}

// --- drivable route ---------------------------------------------------------
{
  console.log('');
  const spawn = terrain.getSpawnPoint();
  report('spawn position', `x=${fmt(spawn.position.x)} y=${fmt(spawn.position.y)} z=${fmt(spawn.position.z)}`);
  report('spawn heading', `${fmt(spawn.heading * DEG)} deg`);
  report('spawn surface', terrain.surfaceAt(spawn.position.x, spawn.position.z));
  report('spawn route mask', terrain.routeAt(spawn.position.x, spawn.position.z));
  const sn = terrain.normalAt(spawn.position.x, spawn.position.z);
  const spawnSlope = Math.acos(Math.min(1, sn.y)) * DEG;
  report('spawn slope', `${fmt(spawnSlope)} deg`);
  check(spawnSlope < 6, 'spawn is flat enough to start on', `${fmt(spawnSlope)} deg`);
  check(terrain.routeAt(spawn.position.x, spawn.position.z) > 0.5, 'spawn is on the route');

  const s = field.layout.scale;
  const goal = { x: field.layout.washCenterX(360 * s), z: 360 * s };
  const path = findRoute(terrain, spawn.position.x, spawn.position.z, goal.x, goal.z, 30);
  report('route goal (north wash)', `x=${fmt(goal.x)} z=${fmt(goal.z)}`);
  if (path) {
    report('route found', `${path.nodes} nodes, ${fmt(path.length)} m, ${fmt(path.maxGrade)} deg worst`);
    report('route explored', `${path.visited} cells`);
  } else {
    report('route found', 'NO');
  }
  check(path !== null, 'a <=30 deg route runs from spawn up the valley');

  const climb = field.layout.corridors.find((c) => c.name === 'steep-climb')!;
  const summit = climb.points[climb.points.length - 1]!;
  const toSummit = findRoute(terrain, spawn.position.x, spawn.position.z, summit[0], summit[1], 32);
  if (toSummit) {
    report('route to climb summit', `${fmt(toSummit.length)} m, ${fmt(toSummit.maxGrade)} deg worst`);
  }
  check(toSummit !== null, 'the climb summit is reachable at <=32 deg');
}

// --- authored landform survival ---------------------------------------------
// Flattening a world is easy; flattening it without erasing the things that
// were placed in it by hand is the whole job. Each of these is a feature the
// level depends on, measured rather than assumed.
{
  console.log('');
  const h = (x: number, z: number): number => terrain.heightAt(x, z);
  const l = field.layout;

  // Bowl: a real depression, measured rim-to-floor.
  let rim = 0;
  for (let i = 0; i < 16; i++) {
    const t = (i / 16) * Math.PI * 2;
    rim += h(l.bowl.x + Math.cos(t) * l.bowl.radius, l.bowl.z + Math.sin(t) * l.bowl.radius);
  }
  const bowlDepth = rim / 16 - h(l.bowl.x, l.bowl.z);
  report('bowl depth (rim to floor)', `${fmt(bowlDepth)} m`);
  check(bowlDepth > 14, 'the banked bowl is still a bowl', `${fmt(bowlDepth)} m`);

  // Wash: a flat floor with ground above it on both sides, the whole way up.
  let worstFloor = 0;
  let leastBank = Infinity;
  let narrowest = Infinity;
  for (let z = -400; z <= 400; z += 10) {
    const cx = l.washCenterX(z);
    const floor = h(cx, z);
    let grade = 0;
    let width = 0;
    for (let dx = -60; dx <= 60; dx += 2) {
      const g = Math.atan(Math.abs(h(cx + dx + 2, z) - h(cx + dx - 2, z)) / 4) * DEG;
      if (Math.abs(dx) <= 12 && g > grade) grade = g;
      if (h(cx + dx, z) - floor < 1.6) width += 2;
    }
    const bothBanks = Math.min(h(cx - 70, z) - floor, h(cx + 70, z) - floor);
    worstFloor = Math.max(worstFloor, grade);
    leastBank = Math.min(leastBank, bothBanks);
    narrowest = Math.min(narrowest, width);
  }
  report('wash floor worst grade', `${fmt(worstFloor)} deg`);
  report('wash narrowest drivable floor', `${fmt(narrowest)} m`);
  report('wash shallowest bank at 70 m', `${fmt(leastBank)} m`);
  check(narrowest >= 12, 'the dry wash keeps a drivable floor its whole length', `${fmt(narrowest)} m`);
  check(leastBank > 2, 'the wash is still incised into the surrounding ground', `${fmt(leastBank)} m`);

  // Rocky chatter: high-frequency roughness, relative to calm ground.
  const roughness = (cx: number, cz: number): number => {
    let sum = 0;
    let n2 = 0;
    for (let dx = -60; dx <= 60; dx += 2) {
      for (let dz = -60; dz <= 60; dz += 2) {
        const x = cx + dx;
        const z = cz + dz;
        const mean = (h(x - 2, z) + h(x + 2, z) + h(x, z - 2) + h(x, z + 2)) * 0.25;
        sum += Math.abs(h(x, z) - mean);
        n2++;
      }
    }
    return sum / n2;
  };
  const rocky = roughness(l.rocky.x, l.rocky.z);
  const calm = roughness(-45 * l.scale, 120 * l.scale);
  report('rocky-section roughness', `${fmt(rocky)} m`);
  report('open-ground roughness', `${fmt(calm)} m`);
  check(rocky > calm * 2.5, 'the rocky chatter section still shakes the suspension', `${fmt(rocky / calm)}x`);

  // Playa: the calibration surface. Measured over the full-strength core, not
  // out to the rim — the rim is where it blends back into the terrain and is
  // supposed to have a grade.
  const panReach = l.pan.radius * 0.6;
  let panSum = 0;
  let panMax = 0;
  let panN = 0;
  for (let dx = -panReach; dx <= panReach; dx += 3) {
    for (let dz = -panReach; dz <= panReach; dz += 3) {
      if (dx * dx + dz * dz > panReach * panReach) continue;
      const x = l.pan.x + dx;
      const z = l.pan.z + dz;
      const g = Math.atan(Math.hypot(h(x + 1, z) - h(x - 1, z), h(x, z + 1) - h(x, z - 1)) / 2) * DEG;
      panSum += g;
      panN++;
      if (g > panMax) panMax = g;
    }
  }
  report('playa mean / max slope', `${fmt(panSum / panN)} / ${fmt(panMax)} deg`);
  check(panSum / panN < 2.5, 'the flat calibration area is flat', `${fmt(panSum / panN)} deg`);
  check(panMax < 10, 'the calibration area has no step in it', `${fmt(panMax)} deg`);

  // Off-camber shelf: the point is the *lateral* tilt across the track.
  const shelf = l.corridors.find((c) => c.name === 'off-camber-shelf')!;
  let camber = 0;
  for (const [px, pz] of shelf.points.slice(1, -1)) {
    camber = Math.max(camber, Math.atan(Math.abs(h(px + 12, pz) - h(px - 12, pz)) / 24) * DEG);
  }
  report('off-camber cross slope', `${fmt(camber)} deg`);
  check(camber > 8, 'the off-camber traverse still leans', `${fmt(camber)} deg`);
}

/* -------------------------------------------------------------------------- */
/* 5. NaN sweep                                                               */
/* -------------------------------------------------------------------------- */

section('5. Finiteness');
{
  const bad = (a: ArrayLike<number>, label: string): void => {
    let n = 0;
    for (let i = 0; i < a.length; i++) if (!Number.isFinite(a[i]!)) n++;
    check(n === 0, `${label} is finite`, n ? `${n} bad values` : `${a.length} values`);
  };
  bad(field.heights, 'heights');
  bad(field.occlusion, 'occlusion');
  bad(field.routeMask, 'routeMask');
  bad(field.control, 'control');

  // Sampler outputs.
  const rnd = mulberry32(0xbeef);
  let sampleBad = 0;
  const nrm = new THREE.Vector3();
  for (let i = 0; i < 20000; i++) {
    const x = (rnd() * 2 - 1) * field.halfSize * 1.05;
    const z = (rnd() * 2 - 1) * field.halfSize * 1.05;
    const h = terrain.heightAt(x, z);
    terrain.normalAt(x, z, nrm);
    if (!Number.isFinite(h) || !Number.isFinite(nrm.x) || !Number.isFinite(nrm.y) || !Number.isFinite(nrm.z)) {
      sampleBad++;
    }
    if (nrm.y <= 0) sampleBad++;
  }
  check(sampleBad === 0, 'heightAt/normalAt finite and up-facing outside the world too');

  // Mesh buffers.
  let meshBad = 0;
  let verts = 0;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const child of terrain.object3d.children) {
    const g = (child as THREE.Mesh).geometry;
    const pos = g.getAttribute('position') as THREE.BufferAttribute;
    const arr = pos.array as Float32Array;
    verts += pos.count;
    for (let i = 0; i < arr.length; i++) {
      if (!Number.isFinite(arr[i]!)) meshBad++;
    }
    for (let i = 1; i < arr.length; i += 3) {
      if (arr[i]! < minY) minY = arr[i]!;
      if (arr[i]! > maxY) maxY = arr[i]!;
    }
    if (!g.boundingSphere || !Number.isFinite(g.boundingSphere.radius)) meshBad++;
  }
  report('mesh vertices', verts.toLocaleString('en-US'));
  report('mesh y range (incl. skirts)', `${fmt(minY)} .. ${fmt(maxY)} m`);
  check(meshBad === 0, 'chunk geometry is finite and bounded');
}

/* -------------------------------------------------------------------------- */
/* 6. Mesh / collider surface agreement + LOD                                 */
/* -------------------------------------------------------------------------- */

section('6. Mesh surface and LOD');
{
  // Every LOD-0 grid vertex must sit exactly on the collider surface.
  const rnd = mulberry32(0x1234);
  let worst = 0;
  const children = terrain.object3d.children;
  for (let t = 0; t < 200; t++) {
    const mesh = children[Math.floor(rnd() * children.length)] as THREE.Mesh;
    const pos = mesh.geometry.getAttribute('position') as THREE.BufferAttribute;
    const i = Math.floor(rnd() * pos.count);
    const x = pos.getX(i);
    const y = pos.getY(i);
    const z = pos.getZ(i);
    const h = terrain.heightAt(x, z);
    // Skirt vertices are deliberately below the surface; skip them.
    if (y < h - 0.001) continue;
    worst = Math.max(worst, Math.abs(y - h));
  }
  report('max |vertex - heightAt|', `${fmt(worst * 1000)} mm`);
  check(worst < 1e-3, 'LOD-0 mesh vertices lie on the collider surface');

  const idxCounts = new Map<number, number>();
  const cam = new THREE.Vector3(0, 60, 0);
  terrain.update(cam);
  for (const child of children) {
    const g = (child as THREE.Mesh).geometry;
    const c = g.index!.count;
    idxCounts.set(c, (idxCounts.get(c) ?? 0) + 1);
  }
  const levels = [...idxCounts.entries()].sort((a, b) => b[0] - a[0]);
  report('LOD spread from world centre', levels.map(([c, n]) => `${n}x${c / 3}tri`).join('  '));
  let drawn = 0;
  for (const [c, n] of idxCounts) drawn += (c / 3) * n;
  report('triangles after LOD (no culling)', drawn.toLocaleString('en-US'));
  report('reduction vs all-LOD0', `${fmt((1 - drawn / terrain.triangleCount) * 100)} %`);
  check(levels.length > 1, 'LOD actually varies across the map');

  // Winding, over every triangle of every chunk at its current LOD.
  //
  // There are two populations in the index buffer and they want different
  // things, which is what made the old "> 90 % of sampled triangles point up"
  // form of this check misleading. The crack-hiding skirt is *deliberately*
  // vertical: all three of its vertices share an x (or a z), so the cross
  // product's y component is exactly zero and no skirt triangle can ever count
  // as up-facing. At LOD 2 a chunk is 512 grid triangles and 128 skirt
  // triangles, so a perfectly wound mesh scores 0.8 and fails — the check was
  // reporting the skirt's existence, not a winding fault.
  //
  // Split them by geometry instead and ask each the question that matters:
  // the surface must face up, the skirt must face out of the chunk. Both are
  // now absolute — one bad triangle fails.
  let surfaceTris = 0;
  let downFacing = 0;
  let skirtTris = 0;
  let skirtInward = 0;
  let degenerate = 0;
  const a = new THREE.Vector3();
  const b = new THREE.Vector3();
  const c = new THREE.Vector3();
  const face = new THREE.Vector3();
  const mid = new THREE.Vector3();

  for (const child of children) {
    const mesh = child as THREE.Mesh;
    const g = mesh.geometry;
    const idx = g.index!;
    const p = g.getAttribute('position') as THREE.BufferAttribute;
    const box = g.boundingBox!;
    const ccx = (box.min.x + box.max.x) * 0.5;
    const ccz = (box.min.z + box.max.z) * 0.5;

    for (let i = 0; i < idx.count; i += 3) {
      a.fromBufferAttribute(p, idx.getX(i));
      b.fromBufferAttribute(p, idx.getX(i + 1));
      c.fromBufferAttribute(p, idx.getX(i + 2));
      mid.copy(a).add(b).add(c).multiplyScalar(1 / 3);
      b.sub(a);
      c.sub(a);
      face.copy(b).cross(c);
      const len = face.length();
      if (len === 0) {
        degenerate++;
        continue;
      }
      if (Math.abs(face.y) > 1e-6 * len) {
        surfaceTris++;
        if (face.y < 0) downFacing++;
      } else {
        skirtTris++;
        // Vertical: its normal must point away from the chunk's centre,
        // otherwise back-face culling eats the crack it is there to hide.
        if (face.x * (mid.x - ccx) + face.z * (mid.z - ccz) <= 0) skirtInward++;
      }
    }
  }

  report('surface triangles checked', surfaceTris.toLocaleString('en-US'));
  report('skirt triangles checked', skirtTris.toLocaleString('en-US'));
  report('down-facing surface triangles', downFacing);
  report('inward-facing skirt triangles', skirtInward);
  check(surfaceTris > 0 && skirtTris > 0, 'both surface and skirt triangles were found');
  check(degenerate === 0, 'no degenerate triangles', `${degenerate}`);
  check(
    downFacing === 0,
    'every surface triangle winds counter-clockwise from above',
    `${downFacing} bad of ${surfaceTris}`,
  );
  check(
    skirtInward === 0,
    'every skirt triangle faces out of its chunk',
    `${skirtInward} bad of ${skirtTris}`,
  );
}

/* -------------------------------------------------------------------------- */
/* 7. Surface classification                                                  */
/* -------------------------------------------------------------------------- */

section('7. Surface mix');
{
  const counts = new Map<string, number>();
  const rnd = mulberry32(0x50f7ace);
  const N = 40000;
  for (let i = 0; i < N; i++) {
    const x = (rnd() * 2 - 1) * field.halfSize;
    const z = (rnd() * 2 - 1) * field.halfSize;
    const s = terrain.surfaceAt(x, z);
    counts.set(s, (counts.get(s) ?? 0) + 1);
  }
  for (const [k, v] of [...counts].sort((a, b) => b[1] - a[1])) {
    report(k, `${fmt((v / N) * 100)} %`);
  }
  check(counts.size >= 4, 'at least four surfaces appear in the world');

  // A wild-outdoors map should read as ground with rock in it, not as a
  // quarry. The splat rules key rock off slope, so when the terrain was too
  // steep this came out 69 % rock and 0.67 % grass and no amount of texture
  // work would have fixed it — the fix was the slope. Gate the mix so the two
  // stay coupled.
  const band: Record<string, [number, number]> = {
    dirt: [0.3, 0.4],
    grass: [0.25, 0.35],
    rock: [0.12, 0.2],
    gravel: [0.1, 0.18],
    sand: [0.05, 0.12],
  };
  for (const [layer, [lo, hi]] of Object.entries(band)) {
    const share = (counts.get(layer) ?? 0) / N;
    check(
      share >= lo && share <= hi,
      `${layer} covers ${fmt(lo * 100)}-${fmt(hi * 100)} % of the world`,
      `${fmt(share * 100)} %`,
    );
  }

  // Grip must match the picture: the wash floor near spawn should not be rock.
  const spawn = terrain.getSpawnPoint();
  const s = terrain.surfaceAt(spawn.position.x, spawn.position.z);
  check(s !== 'rock', 'spawn is not on bare rock', s);
}

/* -------------------------------------------------------------------------- */
/* Teardown                                                                   */
/* -------------------------------------------------------------------------- */

terrain.dispose();
check(world.colliders.len() === 0, 'dispose removes the collider from the world');

console.log(`\n${failures === 0 ? 'OK' : 'FAILED'} — ${checks - failures}/${checks} checks passed`);
process.exit(failures === 0 ? 0 : 1);

/* -------------------------------------------------------------------------- */
/* Measurements                                                               */
/* -------------------------------------------------------------------------- */

interface SlopeStats {
  mean: number;
  median: number;
  max: number;
  under15: number;
  under20: number;
  over45: number;
  /** Height range over the sampled region. */
  relief: number;
  minHeight: number;
  maxHeight: number;
}

/**
 * Slope distribution over the grid, optionally restricted to a region.
 *
 * `interiorOnly` excludes the boundary range. Both numbers matter and they
 * answer different questions: the whole-world figures are what the player sees
 * on the map, and the interior figures are the ones that describe the ground
 * the vehicle is expected to cover. Quoting only the first hides an unclimbable
 * world behind a tall wall's statistics; quoting only the second hides a wall
 * that has eaten half the map.
 */
function measureSlopes(f: TerrainField, interiorOnly = false): SlopeStats {
  const n = f.samples;
  const step = 2;
  const vals: number[] = [];
  const limit = BOUNDARY_INNER * f.layout.scale;
  let sum = 0;
  let max = 0;
  let under15 = 0;
  let under20 = 0;
  let over45 = 0;
  let count = 0;
  let minHeight = Infinity;
  let maxHeight = -Infinity;

  for (let ix = 1; ix < n - 1; ix += step) {
    const x = ix * f.spacing - f.halfSize;
    for (let iz = 1; iz < n - 1; iz += step) {
      const z = iz * f.spacing - f.halfSize;
      if (interiorOnly && boundaryDistance(x, z) >= limit) continue;
      const dhx = (f.heights[(ix + 1) * n + iz]! - f.heights[(ix - 1) * n + iz]!) / (2 * f.spacing);
      const dhz = (f.heights[ix * n + iz + 1]! - f.heights[ix * n + iz - 1]!) / (2 * f.spacing);
      const deg = Math.atan(Math.hypot(dhx, dhz)) * DEG;
      sum += deg;
      count++;
      if (deg > max) max = deg;
      if (deg < 15) under15++;
      if (deg < 20) under20++;
      if (deg > 45) over45++;
      const h = f.heights[ix * n + iz]!;
      if (h < minHeight) minHeight = h;
      if (h > maxHeight) maxHeight = h;
      if ((ix * n + iz) % 7 === 0) vals.push(deg);
    }
  }
  vals.sort((a, b) => a - b);
  return {
    mean: sum / count,
    median: vals[Math.floor(vals.length / 2)] ?? 0,
    max,
    under15: under15 / count,
    under20: under20 / count,
    over45: over45 / count,
    relief: maxHeight - minHeight,
    minHeight,
    maxHeight,
  };
}

/** Along-track gradient of a corridor centreline, in degrees. */
function measureCorridorGrade(
  t: Terrain,
  points: ReadonlyArray<readonly [number, number]>,
): { mean: number; median: number; p10: number; p90: number; rise: number; run: number } {
  const step = 2;
  const grades: number[] = [];
  let run = 0;
  let first = 0;
  let last = 0;
  let got = false;

  for (let i = 0; i < points.length - 1; i++) {
    const [ax, az] = points[i]!;
    const [bx, bz] = points[i + 1]!;
    const len = Math.hypot(bx - ax, bz - az);
    const n = Math.max(1, Math.round(len / step));
    for (let k = 0; k < n; k++) {
      const t0 = k / n;
      const t1 = (k + 1) / n;
      const h0 = t.heightAt(ax + (bx - ax) * t0, az + (bz - az) * t0);
      const h1 = t.heightAt(ax + (bx - ax) * t1, az + (bz - az) * t1);
      const ds = len / n;
      grades.push(Math.atan(Math.abs(h1 - h0) / ds) * DEG);
      if (!got) {
        first = h0;
        got = true;
      }
      last = h1;
      run += ds;
    }
  }
  grades.sort((a, b) => a - b);
  const mean = grades.reduce((s, v) => s + v, 0) / grades.length;
  return {
    mean,
    median: grades[Math.floor(grades.length / 2)]!,
    p10: grades[Math.floor(grades.length * 0.1)]!,
    p90: grades[Math.floor(grades.length * 0.9)]!,
    rise: last - first,
    run,
  };
}

/**
 * A* over a coarse lattice, where a move is legal only if the grade between the
 * two cells stays under `maxDeg`. This is the question that actually matters:
 * not "is there a path" in the topological sense, but "can a vehicle that
 * cannot climb past 30 degrees get from the start to the far end of the map".
 */
function findRoute(
  t: Terrain,
  sx: number,
  sz: number,
  gx: number,
  gz: number,
  maxDeg: number,
): { nodes: number; length: number; maxGrade: number; visited: number } | null {
  const cell = 6;
  const half = t.halfSize;
  const dim = Math.floor((half * 2) / cell);
  const toI = (x: number): number => Math.round((x + half) / cell);
  const toW = (i: number): number => i * cell - half;
  const key = (a: number, b: number): number => a * dim + b;
  const maxRise = Math.tan((maxDeg * Math.PI) / 180) * cell;

  const start = key(toI(sx), toI(sz));
  const goal = key(toI(gx), toI(gz));

  const heights = new Float32Array(dim * dim);
  for (let i = 0; i < dim; i++) {
    for (let j = 0; j < dim; j++) heights[key(i, j)] = t.heightAt(toW(i), toW(j));
  }

  const gScore = new Float64Array(dim * dim).fill(Infinity);
  const came = new Int32Array(dim * dim).fill(-1);
  const closed = new Uint8Array(dim * dim);
  gScore[start] = 0;

  // Small binary heap; the lattice is ~170^2 so this stays comfortably fast.
  const heap: Array<[number, number]> = [[hEuclid(start), start]];
  const push = (f: number, node: number): void => {
    heap.push([f, node]);
    let i = heap.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (heap[p]![0] <= heap[i]![0]) break;
      [heap[p], heap[i]] = [heap[i]!, heap[p]!];
      i = p;
    }
  };
  const pop = (): [number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length) {
      heap[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1;
        const r = l + 1;
        let m = i;
        if (l < heap.length && heap[l]![0] < heap[m]![0]) m = l;
        if (r < heap.length && heap[r]![0] < heap[m]![0]) m = r;
        if (m === i) break;
        [heap[m], heap[i]] = [heap[i]!, heap[m]!];
        i = m;
      }
    }
    return top;
  };

  function hEuclid(node: number): number {
    const a = Math.floor(node / dim);
    const b = node % dim;
    const ga = Math.floor(goal / dim);
    const gb = goal % dim;
    return Math.hypot(a - ga, b - gb) * cell;
  }

  let visited = 0;
  while (heap.length) {
    const [, node] = pop();
    if (closed[node]) continue;
    closed[node] = 1;
    visited++;
    if (node === goal) break;

    const a = Math.floor(node / dim);
    const b = node % dim;
    const h0 = heights[node]!;

    for (let da = -1; da <= 1; da++) {
      for (let db = -1; db <= 1; db++) {
        if (da === 0 && db === 0) continue;
        const na = a + da;
        const nb = b + db;
        if (na < 1 || nb < 1 || na >= dim - 1 || nb >= dim - 1) continue;
        const nk = key(na, nb);
        if (closed[nk]) continue;
        const dist = Math.hypot(da, db) * cell;
        const rise = Math.abs(heights[nk]! - h0);
        if (rise > maxRise * (dist / cell)) continue;
        const g = gScore[node]! + dist + rise * 0.5;
        if (g < gScore[nk]!) {
          gScore[nk] = g;
          came[nk] = node;
          push(g + hEuclid(nk), nk);
        }
      }
    }
  }

  if (!closed[goal] || came[goal] === -1) return null;

  let node = goal;
  let nodes = 0;
  let length = 0;
  let maxGrade = 0;
  while (node !== start && came[node] !== -1) {
    const prev = came[node]!;
    const a = Math.floor(node / dim);
    const b = node % dim;
    const pa = Math.floor(prev / dim);
    const pb = prev % dim;
    const d = Math.hypot(a - pa, b - pb) * cell;
    const rise = Math.abs(heights[node]! - heights[prev]!);
    length += Math.hypot(d, rise);
    maxGrade = Math.max(maxGrade, Math.atan(rise / d) * DEG);
    nodes++;
    node = prev;
  }
  return { nodes, length, maxGrade, visited };
}
