/**
 * Terrain synthesis — pure height math, no Three.js, no Rapier.
 *
 * The output of `generateTerrain` is the single source of truth for the whole
 * subsystem: the visual mesh, the Rapier collider, `heightAt`, `normalAt` and
 * `surfaceAt` all read from these arrays. Nothing is recomputed analytically
 * afterwards, so there is no way for the render and physics representations to
 * drift apart.
 *
 * ── Grid layout ────────────────────────────────────────────────────────────
 * The sample layout is dictated by Rapier's heightfield collider, which is
 * what we have to match. Verified empirically against rapier3d-compat 0.19:
 *
 *   index(ix, iz) = ix * (cells + 1) + iz
 *   x = (ix / cells - 0.5) * size
 *   z = (iz / cells - 0.5) * size
 *
 * and each cell is split into two triangles along the **anti-diagonal**, the
 * one joining (ix+1, iz) to (ix, iz+1). `sampleHeight` below reproduces that
 * split exactly, which is why terrain queries agree with the collider to
 * floating-point precision rather than to "within a few centimetres".
 *
 * ── Generation order ───────────────────────────────────────────────────────
 * The order matters and is deliberate:
 *   1. macro landform + fractal detail (domain warped)
 *   2. erosion (hydraulic droplets, then thermal talus)   ← natural terrain
 *   2b. boundary range                                    ← after erosion, so
 *      the talus pass cannot open a ramp through the wall
 *   3. valley carve + playa flattening                    ← authored landform
 *   4. route corridors (gradient-regulated, optionally off-camber)
 *   5. drivability clamp over the route mask
 *   6. jump crests                                        ← after the clamp,
 *      so their lips keep the curvature that gets you airborne
 *   7. rock chatter
 *   8. bake normals, AO and the splat control map
 *
 * ── Slope budget ───────────────────────────────────────────────────────────
 * A vehicle can only use ground it can climb, so the generator is tuned to a
 * distribution, not just to a look: median slope 12-18 deg, over half the map
 * under 20 deg, and under 12 % of it over 45 deg, with 110-200 m of relief
 * inside the boundary range. The two places that budget is easiest to blow are
 * the fBm gain (see the landform stage) and the width of the boundary range
 * (see `addBoundaryRange`); both are commented where they are set, and
 * `terrain.test.ts` asserts the resulting distribution.
 */

import {
  Noise2D,
  clamp,
  domainWarp,
  hash2i,
  lerp,
  mulberry32,
  segmentProject,
  smootherstep,
  smoothstep,
} from './noise';
import type { SurfaceKind } from '../types';

/* -------------------------------------------------------------------------- */
/* Surfaces                                                                   */
/* -------------------------------------------------------------------------- */

/**
 * Splat layer order. The control map stores weights for layers 1..4 in RGBA;
 * layer 0 (dirt) is the implicit remainder `1 - (r+g+b+a)`. The shader and
 * `surfaceAt` both use this ordering, so what the tyre grips is what you see.
 */
export const SURFACE_LAYERS = ['dirt', 'grass', 'rock', 'gravel', 'sand'] as const;
export type TerrainSurface = (typeof SURFACE_LAYERS)[number];
export const LAYER_COUNT = SURFACE_LAYERS.length;

/** Narrowing helper: every entry of SURFACE_LAYERS is a valid SurfaceKind. */
const SURFACE_KINDS: SurfaceKind[] = ['dirt', 'grass', 'rock', 'gravel', 'sand'];

/* -------------------------------------------------------------------------- */
/* Authored layout                                                            */
/* -------------------------------------------------------------------------- */

/**
 * Feature positions, authored for a 1024 m world. `generateTerrain` scales
 * them if you ask for a different size, so the level composition survives a
 * change of `size` instead of turning into noise.
 */
export const DESIGN_SIZE = 1024;

export interface KickerSpec {
  readonly name: string;
  /** Lip position. */
  readonly x: number;
  readonly z: number;
  /** Travel direction the jump is built for (unit vector, world XZ). */
  readonly dx: number;
  readonly dz: number;
  /** Height of the broad rise, metres. */
  readonly height: number;
  /** Height of the sharp convex lip added on top, metres. */
  readonly lip: number;
  /** Approach ramp length, metres. */
  readonly approach: number;
  /** Landing ramp length, metres. */
  readonly landing: number;
  /** Half-width of the lip cap, metres — controls how sharp the launch is. */
  readonly lipWidth: number;
  /** Lateral extent of the crest, metres. */
  readonly span: number;
}

export interface CorridorSpec {
  readonly name: string;
  readonly points: ReadonlyArray<readonly [number, number]>;
  /** Full-strength half width, metres. */
  readonly halfWidth: number;
  /** Feather distance beyond `halfWidth` where it eases back to terrain. */
  readonly shoulder: number;
  /** Lateral tilt in degrees; positive drops the left-hand side of travel. */
  readonly tiltDeg: number;
  /** Along-track gradient is clamped into this band (degrees). */
  readonly minSlopeDeg: number;
  readonly maxSlopeDeg: number;
  /** Max blend toward the regulated profile, 0..1. */
  readonly strength: number;
}

/** Centreline of the dry wash as a function of z. Gentle, drivable meander. */
export function washCenterX(z: number): number {
  return 46 * Math.sin(z / 175) + 78 * Math.sin(z / 520 + 0.9) - 20;
}

/** Elevation of the wash floor. It drains to the south, like a real wash. */
export function washFloorY(z: number): number {
  return 12 + (z + 512) * 0.028 + 2.2 * Math.sin(z / 130);
}

export const FEATURES = {
  /** Flat playa by the south end of the wash — handling calibration. */
  pan: { x: -45, z: -380, radius: 96 },

  /** Natural amphitheatre for lateral-grip testing. */
  bowl: { x: 215, z: -70, radius: 96, depth: 25 },

  /** Broken ground that shakes the suspension. */
  rocky: { x: 95, z: 325, rx: 145, rz: 125 },

  /** Big landform masses, added under the fractal detail. */
  massifs: [
    { x: -300, z: 215, radius: 235, height: 74 }, // west range
    { x: -140, z: 215, radius: 152, height: 40 }, // the hill the climb goes up
    { x: 300, z: 210, radius: 210, height: 68 }, // east shoulder
    { x: -60, z: 430, radius: 190, height: 52 }, // north headwall
    { x: 330, z: -320, radius: 175, height: 44 }, // south-east knoll
  ],

  kickers: [
    { name: 'wash-1', x: washCenterX(-250), z: -250, dx: 0, dz: 1, height: 3.2, lip: 1.0, approach: 30, landing: 40, lipWidth: 7.0, span: 58 },
    { name: 'wash-2', x: washCenterX(-70), z: -70, dx: 0, dz: 1, height: 4.6, lip: 1.4, approach: 32, landing: 44, lipWidth: 7.5, span: 62 },
    { name: 'wash-3', x: washCenterX(115), z: 115, dx: 0, dz: 1, height: 5.6, lip: 1.7, approach: 36, landing: 52, lipWidth: 8.0, span: 66 },
    { name: 'wash-4', x: washCenterX(300), z: 300, dx: 0, dz: 1, height: 4.0, lip: 1.25, approach: 28, landing: 38, lipWidth: 6.5, span: 54 },
    { name: 'bowl-approach', x: 78, z: -72, dx: 1, dz: 0, height: 4.0, lip: 1.3, approach: 28, landing: 40, lipWidth: 7.0, span: 46 },
  ] as const satisfies readonly KickerSpec[],

  corridors: [
    {
      name: 'steep-climb',
      points: [[30, 140], [-90, 190], [-150, 225]],
      halfWidth: 13,
      shoulder: 20,
      tiltDeg: 0,
      // A corridor's band has to be able to *reach* the far end. The regulator
      // clamps every step into [min, max], so if the terrain's two endpoints
      // are further apart than `maxSlopeDeg` over the corridor's length the
      // profile simply cannot get there, and the leftover rise reappears as a
      // step where the corridor fades back into the hillside — a 50 degree
      // wall at the bottom of what is supposed to be the 26 degree climb.
      // `terrain.test.ts` now asserts every band is reachable; keep them so.
      minSlopeDeg: 21,
      maxSlopeDeg: 28,
      strength: 1,
    },
    {
      name: 'bail-out',
      points: [[30, 140], [-40, 60], [-160, 95], [-205, 190], [-152, 230]],
      halfWidth: 11,
      shoulder: 18,
      tiltDeg: 0,
      minSlopeDeg: 0,
      maxSlopeDeg: 15,
      strength: 0.95,
    },
    {
      name: 'off-camber-shelf',
      points: [[75, 45], [185, 110], [255, 200], [290, 300]],
      halfWidth: 17,
      shoulder: 22,
      tiltDeg: 16,
      minSlopeDeg: 0,
      maxSlopeDeg: 12,
      strength: 0.95,
    },
    {
      name: 'bowl-approach',
      points: [[26, -76], [90, -74], [135, -70]],
      halfWidth: 16,
      shoulder: 20,
      tiltDeg: 0,
      minSlopeDeg: 0,
      maxSlopeDeg: 17,
      strength: 0.85,
    },
  ] as const satisfies readonly CorridorSpec[],
} as const;

/* -------------------------------------------------------------------------- */
/* Public result                                                              */
/* -------------------------------------------------------------------------- */

export interface TerrainFieldOptions {
  seed?: number;
  /** Side length of the square world, metres. */
  size?: number;
  /** Number of cells per side. Sample count is (resolution + 1)^2. */
  resolution?: number;
  /** Skip erosion. Only used by tooling that wants a fast preview. */
  erosion?: boolean;
}

export interface TerrainField {
  readonly seed: number;
  readonly size: number;
  readonly halfSize: number;
  /** Cells per side. */
  readonly cells: number;
  /** Samples per side = cells + 1. */
  readonly samples: number;
  /** Metres between samples. */
  readonly spacing: number;
  /** Heights, indexed `ix * samples + iz`, metres. */
  readonly heights: Float32Array;
  /** Baked ambient occlusion, 0 (dark) .. 1 (open), same indexing. */
  readonly occlusion: Float32Array;
  /**
   * Splat control map, `controlSize^2` RGBA8 texels covering the whole world.
   * R=grass G=rock B=gravel A=sand; dirt is the remainder.
   */
  readonly control: Uint8Array;
  readonly controlSize: number;
  /** 0..1 mask of "this is an intended driving route". */
  readonly routeMask: Float32Array;
  readonly minHeight: number;
  readonly maxHeight: number;
  /** Scaled copies of the authored layout, in world units. */
  readonly layout: ScaledLayout;
  /** Wall-clock generation cost, milliseconds, per stage. */
  readonly timings: Record<string, number>;
}

export interface ScaledLayout {
  scale: number;
  pan: { x: number; z: number; radius: number };
  bowl: { x: number; z: number; radius: number; depth: number };
  rocky: { x: number; z: number; rx: number; rz: number };
  kickers: KickerSpec[];
  corridors: CorridorSpec[];
  washCenterX(z: number): number;
  washFloorY(z: number): number;
}

/* -------------------------------------------------------------------------- */
/* Grid helpers                                                               */
/* -------------------------------------------------------------------------- */

/**
 * Bilinear read of a grid stored as `ix * n + iz`, in grid coordinates.
 *
 * The clamp is to `n - 1` exactly, not `n - 1 - eps`. An epsilon here looks
 * harmless but it means the very last row and column are never read at their
 * own coordinate: a query at `fx = n - 1` lands at `n - 1 - eps` and comes back
 * interpolated, off by `eps` times the local height difference. That is the
 * millimetre-scale error the sampler tests catch. Floor/next-index selection
 * below already handles `x === n - 1` by collapsing the far tap onto the near
 * one, so no epsilon is needed to stay in bounds.
 */
function gridBilinear(data: Float32Array, n: number, fx: number, fz: number): number {
  const x = clamp(fx, 0, n - 1);
  const z = clamp(fz, 0, n - 1);
  const ix = Math.floor(x);
  const iz = Math.floor(z);
  const tx = x - ix;
  const tz = z - iz;
  const i0 = ix * n + iz;
  const i1 = (ix + 1 < n ? ix + 1 : ix) * n + iz;
  const h00 = data[i0]!;
  const h01 = data[i0 + (iz + 1 < n ? 1 : 0)]!;
  const h10 = data[i1]!;
  const h11 = data[i1 + (iz + 1 < n ? 1 : 0)]!;
  return lerp(lerp(h00, h10, tx), lerp(h01, h11, tx), tz);
}

/**
 * Height at fractional grid coordinates using **the same triangulation Rapier
 * uses**: each cell is split along the anti-diagonal joining (ix+1, iz) to
 * (ix, iz+1). Plain bilinear interpolation would disagree with the collider by
 * up to a quarter of the cell's height variation — decimetres on this terrain,
 * not millimetres — and the car would visibly float or sink.
 *
 * As in `gridBilinear`, the clamp is to `n - 1` exactly; see the note there.
 */
export function sampleGridTriangulated(
  data: Float32Array,
  n: number,
  fx: number,
  fz: number,
): number {
  const x = clamp(fx, 0, n - 1);
  const z = clamp(fz, 0, n - 1);
  const ix = Math.floor(x);
  const iz = Math.floor(z);
  const u = x - ix;
  const v = z - iz;
  const base = ix * n + iz;
  const nextX = ix + 1 < n ? n : 0;
  const nextZ = iz + 1 < n ? 1 : 0;
  const h00 = data[base]!;
  const h10 = data[base + nextX]!;
  const h01 = data[base + nextZ]!;
  const h11 = data[base + nextX + nextZ]!;
  if (u + v <= 1) {
    return h00 + (h10 - h00) * u + (h01 - h00) * v;
  }
  return h11 + (h01 - h11) * (1 - u) + (h10 - h11) * (1 - v);
}

/* -------------------------------------------------------------------------- */
/* Landform primitives                                                        */
/* -------------------------------------------------------------------------- */

/**
 * A jump crest: a broad rise with a sharp convex cap on top.
 *
 * The broad rise alone is useless — a quintic ramp has zero curvature at its
 * summit, so you just drive over it. The cap is a parabola whose curvature is
 * `-2*lip/lipWidth^2`; you leave the ground when that exceeds `g / v^2`, so
 * with lip=1.5 and lipWidth=7.5 the launch speed is about 13 m/s. The far side
 * is deliberately longer than the approach so the landing is a ramp, not a
 * flat slam.
 */
function kickerProfile(s: number, k: KickerSpec): number {
  let h: number;
  if (s <= -k.approach || s >= k.landing) {
    h = 0;
  } else if (s <= 0) {
    h = k.height * smootherstep(-k.approach, 0, s);
  } else {
    h = k.height * (1 - smootherstep(0, k.landing, s));
  }
  if (s > -k.lipWidth && s < k.lipWidth) {
    const t = s / k.lipWidth;
    h += k.lip * (1 - t * t);
  }
  return h;
}

/**
 * Design-space superellipse distance at which the boundary range starts to
 * lift the ground. Everything inside this is the drivable world; the relief
 * and slope budgets are quoted for that region, not for the wall.
 */
export const BOUNDARY_INNER = 358;

/**
 * Superellipse distance — a rounded square. Used for the boundary range.
 *
 * The exponent is 8 rather than 4 on purpose. The world is a square, so a ring
 * of *constant* metric distance from the centre bulges deep into the corners:
 * at p=4 the corner of a 1024 m map sits at distance 609 while the edge
 * mid-points sit at 512, so a "60 m wide" range is 60 m wide on the edges and
 * 160 m wide in the corners. Ring area is the scarcest resource in the whole
 * balance — every metre of width costs about 0.4 % of the map — so the metric
 * needs to hug the square. p=8 puts the corners at 558.
 */
export function boundaryDistance(x: number, z: number): number {
  const x2 = x * x;
  const z2 = z * z;
  const x4 = x2 * x2;
  const z4 = z2 * z2;
  return Math.sqrt(Math.sqrt(Math.sqrt(x4 * x4 + z4 * z4)));
}

/* -------------------------------------------------------------------------- */
/* Erosion                                                                    */
/* -------------------------------------------------------------------------- */

interface Brush {
  offsets: Int32Array;
  weights: Float32Array;
}

function buildBrush(radius: number, n: number): Brush {
  const offs: number[] = [];
  const wts: number[] = [];
  let total = 0;
  for (let dx = -radius; dx <= radius; dx++) {
    for (let dz = -radius; dz <= radius; dz++) {
      const d2 = dx * dx + dz * dz;
      if (d2 > radius * radius) continue;
      const w = 1 - Math.sqrt(d2) / radius;
      offs.push(dx * n + dz);
      wts.push(w);
      total += w;
    }
  }
  const weights = new Float32Array(wts.length);
  for (let i = 0; i < wts.length; i++) weights[i] = wts[i]! / total;
  return { offsets: new Int32Array(offs), weights };
}

/**
 * Particle-based hydraulic erosion. Droplets follow the gradient, pick up
 * sediment on steep ground and drop it where the slope eases — which is
 * exactly what produces the two things that make terrain read as a real place:
 * dendritic gullies on the upper slopes and flat, filled valley floors.
 */
function hydraulicErode(
  heights: Float32Array,
  n: number,
  seed: number,
  droplets: number,
): void {
  const rnd = mulberry32(seed ^ 0x5eed10a3);
  const brush = buildBrush(3, n);
  const { offsets, weights } = brush;
  const brushLen = offsets.length;

  const inertia = 0.05;
  const capacityFactor = 3.4;
  const minCapacity = 0.008;
  const erodeSpeed = 0.34;
  const depositSpeed = 0.34;
  const evaporate = 0.012;
  const gravity = 5;
  const maxLifetime = 46;
  const margin = 4;

  for (let d = 0; d < droplets; d++) {
    let px = margin + rnd() * (n - 1 - 2 * margin);
    let pz = margin + rnd() * (n - 1 - 2 * margin);
    let dirX = 0;
    let dirZ = 0;
    let speed = 1;
    let water = 1;
    let sediment = 0;

    for (let life = 0; life < maxLifetime; life++) {
      const ix = Math.floor(px);
      const iz = Math.floor(pz);
      if (ix < 1 || iz < 1 || ix >= n - 2 || iz >= n - 2) break;
      const u = px - ix;
      const v = pz - iz;
      const base = ix * n + iz;
      const h00 = heights[base]!;
      const h10 = heights[base + n]!;
      const h01 = heights[base + 1]!;
      const h11 = heights[base + n + 1]!;

      // Bilinear gradient.
      const gx = (h10 - h00) * (1 - v) + (h11 - h01) * v;
      const gz = (h01 - h00) * (1 - u) + (h11 - h10) * u;
      const hOld = (h00 * (1 - u) + h10 * u) * (1 - v) + (h01 * (1 - u) + h11 * u) * v;

      dirX = dirX * inertia - gx * (1 - inertia);
      dirZ = dirZ * inertia - gz * (1 - inertia);
      const dl = Math.hypot(dirX, dirZ);
      if (dl < 1e-6) break;
      dirX /= dl;
      dirZ /= dl;
      px += dirX;
      pz += dirZ;

      const nix = Math.floor(px);
      const niz = Math.floor(pz);
      if (nix < 1 || niz < 1 || nix >= n - 2 || niz >= n - 2) break;
      const nu = px - nix;
      const nv = pz - niz;
      const nb = nix * n + niz;
      const n00 = heights[nb]!;
      const n10 = heights[nb + n]!;
      const n01 = heights[nb + 1]!;
      const n11 = heights[nb + n + 1]!;
      const hNew = (n00 * (1 - nu) + n10 * nu) * (1 - nv) + (n01 * (1 - nu) + n11 * nu) * nv;
      const dh = hNew - hOld;

      const capacity = Math.max(-dh * speed * water * capacityFactor, minCapacity);

      if (sediment > capacity || dh > 0) {
        // Uphill or oversaturated: drop sediment into the four cells we left.
        const amount = dh > 0 ? Math.min(dh, sediment) : (sediment - capacity) * depositSpeed;
        sediment -= amount;
        heights[base] = h00 + amount * (1 - u) * (1 - v);
        heights[base + n] = h10 + amount * u * (1 - v);
        heights[base + 1] = h01 + amount * (1 - u) * v;
        heights[base + n + 1] = h11 + amount * u * v;
      } else {
        const amount = Math.min((capacity - sediment) * erodeSpeed, -dh);
        for (let b = 0; b < brushLen; b++) {
          const idx = base + offsets[b]!;
          if (idx < 0 || idx >= heights.length) continue;
          const w = weights[b]! * amount;
          const cur = heights[idx]!;
          const taken = cur < w ? cur : w;
          heights[idx] = cur - taken;
          sediment += taken;
        }
      }

      speed = Math.sqrt(Math.max(0, speed * speed + -dh * gravity));
      water *= 1 - evaporate;
      if (water < 0.01) break;
    }
  }
}

/**
 * Thermal erosion. Any slope steeper than the talus angle sheds material to
 * its lowest neighbour. Cheap, and it is what gives scree slopes a consistent
 * angle of repose instead of noise-shaped cliffs.
 */
function thermalErode(
  heights: Float32Array,
  n: number,
  spacing: number,
  iterations: number,
  talusDeg: number,
): void {
  const talus = Math.tan((talusDeg * Math.PI) / 180) * spacing;
  const delta = new Float32Array(heights.length);
  for (let it = 0; it < iterations; it++) {
    delta.fill(0);
    for (let ix = 1; ix < n - 1; ix++) {
      const row = ix * n;
      for (let iz = 1; iz < n - 1; iz++) {
        const i = row + iz;
        const h = heights[i]!;
        let bestIdx = -1;
        let bestDrop = talus;
        const nb0 = heights[i - n]!;
        const nb1 = heights[i + n]!;
        const nb2 = heights[i - 1]!;
        const nb3 = heights[i + 1]!;
        if (h - nb0 > bestDrop) {
          bestDrop = h - nb0;
          bestIdx = i - n;
        }
        if (h - nb1 > bestDrop) {
          bestDrop = h - nb1;
          bestIdx = i + n;
        }
        if (h - nb2 > bestDrop) {
          bestDrop = h - nb2;
          bestIdx = i - 1;
        }
        if (h - nb3 > bestDrop) {
          bestDrop = h - nb3;
          bestIdx = i + 1;
        }
        if (bestIdx >= 0) {
          const move = (bestDrop - talus) * 0.4;
          delta[i] = delta[i]! - move;
          delta[bestIdx] = delta[bestIdx]! + move;
        }
      }
    }
    for (let i = 0; i < heights.length; i++) heights[i] = heights[i]! + delta[i]!;
  }
}

/* -------------------------------------------------------------------------- */
/* Boundary range                                                             */
/* -------------------------------------------------------------------------- */

/**
 * The wall that closes the world, in three bands.
 *
 * Ring area is the tightest constraint in the whole terrain balance. The map's
 * perimeter is 4 km, so a ring one metre wide is already 0.4 % of the world;
 * an honest-looking 150 m deep mountain range would be 55 % of the map and
 * every square metre of it steeper than anything the Jeep can climb. So the
 * range is built the way a real escarpment is, not the way a hill is:
 *
 *   - a broad, gentle **apron** over the outer quarter of the map. It reads as
 *     foothills, it drains inward toward the wash, and at 10-20 degrees it
 *     stays inside the drivable budget.
 *   - a short, violent **escarpment** in the last ~25 m before the edge. This
 *     is the part that is genuinely impassable, and it is deliberately narrow
 *     because steepness is what costs.
 *   - a **corner mass** beyond the square's edge distance, which only the four
 *     corners reach. It gives the skyline some variety in height without
 *     costing any extra area in the middle of each edge.
 *
 * Applied after erosion, so none of it gets eroded flat.
 */
function addBoundaryRange(
  heights: Float32Array,
  n: number,
  size: number,
  scale: number,
  warpA: Noise2D,
  warpB: Noise2D,
  mount: Noise2D,
): void {
  const cells = n - 1;
  const APRON_IN = BOUNDARY_INNER;
  const APRON_OUT = 496;
  const APRON_RISE = 20;
  const CLIFF_IN = 493;
  const CLIFF_OUT = 519;
  const CORNER_IN = 517;
  const CORNER_OUT = 561;

  for (let ix = 0; ix < n; ix++) {
    const x = (ix / cells - 0.5) * size;
    for (let iz = 0; iz < n; iz++) {
      const z = (iz / cells - 0.5) * size;
      const bd = boundaryDistance(x, z) / scale;
      if (bd <= APRON_IN) continue;

      let h = APRON_RISE * smootherstep(APRON_IN, APRON_OUT, bd) * scale;

      const cliff = smootherstep(CLIFF_IN, CLIFF_OUT, bd);
      if (cliff > 0) {
        // Only warp and evaluate the expensive ridge noise where it shows.
        const w = domainWarp(warpA, warpB, x, z, 1 / 540, 68 * scale, 3);
        const rugged = mount.ridged(w.x / 165, w.z / 165, 4, 2.1, 0.5);
        h += cliff * (74 + 88 * rugged) * scale;
        h += 105 * smootherstep(CORNER_IN, CORNER_OUT, bd) * scale;
      }

      heights[ix * n + iz] += h;
    }
  }
}

/* -------------------------------------------------------------------------- */
/* Route corridors                                                            */
/* -------------------------------------------------------------------------- */

interface Station {
  x: number;
  z: number;
  s: number;
}

function resampleCorridor(points: ReadonlyArray<readonly [number, number]>, step: number): Station[] {
  const out: Station[] = [];
  let s = 0;
  for (let i = 0; i < points.length - 1; i++) {
    const [ax, az] = points[i]!;
    const [bx, bz] = points[i + 1]!;
    const len = Math.hypot(bx - ax, bz - az);
    const n = Math.max(1, Math.round(len / step));
    for (let k = 0; k < n; k++) {
      const t = k / n;
      out.push({ x: lerp(ax, bx, t), z: lerp(az, bz, t), s: s + len * t });
    }
    s += len;
  }
  const last = points[points.length - 1]!;
  out.push({ x: last[0], z: last[1], s });
  return out;
}

/**
 * Applies a corridor to the grid: samples the terrain along the centreline,
 * regulates the gradient into the requested band, then blends the terrain
 * toward that regulated profile (plus a lateral tilt, for off-camber shelves).
 *
 * Regulating a *sampled* profile rather than forcing an absolute ramp is what
 * keeps these reading as trails cut into the hillside instead of concrete
 * ramps floating over it.
 */
function applyCorridor(
  heights: Float32Array,
  routeMask: Float32Array,
  n: number,
  size: number,
  spec: CorridorSpec,
): void {
  const half = size / 2;
  const spacing = size / (n - 1);
  const toGridX = (x: number): number => (x + half) / spacing;
  const toGridZ = (z: number): number => (z + half) / spacing;

  const step = 4;
  const stations = resampleCorridor(spec.points, step);
  const m = stations.length;
  const profile = new Float64Array(m);
  for (let i = 0; i < m; i++) {
    const st = stations[i]!;
    profile[i] = gridBilinear(heights, n, toGridX(st.x), toGridZ(st.z));
  }

  const maxGrad = Math.tan((spec.maxSlopeDeg * Math.PI) / 180);
  const minGrad = Math.tan((spec.minSlopeDeg * Math.PI) / 180);
  const ascending = profile[m - 1]! > profile[0]!;

  const smoothed = new Float64Array(m);
  for (let pass = 0; pass < 90; pass++) {
    // 1-2-1 smoothing keeps the profile from following every erosion gully.
    smoothed[0] = profile[0]!;
    smoothed[m - 1] = profile[m - 1]!;
    for (let i = 1; i < m - 1; i++) {
      smoothed[i] = (profile[i - 1]! + 2 * profile[i]! + profile[i + 1]!) * 0.25;
    }
    profile.set(smoothed);

    // Gradient band, enforced symmetrically so the error spreads both ways.
    for (let sweep = 0; sweep < 2; sweep++) {
      const forward = sweep === 0;
      for (let k = 1; k < m; k++) {
        const i = forward ? k : m - 1 - k;
        const j = forward ? i - 1 : i + 1;
        const ds = Math.abs(stations[i]!.s - stations[j]!.s) || step;
        let dh = profile[i]! - profile[j]!;
        const sign = forward ? 1 : -1;
        // "Ascending" is measured along +s regardless of sweep direction.
        let along = dh * sign;
        const hi = maxGrad * ds;
        const lo = minGrad * ds;
        if (along > hi) along = hi;
        else if (along < -hi) along = -hi;
        if (minGrad > 0) {
          if (ascending && along < lo) along = lo;
          else if (!ascending && along > -lo) along = -lo;
        }
        dh = along * sign;
        const target = profile[j]! + dh;
        // Split the correction so neither end gets dragged the whole way.
        const err = target - profile[i]!;
        profile[i] = profile[i]! + err * 0.7;
        profile[j] = profile[j]! - err * 0.15;
      }
    }
  }

  const tilt = Math.tan((spec.tiltDeg * Math.PI) / 180);
  const reach = spec.halfWidth + spec.shoulder;
  const total = stations[m - 1]!.s;

  let minX = Infinity;
  let maxX = -Infinity;
  let minZ = Infinity;
  let maxZ = -Infinity;
  for (const p of spec.points) {
    minX = Math.min(minX, p[0]);
    maxX = Math.max(maxX, p[0]);
    minZ = Math.min(minZ, p[1]);
    maxZ = Math.max(maxZ, p[1]);
  }
  const ix0 = Math.max(0, Math.floor(toGridX(minX - reach)));
  const ix1 = Math.min(n - 1, Math.ceil(toGridX(maxX + reach)));
  const iz0 = Math.max(0, Math.floor(toGridZ(minZ - reach)));
  const iz1 = Math.min(n - 1, Math.ceil(toGridZ(maxZ + reach)));

  // Cumulative arc length at each authored vertex, for the nearest-segment test.
  const segStart: number[] = [0];
  for (let i = 0; i < spec.points.length - 1; i++) {
    const a = spec.points[i]!;
    const b = spec.points[i + 1]!;
    segStart.push(segStart[i]! + Math.hypot(b[0] - a[0], b[1] - a[1]));
  }

  const profileAt = (s: number): number => {
    const f = clamp(s / step, 0, m - 1.0001);
    const i = Math.floor(f);
    return lerp(profile[i]!, profile[Math.min(i + 1, m - 1)]!, f - i);
  };

  for (let ix = ix0; ix <= ix1; ix++) {
    const wx = (ix / (n - 1) - 0.5) * size;
    for (let iz = iz0; iz <= iz1; iz++) {
      const wz = (iz / (n - 1) - 0.5) * size;

      let bestDist = Infinity;
      let bestS = 0;
      let bestSide = 0;
      for (let sgi = 0; sgi < spec.points.length - 1; sgi++) {
        const a = spec.points[sgi]!;
        const b = spec.points[sgi + 1]!;
        const pr = segmentProject(wx, wz, a[0], a[1], b[0], b[1]);
        if (pr.dist < bestDist) {
          bestDist = pr.dist;
          bestS = segStart[sgi]! + pr.t * pr.len;
          bestSide = pr.side;
        }
      }
      if (bestDist > reach) continue;

      const lateral = 1 - smootherstep(spec.halfWidth, reach, bestDist);
      // Ease in and out over 36 m rather than 26. A corridor's regulated
      // profile never matches the raw hillside exactly at its two ends, and
      // whatever mismatch is left has to be spent somewhere; spreading it over
      // half again as much ground is the difference between a corridor that
      // merges into the slope and one that steps off it.
      const endFade = smootherstep(0, 36, bestS) * smootherstep(0, 36, total - bestS);
      const w = spec.strength * lateral * endFade;
      if (w <= 0.001) continue;

      const target = profileAt(bestS) + tilt * bestSide * Math.min(bestDist, spec.halfWidth);
      const i = ix * n + iz;
      heights[i] = lerp(heights[i]!, target, w);
      const rm = spec.strength * (1 - smootherstep(spec.halfWidth * 0.6, reach * 0.9, bestDist)) * endFade;
      if (rm > routeMask[i]!) routeMask[i] = rm;
    }
  }
}

/* -------------------------------------------------------------------------- */
/* Drivability clamp                                                          */
/* -------------------------------------------------------------------------- */

/**
 * Relaxes any slope steeper than `maxDeg` — but only where `mask` says this is
 * an intended route. Everywhere else the terrain keeps its cliffs, which is
 * what makes the boundary range genuinely impassable rather than merely
 * discouraging.
 */
function clampDrivableSlopes(
  heights: Float32Array,
  mask: Float32Array,
  n: number,
  spacing: number,
  maxDeg: number,
  iterations: number,
): number {
  const maxDelta = Math.tan((maxDeg * Math.PI) / 180) * spacing;
  const delta = new Float32Array(heights.length);
  let moved = 0;
  for (let it = 0; it < iterations; it++) {
    delta.fill(0);
    for (let ix = 1; ix < n - 1; ix++) {
      const row = ix * n;
      for (let iz = 1; iz < n - 1; iz++) {
        const i = row + iz;
        const m = mask[i]!;
        if (m <= 0.01) continue;
        const h = heights[i]!;
        for (let k = 0; k < 4; k++) {
          const j = k === 0 ? i - n : k === 1 ? i + n : k === 2 ? i - 1 : i + 1;
          const d = h - heights[j]!;
          if (d > maxDelta) {
            // Weight by the *lower* of the two masks so the correction never
            // leaks out of the corridor and flattens the scenery.
            const w = Math.min(m, Math.max(mask[j]!, 0.35)) * 0.25;
            const move = (d - maxDelta) * 0.5 * w;
            delta[i] = delta[i]! - move;
            delta[j] = delta[j]! + move;
            moved += move;
          }
        }
      }
    }
    for (let i = 0; i < heights.length; i++) heights[i] = heights[i]! + delta[i]!;
  }
  return moved;
}

/* -------------------------------------------------------------------------- */
/* Ambient occlusion                                                          */
/* -------------------------------------------------------------------------- */

/**
 * Horizon-angle ambient occlusion, computed on a coarse grid and upsampled,
 * plus a full-resolution cavity term. Multiplied into albedo it costs nothing
 * at runtime and does more for the sense of three-dimensional relief than any
 * amount of extra lighting.
 */
function bakeOcclusion(heights: Float32Array, n: number, spacing: number): Float32Array {
  const out = new Float32Array(heights.length);

  // --- coarse horizon AO -------------------------------------------------
  const cN = 257;
  const coarse = new Float32Array(cN * cN);
  const stride = (n - 1) / (cN - 1);
  const dirs = 8;
  const steps = [1.5, 3, 6, 12, 24, 48, 96];
  for (let cx = 0; cx < cN; cx++) {
    const fx = cx * stride;
    for (let cz = 0; cz < cN; cz++) {
      const fz = cz * stride;
      const h0 = gridBilinear(heights, n, fx, fz);
      let occ = 0;
      for (let d = 0; d < dirs; d++) {
        const a = (d / dirs) * Math.PI * 2 + 0.31;
        const dx = Math.cos(a);
        const dz = Math.sin(a);
        let maxTan = 0;
        for (const r of steps) {
          const sx = fx + (dx * r) / spacing;
          const sz = fz + (dz * r) / spacing;
          if (sx < 0 || sz < 0 || sx > n - 1 || sz > n - 1) continue;
          const hs = gridBilinear(heights, n, sx, sz);
          const t = (hs - h0) / r;
          if (t > maxTan) maxTan = t;
        }
        occ += maxTan / Math.sqrt(1 + maxTan * maxTan); // sin of the horizon angle
      }
      coarse[cx * cN + cz] = 1 - occ / dirs;
    }
  }

  // --- full-res cavity + upsample ---------------------------------------
  const r = 3;
  for (let ix = 0; ix < n; ix++) {
    const cfx = (ix / (n - 1)) * (cN - 1);
    for (let iz = 0; iz < n; iz++) {
      const i = ix * n + iz;
      const h = heights[i]!;
      let sum = 0;
      let count = 0;
      for (let dx = -r; dx <= r; dx += r) {
        const jx = clamp(ix + dx, 0, n - 1);
        for (let dz = -r; dz <= r; dz += r) {
          const jz = clamp(iz + dz, 0, n - 1);
          sum += heights[jx * n + jz]!;
          count++;
        }
      }
      const cavity = clamp(0.5 + (h - sum / count) * 0.55, 0.0, 1.0);
      const macro = gridBilinear(coarse, cN, cfx, (iz / (n - 1)) * (cN - 1));
      out[i] = clamp(macro * 0.72 + cavity * 0.28, 0.06, 1);
    }
  }
  return out;
}

/* -------------------------------------------------------------------------- */
/* Main generator                                                             */
/* -------------------------------------------------------------------------- */

export function generateTerrain(opts: TerrainFieldOptions = {}): TerrainField {
  const seed = opts.seed ?? 20250807;
  const size = opts.size ?? DESIGN_SIZE;
  const cells = opts.resolution ?? 1024;
  const doErosion = opts.erosion ?? true;
  const n = cells + 1;
  const spacing = size / cells;
  const half = size / 2;
  const scale = size / DESIGN_SIZE;
  const timings: Record<string, number> = {};
  const now = (): number =>
    typeof performance !== 'undefined' ? performance.now() : Date.now();

  const layout = scaleLayout(scale);

  const heights = new Float32Array(n * n);
  const routeMask = new Float32Array(n * n);

  // Independent noise fields. Deriving each from `seed` keeps the whole world
  // reproducible from a single number.
  const nBase = new Noise2D(seed + 1);
  const nRidge = new Noise2D(seed + 2);
  const nWarpA = new Noise2D(seed + 3);
  const nWarpB = new Noise2D(seed + 4);
  const nFine = new Noise2D(seed + 5);
  const nValley = new Noise2D(seed + 6);
  const nMount = new Noise2D(seed + 7);
  const nRock = new Noise2D(seed + 8);
  const nRock2 = new Noise2D(seed + 9);
  const nSplatA = new Noise2D(seed + 10);
  const nSplatB = new Noise2D(seed + 11);
  const nSplatC = new Noise2D(seed + 12);

  /* -- 1. macro landform + fractal detail -------------------------------- */
  let t0 = now();
  for (let ix = 0; ix < n; ix++) {
    const x = (ix / cells - 0.5) * size;
    for (let iz = 0; iz < n; iz++) {
      const z = (iz / cells - 0.5) * size;

      // Domain warp first: everything downstream inherits the bent coordinate
      // frame, so ridgelines curve instead of running dead straight.
      const w = domainWarp(nWarpA, nWarpB, x, z, 1 / 540, 68 * scale, 3);

      // ── Why these numbers ────────────────────────────────────────────────
      // The slope a noise layer contributes is amplitude / wavelength, and in
      // an fBm each octave multiplies that by `gain * lacunarity`. With the
      // usual gain 0.5 and lacunarity 2 that product is 1: every octave adds
      // the *same* gradient as the one before, so an 6-octave fBm is six times
      // as steep as its base octave while being barely any taller. That is a
      // fine choice for a mountain renderer and a terrible one for a vehicle,
      // and it is what made this world unreadable at a 49 degree median.
      //
      // Here every layer runs `gain * lacunarity` below 1, so the fractal
      // detail decays in slope as it gets finer: relief comes from the long
      // wavelengths, texture from the short ones, and nothing in between
      // stacks up into a wall. Drama is spent deliberately instead — on the
      // massifs, the wash banks, the bowl and the boundary range.
      let h = 33;
      // Broad basins and divides. 470 m base wavelength: one basin is most of
      // a drive across the map.
      h += 23.5 * nBase.fbm(w.x / 470, w.z / 470, 5, 2.03, 0.45);
      // Ridge structure — drainage divides and spurs, widened so the same
      // relief is spread over half again as much ground as before.
      h += 18.5 * (nRidge.ridged(w.x / 640, w.z / 640, 4, 2.05, 0.44) - 0.33);
      // Hillside undulation: enough to make a traverse interesting.
      h += 3.7 * nFine.fbm(x / 150, z / 150, 3, 2.0, 0.42);
      // Micro-relief so the ground is never a plane under the wheels.
      h += 1.0 * nFine.fbm(x / 34, z / 34, 3, 2.0, 0.4);

      // Authored masses under the fractal, so the map has real landmarks.
      // These carry most of the world's relief now, which is the point: relief
      // you placed reads as a landscape, relief the fBm sprayed everywhere
      // reads as noise.
      for (const m of layout.massifs) {
        const d = Math.hypot(x - m.x, z - m.z) / m.radius;
        if (d < 2.2) h += m.height * Math.exp(-d * d);
      }

      heights[ix * n + iz] = h;
    }
  }
  timings.landform = now() - t0;

  /* -- 2. erosion --------------------------------------------------------- */
  // Runs before the boundary range is added, for two reasons: the droplets
  // would otherwise spend most of their lifetime sliding down a 70 degree wall
  // instead of shaping the ground the player drives on, and the talus pass
  // would chew the wall down to its angle of repose and open a way out.
  if (doErosion) {
    t0 = now();
    // Droplet count scales with area so a smaller world is not over-eroded.
    const droplets = Math.round(78000 * (n * n) / (1025 * 1025));
    hydraulicErode(heights, n, seed, droplets);
    timings.hydraulic = now() - t0;

    t0 = now();
    // 28 degrees of talus, not 40. Anything above the angle of repose sheds
    // material until it reaches it, so this is the single most effective
    // control on how much of the map is undrivable — and unlike scaling the
    // heights down it removes only the *excess* slope and leaves the relief.
    thermalErode(heights, n, spacing, 36, 24);
    timings.thermal = now() - t0;
  }

  /* -- 2b. boundary range -------------------------------------------------- */
  t0 = now();
  addBoundaryRange(heights, n, size, scale, nWarpA, nWarpB, nMount);
  timings.boundary = now() - t0;

  /* -- 3. valley carve + playa ------------------------------------------- */
  t0 = now();
  const pan = layout.pan;
  for (let ix = 0; ix < n; ix++) {
    const x = (ix / cells - 0.5) * size;
    for (let iz = 0; iz < n; iz++) {
      const z = (iz / cells - 0.5) * size;
      const i = ix * n + iz;
      let h = heights[i]!;

      // -- dry wash ------------------------------------------------------
      const cx = layout.washCenterX(z);
      const wobble = 9 * scale * nValley.fbm(x / 70, z / 70, 3);
      const d = Math.abs(x - cx) + wobble;
      const taper = 1 - smootherstep(330 * scale, 452 * scale, Math.abs(z));
      if (taper > 0.001) {
        const floorY = layout.washFloorY(z);
        const halfFloor = (26 + 9 * nValley.sample(z / 90, 3.3)) * scale;
        // The bank ramp used to be 34 m wide, which on ground sitting 40 m
        // above the floor is a 50 degree canyon wall: you could drive the wash
        // but never enter or leave it, and it accounted for more of the map's
        // undrivable ground than every natural hillside put together. At 58 m
        // it is a valley side you can pick a line up, which is what a dry wash
        // in rolling country actually looks like.
        const bankW = 70 * scale;
        const t = smootherstep(halfFloor, halfFloor + bankW, d);
        const influence = 1 - smootherstep(halfFloor + bankW, halfFloor + bankW + 62 * scale, d);
        const bankTarget = Math.max(h, floorY + 7.5 * scale);
        const carved = lerp(floorY, bankTarget, t);
        h = lerp(h, carved, influence * taper);
        const rm = (1 - smootherstep(halfFloor * 0.4, halfFloor + bankW * 0.85, d)) * taper;
        if (rm > routeMask[i]!) routeMask[i] = rm;
      }

      // -- bowl ----------------------------------------------------------
      const bd = Math.hypot(x - layout.bowl.x, z - layout.bowl.z) / layout.bowl.radius;
      if (bd < 1) {
        const k = 1 - bd * bd;
        h -= layout.bowl.depth * k * k;
        const rm = 0.85 * smootherstep(0, 0.25, 1 - bd);
        if (rm > routeMask[i]!) routeMask[i] = rm;
      }

      // -- playa near spawn ----------------------------------------------
      const pd = Math.hypot(x - pan.x, z - pan.z);
      if (pd < pan.radius) {
        const w = 1 - smootherstep(pan.radius * 0.62, pan.radius, pd);
        const panY = layout.washFloorY(pan.z) + 0.35 * nFine.fbm(x / 40, z / 40, 2);
        h = lerp(h, panY, w * 0.96);
        if (w > routeMask[i]!) routeMask[i] = w;
      }

      heights[i] = h;
    }
  }
  timings.landmarks = now() - t0;

  /* -- 4. route corridors ------------------------------------------------- */
  t0 = now();
  for (const spec of layout.corridors) applyCorridor(heights, routeMask, n, size, spec);
  timings.corridors = now() - t0;

  /* -- 5. drivability clamp ---------------------------------------------- */
  t0 = now();
  // 30 degrees, down from 34. This only touches ground the route mask claims,
  // so it costs no scenery, and 34 was letting the shoulders of the corridors
  // keep steps the regulated centreline had already ruled out.
  clampDrivableSlopes(heights, routeMask, n, spacing, 28, 42);
  timings.clamp = now() - t0;

  /* -- 6. jump crests ----------------------------------------------------- */
  t0 = now();
  for (const k of layout.kickers) {
    const reach = Math.max(k.approach, k.landing) + 4;
    const span = k.span * 2.2;
    const px = -k.dz;
    const pz = k.dx;
    const ix0 = Math.max(0, Math.floor(((k.x - reach - span + half) / spacing)));
    const ix1 = Math.min(n - 1, Math.ceil(((k.x + reach + span + half) / spacing)));
    const iz0 = Math.max(0, Math.floor(((k.z - reach - span + half) / spacing)));
    const iz1 = Math.min(n - 1, Math.ceil(((k.z + reach + span + half) / spacing)));
    for (let ix = ix0; ix <= ix1; ix++) {
      const x = (ix / cells - 0.5) * size;
      for (let iz = iz0; iz <= iz1; iz++) {
        const z = (iz / cells - 0.5) * size;
        const ox = x - k.x;
        const oz = z - k.z;
        const s = ox * k.dx + oz * k.dz;
        const lateral = ox * px + oz * pz;
        const lat = Math.exp(-(lateral / k.span) * (lateral / k.span));
        if (lat < 0.02) continue;
        const add = kickerProfile(s, k) * lat;
        if (add === 0) continue;
        const i = ix * n + iz;
        heights[i] = heights[i]! + add;
        const rm = lat * 0.9;
        if (rm > routeMask[i]!) routeMask[i] = rm;
      }
    }
  }
  timings.kickers = now() - t0;

  /* -- 7. rock chatter ---------------------------------------------------- */
  t0 = now();
  const rocky = layout.rocky;
  for (let ix = 0; ix < n; ix++) {
    const x = (ix / cells - 0.5) * size;
    const rxn = (x - rocky.x) / rocky.rx;
    if (Math.abs(rxn) > 1.35) continue;
    for (let iz = 0; iz < n; iz++) {
      const z = (iz / cells - 0.5) * size;
      const rzn = (z - rocky.z) / rocky.rz;
      const rd = Math.hypot(rxn, rzn);
      if (rd > 1.25) continue;
      const m = 1 - smootherstep(0.72, 1.25, rd);
      const i = ix * n + iz;
      // High-frequency chatter plus scattered boulders. Amplitudes are small
      // enough not to matter for route-finding but large enough that the
      // suspension has something to do.
      const chatter = 0.34 * nRock.fbm(x / 2.6, z / 2.6, 3);
      const bars = 0.55 * (nRock.ridged(x / 7.5, z / 7.5, 3) - 0.4);
      const boulder = 1.15 * Math.pow(nRock2.ridged(x / 13, z / 13, 2), 3);
      heights[i] = heights[i]! + m * (chatter + bars + boulder);
    }
  }
  timings.chatter = now() - t0;

  /* -- 8. bakes ----------------------------------------------------------- */
  t0 = now();
  let minHeight = Infinity;
  let maxHeight = -Infinity;
  for (let i = 0; i < heights.length; i++) {
    const h = heights[i]!;
    if (h < minHeight) minHeight = h;
    if (h > maxHeight) maxHeight = h;
  }
  const occlusion = bakeOcclusion(heights, n, spacing);
  timings.occlusion = now() - t0;

  t0 = now();
  const controlSize = Math.min(1024, cells);
  const control = bakeControlMap(
    heights,
    routeMask,
    n,
    size,
    controlSize,
    layout,
    { a: nSplatA, b: nSplatB, c: nSplatC },
    seed,
  );
  timings.control = now() - t0;

  return {
    seed,
    size,
    halfSize: half,
    cells,
    samples: n,
    spacing,
    heights,
    occlusion,
    control,
    controlSize,
    routeMask,
    minHeight,
    maxHeight,
    layout,
    timings,
  };
}

function scaleLayout(scale: number): ScaledLayout & { massifs: typeof FEATURES.massifs } {
  const s = scale;
  return {
    scale: s,
    massifs: FEATURES.massifs.map((m) => ({
      x: m.x * s,
      z: m.z * s,
      radius: m.radius * s,
      height: m.height * s,
    })) as unknown as typeof FEATURES.massifs,
    pan: { x: FEATURES.pan.x * s, z: FEATURES.pan.z * s, radius: FEATURES.pan.radius * s },
    bowl: {
      x: FEATURES.bowl.x * s,
      z: FEATURES.bowl.z * s,
      radius: FEATURES.bowl.radius * s,
      depth: FEATURES.bowl.depth * s,
    },
    rocky: {
      x: FEATURES.rocky.x * s,
      z: FEATURES.rocky.z * s,
      rx: FEATURES.rocky.rx * s,
      rz: FEATURES.rocky.rz * s,
    },
    kickers: FEATURES.kickers.map((k) => ({
      name: k.name,
      x: k.x * s,
      z: k.z * s,
      dx: k.dx,
      dz: k.dz,
      height: k.height * s,
      lip: k.lip * s,
      approach: k.approach * s,
      landing: k.landing * s,
      lipWidth: k.lipWidth * s,
      span: k.span * s,
    })),
    corridors: FEATURES.corridors.map((c) => ({
      name: c.name,
      points: c.points.map((p) => [p[0] * s, p[1] * s] as const),
      halfWidth: c.halfWidth * s,
      shoulder: c.shoulder * s,
      tiltDeg: c.tiltDeg,
      minSlopeDeg: c.minSlopeDeg,
      maxSlopeDeg: c.maxSlopeDeg,
      strength: c.strength,
    })),
    washCenterX: (z: number) => washCenterX(z / s) * s,
    washFloorY: (z: number) => washFloorY(z / s) * s,
  };
}

/* -------------------------------------------------------------------------- */
/* Splat control map                                                          */
/* -------------------------------------------------------------------------- */

/**
 * Computes the five layer weights at a world position. This is the *only*
 * place the splat rules live: the control map the shader samples and the
 * answer `surfaceAt` gives are both derived from this function, so the grip
 * model and the picture can never disagree about what the ground is.
 *
 * Returns normalised weights in SURFACE_LAYERS order.
 */
export function surfaceWeights(
  out: Float64Array,
  x: number,
  z: number,
  height: number,
  slope: number,
  route: number,
  layout: ScaledLayout,
  noise: { a: Noise2D; b: Noise2D; c: Noise2D },
): void {
  const s = layout.scale;
  const patch = noise.a.fbm(x / (95 * s), z / (95 * s), 3);
  const fine = noise.b.fbm(x / (38 * s), z / (38 * s), 3);
  const moisture = 0.5 + 0.5 * noise.c.fbm(x / (210 * s), z / (210 * s), 3);

  // How "wash-like" is this point — used for sand on the floor and gravel on
  // the banks and alluvial fans.
  const cx = layout.washCenterX(z);
  const wd = Math.abs(x - cx) + 13 * s * noise.a.fbm(x / (70 * s), z / (70 * s), 3);
  const taper = 1 - smootherstep(330 * s, 452 * s, Math.abs(z));
  const floorNess = (1 - smootherstep(38 * s, 76 * s, wd)) * taper;
  const bankNess = (smootherstep(30 * s, 58 * s, wd) - smootherstep(74 * s, 124 * s, wd)) * taper;

  const rockyD = Math.hypot((x - layout.rocky.x) / layout.rocky.rx, (z - layout.rocky.z) / layout.rocky.rz);
  const rockyNess = 1 - smootherstep(0.6, 1.2, rockyD);

  const panD = Math.hypot(x - layout.pan.x, z - layout.pan.z) / layout.pan.radius;
  const panNess = 1 - smootherstep(0.55, 1.05, panD);

  const highland = smoothstep(112 * s, 176 * s, height);

  // Every threshold below is a *gradient* (rise over run), not an angle:
  // 0.27 is 15 deg, 0.36 is 20 deg, 0.58 is 30 deg, 1.0 is 45 deg.
  //
  // rock — genuinely steep ground and bare summits. The old rule started
  // fading rock in at 19 deg, which on the old 49 deg world meant everything;
  // even on gentle ground it would have painted every hillside grey. Bare rock
  // belongs on ground you cannot stand on, so it starts at 27 deg.
  let wRock = smoothstep(0.42, 0.84, slope + 0.1 * fine) * 2.5 + highland * 2.1;
  // sand — wash bottom and the playa, wherever it is flat. Weighted to
  // actually win there: this is the surface the level is calibrated on and it
  // used to lose to `dirt` on every square metre of it.
  let wSand =
    (1 - smoothstep(0.16, 0.44, slope)) *
    (floorNess * 5.2 + panNess * 5.8) *
    (0.8 + 0.32 * patch);
  // gravel — banks, fans, broken ground, mid slopes
  let wGravel =
    bankNess * 1.15 +
    rockyNess * 1.5 +
    smoothstep(0.33, 0.66, slope) * 1.0 * (1 - highland) +
    0.16 * fine;
  // grass — gentle, wetter ground away from the wash. The old rule needed
  // 28 m of elevation before any grass appeared and cut it off again by
  // 120 m; on a world whose ground ran 100-300 m that window was empty, which
  // is why the map came out as a quarry.
  let wGrass =
    (1 - smoothstep(0.30, 0.58, slope)) *
    smoothstep(20 * s, 42 * s, height) *
    (1 - smoothstep(116 * s, 170 * s, height)) *
    (0.42 + 0.86 * moisture) *
    (0.7 + 0.75 * patch) *
    1.0;
  // dirt — the default ground, and the trails. `route` used to add 1.9 here,
  // which is more than any other layer's total: every graded trail came out
  // dirt no matter what it was cut through, including the sand of the wash.
  // The `trail` multipliers below already do that job, selectively.
  let wDirt = 0.42 + route * 0.55 + 0.24 * fine + 0.2 * (1 - highland) * (1 - moisture);

  if (wRock < 0) wRock = 0;
  if (wSand < 0) wSand = 0;
  if (wGravel < 0) wGravel = 0;
  if (wGrass < 0) wGrass = 0;
  if (wDirt < 0) wDirt = 0;

  // Trails cut through everything: a graded track is dirt, not grass.
  const trail = smootherstep(0.35, 0.85, route);
  wGrass *= 1 - trail * 0.92;
  wSand *= 1 - trail * 0.2;
  wRock *= 1 - trail * 0.6;

  const sum = wDirt + wGrass + wRock + wGravel + wSand;
  const inv = sum > 1e-6 ? 1 / sum : 0;
  out[0] = wDirt * inv;
  out[1] = wGrass * inv;
  out[2] = wRock * inv;
  out[3] = wGravel * inv;
  out[4] = wSand * inv;
}

function bakeControlMap(
  heights: Float32Array,
  routeMask: Float32Array,
  n: number,
  size: number,
  controlSize: number,
  layout: ScaledLayout,
  noise: { a: Noise2D; b: Noise2D; c: Noise2D },
  seed: number,
): Uint8Array {
  const out = new Uint8Array(controlSize * controlSize * 4);
  const half = size / 2;
  const texel = size / controlSize;
  const spacing = size / (n - 1);
  const w = new Float64Array(LAYER_COUNT);

  for (let cz = 0; cz < controlSize; cz++) {
    const z = -half + (cz + 0.5) * texel;
    const fz = (z + half) / spacing;
    for (let cx = 0; cx < controlSize; cx++) {
      const x = -half + (cx + 0.5) * texel;
      const fx = (x + half) / spacing;

      const h = gridBilinear(heights, n, fx, fz);
      const hx = gridBilinear(heights, n, fx + 1, fz) - gridBilinear(heights, n, fx - 1, fz);
      const hz = gridBilinear(heights, n, fx, fz + 1) - gridBilinear(heights, n, fx, fz - 1);
      const slope = Math.hypot(hx, hz) / (2 * spacing);
      const route = gridBilinear(routeMask, n, fx, fz);

      surfaceWeights(w, x, z, h, slope, route, layout, noise);

      // Deterministic ordered dither so 8-bit quantisation does not band the
      // long, shallow gradients where one layer fades into the next.
      const d = (hash2i(cx, cz, seed) - 0.5) * (1 / 255);
      const o = (cz * controlSize + cx) * 4;
      out[o] = clamp(Math.round((w[1]! + d) * 255), 0, 255);
      out[o + 1] = clamp(Math.round((w[2]! + d) * 255), 0, 255);
      out[o + 2] = clamp(Math.round((w[3]! + d) * 255), 0, 255);
      out[o + 3] = clamp(Math.round((w[4]! + d) * 255), 0, 255);
    }
  }
  return out;
}

/**
 * Bilinear sample of the control map that reproduces exactly what a GPU does
 * with `texture2D(control, uv)` at mip level 0 — same texel centres, same
 * weights — then returns the dominant layer.
 */
export function sampleSurface(
  control: Uint8Array,
  controlSize: number,
  size: number,
  x: number,
  z: number,
): SurfaceKind {
  const half = size / 2;
  const fx = clamp(((x + half) / size) * controlSize - 0.5, 0, controlSize - 1.0001);
  const fz = clamp(((z + half) / size) * controlSize - 0.5, 0, controlSize - 1.0001);
  const x0 = Math.floor(fx);
  const z0 = Math.floor(fz);
  const tx = fx - x0;
  const tz = fz - z0;
  const x1 = Math.min(x0 + 1, controlSize - 1);
  const z1 = Math.min(z0 + 1, controlSize - 1);
  const o00 = (z0 * controlSize + x0) * 4;
  const o10 = (z0 * controlSize + x1) * 4;
  const o01 = (z1 * controlSize + x0) * 4;
  const o11 = (z1 * controlSize + x1) * 4;

  let best = 0;
  let bestW = 0;
  let sum = 0;
  for (let c = 0; c < 4; c++) {
    const v =
      lerp(lerp(out8(control, o00 + c), out8(control, o10 + c), tx),
           lerp(out8(control, o01 + c), out8(control, o11 + c), tx), tz);
    sum += v;
    if (v > bestW) {
      bestW = v;
      best = c + 1;
    }
  }
  const dirt = 1 - sum;
  if (dirt > bestW) return SURFACE_KINDS[0]!;
  return SURFACE_KINDS[best]!;
}

function out8(a: Uint8Array, i: number): number {
  return a[i]! / 255;
}
