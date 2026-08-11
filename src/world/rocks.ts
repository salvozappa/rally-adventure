/**
 * Procedural rock geometry.
 *
 * Every rock is a subdivided icosahedron whose vertices are pushed along their
 * own direction by fractal noise, then squashed anisotropically and cut off
 * flat underneath so it sits on the ground instead of balancing on a point.
 * The result is converted to non-indexed geometry, which gives hard flat facets
 * — the same trick the era used, and the reason low-poly rocks read as *rock*
 * rather than as a lumpy sphere.
 *
 * Three details do most of the work:
 *
 *  - **Per-face triplanar UVs.** The spherical UVs a polyhedron ships with have
 *    a seam and stretch badly at the poles. Projecting each face along its own
 *    dominant axis removes both problems and costs nothing at runtime.
 *  - **Cavity shading baked into vertex colour.** Vertices that the noise pushed
 *    *inward* are darkened; upward faces are lifted. That single term is what
 *    makes the facets read as relief under flat retro lighting.
 *  - **Convex hull points harvested during the build.** The collider is built
 *    from the same displaced vertices the mesh uses, so what you hit is what you
 *    see.
 */

import * as THREE from 'three';
import { Rng, clamp01, fbm3 } from './scatterTextures';
import type { ColliderShape, ScatterModel } from './vegetation';

/* -------------------------------------------------------------------------- */
/* Specs                                                                      */
/* -------------------------------------------------------------------------- */

export type RockClass = 'boulder' | 'stone' | 'scree';

interface RockSpec {
  name: string;
  /** Base radius, metres. Instances scale around this. */
  size: number;
  /** Non-uniform stretch. */
  sx: number;
  sy: number;
  sz: number;
  /** Displacement amplitude, 0..1 of the radius. */
  amp: number;
  /** Noise frequency — low is blobby, high is broken. */
  freq: number;
  /** 0 rounded, 1 hard angular facets. */
  angular: number;
  /** How much of the sphere is buried, 0..1. */
  bury: number;
  /** Grey tint multiplier. */
  tint: number;
  /** Moss/lichen coverage on upward faces, 0..1. */
  moss: number;
}

const BOULDERS: RockSpec[] = [
  { name: 'round', size: 1.6, sx: 1.1, sy: 0.85, sz: 1.0, amp: 0.22, freq: 1.5, angular: 0.15, bury: 0.3, tint: 1.0, moss: 0.45 },
  { name: 'blocky', size: 1.9, sx: 1.0, sy: 0.95, sz: 1.15, amp: 0.3, freq: 1.1, angular: 0.85, bury: 0.22, tint: 0.9, moss: 0.25 },
  { name: 'slab', size: 2.3, sx: 1.45, sy: 0.42, sz: 1.2, amp: 0.24, freq: 1.8, angular: 0.6, bury: 0.18, tint: 1.05, moss: 0.35 },
  { name: 'tall', size: 1.5, sx: 0.8, sy: 1.55, sz: 0.9, amp: 0.28, freq: 1.4, angular: 0.55, bury: 0.25, tint: 0.95, moss: 0.3 },
  { name: 'split', size: 2.0, sx: 1.2, sy: 0.9, sz: 0.75, amp: 0.38, freq: 2.4, angular: 0.75, bury: 0.28, tint: 0.86, moss: 0.2 },
  { name: 'mossy', size: 1.35, sx: 1.15, sy: 0.8, sz: 1.15, amp: 0.2, freq: 1.7, angular: 0.1, bury: 0.36, tint: 0.92, moss: 0.85 },
  { name: 'weathered', size: 1.75, sx: 1.05, sy: 1.0, sz: 1.0, amp: 0.33, freq: 2.9, angular: 0.4, bury: 0.24, tint: 1.1, moss: 0.4 },
  { name: 'wedge', size: 2.1, sx: 1.3, sy: 0.7, sz: 0.85, amp: 0.3, freq: 1.3, angular: 0.95, bury: 0.2, tint: 0.98, moss: 0.15 },
];

const STONES: RockSpec[] = [
  { name: 'cobble', size: 0.42, sx: 1.15, sy: 0.7, sz: 1.0, amp: 0.2, freq: 2.2, angular: 0.2, bury: 0.35, tint: 1.05, moss: 0.3 },
  { name: 'flat', size: 0.55, sx: 1.35, sy: 0.34, sz: 1.1, amp: 0.18, freq: 2.6, angular: 0.5, bury: 0.3, tint: 1.0, moss: 0.2 },
  { name: 'chunk', size: 0.5, sx: 1.0, sy: 0.9, sz: 1.0, amp: 0.3, freq: 2.0, angular: 0.7, bury: 0.28, tint: 0.92, moss: 0.25 },
  { name: 'shard', size: 0.62, sx: 0.7, sy: 1.1, sz: 1.25, amp: 0.34, freq: 1.8, angular: 1.0, bury: 0.22, tint: 0.88, moss: 0.1 },
  { name: 'pebble', size: 0.26, sx: 1.2, sy: 0.72, sz: 1.05, amp: 0.16, freq: 3.4, angular: 0.1, bury: 0.4, tint: 1.12, moss: 0.15 },
  { name: 'mossy-stone', size: 0.48, sx: 1.05, sy: 0.75, sz: 1.15, amp: 0.22, freq: 2.4, angular: 0.15, bury: 0.38, tint: 0.9, moss: 0.8 },
];

const SCREE: RockSpec[] = [
  { name: 'plate', size: 0.3, sx: 1.5, sy: 0.24, sz: 1.2, amp: 0.16, freq: 3.0, angular: 1.0, bury: 0.42, tint: 1.0, moss: 0.0 },
  { name: 'spall', size: 0.24, sx: 1.1, sy: 0.45, sz: 0.9, amp: 0.24, freq: 3.6, angular: 1.0, bury: 0.4, tint: 0.94, moss: 0.0 },
  { name: 'grit', size: 0.16, sx: 1.0, sy: 0.7, sz: 1.0, amp: 0.28, freq: 4.5, angular: 0.8, bury: 0.45, tint: 1.08, moss: 0.0 },
  { name: 'blade', size: 0.36, sx: 0.6, sy: 0.9, sz: 1.35, amp: 0.3, freq: 2.8, angular: 1.0, bury: 0.35, tint: 0.9, moss: 0.05 },
  { name: 'flake', size: 0.28, sx: 1.3, sy: 0.3, sz: 1.35, amp: 0.2, freq: 3.2, angular: 1.0, bury: 0.44, tint: 1.04, moss: 0.0 },
  { name: 'nub', size: 0.2, sx: 1.0, sy: 0.85, sz: 1.0, amp: 0.22, freq: 4.0, angular: 0.6, bury: 0.46, tint: 0.98, moss: 0.0 },
];

/**
 * Rock albedo. As with the vegetation palette these are the *whole* albedo:
 * `scatter.rock` is a detail map averaging ~0.7 linear, so the rendered value is
 * about 0.7 × these. Dry limestone and granite sit around 0.25-0.35 linear, so
 * the base greys have to be authored well up in the light half of the ramp.
 * Authored dark (the previous 0x6d6a62 against a 0.16-linear texture gave a
 * 0.024 albedo) they render as holes punched in the ground, not stone.
 */
const ROCK_GREY = new THREE.Color(0x9d998c);
const ROCK_WARM = new THREE.Color(0xa89a80);
const ROCK_SHADE = new THREE.Color(0x676054);
/**
 * Lichen, not moss. It tints the upward faces of a boulder; it does not cover
 * them. Pushed too far — a saturated green at 85% coverage — the rocks stop
 * reading as stone at all and turn into green crystals sitting in the grass,
 * which is worse than having no lichen.
 */
const LICHEN = new THREE.Color(0x8e8c67);
const LICHEN_MAX = 0.45;

/* -------------------------------------------------------------------------- */
/* Build                                                                      */
/* -------------------------------------------------------------------------- */

const _a = new THREE.Vector3();
const _b = new THREE.Vector3();
const _c = new THREE.Vector3();
const _n = new THREE.Vector3();
const _e = new THREE.Vector3();
const _col = new THREE.Color();

/**
 * Displace one icosahedron into a rock. `detail` picks the LOD: 1 is 80 faces
 * (the near model), 0 is 20 faces (mid distance and scree).
 */
function buildRockGeometry(
  spec: RockSpec,
  rng: Rng,
  detail: 0 | 1,
): { geo: THREE.BufferGeometry; hull: Float32Array; height: number; radius: number } {
  const src = new THREE.IcosahedronGeometry(1, detail);
  const srcPos = src.getAttribute('position') as THREE.BufferAttribute;
  const count = srcPos.count;
  const pos = new Float32Array(count * 3);
  const uv = new Float32Array(count * 2);
  const col = new Float32Array(count * 3);

  const seedA = rng.int(0xffff);
  const seedB = rng.int(0xffff);
  // Random orientation of the noise field, so two rocks from the same spec do
  // not share a silhouette.
  const ox = rng.range(-40, 40);
  const oy = rng.range(-40, 40);
  const oz = rng.range(-40, 40);

  const displaced = new Float32Array(count * 3);
  const push = new Float32Array(count);

  for (let i = 0; i < count; i++) {
    const dx = srcPos.getX(i);
    const dy = srcPos.getY(i);
    const dz = srcPos.getZ(i);

    let n = fbm3(dx * spec.freq + ox, dy * spec.freq + oy, dz * spec.freq + oz, seedA, 3) - 0.5;
    if (spec.angular > 0) {
      // Quantising the noise creates flat steps in the surface: the ledges and
      // fracture planes that separate an angular rock from a potato.
      const steps = 4;
      const q = Math.round(n * steps) / steps;
      n = n * (1 - spec.angular) + q * spec.angular;
    }
    const d = 1 + n * spec.amp * 2;
    push[i] = n;

    let x = dx * d * spec.sx;
    let y = dy * d * spec.sy;
    let z = dz * d * spec.sz;

    // Flat bottom. Everything below the cut plane collapses onto it, which
    // gives the rock a footprint instead of a point of contact.
    const cut = -spec.sy * (1 - spec.bury * 2);
    if (y < cut) y = cut;

    displaced[i * 3] = x;
    displaced[i * 3 + 1] = y;
    displaced[i * 3 + 2] = z;
  }

  let minY = Infinity;
  let maxY = -Infinity;
  let maxR = 0;
  for (let i = 0; i < count; i++) {
    const y = displaced[i * 3 + 1]!;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
    const r = Math.hypot(displaced[i * 3]!, displaced[i * 3 + 2]!);
    if (r > maxR) maxR = r;
  }
  const lift = -minY;

  const scale = spec.size;
  for (let f = 0; f < count; f += 3) {
    _a.fromArray(displaced, f * 3);
    _b.fromArray(displaced, (f + 1) * 3);
    _c.fromArray(displaced, (f + 2) * 3);
    _e.subVectors(_a, _b);
    _n.subVectors(_c, _b).cross(_e).normalize();

    const ax = Math.abs(_n.x);
    const ay = Math.abs(_n.y);
    const az = Math.abs(_n.z);
    const axis = ay > ax && ay > az ? 1 : ax > az ? 0 : 2;

    for (let k = 0; k < 3; k++) {
      const i = f + k;
      const x = displaced[i * 3]! * scale;
      const y = (displaced[i * 3 + 1]! + lift) * scale;
      const z = displaced[i * 3 + 2]! * scale;
      pos[i * 3] = x;
      pos[i * 3 + 1] = y;
      pos[i * 3 + 2] = z;

      // Triplanar UV at roughly 1 m per texture repeat.
      const u = axis === 0 ? z : x;
      const v = axis === 1 ? z : y;
      uv[i * 2] = u * 0.55;
      uv[i * 2 + 1] = v * 0.55;

      // Shading: up-facing lifts, inward displacement darkens.
      const upness = clamp01(_n.y * 0.5 + 0.5);
      const cavity = clamp01(0.5 + push[i]! * 1.8);
      _col.copy(ROCK_GREY).lerp(ROCK_WARM, clamp01(fbm3(x * 0.7, y * 0.7, z * 0.7, seedB, 2)));
      // Cavity darkening stays strong — it is relief the lighting cannot know
      // about. The up-facing term is deliberately weak: real directional light
      // already shades a downward face, and baking a second helping of the same
      // thing into the albedo is what sends a backlit boulder to black.
      _col.lerp(ROCK_SHADE, (1 - cavity) * 0.5 + (1 - upness) * 0.12);
      _col.multiplyScalar(spec.tint * (0.9 + 0.2 * cavity));
      if (spec.moss > 0) {
        const m =
          spec.moss *
          clamp01((_n.y - 0.35) * 2.2) *
          clamp01(fbm3(x * 1.3 + 11, y * 1.3, z * 1.3, seedB + 7, 2) * 2.1 - 0.55);
        if (m > 0) _col.lerp(LICHEN, Math.min(LICHEN_MAX, m));
      }
      col[i * 3] = _col.r;
      col[i * 3 + 1] = _col.g;
      col[i * 3 + 2] = _col.b;
    }
  }

  src.dispose();

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
  geo.setAttribute('aFlex', new THREE.BufferAttribute(new Float32Array(count), 1));
  geo.computeVertexNormals();
  geo.computeBoundingBox();
  geo.computeBoundingSphere();

  // Unique points for the convex hull. Quantising to a centimetre folds the
  // duplicated non-indexed vertices back together.
  const seen = new Set<string>();
  const hullPts: number[] = [];
  for (let i = 0; i < count; i++) {
    const x = pos[i * 3]!;
    const y = pos[i * 3 + 1]!;
    const z = pos[i * 3 + 2]!;
    const key = `${Math.round(x * 100)},${Math.round(y * 100)},${Math.round(z * 100)}`;
    if (seen.has(key)) continue;
    seen.add(key);
    hullPts.push(x, y, z);
  }

  return {
    geo,
    hull: new Float32Array(hullPts),
    height: (maxY + lift) * scale,
    radius: maxR * scale,
  };
}

function specsFor(cls: RockClass): RockSpec[] {
  return cls === 'boulder' ? BOULDERS : cls === 'stone' ? STONES : SCREE;
}

/**
 * Rock models for one size class. Boulders get two LODs and a convex-hull
 * collider; stones and scree get one LOD and, being ankle height, no collider
 * at all — a thousand static hulls that only ever scuff a tyre is not a trade
 * worth making.
 */
export function buildRockModels(cls: RockClass, seed: number): ScatterModel[] {
  const specs = specsFor(cls);
  return specs.map((spec, i) => {
    const near = buildRockGeometry(spec, new Rng(seed + i * 3221), cls === 'boulder' ? 1 : 0);
    const lods: THREE.BufferGeometry[] = [near.geo];
    if (cls === 'boulder') {
      lods.push(buildRockGeometry(spec, new Rng(seed + i * 3221), 0).geo);
    }
    const collider: ColliderShape | null =
      cls === 'boulder' ? { kind: 'hull', points: near.hull } : null;
    return {
      name: `${cls}-${spec.name}`,
      lods,
      height: near.height,
      radius: near.radius,
      spacing: Math.max(0.4, near.radius * 0.9),
      collider,
    };
  });
}
