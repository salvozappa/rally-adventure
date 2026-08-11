/**
 * Procedural vegetation: conifers, broadleaf trees, bushes, deadwood and grass.
 *
 * Everything here is *model* work — geometry only, no placement, no materials.
 * `Scatter` decides where the models go and what they are made of; this module
 * only guarantees a few invariants that placement relies on:
 *
 *   - the base of every model sits at y = 0 and the model is centred on the
 *     origin in XZ, so an instance matrix is just translate * rotate * scale,
 *   - geometry is built at **unit scale in metres** and reports its own height
 *     and radius, so the scatter can size colliders and impostors from it,
 *   - every geometry carries an `aFlex` attribute, 0 at the planted base and 1
 *     at the parts the wind is allowed to move. The vertex shader multiplies
 *     its sway by this, which is what keeps trunks still while canopies move.
 *
 * All geometry is non-indexed and flat-shaded. Faceted foliage is the correct
 * look for the 1997 target and it means normals never have to be authored.
 *
 * Polygon budget per LOD 0 model: ~150-220 triangles for a tree, ~110 for a
 * bush. That is generous for the era but the whole world is instanced, so the
 * cost that matters is draw calls, not triangles.
 */

import * as THREE from 'three';
import {
  CANOPY_TILE,
  MeshBuilder,
  Rng,
  clamp01,
  type UvTile,
} from './scatterTextures';

/* -------------------------------------------------------------------------- */
/* Shared model contract                                                      */
/* -------------------------------------------------------------------------- */

/**
 * A collider Rapier can build without knowing anything about the mesh. Only
 * objects big enough to stop or unsettle a car get one; see `Scatter`.
 */
export type ColliderShape =
  | { kind: 'cylinder'; radius: number; halfHeight: number; y: number }
  | { kind: 'ball'; radius: number; y: number }
  | { kind: 'cuboid'; hx: number; hy: number; hz: number; y: number }
  | { kind: 'hull'; points: Float32Array };

/** One scatterable thing, at unit scale, with its LOD chain. */
export interface ScatterModel {
  readonly name: string;
  /** Index 0 is full detail. Later entries are progressively cheaper. */
  readonly lods: THREE.BufferGeometry[];
  /** Overall height in metres at scale 1. */
  readonly height: number;
  /** Horizontal half-extent in metres at scale 1 — impostor width, spacing. */
  readonly radius: number;
  /** Null when the object is too small to be worth a static collider. */
  readonly collider: ColliderShape | null;
  /** Minimum spacing to any other large object, metres at scale 1. */
  readonly spacing: number;
}

/* -------------------------------------------------------------------------- */
/* Palette                                                                    */
/* -------------------------------------------------------------------------- */

/**
 * Species palette — and the **albedo of record** for everything in this file.
 *
 * These are not shading tints. The canopy atlas is a detail map averaging about
 * 0.7 in linear light (see `scatterTextures`), so the rendered albedo is roughly
 * `0.7 × the value below` and everything after that is lighting. Author these as
 * the colour you want the leaf to *be* in open sun, one stop up — never as a
 * dark "foliage green". Paired with a dark texture, dark greens here are exactly
 * what produced the black-spike forest: two albedos multiplied is not an albedo.
 *
 * Hue target is Screamer 4x4's dry olive country, not alpine emerald.
 */
const C = {
  barkDark: new THREE.Color(0x584434),
  barkMid: new THREE.Color(0x6f5b42),
  barkPale: new THREE.Color(0x8f7c60),
  barkAspen: new THREE.Color(0x8f8e78),
  barkAspenTop: new THREE.Color(0xa8a790),
  barkDead: new THREE.Color(0x8b7e69),
  needleDark: new THREE.Color(0x3f5432),
  needleMid: new THREE.Color(0x63834a),
  needleLight: new THREE.Color(0x8ea85c),
  needleBlue: new THREE.Color(0x4f6d5c),
  leafDark: new THREE.Color(0x4a6033),
  leafMid: new THREE.Color(0x6f8f47),
  leafLight: new THREE.Color(0x9fb463),
  leafGold: new THREE.Color(0xb5a851),
  dryDark: new THREE.Color(0x6b6440),
  dryLight: new THREE.Color(0xb0a26c),
  moss: new THREE.Color(0x6b7f3e),
  // Grass is authored a good deal lighter than it looks like it should be. It
  // receives shadow but its normals point at the sky, so a tuft under a tree
  // loses the sun and keeps nothing else; at a leaf-green albedo that lands on
  // black, and a shadow full of black tufts reads as dirt, not as grass. Lifted
  // to a pale khaki it stays legible in shade and matches the sunlit ground
  // rather than sitting on top of it as bright confetti.
  grassBase: new THREE.Color(0x86864f),
  grassTip: new THREE.Color(0xaba770),
  grassDry: new THREE.Color(0xb29c62),
} as const;

/* Shared blob primitives. Built once at import; they are only read from. */
const ICO0 = new THREE.IcosahedronGeometry(1, 0);
const ICO1 = new THREE.IcosahedronGeometry(1, 1);

/* -------------------------------------------------------------------------- */
/* Build helpers                                                              */
/* -------------------------------------------------------------------------- */

const _tmpColor = new THREE.Color();

/**
 * Every LOD of one model is built from the same seed, and every level must
 * therefore draw from that seed in the *same order* or the two LODs are not the
 * same tree — they are two different trees that swap places as you approach.
 *
 * That matters more than it sounds. The LOD cross-fade dissolves one mesh into
 * the other through a screen-door pattern, and the pattern only hides the swap
 * if the two silhouettes agree. When they do not, half the pixels of whichever
 * parts do not overlap are simply missing, and a tree at the transition
 * distance appears shot through with holes.
 *
 * The trap is that the geometry helpers themselves consume randomness in
 * proportion to their segment count — a 7-sided trunk draws more numbers than a
 * 5-sided one — so passing the shape rng straight to `MeshBuilder.cylinder`
 * silently desynchronises everything built after it. Shape decisions therefore
 * take the shared stream and tessellation jitter takes a fork of it, which
 * leaves the parent stream untouched no matter how many segments are drawn.
 */
function jitterStream(rng: Rng, salt: number): Rng {
  return rng.fork(salt);
}

/**
 * Wind flex weight per vertex: zero over the planted part of the model, then
 * easing to 1 at the top. `bare` is the fraction of the height that stays put.
 */
function addFlex(geo: THREE.BufferGeometry, height: number, bare: number, power = 1.5): THREE.BufferGeometry {
  const pos = geo.getAttribute('position') as THREE.BufferAttribute;
  const flex = new Float32Array(pos.count);
  const y0 = height * bare;
  const span = Math.max(1e-3, height - y0);
  for (let i = 0; i < pos.count; i++) {
    flex[i] = Math.pow(clamp01((pos.getY(i) - y0) / span), power);
  }
  geo.setAttribute('aFlex', new THREE.BufferAttribute(flex, 1));
  return geo;
}

/** No wind at all — used for logs, stumps, rocks and props. */
function addRigidFlex(geo: THREE.BufferGeometry): THREE.BufferGeometry {
  const pos = geo.getAttribute('position') as THREE.BufferAttribute;
  geo.setAttribute('aFlex', new THREE.BufferAttribute(new Float32Array(pos.count), 1));
  return geo;
}

/** Model dimensions straight from the built geometry, so they can never lie. */
function measure(geo: THREE.BufferGeometry): { height: number; radius: number } {
  geo.computeBoundingBox();
  const bb = geo.boundingBox!;
  return {
    height: bb.max.y,
    radius: Math.max(Math.abs(bb.min.x), bb.max.x, Math.abs(bb.min.z), bb.max.z),
  };
}

/**
 * Silhouette compensation for the low-detail clump.
 *
 * A 20-face icosahedron and its once-subdivided version have their vertices on
 * the same sphere, but the coarse one's outline cuts across chords between
 * them, so it draws about 6% narrower. That is invisible on its own and very
 * visible during a LOD cross-fade: the two levels no longer cover the same
 * pixels, and the screen-door dither leaves the canopy looking like wire mesh
 * for the whole width of the transition band. Growing the coarse clump to the
 * finer one's apparent size costs nothing and makes the swap disappear.
 */
const ICO0_SILHOUETTE_MATCH = 1.06;

/**
 * A squashed, rotated icosahedron — the foliage clump primitive. Vertical
 * position inside the blob drives the colour so canopies are lit on top and
 * dark underneath without needing a single extra light.
 */
function blob(
  mb: MeshBuilder,
  rng: Rng,
  rx: number,
  ry: number,
  rz: number,
  dark: THREE.Color,
  light: THREE.Color,
  tile: UvTile,
  detail: 0 | 1,
): void {
  const k = detail === 1 ? 1 : ICO0_SILHOUETTE_MATCH;
  mb.push();
  mb.rotateY(rng.next() * Math.PI * 2);
  mb.rotateZ(rng.range(-0.25, 0.25));
  mb.scale(rx * k, ry * k, rz * k);
  mb.addGeometry(detail === 1 ? ICO1 : ICO0, tile, (_x, y, _z) =>
    _tmpColor.copy(dark).lerp(light, clamp01(0.42 + y * 0.62)),
  );
  mb.pop();
}

/**
 * One conifer branch whorl: a drooping, two-ring skirt of needles.
 *
 * The single-ring version this replaces was a paper fan — apex, rim, hub — and
 * from any distance it collapsed to a flat triangle, which is what made a stand
 * of them read as spikes rather than trees. Adding a shoulder ring at 55% of the
 * radius costs one more triangle strip and buys the two things a conifer needs:
 * a shaded upper surface that falls away from the leader, and a rim that hangs
 * *below* the branch it grows from, so the silhouette has depth from the side
 * and mass from below.
 *
 * Colour runs bright at the tips and dark into the hub — spruce reads as lit
 * needle ends against a dark interior, and no amount of extra lighting
 * substitutes for painting that in.
 */
function whorl(
  mb: MeshBuilder,
  rng: Rng,
  y: number,
  r: number,
  top: number,
  droop: number,
  seg: number,
  inner: THREE.Color,
  outer: THREE.Color,
  tile: UvTile,
  shoulders: boolean,
): void {
  const uc = tile.u0 + tile.w * 0.5;
  const vLo = tile.v0 + tile.h * 0.08;
  const vMid = tile.v0 + tile.h * 0.5;
  const vHi = tile.v0 + tile.h * 0.92;
  // Angular UV that wraps the tile a couple of times without a hard seam: the
  // atlas is noise, so all this has to do is avoid stretching one texel across
  // a whole branch.
  const uAt = (a: number) => tile.u0 + tile.w * (0.5 + 0.48 * Math.cos(a * 2.0));

  const apex = mb.vertex(0, y + top, 0, uc, vHi, inner);
  const hub = mb.vertex(0, y + top * 0.14, 0, uc, vLo, inner);

  const mid: number[] = [];
  const rim: number[] = [];
  const shoulder = _tmpColor.copy(inner).lerp(outer, 0.5).clone();
  // Per-vertex jitter is proportional to `seg`, so it runs on a fork.
  const jit = jitterStream(rng, 0x9f13 + seg);

  for (let i = 0; i < seg; i++) {
    const a = (i / seg) * Math.PI * 2 + jit.range(-0.11, 0.11);
    const ca = Math.cos(a);
    const sa = Math.sin(a);
    const rr = r * jit.range(0.72, 1.16);
    const yy = y - droop * jit.range(0.6, 1.35);
    _tmpColor.copy(outer).multiplyScalar(jit.range(0.86, 1.16));
    rim.push(mb.vertex(ca * rr, yy, sa * rr, uAt(a), vMid, _tmpColor));
    if (shoulders) {
      const rm = rr * jit.range(0.48, 0.62);
      _tmpColor.copy(shoulder).multiplyScalar(jit.range(0.9, 1.1));
      mid.push(mb.vertex(ca * rm, y + top * 0.5, sa * rm, uAt(a + 0.7), vHi, _tmpColor));
    }
  }

  for (let i = 0; i < seg; i++) {
    const j = (i + 1) % seg;
    if (shoulders) {
      mb.tri(apex, mid[j]!, mid[i]!);
      mb.quad(mid[i]!, mid[j]!, rim[j]!, rim[i]!);
    } else {
      mb.tri(apex, rim[j]!, rim[i]!);
    }
    // Underside, facing down: this is the mass you see from a car window.
    mb.tri(hub, rim[i]!, rim[j]!);
  }
}

/* -------------------------------------------------------------------------- */
/* Conifers                                                                   */
/* -------------------------------------------------------------------------- */

interface ConiferSpec {
  name: string;
  height: number;
  trunk: number;
  crown: number;
  whorls: number;
  bare: number;
  droop: number;
  needle: THREE.Color;
  needleTip: THREE.Color;
  bark: THREE.Color;
  /** Dead standing snag: bare branches, no needles. */
  snag?: boolean;
}

/**
 * Crown radius is the number that decides whether these read as trees.
 *
 * A 17 m spruce with a 3 m crown is a 1:6 sliver — accurate for a dense
 * even-aged plantation, wrong for the open country this level is, and it is
 * half of why the first build looked like antennae. Field-grown conifers
 * standing alone are closer to 1:2.5, so the crowns below are wide and the
 * heights modest. They are accents on ridgelines here, not a forest.
 */
const CONIFERS: ConiferSpec[] = [
  { name: 'spruce', height: 13.5, trunk: 0.36, crown: 4.4, whorls: 9, bare: 0.11, droop: 0.6, needle: C.needleDark, needleTip: C.needleLight, bark: C.barkDark },
  { name: 'fir-broad', height: 10.5, trunk: 0.42, crown: 4.6, whorls: 8, bare: 0.08, droop: 0.46, needle: C.needleMid, needleTip: C.needleLight, bark: C.barkMid },
  { name: 'pine-tall', height: 15.5, trunk: 0.46, crown: 4.5, whorls: 6, bare: 0.42, droop: 0.34, needle: C.needleMid, needleTip: C.needleLight, bark: C.barkPale },
  { name: 'pine-scrub', height: 6.2, trunk: 0.26, crown: 3.0, whorls: 5, bare: 0.14, droop: 0.36, needle: C.needleBlue, needleTip: C.needleMid, bark: C.barkMid },
  { name: 'spruce-blue', height: 12.0, trunk: 0.32, crown: 3.8, whorls: 9, bare: 0.1, droop: 0.66, needle: C.needleBlue, needleTip: C.needleMid, bark: C.barkDark },
  { name: 'sapling', height: 3.6, trunk: 0.14, crown: 1.5, whorls: 5, bare: 0.08, droop: 0.24, needle: C.needleMid, needleTip: C.needleLight, bark: C.barkMid },
  { name: 'snag', height: 9.0, trunk: 0.32, crown: 2.0, whorls: 6, bare: 0.3, droop: 0.0, needle: C.barkDead, needleTip: C.barkDead, bark: C.barkDead, snag: true },
];

function buildConifer(spec: ConiferSpec, rng: Rng, lod: number): THREE.BufferGeometry {
  const mb = new MeshBuilder();
  const H = spec.height;
  const seg = lod === 0 ? 7 : 5;
  const trunkTop = spec.snag ? 0.9 : 0.75;

  mb.cylinder(
    spec.trunk,
    spec.trunk * 0.16,
    H * trunkTop,
    seg,
    CANOPY_TILE.bark,
    spec.bark,
    _tmpColor.copy(spec.bark).multiplyScalar(1.25).clone(),
    false,
    0.09,
    jitterStream(rng, 0x71c0),
  );

  if (spec.snag) {
    // A dead standing tree: a handful of broken limbs, no canopy at all. It is
    // the cheapest possible "this forest has a history" cue.
    const limbs = 6;
    for (let i = 0; i < limbs; i++) {
      const t = 0.34 + (i / limbs) * 0.58;
      const len = spec.crown * (1.15 - t) * rng.range(0.6, 1.15);
      mb.push();
      mb.translate(0, H * t, 0);
      mb.rotateY(rng.next() * Math.PI * 2);
      mb.rotateX(rng.range(-1.25, -0.55));
      mb.cylinder(spec.trunk * 0.34, spec.trunk * 0.08, len, 4, CANOPY_TILE.bark, spec.bark, C.barkDead, false);
      mb.pop();
    }
    return finishTree(mb, H, 0.16);
  }

  // Whorl *count* is the silhouette, so it does not change with LOD — only the
  // number of sides per whorl and the shoulder ring do. Dropping layers at LOD1
  // changes the outline and the cross-fade then shows through it.
  const n = spec.whorls;
  const rseg = lod === 0 ? 8 : 5;
  const y0 = H * spec.bare;
  const span = H * (0.9 - spec.bare);
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1);
    // A little compression toward the top keeps the whorls dense where the
    // silhouette is thinnest instead of leaving a bald spire.
    const y = y0 + span * Math.pow(t, 1.08);
    // Gentle taper. The old `pow(1 - t*0.94, 0.78)` had shed 89% of the radius
    // by the top whorl, which is a cone, not a crown — the profile now keeps
    // real width through the upper third and only closes in the last layer.
    const r = spec.crown * Math.pow(1 - t * 0.82, 0.58) * rng.range(0.86, 1.12);
    const layer = (span / n) * rng.range(1.5, 2.2);
    const shade = 0.78 + 0.38 * t;
    _tmpColor.copy(spec.needleTip).multiplyScalar(shade);
    whorl(mb, rng, y, r, layer, spec.droop * layer, rseg, spec.needle, _tmpColor.clone(), CANOPY_TILE.needle, lod === 0);
  }
  // Leader spike, so the tree does not end in a flat disc.
  whorl(mb, rng, y0 + span, spec.crown * 0.22, H * 0.09, 0.02, lod === 0 ? 6 : 4, spec.needle, spec.needleTip, CANOPY_TILE.needle, false);

  return finishTree(mb, H, spec.bare * 0.9);
}

function finishTree(mb: MeshBuilder, height: number, bare: number): THREE.BufferGeometry {
  const geo = mb.build(true);
  addFlex(geo, height, bare, 1.6);
  return geo;
}

/* -------------------------------------------------------------------------- */
/* Broadleaf trees                                                            */
/* -------------------------------------------------------------------------- */

interface BroadleafSpec {
  name: string;
  height: number;
  trunk: number;
  /** Fraction of the height before the first fork. */
  fork: number;
  spread: number;
  blobR: number;
  bark: THREE.Color;
  barkTop: THREE.Color;
  leafDark: THREE.Color;
  leafLight: THREE.Color;
  tile: UvTile;
  /** Dead tree — branches only. */
  bare?: boolean;
  /** Branch tips droop (willow). */
  weep?: number;
}

/**
 * `fork` is the parameter that decides whether these read as trees or as
 * lollipops. It is the fraction of the height spent on bare trunk before the
 * first split, so a 16 m tree at fork 0.72 hangs two small leaf blobs on top of
 * an 11 m pole — which is precisely what the first build looked like from the
 * driver's seat. Field trees in the reference put the crown over roughly the
 * top two-thirds, so nothing here forks above 0.5 and `blobR` is sized so the
 * crown ends up at least as wide as the bare trunk beneath it is tall.
 */
const BROADLEAF: BroadleafSpec[] = [
  { name: 'aspen', height: 10.5, trunk: 0.24, fork: 0.42, spread: 0.5, blobR: 2.2, bark: C.barkAspen, barkTop: C.barkAspenTop, leafDark: C.leafDark, leafLight: C.leafLight, tile: CANOPY_TILE.leaf },
  { name: 'aspen-slim', height: 12.5, trunk: 0.2, fork: 0.5, spread: 0.4, blobR: 2.0, bark: C.barkMid, barkTop: C.barkPale, leafDark: C.leafDark, leafLight: C.leafLight, tile: CANOPY_TILE.leaf },
  { name: 'oak', height: 9.0, trunk: 0.55, fork: 0.28, spread: 0.8, blobR: 3.0, bark: C.barkDark, barkTop: C.barkMid, leafDark: C.leafDark, leafLight: C.leafMid, tile: CANOPY_TILE.leaf },
  { name: 'gold-birch', height: 9.5, trunk: 0.22, fork: 0.4, spread: 0.55, blobR: 2.3, bark: C.barkAspen, barkTop: C.barkAspenTop, leafDark: C.leafMid, leafLight: C.leafGold, tile: CANOPY_TILE.leaf },
  { name: 'willow', height: 7.5, trunk: 0.38, fork: 0.3, spread: 0.9, blobR: 2.4, bark: C.barkMid, barkTop: C.barkPale, leafDark: C.leafDark, leafLight: C.leafMid, tile: CANOPY_TILE.leaf, weep: 0.5 },
  { name: 'thorn', height: 4.6, trunk: 0.22, fork: 0.28, spread: 1.0, blobR: 1.7, bark: C.barkPale, barkTop: C.barkPale, leafDark: C.dryDark, leafLight: C.dryLight, tile: CANOPY_TILE.dry },
  { name: 'dead-oak', height: 7.0, trunk: 0.4, fork: 0.3, spread: 0.9, blobR: 0, bark: C.barkDead, barkTop: C.barkDead, leafDark: C.barkDead, leafLight: C.barkDead, tile: CANOPY_TILE.bark, bare: true },
];

/**
 * Recursive branch. Each level shortens and thins; the leaves go on at the
 * last level, which is what makes the canopy sit on the structure rather than
 * being a ball hovering over a stick.
 */
function branch(
  mb: MeshBuilder,
  rng: Rng,
  spec: BroadleafSpec,
  len: number,
  r: number,
  depth: number,
  maxDepth: number,
  lod: number,
): void {
  const seg = depth === 0 ? (lod === 0 ? 6 : 4) : 4;
  const r1 = r * 0.62;
  const c0 = _tmpColor.copy(spec.bark).clone();
  const c1 = _tmpColor.copy(spec.bark).lerp(spec.barkTop, 0.55 + depth * 0.15).clone();
  mb.cylinder(r, r1, len, seg, CANOPY_TILE.bark, c0, c1, false, 0.07, jitterStream(rng, 0x5b1 + depth));

  mb.push();
  mb.translate(0, len, 0);

  if (depth >= maxDepth) {
    if (!spec.bare && spec.blobR > 0) {
      const s = spec.blobR * rng.range(0.8, 1.3);
      blob(mb, rng, s, s * rng.range(0.62, 0.86), s, spec.leafDark, spec.leafLight, spec.tile, lod === 0 ? 1 : 0);
      // One cheap satellite clump offset off the branch tip. A single blob per
      // tip leaves a canopy of separate balls with sky between them; the
      // satellite is what fuses them into one rounded mass, and at 20 faces it
      // costs a quarter of what a second full-detail blob would.
      const off = s * rng.range(0.5, 0.85);
      const a = rng.next() * Math.PI * 2;
      mb.push();
      mb.translate(Math.cos(a) * off, s * rng.range(-0.28, 0.34), Math.sin(a) * off);
      const s2 = s * rng.range(0.55, 0.8);
      blob(mb, rng, s2, s2 * rng.range(0.62, 0.88), s2, spec.leafDark, spec.leafLight, spec.tile, 0);
      mb.pop();
    }
    mb.pop();
    return;
  }

  const kids = depth === 0 ? 3 : 2;
  const baseYaw = rng.next() * Math.PI * 2;
  for (let i = 0; i < kids; i++) {
    const yaw = baseYaw + (i / kids) * Math.PI * 2 + rng.range(-0.5, 0.5);
    const pitch = spec.spread * rng.range(0.55, 1.35) + (spec.weep ?? 0) * depth;
    mb.push();
    mb.rotateY(yaw);
    mb.rotateX(pitch);
    branch(mb, rng, spec, len * rng.range(0.55, 0.78), r1 * rng.range(0.72, 0.95), depth + 1, maxDepth, lod);
    mb.pop();
  }
  mb.pop();
}

function buildBroadleaf(spec: BroadleafSpec, rng: Rng, lod: number): THREE.BufferGeometry {
  const mb = new MeshBuilder();
  // Both LODs branch to the same depth: the branch tree *is* the crown outline,
  // so cutting a level off at LOD1 does not simplify the tree, it replaces it.
  // The saving comes from coarser trunks and 20-face leaf clumps instead of 80.
  branch(mb, rng, spec, spec.height * spec.fork, spec.trunk, 0, 2, lod);
  return finishTree(mb, spec.height, spec.fork * 0.55);
}

/* -------------------------------------------------------------------------- */
/* Bushes and scrub                                                           */
/* -------------------------------------------------------------------------- */

interface BushSpec {
  name: string;
  height: number;
  spread: number;
  clumps: number;
  dark: THREE.Color;
  light: THREE.Color;
  tile: UvTile;
  /** Visible woody stems poking out of the mass. */
  stems: number;
  /** 0 = ball, 1 = flat mat. */
  flat: number;
}

/**
 * Scrub is knee-to-waist height. A 2.1 m juniper mat scaled up by 1.5 is a
 * three-metre green dome, which at driver's eye reads as a mossy boulder rather
 * than as a bush — and next to actual boulders the two become impossible to
 * tell apart, which matters when one of them stops the car and the other does
 * not. Nothing here spreads past about 1.3 m at unit scale.
 */
const BUSHES: BushSpec[] = [
  { name: 'shrub', height: 1.2, spread: 0.95, clumps: 6, dark: C.leafDark, light: C.leafMid, tile: CANOPY_TILE.leaf, stems: 3, flat: 0.25 },
  { name: 'broom', height: 1.7, spread: 0.75, clumps: 5, dark: C.leafDark, light: C.leafLight, tile: CANOPY_TILE.leaf, stems: 5, flat: 0.0 },
  { name: 'sage', height: 0.7, spread: 1.0, clumps: 8, dark: C.dryDark, light: C.dryLight, tile: CANOPY_TILE.dry, stems: 2, flat: 0.5 },
  { name: 'thorn-scrub', height: 0.95, spread: 1.1, clumps: 7, dark: C.dryDark, light: C.leafMid, tile: CANOPY_TILE.dry, stems: 6, flat: 0.45 },
  { name: 'juniper-mat', height: 0.5, spread: 1.0, clumps: 9, dark: C.needleDark, light: C.needleMid, tile: CANOPY_TILE.needle, stems: 2, flat: 0.6 },
  { name: 'tussock', height: 0.85, spread: 0.6, clumps: 4, dark: C.dryDark, light: C.grassTip, tile: CANOPY_TILE.dry, stems: 4, flat: 0.1 },
  { name: 'dead-bush', height: 1.0, spread: 0.9, clumps: 3, dark: C.barkDead, light: C.dryLight, tile: CANOPY_TILE.dry, stems: 9, flat: 0.3 },
];

function buildBush(spec: BushSpec, rng: Rng, lod: number): THREE.BufferGeometry {
  const mb = new MeshBuilder();

  // The woody stems are what LOD1 drops: they are 3-sided twigs 4 cm across and
  // invisible past about fifteen metres, so losing them costs nothing visible
  // while the leaf clumps — which are the whole silhouette — stay identical
  // between levels and the cross-fade has nothing to reveal.
  const stems = lod === 0 ? spec.stems : 0;
  // Forked *before* the stem loop, which is the only LOD-dependent thing here:
  // taken after it, LOD1 would skip the stems' draws and land on a different
  // point in the stream, giving the two levels different clumps.
  const rng2 = jitterStream(rng, 0xb115);
  for (let i = 0; i < stems; i++) {
    mb.push();
    mb.rotateY(rng.next() * Math.PI * 2);
    mb.rotateX(rng.range(0.15, 0.7));
    mb.cylinder(0.045, 0.015, spec.height * rng.range(0.6, 1.05), 3, CANOPY_TILE.bark, C.barkDark, C.barkPale, false);
    mb.pop();
  }

  const n = spec.clumps;
  for (let i = 0; i < n; i++) {
    // Many small clumps rather than a few large ones. A 20-face icosahedron
    // scaled to most of the bush's radius is a smooth green solid that reads as
    // a mossy rock at driver's eye; the same volume broken into eight lumps
    // reads as foliage, for the same triangle count.
    const a = (i / n) * Math.PI * 2 + rng2.range(-0.6, 0.6);
    const d = spec.spread * rng2.range(0.05, 0.6);
    const s = spec.spread * rng2.range(0.3, 0.52);
    mb.push();
    mb.translate(Math.cos(a) * d, spec.height * rng2.range(0.3, 0.66), Math.sin(a) * d);
    blob(mb, rng2, s, s * (1 - spec.flat * 0.72) * rng2.range(0.7, 1.1), s, spec.dark, spec.light, spec.tile, 0);
    mb.pop();
  }

  const geo = mb.build(true);
  // Clip anything that ended up below ground, so a bush never floats.
  liftToGround(geo);
  addFlex(geo, spec.height, 0.05, 1.1);
  return geo;
}

/** Push the geometry up so its lowest vertex sits exactly on y = 0. */
function liftToGround(geo: THREE.BufferGeometry): void {
  geo.computeBoundingBox();
  const dy = -geo.boundingBox!.min.y;
  if (Math.abs(dy) > 1e-4) geo.translate(0, dy, 0);
  geo.computeBoundingBox();
  geo.computeBoundingSphere();
}

/* -------------------------------------------------------------------------- */
/* Deadwood                                                                   */
/* -------------------------------------------------------------------------- */

interface DeadwoodSpec {
  name: string;
  kind: 'log' | 'stump' | 'pile';
  length: number;
  radius: number;
}

const DEADWOOD: DeadwoodSpec[] = [
  { name: 'log-long', kind: 'log', length: 6.5, radius: 0.36 },
  { name: 'log-short', kind: 'log', length: 3.4, radius: 0.28 },
  { name: 'log-fat', kind: 'log', length: 4.6, radius: 0.52 },
  { name: 'stump-broad', kind: 'stump', length: 0.9, radius: 0.55 },
  { name: 'stump-tall', kind: 'stump', length: 1.6, radius: 0.34 },
  { name: 'branch-pile', kind: 'pile', length: 2.6, radius: 0.12 },
];

function buildDeadwood(spec: DeadwoodSpec, rng: Rng): THREE.BufferGeometry {
  const mb = new MeshBuilder();
  const wood = C.barkDead;
  const woodDark = _tmpColor.copy(C.barkDark).lerp(C.barkDead, 0.4).clone();

  if (spec.kind === 'log') {
    const r = spec.radius;
    mb.push();
    mb.translate(-spec.length * 0.5, r * 0.85, 0);
    // -90° about Z takes the builder's +Y to +X, so the trunk runs from
    // -length/2 to +length/2 and the model stays centred on its own origin. A
    // +90° here sends it to -X instead: the mesh lands a whole length away from
    // the branch stubs, the moss and — the part you can feel — the cuboid
    // collider, which is built centred.
    mb.rotateZ(-Math.PI * 0.5 + 0.03);
    mb.rotateY(rng.range(-0.15, 0.15));
    // Slight taper and a big wobble so it reads as a snapped trunk.
    mb.cylinder(r, r * 0.72, spec.length, 8, CANOPY_TILE.bark, woodDark, wood, true, 0.11, rng);
    mb.pop();
    // Broken branch stubs.
    for (let i = 0; i < 3; i++) {
      const t = rng.range(0.15, 0.85);
      mb.push();
      mb.translate(-spec.length * 0.5 + spec.length * t, r * 0.95, 0);
      mb.rotateY(rng.next() * Math.PI * 2);
      mb.rotateX(rng.range(-0.9, -0.3));
      mb.cylinder(r * 0.3, r * 0.1, r * rng.range(1.6, 3.4), 4, CANOPY_TILE.bark, wood, C.barkDead, false);
      mb.pop();
    }
    // Moss on the upper surface: one flattened blob riding the log's back.
    mb.push();
    mb.translate(0, r * 1.5, 0);
    mb.scale(1, 1, 1);
    blob(mb, rng, spec.length * 0.32, r * 0.3, r * 0.62, C.moss, C.leafMid, CANOPY_TILE.needle, 0);
    mb.pop();
  } else if (spec.kind === 'stump') {
    const r = spec.radius;
    const h = spec.length;
    mb.cylinder(r * 1.18, r, h, 9, CANOPY_TILE.bark, woodDark, wood, false, 0.08, rng);
    // Jagged snapped top rather than a clean disc.
    const ring: number[] = [];
    const seg = 9;
    for (let i = 0; i < seg; i++) {
      const a = (i / seg) * Math.PI * 2;
      ring.push(
        mb.vertex(
          Math.cos(a) * r,
          h + rng.range(-0.04, 0.16),
          Math.sin(a) * r,
          CANOPY_TILE.bark.u0 + CANOPY_TILE.bark.w * (0.5 + 0.4 * Math.cos(a)),
          CANOPY_TILE.bark.v0 + CANOPY_TILE.bark.h * (0.5 + 0.4 * Math.sin(a)),
          C.barkPale,
        ),
      );
    }
    const mid = mb.vertex(0, h + rng.range(0.04, 0.22), 0, CANOPY_TILE.bark.u0 + CANOPY_TILE.bark.w * 0.5, CANOPY_TILE.bark.v0 + CANOPY_TILE.bark.h * 0.5, C.barkPale);
    for (let i = 0; i < seg; i++) mb.tri(mid, ring[i]!, ring[(i + 1) % seg]!);
    // Root flares.
    for (let i = 0; i < 4; i++) {
      const a = (i / 4) * Math.PI * 2 + rng.range(-0.4, 0.4);
      mb.push();
      mb.translate(Math.cos(a) * r * 0.85, 0.02, Math.sin(a) * r * 0.85);
      mb.rotateY(-a);
      mb.rotateZ(-1.15);
      mb.cylinder(r * 0.34, r * 0.1, r * rng.range(0.9, 1.5), 4, CANOPY_TILE.bark, woodDark, wood, false);
      mb.pop();
    }
  } else {
    for (let i = 0; i < 7; i++) {
      mb.push();
      mb.translate(rng.range(-0.7, 0.7), spec.radius * rng.range(0.6, 2.2), rng.range(-0.7, 0.7));
      mb.rotateY(rng.next() * Math.PI * 2);
      mb.rotateZ(Math.PI * 0.5 + rng.range(-0.45, 0.45));
      mb.cylinder(spec.radius, spec.radius * 0.5, spec.length * rng.range(0.5, 1.2), 4, CANOPY_TILE.bark, woodDark, wood, false);
      mb.pop();
    }
  }

  const geo = mb.build(true);
  liftToGround(geo);
  addRigidFlex(geo);
  return geo;
}

/* -------------------------------------------------------------------------- */
/* Grass                                                                      */
/* -------------------------------------------------------------------------- */

/**
 * A grass tuft: three crossed, slightly arched cards. Alpha-tested, never
 * blended — blending a hundred thousand overlapping quads is a sorting problem
 * with no good answer, and a hard alpha edge is the right look here anyway.
 *
 * The cards lean away from each other so the tuft has volume from every angle,
 * and the top edge is offset so the silhouette is not a rectangle.
 *
 * Every normal is forced to straight up. A card's true normal is horizontal, so
 * lit honestly a field of grass is a field of near-black slivers standing on
 * bright ground — the light is coming from above and the geometry is edge-on to
 * it. Pointing the normals at the sky makes each tuft take the same light as the
 * ground it grows out of, which is what makes ground cover read as *cover*
 * rather than as thousands of dark spikes. It is the oldest trick in foliage
 * rendering and there is no substitute for it here.
 *
 * Each card is therefore emitted **twice, wound both ways**, and the material
 * renders front faces only. The obvious alternative — one card with
 * `side: DoubleSide` — undoes the whole trick: three flips the normal on back
 * faces, so the reverse of every blade is shaded as though it pointed at the
 * ground and comes out black. Half of every tuft in the world goes dark, and
 * from a moving car that reads as flickering soot on the hillside.
 */
export function buildGrassVariants(seed: number): THREE.BufferGeometry[] {
  const out: THREE.BufferGeometry[] = [];
  // Ankle height, not knee height. A tuft the size of a bush reads as a bush;
  // ground cover only works when there is a lot of it and each piece is small.
  // Variants 0-1 are the lush pair, 2-3 the pale dry pair — `Scatter` picks by
  // surface material and relies on that ordering.
  const shapes = [
    { cards: 3, w: 0.7, h: 0.2, arch: 0.06, tip: C.grassTip },
    { cards: 3, w: 0.58, h: 0.29, arch: 0.1, tip: C.grassTip },
    { cards: 3, w: 0.78, h: 0.15, arch: 0.04, tip: C.grassDry },
    { cards: 3, w: 0.64, h: 0.24, arch: 0.07, tip: C.grassDry },
  ];
  for (let v = 0; v < shapes.length; v++) {
    const s = shapes[v]!;
    const rng = new Rng(seed + v * 7717);
    const mb = new MeshBuilder();
    for (let i = 0; i < s.cards; i++) {
      const yaw = (i / s.cards) * Math.PI + rng.range(-0.2, 0.2);
      const w = s.w * rng.range(0.85, 1.15);
      const h = s.h * rng.range(0.8, 1.25);
      const lean = rng.range(-s.arch, s.arch);
      const ca = Math.cos(yaw);
      const sa = Math.sin(yaw);
      const base = _tmpColor.copy(C.grassBase).multiplyScalar(rng.range(0.8, 1.1)).clone();
      const tip = _tmpColor.copy(s.tip).multiplyScalar(rng.range(0.85, 1.15)).clone();
      const t = { u0: 0, v0: 0, w: 1, h: 1 };
      const a = mb.vertex(-ca * w * 0.5, 0, -sa * w * 0.5, t.u0, t.v0, base);
      const b = mb.vertex(ca * w * 0.5, 0, sa * w * 0.5, t.u0 + t.w, t.v0, base);
      const c = mb.vertex(ca * w * 0.42 + lean, h, sa * w * 0.42 + lean * 0.6, t.u0 + t.w, t.v0 + t.h, tip);
      const d = mb.vertex(-ca * w * 0.42 + lean, h, -sa * w * 0.42 + lean * 0.6, t.u0, t.v0 + t.h, tip);
      mb.quad(a, b, c, d);
      mb.quad(d, c, b, a);
    }
    const geo = mb.build(false);
    const nrm = geo.getAttribute('normal') as THREE.BufferAttribute;
    for (let i = 0; i < nrm.count; i++) nrm.setXYZ(i, 0, 1, 0);
    nrm.needsUpdate = true;
    addFlex(geo, s.h, 0.0, 1.15);
    out.push(geo);
  }
  return out;
}

/* -------------------------------------------------------------------------- */
/* Public builders                                                            */
/* -------------------------------------------------------------------------- */

function treeModel(
  name: string,
  lods: THREE.BufferGeometry[],
  trunkRadius: number,
  collides: boolean,
): ScatterModel {
  const m = measure(lods[0]!);
  return {
    name,
    lods,
    height: m.height,
    radius: m.radius,
    // Deliberately tight: real stands have overlapping canopies, and a
    // separation of one full crown radius reads as an orchard.
    spacing: Math.max(1.1, m.radius * 0.42),
    collider: collides
      ? { kind: 'cylinder', radius: Math.max(0.16, trunkRadius * 1.35), halfHeight: m.height * 0.5, y: m.height * 0.5 }
      : null,
  };
}

export function buildConiferModels(seed: number): ScatterModel[] {
  return CONIFERS.map((spec, i) => {
    const lods = [
      buildConifer(spec, new Rng(seed + i * 977), 0),
      buildConifer(spec, new Rng(seed + i * 977), 1),
    ];
    return treeModel(`conifer-${spec.name}`, lods, spec.trunk, spec.height > 5);
  });
}

export function buildBroadleafModels(seed: number): ScatterModel[] {
  return BROADLEAF.map((spec, i) => {
    const lods = [
      buildBroadleaf(spec, new Rng(seed + i * 1361), 0),
      buildBroadleaf(spec, new Rng(seed + i * 1361), 1),
    ];
    return treeModel(`broadleaf-${spec.name}`, lods, spec.trunk, true);
  });
}

export function buildBushModels(seed: number): ScatterModel[] {
  return BUSHES.map((spec, i) => {
    const lods = [
      buildBush(spec, new Rng(seed + i * 613), 0),
      buildBush(spec, new Rng(seed + i * 613), 1),
    ];
    const m = measure(lods[0]!);
    return {
      name: `bush-${spec.name}`,
      lods,
      height: m.height,
      radius: m.radius,
      spacing: Math.max(0.6, m.radius * 0.35),
      collider: null,
    };
  });
}

export function buildDeadwoodModels(seed: number): ScatterModel[] {
  return DEADWOOD.map((spec, i) => {
    const geo = buildDeadwood(spec, new Rng(seed + i * 421));
    const m = measure(geo);
    const collider: ColliderShape | null =
      spec.kind === 'log'
        ? { kind: 'cuboid', hx: spec.length * 0.5, hy: spec.radius, hz: spec.radius, y: spec.radius }
        : spec.kind === 'stump'
          ? { kind: 'cylinder', radius: spec.radius, halfHeight: spec.length * 0.5, y: spec.length * 0.5 }
          : null;
    return {
      name: `deadwood-${spec.name}`,
      lods: [geo],
      height: m.height,
      radius: m.radius,
      spacing: Math.max(1.2, m.radius * 0.8),
      collider,
    };
  });
}
