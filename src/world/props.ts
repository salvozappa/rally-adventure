/**
 * Trail furniture — the man-made dressing along the driving line.
 *
 * These are the strongest speed cues in the whole world. A tree 40 m off the
 * track barely moves across the screen; a marker post 2 m from the wheel is a
 * strobe. Dense, low, close-in roadside objects are exactly what made the 1997
 * rally games feel fast, and they cost almost nothing because there are only a
 * few hundred of them in the entire level.
 *
 * `role` is what `Scatter` places by: it walks the authored route polylines and
 * asks for a marker here, a chicane there, a wreck once in a while.
 */

import * as THREE from 'three';
import { MeshBuilder, PROP_TILE, Rng, type UvTile } from './scatterTextures';
import type { ColliderShape, ScatterModel } from './vegetation';

export type PropRole =
  | 'marker'
  | 'stripe'
  | 'tyres'
  | 'bale'
  | 'fence'
  | 'sign'
  | 'wreck'
  | 'barrier'
  | 'drum';

export interface PropModel extends ScatterModel {
  readonly role: PropRole;
  /** 0 = stays vertical, 1 = fully follows the ground normal. */
  readonly align: number;
}

const WHITE = new THREE.Color(0xffffff);
const SHADE = new THREE.Color(0x8e8e8e);
const DEEP = new THREE.Color(0x5a5a5a);
const _c = new THREE.Color();

function tint(base: THREE.Color, k: number): THREE.Color {
  return _c.copy(base).multiplyScalar(k).clone();
}

/* -------------------------------------------------------------------------- */
/* Primitives                                                                 */
/* -------------------------------------------------------------------------- */

/** Squat cylinder used for posts and drums, base at y = 0. */
function post(
  mb: MeshBuilder,
  r: number,
  h: number,
  seg: number,
  tile: UvTile,
  c0: THREE.Color,
  c1: THREE.Color,
): void {
  mb.cylinder(r, r * 0.94, h, seg, tile, c0, c1, true);
}

const TYRE = new THREE.TorusGeometry(0.34, 0.13, 4, 9);
TYRE.rotateX(Math.PI * 0.5);

function tyre(mb: MeshBuilder, rng: Rng, y: number, scale: number): void {
  mb.push();
  mb.translate(rng.range(-0.05, 0.05), y, rng.range(-0.05, 0.05));
  mb.rotateY(rng.next() * Math.PI * 2);
  mb.scale(scale, scale, scale);
  mb.addGeometry(TYRE, PROP_TILE.rubber, (_x, yy) => tint(WHITE, 0.75 + yy * 0.5));
  mb.pop();
}

/* -------------------------------------------------------------------------- */
/* Builders                                                                   */
/* -------------------------------------------------------------------------- */

function buildMarker(rng: Rng): THREE.BufferGeometry {
  const mb = new MeshBuilder();
  const h = rng.range(1.05, 1.45);
  mb.push();
  mb.rotateY(rng.next() * Math.PI * 2);
  mb.rotateZ(rng.range(-0.09, 0.09));
  mb.box(0.085, h, 0.085, PROP_TILE.wood, WHITE, true);
  // Painted cap — the bit that actually reads at 100 km/h.
  mb.push();
  mb.translate(0, h - 0.16, 0);
  mb.box(0.1, 0.3, 0.1, PROP_TILE.stripe, WHITE, true);
  mb.pop();
  mb.pop();
  return mb.build(true);
}

function buildStripePole(rng: Rng): THREE.BufferGeometry {
  const mb = new MeshBuilder();
  const h = rng.range(1.5, 1.9);
  mb.push();
  mb.rotateZ(rng.range(-0.06, 0.06));
  post(mb, 0.055, h, 6, PROP_TILE.stripe, tint(WHITE, 0.9), WHITE);
  mb.pop();
  // Concrete-ish foot so it does not look pushed into thin air.
  mb.box(0.28, 0.1, 0.28, PROP_TILE.metal, SHADE, true);
  return mb.build(true);
}

function buildTyreStack(rng: Rng): THREE.BufferGeometry {
  const mb = new MeshBuilder();
  const n = rng.int(2) + 3;
  for (let i = 0; i < n; i++) tyre(mb, rng, 0.13 + i * 0.24, rng.range(0.92, 1.06));
  return mb.build(true);
}

function buildBale(rng: Rng): THREE.BufferGeometry {
  const mb = new MeshBuilder();
  const r = rng.range(0.55, 0.68);
  const len = rng.range(1.1, 1.35);
  mb.push();
  mb.translate(-len * 0.5, r, 0);
  mb.rotateZ(-Math.PI * 0.5);
  mb.cylinder(r, r, len, 10, PROP_TILE.hay, tint(WHITE, 0.8), WHITE, true, 0.03, rng);
  mb.pop();
  // Baler twine.
  for (let i = 0; i < 2; i++) {
    mb.push();
    mb.translate((i === 0 ? -1 : 1) * len * 0.25, r, 0);
    mb.rotateZ(-Math.PI * 0.5);
    mb.cylinder(r * 1.02, r * 1.02, 0.035, 10, PROP_TILE.wood, DEEP, DEEP, false);
    mb.pop();
  }
  return mb.build(true);
}

function buildFence(rng: Rng): THREE.BufferGeometry {
  const mb = new MeshBuilder();
  const span = 4.4;
  const h = 1.15;
  for (let i = 0; i < 3; i++) {
    mb.push();
    mb.translate(-span * 0.5 + (span * i) / 2, 0, 0);
    mb.rotateZ(rng.range(-0.05, 0.05));
    mb.box(0.11, h * rng.range(0.94, 1.06), 0.11, PROP_TILE.wood, tint(WHITE, 0.92), true);
    mb.pop();
  }
  for (let i = 0; i < 2; i++) {
    mb.push();
    mb.translate(0, h * (0.45 + i * 0.42), 0);
    mb.rotateZ(rng.range(-0.02, 0.02));
    mb.box(span, 0.11, 0.055, PROP_TILE.wood, WHITE);
    mb.pop();
  }
  return mb.build(true);
}

function buildSign(rng: Rng): THREE.BufferGeometry {
  const mb = new MeshBuilder();
  const h = rng.range(1.6, 2.0);
  post(mb, 0.055, h, 5, PROP_TILE.wood, tint(WHITE, 0.85), WHITE);
  mb.push();
  mb.translate(0, h - 0.32, 0);
  mb.rotateY(rng.range(-0.25, 0.25));
  mb.box(0.95, 0.62, 0.05, PROP_TILE.sign, WHITE);
  mb.pop();
  return mb.build(true);
}

function buildBarrier(rng: Rng): THREE.BufferGeometry {
  const mb = new MeshBuilder();
  const span = 2.6;
  for (let i = 0; i < 2; i++) {
    mb.push();
    mb.translate(-span * 0.5 + span * i, 0, 0);
    mb.box(0.11, 0.95, 0.11, PROP_TILE.wood, tint(WHITE, 0.9), true);
    mb.pop();
  }
  mb.push();
  mb.translate(0, 0.72, 0);
  mb.rotateZ(rng.range(-0.03, 0.03));
  mb.box(span + 0.2, 0.24, 0.06, PROP_TILE.stripe, WHITE);
  mb.pop();
  return mb.build(true);
}

function buildDrum(rng: Rng): THREE.BufferGeometry {
  const mb = new MeshBuilder();
  const h = 0.88;
  mb.push();
  mb.rotateZ(rng.range(-0.04, 0.04));
  post(mb, 0.29, h, 9, PROP_TILE.rust, tint(WHITE, 0.8), WHITE);
  for (let i = 0; i < 2; i++) {
    mb.push();
    mb.translate(0, h * (0.3 + i * 0.4), 0);
    mb.cylinder(0.31, 0.31, 0.05, 9, PROP_TILE.metal, SHADE, WHITE, false);
    mb.pop();
  }
  mb.pop();
  return mb.build(true);
}

/**
 * A burnt-out shell at the side of the road. Crude on purpose: three boxes and
 * a pair of collapsed wheels read as "wreck" from a moving car, and anything
 * more detailed would compete with the player's own vehicle.
 */
function buildWreck(rng: Rng): THREE.BufferGeometry {
  const mb = new MeshBuilder();
  mb.push();
  mb.rotateY(rng.range(-0.4, 0.4));
  mb.rotateZ(rng.range(-0.13, 0.13));

  mb.push();
  mb.translate(0, 0.42, 0);
  mb.box(3.7, 0.62, 1.62, PROP_TILE.rust, tint(WHITE, 0.95));
  mb.pop();

  mb.push();
  mb.translate(-0.15, 0.95, 0);
  mb.rotateZ(rng.range(-0.05, 0.05));
  mb.box(1.85, 0.62, 1.48, PROP_TILE.paint, tint(WHITE, 0.88));
  mb.pop();

  // Bonnet, sprung open.
  mb.push();
  mb.translate(1.45, 0.78, 0);
  mb.rotateZ(rng.range(0.35, 0.75));
  mb.box(1.2, 0.06, 1.4, PROP_TILE.paint, tint(WHITE, 0.8));
  mb.pop();

  for (let i = 0; i < 4; i++) {
    const fx = i < 2 ? 1.25 : -1.25;
    const fz = i % 2 === 0 ? 0.82 : -0.82;
    if (rng.bool(0.25)) continue; // some wheels are simply gone
    mb.push();
    mb.translate(fx, 0.22, fz);
    mb.rotateX(Math.PI * 0.5);
    mb.rotateY(rng.range(-0.3, 0.3));
    mb.scale(0.62, 0.62, 0.62);
    mb.addGeometry(TYRE, PROP_TILE.rubber, () => tint(WHITE, 0.7));
    mb.pop();
  }
  mb.pop();
  return mb.build(true);
}

/* -------------------------------------------------------------------------- */
/* Model table                                                                */
/* -------------------------------------------------------------------------- */

interface PropDef {
  role: PropRole;
  name: string;
  build: (rng: Rng) => THREE.BufferGeometry;
  collider: (h: number, r: number) => ColliderShape | null;
  align: number;
  spacing: number;
}

const DEFS: PropDef[] = [
  { role: 'marker', name: 'marker-a', build: buildMarker, collider: () => null, align: 0.7, spacing: 0.4 },
  { role: 'marker', name: 'marker-b', build: buildMarker, collider: () => null, align: 0.7, spacing: 0.4 },
  { role: 'marker', name: 'marker-c', build: buildMarker, collider: () => null, align: 0.7, spacing: 0.4 },
  { role: 'stripe', name: 'stripe-pole', build: buildStripePole, collider: () => null, align: 0.3, spacing: 0.5 },
  {
    role: 'tyres',
    name: 'tyre-stack',
    build: buildTyreStack,
    collider: (h) => ({ kind: 'cylinder', radius: 0.48, halfHeight: h * 0.5, y: h * 0.5 }),
    align: 0.9,
    spacing: 1.1,
  },
  {
    role: 'bale',
    name: 'hay-bale',
    build: buildBale,
    collider: (h, r) => ({ kind: 'cuboid', hx: r, hy: h * 0.5, hz: h * 0.5, y: h * 0.5 }),
    align: 0.9,
    spacing: 1.6,
  },
  {
    role: 'fence',
    name: 'fence-run',
    build: buildFence,
    collider: (h) => ({ kind: 'cuboid', hx: 2.2, hy: h * 0.5, hz: 0.09, y: h * 0.5 }),
    align: 0.8,
    spacing: 2.4,
  },
  {
    role: 'sign',
    name: 'signpost',
    build: buildSign,
    collider: (h) => ({ kind: 'cylinder', radius: 0.1, halfHeight: h * 0.5, y: h * 0.5 }),
    align: 0.4,
    spacing: 1.0,
  },
  {
    role: 'barrier',
    name: 'plank-barrier',
    build: buildBarrier,
    collider: (h) => ({ kind: 'cuboid', hx: 1.4, hy: h * 0.5, hz: 0.1, y: h * 0.5 }),
    align: 0.85,
    spacing: 1.8,
  },
  {
    role: 'drum',
    name: 'oil-drum',
    build: buildDrum,
    collider: (h) => ({ kind: 'cylinder', radius: 0.31, halfHeight: h * 0.5, y: h * 0.5 }),
    align: 0.9,
    spacing: 0.9,
  },
  {
    role: 'wreck',
    name: 'wreck',
    build: buildWreck,
    collider: () => ({ kind: 'cuboid', hx: 1.9, hy: 0.6, hz: 0.85, y: 0.6 }),
    align: 0.95,
    spacing: 3.2,
  },
];

export function buildPropModels(seed: number): PropModel[] {
  return DEFS.map((def, i) => {
    const geo = def.build(new Rng(seed + i * 5231));
    geo.computeBoundingBox();
    const bb = geo.boundingBox!;
    const dy = -bb.min.y;
    if (Math.abs(dy) > 1e-4) {
      geo.translate(0, dy, 0);
      geo.computeBoundingBox();
    }
    const box = geo.boundingBox!;
    const height = box.max.y;
    const radius = Math.max(Math.abs(box.min.x), box.max.x, Math.abs(box.min.z), box.max.z);
    geo.setAttribute(
      'aFlex',
      new THREE.BufferAttribute(new Float32Array((geo.getAttribute('position') as THREE.BufferAttribute).count), 1),
    );
    geo.computeBoundingSphere();
    return {
      name: def.name,
      role: def.role,
      align: def.align,
      lods: [geo],
      height,
      radius,
      spacing: def.spacing,
      collider: def.collider(height, radius),
    };
  });
}
