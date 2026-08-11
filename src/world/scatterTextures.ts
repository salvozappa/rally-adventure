/**
 * Procedural primitives for the world-dressing layer.
 *
 * This module is the lowest level of the scatter stack: seeded RNG, value
 * noise, a small mesh builder, and the fallback procedural textures used by
 * vegetation / rocks / props.
 *
 * ── The one rule that matters here ─────────────────────────────────────────
 * The canopy, rock and grass maps are **detail maps, not albedos**. The albedo
 * of every scattered object lives in its *vertex colours*; these textures only
 * modulate it. So each tile is drawn high-key — mean around sRGB 0.85, i.e.
 * linear ~0.7 — with the variation carried by contrast, not by darkness.
 *
 * Getting that wrong is expensive and silent. A `MeshLambertMaterial` with both
 * `map` and `vertexColors` multiplies the two, *after* both have been decoded
 * from sRGB to linear. Two plausible-looking dark greens — say a 0x2c4520
 * vertex colour and a needle texture averaging 0x31502a — multiply out to a
 * linear albedo of about 0.004. Nothing you can do with lights recovers that:
 * the tree renders black no matter how hard you light it. It was exactly this
 * that turned the conifer forest into black spikes.
 *
 * The prop atlas is the deliberate exception: props are drawn with white
 * vertex colours, so *that* atlas is the albedo and is authored at full range.
 *
 * Texture seam: if `src/render/textures.ts` lands with a richer bark/foliage/
 * rock set, whoever owns it calls `provideScatterTextures({...})` once at boot
 * and every scatter material picks them up. Until then these are used. Anything
 * supplied through that seam must obey the same detail-map rule.
 */
import * as THREE from 'three';

/* ------------------------------------------------------------------ */
/* Seeded RNG                                                          */
/* ------------------------------------------------------------------ */

/** mulberry32 — small, fast, good enough for placement and geometry jitter. */
export class Rng {
  private s: number;

  constructor(seed: number) {
    this.s = seed >>> 0;
  }

  /** 0 .. 1 */
  next(): number {
    this.s = (this.s + 0x6d2b79f5) >>> 0;
    let t = this.s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /** `lo` .. `hi` */
  range(lo: number, hi: number): number {
    return lo + (hi - lo) * this.next();
  }

  /** 0 .. n-1 */
  int(n: number): number {
    return Math.min(n - 1, (this.next() * n) | 0);
  }

  pick<T>(items: readonly T[]): T {
    return items[this.int(items.length)]!;
  }

  /** Roughly normal, mean 0, sd 1 (sum of three uniforms). */
  gauss(): number {
    return (this.next() + this.next() + this.next() - 1.5) * 1.4142;
  }

  bool(p: number): boolean {
    return this.next() < p;
  }

  /** A fresh independent stream, derived deterministically. */
  fork(salt: number): Rng {
    return new Rng((Math.imul(this.s ^ salt, 0x9e3779b1) ^ (salt << 7)) >>> 0);
  }
}

/* ------------------------------------------------------------------ */
/* Noise                                                               */
/* ------------------------------------------------------------------ */

function hash2i(x: number, y: number, seed: number): number {
  let h = (Math.imul(x, 374761393) + Math.imul(y, 668265263) + Math.imul(seed, 1274126177)) | 0;
  h = (h ^ (h >>> 13)) | 0;
  h = Math.imul(h, 1274126177) | 0;
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

function hash3i(x: number, y: number, z: number, seed: number): number {
  let h =
    (Math.imul(x, 374761393) + Math.imul(y, 668265263) + Math.imul(z, 2147483647) + Math.imul(seed, 1274126177)) | 0;
  h = (h ^ (h >>> 13)) | 0;
  h = Math.imul(h, 1274126177) | 0;
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

function smooth(t: number): number {
  return t * t * (3 - 2 * t);
}

/** Value noise, output 0..1. */
export function valueNoise2(x: number, y: number, seed: number): number {
  const xi = Math.floor(x);
  const yi = Math.floor(y);
  const xf = smooth(x - xi);
  const yf = smooth(y - yi);
  const a = hash2i(xi, yi, seed);
  const b = hash2i(xi + 1, yi, seed);
  const c = hash2i(xi, yi + 1, seed);
  const d = hash2i(xi + 1, yi + 1, seed);
  return (a + (b - a) * xf) * (1 - yf) + (c + (d - c) * xf) * yf;
}

/** Value noise in 3D, output 0..1. */
export function valueNoise3(x: number, y: number, z: number, seed: number): number {
  const xi = Math.floor(x);
  const yi = Math.floor(y);
  const zi = Math.floor(z);
  const xf = smooth(x - xi);
  const yf = smooth(y - yi);
  const zf = smooth(z - zi);
  let acc = 0;
  for (let dz = 0; dz < 2; dz++) {
    const wz = dz === 0 ? 1 - zf : zf;
    for (let dy = 0; dy < 2; dy++) {
      const wy = dy === 0 ? 1 - yf : yf;
      for (let dx = 0; dx < 2; dx++) {
        const wx = dx === 0 ? 1 - xf : xf;
        acc += hash3i(xi + dx, yi + dy, zi + dz, seed) * wx * wy * wz;
      }
    }
  }
  return acc;
}

/** Fractal value noise, output ~0..1. */
export function fbm2(x: number, y: number, seed: number, octaves = 4): number {
  let sum = 0;
  let amp = 0.5;
  let norm = 0;
  let fx = x;
  let fy = y;
  for (let i = 0; i < octaves; i++) {
    sum += valueNoise2(fx, fy, seed + i * 101) * amp;
    norm += amp;
    amp *= 0.5;
    fx *= 2.02;
    fy *= 2.02;
  }
  return sum / norm;
}

export function fbm3(x: number, y: number, z: number, seed: number, octaves = 3): number {
  let sum = 0;
  let amp = 0.5;
  let norm = 0;
  let s = 1;
  for (let i = 0; i < octaves; i++) {
    sum += valueNoise3(x * s, y * s, z * s, seed + i * 71) * amp;
    norm += amp;
    amp *= 0.5;
    s *= 2.03;
  }
  return sum / norm;
}

/** Ridged noise — sharp crests, good for scree fields and rock veins. */
export function ridged2(x: number, y: number, seed: number, octaves = 3): number {
  let sum = 0;
  let amp = 0.5;
  let norm = 0;
  let fx = x;
  let fy = y;
  for (let i = 0; i < octaves; i++) {
    const n = 1 - Math.abs(valueNoise2(fx, fy, seed + i * 313) * 2 - 1);
    sum += n * n * amp;
    norm += amp;
    amp *= 0.5;
    fx *= 2.05;
    fy *= 2.05;
  }
  return sum / norm;
}

/** Deterministic 0..1 from a pair of integers — used by the grass field. */
export function hashCell(x: number, y: number, seed: number): number {
  return hash2i(x, y, seed);
}

export function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

/** 0 at `edge0`, 1 at `edge1`, smooth between. */
export function smoothstep(edge0: number, edge1: number, v: number): number {
  if (edge0 === edge1) return v < edge0 ? 0 : 1;
  return smooth(clamp01((v - edge0) / (edge1 - edge0)));
}

/** Band-pass: 0 outside [lo,hi], 1 in the middle, `fade` wide shoulders. */
export function band(v: number, lo: number, hi: number, fade: number): number {
  return smoothstep(lo - fade, lo + fade, v) * (1 - smoothstep(hi - fade, hi + fade, v));
}

/* ------------------------------------------------------------------ */
/* Mesh builder                                                        */
/* ------------------------------------------------------------------ */

const _v = new THREE.Vector3();

/**
 * A tiny transform-stacked mesh builder. Normals are always derived at build
 * time (smooth for indexed, flat for non-indexed) so callers never write them.
 */
export class MeshBuilder {
  readonly pos: number[] = [];
  readonly uv: number[] = [];
  readonly col: number[] = [];
  readonly idx: number[] = [];

  private m = new THREE.Matrix4();
  private stack: THREE.Matrix4[] = [];

  push(): this {
    this.stack.push(this.m.clone());
    return this;
  }

  pop(): this {
    const m = this.stack.pop();
    if (m) this.m.copy(m);
    return this;
  }

  identity(): this {
    this.m.identity();
    return this;
  }

  translate(x: number, y: number, z: number): this {
    this.m.multiply(new THREE.Matrix4().makeTranslation(x, y, z));
    return this;
  }

  rotateX(a: number): this {
    this.m.multiply(new THREE.Matrix4().makeRotationX(a));
    return this;
  }

  rotateY(a: number): this {
    this.m.multiply(new THREE.Matrix4().makeRotationY(a));
    return this;
  }

  rotateZ(a: number): this {
    this.m.multiply(new THREE.Matrix4().makeRotationZ(a));
    return this;
  }

  scale(x: number, y: number, z: number): this {
    this.m.multiply(new THREE.Matrix4().makeScale(x, y, z));
    return this;
  }

  get vertexCount(): number {
    return this.pos.length / 3;
  }

  /** Append one vertex through the current transform. Returns its index. */
  vertex(x: number, y: number, z: number, u: number, v: number, c: THREE.Color): number {
    _v.set(x, y, z).applyMatrix4(this.m);
    this.pos.push(_v.x, _v.y, _v.z);
    this.uv.push(u, v);
    this.col.push(c.r, c.g, c.b);
    return this.pos.length / 3 - 1;
  }

  tri(a: number, b: number, c: number): void {
    this.idx.push(a, b, c);
  }

  quad(a: number, b: number, c: number, d: number): void {
    this.idx.push(a, b, c, a, c, d);
  }

  /**
   * Tapered cylinder along +Y from y=0 to y=h. `open` skips the caps.
   * UVs are mapped into `tile`.
   */
  cylinder(
    r0: number,
    r1: number,
    h: number,
    seg: number,
    tile: UvTile,
    c0: THREE.Color,
    c1: THREE.Color,
    caps = true,
    wobble = 0,
    rng?: Rng,
  ): void {
    const ring0: number[] = [];
    const ring1: number[] = [];
    for (let i = 0; i < seg; i++) {
      const t = i / seg;
      const a = t * Math.PI * 2;
      const w0 = wobble && rng ? 1 + rng.range(-wobble, wobble) : 1;
      const w1 = wobble && rng ? 1 + rng.range(-wobble, wobble) : 1;
      const u = tile.u0 + tile.w * (t * 2 % 1);
      ring0.push(
        this.vertex(Math.cos(a) * r0 * w0, 0, Math.sin(a) * r0 * w0, u, tile.v0 + tile.h * 0.02, c0),
      );
      ring1.push(
        this.vertex(Math.cos(a) * r1 * w1, h, Math.sin(a) * r1 * w1, u, tile.v0 + tile.h * 0.98, c1),
      );
    }
    for (let i = 0; i < seg; i++) {
      const j = (i + 1) % seg;
      this.quad(ring0[i]!, ring0[j]!, ring1[j]!, ring1[i]!);
    }
    if (caps) {
      const uc = tile.u0 + tile.w * 0.5;
      const vc = tile.v0 + tile.h * 0.5;
      if (r1 > 1e-4) {
        const top = this.vertex(0, h, 0, uc, vc, c1);
        for (let i = 0; i < seg; i++) this.tri(top, ring1[i]!, ring1[(i + 1) % seg]!);
      }
      if (r0 > 1e-4) {
        const bot = this.vertex(0, 0, 0, uc, vc, c0);
        for (let i = 0; i < seg; i++) this.tri(bot, ring0[(i + 1) % seg]!, ring0[i]!);
      }
    }
  }

  /** Axis-aligned box centred on the current origin. */
  box(w: number, h: number, d: number, tile: UvTile, c: THREE.Color, yBase = false): void {
    const x = w * 0.5;
    const z = d * 0.5;
    const y0 = yBase ? 0 : -h * 0.5;
    const y1 = yBase ? h : h * 0.5;
    const u0 = tile.u0 + tile.w * 0.03;
    const u1 = tile.u0 + tile.w * 0.97;
    const v0 = tile.v0 + tile.h * 0.03;
    const v1 = tile.v0 + tile.h * 0.97;
    const p: [number, number, number][] = [
      [-x, y0, -z],
      [x, y0, -z],
      [x, y0, z],
      [-x, y0, z],
      [-x, y1, -z],
      [x, y1, -z],
      [x, y1, z],
      [-x, y1, z],
    ];
    const faces: [number, number, number, number][] = [
      [0, 3, 2, 1],
      [4, 5, 6, 7],
      [0, 1, 5, 4],
      [1, 2, 6, 5],
      [2, 3, 7, 6],
      [3, 0, 4, 7],
    ];
    const uvs: [number, number][] = [
      [u0, v0],
      [u1, v0],
      [u1, v1],
      [u0, v1],
    ];
    for (const f of faces) {
      const ids: number[] = [];
      for (let i = 0; i < 4; i++) {
        const pt = p[f[i]!]!;
        const uvp = uvs[i]!;
        ids.push(this.vertex(pt[0], pt[1], pt[2], uvp[0], uvp[1], c));
      }
      this.quad(ids[0]!, ids[1]!, ids[2]!, ids[3]!);
    }
  }

  /** A flat quad in the XY plane, centred horizontally, base at y=0. */
  plane(w: number, h: number, tile: UvTile, c: THREE.Color, doubleSided = false): void {
    const x = w * 0.5;
    const a = this.vertex(-x, 0, 0, tile.u0, tile.v0, c);
    const b = this.vertex(x, 0, 0, tile.u0 + tile.w, tile.v0, c);
    const cc = this.vertex(x, h, 0, tile.u0 + tile.w, tile.v0 + tile.h, c);
    const d = this.vertex(-x, h, 0, tile.u0, tile.v0 + tile.h, c);
    this.quad(a, b, cc, d);
    if (doubleSided) this.quad(d, cc, b, a);
  }

  /** Merge an existing geometry through the current transform. */
  addGeometry(geo: THREE.BufferGeometry, tile: UvTile, color: (x: number, y: number, z: number) => THREE.Color): void {
    const src = geo.getAttribute('position') as THREE.BufferAttribute;
    const base = this.vertexCount;
    const tmp = new THREE.Color();
    for (let i = 0; i < src.count; i++) {
      const x = src.getX(i);
      const y = src.getY(i);
      const z = src.getZ(i);
      tmp.copy(color(x, y, z));
      const len = Math.hypot(x, y, z) || 1;
      const u = tile.u0 + tile.w * (0.5 + Math.atan2(z, x) / (Math.PI * 2));
      const v = tile.v0 + tile.h * (0.5 + (y / len) * 0.5);
      this.vertex(x, y, z, u, v, tmp);
    }
    const index = geo.getIndex();
    if (index) {
      for (let i = 0; i < index.count; i += 3) {
        this.tri(base + index.getX(i), base + index.getX(i + 1), base + index.getX(i + 2));
      }
    } else {
      for (let i = 0; i < src.count; i += 3) this.tri(base + i, base + i + 1, base + i + 2);
    }
  }

  build(flat: boolean): THREE.BufferGeometry {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(this.pos, 3));
    g.setAttribute('uv', new THREE.Float32BufferAttribute(this.uv, 2));
    g.setAttribute('color', new THREE.Float32BufferAttribute(this.col, 3));
    g.setIndex(this.idx);
    const out = flat ? g.toNonIndexed() : g;
    if (flat) g.dispose();
    out.computeVertexNormals();
    out.computeBoundingSphere();
    out.computeBoundingBox();
    return out;
  }
}

/* ------------------------------------------------------------------ */
/* Texture atlases                                                     */
/* ------------------------------------------------------------------ */

export interface UvTile {
  u0: number;
  v0: number;
  w: number;
  h: number;
}

function tile(u0: number, v0: number, w: number, h: number): UvTile {
  /* inset by half a texel-ish so neighbouring tiles never bleed */
  const i = 0.004;
  return { u0: u0 + i, v0: v0 + i, w: w - i * 2, h: h - i * 2 };
}

/** Canopy atlas layout (bark / conifer needles / broadleaf / dry scrub). */
export const CANOPY_TILE = {
  bark: tile(0, 0, 0.5, 0.5),
  needle: tile(0.5, 0, 0.5, 0.5),
  leaf: tile(0, 0.5, 0.5, 0.5),
  dry: tile(0.5, 0.5, 0.5, 0.5),
} as const;

/** Prop atlas layout — 4 x 2 tiles. */
export const PROP_TILE = {
  wood: tile(0, 0.5, 0.25, 0.5),
  stripe: tile(0.25, 0.5, 0.25, 0.5),
  rubber: tile(0.5, 0.5, 0.25, 0.5),
  hay: tile(0.75, 0.5, 0.25, 0.5),
  rust: tile(0, 0, 0.25, 0.5),
  metal: tile(0.25, 0, 0.25, 0.5),
  sign: tile(0.5, 0, 0.25, 0.5),
  paint: tile(0.75, 0, 0.25, 0.5),
} as const;

export const FULL_TILE: UvTile = { u0: 0, v0: 0, w: 1, h: 1 };

/** Rect in canvas pixels for a UV tile, accounting for the default flipY. */
function rect(t: { u0: number; v0: number; w: number; h: number }, size: number, aspectH = size) {
  return {
    x: Math.round(t.u0 * size),
    y: Math.round(aspectH - (t.v0 + t.h) * aspectH),
    w: Math.round(t.w * size),
    h: Math.round(t.h * aspectH),
  };
}

function canvas(w: number, h: number): { cv: HTMLCanvasElement; ctx: CanvasRenderingContext2D } {
  const cv = document.createElement('canvas');
  cv.width = w;
  cv.height = h;
  const ctx = cv.getContext('2d');
  if (!ctx) throw new Error('scatterTextures: 2D canvas unavailable');
  return { cv, ctx };
}

function rgb(r: number, g: number, b: number): string {
  return `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)})`;
}

/** Uniform noise fill inside a rect. */
function speckle(
  ctx: CanvasRenderingContext2D,
  r: { x: number; y: number; w: number; h: number },
  rng: Rng,
  base: [number, number, number],
  spread: number,
  count: number,
  minR: number,
  maxR: number,
): void {
  for (let i = 0; i < count; i++) {
    const k = rng.range(-spread, spread);
    ctx.fillStyle = rgb(clamp01(base[0] + k), clamp01(base[1] + k * 1.05), clamp01(base[2] + k * 0.85));
    const rad = rng.range(minR, maxR);
    ctx.beginPath();
    ctx.arc(r.x + rng.next() * r.w, r.y + rng.next() * r.h, rad, 0, Math.PI * 2);
    ctx.fill();
  }
}

function fillTile(
  ctx: CanvasRenderingContext2D,
  r: { x: number; y: number; w: number; h: number },
  c: [number, number, number],
): void {
  ctx.fillStyle = rgb(c[0], c[1], c[2]);
  ctx.fillRect(r.x, r.y, r.w, r.h);
}

/**
 * Mid-grey the detail tiles are drawn around. sRGB 0.86 decodes to a linear
 * ~0.71, so a tile at this value dims the vertex albedo by roughly a third —
 * enough headroom for the highlights to sit above it without the mean going
 * dark. Every detail tile must average close to this.
 */
const DETAIL_MID = 0.86;

/** A detail value with a hue push: `k` in stops around DETAIL_MID, `hue` a bias. */
function detail(k: number, hue: [number, number, number] = [0, 0, 0]): string {
  return rgb(
    clamp01(DETAIL_MID * k + hue[0]),
    clamp01(DETAIL_MID * k + hue[1]),
    clamp01(DETAIL_MID * k + hue[2]),
  );
}

/** Faintly warm, so trunks pick up a hint of brown even under a grey palette. */
const BARK_HUE: [number, number, number] = [0.035, 0.005, -0.03];

function drawBark(ctx: CanvasRenderingContext2D, r: { x: number; y: number; w: number; h: number }, rng: Rng): void {
  ctx.fillStyle = detail(1, BARK_HUE);
  ctx.fillRect(r.x, r.y, r.w, r.h);
  ctx.save();
  ctx.beginPath();
  ctx.rect(r.x, r.y, r.w, r.h);
  ctx.clip();
  /* vertical fissures — the ridges between them read as the lit side */
  for (let i = 0; i < 140; i++) {
    const x = r.x + rng.next() * r.w;
    ctx.strokeStyle = detail(rng.range(0.82, 1.16), BARK_HUE);
    ctx.lineWidth = rng.range(1, 5);
    ctx.beginPath();
    let y = r.y - 4;
    let cx = x;
    ctx.moveTo(cx, y);
    while (y < r.y + r.h + 4) {
      y += rng.range(6, 16);
      cx += rng.range(-3, 3);
      ctx.lineTo(cx, y);
    }
    ctx.stroke();
  }
  /* cracks: contrast, not blackness — a 0.5 factor is already a deep groove */
  for (let i = 0; i < 46; i++) {
    ctx.strokeStyle = detail(rng.range(0.5, 0.66), BARK_HUE);
    ctx.lineWidth = rng.range(1, 2.5);
    ctx.beginPath();
    let y = r.y - 4;
    let cx = r.x + rng.next() * r.w;
    ctx.moveTo(cx, y);
    while (y < r.y + r.h + 4) {
      y += rng.range(8, 20);
      cx += rng.range(-2.5, 2.5);
      ctx.lineTo(cx, y);
    }
    ctx.stroke();
  }
  ctx.restore();
}

/**
 * Foliage break-up: clumps of needles or leaves as a *luminance* pattern.
 *
 * `hue` is a small push away from neutral — just enough that the texture is not
 * sterile grey when it lands on bark-coloured or dry-scrub-coloured geometry.
 * The species colour itself comes from the model's vertex colours.
 */
function drawFoliage(
  ctx: CanvasRenderingContext2D,
  r: { x: number; y: number; w: number; h: number },
  rng: Rng,
  hue: [number, number, number],
  clumps: number,
): void {
  ctx.fillStyle = detail(1, hue);
  ctx.fillRect(r.x, r.y, r.w, r.h);
  ctx.save();
  ctx.beginPath();
  ctx.rect(r.x, r.y, r.w, r.h);
  ctx.clip();
  for (let i = 0; i < clumps; i++) {
    ctx.fillStyle = detail(rng.range(0.7, 1.17), hue);
    const x = r.x + rng.next() * r.w;
    const y = r.y + rng.next() * r.h;
    const rad = rng.range(2, 9);
    ctx.beginPath();
    ctx.ellipse(x, y, rad, rad * rng.range(0.4, 1), rng.next() * Math.PI, 0, Math.PI * 2);
    ctx.fill();
  }
  /* Shadow pockets between the clumps, so it does not read as flat mush. Small
     and shallow on purpose: a canopy blob is only a metre or two across on
     screen, so a big deep pocket does not read as depth, it reads as a hole
     punched in the leaves. */
  for (let i = 0; i < clumps * 0.3; i++) {
    ctx.fillStyle = detail(rng.range(0.66, 0.8), hue);
    ctx.beginPath();
    ctx.arc(r.x + rng.next() * r.w, r.y + rng.next() * r.h, rng.range(2.5, 7), 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function buildCanopyAtlas(): THREE.Texture {
  const S = 512;
  const { cv, ctx } = canvas(S, S);
  const rng = new Rng(0x51a7c0de);
  drawBark(ctx, rect(CANOPY_TILE.bark, S), rng);
  drawFoliage(ctx, rect(CANOPY_TILE.needle, S), rng, [-0.03, 0.015, -0.04], 900);
  drawFoliage(ctx, rect(CANOPY_TILE.leaf, S), rng, [-0.015, 0.02, -0.045], 800);
  drawFoliage(ctx, rect(CANOPY_TILE.dry, S), rng, [0.02, 0.005, -0.05], 700);
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  tex.name = 'scatter.canopy';
  return tex;
}

function buildRockTexture(): THREE.Texture {
  const S = 256;
  const { cv, ctx } = canvas(S, S);
  const rng = new Rng(0x0c0ffee1);
  const r = { x: 0, y: 0, w: S, h: S };
  const grain: [number, number, number] = [DETAIL_MID, DETAIL_MID * 0.99, DETAIL_MID * 0.95];
  fillTile(ctx, r, grain);
  speckle(ctx, r, rng, grain, 0.13, 2600, 1, 5);
  /* fracture lines */
  for (let i = 0; i < 26; i++) {
    ctx.strokeStyle = detail(rng.range(0.56, 0.72));
    ctx.lineWidth = rng.range(1, 3);
    ctx.beginPath();
    let x = rng.next() * S;
    let y = rng.next() * S;
    ctx.moveTo(x, y);
    for (let k = 0; k < 6; k++) {
      x += rng.range(-40, 40);
      y += rng.range(-40, 40);
      ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
  /* lichen — a hue push only; the mean must stay put */
  for (let i = 0; i < 90; i++) {
    ctx.fillStyle = `rgba(${Math.round(rng.range(185, 215))},${Math.round(rng.range(200, 230))},${Math.round(rng.range(140, 170))},0.35)`;
    ctx.beginPath();
    ctx.arc(rng.next() * S, rng.next() * S, rng.range(3, 13), 0, Math.PI * 2);
    ctx.fill();
  }
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.anisotropy = 4;
  tex.name = 'scatter.rock';
  return tex;
}

/**
 * A clump of blades on a transparent tile — alpha-tested by the grass shader.
 *
 * Luminance only, same rule as the rest: the blade's colour is the card's
 * vertex gradient, this only supplies the shape and the shading down the blade.
 * The tile is drawn *wider than tall at the base* so a card cut from it reads as
 * several blades rather than one leaf.
 */
function buildGrassTexture(): THREE.Texture {
  const S = 128;
  const { cv, ctx } = canvas(S, S);
  const rng = new Rng(0x9ea55);
  ctx.clearRect(0, 0, S, S);
  const blades = 22;
  for (let i = 0; i < blades; i++) {
    const baseX = ((i + 0.5) / blades) * S + rng.range(-5, 5);
    const tipX = baseX + rng.range(-20, 20);
    const height = rng.range(0.5, 1.0) * S;
    const w = rng.range(3.2, 6.5);
    const shade = rng.range(0.86, 1.12);
    const g = ctx.createLinearGradient(0, S, 0, S - height);
    g.addColorStop(0, detail(0.62 * shade));
    g.addColorStop(0.55, detail(0.92 * shade));
    g.addColorStop(1, detail(1.16 * shade));
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.moveTo(baseX - w, S);
    ctx.quadraticCurveTo(baseX - w * 0.6, S - height * 0.5, tipX, S - height);
    ctx.quadraticCurveTo(baseX + w * 0.6, S - height * 0.5, baseX + w, S);
    ctx.closePath();
    ctx.fill();
  }
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  tex.name = 'scatter.grass';
  return tex;
}

function buildPropAtlas(): THREE.Texture {
  const W = 512;
  const H = 256;
  const { cv, ctx } = canvas(W, H);
  const rng = new Rng(0x1234abcd);

  const r = (t: UvTile) => rect(t, W, H);

  /* wood — plank grain */
  {
    const q = r(PROP_TILE.wood);
    fillTile(ctx, q, [0.45, 0.33, 0.2]);
    ctx.save();
    ctx.beginPath();
    ctx.rect(q.x, q.y, q.w, q.h);
    ctx.clip();
    for (let i = 0; i < 90; i++) {
      const k = rng.range(-0.09, 0.09);
      ctx.strokeStyle = rgb(clamp01(0.45 + k), clamp01(0.33 + k), clamp01(0.2 + k * 0.9));
      ctx.lineWidth = rng.range(1, 4);
      const y = q.y + rng.next() * q.h;
      ctx.beginPath();
      ctx.moveTo(q.x, y);
      ctx.bezierCurveTo(q.x + q.w * 0.33, y + rng.range(-5, 5), q.x + q.w * 0.66, y + rng.range(-5, 5), q.x + q.w, y);
      ctx.stroke();
    }
    for (let i = 0; i < 5; i++) {
      ctx.strokeStyle = 'rgba(40,26,14,0.5)';
      ctx.lineWidth = 2;
      const y = q.y + ((i + 0.5) / 5) * q.h;
      ctx.beginPath();
      ctx.moveTo(q.x, y);
      ctx.lineTo(q.x + q.w, y);
      ctx.stroke();
    }
    ctx.restore();
  }
  /* red / white hazard stripe (diagonal) */
  {
    const q = r(PROP_TILE.stripe);
    fillTile(ctx, q, [0.88, 0.87, 0.84]);
    ctx.save();
    ctx.beginPath();
    ctx.rect(q.x, q.y, q.w, q.h);
    ctx.clip();
    ctx.fillStyle = 'rgb(176,38,32)';
    const bands = 6;
    for (let i = 0; i < bands; i++) {
      const y = q.y + (i / bands) * q.h;
      ctx.fillRect(q.x, y, q.w, q.h / (bands * 2));
    }
    /* grime */
    for (let i = 0; i < 60; i++) {
      ctx.fillStyle = `rgba(70,55,40,${rng.range(0.05, 0.2)})`;
      ctx.beginPath();
      ctx.arc(q.x + rng.next() * q.w, q.y + rng.next() * q.h, rng.range(2, 9), 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }
  /* rubber */
  {
    const q = r(PROP_TILE.rubber);
    fillTile(ctx, q, [0.1, 0.1, 0.11]);
    speckle(ctx, q, rng, [0.12, 0.12, 0.13], 0.05, 500, 1, 4);
  }
  /* hay */
  {
    const q = r(PROP_TILE.hay);
    fillTile(ctx, q, [0.62, 0.53, 0.27]);
    ctx.save();
    ctx.beginPath();
    ctx.rect(q.x, q.y, q.w, q.h);
    ctx.clip();
    for (let i = 0; i < 400; i++) {
      const k = rng.range(-0.16, 0.16);
      ctx.strokeStyle = rgb(clamp01(0.62 + k), clamp01(0.53 + k), clamp01(0.27 + k));
      ctx.lineWidth = rng.range(0.8, 2);
      const x = q.x + rng.next() * q.w;
      const y = q.y + rng.next() * q.h;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x + rng.range(-14, 14), y + rng.range(-4, 4));
      ctx.stroke();
    }
    ctx.restore();
  }
  /* rust */
  {
    const q = r(PROP_TILE.rust);
    fillTile(ctx, q, [0.42, 0.24, 0.13]);
    speckle(ctx, q, rng, [0.42, 0.24, 0.13], 0.16, 900, 1, 8);
    for (let i = 0; i < 40; i++) {
      ctx.fillStyle = `rgba(${Math.round(rng.range(60, 110))},${Math.round(rng.range(60, 100))},${Math.round(rng.range(60, 95))},0.5)`;
      ctx.beginPath();
      ctx.arc(q.x + rng.next() * q.w, q.y + rng.next() * q.h, rng.range(3, 12), 0, Math.PI * 2);
      ctx.fill();
    }
  }
  /* bare metal */
  {
    const q = r(PROP_TILE.metal);
    fillTile(ctx, q, [0.5, 0.52, 0.55]);
    speckle(ctx, q, rng, [0.5, 0.52, 0.55], 0.09, 500, 1, 5);
  }
  /* sign face — white with a chevron */
  {
    const q = r(PROP_TILE.sign);
    fillTile(ctx, q, [0.85, 0.84, 0.8]);
    ctx.save();
    ctx.beginPath();
    ctx.rect(q.x, q.y, q.w, q.h);
    ctx.clip();
    ctx.strokeStyle = 'rgb(30,30,32)';
    ctx.lineWidth = Math.max(2, q.w * 0.05);
    ctx.strokeRect(q.x + q.w * 0.08, q.y + q.h * 0.08, q.w * 0.84, q.h * 0.84);
    ctx.fillStyle = 'rgb(30,30,32)';
    ctx.beginPath();
    ctx.moveTo(q.x + q.w * 0.25, q.y + q.h * 0.3);
    ctx.lineTo(q.x + q.w * 0.72, q.y + q.h * 0.5);
    ctx.lineTo(q.x + q.w * 0.25, q.y + q.h * 0.7);
    ctx.lineTo(q.x + q.w * 0.42, q.y + q.h * 0.5);
    ctx.closePath();
    ctx.fill();
    for (let i = 0; i < 40; i++) {
      ctx.fillStyle = `rgba(80,62,45,${rng.range(0.04, 0.16)})`;
      ctx.beginPath();
      ctx.arc(q.x + rng.next() * q.w, q.y + rng.next() * q.h, rng.range(2, 8), 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }
  /* faded paint (wreck bodywork) */
  {
    const q = r(PROP_TILE.paint);
    fillTile(ctx, q, [0.55, 0.45, 0.35]);
    speckle(ctx, q, rng, [0.5, 0.4, 0.31], 0.14, 700, 2, 10);
  }

  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  tex.name = 'scatter.props';
  return tex;
}

export interface ScatterTextureSet {
  /** bark + conifer needle + broadleaf + dry scrub, see CANOPY_TILE */
  canopy: THREE.Texture;
  rock: THREE.Texture;
  grass: THREE.Texture;
  /** wood, stripe, rubber, hay, rust, metal, sign, paint — see PROP_TILE */
  props: THREE.Texture;
}

let provided: Partial<ScatterTextureSet> | null = null;
let cache: ScatterTextureSet | null = null;
let owned: THREE.Texture[] = [];

/**
 * Seam for `src/render/textures.ts`: call once before creating the Scatter to
 * override any of the fallback textures with the shared art-directed set.
 */
export function provideScatterTextures(set: Partial<ScatterTextureSet>): void {
  provided = set;
  cache = null;
}

export function getScatterTextures(): ScatterTextureSet {
  if (cache) return cache;
  const mk = <K extends keyof ScatterTextureSet>(k: K, build: () => THREE.Texture): THREE.Texture => {
    const given = provided?.[k];
    if (given) return given;
    const t = build();
    owned.push(t);
    return t;
  };
  cache = {
    canopy: mk('canopy', buildCanopyAtlas),
    rock: mk('rock', buildRockTexture),
    grass: mk('grass', buildGrassTexture),
    props: mk('props', buildPropAtlas),
  };
  return cache;
}

/** Release only the textures this module built itself. */
export function disposeScatterTextures(): void {
  for (const t of owned) t.dispose();
  owned = [];
  cache = null;
}
