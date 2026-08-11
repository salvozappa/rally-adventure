/**
 * Deterministic noise primitives for terrain and texture synthesis.
 *
 * Everything here is pure and seeded: the same seed always produces the same
 * numbers, on every machine, in every run. No Math.random, no Date, no
 * dependence on iteration order. That is a hard requirement — the physics
 * heightfield, the visual mesh and the splat control map are all derived from
 * these functions and must agree exactly.
 */

/* -------------------------------------------------------------------------- */
/* Scalar helpers                                                             */
/* -------------------------------------------------------------------------- */

export function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}

export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/** Hermite fade, zero first derivative at both ends. */
export function smoothstep(edge0: number, edge1: number, x: number): number {
  if (edge0 === edge1) return x < edge0 ? 0 : 1;
  const t = clamp((x - edge0) / (edge1 - edge0), 0, 1);
  return t * t * (3 - 2 * t);
}

/** Quintic fade, zero first *and* second derivative at both ends. */
export function smootherstep(edge0: number, edge1: number, x: number): number {
  if (edge0 === edge1) return x < edge0 ? 0 : 1;
  const t = clamp((x - edge0) / (edge1 - edge0), 0, 1);
  return t * t * t * (t * (t * 6 - 15) + 10);
}

/** Signed distance from a point to a line segment, plus the projection param. */
export function segmentProject(
  px: number,
  pz: number,
  ax: number,
  az: number,
  bx: number,
  bz: number,
): { t: number; dist: number; side: number; len: number } {
  const dx = bx - ax;
  const dz = bz - az;
  const len2 = dx * dx + dz * dz;
  const len = Math.sqrt(len2);
  const t = len2 > 0 ? clamp(((px - ax) * dx + (pz - az) * dz) / len2, 0, 1) : 0;
  const cx = ax + dx * t;
  const cz = az + dz * t;
  const ox = px - cx;
  const oz = pz - cz;
  const dist = Math.hypot(ox, oz);
  // Sign of the perpendicular offset: positive to the left of a->b.
  const side = len > 0 ? Math.sign(ox * dz - oz * dx) : 0;
  return { t, dist, side, len };
}

/* -------------------------------------------------------------------------- */
/* PRNG                                                                       */
/* -------------------------------------------------------------------------- */

/** Small, fast, well-distributed 32-bit PRNG. Returns floats in [0, 1). */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return function next(): number {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Stateless integer hash -> [0, 1). Useful for per-cell decisions. */
export function hash2i(x: number, y: number, seed: number): number {
  let h = Math.imul(x | 0, 0x27d4eb2d) ^ Math.imul(y | 0, 0x165667b1) ^ Math.imul(seed | 0, 0x9e3779b1);
  h = Math.imul(h ^ (h >>> 15), 0x85ebca6b);
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

/* -------------------------------------------------------------------------- */
/* 2D simplex noise                                                           */
/* -------------------------------------------------------------------------- */

const F2 = 0.5 * (Math.sqrt(3) - 1);
const G2 = (3 - Math.sqrt(3)) / 6;

/** 12 evenly spread gradient directions, as used by Perlin's reference code. */
const GRAD2 = new Float32Array([
  1, 1, -1, 1, 1, -1, -1, -1, 1, 0, -1, 0, 1, 0, -1, 0, 0, 1, 0, -1, 0, 1, 0, -1,
]);

/**
 * Seeded 2D simplex noise. Output is roughly in [-1, 1] (the classic scaling
 * constant 70 puts the theoretical maximum just under 1).
 *
 * Simplex rather than Perlin because Perlin's value lattice leaves visible
 * axis-aligned creases, and on a heightfield those read as an obviously
 * artificial grid when the sun rakes across at a low angle.
 */
export class Noise2D {
  private readonly perm = new Uint8Array(512);
  private readonly permMod12 = new Uint8Array(512);

  constructor(seed: number) {
    const rnd = mulberry32(seed);
    const p = new Uint8Array(256);
    for (let i = 0; i < 256; i++) p[i] = i;
    // Fisher-Yates with a seeded PRNG: deterministic permutation.
    for (let i = 255; i > 0; i--) {
      const j = Math.floor(rnd() * (i + 1));
      const t = p[i]!;
      p[i] = p[j]!;
      p[j] = t;
    }
    for (let i = 0; i < 512; i++) {
      this.perm[i] = p[i & 255]!;
      this.permMod12[i] = this.perm[i]! % 12;
    }
  }

  /** Noise value at (x, y), in approximately [-1, 1]. */
  sample(x: number, y: number): number {
    const perm = this.perm;
    const permMod12 = this.permMod12;

    const s = (x + y) * F2;
    const i = Math.floor(x + s);
    const j = Math.floor(y + s);
    const t = (i + j) * G2;
    const x0 = x - (i - t);
    const y0 = y - (j - t);

    let i1: number;
    let j1: number;
    if (x0 > y0) {
      i1 = 1;
      j1 = 0;
    } else {
      i1 = 0;
      j1 = 1;
    }

    const x1 = x0 - i1 + G2;
    const y1 = y0 - j1 + G2;
    const x2 = x0 - 1 + 2 * G2;
    const y2 = y0 - 1 + 2 * G2;

    const ii = i & 255;
    const jj = j & 255;

    let n = 0;

    let t0 = 0.5 - x0 * x0 - y0 * y0;
    if (t0 > 0) {
      const gi = permMod12[ii + perm[jj]!]! * 2;
      t0 *= t0;
      n += t0 * t0 * (GRAD2[gi]! * x0 + GRAD2[gi + 1]! * y0);
    }

    let t1 = 0.5 - x1 * x1 - y1 * y1;
    if (t1 > 0) {
      const gi = permMod12[ii + i1 + perm[jj + j1]!]! * 2;
      t1 *= t1;
      n += t1 * t1 * (GRAD2[gi]! * x1 + GRAD2[gi + 1]! * y1);
    }

    let t2 = 0.5 - x2 * x2 - y2 * y2;
    if (t2 > 0) {
      const gi = permMod12[ii + 1 + perm[jj + 1]!]! * 2;
      t2 *= t2;
      n += t2 * t2 * (GRAD2[gi]! * x2 + GRAD2[gi + 1]! * y2);
    }

    return 70 * n;
  }

  /** Simplex remapped to [0, 1]. */
  sample01(x: number, y: number): number {
    return this.sample(x, y) * 0.5 + 0.5;
  }

  /**
   * Fractional Brownian motion. Normalised so the result stays in [-1, 1]
   * regardless of octave count — otherwise changing `octaves` silently
   * rescales the whole world.
   */
  fbm(x: number, y: number, octaves: number, lacunarity = 2.0, gain = 0.5): number {
    let amp = 1;
    let freq = 1;
    let sum = 0;
    let norm = 0;
    for (let o = 0; o < octaves; o++) {
      sum += amp * this.sample(x * freq, y * freq);
      norm += amp;
      amp *= gain;
      freq *= lacunarity;
    }
    return norm > 0 ? sum / norm : 0;
  }

  /**
   * Ridged multifractal. Produces sharp crest lines rather than blobs — this
   * is what turns "lumpy noise" into something with recognisable ridgelines
   * and drainage divides. Output in [0, 1].
   */
  ridged(x: number, y: number, octaves: number, lacunarity = 2.05, gain = 0.5): number {
    let amp = 1;
    let freq = 1;
    let sum = 0;
    let norm = 0;
    let prev = 1;
    for (let o = 0; o < octaves; o++) {
      let n = 1 - Math.abs(this.sample(x * freq, y * freq));
      n *= n;
      // Weight each octave by the previous one so detail concentrates on the
      // crests instead of spraying evenly over the flats.
      n *= prev;
      prev = clamp(n * 1.6, 0, 1);
      sum += amp * n;
      norm += amp;
      amp *= gain;
      freq *= lacunarity;
    }
    return norm > 0 ? sum / norm : 0;
  }

  /**
   * Billowy noise — |fbm|, inverted. Reads as soft mounded dunes and is a good
   * counterweight to `ridged` in the low ground.
   */
  billow(x: number, y: number, octaves: number, lacunarity = 2.0, gain = 0.5): number {
    let amp = 1;
    let freq = 1;
    let sum = 0;
    let norm = 0;
    for (let o = 0; o < octaves; o++) {
      sum += amp * Math.abs(this.sample(x * freq, y * freq));
      norm += amp;
      amp *= gain;
      freq *= lacunarity;
    }
    return norm > 0 ? sum / norm : 0;
  }
}

/**
 * Domain warp: displace the sample point by a low-frequency vector field
 * before evaluating the real noise. This is the single cheapest way to stop
 * fractal terrain looking like fractal terrain — it bends ridgelines into
 * curves and destroys the tell-tale isotropy of raw fBm.
 */
export function domainWarp(
  wx: Noise2D,
  wz: Noise2D,
  x: number,
  z: number,
  frequency: number,
  amplitude: number,
  octaves = 3,
): { x: number; z: number } {
  const nx = wx.fbm(x * frequency, z * frequency, octaves);
  const nz = wz.fbm(x * frequency + 31.7, z * frequency - 17.3, octaves);
  return { x: x + nx * amplitude, z: z + nz * amplitude };
}

/**
 * Bilinear sample of a scalar grid, with clamped edges.
 * `w` is the number of columns; `data.length` must be `w * h`.
 */
export function bilinearGrid(
  data: Float32Array,
  w: number,
  h: number,
  fx: number,
  fy: number,
): number {
  const x = clamp(fx, 0, w - 1.0001);
  const y = clamp(fy, 0, h - 1.0001);
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const tx = x - x0;
  const ty = y - y0;
  const x1 = x0 + 1 < w ? x0 + 1 : x0;
  const y1 = y0 + 1 < h ? y0 + 1 : y0;
  const a = data[y0 * w + x0]!;
  const b = data[y0 * w + x1]!;
  const c = data[y1 * w + x0]!;
  const d = data[y1 * w + x1]!;
  return lerp(lerp(a, b, tx), lerp(c, d, tx), ty);
}
