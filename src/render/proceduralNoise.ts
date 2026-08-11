/**
 * proceduralNoise.ts — pattern-synthesis primitives for the Rally Adventure texture library.
 *
 * Everything here is pure math over Float32 scalar fields ("Field") and RGBA byte
 * buffers. No THREE dependency, no DOM dependency — so it runs identically in a
 * worker, in node, or on the main thread.
 *
 * SEAMLESS BY CONSTRUCTION: every noise function takes an integer `period` and
 * wraps its lattice with a modulo, so a field sampled over uv in [0,1) tiles
 * perfectly. fBm keeps this property by rounding each octave's period to an
 * integer and using that same integer as the frequency. Domain warping keeps it
 * too, because a periodic function of a periodic offset is still periodic.
 */

/* ------------------------------------------------------------------ *
 * Types
 * ------------------------------------------------------------------ */

/**
 * A single-channel scalar field, size*size, row-major, conventionally 0..1.
 *
 * Pinned to `ArrayBuffer` rather than the default `ArrayBufferLike`: TypeScript
 * 5.7+ made the typed arrays generic over their backing buffer, and leaving it
 * open admits `SharedArrayBuffer`, which then fails to assign to the plain
 * `Float32Array` that local `let` bindings infer.
 */
export type Field = Float32Array<ArrayBuffer>;

/** RGBA8 image buffer, size*size*4. */
export type RGBA = Uint8ClampedArray<ArrayBuffer>;

/** A discrete colour palette: n entries of packed RGB bytes (length n*3). */
export type Palette = Uint8Array<ArrayBuffer>;

/* ------------------------------------------------------------------ *
 * Random / hashing
 * ------------------------------------------------------------------ */

/** Small fast deterministic PRNG. Returns a function producing [0,1). */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return function rnd(): number {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Integer 2D hash -> uint32. */
export function ihash2(x: number, y: number, seed: number): number {
  let h = (Math.imul(x | 0, 374761393) ^ Math.imul(y | 0, 668265263) ^ Math.imul(seed | 0, 2246822519)) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return (h ^ (h >>> 16)) >>> 0;
}

/** Integer 2D hash -> [0,1). */
export function hash2(x: number, y: number, seed: number): number {
  return ihash2(x, y, seed) / 4294967296;
}

/* ------------------------------------------------------------------ *
 * Core noise
 * ------------------------------------------------------------------ */

const GRAD_N = 256;
const GRAD = new Float32Array(GRAD_N * 2);
for (let i = 0; i < GRAD_N; i++) {
  const a = (i / GRAD_N) * Math.PI * 2;
  GRAD[i * 2] = Math.cos(a);
  GRAD[i * 2 + 1] = Math.sin(a);
}

function quintic(t: number): number {
  return t * t * t * (t * (t * 6 - 15) + 10);
}

function wrapInt(v: number, p: number): number {
  const m = v % p;
  return m < 0 ? m + p : m;
}

/**
 * Periodic Perlin (gradient) noise. Returns roughly -1..1.
 * `period` must be an integer >= 1; the field repeats every `period` units of x/y.
 */
export function perlin2(x: number, y: number, period: number, seed: number): number {
  const P = Math.max(1, period | 0);
  const xi = Math.floor(x);
  const yi = Math.floor(y);
  const fx = x - xi;
  const fy = y - yi;
  const x0 = wrapInt(xi, P);
  const y0 = wrapInt(yi, P);
  const x1 = x0 + 1 === P ? 0 : x0 + 1;
  const y1 = y0 + 1 === P ? 0 : y0 + 1;
  const u = quintic(fx);
  const v = quintic(fy);

  const g = (gx: number, gy: number, dx: number, dy: number): number => {
    const idx = (ihash2(gx, gy, seed) & (GRAD_N - 1)) * 2;
    return GRAD[idx]! * dx + GRAD[idx + 1]! * dy;
  };

  const n00 = g(x0, y0, fx, fy);
  const n10 = g(x1, y0, fx - 1, fy);
  const n01 = g(x0, y1, fx, fy - 1);
  const n11 = g(x1, y1, fx - 1, fy - 1);
  const a = n00 + u * (n10 - n00);
  const b = n01 + u * (n11 - n01);
  return (a + v * (b - a)) * 1.4142;
}

/** Periodic value noise. Returns 0..1. Blockier than Perlin — good for grit. */
export function value2(x: number, y: number, period: number, seed: number): number {
  const P = Math.max(1, period | 0);
  const xi = Math.floor(x);
  const yi = Math.floor(y);
  const fx = quintic(x - xi);
  const fy = quintic(y - yi);
  const x0 = wrapInt(xi, P);
  const y0 = wrapInt(yi, P);
  const x1 = x0 + 1 === P ? 0 : x0 + 1;
  const y1 = y0 + 1 === P ? 0 : y0 + 1;
  const n00 = hash2(x0, y0, seed);
  const n10 = hash2(x1, y0, seed);
  const n01 = hash2(x0, y1, seed);
  const n11 = hash2(x1, y1, seed);
  const a = n00 + fx * (n10 - n00);
  const b = n01 + fx * (n11 - n01);
  return a + fy * (b - a);
}

export interface FbmParams {
  /** Number of octaves. */
  octaves?: number;
  /** Base period in tiles across the unit square (integer-ish). */
  period?: number;
  /** Frequency multiplier per octave. */
  lacunarity?: number;
  /** Amplitude multiplier per octave. */
  gain?: number;
  seed?: number;
  /** 1 - |n| : sharp creases at the top of the range. Cliffs, cracks, ridges. */
  ridged?: boolean;
  /** |n| : billowy lumps. Clouds, smoke, rust. */
  turbulence?: boolean;
  /** Non-uniform frequency scaling, e.g. [1, 6] stretches the pattern horizontally. */
  stretch?: [number, number];
  /** Use value noise instead of Perlin (blockier, grittier). */
  value?: boolean;
}

/** fBm over periodic noise, evaluated at uv in [0,1). Returns 0..1. */
export function fbm2(x: number, y: number, p: FbmParams): number {
  const oct = p.octaves ?? 4;
  const lac = p.lacunarity ?? 2;
  const gain = p.gain ?? 0.5;
  const seed = p.seed ?? 1;
  const sx = p.stretch ? p.stretch[0] : 1;
  const sy = p.stretch ? p.stretch[1] : 1;
  let per = p.period ?? 4;
  let amp = 1;
  let sum = 0;
  let norm = 0;

  for (let o = 0; o < oct; o++) {
    const P = Math.max(1, Math.round(per));
    // Non-uniform stretch: periods must stay integers on BOTH axes to tile, so we
    // wrap x and y on independent integer periods via an anisotropic lattice.
    const PX = Math.max(1, Math.round(P * sx));
    const PY = Math.max(1, Math.round(P * sy));
    let n: number;
    if (p.value) {
      n = value2Aniso(x * PX, y * PY, PX, PY, seed + o * 1013) * 2 - 1;
    } else {
      n = perlin2Aniso(x * PX, y * PY, PX, PY, seed + o * 1013);
    }
    if (p.ridged) n = 1 - Math.abs(n);
    else if (p.turbulence) n = Math.abs(n);
    else n = n * 0.5 + 0.5;
    sum += n * amp;
    norm += amp;
    amp *= gain;
    per *= lac;
  }
  return norm > 0 ? sum / norm : 0;
}

/** Perlin with independent x/y periods (for stretched/anisotropic patterns). */
export function perlin2Aniso(x: number, y: number, px: number, py: number, seed: number): number {
  const PX = Math.max(1, px | 0);
  const PY = Math.max(1, py | 0);
  const xi = Math.floor(x);
  const yi = Math.floor(y);
  const fx = x - xi;
  const fy = y - yi;
  const x0 = wrapInt(xi, PX);
  const y0 = wrapInt(yi, PY);
  const x1 = x0 + 1 === PX ? 0 : x0 + 1;
  const y1 = y0 + 1 === PY ? 0 : y0 + 1;
  const u = quintic(fx);
  const v = quintic(fy);
  const g = (gx: number, gy: number, dx: number, dy: number): number => {
    const idx = (ihash2(gx, gy, seed) & (GRAD_N - 1)) * 2;
    return GRAD[idx]! * dx + GRAD[idx + 1]! * dy;
  };
  const n00 = g(x0, y0, fx, fy);
  const n10 = g(x1, y0, fx - 1, fy);
  const n01 = g(x0, y1, fx, fy - 1);
  const n11 = g(x1, y1, fx - 1, fy - 1);
  const a = n00 + u * (n10 - n00);
  const b = n01 + u * (n11 - n01);
  return (a + v * (b - a)) * 1.4142;
}

/** Value noise with independent x/y periods. Returns 0..1. */
export function value2Aniso(x: number, y: number, px: number, py: number, seed: number): number {
  const PX = Math.max(1, px | 0);
  const PY = Math.max(1, py | 0);
  const xi = Math.floor(x);
  const yi = Math.floor(y);
  const fx = quintic(x - xi);
  const fy = quintic(y - yi);
  const x0 = wrapInt(xi, PX);
  const y0 = wrapInt(yi, PY);
  const x1 = x0 + 1 === PX ? 0 : x0 + 1;
  const y1 = y0 + 1 === PY ? 0 : y0 + 1;
  const n00 = hash2(x0, y0, seed);
  const n10 = hash2(x1, y0, seed);
  const n01 = hash2(x0, y1, seed);
  const n11 = hash2(x1, y1, seed);
  const a = n00 + fx * (n10 - n00);
  const b = n01 + fx * (n11 - n01);
  return a + fy * (b - a);
}

/* ------------------------------------------------------------------ *
 * Field construction
 * ------------------------------------------------------------------ */

export function makeField(size: number, fill = 0): Field {
  const f = new Float32Array(size * size);
  if (fill !== 0) f.fill(fill);
  return f;
}

export interface FieldParams extends FbmParams {
  /** Domain-warp amount in uv units (0.05–0.3 typical). */
  warp?: number;
  /** Period of the warp field (defaults to the base period). */
  warpPeriod?: number;
  /** Octaves in the warp field. */
  warpOctaves?: number;
}

/** Build a size*size fBm field, optionally domain-warped. Seamless. */
export function fbmField(size: number, p: FieldParams): Field {
  const f = new Float32Array(size * size);
  const seed = p.seed ?? 1;
  const warp = p.warp ?? 0;
  const wp: FbmParams = {
    octaves: p.warpOctaves ?? 2,
    period: p.warpPeriod ?? p.period ?? 4,
    lacunarity: p.lacunarity ?? 2,
    gain: 0.5,
  };
  const inv = 1 / size;
  for (let y = 0; y < size; y++) {
    const v = y * inv;
    for (let x = 0; x < size; x++) {
      const u = x * inv;
      let uu = u;
      let vv = v;
      if (warp !== 0) {
        const wx = fbm2(u, v, { ...wp, seed: seed + 7717 }) - 0.5;
        const wy = fbm2(u, v, { ...wp, seed: seed + 3313 }) - 0.5;
        uu = u + wx * warp * 2;
        vv = v + wy * warp * 2;
      }
      f[y * size + x] = fbm2(uu, vv, p);
    }
  }
  return f;
}

/** Fill a field from a callback over uv in [0,1). */
export function fieldFromUV(size: number, fn: (u: number, v: number, x: number, y: number) => number): Field {
  const f = new Float32Array(size * size);
  const inv = 1 / size;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      f[y * size + x] = fn(x * inv, y * inv, x, y);
    }
  }
  return f;
}

/* ------------------------------------------------------------------ *
 * Worley / Voronoi
 * ------------------------------------------------------------------ */

export interface WorleyResult {
  /** Distance to nearest feature point, normalised so ~1 == one cell width. */
  f1: Field;
  /** Distance to second-nearest. */
  f2: Field;
  /** Stable per-cell random value 0..1 — use for per-pebble colour variation. */
  id: Field;
  /** Vector from pixel to the owning feature point, in cell units. */
  dx: Field;
  dy: Field;
}

export type WorleyMetric = 'euclidean' | 'manhattan' | 'chebyshev';

/**
 * Periodic Worley noise on a `cells x cells` jittered grid over the unit square.
 * Seamless. `jitter` 0 = perfect grid, 1 = fully random within the cell.
 */
export function worley(
  size: number,
  cells: number,
  seed: number,
  jitter = 1,
  metric: WorleyMetric = 'euclidean',
  stretchY = 1,
): WorleyResult {
  const CX = Math.max(1, cells | 0);
  const CY = Math.max(1, Math.round(cells * stretchY));
  const f1 = new Float32Array(size * size);
  const f2 = new Float32Array(size * size);
  const id = new Float32Array(size * size);
  const dxF = new Float32Array(size * size);
  const dyF = new Float32Array(size * size);

  // Precompute feature points per cell.
  const px = new Float32Array(CX * CY);
  const py = new Float32Array(CX * CY);
  const pid = new Float32Array(CX * CY);
  for (let cy = 0; cy < CY; cy++) {
    for (let cx = 0; cx < CX; cx++) {
      const i = cy * CX + cx;
      const h1 = hash2(cx, cy, seed);
      const h2 = hash2(cx, cy, seed + 9187);
      px[i] = cx + 0.5 + (h1 - 0.5) * jitter;
      py[i] = cy + 0.5 + (h2 - 0.5) * jitter;
      pid[i] = hash2(cx, cy, seed + 4441);
    }
  }

  const dist = (ax: number, ay: number): number => {
    if (metric === 'manhattan') return Math.abs(ax) + Math.abs(ay);
    if (metric === 'chebyshev') return Math.max(Math.abs(ax), Math.abs(ay));
    return Math.sqrt(ax * ax + ay * ay);
  };

  const inv = 1 / size;
  for (let y = 0; y < size; y++) {
    const gy = y * inv * CY;
    const by = Math.floor(gy);
    for (let x = 0; x < size; x++) {
      const gx = x * inv * CX;
      const bx = Math.floor(gx);
      let best = Infinity;
      let second = Infinity;
      let bestId = 0;
      let bestDX = 0;
      let bestDY = 0;
      for (let oy = -1; oy <= 1; oy++) {
        const wy = wrapInt(by + oy, CY);
        for (let ox = -1; ox <= 1; ox++) {
          const wx = wrapInt(bx + ox, CX);
          const i = wy * CX + wx;
          // Unwrapped feature position relative to this pixel's neighbourhood.
          const fx = px[i]! - wx + (bx + ox);
          const fy = py[i]! - wy + (by + oy);
          const ax = fx - gx;
          const ay = fy - gy;
          const d = dist(ax, ay);
          if (d < best) {
            second = best;
            best = d;
            bestId = pid[i]!;
            bestDX = ax;
            bestDY = ay;
          } else if (d < second) {
            second = d;
          }
        }
      }
      const o = y * size + x;
      f1[o] = best;
      f2[o] = second;
      id[o] = bestId;
      dxF[o] = bestDX;
      dyF[o] = bestDY;
    }
  }
  return { f1, f2, id, dx: dxF, dy: dyF };
}

/** f2 - f1: bright ridges along cell borders. Cracks, mud plates, dry earth. */
export function worleyEdges(w: WorleyResult): Field {
  const out = new Float32Array(w.f1.length);
  for (let i = 0; i < out.length; i++) out[i] = w.f2[i]! - w.f1[i]!;
  return out;
}

/* ------------------------------------------------------------------ *
 * Field arithmetic
 * ------------------------------------------------------------------ */

export function cloneField(f: Field): Field {
  return new Float32Array(f);
}

export function mapField(f: Field, fn: (v: number, i: number) => number): Field {
  const out = new Float32Array(f.length);
  for (let i = 0; i < f.length; i++) out[i] = fn(f[i]!, i);
  return out;
}

export function mapFieldInPlace(f: Field, fn: (v: number, i: number) => number): Field {
  for (let i = 0; i < f.length; i++) f[i] = fn(f[i]!, i);
  return f;
}

export function combine(a: Field, b: Field, fn: (x: number, y: number) => number): Field {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = fn(a[i]!, b[i]!);
  return out;
}

export function addField(a: Field, b: Field, scale = 1): Field {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i]! + b[i]! * scale;
  return out;
}

export function mulField(a: Field, b: Field): Field {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i]! * b[i]!;
  return out;
}

export function mixField(a: Field, b: Field, t: Field | number): Field {
  const out = new Float32Array(a.length);
  if (typeof t === 'number') {
    for (let i = 0; i < a.length; i++) out[i] = a[i]! + (b[i]! - a[i]!) * t;
  } else {
    for (let i = 0; i < a.length; i++) out[i] = a[i]! + (b[i]! - a[i]!) * t[i]!;
  }
  return out;
}

export function clamp(v: number, lo = 0, hi = 1): number {
  return v < lo ? lo : v > hi ? hi : v;
}

export function smoothstep(e0: number, e1: number, x: number): number {
  const t = clamp((x - e0) / (e1 - e0));
  return t * t * (3 - 2 * t);
}

export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/** Rescale a field so its min/max become 0/1. */
export function normalizeField(f: Field): Field {
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < f.length; i++) {
    const v = f[i]!;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  const d = hi - lo;
  const out = new Float32Array(f.length);
  if (d < 1e-9) return out;
  for (let i = 0; i < f.length; i++) out[i] = (f[i]! - lo) / d;
  return out;
}

/** S-curve contrast around a pivot. amount > 1 hardens, < 1 softens. */
export function contrastField(f: Field, amount: number, pivot = 0.5): Field {
  return mapField(f, (v) => clamp((v - pivot) * amount + pivot));
}

/** Remap an input window to 0..1 with smooth edges. Great for isolating features. */
export function thresholdField(f: Field, lo: number, hi: number): Field {
  return mapField(f, (v) => smoothstep(lo, hi, v));
}

/** Quantise a field to n discrete levels — the poster-step look, pre-palette. */
export function posterizeField(f: Field, levels: number): Field {
  const n = Math.max(2, levels | 0) - 1;
  return mapField(f, (v) => Math.round(clamp(v) * n) / n);
}

/* ------------------------------------------------------------------ *
 * Sampling / filtering (all wrap)
 * ------------------------------------------------------------------ */

/** Nearest-neighbour fetch with wrapping. */
export function fetchWrap(f: Field, size: number, x: number, y: number): number {
  return f[wrapInt(y | 0, size) * size + wrapInt(x | 0, size)]!;
}

/** Bilinear sample at fractional pixel coordinates, wrapping. */
export function sampleWrap(f: Field, size: number, x: number, y: number): number {
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const fx = x - x0;
  const fy = y - y0;
  const xa = wrapInt(x0, size);
  const ya = wrapInt(y0, size);
  const xb = xa + 1 === size ? 0 : xa + 1;
  const yb = ya + 1 === size ? 0 : ya + 1;
  const n00 = f[ya * size + xa]!;
  const n10 = f[ya * size + xb]!;
  const n01 = f[yb * size + xa]!;
  const n11 = f[yb * size + xb]!;
  const a = n00 + fx * (n10 - n00);
  const b = n01 + fx * (n11 - n01);
  return a + fy * (b - a);
}

/** Separable box blur, wrapping, `passes` iterations approximate a gaussian. */
export function blurField(f: Field, size: number, radius: number, passes = 2): Field {
  if (radius <= 0) return cloneField(f);
  let src = cloneField(f);
  let dst = new Float32Array(f.length);
  const r = Math.max(1, Math.round(radius));
  const n = r * 2 + 1;
  for (let p = 0; p < passes; p++) {
    // horizontal
    for (let y = 0; y < size; y++) {
      const row = y * size;
      let sum = 0;
      for (let k = -r; k <= r; k++) sum += src[row + wrapInt(k, size)]!;
      for (let x = 0; x < size; x++) {
        dst[row + x] = sum / n;
        sum -= src[row + wrapInt(x - r, size)]!;
        sum += src[row + wrapInt(x + r + 1, size)]!;
      }
    }
    const t = src;
    src = dst;
    dst = t;
    // vertical
    for (let x = 0; x < size; x++) {
      let sum = 0;
      for (let k = -r; k <= r; k++) sum += src[wrapInt(k, size) * size + x]!;
      for (let y = 0; y < size; y++) {
        dst[y * size + x] = sum / n;
        sum -= src[wrapInt(y - r, size) * size + x]!;
        sum += src[wrapInt(y + r + 1, size) * size + x]!;
      }
    }
    const t2 = src;
    src = dst;
    dst = t2;
  }
  return src;
}

/** Unsharp mask — pushes local contrast up. Essential for the "punched" era look. */
export function sharpenField(f: Field, size: number, radius = 1, amount = 0.6): Field {
  const b = blurField(f, size, radius, 1);
  return mapField(f, (v, i) => clamp(v + (v - b[i]!) * amount));
}

/**
 * Directional smear: accumulates an exponentially decaying trail along (dx,dy).
 * Tyre drag on dirt, rain erosion on rock, brushed metal, wind on snow.
 */
export function streakField(f: Field, size: number, dx: number, dy: number, length: number, decay = 0.82): Field {
  const out = new Float32Array(f.length);
  const steps = Math.max(1, Math.round(length));
  const len = Math.hypot(dx, dy) || 1;
  const ux = dx / len;
  const uy = dy / len;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      let acc = 0;
      let w = 1;
      let tot = 0;
      for (let s = 0; s < steps; s++) {
        acc += sampleWrap(f, size, x - ux * s, y - uy * s) * w;
        tot += w;
        w *= decay;
      }
      out[y * size + x] = acc / tot;
    }
  }
  return out;
}

/* ------------------------------------------------------------------ *
 * Stamping (all wrap)
 * ------------------------------------------------------------------ */

export type StampOp = 'add' | 'max' | 'min' | 'sub' | 'set';

function applyOp(cur: number, v: number, op: StampOp): number {
  switch (op) {
    case 'add':
      return cur + v;
    case 'max':
      return Math.max(cur, v);
    case 'min':
      return Math.min(cur, v);
    case 'sub':
      return cur - v;
    default:
      return v;
  }
}

/**
 * Stamp a radial blob. `pow` shapes the falloff: 0.5 domed, 1 linear, 3 spiky.
 * `flat` (0..1) keeps the centre at full amplitude before the falloff starts —
 * use it for pebbles that should read as solid objects, not gaussian smudges.
 */
export function stampDisc(
  f: Field,
  size: number,
  cx: number,
  cy: number,
  r: number,
  amp: number,
  op: StampOp = 'max',
  pow = 0.5,
  flat = 0,
): void {
  const ri = Math.ceil(r) + 1;
  for (let oy = -ri; oy <= ri; oy++) {
    const y = wrapInt(Math.round(cy) + oy, size);
    for (let ox = -ri; ox <= ri; ox++) {
      const dx = Math.round(cx) + ox - cx;
      const dy = Math.round(cy) + oy - cy;
      const d = Math.hypot(dx, dy) / r;
      if (d >= 1) continue;
      let t = 1 - d;
      if (flat > 0) t = clamp((1 - d) / Math.max(1e-4, 1 - flat));
      const v = Math.pow(t, pow) * amp;
      const x = wrapInt(Math.round(cx) + ox, size);
      const i = y * size + x;
      f[i] = applyOp(f[i]!, v, op);
    }
  }
}

/** Stamp an axis-aligned-ish ellipse with rotation. Pebbles, leaves, scuffs. */
export function stampEllipse(
  f: Field,
  size: number,
  cx: number,
  cy: number,
  rx: number,
  ry: number,
  angle: number,
  amp: number,
  op: StampOp = 'max',
  pow = 0.5,
  flat = 0,
): void {
  const ca = Math.cos(-angle);
  const sa = Math.sin(-angle);
  const ri = Math.ceil(Math.max(rx, ry)) + 1;
  for (let oy = -ri; oy <= ri; oy++) {
    for (let ox = -ri; ox <= ri; ox++) {
      const dx = Math.round(cx) + ox - cx;
      const dy = Math.round(cy) + oy - cy;
      const lx = (dx * ca - dy * sa) / rx;
      const ly = (dx * sa + dy * ca) / ry;
      const d = Math.hypot(lx, ly);
      if (d >= 1) continue;
      let t = 1 - d;
      if (flat > 0) t = clamp((1 - d) / Math.max(1e-4, 1 - flat));
      const v = Math.pow(t, pow) * amp;
      const i = wrapInt(Math.round(cy) + oy, size) * size + wrapInt(Math.round(cx) + ox, size);
      f[i] = applyOp(f[i]!, v, op);
    }
  }
}

/** Stamp a line as a chain of discs. Blades of grass, scratches, cracks, twigs. */
export function stampLine(
  f: Field,
  size: number,
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  width: number,
  amp: number,
  op: StampOp = 'max',
  taper = 0,
): void {
  const len = Math.hypot(x1 - x0, y1 - y0);
  const steps = Math.max(1, Math.ceil(len));
  for (let s = 0; s <= steps; s++) {
    const t = s / steps;
    const w = width * (1 - taper * t);
    if (w <= 0.05) continue;
    stampDisc(f, size, x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, Math.max(0.6, w), amp * (1 - taper * t * 0.35), op, 0.4, 0.5);
  }
}

/**
 * Scatter helper: calls `fn` `count` times with a deterministic RNG and a random
 * wrapped position. Use it to place discrete features at texel scale — the thing
 * that separates a real-looking era texture from a noise field.
 */
export function scatter(
  size: number,
  count: number,
  seed: number,
  fn: (cx: number, cy: number, rnd: () => number, i: number) => void,
): void {
  const rnd = mulberry32(seed);
  for (let i = 0; i < count; i++) {
    fn(rnd() * size, rnd() * size, rnd, i);
  }
}

/* ------------------------------------------------------------------ *
 * Ordered dithering
 * ------------------------------------------------------------------ */

/** 4x4 Bayer matrix, normalised to 0..1 (values are (n+0.5)/16). */
export const BAYER4 = ((): Float32Array => {
  const m = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5];
  const f = new Float32Array(16);
  for (let i = 0; i < 16; i++) f[i] = (m[i]! + 0.5) / 16;
  return f;
})();

/** 8x8 Bayer matrix, normalised to 0..1. Generated by recursive expansion. */
export const BAYER8 = ((): Float32Array => {
  const b4 = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5];
  const f = new Float32Array(64);
  for (let y = 0; y < 8; y++) {
    for (let x = 0; x < 8; x++) {
      const q = (y & 4 ? 2 : 0) + (x & 4 ? 1 : 0);
      const quadOrder = [0, 2, 3, 1];
      const inner = b4[(y & 3) * 4 + (x & 3)]!;
      f[y * 8 + x] = (quadOrder[q]! * 16 + inner + 0.5) / 64;
    }
  }
  return f;
})();

/** Ordered-dither threshold for a pixel. `level` is 4 or 8. Returns 0..1. */
export function bayer(x: number, y: number, level: 4 | 8 = 4): number {
  return level === 8 ? BAYER8[(y & 7) * 8 + (x & 7)]! : BAYER4[(y & 3) * 4 + (x & 3)]!;
}

/** Bayer value centred on zero: -0.5..0.5. Add to an index before rounding. */
export function bayerSigned(x: number, y: number, level: 4 | 8 = 4): number {
  return bayer(x, y, level) - 0.5;
}

/* ------------------------------------------------------------------ *
 * Colour / palettes
 * ------------------------------------------------------------------ */

export function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace('#', '');
  const n = parseInt(h.length === 3 ? h[0]! + h[0]! + h[1]! + h[1]! + h[2]! + h[2]! : h, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

export function rgbToHex(r: number, g: number, b: number): string {
  const c = (v: number) => Math.round(clamp(v, 0, 255)).toString(16).padStart(2, '0');
  return `#${c(r)}${c(g)}${c(b)}`;
}

/**
 * Build a discrete `n`-entry palette by piecewise-linear interpolation through
 * the given hex stops. This is THE ramp: textures index into it, so the whole
 * texture is guaranteed to use only these n colours.
 */
export function buildRamp(stops: readonly string[], n = 16): Palette {
  const rgb = stops.map(hexToRgb);
  const out = new Uint8Array(n * 3);
  const segs = rgb.length - 1;
  for (let i = 0; i < n; i++) {
    const t = segs === 0 ? 0 : (i / (n - 1)) * segs;
    const s = Math.min(segs - 1, Math.floor(t));
    const ft = segs === 0 ? 0 : t - s;
    const a = rgb[Math.max(0, s)]!;
    const b = rgb[Math.min(rgb.length - 1, s + 1)]!;
    out[i * 3] = Math.round(a[0] + (b[0] - a[0]) * ft);
    out[i * 3 + 1] = Math.round(a[1] + (b[1] - a[1]) * ft);
    out[i * 3 + 2] = Math.round(a[2] + (b[2] - a[2]) * ft);
  }
  return out;
}

/** A palette straight from an explicit list of hexes (no interpolation). */
export function paletteFrom(hexes: readonly string[]): Palette {
  const out = new Uint8Array(hexes.length * 3);
  hexes.forEach((h, i) => {
    const c = hexToRgb(h);
    out[i * 3] = c[0];
    out[i * 3 + 1] = c[1];
    out[i * 3 + 2] = c[2];
  });
  return out;
}

export function rampSize(p: Palette): number {
  return p.length / 3;
}

/** Multiply every entry of a ramp (tint / darken a whole material). */
export function tintRamp(p: Palette, r: number, g: number, b: number): Palette {
  const out = new Uint8Array(p.length);
  for (let i = 0; i < p.length; i += 3) {
    out[i] = clamp(p[i]! * r, 0, 255);
    out[i + 1] = clamp(p[i + 1]! * g, 0, 255);
    out[i + 2] = clamp(p[i + 2]! * b, 0, 255);
  }
  return out;
}

/* ------------------------------------------------------------------ *
 * Rasterisation: field -> RGBA
 * ------------------------------------------------------------------ */

export function makeRGBA(size: number, r = 0, g = 0, b = 0, a = 255): RGBA {
  const buf = new Uint8ClampedArray(size * size * 4);
  for (let i = 0; i < buf.length; i += 4) {
    buf[i] = r;
    buf[i + 1] = g;
    buf[i + 2] = b;
    buf[i + 3] = a;
  }
  return buf;
}

export interface ShadeOptions {
  /** Dither spread in ramp-index units. ~1.0 dithers between adjacent entries. */
  dither?: number;
  /** Bayer matrix size. */
  bayerLevel?: 4 | 8;
  /** Preserve existing alpha in `out` instead of forcing 255. */
  keepAlpha?: boolean;
}

/**
 * Paint a scalar field through a discrete palette with ordered dithering.
 * This is the workhorse: it is what makes output read as 1997 rather than
 * "small modern texture".
 */
export function shadeRamp(out: RGBA, size: number, t: Field, ramp: Palette, opts: ShadeOptions = {}): RGBA {
  const n = rampSize(ramp);
  const dither = opts.dither ?? 1;
  const level = opts.bayerLevel ?? 4;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = y * size + x;
      const d = dither === 0 ? 0 : bayerSigned(x, y, level) * dither;
      let idx = Math.round(clamp(t[i]!) * (n - 1) + d);
      idx = idx < 0 ? 0 : idx > n - 1 ? n - 1 : idx;
      const o = i * 4;
      const p = idx * 3;
      out[o] = ramp[p]!;
      out[o + 1] = ramp[p + 1]!;
      out[o + 2] = ramp[p + 2]!;
      if (!opts.keepAlpha) out[o + 3] = 255;
    }
  }
  return out;
}

/**
 * Two ramps blended by a mask, with the *selection itself* dithered — a hard
 * 1-bit choice per texel rather than a smooth blend. Period-correct and much
 * more readable than lerping colours (which mushes both materials into grey).
 */
export function shadeTwoRamps(
  out: RGBA,
  size: number,
  t: Field,
  mask: Field,
  rampA: Palette,
  rampB: Palette,
  opts: ShadeOptions = {},
): RGBA {
  const nA = rampSize(rampA);
  const nB = rampSize(rampB);
  const dither = opts.dither ?? 1;
  const level = opts.bayerLevel ?? 4;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = y * size + x;
      const bs = bayerSigned(x, y, level);
      const useB = mask[i]! + bs * 0.55 > 0.5;
      const ramp = useB ? rampB : rampA;
      const n = useB ? nB : nA;
      let idx = Math.round(clamp(t[i]!) * (n - 1) + bs * dither);
      idx = idx < 0 ? 0 : idx > n - 1 ? n - 1 : idx;
      const o = i * 4;
      const p = idx * 3;
      out[o] = ramp[p]!;
      out[o + 1] = ramp[p + 1]!;
      out[o + 2] = ramp[p + 2]!;
      if (!opts.keepAlpha) out[o + 3] = 255;
    }
  }
  return out;
}

/** Write a field into the alpha channel, with optional dithered hard cutoff. */
export function writeAlpha(out: RGBA, size: number, a: Field, cutoff?: number, ditherLevel: 4 | 8 = 4): RGBA {
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = y * size + x;
      let v = clamp(a[i]!);
      if (cutoff !== undefined) {
        v = v + bayerSigned(x, y, ditherLevel) * 0.28 > cutoff ? 1 : 0;
      }
      out[i * 4 + 3] = Math.round(v * 255);
    }
  }
  return out;
}

/** Grey RGBA from a field — used for the height map output. */
export function fieldToGrey(size: number, f: Field): RGBA {
  const out = new Uint8ClampedArray(size * size * 4);
  for (let i = 0; i < f.length; i++) {
    const v = Math.round(clamp(f[i]!) * 255);
    out[i * 4] = v;
    out[i * 4 + 1] = v;
    out[i * 4 + 2] = v;
    out[i * 4 + 3] = 255;
  }
  return out;
}

/* ------------------------------------------------------------------ *
 * Lighting
 * ------------------------------------------------------------------ */

export interface ShadeLightOptions {
  /** Light direction in texture space. Default is upper-left, the era default. */
  lx?: number;
  ly?: number;
  lz?: number;
  /** How strongly the height slope tilts the surface normal. */
  scale?: number;
  /** Ambient occlusion strength from local cavity. */
  ao?: number;
  /** Radius of the cavity comparison blur. */
  aoRadius?: number;
}

/**
 * Bake soft directional light + cavity AO from a height field.
 * Returns a field centred on 0.5 (0.5 = flat lit); add it into your tone field.
 * Era artists painted this in by hand; deriving it from the same height field
 * that produces the normal map keeps albedo and geometry in agreement.
 */
export function heightShade(size: number, h: Field, opts: ShadeLightOptions = {}): Field {
  const lx = opts.lx ?? -0.55;
  const ly = opts.ly ?? -0.55;
  const lz = opts.lz ?? 0.63;
  const ll = Math.hypot(lx, ly, lz) || 1;
  const nlx = lx / ll;
  const nly = ly / ll;
  const nlz = lz / ll;
  const scale = opts.scale ?? 8;
  const aoAmt = opts.ao ?? 0.5;
  const aoR = opts.aoRadius ?? 3;
  const blurred = aoAmt > 0 ? blurField(h, size, aoR, 2) : h;
  const out = new Float32Array(h.length);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = y * size + x;
      const l = fetchWrap(h, size, x - 1, y);
      const r = fetchWrap(h, size, x + 1, y);
      const u = fetchWrap(h, size, x, y - 1);
      const d = fetchWrap(h, size, x, y + 1);
      const dx = (r - l) * scale;
      const dy = (d - u) * scale;
      const inv = 1 / Math.sqrt(dx * dx + dy * dy + 1);
      const nx = -dx * inv;
      const ny = -dy * inv;
      const nz = inv;
      const lam = nx * nlx + ny * nly + nz * nlz;
      const cav = aoAmt > 0 ? clamp(0.5 + (h[i]! - blurred[i]!) * 4) : 0.5;
      out[i] = clamp(lam * 0.5 + 0.5 + (cav - 0.5) * aoAmt * 2 - 0.5 + 0.5);
    }
  }
  return out;
}

/**
 * Tangent-space normal map from a height field. Wraps, so the normal map is as
 * seamless as the height it came from.
 */
export function heightToNormalRGBA(size: number, h: Field, strength = 3): RGBA {
  const out = new Uint8ClampedArray(size * size * 4);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const l = fetchWrap(h, size, x - 1, y);
      const r = fetchWrap(h, size, x + 1, y);
      const u = fetchWrap(h, size, x, y - 1);
      const d = fetchWrap(h, size, x, y + 1);
      const dx = (r - l) * strength * size * 0.02;
      const dy = (d - u) * strength * size * 0.02;
      const inv = 1 / Math.sqrt(dx * dx + dy * dy + 1);
      const nx = -dx * inv;
      // Three.js tangent-space normal maps use +Y up (OpenGL convention); image
      // rows run downward, so flip the vertical derivative sign.
      const ny = dy * inv;
      const nz = inv;
      const o = (y * size + x) * 4;
      out[o] = Math.round((nx * 0.5 + 0.5) * 255);
      out[o + 1] = Math.round((ny * 0.5 + 0.5) * 255);
      out[o + 2] = Math.round((nz * 0.5 + 0.5) * 255);
      out[o + 3] = 255;
    }
  }
  return out;
}
