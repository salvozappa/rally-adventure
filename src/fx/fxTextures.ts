/**
 * fxTextures.ts — procedural sprite atlas + decal texture for the drive FX.
 *
 * These are the *fallback* textures: if another module ships a hand-authored
 * (or better procedural) set, inject it once at boot with
 * `setFxTextureProvider()` and DriveFx will pick it up. Nothing else in the
 * FX package reaches for a texture directly.
 *
 * Everything is generated on a canvas at module-request time and cached, so
 * there is no network fetch and no load-order dance.
 */

import * as THREE from 'three';
import { fbm2, mulberry32, smoothstep, clamp } from '../render/proceduralNoise';

/* ------------------------------------------------------------------ *
 * The set the FX package consumes
 * ------------------------------------------------------------------ */

export interface FxTextureSet {
  /**
   * 2x2 sprite atlas. Cells, in `atlasCell` index order:
   *   0 — soft dust puff (very low contrast, no hard edge)
   *   1 — smoke wisp (higher contrast, torn edges)
   *   2 — dirt clod (small solid irregular chunk)
   *   3 — spark streak (bright core, vertical smear)
   */
  atlas: THREE.Texture;
  /** Tyre track ribbon: tread across U, tiles along V. Alpha only, in .a */
  track: THREE.Texture;
  /** Low-frequency turbulence, tiling. Used by the heat haze shimmer. */
  haze: THREE.Texture;
}

/** Number of cells per axis in the sprite atlas. */
export const ATLAS_GRID = 2;

export const ATLAS_DUST = 0;
export const ATLAS_SMOKE = 1;
export const ATLAS_CLOD = 2;
export const ATLAS_SPARK = 3;

/* ------------------------------------------------------------------ *
 * Injection seam
 * ------------------------------------------------------------------ */

let provider: (() => FxTextureSet) | null = null;
let cached: FxTextureSet | null = null;

/**
 * Hand the FX package a better texture set. Call before constructing DriveFx.
 * Passing `null` restores the built-in procedural fallbacks.
 */
export function setFxTextureProvider(fn: (() => FxTextureSet) | null): void {
  provider = fn;
  cached = null;
}

/** The texture set in use. Built once, then cached for the process. */
export function getFxTextures(): FxTextureSet {
  if (cached) return cached;
  cached = provider ? provider() : buildFallbackSet();
  return cached;
}

/** Drop the cache and free GPU memory. Only the owner of the set should call. */
export function disposeFxTextures(): void {
  if (!cached) return;
  cached.atlas.dispose();
  cached.track.dispose();
  cached.haze.dispose();
  cached = null;
}

/* ------------------------------------------------------------------ *
 * Canvas helpers
 * ------------------------------------------------------------------ */

function makeCanvas(size: number): { c: HTMLCanvasElement; d: ImageData; px: Uint8ClampedArray } {
  const c = document.createElement('canvas');
  c.width = size;
  c.height = size;
  const ctx = c.getContext('2d');
  if (!ctx) throw new Error('fxTextures: 2d context unavailable');
  const d = ctx.createImageData(size, size);
  return { c, d, px: d.data };
}

function commit(c: HTMLCanvasElement, d: ImageData): void {
  const ctx = c.getContext('2d');
  if (ctx) ctx.putImageData(d, 0, 0);
}

/* ------------------------------------------------------------------ *
 * Sprite atlas
 * ------------------------------------------------------------------ */

const CELL = 256;
const ATLAS_SIZE = CELL * ATLAS_GRID;

/**
 * Every sprite is written white with the shape carried entirely in alpha —
 * particle colour comes from the per-instance attribute, so one atlas serves
 * ochre dirt, grey rock smoke and green grass clippings alike.
 */
type CellPainter = (u: number, v: number, rnd: () => number) => { l: number; a: number };

function paintCell(
  px: Uint8ClampedArray,
  atlasSize: number,
  cx: number,
  cy: number,
  fn: (u: number, v: number) => { l: number; a: number },
): void {
  const x0 = cx * CELL;
  const y0 = cy * CELL;
  for (let y = 0; y < CELL; y++) {
    for (let x = 0; x < CELL; x++) {
      const u = (x + 0.5) / CELL;
      const v = (y + 0.5) / CELL;
      const { l, a } = fn(u, v);
      const i = ((y0 + y) * atlasSize + (x0 + x)) * 4;
      const lum = Math.round(clamp(l) * 255);
      px[i] = lum;
      px[i + 1] = lum;
      px[i + 2] = lum;
      px[i + 3] = Math.round(clamp(a) * 255);
    }
  }
}

/** Distance from centre in cell space, 0 at middle, 1 at the edge midpoint. */
function radial(u: number, v: number): number {
  const dx = u - 0.5;
  const dy = v - 0.5;
  return Math.sqrt(dx * dx + dy * dy) * 2;
}

function buildAtlas(): THREE.Texture {
  const { c, d, px } = makeCanvas(ATLAS_SIZE);

  // --- cell 0: dust puff -------------------------------------------------
  // A very soft ball with a gentle internal density variation. The falloff is
  // deliberately long: several of these stacked read as a volume, and no
  // single one ever shows a silhouette.
  paintCell(px, ATLAS_SIZE, 0, 0, (u, v) => {
    const r = radial(u, v);
    const warp = fbm2(u, v, { octaves: 3, period: 3, seed: 71 });
    // Push the silhouette in and out a little so it isn't a perfect circle.
    const rr = r * (1 + (warp - 0.5) * 0.35);
    let a = 1 - smoothstep(0.05, 1.0, rr);
    a *= a; // long tail
    // Interior mottling keeps big sprites from looking like an airbrush dot.
    const grain = fbm2(u, v, { octaves: 4, period: 5, seed: 913, gain: 0.55 });
    a *= 0.72 + grain * 0.56;
    // Slightly brighter core reads as light scattering through the plume.
    const l = 0.82 + (1 - smoothstep(0.0, 0.7, rr)) * 0.18;
    return { l, a: a * 0.95 };
  });

  // --- cell 1: smoke wisp ------------------------------------------------
  // Torn, higher-contrast, off-centre. Tyre smoke and exhaust want structure.
  paintCell(px, ATLAS_SIZE, 1, 0, (u, v) => {
    const r = radial(u, v);
    const w1 = fbm2(u, v, { octaves: 5, period: 4, seed: 4001, gain: 0.58, turbulence: true });
    const rr = r * (1 + (w1 - 0.5) * 0.85);
    let a = 1 - smoothstep(0.0, 0.95, rr);
    a = a * a * (0.35 + w1 * 1.05);
    // Erode a few holes so the edge is ragged rather than feathered.
    const holes = fbm2(u, v, { octaves: 3, period: 8, seed: 5507 });
    a *= smoothstep(0.22, 0.55, holes * 0.55 + (1 - rr) * 0.6);
    const l = 0.75 + (1 - smoothstep(0.0, 0.6, rr)) * 0.25;
    return { l, a: clamp(a) * 0.9 };
  });

  // --- cell 2: dirt clod -------------------------------------------------
  // Small, opaque, irregular. Read at 6-20 px on screen, so silhouette is
  // everything: keep the alpha near-binary with a 1px soft rim.
  paintCell(px, ATLAS_SIZE, 0, 1, (u, v) => {
    const dx = u - 0.5;
    const dy = v - 0.5;
    const ang = Math.atan2(dy, dx);
    const r = Math.sqrt(dx * dx + dy * dy) * 2;
    // Angular lumpiness — a potato, not a pebble.
    const lump =
      0.62 +
      0.13 * Math.sin(ang * 3.0 + 0.7) +
      0.08 * Math.sin(ang * 5.0 - 1.9) +
      0.05 * Math.sin(ang * 8.0 + 2.7);
    let a = 1 - smoothstep(lump - 0.09, lump + 0.02, r);
    // A couple of detached crumbs travelling with the main chunk.
    const crumb = (ox: number, oy: number, rad: number): number => {
      const q = Math.sqrt((u - ox) ** 2 + (v - oy) ** 2);
      return 1 - smoothstep(rad * 0.6, rad, q);
    };
    a = Math.max(a, crumb(0.79, 0.31, 0.075) * 0.9);
    a = Math.max(a, crumb(0.24, 0.78, 0.055) * 0.85);
    // Shade the underside so a tumbling clod has some form.
    const shade = fbm2(u, v, { octaves: 3, period: 6, seed: 313 });
    const l = clamp(0.55 + shade * 0.5 - (v - 0.5) * 0.45, 0.25, 1.2);
    return { l, a: clamp(a) };
  });

  // --- cell 3: spark streak ---------------------------------------------
  // Bright core, vertical smear. Drawn additively and stretched along
  // velocity by the STREAK mode, so the sprite itself stays modest.
  paintCell(px, ATLAS_SIZE, 1, 1, (u, v) => {
    const dx = (u - 0.5) * 2.6; // narrow
    const dy = (v - 0.5) * 1.0;
    const q = Math.sqrt(dx * dx + dy * dy);
    let a = 1 - smoothstep(0.0, 0.85, q);
    a = a * a * a;
    // Hot white core.
    const core = 1 - smoothstep(0.0, 0.28, q);
    return { l: 0.6 + core * 0.75, a: clamp(a + core * 0.5) };
  });

  commit(c, d);

  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.generateMipmaps = true;
  tex.anisotropy = 2;
  // Cells are addressed by index, and the index arithmetic in the particle
  // shader counts rows downwards from the top of the image. Uploading flipped
  // would swap cell 0 with cell 2 and cell 1 with cell 3 — dust would draw as
  // a dirt clod, which is a very confusing bug to look at.
  tex.flipY = false;
  tex.needsUpdate = true;
  tex.name = 'fx.atlas';
  return tex;
}

/* ------------------------------------------------------------------ *
 * Tyre track ribbon
 * ------------------------------------------------------------------ */

/**
 * Tread pattern. U runs across the tyre (0..1 = shoulder to shoulder), V runs
 * along the direction of travel and tiles. The shoulders fade to zero so the
 * ribbon has no cut edge; the lugs give the track a rhythm you can read at
 * speed without it turning into a stripe.
 */
function buildTrack(): THREE.Texture {
  const SIZE = 128;
  const { c, d, px } = makeCanvas(SIZE);
  const rnd = mulberry32(20260807);
  // Pre-roll so the pattern isn't correlated with the noise field below.
  for (let i = 0; i < 16; i++) rnd();

  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const u = (x + 0.5) / SIZE;
      const v = (y + 0.5) / SIZE;

      // Across-tyre profile: full in the middle, feathered at the shoulders,
      // with a slightly heavier print where the block edges bite.
      const s = Math.abs(u - 0.5) * 2;
      let across = 1 - smoothstep(0.55, 1.0, s);
      across *= 0.78 + 0.22 * Math.cos(u * Math.PI * 2 * 2);

      // Chevron lugs. Period 4 along V so the 128px tile repeats cleanly.
      const chev = Math.abs(((v * 4 + (u < 0.5 ? u : 1 - u) * 0.9) % 1) - 0.5) * 2;
      const lug = smoothstep(0.30, 0.62, chev);

      // Broken-up soil, so no two metres of track look identical.
      const grain = fbm2(u, v, { octaves: 4, period: 6, seed: 8821, gain: 0.55 });

      let a = across * (0.55 + lug * 0.45) * (0.62 + grain * 0.62);
      a = clamp(a);

      // Luminance carries a subtle relief: the rut floor is darker than the
      // ridge between the lugs, which is what sells a track as an impression
      // in soil rather than a painted stripe.
      const l = clamp(0.45 + lug * 0.4 + grain * 0.3, 0, 1);

      const i = (y * SIZE + x) * 4;
      const lum = Math.round(l * 255);
      px[i] = lum;
      px[i + 1] = lum;
      px[i + 2] = lum;
      px[i + 3] = Math.round(a * 255);
    }
  }
  commit(c, d);

  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.generateMipmaps = true;
  tex.anisotropy = 4;
  tex.needsUpdate = true;
  tex.name = 'fx.track';
  return tex;
}

/* ------------------------------------------------------------------ *
 * Heat haze turbulence
 * ------------------------------------------------------------------ */

function buildHaze(): THREE.Texture {
  const SIZE = 128;
  const { c, d, px } = makeCanvas(SIZE);
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const u = (x + 0.5) / SIZE;
      const v = (y + 0.5) / SIZE;
      // Two decorrelated octave stacks, stored in R and G, so the shader can
      // scroll them at different rates and get a non-repeating shimmer.
      const a = fbm2(u, v, { octaves: 4, period: 4, seed: 1201, gain: 0.55 });
      const b = fbm2(u, v, { octaves: 4, period: 6, seed: 7717, gain: 0.55 });
      const i = (y * SIZE + x) * 4;
      px[i] = Math.round(clamp(a) * 255);
      px[i + 1] = Math.round(clamp(b) * 255);
      px[i + 2] = 128;
      px[i + 3] = 255;
    }
  }
  commit(c, d);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.NoColorSpace;
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.generateMipmaps = true;
  tex.needsUpdate = true;
  tex.name = 'fx.haze';
  return tex;
}

function buildFallbackSet(): FxTextureSet {
  return { atlas: buildAtlas(), track: buildTrack(), haze: buildHaze() };
}

// `CellPainter` documents the shape of the per-cell closures above; exported so
// a replacement provider can reuse the convention if it wants to.
export type { CellPainter };
