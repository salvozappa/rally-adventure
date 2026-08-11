import * as THREE from 'three';
import {
  PATTERNS,
  DEFAULT_SIZE,
  DEFAULT_SEED,
  TEXTURE_NAMES,
  RALLY_PALETTE,
  type PatternResult,
  type PatternOptions,
  type TextureName,
} from './texturePatterns';
import type { Field, RGBA } from './proceduralNoise';

export { RALLY_PALETTE, TEXTURE_NAMES };
export type { TextureName };

export interface TextureSet {
  map: THREE.Texture;
  normalMap?: THREE.Texture;
  /** Grayscale height, used by the terrain shader for height-based blending. */
  heightMap?: THREE.Texture;
  /** Source resolution, for shaders that need texel size. */
  size: number;
}

export interface TextureOptions extends PatternOptions {
  /** Texture repeat. Applied to every map in the set. */
  repeat?: number | [number, number];
  /**
   * Magnification filter. Nearest is the retro default and is what keeps
   * texels crisp instead of the smeared bilinear look of actual 1997 hardware —
   * period-authentic blur reads as "low quality" on a modern display, whereas
   * crisp texels read as a deliberate style.
   */
  magFilter?: THREE.MagnificationTextureFilter;
  /** Skip normal/height generation when the caller only needs albedo. */
  albedoOnly?: boolean;
}

/**
 * Procedural texture library.
 *
 * Everything is synthesised at load time from `texturePatterns` — there are no
 * image assets in this project. Results are cached by a key derived from the
 * name and every option that affects the pixels, so repeated requests share one
 * GPU upload.
 */
const cache = new Map<string, TextureSet>();
let anisotropyCap = 8;
const genTimes = new Map<TextureName, number>();

/** Called once by the Engine so we can request the driver's real anisotropy limit. */
export function configureTextures(renderer: THREE.WebGLRenderer): void {
  anisotropyCap = Math.min(16, renderer.capabilities.getMaxAnisotropy());
}

export function getTexture(name: TextureName, opts: TextureOptions = {}): TextureSet {
  const size = opts.size ?? DEFAULT_SIZE[name] ?? 128;
  const seed = opts.seed ?? DEFAULT_SEED[name] ?? 1;
  const key = [
    name,
    size,
    seed,
    opts.color ?? '-',
    opts.albedoOnly ? 'a' : 'f',
    opts.magFilter ?? 'n',
    Array.isArray(opts.repeat) ? opts.repeat.join('x') : (opts.repeat ?? 1),
  ].join('|');

  const hit = cache.get(key);
  if (hit) return hit;

  const t0 = performance.now();
  const fn = PATTERNS[name];
  if (!fn) throw new Error(`textures: unknown pattern "${name}"`);
  const result = fn({ size, seed, color: opts.color });
  const set = toTextureSet(result, opts);
  genTimes.set(name, performance.now() - t0);

  cache.set(key, set);
  return set;
}

/** Albedo only — the common case for props and UI. */
export function getMap(name: TextureName, opts: TextureOptions = {}): THREE.Texture {
  return getTexture(name, { ...opts, albedoOnly: true }).map;
}

/** Generation cost per texture, for the preview page and perf reporting. */
export function getGenerationTimes(): ReadonlyMap<TextureName, number> {
  return genTimes;
}

// ------------------------------------------------------------------ internals

function toTextureSet(r: PatternResult, opts: TextureOptions): TextureSet {
  const clampMode = r.clamp ? THREE.ClampToEdgeWrapping : THREE.RepeatWrapping;
  const mag = opts.magFilter ?? THREE.NearestFilter;

  const map = makeTexture(r.albedo, r.size, THREE.SRGBColorSpace, clampMode, mag);
  // Sprites and cutouts need alpha preserved through the whole chain.
  map.premultiplyAlpha = false;
  applyRepeat(map, opts.repeat);

  const set: TextureSet = { map, size: r.size };

  if (!opts.albedoOnly && !r.flat) {
    const nrm = makeTexture(
      heightToNormalRGBA(r.height, r.size, r.normalStrength ?? 1),
      r.size,
      THREE.NoColorSpace,
      clampMode,
      mag,
    );
    applyRepeat(nrm, opts.repeat);
    set.normalMap = nrm;

    const hgt = makeTexture(
      fieldToGrayRGBA(r.height),
      r.size,
      THREE.NoColorSpace,
      clampMode,
      mag,
    );
    applyRepeat(hgt, opts.repeat);
    set.heightMap = hgt;
  }

  return set;
}

function makeTexture(
  data: RGBA,
  size: number,
  colorSpace: THREE.ColorSpace,
  wrap: THREE.Wrapping,
  mag: THREE.MagnificationTextureFilter,
): THREE.DataTexture {
  const tex = new THREE.DataTexture(data, size, size, THREE.RGBAFormat, THREE.UnsignedByteType);
  tex.colorSpace = colorSpace;
  tex.wrapS = wrap;
  tex.wrapT = wrap;
  // Nearest magnification keeps texels crisp; trilinear minification kills the
  // shimmer that would otherwise crawl all over the terrain at speed. That
  // pairing is the whole retro-but-not-broken trick.
  tex.magFilter = mag;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.generateMipmaps = true;
  tex.anisotropy = anisotropyCap;
  tex.needsUpdate = true;
  return tex;
}

function applyRepeat(tex: THREE.Texture, repeat?: number | [number, number]): void {
  if (repeat === undefined) return;
  const [rx, ry] = Array.isArray(repeat) ? repeat : [repeat, repeat];
  tex.repeat.set(rx, ry);
}

/**
 * Sobel the height field into a tangent-space normal map.
 *
 * Wraps at the edges so the normal map stays seamless wherever the albedo is —
 * a non-wrapping derivative is a classic source of visible tile seams that only
 * show up under lighting, which makes it maddening to debug later.
 */
function heightToNormalRGBA(height: Field, size: number, strength: number): RGBA {
  const out = new Uint8ClampedArray(size * size * 4);
  const at = (x: number, y: number) => height[((y + size) % size) * size + ((x + size) % size)];
  // Scale so the perceived relief is roughly resolution-independent.
  const scale = strength * size * 0.03;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const tl = at(x - 1, y - 1), t = at(x, y - 1), tr = at(x + 1, y - 1);
      const l = at(x - 1, y), rr = at(x + 1, y);
      const bl = at(x - 1, y + 1), b = at(x, y + 1), br = at(x + 1, y + 1);

      const dx = (tr + 2 * rr + br) - (tl + 2 * l + bl);
      const dy = (bl + 2 * b + br) - (tl + 2 * t + tr);

      let nx = -dx * scale;
      let ny = -dy * scale;
      const nz = 1;
      const inv = 1 / Math.hypot(nx, ny, nz);
      nx *= inv;
      ny *= inv;

      const i = (y * size + x) * 4;
      out[i] = (nx * 0.5 + 0.5) * 255;
      out[i + 1] = (ny * 0.5 + 0.5) * 255;
      out[i + 2] = (nz * inv * 0.5 + 0.5) * 255;
      out[i + 3] = 255;
    }
  }
  return out;
}

function fieldToGrayRGBA(height: Field): RGBA {
  const out = new Uint8ClampedArray(height.length * 4);
  for (let i = 0; i < height.length; i++) {
    const v = height[i] * 255;
    const o = i * 4;
    out[o] = v;
    out[o + 1] = v;
    out[o + 2] = v;
    out[o + 3] = 255;
  }
  return out;
}

export function disposeTextures(): void {
  for (const set of cache.values()) {
    set.map.dispose();
    set.normalMap?.dispose();
    set.heightMap?.dispose();
  }
  cache.clear();
  genTimes.clear();
}
