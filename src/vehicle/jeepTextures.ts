/**
 * Procedural textures for the Jeep.
 *
 * These are the *fallback* set: if the renderer's shared texture library
 * (`src/render/textures.ts`) is available, the host can inject it through
 * `setJeepTextureProvider()` before the first `JeepModel` is constructed and
 * the model will use those instead. Nothing in this file imports the render
 * package, so the vehicle builds standalone (see `jeep.preview.ts`).
 *
 * Everything is canvas-generated, tileable, and filtered with NearestFilter on
 * magnification for the low-res era look.
 */
import * as THREE from 'three';

export interface JeepTextureSet {
  /** Weathered painted panel. Luminance only — tinted by material.color. */
  paint: THREE.Texture;
  /** Same weathering at a coarser scale, for large flat panels (hood/doors). */
  paintPanel: THREE.Texture;
  /** Scuffed bare/painted steel: chassis, axles, bumpers. */
  steel: THREE.Texture;
  /** Bright polished trim. */
  chrome: THREE.Texture;
  /** Tyre tread rubber. */
  rubber: THREE.Texture;
  /** Tyre sidewall, with moulded lettering ring. */
  sidewall: THREE.Texture;
  /** Vinyl seat / dash grain. */
  vinyl: THREE.Texture;
  /** Dark textured plastic: flares, grille surround, mud flaps. */
  plastic: THREE.Texture;
  /** Grubby underbody: mud-caked steel. */
  grime: THREE.Texture;
  /** Instrument cluster face (two round gauges). */
  gauges: THREE.Texture;
  /** Glass: faint smear + tint. */
  glass: THREE.Texture;
}

export type JeepTextureProvider = () => JeepTextureSet;

let provider: JeepTextureProvider | null = null;
let cached: JeepTextureSet | null = null;

/** Swap in an externally-authored texture set (e.g. the shared render library). */
export function setJeepTextureProvider(p: JeepTextureProvider | null): void {
  provider = p;
  cached = null;
}

export function getJeepTextures(): JeepTextureSet {
  if (!cached) cached = provider ? provider() : buildJeepTextures();
  return cached;
}

export function disposeJeepTextures(): void {
  if (!cached) return;
  for (const t of Object.values(cached) as THREE.Texture[]) t.dispose();
  cached = null;
}

// ---------------------------------------------------------------------------
// noise
// ---------------------------------------------------------------------------

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Tileable value noise on an integer lattice of `period` cells. */
function latticeNoise(period: number, seed: number): (x: number, y: number) => number {
  const rnd = mulberry32(seed);
  const g = new Float32Array(period * period);
  for (let i = 0; i < g.length; i++) g[i] = rnd();
  const at = (ix: number, iy: number): number =>
    g[(((iy % period) + period) % period) * period + (((ix % period) + period) % period)]!;
  const smooth = (t: number): number => t * t * (3 - 2 * t);
  return (x, y) => {
    const xi = Math.floor(x);
    const yi = Math.floor(y);
    const tx = smooth(x - xi);
    const ty = smooth(y - yi);
    const a = at(xi, yi) * (1 - tx) + at(xi + 1, yi) * tx;
    const b = at(xi, yi + 1) * (1 - tx) + at(xi + 1, yi + 1) * tx;
    return a * (1 - ty) + b * ty;
  };
}

/** Tileable fbm in [0,1]; `base` cells across the (unit) texture. */
function fbm(base: number, octaves: number, seed: number): (u: number, v: number) => number {
  const layers: Array<{ n: (x: number, y: number) => number; f: number; a: number }> = [];
  let amp = 1;
  let sum = 0;
  for (let o = 0; o < octaves; o++) {
    const f = base * 2 ** o;
    layers.push({ n: latticeNoise(f, seed + o * 7919), f, a: amp });
    sum += amp;
    amp *= 0.5;
  }
  return (u, v) => {
    let t = 0;
    for (const l of layers) t += l.n(u * l.f, v * l.f) * l.a;
    return t / sum;
  };
}

// ---------------------------------------------------------------------------
// canvas helpers
// ---------------------------------------------------------------------------

interface Painter {
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
  size: number;
}

function painter(size: number): Painter {
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('jeepTextures: 2d context unavailable');
  return { canvas, ctx, size };
}

/** Per-pixel pass: `fn` returns [r,g,b] in 0..255. */
function pixels(p: Painter, fn: (u: number, v: number, x: number, y: number) => [number, number, number]): void {
  const img = p.ctx.getImageData(0, 0, p.size, p.size);
  const d = img.data;
  for (let y = 0; y < p.size; y++) {
    for (let x = 0; x < p.size; x++) {
      const [r, g, b] = fn((x + 0.5) / p.size, (y + 0.5) / p.size, x, y);
      const i = (y * p.size + x) * 4;
      d[i] = r;
      d[i + 1] = g;
      d[i + 2] = b;
      d[i + 3] = 255;
    }
  }
  p.ctx.putImageData(img, 0, 0);
}

function finish(p: Painter, repeat = 1, srgb = true): THREE.Texture {
  const t = new THREE.CanvasTexture(p.canvas);
  t.wrapS = THREE.RepeatWrapping;
  t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(repeat, repeat);
  t.magFilter = THREE.NearestFilter;
  t.minFilter = THREE.LinearMipmapLinearFilter;
  t.generateMipmaps = true;
  t.anisotropy = 4;
  if (srgb) t.colorSpace = THREE.SRGBColorSpace;
  t.needsUpdate = true;
  return t;
}

const clamp255 = (v: number): number => (v < 0 ? 0 : v > 255 ? 255 : v | 0);
const grey = (v: number): [number, number, number] => [clamp255(v), clamp255(v), clamp255(v)];

// ---------------------------------------------------------------------------
// the set
// ---------------------------------------------------------------------------

export function buildJeepTextures(): JeepTextureSet {
  return {
    paint: paintTexture(128, 1.0, 11),
    paintPanel: paintTexture(128, 0.55, 47),
    steel: steelTexture(),
    chrome: chromeTexture(),
    rubber: rubberTexture(),
    sidewall: sidewallTexture(),
    vinyl: vinylTexture(),
    plastic: plasticTexture(),
    grime: grimeTexture(),
    gauges: gaugeTexture(),
    glass: glassTexture(),
  };
}

/**
 * Weathered paint. Mid-bright base so `material.color` does the tinting: dirt
 * settles into a low-frequency mottle, with chalky sun-faded highlights and a
 * scattering of stone chips that punch through to primer.
 */
function paintTexture(size: number, scale: number, seed: number): THREE.Texture {
  const p = painter(size);
  const dirt = fbm(Math.max(2, Math.round(6 * scale)), 4, seed);
  const fade = fbm(Math.max(2, Math.round(3 * scale)), 2, seed + 101);
  const speck = fbm(Math.round(32 * scale) || 16, 1, seed + 202);
  pixels(p, (u, v) => {
    let l = 196;
    l += (fade(u, v) - 0.5) * 26; // uneven fade
    l -= (1 - dirt(u, v)) * 34; // grime pooling
    l += (speck(u, v) - 0.5) * 12; // orange-peel
    // stone chips: sparse dark pits
    if (speck(u * 1.7 + 0.3, v * 1.7 + 0.11) > 0.87) l -= 62;
    return grey(l);
  });
  // fine horizontal polish grain, drawn after so it survives the pixel pass
  p.ctx.globalAlpha = 0.06;
  p.ctx.strokeStyle = '#ffffff';
  const rnd = mulberry32(seed + 900);
  for (let i = 0; i < size * 0.5; i++) {
    const y = Math.floor(rnd() * size) + 0.5;
    p.ctx.beginPath();
    p.ctx.moveTo(0, y);
    p.ctx.lineTo(size, y);
    p.ctx.stroke();
  }
  p.ctx.globalAlpha = 1;
  return finish(p, 1);
}

/** Scuffed steel for the frame, axles and bumpers. */
function steelTexture(): THREE.Texture {
  const p = painter(128);
  const n = fbm(8, 4, 313);
  const rust = fbm(5, 3, 707);
  pixels(p, (u, v) => {
    const l = 118 + (n(u, v) - 0.5) * 60;
    const r = Math.max(0, rust(u, v) - 0.62) * 2.4; // patches of oxide
    return [clamp255(l + r * 60), clamp255(l + r * 22), clamp255(l - r * 8)];
  });
  return finish(p, 1);
}

/** Polished trim — bright with a soft anisotropic banding. */
function chromeTexture(): THREE.Texture {
  const p = painter(64);
  const n = fbm(6, 3, 1201);
  pixels(p, (u, v) => {
    const band = 0.5 + 0.5 * Math.sin(v * Math.PI * 4);
    const l = 168 + band * 58 + (n(u, v) - 0.5) * 26;
    return grey(l);
  });
  return finish(p, 1);
}

/** Tread rubber: dark, matte, with a moulding-grain speckle. */
function rubberTexture(): THREE.Texture {
  const p = painter(64);
  const n = fbm(10, 3, 1777);
  const dust = fbm(4, 2, 1888);
  pixels(p, (u, v) => {
    const l = 40 + (n(u, v) - 0.5) * 26;
    const d = Math.max(0, dust(u, v) - 0.55) * 1.6; // dried dirt film
    return [clamp255(l + d * 66), clamp255(l + d * 48), clamp255(l + d * 30)];
  });
  return finish(p, 1);
}

/**
 * Sidewall. `u` runs the circumference and `v` across the profile, so the
 * lettering band lands as a moulded ring once the texture repeats around.
 */
function sidewallTexture(): THREE.Texture {
  const size = 128;
  const p = painter(size);
  const n = fbm(8, 3, 2211);
  pixels(p, (u, v) => grey(38 + (n(u, v) - 0.5) * 22 + Math.sin(v * Math.PI * 18) * 4));
  const ctx = p.ctx;
  // raised ring + moulded lettering, centred on the sidewall band
  ctx.strokeStyle = 'rgba(120,118,112,0.55)';
  ctx.lineWidth = 2;
  for (const y of [size * 0.2, size * 0.86]) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(size, y);
    ctx.stroke();
  }
  ctx.fillStyle = 'rgba(146,142,134,0.8)';
  ctx.font = `bold ${Math.round(size * 0.115)}px sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText('MUD  TERRAIN', size * 0.5, size * 0.53);
  ctx.font = `bold ${Math.round(size * 0.085)}px sans-serif`;
  ctx.fillStyle = 'rgba(120,116,110,0.7)';
  ctx.fillText('35x12.5 R15', size * 0.5, size * 0.71);
  return finish(p, 1);
}

/** Seat and dash vinyl: a coarse pebbled grain. */
function vinylTexture(): THREE.Texture {
  const p = painter(96);
  const n = fbm(24, 2, 3131);
  const wear = fbm(4, 2, 3232);
  pixels(p, (u, v) => {
    const l = 92 + (n(u, v) - 0.5) * 44 + (wear(u, v) - 0.5) * 26;
    return [clamp255(l * 1.02), clamp255(l * 0.98), clamp255(l * 0.94)];
  });
  return finish(p, 1);
}

/** Flares / bumper caps: textured black plastic, sun-chalked. */
function plasticTexture(): THREE.Texture {
  const p = painter(96);
  const n = fbm(20, 3, 4141);
  const chalk = fbm(5, 2, 4242);
  pixels(p, (u, v) => grey(56 + (n(u, v) - 0.5) * 30 + Math.max(0, chalk(u, v) - 0.5) * 46));
  return finish(p, 1);
}

/** Caked mud over dark steel, for everything under the floor. */
function grimeTexture(): THREE.Texture {
  const p = painter(128);
  const mud = fbm(6, 4, 5151);
  const grit = fbm(28, 2, 5252);
  pixels(p, (u, v) => {
    const m = mud(u, v);
    const g = (grit(u, v) - 0.5) * 24;
    const l = 62 + m * 52 + g;
    return [clamp255(l * 1.12), clamp255(l * 0.94), clamp255(l * 0.72)];
  });
  return finish(p, 1);
}

/** Two round dials with needles, sitting in a dark binnacle. */
function gaugeTexture(): THREE.Texture {
  const size = 128;
  const p = painter(size);
  const ctx = p.ctx;
  ctx.fillStyle = '#161513';
  ctx.fillRect(0, 0, size, size);
  const dial = (cx: number, cy: number, r: number, needle: number): void => {
    ctx.fillStyle = '#0d0f0e';
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#8a8f8a';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.strokeStyle = '#d8d4c4';
    ctx.lineWidth = 1.5;
    for (let i = 0; i <= 10; i++) {
      const a = Math.PI * 0.75 + (i / 10) * Math.PI * 1.5;
      const c = Math.cos(a);
      const s = Math.sin(a);
      ctx.beginPath();
      ctx.moveTo(cx + c * r * 0.68, cy + s * r * 0.68);
      ctx.lineTo(cx + c * r * 0.86, cy + s * r * 0.86);
      ctx.stroke();
    }
    const a = Math.PI * 0.75 + needle * Math.PI * 1.5;
    ctx.strokeStyle = '#e2523a';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + Math.cos(a) * r * 0.78, cy + Math.sin(a) * r * 0.78);
    ctx.stroke();
  };
  dial(size * 0.27, size * 0.5, size * 0.22, 0.18);
  dial(size * 0.73, size * 0.5, size * 0.22, 0.3);
  // warning lamps between the dials
  for (let i = 0; i < 3; i++) {
    ctx.fillStyle = ['#c8a022', '#3a7f4a', '#b03a2a'][i]!;
    ctx.fillRect(size * 0.46, size * 0.3 + i * size * 0.14, size * 0.08, size * 0.06);
  }
  const t = finish(p, 1);
  t.wrapS = THREE.ClampToEdgeWrapping;
  t.wrapT = THREE.ClampToEdgeWrapping;
  return t;
}

/** Windscreen: faint wiper arc + dust film so glass isn't perfectly clean. */
function glassTexture(): THREE.Texture {
  const size = 128;
  const p = painter(size);
  const n = fbm(6, 3, 6161);
  pixels(p, (u, v) => {
    const l = 168 + (n(u, v) - 0.5) * 34 - v * 18;
    return [clamp255(l * 0.92), clamp255(l * 0.98), clamp255(l)];
  });
  const ctx = p.ctx;
  ctx.globalCompositeOperation = 'lighter';
  ctx.strokeStyle = 'rgba(255,255,255,0.10)';
  ctx.lineWidth = size * 0.24;
  for (const cx of [size * 0.3, size * 0.72]) {
    ctx.beginPath();
    ctx.arc(cx, size * 1.08, size * 0.62, Math.PI * 1.18, Math.PI * 1.82);
    ctx.stroke();
  }
  ctx.globalCompositeOperation = 'source-over';
  const t = finish(p, 1);
  t.wrapS = THREE.ClampToEdgeWrapping;
  t.wrapT = THREE.ClampToEdgeWrapping;
  return t;
}
