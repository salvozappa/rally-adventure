/**
 * texturePatterns.ts — the individual procedural texture generators.
 *
 * Every generator returns raw pixel data (RGBA albedo + a Float32 height field).
 * Wrapping them into THREE textures, deriving normal maps and caching is
 * textures.ts's job; this file is pure synthesis and has no THREE dependency.
 *
 * HOUSE STYLE — the 1997 recipe, applied consistently:
 *  1. Build a height field out of DISCRETE features (pebbles, blades, planks,
 *     lugs), not a smooth noise blur. Era textures are busy and readable.
 *  2. Bake directional light + cavity AO from that same height field into the
 *     tone, so albedo and normal always agree.
 *  3. Sharpen, then push contrast. Low-res without contrast is just mush.
 *  4. Paint through a 12–24 entry palette with 4x4 ordered dithering. The
 *     dithered ramp is what makes it read as period-authentic rather than small.
 */

import {
  BAYER4,
  bayer,
  bayerSigned,
  blurField,
  buildRamp,
  clamp,
  cloneField,
  combine,
  contrastField,
  fbmField,
  fieldFromUV,
  fieldToGrey,
  hexToRgb,
  lerp,
  makeField,
  makeRGBA,
  mapField,
  mapFieldInPlace,
  mixField,
  mulberry32,
  normalizeField,
  paletteFrom,
  posterizeField,
  rgbToHex,
  sampleWrap,
  scatter,
  shadeRamp,
  shadeTwoRamps,
  sharpenField,
  smoothstep,
  stampDisc,
  stampEllipse,
  stampLine,
  streakField,
  thresholdField,
  worley,
  worleyEdges,
  writeAlpha,
  heightShade,
} from './proceduralNoise';
import type { Field, Palette, RGBA } from './proceduralNoise';

/* ------------------------------------------------------------------ *
 * Palette — sampled-from-reference rally-era colours.
 *
 * Warm, desaturated earth; olive-leaning greens; warm-grey rock. Mid-90s
 * hardware rendering ran everything through a dark, slightly muddy gamma, so
 * midtones sit low and highlights punch hard rather than rolling off.
 * ------------------------------------------------------------------ */

export const RALLY_PALETTE = {
  dirtDark: '#241a12',
  dirtMid: '#553d28',
  dirtLight: '#8c6b47',
  dirtDust: '#b89870',

  grassDark: '#15220f',
  grassMid: '#3a5a20',
  grassLight: '#6b9138',
  strawDark: '#3c3117',
  strawMid: '#7a6630',
  strawLight: '#bda264',

  rockDark: '#1f1b18',
  rockMid: '#4b433b',
  rockLight: '#8b8073',
  rockHigh: '#b3a898',

  sandDark: '#7a5c33',
  sandMid: '#bd9a63',
  sandLight: '#e6cf9c',

  mudDark: '#0e0906',
  mudMid: '#33220f',
  mudLight: '#5c452a',

  snowShadow: '#5e7196',
  snowMid: '#a9b8d0',
  snowLight: '#f4f8ff',

  barkDark: '#1b1410',
  barkMid: '#4a382a',
  barkLight: '#856a4f',

  metalDark: '#191c20',
  metalMid: '#5a626b',
  metalLight: '#a9b3bd',
  rust: '#7a3d1c',

  skyHaze: '#9db9cf',
  fog: '#a8b6b2',
} as const;

/* ------------------------------------------------------------------ *
 * Result shape
 * ------------------------------------------------------------------ */

export interface PatternResult {
  size: number;
  /** RGBA8 albedo. Alpha is meaningful only when `hasAlpha` is set. */
  albedo: RGBA;
  /** 0..1 height field; normal + height maps are derived from this. */
  height: Field;
  /** Cutout / soft alpha present in the albedo. */
  hasAlpha?: boolean;
  /** Suggested normal map strength for this material. */
  normalStrength?: number;
  /** Sprites and atlases must not repeat. */
  clamp?: boolean;
  /** Skip the normal/height outputs entirely (font atlas, flat sprites). */
  flat?: boolean;
}

export interface PatternOptions {
  size?: number;
  seed?: number;
  /** Base colour for tintable materials (painted metal). */
  color?: string;
}

export type PatternArgs = { size: number; seed: number; color?: string };
export type PatternFn = (a: PatternArgs) => PatternResult;

export type TextureName =
  // terrain
  | 'dirt'
  | 'grass'
  | 'rock'
  | 'gravel'
  | 'sand'
  | 'mud'
  | 'snow'
  // vehicle
  | 'paintedMetal'
  | 'tyre'
  | 'glass'
  | 'chrome'
  | 'vinyl'
  | 'headlight'
  // world
  | 'bark'
  | 'foliage'
  | 'rockFace'
  | 'planks'
  | 'corrugatedMetal'
  | 'scrub'
  | 'cliffStrata'
  // effects / ui
  | 'dust'
  | 'smoke'
  | 'skid'
  | 'fontAtlas';

/* ------------------------------------------------------------------ *
 * Shared helpers
 * ------------------------------------------------------------------ */

/** Bake light + AO from `h` into `tone`. The core "it looks 3D" step. */
function lit(size: number, tone: Field, h: Field, amount = 0.55, scale = 8, ao = 0.5): Field {
  const s = heightShade(size, h, { scale, ao });
  return mapField(tone, (v, i) => clamp(v + (s[i]! - 0.5) * amount));
}

/** Sharpen then push contrast — always the last tonal step before painting. */
function punch(size: number, f: Field, sharpen = 0.55, contrast = 1.25, pivot = 0.5): Field {
  return contrastField(sharpenField(f, size, 1, sharpen), contrast, pivot);
}

/** Fine per-texel grain. Nothing at this resolution should be perfectly smooth. */
function grain(size: number, seed: number, amount: number): Field {
  const rnd = mulberry32(seed);
  const f = new Float32Array(size * size);
  for (let i = 0; i < f.length; i++) f[i] = (rnd() - 0.5) * amount;
  return f;
}

function addInto(dst: Field, src: Field, scale = 1): Field {
  for (let i = 0; i < dst.length; i++) dst[i] = dst[i]! + src[i]! * scale;
  return dst;
}

/** Build a light-to-dark ramp around an arbitrary base colour. */
function rampFromColor(hex: string, n = 16): Palette {
  const [r, g, b] = hexToRgb(hex);
  const up = (t: number) => rgbToHex(r + (255 - r) * t, g + (255 - g) * t, b + (255 - b) * t);
  const down = (t: number) => rgbToHex(r * t, g * t, b * t + b * t * 0.12);
  return buildRamp([down(0.14), down(0.34), down(0.62), hex, up(0.34), up(0.72)], n);
}

const RAMP = {
  dirt: buildRamp(['#1b120b', '#33241633', '#553d28', '#7d5f3e', '#a37e55', '#c9a87c'].filter((s) => s.length <= 7), 20),
  dirtCool: buildRamp(['#191410', '#3a3026', '#5f5040', '#8a7458', '#b39a78'], 16),
  grass: buildRamp(['#0d1608', '#1d3010', '#33511b', '#4c7527', '#6b9c39', '#93bd58'], 20),
  straw: buildRamp(['#241c0d', '#463818', '#6e5a29', '#9a8140', '#c3a866', '#dfc98e'], 20),
  rock: buildRamp(['#161311', '#2c2723', '#484038', '#6c6355', '#948a7a', '#bcb2a0'], 22),
  gravel: buildRamp(['#2c2822', '#443e36', '#635c51', '#87806f', '#aaa294', '#ccc5b5'], 22),
  sand: buildRamp(['#5b421f', '#8a6a3c', '#b58f57', '#d6b478', '#eed6a2', '#f8ebc9'], 20),
  mud: buildRamp(['#070403', '#1a1008', '#301f10', '#48331c', '#63492c', '#8a6c46'], 20),
  snow: buildRamp(['#414f6d', '#65789b', '#93a6c4', '#c0cee2', '#e4ecf7', '#ffffff'], 20),
  bark: buildRamp(['#0e0a07', '#241a12', '#3f2f21', '#5c4632', '#836548', '#a98a67'], 20),
  leaf: buildRamp(['#0b1607', '#17280d', '#274318', '#3a6224', '#548a33', '#78b04b'], 18),
  scrub: buildRamp(['#161206', '#2e2711', '#4c401d', '#6e5c2c', '#94803f', '#b8a45f'], 18),
  wood: buildRamp(['#150e08', '#2f2013', '#4d3620', '#6d4f31', '#8f6d47', '#b08d63'], 20),
  steel: buildRamp(['#0d0f12', '#22272d', '#3d454f', '#5e6874', '#8b95a1', '#c3ccd6'], 22),
  chrome: buildRamp(['#0a0c10', '#1c222b', '#3a4552', '#6b7a8b', '#a6b4c2', '#e8f0f8'], 24),
  rubber: buildRamp(['#080809', '#131417', '#1f2125', '#2d3036', '#3e424a', '#565b64'], 16),
  glass: buildRamp(['#08101a', '#0f1e2e', '#1a3247', '#2b4d66', '#4a7893', '#8fb6cc'], 18),
  vinyl: buildRamp(['#0a0908', '#171412', '#26211d', '#38312a', '#4c433a', '#655a4e'], 16),
  headlight: buildRamp(['#3a4247', '#68767e', '#9aa9b0', '#c8d6da', '#eef6f8', '#ffffff'], 18),
  dust: buildRamp(['#6b5236', '#94764f', '#b8996b', '#d6bb8e', '#eddcb6'], 14),
  smoke: buildRamp(['#22201e', '#3d3a36', '#5c5854', '#807a74', '#a8a29b'], 14),
} as const;

/* ================================================================== *
 * TERRAIN
 * ================================================================== */

/** Packed earth trail — the default driving surface. */
function dirt({ size, seed }: PatternArgs): PatternResult {
  // Broad tonal patches: where the trail is packed hard vs loose and dusty.
  const base = fbmField(size, { octaves: 5, period: 3, seed, gain: 0.55, warp: 0.09 });
  const patches = fbmField(size, { octaves: 2, period: 2, seed: seed + 91 });

  // Embedded stones. Two passes: a few chunky ones, lots of grit.
  const stones = makeField(size);
  const stoneId = makeField(size);
  scatter(size, Math.round(size * 3.4), seed + 3, (cx, cy, rnd) => {
    const r = 0.9 + rnd() * rnd() * 3.4;
    const id = rnd();
    stampEllipse(stones, size, cx, cy, r, r * (0.6 + rnd() * 0.5), rnd() * Math.PI, 0.55 + rnd() * 0.45, 'max', 0.5, 0.4);
    stampEllipse(stoneId, size, cx, cy, r, r * 0.8, 0, id, 'max', 0.05, 0.9);
  });

  // Hairline cracks in the packed crust. Kept fine and faint on purpose: big
  // polygonal plates read as desert playa, which is the wrong material for a
  // trail surface and, worse, is the same motif the rock family already uses —
  // at a glance the whole terrain then looks like one texture in five tints.
  const w = worley(size, 19, seed + 7, 0.95);
  const cracks = mapField(worleyEdges(w), (v) => 1 - smoothstep(0.01, 0.07, v));

  // Wheel drag smears the loose top layer along the direction of travel.
  const drag = streakField(base, size, 0, 1, 9, 0.8);

  const grit = fbmField(size, { octaves: 2, period: Math.max(8, size >> 3), seed: seed + 41, value: true, gain: 0.55 });

  // Stones and grit carry the surface; the crack layer is a garnish.
  const height = makeField(size);
  for (let i = 0; i < height.length; i++) {
    height[i] = clamp(
      0.34 * base[i]! + 0.16 * drag[i]! + 0.52 * stones[i]! + 0.2 * grit[i]! - 0.09 * cracks[i]! + 0.1,
    );
  }

  let tone = makeField(size);
  for (let i = 0; i < tone.length; i++) {
    const stoneTint = stones[i]! > 0.25 ? (stoneId[i]! - 0.5) * 0.42 + 0.12 : 0;
    tone[i] =
      0.40 +
      (base[i]! - 0.5) * 0.62 +
      (patches[i]! - 0.5) * 0.34 +
      (drag[i]! - 0.5) * 0.18 +
      stoneTint +
      (grit[i]! - 0.5) * 0.30 -
      cracks[i]! * 0.11;
  }
  addInto(tone, grain(size, seed + 5, 0.09));
  tone = lit(size, tone, height, 0.6, 9, 0.55);
  tone = punch(size, tone, 0.6, 1.3, 0.46);

  const albedo = makeRGBA(size);
  shadeRamp(albedo, size, tone, RAMP.dirt, { dither: 1.0 });
  return { size, albedo, height, normalStrength: 2.4 };
}

/** Patchy grass with dead-straw variation. */
function grass({ size, seed }: PatternArgs): PatternResult {
  // Where the turf is alive vs burnt off.
  const health = fbmField(size, { octaves: 3, period: 3, seed: seed + 17, warp: 0.14, gain: 0.55 });
  // Clump structure — grass grows in tufts, not as a lawn.
  const clumps = fbmField(size, { octaves: 4, period: 7, seed, warp: 0.08, gain: 0.5 });

  const blades = makeField(size);
  const bladeTip = makeField(size);
  const rnd = mulberry32(seed + 77);
  const bladeCount = Math.round(size * size * 0.055);
  for (let i = 0; i < bladeCount; i++) {
    const cx = rnd() * size;
    const cy = rnd() * size;
    // Denser blades where the clump field is high.
    if (sampleWrap(clumps, size, cx, cy) < 0.35 + rnd() * 0.35) continue;
    const len = 2.2 + rnd() * 5.0;
    const ang = -Math.PI / 2 + (rnd() - 0.5) * 1.5;
    const ex = cx + Math.cos(ang) * len;
    const ey = cy + Math.sin(ang) * len;
    stampLine(blades, size, cx, cy, ex, ey, 0.85 + rnd() * 0.5, 0.55 + rnd() * 0.45, 'max', 0.55);
    stampDisc(bladeTip, size, ex, ey, 1.1, 0.7 + rnd() * 0.3, 'max', 0.6);
  }

  // Bare soil showing through the gaps.
  const soil = mapField(clumps, (v) => 1 - smoothstep(0.24, 0.5, v));

  const height = makeField(size);
  for (let i = 0; i < height.length; i++) {
    height[i] = clamp(0.22 + 0.3 * clumps[i]! + 0.55 * blades[i]! - 0.2 * soil[i]!);
  }

  let tone = makeField(size);
  for (let i = 0; i < tone.length; i++) {
    tone[i] =
      0.34 +
      (clumps[i]! - 0.5) * 0.55 +
      blades[i]! * 0.42 +
      bladeTip[i]! * 0.3 -
      soil[i]! * 0.28;
  }
  addInto(tone, grain(size, seed + 9, 0.11));
  tone = lit(size, tone, height, 0.45, 7, 0.5);
  tone = punch(size, tone, 0.7, 1.35, 0.44);

  // Straw where health is low, green where it's high — dithered hard selection
  // so both materials stay saturated instead of averaging into khaki mush.
  const mask = mapField(health, (v) => clamp((0.62 - v) * 2.6 + 0.5));
  const albedo = makeRGBA(size);
  shadeTwoRamps(albedo, size, tone, mask, RAMP.grass, RAMP.straw, { dither: 1.0 });

  // Bare soil patches punched through in dirt colours.
  const dirtRamp = RAMP.dirt;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = y * size + x;
      if (soil[i]! + bayerSigned(x, y, 4) * 0.3 > 0.72) {
        const n = dirtRamp.length / 3;
        let idx = Math.round(clamp(tone[i]! * 0.9 + 0.05) * (n - 1) + bayerSigned(x, y, 4));
        idx = clamp(idx, 0, n - 1) | 0;
        albedo[i * 4] = dirtRamp[idx * 3]!;
        albedo[i * 4 + 1] = dirtRamp[idx * 3 + 1]!;
        albedo[i * 4 + 2] = dirtRamp[idx * 3 + 2]!;
      }
    }
  }
  return { size, albedo, height, normalStrength: 1.9 };
}

/** Stratified rock — horizontal sedimentary banding with vertical erosion. */
function stratifiedRock(
  { size, seed }: PatternArgs,
  bands: number,
  ramp: Palette,
  chaos: number,
  erosion: number,
): PatternResult {
  // Bend the strata: real bedding planes are never flat.
  const bend = fbmField(size, { octaves: 3, period: 2, seed: seed + 5, gain: 0.55 });
  const rough = fbmField(size, { octaves: 5, period: 5, seed: seed + 13, gain: 0.55, warp: chaos * 0.12 });
  const fine = fbmField(size, { octaves: 3, period: Math.max(10, size >> 3), seed: seed + 29, value: true });

  // Per-band tone lookup; bands must be an integer so floor() wraps cleanly.
  const bandRnd = mulberry32(seed + 300);
  const bandTone = new Float32Array(bands);
  const bandDepth = new Float32Array(bands);
  for (let i = 0; i < bands; i++) {
    bandTone[i] = bandRnd();
    bandDepth[i] = bandRnd();
  }

  const strata = makeField(size);
  const stratLine = makeField(size);
  const stratDepth = makeField(size);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = y * size + x;
      const vv = y / size + (bend[i]! - 0.5) * 0.13 * chaos + (rough[i]! - 0.5) * 0.03;
      const g = vv * bands;
      const bi = ((Math.floor(g) % bands) + bands) % bands;
      const frac = g - Math.floor(g);
      strata[i] = bandTone[bi]!;
      stratDepth[i] = bandDepth[bi]!;
      // Dark bedding line at the boundary between bands.
      stratLine[i] = Math.max(smoothstep(0.09, 0.0, frac), smoothstep(0.91, 1.0, frac));
    }
  }

  // Fracture network across the face.
  const w = worley(size, 5, seed + 61, 0.95, 'euclidean', 1.6);
  const fract = mapField(worleyEdges(w), (v) => 1 - smoothstep(0.02, 0.14, v));

  // Water erosion runs vertically down the face.
  const ero = streakField(rough, size, 0, 1, Math.round(18 * erosion) + 2, 0.9);

  const height = makeField(size);
  for (let i = 0; i < height.length; i++) {
    height[i] = clamp(
      0.2 +
        (stratDepth[i]! - 0.5) * 0.42 +
        rough[i]! * 0.36 +
        fine[i]! * 0.14 +
        (ero[i]! - 0.5) * 0.22 * erosion -
        stratLine[i]! * 0.42 -
        fract[i]! * 0.3,
    );
  }

  let tone = makeField(size);
  for (let i = 0; i < tone.length; i++) {
    tone[i] =
      0.40 +
      (strata[i]! - 0.5) * 0.5 +
      (rough[i]! - 0.5) * 0.5 +
      (fine[i]! - 0.5) * 0.2 +
      (ero[i]! - 0.5) * 0.26 * erosion -
      stratLine[i]! * 0.4 -
      fract[i]! * 0.3;
  }
  addInto(tone, grain(size, seed + 31, 0.08));
  tone = lit(size, tone, height, 0.66, 10, 0.6);
  tone = punch(size, tone, 0.65, 1.32, 0.46);

  const albedo = makeRGBA(size);
  shadeRamp(albedo, size, tone, ramp, { dither: 1.0 });
  return { size, albedo, height, normalStrength: 3.0 };
}

function rock(a: PatternArgs): PatternResult {
  return stratifiedRock(a, 6, RAMP.rock, 1.0, 1.0);
}

function rockFace(a: PatternArgs): PatternResult {
  // Broken-up boulder face rather than clean bedding: fewer, more chaotic bands
  // plus a heavy plate fracture pass on top.
  const r = stratifiedRock({ ...a, seed: a.seed + 404 }, 3, RAMP.rock, 2.2, 0.5);
  const { size } = a;
  const w = worley(size, 4, a.seed + 707, 1.0);
  const plates = worleyEdges(w);
  const edge = mapField(plates, (v) => 1 - smoothstep(0.02, 0.18, v));
  const facet = mapField(w.id, (v) => (v - 0.5) * 0.5);

  for (let i = 0; i < r.height.length; i++) {
    r.height[i] = clamp(r.height[i]! * 0.7 + 0.3 * (0.5 + facet[i]!) - edge[i]! * 0.4);
  }
  let tone = makeField(size);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = y * size + x;
      // Recover tone from the albedo we already produced, then re-facet it.
      tone[i] = clamp(r.albedo[i * 4 + 1]! / 255 + facet[i]! * 0.6 - edge[i]! * 0.35);
    }
  }
  tone = lit(size, tone, r.height, 0.5, 9, 0.5);
  tone = punch(size, tone, 0.5, 1.28, 0.46);
  shadeRamp(r.albedo, size, tone, RAMP.rock, { dither: 1.0 });
  r.normalStrength = 3.4;
  return r;
}

function cliffStrata(a: PatternArgs): PatternResult {
  return stratifiedRock({ ...a, seed: a.seed + 909 }, 11, RAMP.rock, 0.55, 1.6);
}

/** Loose gravel — a dense bed of discrete stones. */
function gravel({ size, seed }: PatternArgs): PatternResult {
  const bed = fbmField(size, { octaves: 3, period: 4, seed, gain: 0.5 });
  const stones = makeField(size);
  const stoneId = makeField(size, 0.5);
  const stoneEdge = makeField(size);

  // Big stones first, then progressively smaller fills — gives real size
  // distribution instead of a uniform bumpy field.
  const passes: [number, number, number][] = [
    [Math.round(size * 1.1), 2.6, 5.2],
    [Math.round(size * 3.2), 1.5, 3.0],
    [Math.round(size * 8.0), 0.8, 1.7],
  ];
  let ps = 0;
  for (const [count, rmin, rmax] of passes) {
    const rnd = mulberry32(seed + 200 + ps * 37);
    for (let i = 0; i < count; i++) {
      const cx = rnd() * size;
      const cy = rnd() * size;
      const r = rmin + rnd() * (rmax - rmin);
      const ry = r * (0.62 + rnd() * 0.5);
      const ang = rnd() * Math.PI;
      const id = rnd();
      stampEllipse(stones, size, cx, cy, r, ry, ang, 0.45 + r / 6, 'max', 0.55, 0.42);
      stampEllipse(stoneId, size, cx, cy, r, ry, ang, id, 'set', 0.02, 0.98);
      stampEllipse(stoneEdge, size, cx, cy, r, ry, ang, 1, 'set', 1.6, 0);
    }
    ps++;
  }
  const grit = fbmField(size, { octaves: 2, period: size >> 2, seed: seed + 55, value: true });

  const height = makeField(size);
  for (let i = 0; i < height.length; i++) {
    height[i] = clamp(0.12 + 0.18 * bed[i]! + 0.72 * stones[i]! + 0.1 * grit[i]!);
  }

  let tone = makeField(size);
  for (let i = 0; i < tone.length; i++) {
    tone[i] = 0.46 + (stoneId[i]! - 0.5) * 0.62 + (bed[i]! - 0.5) * 0.24 + (grit[i]! - 0.5) * 0.22 + stones[i]! * 0.18;
    // Dark gaps between stones read as depth. Kept shallow: the gaps cover most
    // of the texture, so a deep subtraction here drags the whole material to
    // near-black and gravel stops reading as a daylit ground surface.
    tone[i] = tone[i]! - (1 - smoothstep(0.02, 0.3, stoneEdge[i]!)) * 0.16;
  }
  addInto(tone, grain(size, seed + 3, 0.1));
  tone = lit(size, tone, height, 0.68, 11, 0.6);
  tone = punch(size, tone, 0.7, 1.34, 0.46);

  const albedo = makeRGBA(size);
  shadeRamp(albedo, size, tone, RAMP.gravel, { dither: 1.0 });
  return { size, albedo, height, normalStrength: 3.0 };
}

/** Wind-rippled sand. */
function sand({ size, seed }: PatternArgs): PatternResult {
  const drift = fbmField(size, { octaves: 4, period: 2, seed, gain: 0.55, warp: 0.1 });
  const warpA = fbmField(size, { octaves: 3, period: 3, seed: seed + 21 });

  // Ripple crests: a wave whose phase is dragged around by the drift field.
  const RIPPLES = 14;
  const ripple = fieldFromUV(size, (u, v, x, y) => {
    const ph = (v + (warpA[y * size + x]! - 0.5) * 0.22 + (drift[y * size + x]! - 0.5) * 0.1) * RIPPLES;
    // Sharp crest, wide trough — how wind ripples actually section.
    const s = Math.sin(ph * Math.PI * 2) * 0.5 + 0.5;
    return Math.pow(s, 1.7);
  });

  const grains = makeField(size);
  scatter(size, Math.round(size * size * 0.02), seed + 8, (cx, cy, rnd) => {
    stampDisc(grains, size, cx, cy, 0.7 + rnd() * 0.5, 0.5 + rnd() * 0.5, 'max', 0.5, 0.6);
  });

  const height = makeField(size);
  for (let i = 0; i < height.length; i++) {
    height[i] = clamp(0.25 + 0.3 * drift[i]! + 0.42 * ripple[i]! + 0.1 * grains[i]!);
  }

  let tone = makeField(size);
  for (let i = 0; i < tone.length; i++) {
    tone[i] = 0.52 + (drift[i]! - 0.5) * 0.3 + (ripple[i]! - 0.5) * 0.24 + grains[i]! * 0.18;
  }
  addInto(tone, grain(size, seed + 12, 0.13));
  // Sand is a low-contrast material, so the ripple SHADOW has to do the work:
  // strong directional light, weak base tone variation.
  tone = lit(size, tone, height, 0.72, 13, 0.35);
  tone = punch(size, tone, 0.5, 1.18, 0.55);

  const albedo = makeRGBA(size);
  shadeRamp(albedo, size, tone, RAMP.sand, { dither: 1.1 });
  return { size, albedo, height, normalStrength: 2.0 };
}

/** Wet churned mud with wheel ruts and standing water. */
function mud({ size, seed }: PatternArgs): PatternResult {
  const base = fbmField(size, { octaves: 5, period: 4, seed, gain: 0.58, warp: 0.14 });
  const smear = streakField(base, size, 0, 1, 14, 0.86);

  // Two parallel ruts running with the direction of travel. Cosine keeps them
  // seamless across the vertical wrap and they never intersect the U seam.
  const ruts = fieldFromUV(size, (u) => {
    const a = Math.exp(-Math.pow((u - 0.27) / 0.1, 2));
    const b = Math.exp(-Math.pow((u - 0.73) / 0.1, 2));
    return Math.max(a, b);
  });
  // Chevron tread imprints inside the ruts.
  const tread = fieldFromUV(size, (u, v, x, y) => {
    const rows = 10;
    const p = (v + Math.abs(u - 0.5) * 0.55) * rows;
    const s = Math.abs((p - Math.floor(p)) - 0.5) * 2;
    return smoothstep(0.55, 0.15, s) * ruts[y * size + x]!;
  });

  const clods = makeField(size);
  scatter(size, Math.round(size * 2.2), seed + 4, (cx, cy, rnd) => {
    const r = 1.2 + rnd() * rnd() * 4.5;
    stampEllipse(clods, size, cx, cy, r, r * (0.55 + rnd() * 0.55), rnd() * Math.PI, 0.5 + rnd() * 0.5, 'max', 0.6, 0.35);
  });

  // Standing water: dark, smooth, and slightly brighter at the rim (sky glint).
  const puddleF = fbmField(size, { octaves: 3, period: 3, seed: seed + 66, warp: 0.12 });
  const puddle = mapField(puddleF, (v) => smoothstep(0.56, 0.68, v));

  const height = makeField(size);
  for (let i = 0; i < height.length; i++) {
    const dry = 0.2 + 0.28 * base[i]! + 0.14 * smear[i]! + 0.45 * clods[i]! - 0.3 * ruts[i]! - 0.22 * tread[i]!;
    height[i] = clamp(lerp(dry, 0.05, puddle[i]!));
  }

  let tone = makeField(size);
  for (let i = 0; i < tone.length; i++) {
    tone[i] =
      0.42 + (base[i]! - 0.5) * 0.62 + (smear[i]! - 0.5) * 0.3 + clods[i]! * 0.24 - ruts[i]! * 0.24 - tread[i]! * 0.22;
  }
  addInto(tone, grain(size, seed + 19, 0.09));
  tone = lit(size, tone, height, 0.62, 10, 0.55);

  // Wet sheen: a hard specular streak baked in where the surface faces the light.
  const sheen = heightShade(size, height, { scale: 16, ao: 0, lx: -0.35, ly: -0.8, lz: 0.5 });
  for (let i = 0; i < tone.length; i++) {
    const wet = clamp(puddle[i]! * 0.85 + 0.28);
    const spec = Math.pow(clamp((sheen[i]! - 0.62) / 0.38), 2.2);
    tone[i] = clamp(tone[i]! * lerp(1, 0.42, puddle[i]!) + spec * wet * 0.75);
  }
  tone = punch(size, tone, 0.6, 1.36, 0.42);

  const albedo = makeRGBA(size);
  shadeRamp(albedo, size, tone, RAMP.mud, { dither: 1.0 });
  return { size, albedo, height, normalStrength: 2.6 };
}

/** Snow — drifted, crusted, with sparkle. */
function snow({ size, seed }: PatternArgs): PatternResult {
  const drift = fbmField(size, { octaves: 4, period: 3, seed, gain: 0.55, warp: 0.12 });
  const windRipple = fieldFromUV(size, (u, v, x, y) => {
    const ph = (u * 0.35 + v + (drift[y * size + x]! - 0.5) * 0.3) * 9;
    return Math.pow(Math.sin(ph * Math.PI * 2) * 0.5 + 0.5, 2.0);
  });
  // Wind-scoured crust breaking into plates. Fine and shallow — snow is shaped
  // by drift and sastrugi, not by the polygonal cracking the rock family uses.
  const w = worley(size, 14, seed + 12, 0.9);
  const crust = mapField(worleyEdges(w), (v) => 1 - smoothstep(0.02, 0.09, v));

  const lumps = makeField(size);
  scatter(size, Math.round(size * 1.6), seed + 21, (cx, cy, rnd) => {
    stampDisc(lumps, size, cx, cy, 1.5 + rnd() * 4, 0.4 + rnd() * 0.4, 'max', 0.7, 0.2);
  });

  const height = makeField(size);
  for (let i = 0; i < height.length; i++) {
    height[i] = clamp(0.3 + 0.34 * drift[i]! + 0.22 * windRipple[i]! + 0.24 * lumps[i]! - 0.24 * crust[i]!);
  }

  let tone = makeField(size);
  for (let i = 0; i < tone.length; i++) {
    tone[i] = 0.74 + (drift[i]! - 0.5) * 0.18 + (windRipple[i]! - 0.5) * 0.14 - crust[i]! * 0.24;
  }
  // Snow has almost no albedo variation — all of the form comes from shadow.
  tone = lit(size, tone, height, 0.9, 15, 0.55);
  addInto(tone, grain(size, seed + 33, 0.07));

  // Ice crystal sparkle: isolated maximum-brightness texels.
  const sp = mulberry32(seed + 501);
  for (let i = 0; i < tone.length; i++) {
    if (sp() < 0.012) tone[i] = 1.0;
  }
  tone = punch(size, tone, 0.45, 1.22, 0.62);

  const albedo = makeRGBA(size);
  shadeRamp(albedo, size, tone, RAMP.snow, { dither: 1.0 });
  return { size, albedo, height, normalStrength: 1.8 };
}

/* ================================================================== *
 * VEHICLE
 * ================================================================== */

/** Painted body panel: weathered, flaked, with dirt accumulation. */
function paintedMetal({ size, seed, color }: PatternArgs): PatternResult {
  const ramp = rampFromColor(color ?? '#b4402a', 20);
  const dirtRamp = RAMP.dirt;

  // Very subtle panel undulation — old game cars were never dead flat.
  const undulate = fbmField(size, { octaves: 3, period: 2, seed, gain: 0.5 });
  // Metallic flake: fine, high-frequency, low amplitude.
  const flake = fbmField(size, { octaves: 2, period: size >> 2, seed: seed + 11, value: true });

  // Scratches: directional, mostly horizontal (airflow / brush marks).
  const scratches = makeField(size);
  const rnd = mulberry32(seed + 700);
  for (let i = 0; i < Math.round(size * 0.9); i++) {
    const x0 = rnd() * size;
    const y0 = rnd() * size;
    const len = 4 + rnd() * rnd() * size * 0.4;
    const ang = (rnd() - 0.5) * 0.5;
    stampLine(scratches, size, x0, y0, x0 + Math.cos(ang) * len, y0 + Math.sin(ang) * len, 0.6, 0.5 + rnd() * 0.5, 'max', 0.6);
  }

  // Chipped paint / rust blooms.
  const chipMask = makeField(size);
  scatter(size, Math.round(size * 0.5), seed + 33, (cx, cy, rnd2) => {
    const r = 0.9 + rnd2() * rnd2() * 4.5;
    stampEllipse(chipMask, size, cx, cy, r, r * (0.6 + rnd2() * 0.6), rnd2() * Math.PI, 1, 'max', 0.5, 0.5);
  });
  const chipEdge = fbmField(size, { octaves: 3, period: size >> 3, seed: seed + 44 });
  const chips = combine(chipMask, chipEdge, (m, e) => smoothstep(0.45, 0.6, m * 0.6 + e * 0.5));

  // Grime accumulating in blotches (kept blotchy, not gradient, so it tiles).
  const grime = fbmField(size, { octaves: 4, period: 3, seed: seed + 55, warp: 0.16, gain: 0.55 });
  const grimeMask = mapField(grime, (v) => smoothstep(0.46, 0.68, v));

  const height = makeField(size);
  for (let i = 0; i < height.length; i++) {
    height[i] = clamp(0.55 + (undulate[i]! - 0.5) * 0.3 - scratches[i]! * 0.18 - chips[i]! * 0.35 + flake[i]! * 0.05);
  }

  let tone = makeField(size);
  for (let i = 0; i < tone.length; i++) {
    tone[i] =
      0.56 +
      (undulate[i]! - 0.5) * 0.24 +
      (flake[i]! - 0.5) * 0.16 +
      scratches[i]! * 0.3 -
      chips[i]! * 0.3 -
      grimeMask[i]! * 0.2;
  }
  tone = lit(size, tone, height, 0.35, 6, 0.4);
  tone = punch(size, tone, 0.4, 1.2, 0.55);

  const albedo = makeRGBA(size);
  shadeRamp(albedo, size, tone, ramp, { dither: 1.0 });

  // Punch rust through the chips and dirt through the grime, both dithered.
  const rustRamp = buildRamp(['#2a1408', '#4d2410', '#7a3d1c', '#9e5c30', '#b8794a'], 14);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = y * size + x;
      const b = bayerSigned(x, y, 4);
      let src: Palette | null = null;
      let t = tone[i]!;
      if (chips[i]! + b * 0.3 > 0.55) {
        src = rustRamp;
        t = clamp(t * 0.8 + 0.1);
      } else if (grimeMask[i]! + b * 0.35 > 0.66) {
        src = dirtRamp;
        t = clamp(t * 0.7 + 0.05);
      }
      if (src) {
        const n = src.length / 3;
        const idx = clamp(Math.round(t * (n - 1) + b), 0, n - 1) | 0;
        albedo[i * 4] = src[idx * 3]!;
        albedo[i * 4 + 1] = src[idx * 3 + 1]!;
        albedo[i * 4 + 2] = src[idx * 3 + 2]!;
      }
    }
  }
  return { size, albedo, height, normalStrength: 1.1 };
}

/** Tyre: block tread pattern with sidewall-grade rubber grain. */
function tyre({ size, seed }: PatternArgs): PatternResult {
  const ROWS = 6; // lug rows around the circumference (V)
  const COLS = 3; // lug columns across the tread (U)
  const grainF = fbmField(size, { octaves: 3, period: size >> 3, seed, value: true, gain: 0.55 });
  const wobble = fbmField(size, { octaves: 3, period: 4, seed: seed + 3 });

  const lugs = fieldFromUV(size, (u, v, x, y) => {
    const i = y * size + x;
    const jit = (wobble[i]! - 0.5) * 0.035;
    // Centre rib runs continuously around the tyre.
    const ribD = Math.abs(u - 0.5);
    const rib = smoothstep(0.085, 0.055, ribD);

    // Shoulder/outer lug blocks, staggered row to row.
    const row = Math.floor((v + jit) * ROWS);
    const stagger = (((row % 2) + 2) % 2) * 0.5;
    const gu = (u + jit) * COLS + stagger;
    const fu = gu - Math.floor(gu);
    const fv = (v + jit) * ROWS - Math.floor((v + jit) * ROWS);
    // Angled leading edge gives the chevron bite.
    const skew = (fu - 0.5) * 0.22;
    const bu = smoothstep(0.06, 0.20, fu) * smoothstep(0.94, 0.80, fu);
    const bv = smoothstep(0.10, 0.26, fv + skew) * smoothstep(0.90, 0.74, fv + skew);
    const block = bu * bv;
    // Suppress blocks where the centre rib lives.
    return Math.max(rib, block * smoothstep(0.10, 0.16, ribD));
  });

  // Sipes: fine cuts across each lug.
  const sipes = fieldFromUV(size, (u, v, x, y) => {
    const i = y * size + x;
    const p = (v + (wobble[i]! - 0.5) * 0.02) * ROWS * 3;
    const s = Math.abs(p - Math.floor(p) - 0.5) * 2;
    return smoothstep(0.3, 0.05, s) * lugs[i]!;
  });

  const height = makeField(size);
  for (let i = 0; i < height.length; i++) {
    height[i] = clamp(0.12 + 0.7 * lugs[i]! + 0.12 * grainF[i]! - 0.3 * sipes[i]!);
  }

  let tone = makeField(size);
  for (let i = 0; i < tone.length; i++) {
    tone[i] = 0.3 + lugs[i]! * 0.34 + (grainF[i]! - 0.5) * 0.26 - sipes[i]! * 0.2;
  }
  addInto(tone, grain(size, seed + 5, 0.09));
  tone = lit(size, tone, height, 0.7, 12, 0.6);

  // Scuffed, dusty highlights on the lug crowns.
  const scuff = makeField(size);
  scatter(size, Math.round(size * 1.4), seed + 9, (cx, cy, rnd) => {
    stampEllipse(scuff, size, cx, cy, 1 + rnd() * 3.5, 1 + rnd() * 2, rnd() * Math.PI, 0.5 + rnd() * 0.5, 'max', 0.7, 0.3);
  });
  for (let i = 0; i < tone.length; i++) tone[i] = clamp(tone[i]! + scuff[i]! * lugs[i]! * 0.22);
  tone = punch(size, tone, 0.55, 1.3, 0.42);

  const albedo = makeRGBA(size);
  shadeRamp(albedo, size, tone, RAMP.rubber, { dither: 1.0 });
  return { size, albedo, height, normalStrength: 3.2 };
}

/** Window glass: dark, with a raked reflection band and edge grime. */
function glass({ size, seed }: PatternArgs): PatternResult {
  const smudge = fbmField(size, { octaves: 4, period: 3, seed, warp: 0.14, gain: 0.55 });
  const dustF = fbmField(size, { octaves: 3, period: 6, seed: seed + 7, warp: 0.1 });

  // Two hard diagonal reflection bands — the era shorthand for "this is glass".
  const refl = fieldFromUV(size, (u, v) => {
    const d = u * 0.72 + v * 0.7;
    const band = (x: number, c: number, w: number) => smoothstep(w, 0, Math.abs(((d - c) % 1 + 1.5) % 1 - 0.5));
    return Math.max(band(d, 0.0, 0.1) * 1.0, band(d, 0.0, 0.22) * 0.45);
  });

  // Wiper arc scratches.
  const wipe = makeField(size);
  const rnd = mulberry32(seed + 61);
  for (let i = 0; i < 26; i++) {
    const cy = rnd() * size;
    const amp = 2 + rnd() * 5;
    const ph = rnd() * Math.PI * 2;
    let px = 0;
    let py = cy + Math.sin(ph) * amp;
    for (let x = 1; x <= size; x++) {
      const y = cy + Math.sin((x / size) * Math.PI * 2 + ph) * amp;
      stampLine(wipe, size, px, py, x, y, 0.6, 0.35 + rnd() * 0.25, 'max', 0);
      px = x;
      py = y;
    }
  }

  const height = makeField(size);
  for (let i = 0; i < height.length; i++) {
    height[i] = clamp(0.5 + (smudge[i]! - 0.5) * 0.08 + wipe[i]! * 0.06);
  }

  let tone = makeField(size);
  for (let i = 0; i < tone.length; i++) {
    tone[i] = 0.3 + refl[i]! * 0.62 + (smudge[i]! - 0.5) * 0.24 + dustF[i]! * 0.2 + wipe[i]! * 0.22;
  }
  addInto(tone, grain(size, seed + 2, 0.05));
  tone = punch(size, tone, 0.4, 1.35, 0.4);

  const albedo = makeRGBA(size);
  shadeRamp(albedo, size, tone, RAMP.glass, { dither: 1.0 });
  return { size, albedo, height, normalStrength: 0.6 };
}

/** Chrome / bare steel trim: brushed vertical banding with hot highlights. */
function chrome({ size, seed }: PatternArgs): PatternResult {
  // Anisotropic noise: fast variation across, slow along -> brushed metal.
  const brush = fbmField(size, { octaves: 4, period: 4, seed, stretch: [8, 1], gain: 0.6 });
  const bandF = fbmField(size, { octaves: 2, period: 3, seed: seed + 5, stretch: [1, 3] });
  // Environment "reflection": bright sky in the upper half, dark ground in the
  // lower, with a hot flare along the horizon where they meet. That single
  // split is how 90s art faked chrome without a cube map, and it's what the eye
  // actually reads as polished metal — repeating sine bands just look like a
  // barcode.
  const env = fieldFromUV(size, (u, v) => {
    const sky = 0.62 + 0.3 * (1 - v);
    const ground = 0.1 + 0.16 * v;
    const horizon = Math.exp(-Math.pow((v - 0.52) / 0.045, 2)) * 0.5;
    return clamp(lerp(sky, ground, smoothstep(0.44, 0.58, v)) + horizon);
  });

  const pits = makeField(size);
  scatter(size, Math.round(size * 0.7), seed + 17, (cx, cy, rnd) => {
    stampDisc(pits, size, cx, cy, 0.7 + rnd() * 1.6, 0.4 + rnd() * 0.6, 'max', 0.6, 0.3);
  });

  const height = makeField(size);
  for (let i = 0; i < height.length; i++) {
    height[i] = clamp(0.6 + (brush[i]! - 0.5) * 0.12 - pits[i]! * 0.4);
  }

  let tone = makeField(size);
  for (let i = 0; i < tone.length; i++) {
    tone[i] = 0.2 + env[i]! * 0.66 + (brush[i]! - 0.5) * 0.34 + (bandF[i]! - 0.5) * 0.18 - pits[i]! * 0.32;
  }
  tone = lit(size, tone, height, 0.3, 8, 0.4);
  tone = punch(size, tone, 0.6, 1.45, 0.48);

  const albedo = makeRGBA(size);
  shadeRamp(albedo, size, tone, RAMP.chrome, { dither: 0.9 });
  return { size, albedo, height, normalStrength: 0.9 };
}

/** Worn interior vinyl: pebble grain with polished wear patches. */
function vinyl({ size, seed }: PatternArgs): PatternResult {
  // Classic embossed leather-grain cell structure.
  const w = worley(size, Math.max(10, size >> 4), seed, 1.0);
  const cell = mapField(worleyEdges(w), (v) => smoothstep(0.0, 0.22, v));
  const fine = worley(size, Math.max(20, size >> 3), seed + 5, 1.0);
  const cellFine = mapField(worleyEdges(fine), (v) => smoothstep(0.0, 0.3, v));

  const wear = fbmField(size, { octaves: 4, period: 3, seed: seed + 31, warp: 0.14 });
  const wearMask = mapField(wear, (v) => smoothstep(0.52, 0.72, v));

  // Cracks in the old vinyl.
  const cracks = makeField(size);
  const rnd = mulberry32(seed + 91);
  for (let i = 0; i < 14; i++) {
    let x = rnd() * size;
    let y = rnd() * size;
    let a = rnd() * Math.PI * 2;
    for (let s = 0; s < 20 + rnd() * 30; s++) {
      const nx = x + Math.cos(a) * 2.5;
      const ny = y + Math.sin(a) * 2.5;
      stampLine(cracks, size, x, y, nx, ny, 0.55, 0.9, 'max', 0);
      x = nx;
      y = ny;
      a += (rnd() - 0.5) * 0.9;
    }
  }

  const height = makeField(size);
  for (let i = 0; i < height.length; i++) {
    height[i] = clamp(0.35 + 0.4 * cell[i]! + 0.2 * cellFine[i]! - 0.4 * cracks[i]! - wearMask[i]! * 0.12);
  }

  let tone = makeField(size);
  for (let i = 0; i < tone.length; i++) {
    tone[i] = 0.4 + (cell[i]! - 0.5) * 0.3 + (cellFine[i]! - 0.5) * 0.18 + wearMask[i]! * 0.26 - cracks[i]! * 0.35;
  }
  addInto(tone, grain(size, seed + 4, 0.07));
  tone = lit(size, tone, height, 0.62, 12, 0.6);
  tone = punch(size, tone, 0.6, 1.28, 0.46);

  const albedo = makeRGBA(size);
  shadeRamp(albedo, size, tone, RAMP.vinyl, { dither: 1.0 });
  return { size, albedo, height, normalStrength: 2.0 };
}

/** Headlight lens: fluted fresnel prisms behind a bright reflector. */
function headlight({ size, seed }: PatternArgs): PatternResult {
  const RIBS_U = 10;
  const RIBS_V = 4;
  const flute = fieldFromUV(size, (u, v) => {
    const fu = u * RIBS_U - Math.floor(u * RIBS_U);
    const fv = v * RIBS_V - Math.floor(v * RIBS_V);
    // Cylindrical lens per rib: bright core, dark seam.
    const a = Math.pow(Math.sin(fu * Math.PI), 0.6);
    const b = 0.72 + 0.28 * Math.pow(Math.sin(fv * Math.PI), 0.5);
    return a * b;
  });
  const seam = fieldFromUV(size, (u, v) => {
    const fu = Math.abs(u * RIBS_U - Math.floor(u * RIBS_U) - 0.5) * 2;
    const fv = Math.abs(v * RIBS_V - Math.floor(v * RIBS_V) - 0.5) * 2;
    return Math.max(smoothstep(0.8, 1.0, fu), smoothstep(0.86, 1.0, fv) * 0.7);
  });
  const dirtF = fbmField(size, { octaves: 3, period: 4, seed, warp: 0.1 });

  const height = makeField(size);
  for (let i = 0; i < height.length; i++) height[i] = clamp(0.2 + 0.7 * flute[i]! - 0.3 * seam[i]!);

  let tone = makeField(size);
  for (let i = 0; i < tone.length; i++) {
    tone[i] = 0.28 + flute[i]! * 0.7 - seam[i]! * 0.34 - (dirtF[i]! - 0.5) * 0.16;
  }
  tone = lit(size, tone, height, 0.4, 9, 0.35);
  tone = punch(size, tone, 0.5, 1.4, 0.5);

  const albedo = makeRGBA(size);
  shadeRamp(albedo, size, tone, RAMP.headlight, { dither: 0.8 });
  return { size, albedo, height, normalStrength: 2.4 };
}

/* ================================================================== *
 * WORLD / SCENERY
 * ================================================================== */

/** Tree bark: vertical ridges split by deep fissures. */
function bark({ size, seed }: PatternArgs): PatternResult {
  // Anisotropic fBm: high frequency across the trunk, low along it.
  const ridges = fbmField(size, { octaves: 5, period: 3, seed, stretch: [5, 1], gain: 0.55, warp: 0.05, ridged: true });
  const plates = worley(size, 7, seed + 11, 0.95, 'euclidean', 0.35);
  const plateEdge = mapField(worleyEdges(plates), (v) => 1 - smoothstep(0.02, 0.2, v));
  const fine = fbmField(size, { octaves: 3, period: size >> 3, seed: seed + 5, stretch: [3, 1], value: true });

  // Deep vertical fissures where the ridged noise troughs line up.
  const fissure = mapField(ridges, (v) => smoothstep(0.42, 0.16, v));

  const height = makeField(size);
  for (let i = 0; i < height.length; i++) {
    height[i] = clamp(0.28 + 0.55 * ridges[i]! + 0.12 * fine[i]! - 0.42 * fissure[i]! - 0.28 * plateEdge[i]!);
  }

  let tone = makeField(size);
  for (let i = 0; i < tone.length; i++) {
    tone[i] =
      0.42 + (ridges[i]! - 0.5) * 0.6 + (fine[i]! - 0.5) * 0.24 + (plates.id[i]! - 0.5) * 0.16 - fissure[i]! * 0.44 - plateEdge[i]! * 0.26;
  }
  addInto(tone, grain(size, seed + 8, 0.09));
  tone = lit(size, tone, height, 0.66, 11, 0.6);
  tone = punch(size, tone, 0.7, 1.34, 0.44);

  const albedo = makeRGBA(size);
  shadeRamp(albedo, size, tone, RAMP.bark, { dither: 1.0 });
  return { size, albedo, height, normalStrength: 3.2 };
}

/** Foliage cluster with alpha cutout — for billboard/cross-plane canopies. */
function foliage({ size, seed }: PatternArgs): PatternResult {
  const cover = makeField(size);
  const depth = makeField(size);
  const leafId = makeField(size, 0.5);
  const rnd = mulberry32(seed + 12);

  // Clump the leaves into a few masses so the cutout reads as a canopy rather
  // than confetti, and so the silhouette has real notches in it.
  const CLUMPS = 7;
  const clumpX: number[] = [];
  const clumpY: number[] = [];
  const clumpR: number[] = [];
  for (let c = 0; c < CLUMPS; c++) {
    clumpX.push(rnd() * size);
    clumpY.push(rnd() * size);
    clumpR.push(size * (0.12 + rnd() * 0.14));
  }

  const leaves = Math.round(size * 5.5);
  for (let i = 0; i < leaves; i++) {
    const c = (rnd() * CLUMPS) | 0;
    const ang = rnd() * Math.PI * 2;
    const rad = Math.pow(rnd(), 0.55) * clumpR[c]!;
    const cx = clumpX[c]! + Math.cos(ang) * rad;
    const cy = clumpY[c]! + Math.sin(ang) * rad;
    const rl = 2.2 + rnd() * 3.4;
    const a = rnd() * Math.PI;
    const d = rnd();
    stampEllipse(cover, size, cx, cy, rl, rl * (0.34 + rnd() * 0.3), a, 1, 'max', 0.8, 0.25);
    stampEllipse(depth, size, cx, cy, rl, rl * 0.4, a, 0.35 + d * 0.65, 'max', 0.5, 0.3);
    stampEllipse(leafId, size, cx, cy, rl * 0.9, rl * 0.34, a, d, 'set', 0.05, 0.95);
  }

  const veins = fbmField(size, { octaves: 3, period: size >> 3, seed: seed + 3, value: true });

  const height = makeField(size);
  for (let i = 0; i < height.length; i++) height[i] = clamp(depth[i]! * 0.8 + veins[i]! * 0.12);

  let tone = makeField(size);
  for (let i = 0; i < tone.length; i++) {
    tone[i] = 0.3 + depth[i]! * 0.42 + (leafId[i]! - 0.5) * 0.42 + (veins[i]! - 0.5) * 0.2;
  }
  tone = lit(size, tone, height, 0.55, 10, 0.55);
  tone = punch(size, tone, 0.6, 1.3, 0.44);

  const albedo = makeRGBA(size);
  shadeRamp(albedo, size, tone, RAMP.leaf, { dither: 1.0, keepAlpha: true });
  // Hard 1-bit cutout — alpha blending in this era was alpha *test*.
  writeAlpha(albedo, size, cover, 0.5);
  return { size, albedo, height, hasAlpha: true, normalStrength: 1.6 };
}

/** Weathered wooden planks. */
function planks({ size, seed }: PatternArgs): PatternResult {
  const PLANKS = 4;
  const rnd = mulberry32(seed + 55);
  const plankTone = new Float32Array(PLANKS);
  const plankShift = new Float32Array(PLANKS);
  for (let i = 0; i < PLANKS; i++) {
    plankTone[i] = rnd();
    plankShift[i] = rnd();
  }

  // Grain: strongly stretched noise, warped so it swirls like real timber.
  const grainF = fbmField(size, { octaves: 5, period: 3, seed, stretch: [1, 9], gain: 0.6, warp: 0.05 });
  const fine = fbmField(size, { octaves: 3, period: size >> 3, seed: seed + 7, stretch: [1, 4], value: true });

  const gapF = makeField(size);
  const plankIdx = makeField(size);
  const plankLocal = makeField(size);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = y * size + x;
      const g = (y / size) * PLANKS;
      const pi = Math.floor(g) % PLANKS;
      const f = g - Math.floor(g);
      plankIdx[i] = pi;
      plankLocal[i] = f;
      gapF[i] = Math.max(smoothstep(0.055, 0.0, f), smoothstep(0.945, 1.0, f));
    }
  }

  // Knots.
  const knots = makeField(size);
  const knotRing = makeField(size);
  scatter(size, 7, seed + 21, (cx, cy, r2) => {
    const rad = size * (0.03 + r2() * 0.05);
    for (let oy = -Math.ceil(rad); oy <= Math.ceil(rad); oy++) {
      for (let ox = -Math.ceil(rad * 1.7); ox <= Math.ceil(rad * 1.7); ox++) {
        const dx = ox / 1.7;
        const dy = oy;
        const d = Math.hypot(dx, dy) / rad;
        if (d >= 1) continue;
        const xx = ((Math.round(cx) + ox) % size + size) % size;
        const yy = ((Math.round(cy) + oy) % size + size) % size;
        const i = yy * size + xx;
        knots[i] = Math.max(knots[i]!, 1 - d);
        knotRing[i] = Math.max(knotRing[i]!, Math.pow(Math.sin(d * Math.PI * 5) * 0.5 + 0.5, 2) * (1 - d));
      }
    }
  });

  // Nails at the plank ends.
  const nails = makeField(size);
  for (let p = 0; p < PLANKS; p++) {
    for (let k = 0; k < 2; k++) {
      const cx = (0.12 + k * 0.76) * size + (plankShift[p]! - 0.5) * 6;
      const cy = (p + 0.5) * (size / PLANKS);
      stampDisc(nails, size, cx, cy, 2.0, 1, 'max', 0.5, 0.55);
    }
  }

  const height = makeField(size);
  for (let i = 0; i < height.length; i++) {
    height[i] = clamp(
      0.55 + (grainF[i]! - 0.5) * 0.24 + (fine[i]! - 0.5) * 0.12 - gapF[i]! * 0.6 - knots[i]! * 0.16 - nails[i]! * 0.35,
    );
  }

  let tone = makeField(size);
  for (let i = 0; i < tone.length; i++) {
    const pi = plankIdx[i]! | 0;
    tone[i] =
      0.46 +
      (plankTone[pi]! - 0.5) * 0.3 +
      (grainF[i]! - 0.5) * 0.62 +
      (fine[i]! - 0.5) * 0.24 -
      gapF[i]! * 0.62 -
      knots[i]! * 0.3 -
      knotRing[i]! * 0.24;
  }
  addInto(tone, grain(size, seed + 3, 0.07));
  tone = lit(size, tone, height, 0.5, 9, 0.5);
  // Nail heads are bright metal, painted over the wood ramp afterwards.
  tone = punch(size, tone, 0.6, 1.28, 0.46);

  const albedo = makeRGBA(size);
  shadeRamp(albedo, size, tone, RAMP.wood, { dither: 1.0 });
  const steel = RAMP.steel;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = y * size + x;
      if (nails[i]! > 0.35) {
        const n = steel.length / 3;
        const t = clamp(0.35 + (1 - nails[i]!) * 0.9 + bayerSigned(x, y, 4) * 0.1);
        const idx = clamp(Math.round(t * (n - 1)), 0, n - 1) | 0;
        albedo[i * 4] = steel[idx * 3]!;
        albedo[i * 4 + 1] = steel[idx * 3 + 1]!;
        albedo[i * 4 + 2] = steel[idx * 3 + 2]!;
      }
    }
  }
  return { size, albedo, height, normalStrength: 2.2 };
}

/** Corrugated metal sheeting, rusted. */
function corrugatedMetal({ size, seed }: PatternArgs): PatternResult {
  const RIBS = 8;
  const corr = fieldFromUV(size, (u) => {
    const s = Math.sin(u * RIBS * Math.PI * 2);
    return s * 0.5 + 0.5;
  });
  // Sharper valley than crest — rolled steel profile, not a pure sine.
  const profile = mapField(corr, (v) => Math.pow(v, 0.75));

  const rustF = fbmField(size, { octaves: 5, period: 3, seed, warp: 0.16, gain: 0.58, turbulence: true });
  const rustMask = mapField(rustF, (v) => smoothstep(0.34, 0.58, v));
  // Rust bleeds downward in streaks.
  const bleed = streakField(rustMask, size, 0, 1, Math.round(size / 6), 0.93);

  const dents = makeField(size);
  scatter(size, Math.round(size * 0.35), seed + 9, (cx, cy, rnd) => {
    stampEllipse(dents, size, cx, cy, 2 + rnd() * 6, 2 + rnd() * 5, rnd() * Math.PI, 0.3 + rnd() * 0.4, 'max', 1.2, 0);
  });

  // Bolt line across the sheet.
  const bolts = makeField(size);
  for (let r = 0; r < RIBS; r++) {
    const cx = ((r + 0.25) / RIBS) * size;
    stampDisc(bolts, size, cx, size * 0.5, 2.2, 1, 'max', 0.5, 0.55);
  }

  const height = makeField(size);
  for (let i = 0; i < height.length; i++) {
    height[i] = clamp(0.15 + 0.72 * profile[i]! - dents[i]! * 0.16 + bolts[i]! * 0.2 - rustMask[i]! * 0.06);
  }

  let tone = makeField(size);
  for (let i = 0; i < tone.length; i++) {
    tone[i] = 0.34 + profile[i]! * 0.44 - dents[i]! * 0.18 + bolts[i]! * 0.2;
  }
  addInto(tone, grain(size, seed + 2, 0.06));
  tone = lit(size, tone, height, 0.62, 10, 0.4);
  tone = punch(size, tone, 0.45, 1.28, 0.48);

  const albedo = makeRGBA(size);
  shadeRamp(albedo, size, tone, RAMP.steel, { dither: 1.0 });
  const rustRamp = buildRamp(['#2a1408', '#4d2410', '#7a3d1c', '#a15a2c', '#c08050'], 14);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = y * size + x;
      const r = Math.max(rustMask[i]!, bleed[i]! * 0.85);
      if (r + bayerSigned(x, y, 8) * 0.4 > 0.5) {
        const n = rustRamp.length / 3;
        const idx = clamp(Math.round(clamp(tone[i]! * 0.95) * (n - 1) + bayerSigned(x, y, 4)), 0, n - 1) | 0;
        albedo[i * 4] = rustRamp[idx * 3]!;
        albedo[i * 4 + 1] = rustRamp[idx * 3 + 1]!;
        albedo[i * 4 + 2] = rustRamp[idx * 3 + 2]!;
      }
    }
  }
  return { size, albedo, height, normalStrength: 2.6 };
}

/** Dry scrub / brush — sparse twiggy cover with alpha. */
function scrub({ size, seed }: PatternArgs): PatternResult {
  const cover = makeField(size);
  const bright = makeField(size);
  const rnd = mulberry32(seed + 44);

  const BUSHES = 5;
  for (let b = 0; b < BUSHES; b++) {
    const bx = rnd() * size;
    const by = rnd() * size;
    const br = size * (0.10 + rnd() * 0.13);
    const twigs = 60 + (rnd() * 60) | 0;
    for (let t = 0; t < twigs; t++) {
      const a0 = rnd() * Math.PI * 2;
      const r0 = Math.pow(rnd(), 0.7) * br * 0.4;
      let x = bx + Math.cos(a0) * r0;
      let y = by + Math.sin(a0) * r0;
      let a = a0 + (rnd() - 0.5) * 1.2;
      const segs = 3 + ((rnd() * 5) | 0);
      for (let s = 0; s < segs; s++) {
        const l = 1.6 + rnd() * 3.2;
        const nx = x + Math.cos(a) * l;
        const ny = y + Math.sin(a) * l;
        stampLine(cover, size, x, y, nx, ny, 0.8, 1, 'max', 0.25);
        stampLine(bright, size, x, y, nx, ny, 0.6, rnd() * 0.9, 'max', 0.4);
        x = nx;
        y = ny;
        a += (rnd() - 0.5) * 1.1;
      }
      // Dry leaf clusters on the tips.
      if (rnd() < 0.45) stampEllipse(cover, size, x, y, 1.4 + rnd() * 1.6, 1 + rnd(), rnd() * Math.PI, 1, 'max', 0.8, 0.3);
    }
  }

  const height = makeField(size);
  for (let i = 0; i < height.length; i++) height[i] = clamp(cover[i]! * 0.7 + bright[i]! * 0.3);

  let tone = makeField(size);
  for (let i = 0; i < tone.length; i++) tone[i] = 0.3 + bright[i]! * 0.55 + cover[i]! * 0.18;
  addInto(tone, grain(size, seed + 6, 0.12));
  tone = lit(size, tone, height, 0.5, 10, 0.5);
  tone = punch(size, tone, 0.7, 1.35, 0.44);

  const albedo = makeRGBA(size);
  shadeRamp(albedo, size, tone, RAMP.scrub, { dither: 1.0, keepAlpha: true });
  writeAlpha(albedo, size, cover, 0.42);
  return { size, albedo, height, hasAlpha: true, normalStrength: 1.4 };
}

/* ================================================================== *
 * EFFECTS / UI
 * ================================================================== */

/** Soft-but-chunky dust puff sprite (alpha). */
function dust({ size, seed }: PatternArgs): PatternResult {
  const lumps = fbmField(size, { octaves: 4, period: 4, seed, turbulence: true, gain: 0.55, warp: 0.12 });
  const alpha = makeField(size);
  const tone = makeField(size);
  const half = size / 2;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = y * size + x;
      const d = Math.hypot(x + 0.5 - half, y + 0.5 - half) / half;
      // Noisy radius so the silhouette is a puff, not a circle.
      const rr = d * (0.75 + lumps[i]! * 0.55);
      alpha[i] = clamp(Math.pow(1 - clamp(rr), 1.15));
      tone[i] = clamp(0.42 + (1 - d) * 0.4 + (lumps[i]! - 0.5) * 0.5);
    }
  }
  // Bake a light side so the puff has volume.
  const h = mapField(alpha, (v, i) => v * (0.6 + lumps[i]! * 0.5));
  const shaded = lit(size, tone, h, 0.55, 6, 0.35);
  const final = punch(size, shaded, 0.4, 1.2, 0.5);

  const albedo = makeRGBA(size);
  shadeRamp(albedo, size, final, RAMP.dust, { dither: 1.0, keepAlpha: true });
  // Quantised alpha ladder — era sprites had ~8 alpha steps, not 256.
  writeAlpha(albedo, size, posterizeField(mapField(alpha, (v) => Math.pow(v, 1.25)), 8));
  return { size, albedo, height: h, hasAlpha: true, clamp: true, normalStrength: 0.8 };
}

/** Billowing smoke puff sprite (alpha). */
function smoke({ size, seed }: PatternArgs): PatternResult {
  const billow = fbmField(size, { octaves: 5, period: 3, seed, turbulence: true, gain: 0.6, warp: 0.18 });
  const alpha = makeField(size);
  const tone = makeField(size);
  const half = size / 2;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = y * size + x;
      const d = Math.hypot(x + 0.5 - half, y + 0.5 - half) / half;
      const rr = d * (0.62 + billow[i]! * 0.8);
      alpha[i] = clamp(Math.pow(1 - clamp(rr), 1.4)) * 0.92;
      tone[i] = clamp(0.35 + (billow[i]! - 0.35) * 0.9 + (1 - d) * 0.25);
    }
  }
  const h = mapField(alpha, (v, i) => v * (0.4 + billow[i]! * 0.8));
  const shaded = lit(size, tone, h, 0.7, 7, 0.45);
  const final = punch(size, shaded, 0.45, 1.22, 0.48);

  const albedo = makeRGBA(size);
  shadeRamp(albedo, size, final, RAMP.smoke, { dither: 1.0, keepAlpha: true });
  writeAlpha(albedo, size, posterizeField(alpha, 8));
  return { size, albedo, height: h, hasAlpha: true, clamp: true, normalStrength: 0.6 };
}

/** Tyre skid decal — tiles along V (the direction of travel), fades across U. */
function skid({ size, seed }: PatternArgs): PatternResult {
  const ROWS = 8;
  const wobble = fbmField(size, { octaves: 3, period: 3, seed });
  const grit = fbmField(size, { octaves: 3, period: size >> 3, seed: seed + 3, value: true });

  const cover = makeField(size);
  const tone = makeField(size);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = y * size + x;
      const u = x / size;
      const v = y / size;
      const wob = (wobble[i]! - 0.5) * 0.12;
      // Width profile across the tyre, with ragged edges.
      const across = 1 - Math.abs(u - 0.5 + wob) * 2.1;
      const body = smoothstep(0.0, 0.35, across);
      // Tread ribs printed into the mark.
      const p = (v + Math.abs(u - 0.5) * 0.3) * ROWS;
      const rib = Math.abs(p - Math.floor(p) - 0.5) * 2;
      const ribMask = 0.55 + 0.45 * smoothstep(0.2, 0.8, rib);
      cover[i] = clamp(body * ribMask * (0.65 + grit[i]! * 0.6));
      tone[i] = clamp(0.16 + (1 - body) * 0.32 + (grit[i]! - 0.5) * 0.3);
    }
  }

  const albedo = makeRGBA(size);
  shadeRamp(albedo, size, punch(size, tone, 0.4, 1.2, 0.4), RAMP.rubber, { dither: 1.0, keepAlpha: true });
  writeAlpha(albedo, size, posterizeField(cover, 6));
  const h = mapField(cover, (v) => v * 0.3);
  return { size, albedo, height: h, hasAlpha: true, clamp: true, flat: true, normalStrength: 0.4 };
}

/* -------- bitmap font atlas -------- */

/**
 * Classic 5x7 bitmap font, ASCII 32..95 (space through underscore — everything
 * a HUD needs). Five column bytes per glyph, bit 0 = top row.
 */
const FONT5X7: readonly number[][] = [
  [0x00, 0x00, 0x00, 0x00, 0x00], // ' '
  [0x00, 0x00, 0x5f, 0x00, 0x00], // !
  [0x00, 0x07, 0x00, 0x07, 0x00], // "
  [0x14, 0x7f, 0x14, 0x7f, 0x14], // #
  [0x24, 0x2a, 0x7f, 0x2a, 0x12], // $
  [0x23, 0x13, 0x08, 0x64, 0x62], // %
  [0x36, 0x49, 0x55, 0x22, 0x50], // &
  [0x00, 0x05, 0x03, 0x00, 0x00], // '
  [0x00, 0x1c, 0x22, 0x41, 0x00], // (
  [0x00, 0x41, 0x22, 0x1c, 0x00], // )
  [0x14, 0x08, 0x3e, 0x08, 0x14], // *
  [0x08, 0x08, 0x3e, 0x08, 0x08], // +
  [0x00, 0x50, 0x30, 0x00, 0x00], // ,
  [0x08, 0x08, 0x08, 0x08, 0x08], // -
  [0x00, 0x60, 0x60, 0x00, 0x00], // .
  [0x20, 0x10, 0x08, 0x04, 0x02], // /
  [0x3e, 0x51, 0x49, 0x45, 0x3e], // 0
  [0x00, 0x42, 0x7f, 0x40, 0x00], // 1
  [0x42, 0x61, 0x51, 0x49, 0x46], // 2
  [0x21, 0x41, 0x45, 0x4b, 0x31], // 3
  [0x18, 0x14, 0x12, 0x7f, 0x10], // 4
  [0x27, 0x45, 0x45, 0x45, 0x39], // 5
  [0x3c, 0x4a, 0x49, 0x49, 0x30], // 6
  [0x01, 0x71, 0x09, 0x05, 0x03], // 7
  [0x36, 0x49, 0x49, 0x49, 0x36], // 8
  [0x06, 0x49, 0x49, 0x29, 0x1e], // 9
  [0x00, 0x36, 0x36, 0x00, 0x00], // :
  [0x00, 0x56, 0x36, 0x00, 0x00], // ;
  [0x00, 0x08, 0x14, 0x22, 0x41], // <
  [0x14, 0x14, 0x14, 0x14, 0x14], // =
  [0x41, 0x22, 0x14, 0x08, 0x00], // >
  [0x02, 0x01, 0x51, 0x09, 0x06], // ?
  [0x32, 0x49, 0x79, 0x41, 0x3e], // @
  [0x7e, 0x11, 0x11, 0x11, 0x7e], // A
  [0x7f, 0x49, 0x49, 0x49, 0x36], // B
  [0x3e, 0x41, 0x41, 0x41, 0x22], // C
  [0x7f, 0x41, 0x41, 0x22, 0x1c], // D
  [0x7f, 0x49, 0x49, 0x49, 0x41], // E
  [0x7f, 0x09, 0x09, 0x01, 0x01], // F
  [0x3e, 0x41, 0x41, 0x51, 0x32], // G
  [0x7f, 0x08, 0x08, 0x08, 0x7f], // H
  [0x00, 0x41, 0x7f, 0x41, 0x00], // I
  [0x20, 0x40, 0x41, 0x3f, 0x01], // J
  [0x7f, 0x08, 0x14, 0x22, 0x41], // K
  [0x7f, 0x40, 0x40, 0x40, 0x40], // L
  [0x7f, 0x02, 0x04, 0x02, 0x7f], // M
  [0x7f, 0x04, 0x08, 0x10, 0x7f], // N
  [0x3e, 0x41, 0x41, 0x41, 0x3e], // O
  [0x7f, 0x09, 0x09, 0x09, 0x06], // P
  [0x3e, 0x41, 0x51, 0x21, 0x5e], // Q
  [0x7f, 0x09, 0x19, 0x29, 0x46], // R
  [0x46, 0x49, 0x49, 0x49, 0x31], // S
  [0x01, 0x01, 0x7f, 0x01, 0x01], // T
  [0x3f, 0x40, 0x40, 0x40, 0x3f], // U
  [0x1f, 0x20, 0x40, 0x20, 0x1f], // V
  [0x7f, 0x20, 0x18, 0x20, 0x7f], // W
  [0x63, 0x14, 0x08, 0x14, 0x63], // X
  [0x03, 0x04, 0x78, 0x04, 0x03], // Y
  [0x61, 0x51, 0x49, 0x45, 0x43], // Z
  [0x00, 0x00, 0x7f, 0x41, 0x41], // [
  [0x02, 0x04, 0x08, 0x10, 0x20], // backslash
  [0x41, 0x41, 0x7f, 0x00, 0x00], // ]
  [0x04, 0x02, 0x01, 0x02, 0x04], // ^
  [0x40, 0x40, 0x40, 0x40, 0x40], // _
];

/** Font atlas metrics, exported so HUD code can lay text out without guessing. */
export const FONT_ATLAS = {
  /** Glyphs per row in the atlas. */
  cols: 8,
  rows: 8,
  /** First ASCII code point present. */
  firstChar: 32,
  charCount: 64,
  /** Glyph bitmap size in atlas pixels (before the cell padding). */
  glyphWidth: 10,
  glyphHeight: 14,
} as const;

/**
 * 8x8 grid of 16x16 cells = 128x128. Glyphs are drawn at 2x with a 1px dark
 * outline so HUD text stays readable over any background.
 */
function fontAtlas({ size }: PatternArgs): PatternResult {
  const S = 128;
  const albedo = makeRGBA(S, 0, 0, 0, 0);
  const cell = 16;
  const scale = 2;

  const mask = makeField(S);
  for (let g = 0; g < FONT5X7.length; g++) {
    const gx = (g % 8) * cell;
    const gy = ((g / 8) | 0) * cell;
    const glyph = FONT5X7[g]!;
    for (let c = 0; c < 5; c++) {
      const bits = glyph[c]!;
      for (let r = 0; r < 7; r++) {
        if (!(bits & (1 << r))) continue;
        for (let sy = 0; sy < scale; sy++) {
          for (let sx = 0; sx < scale; sx++) {
            const px = gx + 2 + c * scale + sx;
            const py = gy + 1 + r * scale + sy;
            if (px < S && py < S) mask[py * S + px] = 1;
          }
        }
      }
    }
  }

  // 1px outline: any transparent texel adjacent to a lit one becomes shadow.
  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) {
      const i = y * S + x;
      if (mask[i]! >= 1) {
        albedo[i * 4] = 255;
        albedo[i * 4 + 1] = 244;
        albedo[i * 4 + 2] = 214;
        albedo[i * 4 + 3] = 255;
      }
    }
  }
  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) {
      const i = y * S + x;
      if (mask[i]! >= 1) continue;
      let near = false;
      for (let oy = -1; oy <= 1 && !near; oy++) {
        for (let ox = -1; ox <= 1; ox++) {
          const xx = x + ox;
          const yy = y + oy;
          if (xx < 0 || yy < 0 || xx >= S || yy >= S) continue;
          if (mask[yy * S + xx]! >= 1) {
            near = true;
            break;
          }
        }
      }
      if (near) {
        albedo[i * 4] = 22;
        albedo[i * 4 + 1] = 16;
        albedo[i * 4 + 2] = 10;
        albedo[i * 4 + 3] = 255;
      }
    }
  }
  return { size: S, albedo, height: makeField(S), hasAlpha: true, clamp: true, flat: true };
}

/* ------------------------------------------------------------------ *
 * Registry
 * ------------------------------------------------------------------ */

export const PATTERNS: Record<TextureName, PatternFn> = {
  dirt,
  grass,
  rock,
  gravel,
  sand,
  mud,
  snow,
  paintedMetal,
  tyre,
  glass,
  chrome,
  vinyl,
  headlight,
  bark,
  foliage,
  rockFace,
  planks,
  corrugatedMetal,
  scrub,
  cliffStrata,
  dust,
  smoke,
  skid,
  fontAtlas,
};

export const TEXTURE_NAMES = Object.keys(PATTERNS) as TextureName[];

/** Grouping used by the preview contact sheet. */
export const TEXTURE_GROUPS: { label: string; names: TextureName[] }[] = [
  { label: 'Terrain', names: ['dirt', 'grass', 'rock', 'gravel', 'sand', 'mud', 'snow'] },
  { label: 'Vehicle', names: ['paintedMetal', 'tyre', 'glass', 'chrome', 'vinyl', 'headlight'] },
  { label: 'World', names: ['bark', 'foliage', 'rockFace', 'planks', 'corrugatedMetal', 'scrub', 'cliffStrata'] },
  { label: 'Effects / UI', names: ['dust', 'smoke', 'skid', 'fontAtlas'] },
];

/** Default resolution per texture. 256 for surfaces you get close to, 128 else. */
export const DEFAULT_SIZE: Record<TextureName, number> = {
  dirt: 256,
  grass: 256,
  rock: 256,
  gravel: 256,
  sand: 256,
  mud: 256,
  snow: 256,
  paintedMetal: 256,
  tyre: 256,
  glass: 128,
  chrome: 128,
  vinyl: 128,
  headlight: 128,
  bark: 256,
  foliage: 256,
  rockFace: 256,
  planks: 256,
  corrugatedMetal: 256,
  scrub: 256,
  cliffStrata: 256,
  dust: 64,
  smoke: 64,
  skid: 128,
  fontAtlas: 128,
};

export const DEFAULT_SEED: Record<TextureName, number> = {
  dirt: 1337,
  grass: 2024,
  rock: 88,
  gravel: 4711,
  sand: 909,
  mud: 6161,
  snow: 3003,
  paintedMetal: 7,
  tyre: 512,
  glass: 71,
  chrome: 313,
  vinyl: 202,
  headlight: 44,
  bark: 1201,
  foliage: 8080,
  rockFace: 55,
  planks: 626,
  corrugatedMetal: 999,
  scrub: 1717,
  cliffStrata: 2929,
  dust: 31,
  smoke: 32,
  skid: 33,
  fontAtlas: 0,
};

/** Convenience for the preview page: raw greyscale RGBA of the height field. */
export function heightPreview(r: PatternResult): RGBA {
  return fieldToGrey(r.size, r.height);
}

// Re-exported so consumers can build matching UI colours without importing the
// noise module directly.
export { BAYER4, bayer, blurField, cloneField, mixField, normalizeField, thresholdField };
