import * as THREE from 'three';

/**
 * Procedural colour-grading LUT.
 *
 * Stored as a strip (N slices of NxN laid out horizontally) rather than a
 * Data3DTexture, because sampler3D needs GLSL ES 3.00 and every other pass in
 * this chain is happily GLSL1. Hardware bilinear handles R/G; we lerp B in the
 * shader. Visually indistinguishable from a real 3D LUT at N=32.
 */

export const LUT_SIZE = 32;

export interface GradeParams {
  /** Overall exposure applied before the curve. */
  exposure: number;
  /** S-curve strength around the pivot. Era games were contrasty. */
  contrast: number;
  /** Pivot for the contrast curve, in gamma space. */
  pivot: number;
  /** Global saturation multiplier. 1.15-1.35 is the Screamer-era range. */
  saturation: number;
  /** Colour pushed into the shadows (a cool teal/blue lift reads as "air"). */
  shadowTint: THREE.ColorRepresentation;
  /** Colour pushed into the highlights (warm sun). */
  highlightTint: THREE.ColorRepresentation;
  /** How hard the tints bite, 0..1. */
  splitStrength: number;
  /** Black point lift — a small positive value stops the shadows going dead. */
  lift: number;
  /** Where the whites clip. <1 crushes highlights, which era hardware did. */
  whitePoint: number;
  /** Per-channel gain, applied last. */
  gain: THREE.ColorRepresentation;
  /** Pulls everything toward a single hue — a cheap "film stock" feel. */
  toneWash: THREE.ColorRepresentation;
  toneWashStrength: number;
}

export const DEFAULT_GRADE: GradeParams = {
  exposure: 1.0,
  contrast: 0.22,
  pivot: 0.47,
  saturation: 1.2,
  shadowTint: 0x2a3f63,
  highlightTint: 0xffd9a0,
  splitStrength: 0.16,
  lift: 0.012,
  whitePoint: 0.985,
  gain: 0xffffff,
  toneWash: 0xffe6c2,
  toneWashStrength: 0.05,
};

export function cloneGrade(g: GradeParams): GradeParams {
  return { ...g };
}

const _shadow = new THREE.Color();
const _high = new THREE.Color();
const _gain = new THREE.Color();
const _wash = new THREE.Color();

function luminance(r: number, g: number, b: number): number {
  return r * 0.2126 + g * 0.7152 + b * 0.0722;
}

/**
 * Builds the strip texture. Input and output are both gamma-space (0..1) —
 * grading in gamma space is what colourists actually do, and it is also what
 * the era's palette hardware effectively operated on.
 */
export function buildGradeLUT(p: GradeParams, existing?: THREE.DataTexture): THREE.DataTexture {
  const N = LUT_SIZE;
  const w = N * N;
  const h = N;
  const data =
    existing && existing.image.data instanceof Uint8Array && existing.image.data.length === w * h * 4
      ? (existing.image.data as Uint8Array)
      : new Uint8Array(w * h * 4);

  _shadow.set(p.shadowTint);
  _high.set(p.highlightTint);
  _gain.set(p.gain);
  _wash.set(p.toneWash);

  const inv = 1 / (N - 1);

  for (let bi = 0; bi < N; bi++) {
    const b0 = bi * inv;
    for (let gi = 0; gi < N; gi++) {
      const g0 = gi * inv;
      for (let ri = 0; ri < N; ri++) {
        const r0 = ri * inv;

        let r = r0 * p.exposure;
        let g = g0 * p.exposure;
        let b = b0 * p.exposure;

        // --- S-curve contrast around the pivot -------------------------
        // Steepens the midtones while leaving both endpoints anchored, so
        // nothing clips that was not already clipping.
        const cs = (v: number): number => {
          const t = v - p.pivot;
          return p.pivot + t * (1 + p.contrast * Math.max(0, 1 - Math.abs(t) * 2));
        };
        r = cs(r);
        g = cs(g);
        b = cs(b);

        // --- saturation ------------------------------------------------
        const lum = luminance(r, g, b);
        r = lum + (r - lum) * p.saturation;
        g = lum + (g - lum) * p.saturation;
        b = lum + (b - lum) * p.saturation;

        // --- split tone -------------------------------------------------
        const l2 = Math.min(1, Math.max(0, luminance(r, g, b)));
        // Shadow weight peaks at black, highlight weight at white; the
        // midtones stay neutral so skin/dirt does not go weird.
        const ws = (1 - l2) * (1 - l2) * p.splitStrength;
        const wh = l2 * l2 * p.splitStrength;
        r += (_shadow.r - 0.5) * ws * 2 + (_high.r - 0.5) * wh * 2;
        g += (_shadow.g - 0.5) * ws * 2 + (_high.g - 0.5) * wh * 2;
        b += (_shadow.b - 0.5) * ws * 2 + (_high.b - 0.5) * wh * 2;

        // --- tone wash ---------------------------------------------------
        r = r * (1 - p.toneWashStrength) + r * _wash.r * p.toneWashStrength * 1.35;
        g = g * (1 - p.toneWashStrength) + g * _wash.g * p.toneWashStrength * 1.35;
        b = b * (1 - p.toneWashStrength) + b * _wash.b * p.toneWashStrength * 1.35;

        // --- lift / white point / gain ------------------------------------
        const range = p.whitePoint - p.lift;
        r = p.lift + r * range * _gain.r;
        g = p.lift + g * range * _gain.g;
        b = p.lift + b * range * _gain.b;

        const o = (gi * w + bi * N + ri) * 4;
        data[o] = Math.round(Math.min(1, Math.max(0, r)) * 255);
        data[o + 1] = Math.round(Math.min(1, Math.max(0, g)) * 255);
        data[o + 2] = Math.round(Math.min(1, Math.max(0, b)) * 255);
        data[o + 3] = 255;
      }
    }
  }

  if (existing) {
    existing.needsUpdate = true;
    return existing;
  }

  const tex = new THREE.DataTexture(data, w, h, THREE.RGBAFormat, THREE.UnsignedByteType);
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.generateMipmaps = false;
  tex.colorSpace = THREE.NoColorSpace;
  tex.needsUpdate = true;
  return tex;
}

/** Shader-side sampler for the strip layout. Matches buildGradeLUT exactly. */
export const LUT_SAMPLE_GLSL = /* glsl */ `
#ifndef LUT_N
#define LUT_N 32.0
#endif
vec3 sampleGradeLUT(sampler2D lut, vec3 c) {
  c = clamp(c, 0.0, 1.0);
  float n = LUT_N;
  float b = c.b * (n - 1.0);
  float b0 = floor(b);
  float b1 = min(b0 + 1.0, n - 1.0);
  float f = b - b0;

  float xs = (n - 1.0) / (n * n);
  float halfX = 0.5 / (n * n);
  float y = (c.g * (n - 1.0) + 0.5) / n;

  float x = halfX + c.r * xs;
  vec3 s0 = texture2D(lut, vec2(x + b0 / n, y)).rgb;
  vec3 s1 = texture2D(lut, vec2(x + b1 / n, y)).rgb;
  return mix(s0, s1, f);
}
`;
