import * as THREE from 'three';
import { FullScreenPass } from './fullscreen';
import { LUT_SAMPLE_GLSL, LUT_SIZE } from './lut';

/**
 * The heart of the retro look. Runs at the LOW internal resolution and, in one
 * pass, does: HDR combine -> soft-clip tonemap -> gamma encode -> LUT grade ->
 * quantise + ordered dither.
 *
 * Order matters enormously here:
 *  - the LUT grades in gamma space, like a real colourist would;
 *  - the dither is the LAST thing that happens, in gamma space, at the LOW
 *    resolution. Dither before grading and the grade smears the pattern back
 *    into mush; dither after the upscale and you get a screen-door artefact
 *    instead of an era palette.
 *
 * The tonemap is a per-channel soft clip rather than ACES on purpose. ACES
 * desaturates highlights, which is exactly the opposite of the punchy,
 * over-saturated 16-bit palette we are chasing.
 */

const FRAG = /* glsl */ `
precision highp float;
varying vec2 vUv;

uniform sampler2D tScene;
uniform sampler2D tBloom;
uniform sampler2D tGodrays;
uniform sampler2D tLut;
uniform sampler2D tDither;

uniform float uExposure;
uniform float uShoulder;
uniform float uBloomStrength;
uniform float uGodrayStrength;
uniform vec3 uGodrayTint;
uniform vec3 uQuantLevels;
uniform float uDitherAmount;
uniform vec2 uDitherTexelScale;

${LUT_SAMPLE_GLSL}

vec3 softClip(vec3 x) {
  // Linear below the shoulder, exponential approach to 1.0 above it. Keeps
  // channel ratios (and therefore saturation) intact where it matters.
  vec3 k = vec3(uShoulder);
  vec3 over = max(x - k, 0.0);
  vec3 head = 1.0 - k;
  return min(x, k) + head * (1.0 - exp(-over / max(head, vec3(1e-3))));
}

void main() {
  vec3 c = texture2D(tScene, vUv).rgb;

#ifdef USE_BLOOM
  c += texture2D(tBloom, vUv).rgb * uBloomStrength;
#endif

#ifdef USE_GODRAYS
  c += texture2D(tGodrays, vUv).rgb * uGodrayTint * uGodrayStrength;
#endif

  c = softClip(max(c * uExposure, 0.0));

  // Linear -> gamma. Everything from here on is display-referred.
  vec3 g = pow(c, vec3(1.0 / 2.2));

#ifdef USE_LUT
  g = sampleGradeLUT(tLut, g);
#endif

#ifdef USE_QUANTISE
  #ifdef USE_DITHER
    float d = texture2D(tDither, gl_FragCoord.xy * uDitherTexelScale).r;
  #else
    float d = 0.5;
  #endif
  // (d - 0.5) * amount + 0.5 lets the amount slide continuously between a hard
  // round (amount = 0) and a full-strength ordered dither (amount = 1).
  vec3 off = vec3((d - 0.5) * uDitherAmount + 0.5);
  g = floor(g * uQuantLevels + off) / uQuantLevels;
#endif

  gl_FragColor = vec4(clamp(g, 0.0, 1.0), 1.0);
}
`;

export function createCompositePass(): FullScreenPass {
  return new FullScreenPass({
    fragmentShader: FRAG,
    uniforms: {
      tScene: { value: null },
      tBloom: { value: null },
      tGodrays: { value: null },
      tLut: { value: null },
      tDither: { value: null },
      uExposure: { value: 1.0 },
      uShoulder: { value: 0.7 },
      uBloomStrength: { value: 0.55 },
      uGodrayStrength: { value: 0.4 },
      uGodrayTint: { value: new THREE.Vector3(1, 0.86, 0.62) },
      uQuantLevels: { value: new THREE.Vector3(31, 63, 31) },
      uDitherAmount: { value: 0.85 },
      uDitherTexelScale: { value: new THREE.Vector2(1 / 8, 1 / 8) },
    },
    defines: {
      LUT_N: `${LUT_SIZE}.0`,
      USE_BLOOM: 1,
      USE_GODRAYS: 1,
      USE_LUT: 1,
      USE_QUANTISE: 1,
      USE_DITHER: 1,
    },
  });
}
