import * as THREE from 'three';

/**
 * The single source of truth for "what colour is the sky in direction D".
 *
 * Both the sky dome and the post-process fog compile this exact function, and
 * they share the *same uniform objects*. That is deliberate: the classic ugly
 * bug in this kind of renderer is a visible seam at the horizon where a
 * constant fog colour meets a gradient sky. If the fog resolves to
 * `atmosphereBase(viewDir)` then, by construction, fully-fogged geometry is
 * pixel-identical to the sky behind it. No seam is possible.
 */

export interface AtmosphereUniforms {
  uSunDirection: THREE.IUniform<THREE.Vector3>;
  uZenithColor: THREE.IUniform<THREE.Color>;
  uHorizonColor: THREE.IUniform<THREE.Color>;
  uGroundColor: THREE.IUniform<THREE.Color>;
  uSunGlowColor: THREE.IUniform<THREE.Color>;
  uSunDiscColor: THREE.IUniform<THREE.Color>;
  /** x: gradient power, y: glow power, z: glow strength, w: haze lift */
  uSkyParams: THREE.IUniform<THREE.Vector4>;
  /** x: disc angular radius, y: disc intensity, z: halo angular size, w: halo strength */
  uSunDiscParams: THREE.IUniform<THREE.Vector4>;
  [key: string]: THREE.IUniform;
}

export function createAtmosphereUniforms(): AtmosphereUniforms {
  return {
    uSunDirection: { value: new THREE.Vector3(0.35, 0.32, -0.88).normalize() },
    uZenithColor: { value: new THREE.Color(0.055, 0.16, 0.42) },
    uHorizonColor: { value: new THREE.Color(0.72, 0.66, 0.53) },
    uGroundColor: { value: new THREE.Color(0.42, 0.36, 0.29) },
    uSunGlowColor: { value: new THREE.Color(1.0, 0.62, 0.28) },
    uSunDiscColor: { value: new THREE.Color(1.0, 0.82, 0.55) },
    uSkyParams: { value: new THREE.Vector4(2.6, 26.0, 0.55, 0.0) },
    uSunDiscParams: { value: new THREE.Vector4(0.028, 22.0, 0.42, 0.55) },
  };
}

/** Uniform declarations — include once per shader that uses the functions. */
export const ATMOSPHERE_UNIFORMS_GLSL = /* glsl */ `
uniform vec3 uSunDirection;
uniform vec3 uZenithColor;
uniform vec3 uHorizonColor;
uniform vec3 uGroundColor;
uniform vec3 uSunGlowColor;
uniform vec3 uSunDiscColor;
uniform vec4 uSkyParams;
uniform vec4 uSunDiscParams;
`;

/**
 * Sky radiance without the sun disc. Linear, HDR-ish (horizon can exceed 1
 * looking into the sun, which is what feeds the bloom).
 */
export const ATMOSPHERE_FUNCTIONS_GLSL = /* glsl */ `
vec3 atmosphereBase(vec3 dir) {
  float y = dir.y;

  // Zenith -> horizon gradient. The power controls how tightly the pale band
  // hugs the horizon; ~2.5 reads as a hazy afternoon, ~5 as clean high air.
  float t = pow(clamp(1.0 - max(y, 0.0), 0.0, 1.0), uSkyParams.x);
  vec3 col = mix(uZenithColor, uHorizonColor, t);

  float sd = dot(dir, uSunDirection);

  // Tight forward-scatter lobe around the sun.
  float tight = pow(max(sd, 0.0), uSkyParams.y) * uSkyParams.z;
  // Broad Mie-ish haze, strongest low in the sky on the sun's side.
  float broad = pow(max(sd, 0.0) * 0.5 + 0.5, 3.0) * t * 0.5;

  col += uSunGlowColor * (tight + broad);
  col += uHorizonColor * uSkyParams.w * t;

  // Below the horizon the air turns to dust haze rather than sky.
  col = mix(col, uGroundColor, smoothstep(0.0, -0.09, y));
  return col;
}

vec3 atmosphereSunDisc(vec3 dir) {
  float cosA = clamp(dot(dir, uSunDirection), -1.0, 1.0);
  float ang = acos(cosA);
  float core = 1.0 - smoothstep(uSunDiscParams.x * 0.70, uSunDiscParams.x, ang);
  float halo = pow(max(0.0, 1.0 - ang / uSunDiscParams.z), 3.5);
  // Sink the disc as the sun drops below the horizon.
  float above = smoothstep(-0.10, 0.02, uSunDirection.y);
  return uSunDiscColor * (core * uSunDiscParams.y + halo * uSunDiscParams.w) * above;
}
`;

/** Cheap hash/value-noise/fbm kit shared by clouds and ridges. */
export const NOISE_GLSL = /* glsl */ `
float hash21(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}

float vnoise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  float a = hash21(i);
  float b = hash21(i + vec2(1.0, 0.0));
  float c = hash21(i + vec2(0.0, 1.0));
  float d = hash21(i + vec2(1.0, 1.0));
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float fbm4(vec2 p) {
  float v = 0.0;
  float a = 0.5;
  for (int i = 0; i < 4; i++) {
    v += a * vnoise(p);
    p = p * 2.03 + vec2(1.7, 9.2);
    a *= 0.5;
  }
  return v;
}

float fbm6(vec2 p) {
  float v = 0.0;
  float a = 0.5;
  for (int i = 0; i < 6; i++) {
    v += a * vnoise(p);
    p = p * 2.03 + vec2(1.7, 9.2);
    a *= 0.5;
  }
  return v;
}
`;
