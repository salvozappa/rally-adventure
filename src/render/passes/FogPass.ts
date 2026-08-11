import * as THREE from 'three';
import { FullScreenPass } from './fullscreen';
import {
  ATMOSPHERE_FUNCTIONS_GLSL,
  ATMOSPHERE_UNIFORMS_GLSL,
  type AtmosphereUniforms,
} from './atmosphere';

/**
 * Depth-driven atmospheric pass. Does two jobs that both need the depth
 * buffer, so they share one set of taps:
 *
 *  1. Height + distance fog that resolves to `atmosphereBase(viewDir)` — the
 *     literal sky function — so distant geometry dissolves into the sky with
 *     no horizon seam.
 *  2. Screen-space contact darkening: a cheap depth-discontinuity outline that
 *     gives the era's flat Lambert shading the separation it otherwise lacks.
 *
 * Edge darkening runs *before* fog on purpose: outlines on a hill 800m away
 * should be washed out by haze, exactly like every other detail.
 */

const FRAG = /* glsl */ `
precision highp float;
varying vec2 vUv;

uniform sampler2D tScene;
uniform sampler2D tDepth;
uniform mat4 uInvProjection;
uniform mat4 uCameraWorld;
uniform vec3 uCameraPos;
uniform vec2 uNearFar;
uniform vec2 uTexel;

/** x: density, y: height falloff, z: ground level, w: start distance */
uniform vec4 uFogParams;
uniform float uFogAmount;
uniform float uEdgeStrength;
uniform float uEdgeWidth;

${ATMOSPHERE_UNIFORMS_GLSL}
${ATMOSPHERE_FUNCTIONS_GLSL}

float viewZ(float depth) {
  // Perspective depth (0..1 window) -> positive distance along -Z in view space.
  float n = uNearFar.x;
  float f = uNearFar.y;
  float z = depth * 2.0 - 1.0;
  return (2.0 * n * f) / (f + n - z * (f - n));
}

void main() {
  vec3 color = texture2D(tScene, vUv).rgb;
  float depth = texture2D(tDepth, vUv).r;

  // Reconstruct the view ray for this pixel: unproject the far-plane point,
  // then normalise so its z is exactly -1. Multiplying that by the linear view
  // depth gives the view-space position of whatever we hit.
  vec4 clip = vec4(vUv * 2.0 - 1.0, 1.0, 1.0);
  vec4 vpos = uInvProjection * clip;
  vpos /= vpos.w;
  vec3 viewRay = vpos.xyz / max(1e-6, -vpos.z);
  vec3 worldRay = (uCameraWorld * vec4(viewRay, 0.0)).xyz;
  vec3 dirWorld = normalize(worldRay);

  bool isSky = depth >= 0.999995;

#ifdef USE_EDGE_DARKEN
  if (!isSky) {
    float z0 = viewZ(depth);
    vec2 o = uTexel * uEdgeWidth;
    float zl = viewZ(texture2D(tDepth, vUv + vec2(-o.x, 0.0)).r);
    float zr = viewZ(texture2D(tDepth, vUv + vec2( o.x, 0.0)).r);
    float zd = viewZ(texture2D(tDepth, vUv + vec2(0.0, -o.y)).r);
    float zu = viewZ(texture2D(tDepth, vUv + vec2(0.0,  o.y)).r);

    // How much closer is the nearest neighbour than us? If a lot, we are the
    // background sitting right behind a silhouette -> darken.
    float nearest = min(min(zl, zr), min(zd, zu));
    float step0 = max(0.0, z0 - nearest);
    // Scale-invariant: a 30cm step at 5m and at 50m read the same.
    float e = clamp(step0 / (z0 * 0.035 + 0.05), 0.0, 1.0);
    e = pow(e, 0.65);
    color *= 1.0 - e * uEdgeStrength;
  }
#endif

#ifdef USE_FOG
  if (!isSky) {
    float z0 = viewZ(depth);
    // World-space offset from the camera to the shaded point.
    vec3 ray = worldRay * z0;
    float dist = length(ray);
    float dEff = max(dist - uFogParams.w, 0.0);

    float H = uFogParams.y;
    float density = uFogParams.x;
    float dy = ray.y;

    // Analytic integral of an exponentially height-decaying medium along the
    // ray. The abs() guard is the classic degenerate case: a horizontal ray.
    float baseD = density * exp(-H * (uCameraPos.y - uFogParams.z));
    float integral;
    if (abs(dy) > 0.01) {
      integral = baseD * (1.0 - exp(-H * dy)) / (H * dy);
    } else {
      integral = baseD;
    }
    float tau = max(0.0, integral) * dEff;
    float f = 1.0 - exp(-tau);
    f = clamp(f * uFogAmount, 0.0, 1.0);

    color = mix(color, atmosphereBase(dirWorld), f);
  }
#endif

  gl_FragColor = vec4(color, 1.0);
}
`;

export interface FogPassUniforms extends Record<string, THREE.IUniform> {
  tScene: THREE.IUniform<THREE.Texture | null>;
  tDepth: THREE.IUniform<THREE.Texture | null>;
  uInvProjection: THREE.IUniform<THREE.Matrix4>;
  uCameraWorld: THREE.IUniform<THREE.Matrix4>;
  uCameraPos: THREE.IUniform<THREE.Vector3>;
  uNearFar: THREE.IUniform<THREE.Vector2>;
  uTexel: THREE.IUniform<THREE.Vector2>;
  uFogParams: THREE.IUniform<THREE.Vector4>;
  uFogAmount: THREE.IUniform<number>;
  uEdgeStrength: THREE.IUniform<number>;
  uEdgeWidth: THREE.IUniform<number>;
}

export function createFogPass(atmosphere: AtmosphereUniforms): FullScreenPass {
  const uniforms: FogPassUniforms = {
    tScene: { value: null },
    tDepth: { value: null },
    uInvProjection: { value: new THREE.Matrix4() },
    uCameraWorld: { value: new THREE.Matrix4() },
    uCameraPos: { value: new THREE.Vector3() },
    uNearFar: { value: new THREE.Vector2(0.15, 4000) },
    uTexel: { value: new THREE.Vector2(1 / 854, 1 / 480) },
    uFogParams: { value: new THREE.Vector4(0.0075, 0.028, 0.0, 24.0) },
    uFogAmount: { value: 1.0 },
    uEdgeStrength: { value: 0.32 },
    uEdgeWidth: { value: 1.25 },
    ...atmosphere,
  };
  return new FullScreenPass({
    fragmentShader: FRAG,
    uniforms,
    defines: { USE_FOG: 1, USE_EDGE_DARKEN: 1 },
  });
}
