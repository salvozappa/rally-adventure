import * as THREE from 'three';
import { FullScreenPass } from './fullscreen';

/**
 * Display pass: nearest-neighbour upscale of the low-res, already-quantised
 * frame, plus the lens-y bits that belong at display resolution.
 *
 * Chromatic aberration and vignette run AFTER the upscale by design — they are
 * lens artefacts, not framebuffer artefacts, and doing them at 480p makes the
 * fringing chunky and obvious. The CA offset is snapped to whole low-res texels
 * so it shifts entire pixel blocks and never softens the hard pixel edges the
 * whole look depends on.
 */

const FRAG = /* glsl */ `
precision highp float;
varying vec2 vUv;

uniform sampler2D tSrc;
uniform vec2 uLowRes;
uniform float uVignetteStrength;
uniform float uVignetteRadius;
uniform float uAberration;
uniform float uScanlineStrength;
uniform float uScanlineCount;
uniform float uApertureStrength;
uniform vec2 uOutputRes;

void main() {
  vec2 uv = vUv;
  vec2 centred = uv - 0.5;
  float r2 = dot(centred, centred);

  vec3 col;
#ifdef USE_ABERRATION
  // Snap the radial offset to whole source texels: keeps the blocks crisp.
  // The offset is expressed in TEXELS, not UV units, and hard-capped at three:
  // as a raw UV offset this trivially exceeds the whole texture and turns the
  // image into rainbow noise, which is exactly what it used to do.
  vec2 texel = 1.0 / uLowRes;
  vec2 dir = centred * (r2 * uAberration);
  vec2 nTexels = clamp(floor(abs(dir) + 0.5), 0.0, 3.0);
  vec2 offs = nTexels * texel * sign(dir);
  col.r = texture2D(tSrc, uv + offs).r;
  col.g = texture2D(tSrc, uv).g;
  col.b = texture2D(tSrc, uv - offs).b;
#else
  col = texture2D(tSrc, uv).rgb;
#endif

#ifdef USE_SCANLINES
  float line = sin(uv.y * uScanlineCount * 3.14159265) * 0.5 + 0.5;
  col *= 1.0 - uScanlineStrength * line;
  // Aperture grille: a soft RGB stripe at display resolution.
  float ph = gl_FragCoord.x * 3.14159265 * 0.6666667;
  vec3 mask = vec3(
    0.5 + 0.5 * cos(ph),
    0.5 + 0.5 * cos(ph - 2.0943951),
    0.5 + 0.5 * cos(ph + 2.0943951)
  );
  col *= mix(vec3(1.0), mask * 1.35, uApertureStrength);
#endif

#ifdef USE_VIGNETTE
  float v = smoothstep(uVignetteRadius, uVignetteRadius - 0.55, sqrt(r2));
  col *= mix(1.0, v, uVignetteStrength);
#endif

  gl_FragColor = vec4(col, 1.0);
}
`;

export function createOutputPass(): FullScreenPass {
  return new FullScreenPass({
    fragmentShader: FRAG,
    uniforms: {
      tSrc: { value: null },
      uLowRes: { value: new THREE.Vector2(854, 480) },
      uOutputRes: { value: new THREE.Vector2(1920, 1080) },
      uVignetteStrength: { value: 0.34 },
      uVignetteRadius: { value: 0.86 },
      uAberration: { value: 1.35 },
      uScanlineStrength: { value: 0.18 },
      uScanlineCount: { value: 480 },
      uApertureStrength: { value: 0.16 },
    },
    defines: { USE_VIGNETTE: 1, USE_ABERRATION: 1 },
  });
}
