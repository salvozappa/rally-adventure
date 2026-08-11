import * as THREE from 'three';
import { FullScreenPass, makeRenderTarget } from './fullscreen';

/**
 * Screen-space light shafts (Mitchell's radial-blur formulation).
 *
 * The occlusion buffer keeps only sky pixels, so anything solid — a ridge, a
 * tree, the car — punches a hole in the shafts. Over a dusty valley with the
 * sun low this is the single most atmospheric thing in the chain.
 */

const OCCLUSION_FRAG = /* glsl */ `
precision highp float;
varying vec2 vUv;
uniform sampler2D tScene;
uniform sampler2D tDepth;
uniform float uThreshold;

void main() {
  float d = texture2D(tDepth, vUv).r;
  // Only unoccluded sky emits. Geometry is a hard black cut-out.
  float sky = step(0.999995, d);
  vec3 c = texture2D(tScene, vUv).rgb;
  float br = max(c.r, max(c.g, c.b));
  float w = max(br - uThreshold, 0.0) / max(br, 1e-4);
  gl_FragColor = vec4(c * w * sky, 1.0);
}
`;

const RADIAL_FRAG = /* glsl */ `
precision highp float;
varying vec2 vUv;
uniform sampler2D tSrc;
uniform vec2 uSunUv;
uniform float uDensity;
uniform float uDecay;
uniform float uWeight;
uniform float uJitter;
uniform vec2 uTexel;

#ifndef GODRAY_SAMPLES
#define GODRAY_SAMPLES 24
#endif

float hash12(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

void main() {
  vec2 delta = (vUv - uSunUv) * (uDensity / float(GODRAY_SAMPLES));
  // Per-pixel start offset breaks the concentric banding the naive loop gives.
  float jitter = hash12(gl_FragCoord.xy) * uJitter;
  vec2 uv = vUv - delta * jitter;
  vec3 acc = vec3(0.0);
  float illum = 1.0;
  for (int i = 0; i < GODRAY_SAMPLES; i++) {
    uv -= delta;
    vec2 cuv = clamp(uv, vec2(0.0), vec2(1.0));
    acc += texture2D(tSrc, cuv).rgb * illum;
    illum *= uDecay;
  }
  gl_FragColor = vec4(acc * (uWeight / float(GODRAY_SAMPLES)), 1.0);
}
`;

export class GodRayChain {
  private occRT: THREE.WebGLRenderTarget;
  private rayRT: THREE.WebGLRenderTarget;
  private w = 1;
  private h = 1;
  private readonly occ: FullScreenPass;
  private readonly radial: FullScreenPass;

  threshold = 0.55;
  density = 0.82;
  decay = 0.93;
  weight = 1.35;
  /** Multiplied into the final strength; driven by how near the sun is to the view axis. */
  visibility = 0;

  constructor(w: number, h: number) {
    this.occ = new FullScreenPass({
      fragmentShader: OCCLUSION_FRAG,
      uniforms: { tScene: { value: null }, tDepth: { value: null }, uThreshold: { value: 0.55 } },
    });
    this.radial = new FullScreenPass({
      fragmentShader: RADIAL_FRAG,
      uniforms: {
        tSrc: { value: null },
        uSunUv: { value: new THREE.Vector2(0.5, 0.5) },
        uDensity: { value: 0.82 },
        uDecay: { value: 0.93 },
        uWeight: { value: 1.35 },
        uJitter: { value: 1.0 },
        uTexel: { value: new THREE.Vector2() },
      },
    });
    this.occRT = makeRenderTarget(1, 1, { half: true });
    this.rayRT = makeRenderTarget(1, 1, { half: true });
    this.resize(w, h);
  }

  get texture(): THREE.Texture {
    return this.rayRT.texture;
  }

  resize(w: number, h: number): void {
    this.occRT.dispose();
    this.rayRT.dispose();
    this.w = Math.max(2, Math.floor(w / 2));
    this.h = Math.max(2, Math.floor(h / 2));
    this.occRT = makeRenderTarget(this.w, this.h, { half: true });
    this.rayRT = makeRenderTarget(this.w, this.h, { half: true });
  }

  render(
    renderer: THREE.WebGLRenderer,
    scene: THREE.Texture,
    depth: THREE.Texture,
    sunUv: THREE.Vector2,
  ): void {
    this.occ.uniforms.tScene.value = scene;
    this.occ.uniforms.tDepth.value = depth;
    this.occ.uniforms.uThreshold.value = this.threshold;
    this.occ.render(renderer, this.occRT);

    const ru = this.radial.uniforms;
    (ru.uSunUv.value as THREE.Vector2).copy(sunUv);
    (ru.uTexel.value as THREE.Vector2).set(1 / this.w, 1 / this.h);
    ru.uDensity.value = this.density;
    ru.uDecay.value = this.decay;
    ru.uWeight.value = this.weight;
    ru.tSrc.value = this.occRT.texture;
    this.radial.render(renderer, this.rayRT);

    // Second, tighter pass. Chaining two radial blurs reaches further than one
    // long one for the same sample count and looks smoother.
    ru.tSrc.value = this.rayRT.texture;
    ru.uDensity.value = this.density * 0.42;
    ru.uWeight.value = 1.0;
    this.radial.render(renderer, this.occRT);
    const tmp = this.rayRT;
    this.rayRT = this.occRT;
    this.occRT = tmp;
  }

  dispose(): void {
    this.occRT.dispose();
    this.rayRT.dispose();
    this.occ.dispose();
    this.radial.dispose();
  }
}
