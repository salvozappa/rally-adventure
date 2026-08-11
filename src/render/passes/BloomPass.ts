import * as THREE from 'three';
import { FullScreenPass, makeRenderTarget } from './fullscreen';

/**
 * Three-mip separable-gaussian bloom.
 *
 * Deliberately modest: the target look is 1997, and 1997 had no bloom at all.
 * A little glow on the sun disc and on specular hits reads as "expensive
 * modern lighting" without breaking the era. Anything more and it turns into a
 * 2007 shooter.
 */

const BRIGHT_FRAG = /* glsl */ `
precision highp float;
varying vec2 vUv;
uniform sampler2D tSrc;
uniform vec2 uTexel;
uniform float uThreshold;
uniform float uKnee;

void main() {
  // 4-tap box downsample first: halves the sampling cost of the blur chain and
  // kills the worst of the temporal fizz on single bright pixels.
  vec3 c = texture2D(tSrc, vUv + vec2(-uTexel.x, -uTexel.y)).rgb;
  c += texture2D(tSrc, vUv + vec2( uTexel.x, -uTexel.y)).rgb;
  c += texture2D(tSrc, vUv + vec2(-uTexel.x,  uTexel.y)).rgb;
  c += texture2D(tSrc, vUv + vec2( uTexel.x,  uTexel.y)).rgb;
  c *= 0.25;

  float br = max(c.r, max(c.g, c.b));
  float knee = max(uKnee, 1e-4);
  float soft = clamp(br - uThreshold + knee, 0.0, 2.0 * knee);
  soft = soft * soft / (4.0 * knee);
  float contrib = max(soft, br - uThreshold) / max(br, 1e-4);
  gl_FragColor = vec4(c * contrib, 1.0);
}
`;

const BLUR_FRAG = /* glsl */ `
precision highp float;
varying vec2 vUv;
uniform sampler2D tSrc;
uniform vec2 uDirection;

void main() {
  // 9-tap gaussian collapsed to 5 hardware-bilinear taps.
  vec3 c = texture2D(tSrc, vUv).rgb * 0.2270270270;
  vec2 o1 = uDirection * 1.3846153846;
  vec2 o2 = uDirection * 3.2307692308;
  c += texture2D(tSrc, vUv + o1).rgb * 0.3162162162;
  c += texture2D(tSrc, vUv - o1).rgb * 0.3162162162;
  c += texture2D(tSrc, vUv + o2).rgb * 0.0702702703;
  c += texture2D(tSrc, vUv - o2).rgb * 0.0702702703;
  gl_FragColor = vec4(c, 1.0);
}
`;

const DOWN_FRAG = /* glsl */ `
precision highp float;
varying vec2 vUv;
uniform sampler2D tSrc;
uniform vec2 uTexel;
void main() {
  vec3 c = texture2D(tSrc, vUv + vec2(-uTexel.x, -uTexel.y)).rgb;
  c += texture2D(tSrc, vUv + vec2( uTexel.x, -uTexel.y)).rgb;
  c += texture2D(tSrc, vUv + vec2(-uTexel.x,  uTexel.y)).rgb;
  c += texture2D(tSrc, vUv + vec2( uTexel.x,  uTexel.y)).rgb;
  gl_FragColor = vec4(c * 0.25, 1.0);
}
`;

const UPCOMBINE_FRAG = /* glsl */ `
precision highp float;
varying vec2 vUv;
uniform sampler2D tSrc;
uniform sampler2D tHigher;
uniform float uMix;
void main() {
  vec3 a = texture2D(tSrc, vUv).rgb;
  vec3 b = texture2D(tHigher, vUv).rgb;
  gl_FragColor = vec4(a + b * uMix, 1.0);
}
`;

interface Mip {
  a: THREE.WebGLRenderTarget;
  b: THREE.WebGLRenderTarget;
  w: number;
  h: number;
}

const MIP_COUNT = 3;

export class BloomChain {
  private mips: Mip[] = [];
  private readonly bright: FullScreenPass;
  private readonly blur: FullScreenPass;
  private readonly down: FullScreenPass;
  private readonly combine: FullScreenPass;

  threshold = 0.82;
  knee = 0.35;
  /** 0..1 — how much of the wide mips survives into the result. */
  radius = 0.72;

  constructor(w: number, h: number) {
    this.bright = new FullScreenPass({
      fragmentShader: BRIGHT_FRAG,
      uniforms: {
        tSrc: { value: null },
        uTexel: { value: new THREE.Vector2() },
        uThreshold: { value: this.threshold },
        uKnee: { value: this.knee },
      },
    });
    this.blur = new FullScreenPass({
      fragmentShader: BLUR_FRAG,
      uniforms: { tSrc: { value: null }, uDirection: { value: new THREE.Vector2() } },
    });
    this.down = new FullScreenPass({
      fragmentShader: DOWN_FRAG,
      uniforms: { tSrc: { value: null }, uTexel: { value: new THREE.Vector2() } },
    });
    this.combine = new FullScreenPass({
      fragmentShader: UPCOMBINE_FRAG,
      uniforms: { tSrc: { value: null }, tHigher: { value: null }, uMix: { value: 1 } },
    });
    this.resize(w, h);
  }

  get texture(): THREE.Texture {
    return this.mips[0].a.texture;
  }

  resize(w: number, h: number): void {
    for (const m of this.mips) {
      m.a.dispose();
      m.b.dispose();
    }
    this.mips = [];
    let mw = Math.max(2, Math.floor(w / 2));
    let mh = Math.max(2, Math.floor(h / 2));
    for (let i = 0; i < MIP_COUNT; i++) {
      this.mips.push({
        a: makeRenderTarget(mw, mh, { half: true }),
        b: makeRenderTarget(mw, mh, { half: true }),
        w: mw,
        h: mh,
      });
      mw = Math.max(2, Math.floor(mw / 2));
      mh = Math.max(2, Math.floor(mh / 2));
    }
  }

  render(renderer: THREE.WebGLRenderer, source: THREE.Texture, srcW: number, srcH: number): void {
    const bu = this.bright.uniforms;
    bu.tSrc.value = source;
    (bu.uTexel.value as THREE.Vector2).set(0.5 / srcW, 0.5 / srcH);
    bu.uThreshold.value = this.threshold;
    bu.uKnee.value = this.knee;
    this.bright.render(renderer, this.mips[0].a);

    for (let i = 0; i < MIP_COUNT; i++) {
      const m = this.mips[i];
      if (i > 0) {
        const prev = this.mips[i - 1];
        this.down.uniforms.tSrc.value = prev.a.texture;
        (this.down.uniforms.uTexel.value as THREE.Vector2).set(0.5 / prev.w, 0.5 / prev.h);
        this.down.render(renderer, m.a);
      }
      // Two-pass separable blur, run twice for a wider kernel on the small mips.
      const passes = i === 0 ? 1 : 2;
      for (let p = 0; p < passes; p++) {
        this.blur.uniforms.tSrc.value = m.a.texture;
        (this.blur.uniforms.uDirection.value as THREE.Vector2).set(1 / m.w, 0);
        this.blur.render(renderer, m.b);
        this.blur.uniforms.tSrc.value = m.b.texture;
        (this.blur.uniforms.uDirection.value as THREE.Vector2).set(0, 1 / m.h);
        this.blur.render(renderer, m.a);
      }
    }

    // Fold the small (wide) mips back up into mip 0.
    for (let i = MIP_COUNT - 1; i > 0; i--) {
      const src = this.mips[i];
      const dst = this.mips[i - 1];
      this.combine.uniforms.tSrc.value = dst.a.texture;
      this.combine.uniforms.tHigher.value = src.a.texture;
      this.combine.uniforms.uMix.value = this.radius;
      this.combine.render(renderer, dst.b);
      // Swap so the accumulated result stays in `a`.
      const tmp = dst.a;
      (dst as { a: THREE.WebGLRenderTarget }).a = dst.b;
      (dst as { b: THREE.WebGLRenderTarget }).b = tmp;
    }
  }

  dispose(): void {
    for (const m of this.mips) {
      m.a.dispose();
      m.b.dispose();
    }
    this.mips = [];
    this.bright.dispose();
    this.blur.dispose();
    this.down.dispose();
    this.combine.dispose();
  }
}
