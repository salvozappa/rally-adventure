import * as THREE from 'three';
import type { Engine } from '../core/Engine';
import { makeRenderTarget } from './passes/fullscreen';
import type { FullScreenPass } from './passes/fullscreen';
import { BloomChain } from './passes/BloomPass';
import { GodRayChain } from './passes/GodRayPass';
import { createFogPass } from './passes/FogPass';
import type { FogPassUniforms } from './passes/FogPass';
import { createCompositePass } from './passes/CompositePass';
import { createOutputPass } from './passes/OutputPass';
import { createBayerTexture, createNoiseTexture } from './passes/bayer';
import { buildGradeLUT, cloneGrade, DEFAULT_GRADE } from './passes/lut';
import type { GradeParams } from './passes/lut';
import { createAtmosphereUniforms } from './passes/atmosphere';
import type { AtmosphereUniforms } from './passes/atmosphere';
import { configureTextures } from './textures';

/**
 * The retro render chain.
 *
 *   scene  -> [HDR half-float RT, low internal resolution, with depth]
 *          -> bloom      (half of low-res)
 *          -> god rays   (half of low-res, needs the depth buffer)
 *          -> fog        (low-res, needs the depth buffer)
 *          -> composite  (low-res: HDR combine, tonemap, gamma, LUT grade,
 *                         quantise + ordered dither)
 *          -> output     (display-res: nearest upscale, chromatic aberration,
 *                         vignette, optional scanlines)
 *
 * Two ordering decisions carry the whole look:
 *
 *  - Everything before `output` runs at the LOW resolution. In particular the
 *    quantise + dither happens there, so the dither pattern is one low-res
 *    texel across and the nearest upscale multiplies it into chunky era pixels.
 *    Dithering after the upscale would put a 1-display-pixel screen door over
 *    the image, which reads as a broken monitor, not as 1997.
 *
 *  - Chromatic aberration and vignette run AFTER the upscale, because they are
 *    lens artefacts rather than framebuffer artefacts.
 */

export type DitherMode = 'bayer' | 'noise';

export interface RetroOptions {
  /** Master switch. False renders the scene straight to the canvas for A/B. */
  enabled: boolean;
  /** Internal render height in pixels. 360 / 480 / 540 are the useful values. */
  internalHeight: number;

  // --- pass toggles -------------------------------------------------------
  bloom: boolean;
  godRays: boolean;
  fog: boolean;
  edgeDarken: boolean;
  grade: boolean;
  quantise: boolean;
  dither: boolean;
  vignette: boolean;
  aberration: boolean;
  scanlines: boolean;

  // --- tuning -------------------------------------------------------------
  exposure: number;
  /** Where the tonemap shoulder starts. Lower = softer highlights. */
  shoulder: number;
  bloomStrength: number;
  bloomThreshold: number;
  godRayStrength: number;
  /** Bits per pixel for the quantise step: 15 (RGB555), 16 (RGB565), 18, 21. */
  quantiseBits: 15 | 16 | 18 | 21;
  ditherMode: DitherMode;
  /** 0 = hard round (banding), 1 = full-strength ordered dither. */
  ditherAmount: number;

  fogDensity: number;
  fogHeightFalloff: number;
  fogGroundLevel: number;
  fogStart: number;
  fogAmount: number;
  edgeStrength: number;

  vignetteStrength: number;
  aberrationStrength: number;
  scanlineStrength: number;
  apertureStrength: number;
}

export const DEFAULT_RETRO_OPTIONS: RetroOptions = {
  enabled: true,
  // 480p: the sweet spot. 360 reads as broken on a high-DPI display and 540
  // starts to lose the chunky-texel signature that makes the style legible.
  internalHeight: 480,

  bloom: true,
  godRays: true,
  fog: true,
  edgeDarken: true,
  grade: true,
  quantise: true,
  dither: true,
  vignette: true,
  aberration: true,
  scanlines: false,

  exposure: 1.0,
  shoulder: 0.72,
  bloomStrength: 0.42,
  bloomThreshold: 0.86,
  godRayStrength: 0.34,
  quantiseBits: 16,
  ditherMode: 'bayer',
  // Restrained on purpose. At 5 bits per channel a full-strength 8x8 Bayer is
  // a visible crosshatch once you upscale it; ~0.55 keeps the banding gone
  // while the pattern stays below the threshold of "that looks like a bug".
  ditherAmount: 0.55,

  fogDensity: 0.0062,
  fogHeightFalloff: 0.02,
  fogGroundLevel: 0,
  fogStart: 30,
  fogAmount: 1,
  edgeStrength: 0.26,

  vignetteStrength: 0.3,
  // Measured in source texels of radial split at the frame corner. Keep it
  // around one texel — two is already visible as a coloured rim, and anything
  // much larger stops reading as a lens artefact.
  aberrationStrength: 1.2,
  scanlineStrength: 0.16,
  apertureStrength: 0.14,
};

/** Bits -> per-channel level count for the quantise step. */
const QUANT_LEVELS: Record<number, [number, number, number]> = {
  15: [31, 31, 31],
  16: [31, 63, 31],
  18: [63, 63, 63],
  21: [127, 127, 127],
};

const _sunWorld = new THREE.Vector3();
const _camDir = new THREE.Vector3();

export class RetroPipeline {
  readonly options: RetroOptions;
  /** Shared with Sky and Lighting so sky, fog and light can never disagree. */
  readonly atmosphere: AtmosphereUniforms;
  readonly grade: GradeParams;

  private readonly engine: Engine;
  private readonly renderer: THREE.WebGLRenderer;

  private sceneRT: THREE.WebGLRenderTarget;
  private fogRT: THREE.WebGLRenderTarget;
  private lowRT: THREE.WebGLRenderTarget;
  private readonly bloom: BloomChain;
  private readonly godRays: GodRayChain;
  private readonly fogPass: FullScreenPass;
  private readonly composite: FullScreenPass;
  private readonly output: FullScreenPass;

  private readonly lut: THREE.DataTexture;
  private readonly bayerTex: THREE.DataTexture;
  private readonly noiseTex: THREE.DataTexture;
  private readonly black: THREE.DataTexture;

  private displayW = 1;
  private displayH = 1;
  private lowW = 1;
  private lowH = 1;
  private sunVisibility = 0;
  private readonly sunUv = new THREE.Vector2(0.5, 0.5);
  private disposed = false;

  constructor(engine: Engine, opts: Partial<RetroOptions> = {}) {
    this.engine = engine;
    this.renderer = engine.renderer;
    this.options = { ...DEFAULT_RETRO_OPTIONS, ...opts };
    this.atmosphere = createAtmosphereUniforms();
    this.grade = cloneGrade(DEFAULT_GRADE);

    // The only place in the app that knows the real device anisotropy limit.
    configureTextures(this.renderer);

    this.lut = buildGradeLUT(this.grade);
    this.bayerTex = createBayerTexture(8);
    this.noiseTex = createNoiseTexture(64);
    this.black = makeBlackTexture();

    this.sceneRT = makeRenderTarget(1, 1, { half: true, depth: true });
    this.fogRT = makeRenderTarget(1, 1, { half: true, nearest: true });
    this.lowRT = makeRenderTarget(1, 1, { nearest: true });

    this.bloom = new BloomChain(1, 1);
    this.godRays = new GodRayChain(1, 1);
    this.fogPass = createFogPass(this.atmosphere);
    this.composite = createCompositePass();
    this.output = createOutputPass();

    this.composite.uniforms.tLut.value = this.lut;

    engine.onResize((w, h) => this.resize(w, h));
    this.applyOptions();
  }

  // ---------------------------------------------------------------- sizing

  resize(w: number, h: number): void {
    this.displayW = Math.max(1, Math.floor(w));
    this.displayH = Math.max(1, Math.floor(h));
    this.rebuildTargets();
  }

  setInternalHeight(px: number): void {
    const clamped = Math.max(120, Math.min(1080, Math.round(px)));
    if (clamped === this.options.internalHeight) return;
    this.options.internalHeight = clamped;
    this.rebuildTargets();
  }

  private rebuildTargets(): void {
    const aspect = this.displayW / Math.max(1, this.displayH);
    // Even dimensions keep every half-resolution chain (bloom, god rays) on an
    // exact texel grid, which stops the mips drifting half a pixel per level.
    const lh = Math.max(2, this.options.internalHeight & ~1);
    const lw = Math.max(2, Math.round(lh * aspect) & ~1);
    if (lw === this.lowW && lh === this.lowH && this.sceneRT.width === lw) {
      // Display size changed but internal size did not; still resize output.
      this.syncOutputUniforms();
      return;
    }
    this.lowW = lw;
    this.lowH = lh;

    this.sceneRT.dispose();
    this.fogRT.dispose();
    this.lowRT.dispose();
    this.sceneRT = makeRenderTarget(lw, lh, { half: true, depth: true });
    this.fogRT = makeRenderTarget(lw, lh, { half: true, nearest: true });
    this.lowRT = makeRenderTarget(lw, lh, { nearest: true });

    this.bloom.resize(lw, lh);
    this.godRays.resize(lw, lh);

    (this.fogPass.uniforms.uTexel.value as THREE.Vector2).set(1 / lw, 1 / lh);
    this.syncOutputUniforms();
  }

  private syncOutputUniforms(): void {
    const ou = this.output.uniforms;
    (ou.uLowRes.value as THREE.Vector2).set(this.lowW, this.lowH);
    (ou.uOutputRes.value as THREE.Vector2).set(this.displayW, this.displayH);
    ou.uScanlineCount.value = this.lowH;
  }

  /** Current internal resolution, for HUDs and preview readouts. */
  get internalSize(): { width: number; height: number } {
    return { width: this.lowW, height: this.lowH };
  }

  // --------------------------------------------------------------- options

  setOption<K extends keyof RetroOptions>(k: K, v: RetroOptions[K]): void {
    if (this.options[k] === v) return;
    this.options[k] = v;
    if (k === 'internalHeight') {
      this.rebuildTargets();
      return;
    }
    this.applyOptions();
  }

  /** Rebuild the grading LUT after mutating `grade`. */
  refreshGrade(): void {
    buildGradeLUT(this.grade, this.lut);
  }

  private applyOptions(): void {
    const o = this.options;

    // --- fog / edge darkening ---------------------------------------------
    const fu = this.fogPass.uniforms as FogPassUniforms;
    this.fogPass.setDefine('USE_FOG', o.fog ? 1 : null);
    this.fogPass.setDefine('USE_EDGE_DARKEN', o.edgeDarken ? 1 : null);
    fu.uFogParams.value.set(o.fogDensity, o.fogHeightFalloff, o.fogGroundLevel, o.fogStart);
    fu.uFogAmount.value = o.fogAmount;
    fu.uEdgeStrength.value = o.edgeStrength;

    // --- bloom -------------------------------------------------------------
    this.bloom.threshold = o.bloomThreshold;

    // --- composite ---------------------------------------------------------
    const cu = this.composite.uniforms;
    this.composite.setDefine('USE_BLOOM', o.bloom ? 1 : null);
    this.composite.setDefine('USE_GODRAYS', o.godRays ? 1 : null);
    this.composite.setDefine('USE_LUT', o.grade ? 1 : null);
    this.composite.setDefine('USE_QUANTISE', o.quantise ? 1 : null);
    this.composite.setDefine('USE_DITHER', o.quantise && o.dither ? 1 : null);
    cu.uExposure.value = o.exposure;
    cu.uShoulder.value = o.shoulder;
    cu.uBloomStrength.value = o.bloomStrength;
    const levels = QUANT_LEVELS[o.quantiseBits] ?? QUANT_LEVELS[16];
    (cu.uQuantLevels.value as THREE.Vector3).set(levels[0], levels[1], levels[2]);
    cu.uDitherAmount.value = o.ditherAmount;
    if (o.ditherMode === 'noise') {
      cu.tDither.value = this.noiseTex;
      (cu.uDitherTexelScale.value as THREE.Vector2).set(1 / 64, 1 / 64);
    } else {
      cu.tDither.value = this.bayerTex;
      (cu.uDitherTexelScale.value as THREE.Vector2).set(1 / 8, 1 / 8);
    }

    // --- output ------------------------------------------------------------
    const ou = this.output.uniforms;
    this.output.setDefine('USE_VIGNETTE', o.vignette ? 1 : null);
    this.output.setDefine('USE_ABERRATION', o.aberration ? 1 : null);
    this.output.setDefine('USE_SCANLINES', o.scanlines ? 1 : null);
    ou.uVignetteStrength.value = o.vignetteStrength;
    ou.uAberration.value = o.aberrationStrength;
    ou.uScanlineStrength.value = o.scanlineStrength;
    ou.uApertureStrength.value = o.apertureStrength;
  }

  // ---------------------------------------------------------------- render

  render(_dt: number): void {
    if (this.disposed) return;
    const { renderer, engine } = this;
    const camera = engine.camera;
    const scene = engine.scene;

    if (!this.options.enabled) {
      renderer.setRenderTarget(null);
      renderer.render(scene, camera);
      return;
    }

    this.applyOptions();
    this.updateSunProjection(camera);

    // 1 --- scene into the low-res HDR buffer -------------------------------
    renderer.setRenderTarget(this.sceneRT);
    renderer.clear(true, true, false);
    renderer.render(scene, camera);

    // 2 --- bloom ------------------------------------------------------------
    if (this.options.bloom) {
      this.bloom.render(renderer, this.sceneRT.texture, this.lowW, this.lowH);
      this.composite.uniforms.tBloom.value = this.bloom.texture;
    } else {
      this.composite.uniforms.tBloom.value = this.black;
    }

    // 3 --- god rays ---------------------------------------------------------
    const depthTex = this.sceneRT.depthTexture as THREE.DepthTexture;
    if (this.options.godRays && this.sunVisibility > 0.001) {
      this.godRays.render(renderer, this.sceneRT.texture, depthTex, this.sunUv);
      this.composite.uniforms.tGodrays.value = this.godRays.texture;
      this.composite.uniforms.uGodrayStrength.value =
        this.options.godRayStrength * this.sunVisibility;
    } else {
      this.composite.uniforms.tGodrays.value = this.black;
      this.composite.uniforms.uGodrayStrength.value = 0;
    }
    // The shafts inherit the sun's own colour so they stay coherent with the
    // sky across every time-of-day preset.
    const glow = this.atmosphere.uSunGlowColor.value;
    (this.composite.uniforms.uGodrayTint.value as THREE.Vector3).set(glow.r, glow.g, glow.b);

    // 4 --- fog + edge darkening ---------------------------------------------
    const fu = this.fogPass.uniforms as FogPassUniforms;
    fu.tScene.value = this.sceneRT.texture;
    fu.tDepth.value = depthTex;
    fu.uInvProjection.value.copy(camera.projectionMatrixInverse);
    fu.uCameraWorld.value.copy(camera.matrixWorld);
    fu.uCameraPos.value.copy(camera.position);
    fu.uNearFar.value.set(camera.near, camera.far);
    this.fogPass.render(renderer, this.fogRT);

    // 5 --- composite: tonemap, grade, quantise + dither, all at low res ------
    this.composite.uniforms.tScene.value = this.fogRT.texture;
    this.composite.render(renderer, this.lowRT);

    // 6 --- nearest upscale + lens artefacts ---------------------------------
    this.output.uniforms.tSrc.value = this.lowRT.texture;
    renderer.setRenderTarget(null);
    this.output.render(renderer, null);
  }

  /**
   * Projects the sun into screen space and derives how strongly the shafts
   * should show. Two independent falloffs: the sun must be in front of the
   * camera at all, and shafts fade as it leaves the frame so they never pop.
   */
  private updateSunProjection(camera: THREE.PerspectiveCamera): void {
    const dir = this.atmosphere.uSunDirection.value;
    camera.getWorldDirection(_camDir);
    const facing = _camDir.dot(dir);
    if (facing <= 0.02 || dir.y < -0.05) {
      this.sunVisibility = 0;
      return;
    }

    _sunWorld.copy(camera.position).addScaledVector(dir, 1000);
    _sunWorld.project(camera);
    this.sunUv.set(_sunWorld.x * 0.5 + 0.5, _sunWorld.y * 0.5 + 0.5);

    // Distance of the sun from the frame, in screen units. 0 inside, growing
    // outside; a soft window keeps the shafts alive just off-frame.
    const dx = Math.max(0, Math.abs(this.sunUv.x - 0.5) - 0.5);
    const dy = Math.max(0, Math.abs(this.sunUv.y - 0.5) - 0.5);
    const off = Math.hypot(dx, dy);
    const frame = 1 - smoothstep(0.0, 0.5, off);
    const axis = smoothstep(0.02, 0.45, facing);
    const above = smoothstep(-0.05, 0.06, dir.y);
    this.sunVisibility = frame * axis * above;
  }

  // --------------------------------------------------------------- teardown

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.sceneRT.dispose();
    this.fogRT.dispose();
    this.lowRT.dispose();
    this.bloom.dispose();
    this.godRays.dispose();
    this.fogPass.dispose();
    this.composite.dispose();
    this.output.dispose();
    this.lut.dispose();
    this.bayerTex.dispose();
    this.noiseTex.dispose();
    this.black.dispose();
  }
}

function smoothstep(e0: number, e1: number, x: number): number {
  const t = Math.max(0, Math.min(1, (x - e0) / (e1 - e0 || 1e-6)));
  return t * t * (3 - 2 * t);
}

/** 1x1 black stand-in so disabled passes cost a texture fetch and nothing else. */
function makeBlackTexture(): THREE.DataTexture {
  const tex = new THREE.DataTexture(
    new Uint8Array([0, 0, 0, 255]),
    1,
    1,
    THREE.RGBAFormat,
    THREE.UnsignedByteType,
  );
  tex.colorSpace = THREE.NoColorSpace;
  tex.needsUpdate = true;
  return tex;
}
