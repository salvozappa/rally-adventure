import * as THREE from 'three';
import {
  ATMOSPHERE_FUNCTIONS_GLSL,
  ATMOSPHERE_UNIFORMS_GLSL,
  NOISE_GLSL,
  createAtmosphereUniforms,
} from '../render/passes/atmosphere';
import type { AtmosphereUniforms } from '../render/passes/atmosphere';

/**
 * Sky dome, clouds, sun disc and the distant mountain silhouettes, plus the
 * time-of-day system that drives all of them (and the fog, and the lights) off
 * one set of numbers.
 *
 * Everything the sky draws is written into the SAME `AtmosphereUniforms`
 * instance the post-process fog samples. The dome fragment shader is literally
 * `atmosphereBase(dir) + sun + clouds`, and fully-fogged geometry resolves to
 * `atmosphereBase(dir)`. Distant terrain therefore dissolves into pixel-exact
 * sky: a horizon seam is not merely unlikely, it is unrepresentable.
 *
 * Draw order and depth:
 *   The dome and the mountain curtain render first with depth test and depth
 *   write OFF. They leave the depth buffer at its cleared 1.0, so the fog pass
 *   and the god-ray occlusion buffer both correctly classify them as sky, and
 *   real terrain (which does write depth) simply paints over them.
 */

// ---------------------------------------------------------------- keyframes

interface RGB {
  r: number;
  g: number;
  b: number;
}

const rgb = (r: number, g: number, b: number): RGB => ({ r, g, b });

/**
 * One instant of the day. All colours are LINEAR radiance — the composite pass
 * gamma-encodes at the very end, so authoring in linear is what keeps the
 * gradient smooth instead of plasticky.
 */
interface DayKey {
  t: number;
  sunElevation: number; // degrees above the horizon
  sunAzimuth: number; // degrees, 0 = +Z, increasing toward +X
  zenith: RGB;
  horizon: RGB;
  ground: RGB;
  sunGlow: RGB;
  sunDisc: RGB;
  /** x: gradient power, y: glow power, z: glow strength, w: haze lift */
  sky: [number, number, number, number];
  /** x: disc radius (rad), y: disc intensity, z: halo size (rad), w: halo strength */
  disc: [number, number, number, number];
  cloudLit: RGB;
  cloudShadow: RGB;
  sunColor: RGB;
  sunIntensity: number;
  ambientSky: RGB;
  ambientGround: RGB;
  ambientIntensity: number;
  fogDensity: number;
  /** How far the mountains have already dissolved into haze, 0..1. */
  ridgeHaze: number;
}

/**
 * The day curve. Colours sampled off the Screamer / Sega Rally reference set:
 * a deep saturated zenith blue collapsing to a pale, faintly warm horizon band,
 * with the whole thing swinging orange at each end of the day.
 */
const DAY: DayKey[] = [
  {
    t: 0.0,
    sunElevation: 1.5,
    sunAzimuth: 95,
    zenith: rgb(0.045, 0.085, 0.26),
    horizon: rgb(0.95, 0.44, 0.24),
    ground: rgb(0.30, 0.20, 0.16),
    sunGlow: rgb(1.0, 0.40, 0.16),
    sunDisc: rgb(1.0, 0.52, 0.26),
    sky: [2.1, 12.0, 0.9, 0.03],
    disc: [0.032, 14.0, 0.5, 0.8],
    cloudLit: rgb(1.0, 0.66, 0.46),
    cloudShadow: rgb(0.30, 0.24, 0.32),
    sunColor: rgb(1.0, 0.54, 0.30),
    sunIntensity: 0.7,
    ambientSky: rgb(0.26, 0.28, 0.44),
    ambientGround: rgb(0.30, 0.20, 0.14),
    ambientIntensity: 0.55,
    fogDensity: 0.0018,
    ridgeHaze: 0.6,
  },
  {
    t: 0.22,
    sunElevation: 27,
    sunAzimuth: 112,
    zenith: rgb(0.048, 0.155, 0.46),
    horizon: rgb(0.74, 0.66, 0.55),
    ground: rgb(0.40, 0.34, 0.27),
    sunGlow: rgb(1.0, 0.72, 0.42),
    sunDisc: rgb(1.0, 0.84, 0.58),
    sky: [2.5, 20.0, 0.62, 0.0],
    disc: [0.028, 20.0, 0.42, 0.62],
    cloudLit: rgb(1.0, 0.95, 0.86),
    cloudShadow: rgb(0.34, 0.36, 0.45),
    sunColor: rgb(1.0, 0.87, 0.71),
    sunIntensity: 1.55,
    ambientSky: rgb(0.24, 0.36, 0.60),
    ambientGround: rgb(0.32, 0.26, 0.19),
    ambientIntensity: 0.52,
    fogDensity: 0.0013,
    ridgeHaze: 0.44,
  },
  {
    t: 0.5,
    sunElevation: 61,
    sunAzimuth: 158,
    zenith: rgb(0.042, 0.175, 0.55),
    horizon: rgb(0.68, 0.685, 0.635),
    ground: rgb(0.44, 0.39, 0.31),
    sunGlow: rgb(1.0, 0.88, 0.68),
    sunDisc: rgb(1.0, 0.95, 0.82),
    sky: [2.8, 26.0, 0.5, 0.0],
    disc: [0.026, 24.0, 0.36, 0.5],
    cloudLit: rgb(1.06, 1.05, 1.0),
    cloudShadow: rgb(0.40, 0.43, 0.52),
    sunColor: rgb(1.0, 0.96, 0.89),
    sunIntensity: 2.05,
    ambientSky: rgb(0.23, 0.39, 0.66),
    ambientGround: rgb(0.34, 0.29, 0.21),
    ambientIntensity: 0.55,
    fogDensity: 0.0010,
    ridgeHaze: 0.36,
  },
  {
    t: 0.74,
    sunElevation: 29,
    sunAzimuth: 238,
    zenith: rgb(0.05, 0.16, 0.49),
    horizon: rgb(0.80, 0.67, 0.50),
    ground: rgb(0.44, 0.36, 0.26),
    sunGlow: rgb(1.0, 0.74, 0.40),
    sunDisc: rgb(1.0, 0.86, 0.62),
    sky: [2.4, 18.0, 0.68, 0.0],
    disc: [0.028, 20.0, 0.44, 0.66],
    cloudLit: rgb(1.04, 0.96, 0.85),
    cloudShadow: rgb(0.36, 0.35, 0.42),
    sunColor: rgb(1.0, 0.86, 0.66),
    sunIntensity: 1.75,
    ambientSky: rgb(0.24, 0.36, 0.58),
    ambientGround: rgb(0.36, 0.28, 0.19),
    ambientIntensity: 0.52,
    fogDensity: 0.0012,
    ridgeHaze: 0.44,
  },
  {
    t: 0.89,
    sunElevation: 7.5,
    sunAzimuth: 261,
    zenith: rgb(0.046, 0.115, 0.35),
    horizon: rgb(1.0, 0.55, 0.26),
    ground: rgb(0.36, 0.25, 0.18),
    sunGlow: rgb(1.0, 0.46, 0.17),
    sunDisc: rgb(1.0, 0.68, 0.34),
    sky: [2.1, 13.0, 0.88, 0.02],
    disc: [0.031, 16.0, 0.48, 0.78],
    cloudLit: rgb(1.05, 0.74, 0.50),
    cloudShadow: rgb(0.33, 0.27, 0.35),
    sunColor: rgb(1.0, 0.70, 0.42),
    sunIntensity: 1.2,
    ambientSky: rgb(0.28, 0.29, 0.44),
    ambientGround: rgb(0.38, 0.26, 0.16),
    ambientIntensity: 0.55,
    fogDensity: 0.0016,
    ridgeHaze: 0.55,
  },
];

// ------------------------------------------------------------------ weather

interface Weather {
  name: string;
  cloudCoverage: number;
  cloudOpacity: number;
  cloudScale: number;
  cloudSpeed: number;
  /** Pulls the whole sky toward its own luminance — the overcast flattener. */
  desaturate: number;
  /** Multiplies the horizon haze lift; the dusty-afternoon knob. */
  hazeLift: number;
  /** Scales the sun disc and the direct light. */
  sunScale: number;
  /** Scales fog density. */
  fogScale: number;
  /** Extra haze on the distant ridges. */
  ridgeHaze: number;
  /** Warm dust pushed into the horizon and the ambient fill. */
  dustTint: RGB;
  dustAmount: number;
  ambientScale: number;
}

const WEATHER: Record<string, Weather> = {
  clear: {
    name: 'clear',
    cloudCoverage: 0.42,
    cloudOpacity: 0.95,
    cloudScale: 3.0,
    cloudSpeed: 1.0,
    desaturate: 0,
    hazeLift: 1,
    sunScale: 1,
    fogScale: 1,
    ridgeHaze: 0,
    dustTint: rgb(1, 1, 1),
    dustAmount: 0,
    ambientScale: 1,
  },
  overcast: {
    name: 'overcast',
    cloudCoverage: 0.93,
    cloudOpacity: 1.0,
    cloudScale: 2.2,
    cloudSpeed: 1.5,
    desaturate: 0.78,
    hazeLift: 2.4,
    sunScale: 0.28,
    fogScale: 1.9,
    ridgeHaze: 0.3,
    dustTint: rgb(0.92, 0.94, 1.0),
    dustAmount: 0.25,
    ambientScale: 1.5,
  },
  dusty: {
    name: 'dusty',
    cloudCoverage: 0.3,
    cloudOpacity: 0.8,
    cloudScale: 3.4,
    cloudSpeed: 0.7,
    desaturate: 0.3,
    hazeLift: 2.8,
    sunScale: 0.88,
    fogScale: 2.3,
    ridgeHaze: 0.26,
    dustTint: rgb(1.0, 0.78, 0.5),
    dustAmount: 0.55,
    ambientScale: 1.15,
  },
};

export interface SkyPreset {
  label: string;
  t: number;
  weather: keyof typeof WEATHER | string;
}

export const SKY_PRESETS: SkyPreset[] = [
  { label: 'Clear Midday', t: 0.5, weather: 'clear' },
  { label: 'Golden Hour', t: 0.89, weather: 'clear' },
  { label: 'Overcast', t: 0.46, weather: 'overcast' },
  { label: 'Dusty Afternoon', t: 0.72, weather: 'dusty' },
];

/** Everything downstream of the sky needs to stay in step with it. */
export interface SkyState {
  label: string;
  timeOfDay: number;
  weather: string;
  sunColor: THREE.Color;
  sunIntensity: number;
  ambientSkyColor: THREE.Color;
  ambientGroundColor: THREE.Color;
  ambientIntensity: number;
  fogDensity: number;
  fogHeightFalloff: number;
  fogStart: number;
}

export interface SkyOptions {
  /** Dome radius. Small is fine: the dome is depth-test-free and camera-locked. */
  radius?: number;
  mountains?: boolean;
  clouds?: boolean;
  timeOfDay?: number;
  weather?: string;
  seed?: number;
  /**
   * Fired whenever the time of day changes. Wire this to the pipeline so fog
   * density tracks the sky:
   *   `new Sky(scene, pipe.atmosphere, { onChange: s => pipe.setOption('fogDensity', s.fogDensity) })`
   */
  onChange?: (state: SkyState) => void;
}

// -------------------------------------------------------------- dome shader

const DOME_VERT = /* glsl */ `
varying vec3 vWorldDir;
void main() {
  vec4 world = modelMatrix * vec4(position, 1.0);
  vWorldDir = world.xyz - cameraPosition;
  gl_Position = projectionMatrix * viewMatrix * world;
}
`;

const DOME_FRAG = /* glsl */ `
precision highp float;
varying vec3 vWorldDir;

${ATMOSPHERE_UNIFORMS_GLSL}
${ATMOSPHERE_FUNCTIONS_GLSL}
${NOISE_GLSL}

uniform float uTime;
uniform vec3 uCloudLit;
uniform vec3 uCloudShadow;
uniform float uCloudCoverage;
uniform float uCloudOpacity;
uniform float uCloudScale;
uniform vec2 uCloudWind;

/**
 * Clouds live on a flat plane one unit above the eye and are read through the
 * view direction, so they converge toward the horizon exactly the way a real
 * cloud deck does. No ray marching, no geometry, correct perspective.
 */
vec2 cloudPlane(vec3 dir) {
  return dir.xz / max(dir.y, 0.012);
}

float cloudDensity(vec2 p, float detailAmount) {
  // A slow domain warp is what turns fbm's cauliflower into cumulus billows.
  vec2 w = vec2(vnoise(p * 0.31 + 11.3), vnoise(p * 0.31 + 41.7)) - 0.5;
  float coarse = fbm4(p * 0.30 + w * 0.75);
  float detail = fbm6(p * 0.68 + w * 1.05);
  return mix(coarse, detail, detailAmount);
}

vec4 cloudLayer(vec3 dir) {
  if (dir.y <= 0.014 || uCloudOpacity <= 0.001) return vec4(0.0);

  vec2 base = cloudPlane(dir) * uCloudScale;
  // Toward the horizon the projection compresses hundreds of texels into one
  // pixel; drop the high octaves there or the deck fizzes into noise.
  float detailAmount = 1.0 - smoothstep(7.0, 30.0, length(base));
  vec2 p = base + uCloudWind * uTime;

  float d = cloudDensity(p, detailAmount);
  float cov = mix(0.80, 0.30, uCloudCoverage);
  float edge = mix(0.24, 0.09, uCloudCoverage);
  float a = smoothstep(cov, cov + edge, d);
  // Dissolve into the haze band rather than terminating on a hard line.
  a *= smoothstep(0.014, 0.115, dir.y);
  a *= uCloudOpacity;
  if (a <= 0.002) return vec4(0.0);

  // One extra density tap displaced toward the sun gives believable self
  // shadowing for the price of a second fbm.
  vec2 sunOff = normalize(uSunDirection.xz + vec2(1e-4, 1e-4)) * 0.62;
  float ds = cloudDensity(p - sunOff, detailAmount);
  float lit = clamp((d - ds) * 2.7 + 0.58, 0.0, 1.0);
  lit = mix(lit, 1.0, 0.16);

  vec3 c = mix(uCloudShadow, uCloudLit, lit);
  float sd = max(dot(dir, uSunDirection), 0.0);
  c += uSunGlowColor * pow(sd, 7.0) * 0.42 * (0.35 + 0.65 * lit);

  // Clouds breathe the same air as everything else: near the horizon they are
  // seen through the full depth of the atmosphere and must go to sky colour.
  float haze = smoothstep(0.42, 0.02, dir.y);
  c = mix(c, atmosphereBase(dir), haze * 0.88);

  return vec4(c, a);
}

void main() {
  vec3 dir = normalize(vWorldDir);
  vec3 col = atmosphereBase(dir) + atmosphereSunDisc(dir);

#ifdef USE_CLOUDS
  vec4 cl = cloudLayer(dir);
  col = mix(col, cl.rgb, cl.a);
#endif

  gl_FragColor = vec4(col, 1.0);
}
`;

// ---------------------------------------------------------- ridge shader

const RIDGE_VERT = /* glsl */ `
attribute float aH;
attribute float aLayer;
varying float vH;
varying float vLayer;
varying vec3 vWorldDir;
void main() {
  vec4 world = modelMatrix * vec4(position, 1.0);
  vWorldDir = world.xyz - cameraPosition;
  vH = aH;
  vLayer = aLayer;
  gl_Position = projectionMatrix * viewMatrix * world;
}
`;

const RIDGE_FRAG = /* glsl */ `
precision highp float;
varying float vH;
varying float vLayer;
varying vec3 vWorldDir;

${ATMOSPHERE_UNIFORMS_GLSL}
${ATMOSPHERE_FUNCTIONS_GLSL}

uniform vec3 uRidgeNear;
uniform vec3 uRidgeFar;
uniform float uRidgeHaze;

void main() {
  vec3 dir = normalize(vWorldDir);
  vec3 sky = atmosphereBase(dir);

  vec3 rock = mix(uRidgeNear, uRidgeFar, vLayer);
  // Ridge lines catch light; valleys and the bases stay in shade.
  rock *= 0.66 + 0.5 * vH;

  // Cheap directional warmth: the flank facing the sun lifts and picks up the
  // sun's own tint, which is what sells these as lit rock rather than decals.
  vec2 dh = normalize(dir.xz + vec2(1e-5));
  vec2 sh = normalize(uSunDirection.xz + vec2(1e-5));
  float s = dot(dh, sh);
  rock *= 1.0 + 0.24 * s;
  rock += uSunGlowColor * max(s, 0.0) * 0.05 * (0.3 + 0.7 * vH);

  // Haze pools in the low ground: the base of a distant range is nearly gone
  // while the summit still reads. Farther layers start further dissolved.
  float presence = mix(0.30, 1.0, vH) * (1.0 - vLayer * 0.42);
  presence *= 1.0 - clamp(uRidgeHaze, 0.0, 0.95);

  gl_FragColor = vec4(mix(sky, rock, clamp(presence, 0.0, 1.0)), 1.0);
}
`;

// ------------------------------------------------------------------- class

const _tmpDir = new THREE.Vector3();

export class Sky {
  readonly object3d: THREE.Group;
  readonly atmosphere: AtmosphereUniforms;
  readonly sunDirection: THREE.Vector3;
  readonly fogColor = new THREE.Color();
  readonly state: SkyState;

  private readonly scene: THREE.Scene;
  private readonly dome: THREE.Mesh;
  private readonly domeMat: THREE.ShaderMaterial;
  private ridges: THREE.Mesh | null = null;
  private ridgeMat: THREE.ShaderMaterial | null = null;
  private readonly onChange?: (s: SkyState) => void;

  private timeOfDay = 0.5;
  private weatherName = 'clear';
  private presetIndex = 0;
  private cloudTime = 0;
  private cloudSpeed = 1;
  private disposed = false;

  constructor(scene: THREE.Scene, atmosphere?: AtmosphereUniforms, opts: SkyOptions = {}) {
    this.scene = scene;
    this.atmosphere = atmosphere ?? createAtmosphereUniforms();
    this.sunDirection = this.atmosphere.uSunDirection.value;
    this.onChange = opts.onChange;

    this.object3d = new THREE.Group();
    this.object3d.name = 'Sky';
    this.object3d.frustumCulled = false;
    this.object3d.matrixAutoUpdate = true;

    const radius = opts.radius ?? 1;
    const useClouds = opts.clouds !== false;

    this.domeMat = new THREE.ShaderMaterial({
      vertexShader: DOME_VERT,
      fragmentShader: DOME_FRAG,
      uniforms: {
        uTime: { value: 0 },
        uCloudLit: { value: new THREE.Color(1, 1, 1) },
        uCloudShadow: { value: new THREE.Color(0.4, 0.43, 0.52) },
        uCloudCoverage: { value: 0.42 },
        uCloudOpacity: { value: 0.95 },
        uCloudScale: { value: 3.0 },
        uCloudWind: { value: new THREE.Vector2(0.055, 0.021) },
        ...this.atmosphere,
      },
      defines: useClouds ? { USE_CLOUDS: 1 } : {},
      side: THREE.BackSide,
      depthTest: false,
      depthWrite: false,
      fog: false,
    });

    this.dome = new THREE.Mesh(new THREE.SphereGeometry(radius, 48, 32), this.domeMat);
    this.dome.frustumCulled = false;
    this.dome.renderOrder = -1000;
    this.object3d.add(this.dome);

    if (opts.mountains !== false) this.buildRidges(opts.seed ?? 1337);

    scene.add(this.object3d);

    this.state = {
      label: SKY_PRESETS[0].label,
      timeOfDay: 0.5,
      weather: 'clear',
      sunColor: new THREE.Color(1, 1, 1),
      sunIntensity: 2,
      ambientSkyColor: new THREE.Color(0.24, 0.39, 0.66),
      ambientGroundColor: new THREE.Color(0.34, 0.29, 0.21),
      ambientIntensity: 0.55,
      fogDensity: 0.0022,
      fogHeightFalloff: 0.02,
      fogStart: 30,
    };

    this.weatherName = opts.weather ?? 'clear';
    this.applyTime(opts.timeOfDay ?? 0.5);
  }

  // ------------------------------------------------------------- geometry

  /**
   * Three concentric curtains of ridge line. Each is a triangle strip around
   * the camera: bottom edge sunk well below the horizon so terrain can never
   * expose a gap under it, top edge following a wrapping harmonic profile.
   */
  private buildRidges(seed: number): void {
    const LAYERS = [
      { radius: 2300, amp: 0.185, base: -0.34, floor: 0.24 },
      { radius: 1750, amp: 0.150, base: -0.32, floor: 0.18 },
      { radius: 1350, amp: 0.115, base: -0.30, floor: 0.12 },
    ];
    const SEG = 384;

    const positions: number[] = [];
    const hAttr: number[] = [];
    const layerAttr: number[] = [];
    const indices: number[] = [];

    let rnd = mulberry(seed);

    // Far layer first: with depth testing off, submission order is draw order,
    // so the nearest range must come last to sit in front.
    for (let li = 0; li < LAYERS.length; li++) {
      const L = LAYERS[li];
      const layerT = 1 - li / (LAYERS.length - 1); // 1 = farthest
      const harmonics = [
        { k: 3, amp: 0.5, ph: rnd() * Math.PI * 2 },
        { k: 5, amp: 0.34, ph: rnd() * Math.PI * 2 },
        { k: 8, amp: 0.22, ph: rnd() * Math.PI * 2 },
        { k: 13, amp: 0.15, ph: rnd() * Math.PI * 2 },
        { k: 21, amp: 0.095, ph: rnd() * Math.PI * 2 },
        { k: 34, amp: 0.055, ph: rnd() * Math.PI * 2 },
        { k: 55, amp: 0.03, ph: rnd() * Math.PI * 2 },
      ];
      const norm = harmonics.reduce((s, h) => s + h.amp, 0);
      const start = positions.length / 3;

      for (let i = 0; i <= SEG; i++) {
        const a = (i / SEG) * Math.PI * 2;
        let v = 0;
        for (const h of harmonics) v += h.amp * Math.sin(h.k * a + h.ph);
        let hN = 0.5 + 0.5 * (v / norm);
        // Sharpen: mountains are mostly low with occasional summits, not a
        // sine wave. The power curve is what gives peaks instead of hills.
        hN = Math.pow(Math.max(0, hN), 2.0);
        const height = L.radius * (L.floor * L.amp + L.amp * hN);
        const baseY = L.radius * L.base;
        const x = Math.sin(a) * L.radius;
        const z = Math.cos(a) * L.radius;

        positions.push(x, baseY, z);
        hAttr.push(0);
        layerAttr.push(layerT);
        positions.push(x, baseY + height, z);
        hAttr.push(1);
        layerAttr.push(layerT);
      }

      for (let i = 0; i < SEG; i++) {
        const a0 = start + i * 2;
        const b0 = a0 + 1;
        const a1 = a0 + 2;
        const b1 = a0 + 3;
        // Wound so the inward-facing side is front: we are inside the ring.
        indices.push(a0, a1, b0, b0, a1, b1);
      }
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geo.setAttribute('aH', new THREE.Float32BufferAttribute(hAttr, 1));
    geo.setAttribute('aLayer', new THREE.Float32BufferAttribute(layerAttr, 1));
    geo.setIndex(indices);

    this.ridgeMat = new THREE.ShaderMaterial({
      vertexShader: RIDGE_VERT,
      fragmentShader: RIDGE_FRAG,
      uniforms: {
        uRidgeNear: { value: new THREE.Color(0.10, 0.12, 0.13) },
        uRidgeFar: { value: new THREE.Color(0.10, 0.13, 0.20) },
        uRidgeHaze: { value: 0.4 },
        ...this.atmosphere,
      },
      side: THREE.DoubleSide,
      depthTest: false,
      depthWrite: false,
      fog: false,
    });

    this.ridges = new THREE.Mesh(geo, this.ridgeMat);
    this.ridges.frustumCulled = false;
    this.ridges.renderOrder = -999;
    this.object3d.add(this.ridges);
  }

  // ------------------------------------------------------------ time of day

  setTimeOfDay(t: number): void {
    this.applyTime(t);
  }

  setWeather(name: string): void {
    if (!WEATHER[name]) return;
    this.weatherName = name;
    this.applyTime(this.timeOfDay);
  }

  /** Applies the next authored preset and returns its label. */
  cycleTimeOfDay(): string {
    this.presetIndex = (this.presetIndex + 1) % SKY_PRESETS.length;
    return this.setPreset(this.presetIndex);
  }

  setPreset(indexOrLabel: number | string): string {
    const idx =
      typeof indexOrLabel === 'number'
        ? ((indexOrLabel % SKY_PRESETS.length) + SKY_PRESETS.length) % SKY_PRESETS.length
        : Math.max(
            0,
            SKY_PRESETS.findIndex((p) => p.label === indexOrLabel),
          );
    const preset = SKY_PRESETS[idx];
    this.presetIndex = idx;
    this.weatherName = preset.weather;
    this.applyTime(preset.t, preset.label);
    return preset.label;
  }

  get presetLabel(): string {
    return this.state.label;
  }

  get timeOfDayValue(): number {
    return this.timeOfDay;
  }

  private applyTime(t: number, label?: string): void {
    this.timeOfDay = ((t % 1) + 1) % 1;
    const k = sampleDay(this.timeOfDay);
    const w = WEATHER[this.weatherName] ?? WEATHER.clear;
    const a = this.atmosphere;

    // --- sun direction -----------------------------------------------------
    const el = (k.sunElevation * Math.PI) / 180;
    const az = (k.sunAzimuth * Math.PI) / 180;
    const ce = Math.cos(el);
    this.sunDirection.set(Math.sin(az) * ce, Math.sin(el), Math.cos(az) * ce).normalize();

    // --- sky palette -------------------------------------------------------
    // Overcast flattens toward luminance; dust pushes a warm wash into the
    // pale band near the horizon where airborne grit actually accumulates.
    setCol(a.uZenithColor.value, mixDust(desat(k.zenith, w.desaturate * 0.85), w, 0.25));
    setCol(a.uHorizonColor.value, mixDust(desat(k.horizon, w.desaturate), w, 1.0));
    setCol(a.uGroundColor.value, mixDust(desat(k.ground, w.desaturate * 0.6), w, 0.8));
    setCol(a.uSunGlowColor.value, scale(desat(k.sunGlow, w.desaturate), w.sunScale * 0.55 + 0.45));
    setCol(a.uSunDiscColor.value, desat(k.sunDisc, w.desaturate));

    a.uSkyParams.value.set(
      k.sky[0],
      k.sky[1],
      k.sky[2] * (0.4 + 0.6 * w.sunScale),
      k.sky[3] * w.hazeLift + (w.hazeLift - 1) * 0.055,
    );
    a.uSunDiscParams.value.set(
      k.disc[0],
      k.disc[1] * w.sunScale,
      k.disc[2],
      k.disc[3] * w.sunScale,
    );

    // --- clouds ------------------------------------------------------------
    const du = this.domeMat.uniforms;
    setCol(du.uCloudLit.value as THREE.Color, desat(k.cloudLit, w.desaturate * 0.7));
    setCol(du.uCloudShadow.value as THREE.Color, desat(k.cloudShadow, w.desaturate * 0.5));
    du.uCloudCoverage.value = w.cloudCoverage;
    du.uCloudOpacity.value = w.cloudOpacity;
    du.uCloudScale.value = w.cloudScale;
    this.cloudSpeed = w.cloudSpeed;

    // --- ridges ------------------------------------------------------------
    if (this.ridgeMat) {
      const ru = this.ridgeMat.uniforms;
      // Distant rock is not brown, it is whatever colour the air between you
      // and it happens to be — so tint it from the horizon, not from a palette.
      const h = a.uHorizonColor.value;
      const z = a.uZenithColor.value;
      (ru.uRidgeNear.value as THREE.Color).setRGB(
        h.r * 0.20 + z.r * 0.32 + 0.020,
        h.g * 0.20 + z.g * 0.32 + 0.026,
        h.b * 0.20 + z.b * 0.32 + 0.034,
      );
      (ru.uRidgeFar.value as THREE.Color).setRGB(
        h.r * 0.34 + z.r * 0.42 + 0.030,
        h.g * 0.34 + z.g * 0.42 + 0.036,
        h.b * 0.34 + z.b * 0.42 + 0.052,
      );
      ru.uRidgeHaze.value = Math.min(0.92, k.ridgeHaze + w.ridgeHaze);
    }

    // --- downstream state ---------------------------------------------------
    const s = this.state;
    s.timeOfDay = this.timeOfDay;
    s.weather = w.name;
    s.label = label ?? `${describeTime(this.timeOfDay)}${w.name === 'clear' ? '' : ` / ${w.name}`}`;
    setCol(s.sunColor, desat(k.sunColor, w.desaturate * 0.5));
    s.sunIntensity = k.sunIntensity * w.sunScale;
    setCol(s.ambientSkyColor, mixDust(desat(k.ambientSky, w.desaturate), w, 0.5));
    setCol(s.ambientGroundColor, mixDust(desat(k.ambientGround, w.desaturate * 0.4), w, 0.7));
    s.ambientIntensity = k.ambientIntensity * w.ambientScale;
    s.fogDensity = k.fogDensity * w.fogScale;
    s.fogHeightFalloff = 0.02;
    s.fogStart = 30;

    // The fog colour is the sky evaluated at the horizon — the same function
    // the fog shader runs — so the two are equal by construction, not by luck.
    _tmpDir.set(this.sunDirection.x, 0.004, this.sunDirection.z).normalize();
    evalAtmosphereBase(a, _tmpDir, this.fogColor);

    this.onChange?.(s);
  }

  // ---------------------------------------------------------------- update

  update(dt: number, cameraPos: THREE.Vector3): void {
    if (this.disposed) return;
    this.cloudTime += dt * this.cloudSpeed;
    this.domeMat.uniforms.uTime.value = this.cloudTime;
    // Camera-locked: the dome and the ranges are at infinity, so they must not
    // parallax. Translating rather than scaling keeps the near plane happy.
    this.object3d.position.copy(cameraPos);
    this.object3d.updateMatrixWorld(true);
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.scene.remove(this.object3d);
    this.dome.geometry.dispose();
    this.domeMat.dispose();
    if (this.ridges) {
      this.ridges.geometry.dispose();
      this.ridgeMat?.dispose();
    }
  }
}

// ----------------------------------------------------------------- helpers

function mulberry(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function lerpRGB(a: RGB, b: RGB, t: number): RGB {
  return {
    r: a.r + (b.r - a.r) * t,
    g: a.g + (b.g - a.g) * t,
    b: a.b + (b.b - a.b) * t,
  };
}

function desat(c: RGB, amount: number): RGB {
  if (amount <= 0) return c;
  const l = c.r * 0.2126 + c.g * 0.7152 + c.b * 0.0722;
  // Overcast is not just grey, it is grey and slightly brighter than the
  // saturated original — flat white light bounced around inside the cloud.
  const grey = l * 1.06;
  return lerpRGB(c, rgb(grey, grey, grey), Math.min(1, amount));
}

function mixDust(c: RGB, w: Weather, weight: number): RGB {
  const amt = w.dustAmount * weight;
  if (amt <= 0) return c;
  const l = c.r * 0.2126 + c.g * 0.7152 + c.b * 0.0722;
  const dust = rgb(w.dustTint.r * l * 1.15, w.dustTint.g * l * 1.15, w.dustTint.b * l * 1.15);
  return lerpRGB(c, dust, Math.min(1, amt));
}

function scale(c: RGB, k: number): RGB {
  return rgb(c.r * k, c.g * k, c.b * k);
}

function setCol(target: THREE.Color, c: RGB): void {
  target.setRGB(c.r, c.g, c.b, THREE.LinearSRGBColorSpace);
}

/** Piecewise-linear walk around the day curve, wrapping at midnight. */
function sampleDay(t: number): DayKey {
  const n = DAY.length;
  let i0 = 0;
  for (let i = 0; i < n; i++) if (DAY[i].t <= t) i0 = i;
  let i1 = (i0 + 1) % n;
  const t0 = DAY[i0].t;
  let t1 = DAY[i1].t;
  if (i1 === 0) t1 = DAY[0].t + 1;
  let tt = t;
  if (t < t0) tt = t + 1;
  const f = smootherstep((tt - t0) / Math.max(1e-6, t1 - t0));

  const a = DAY[i0];
  const b = DAY[i1];
  // Azimuth is interpolated the long way round on purpose: the sun crosses the
  // sky from east to west, it does not take the short arc back through north.
  let azA = a.sunAzimuth;
  let azB = b.sunAzimuth;
  if (azB < azA) azB += 360;
  return {
    t,
    sunElevation: lerp(a.sunElevation, b.sunElevation, f),
    sunAzimuth: lerp(azA, azB, f),
    zenith: lerpRGB(a.zenith, b.zenith, f),
    horizon: lerpRGB(a.horizon, b.horizon, f),
    ground: lerpRGB(a.ground, b.ground, f),
    sunGlow: lerpRGB(a.sunGlow, b.sunGlow, f),
    sunDisc: lerpRGB(a.sunDisc, b.sunDisc, f),
    sky: [
      lerp(a.sky[0], b.sky[0], f),
      lerp(a.sky[1], b.sky[1], f),
      lerp(a.sky[2], b.sky[2], f),
      lerp(a.sky[3], b.sky[3], f),
    ],
    disc: [
      lerp(a.disc[0], b.disc[0], f),
      lerp(a.disc[1], b.disc[1], f),
      lerp(a.disc[2], b.disc[2], f),
      lerp(a.disc[3], b.disc[3], f),
    ],
    cloudLit: lerpRGB(a.cloudLit, b.cloudLit, f),
    cloudShadow: lerpRGB(a.cloudShadow, b.cloudShadow, f),
    sunColor: lerpRGB(a.sunColor, b.sunColor, f),
    sunIntensity: lerp(a.sunIntensity, b.sunIntensity, f),
    ambientSky: lerpRGB(a.ambientSky, b.ambientSky, f),
    ambientGround: lerpRGB(a.ambientGround, b.ambientGround, f),
    ambientIntensity: lerp(a.ambientIntensity, b.ambientIntensity, f),
    fogDensity: lerp(a.fogDensity, b.fogDensity, f),
    ridgeHaze: lerp(a.ridgeHaze, b.ridgeHaze, f),
  };
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function smootherstep(x: number): number {
  const t = Math.max(0, Math.min(1, x));
  return t * t * t * (t * (t * 6 - 15) + 10);
}

function describeTime(t: number): string {
  if (t < 0.08 || t > 0.96) return 'Dawn';
  if (t < 0.3) return 'Morning';
  if (t < 0.62) return 'Midday';
  if (t < 0.82) return 'Afternoon';
  return 'Golden Hour';
}

/**
 * CPU mirror of `atmosphereBase` from atmosphere.ts, used for the fog colour
 * that Lighting and any UI read. Kept in lockstep with the GLSL above it.
 */
function evalAtmosphereBase(a: AtmosphereUniforms, dir: THREE.Vector3, out: THREE.Color): void {
  const p = a.uSkyParams.value;
  const zen = a.uZenithColor.value;
  const hor = a.uHorizonColor.value;
  const grd = a.uGroundColor.value;
  const glow = a.uSunGlowColor.value;
  const sun = a.uSunDirection.value;

  const t = Math.pow(Math.max(0, Math.min(1, 1 - Math.max(dir.y, 0))), p.x);
  let r = zen.r + (hor.r - zen.r) * t;
  let g = zen.g + (hor.g - zen.g) * t;
  let b = zen.b + (hor.b - zen.b) * t;

  const sd = dir.dot(sun);
  const tight = Math.pow(Math.max(sd, 0), p.y) * p.z;
  const broad = Math.pow(Math.max(sd, 0) * 0.5 + 0.5, 3) * t * 0.5;
  r += glow.r * (tight + broad) + hor.r * p.w * t;
  g += glow.g * (tight + broad) + hor.g * p.w * t;
  b += glow.b * (tight + broad) + hor.b * p.w * t;

  const below = smoothstepDown(0, -0.09, dir.y);
  r += (grd.r - r) * below;
  g += (grd.g - g) * below;
  b += (grd.b - b) * below;

  out.setRGB(r, g, b, THREE.LinearSRGBColorSpace);
}

function smoothstepDown(e0: number, e1: number, x: number): number {
  const t = Math.max(0, Math.min(1, (x - e0) / (e1 - e0 || 1e-6)));
  return t * t * (3 - 2 * t);
}
