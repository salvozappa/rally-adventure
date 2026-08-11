/**
 * World dressing: everything that turns the heightfield into a *place*.
 *
 * ── What this file is responsible for ──────────────────────────────────────
 *   1. Ecological placement — deciding where trees, rocks, scrub and trail
 *      furniture belong, from slope, altitude, ground material, moisture and
 *      the authored route mask.
 *   2. Rendering it fast — instancing, LOD, billboard impostors baked from the
 *      real models, per-instance frustum and distance culling, wind in the
 *      vertex shader.
 *   3. Colliding with it — Rapier fixed colliders, created and destroyed in a
 *      small radius around the player so a forest costs a handful of shapes
 *      instead of tens of thousands.
 *
 * ── Placement ──────────────────────────────────────────────────────────────
 * Random scatter reads as confetti. Real landscapes have communities: conifers
 * band by altitude and thin above a treeline, broadleaf follows water, scree
 * collects under steep faces, nothing grows past the angle of repose. So one
 * jittered candidate grid sweeps the whole map, each point is evaluated once
 * against the terrain, and the layers *compete* for it with weights derived
 * from that evaluation. A low-frequency noise mask on top gives groves and
 * clearings, and a shared occupancy grid enforces Poisson-ish spacing so
 * nothing interpenetrates.
 *
 * Because every layer draws from the same candidate stream, a spot can only be
 * won once — which is why the result has the mutually-exclusive look of real
 * ground cover rather than several independent scatters laid on top.
 *
 * ── Rendering ──────────────────────────────────────────────────────────────
 * Instance transforms are baked into a Float32Array at build time, packed in
 * spatial-grid order. A rebuild is then a cell-frustum test plus a run of
 * 16-float copies into the right InstancedMesh — no matrix maths at all. That
 * is cheap enough to do on a dirty check (camera moved 12 m or turned 10°)
 * rather than every frame, and the LOD fade band is sized to match, so an
 * instance is never in the wrong bucket by more than the band width.
 *
 * The far LOD is a billboard impostor rendered *from the real model* at boot,
 * 8 angles per variant into one atlas. That is what makes a whole forest
 * affordable: one draw call, two triangles per tree, and a silhouette that
 * still has branches in it.
 *
 * ── Wind ───────────────────────────────────────────────────────────────────
 * Every model carries an `aFlex` vertex attribute — 0 where it is planted, 1
 * where it may move. The vertex shader sways by `aFlex`, so trunks stay put
 * while canopies swing, and grass ripples in travelling waves. It costs a
 * dozen instructions and it is the single biggest "this world is alive" cue.
 */

import * as THREE from 'three';
import type RAPIER from '@dimforge/rapier3d-compat';
import { GROUP, interactionGroups, type PhysicsContext, type TerrainSampler } from '../types';
import {
  Rng,
  band,
  clamp01,
  fbm2,
  getScatterTextures,
  ridged2,
  smoothstep,
} from './scatterTextures';
import {
  buildBroadleafModels,
  buildBushModels,
  buildConiferModels,
  buildDeadwoodModels,
  buildGrassVariants,
  type ColliderShape,
  type ScatterModel,
} from './vegetation';
import { buildRockModels } from './rocks';
import { buildPropModels, type PropModel, type PropRole } from './props';
import { DESIGN_SIZE, FEATURES, washCenterX } from './heightfield';

/* -------------------------------------------------------------------------- */
/* Tuning                                                                     */
/* -------------------------------------------------------------------------- */

/** Camera movement, metres, before instance buffers are rebuilt. */
const REBUILD_MOVE = 8;
/** Camera rotation, radians, before instance buffers are rebuilt. */
const REBUILD_TURN = 0.17;
/**
 * Width of the LOD cross-fade band, metres. Must exceed REBUILD_MOVE or an
 * instance can sit in the wrong bucket with nothing hiding the pop.
 *
 * Kept as narrow as that rule allows. A mesh and the billboard baked from it
 * never have exactly the same outline, so wherever only one of the pair covers
 * a pixel the screen door has nothing to fill in against and the tree stipples.
 * A wide band spreads that over more distance and therefore over more trees at
 * once; a narrow one confines it to a thin shell you drive through.
 */
const FADE = 12;
/** Spatial grid cell for culling, metres. */
const CELL = 40;

const COLLIDER_RADIUS = 54;
const COLLIDER_MOVE = 7;

/**
 * Near-field ground cover. The radius is what the driver reads as "the ground
 * has stuff growing on it"; the cell is what decides whether it reads as cover
 * or as scattered weeds. At 1.7 m a tuft every 2.9 m² was sparse enough that on
 * a khaki hillside you simply could not see it.
 */
const GRASS_RADIUS = 46;
const GRASS_MOVE = 5;
const GRASS_CELL = 0.95;
/** Instances per grass variant. Four variants, so this is a quarter of the cap. */
const GRASS_PER_VARIANT = 4000;

/** Candidate spacing for the placement sweep, metres (design scale). */
const CANDIDATE_STEP = 3.5;

const IMPOSTOR_ANGLES = 8;
const IMPOSTOR_TILE_W = 96;
const IMPOSTOR_TILE_H = 128;

/* -------------------------------------------------------------------------- */
/* Public options                                                             */
/* -------------------------------------------------------------------------- */

export interface ScatterOptions {
  seed?: number;
  /** Global multiplier on every layer's acceptance probability. */
  density?: number;
  /**
   * Renderer used to bake the impostor atlas. Optional: without one a private
   * WebGL context is created for the bake and thrown away. Pass the game's
   * renderer when you have it — it saves a context and a few milliseconds.
   */
  renderer?: THREE.WebGLRenderer;
  /** Skip impostor baking entirely (headless tooling, tests). */
  impostors?: boolean;
}

/** Live statistics, for the preview HUD and for budgeting. */
export interface ScatterStats {
  readonly totalInstances: number;
  readonly drawnInstances: number;
  readonly drawnMeshes: number;
  readonly colliders: number;
  readonly grass: number;
  readonly buildMs: number;
  readonly impostorMs: number;
  readonly lastRebuildMs: number;
  readonly perLayer: { id: string; total: number; drawn: number; perLod: number[] }[];
}

/* -------------------------------------------------------------------------- */
/* Layer definitions                                                          */
/* -------------------------------------------------------------------------- */

type MaterialKind = 'canopy' | 'rock' | 'prop';

interface LayerSpec {
  id: string;
  material: MaterialKind;
  /** Upper distance of each LOD bucket. The last entry is the cull distance. */
  lodRanges: number[];
  impostor: boolean;
  castShadow: boolean;
  /** vec4(windSpeed, amplitude m, spatial frequency, gust) */
  wind: [number, number, number, number];
  /** Tint jitter range, multiplied into the albedo. */
  tintA: number;
  tintB: number;
  /** 0 = always upright, 1 = fully aligned to the ground normal. */
  align: number;
  scaleMin: number;
  scaleMax: number;
  /** Max random lean, radians. */
  lean: number;
  /** Fraction of the model height buried, so nothing floats on a slope. */
  sink: number;
  /** Registers in the occupancy grid (blocks later placements). */
  occupies: boolean;
  /** Tested against the occupancy grid (may be blocked). */
  respects: number;
}

/* -------------------------------------------------------------------------- */
/* Internals                                                                  */
/* -------------------------------------------------------------------------- */

interface Bucket {
  mesh: THREE.InstancedMesh;
  array: Float32Array;
  attr: THREE.InstancedBufferAttribute;
  n: number;
}

class Layer {
  spec: LayerSpec;
  models: ScatterModel[];
  n = 0;
  /** 16 floats per instance, packed in spatial-grid order. */
  matrices = new Float32Array(0);
  pos = new Float32Array(0);
  quat = new Float32Array(0);
  scale = new Float32Array(0);
  variant = new Uint8Array(0);
  /** CSR offsets into the packed arrays, one per grid cell (+1). */
  cellStart = new Int32Array(0);
  /** xyz + radius per grid cell. */
  cellSphere = new Float32Array(0);
  /** [lod][variant] */
  buckets: Bucket[][] = [];
  impostor: {
    mesh: THREE.InstancedMesh;
    matrix: Float32Array;
    imp: Float32Array;
    matrixAttr: THREE.InstancedBufferAttribute;
    impAttr: THREE.InstancedBufferAttribute;
    n: number;
  } | null = null;
  /** Impostor quad half-width and height per variant, unit scale. */
  impostorSizes: { w: number; h: number }[] | undefined;
  drawn = 0;
  hasColliders = false;

  constructor(spec: LayerSpec, models: ScatterModel[]) {
    this.spec = spec;
    this.models = models;
    this.hasColliders = models.some((m) => m.collider !== null);
  }
}

/** Accumulates placements before they are packed into typed arrays. */
class Collector {
  x: number[] = [];
  y: number[] = [];
  z: number[] = [];
  qx: number[] = [];
  qy: number[] = [];
  qz: number[] = [];
  qw: number[] = [];
  s: number[] = [];
  v: number[] = [];

  push(p: THREE.Vector3, q: THREE.Quaternion, scale: number, variant: number): void {
    this.x.push(p.x);
    this.y.push(p.y);
    this.z.push(p.z);
    this.qx.push(q.x);
    this.qy.push(q.y);
    this.qz.push(q.z);
    this.qw.push(q.w);
    this.s.push(scale);
    this.v.push(variant);
  }

  get length(): number {
    return this.x.length;
  }
}

const _v = new THREE.Vector3();
const _v2 = new THREE.Vector3();
const _q = new THREE.Quaternion();
const _q2 = new THREE.Quaternion();
const _m = new THREE.Matrix4();
const _scaleVec = new THREE.Vector3();
const _up = new THREE.Vector3(0, 1, 0);
const _frustum = new THREE.Frustum();
const _projScreen = new THREE.Matrix4();
const _sphere = new THREE.Sphere();

function nowMs(): number {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}

function markRange(attr: THREE.InstancedBufferAttribute, floats: number): void {
  const a = attr as unknown as {
    clearUpdateRanges?: () => void;
    addUpdateRange?: (start: number, count: number) => void;
  };
  if (typeof a.clearUpdateRanges === 'function' && typeof a.addUpdateRange === 'function') {
    a.clearUpdateRanges();
    if (floats > 0) a.addUpdateRange(0, floats);
  }
  attr.needsUpdate = true;
}

/* -------------------------------------------------------------------------- */
/* Shaders                                                                    */
/* -------------------------------------------------------------------------- */

/**
 * Shared GLSL. Two things live here that would otherwise need per-instance
 * buffers: the tint (hashed from world position, so groves share a hue and
 * neighbours differ) and the LOD fade (dithered from the instance's distance).
 * Both are free compared with uploading a colour per instance every rebuild.
 */
const GLSL_COMMON = /* glsl */ `
float sScatterHash(vec2 p) {
  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}
float sScatterNoise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(sScatterHash(i), sScatterHash(i + vec2(1.0, 0.0)), u.x),
             mix(sScatterHash(i + vec2(0.0, 1.0)), sScatterHash(i + vec2(1.0, 1.0)), u.x), u.y);
}
/* Interleaved gradient noise — a good dissolve pattern with no lookup table. */
float sScatterDither(vec2 p) {
  return fract(52.9829189 * fract(0.06711056 * p.x + 0.00583715 * p.y));
}

/**
 * Screen-door cross-fade between two LOD buckets.
 *
 * f is a vec2 of (fade-in, fade-out). An instance inside a transition band is
 * written to *both* buckets, and the two must claim disjoint halves of the
 * dither pattern or the object is full of holes. They complement only if the
 * bucket that is fading *in* takes the far half and the one fading *out* takes
 * the near half. Testing both against "dither < fade" — the obvious thing to
 * write — makes the two sets nested instead of complementary, and roughly 40%
 * of the object's pixels are then discarded by both buckets. That is the
 * screen-door ghosting on trees at a LOD boundary.
 */
bool sScatterDissolve(vec2 f, vec2 coord) {
  float d = sScatterDither(coord);
  if (f.x < 0.999 && d < 1.0 - f.x) return true;
  if (f.y < 0.999 && d >= f.y) return true;
  return false;
}
`;

const VERT_PARS = /* glsl */ `
attribute float aFlex;
uniform float uTime;
uniform vec2 uWindDir;
uniform vec4 uWind;
uniform vec4 uFade;
uniform vec3 uTintA;
uniform vec3 uTintB;
uniform float uShrink;
varying vec2 vScatterFade;
varying vec3 vScatterTint;
${GLSL_COMMON}
`;

const VERT_BODY = /* glsl */ `
{
  vec3 sOrigin = (modelMatrix * instanceMatrix * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
  float sD = distance(cameraPosition, sOrigin);
  float sFadeIn = smoothstep(uFade.x, uFade.y, sD);
  float sFadeOut = 1.0 - smoothstep(uFade.z, uFade.w, sD);

  /* The last bucket of a layer has nothing behind it to dissolve into, so its
     far edge is a fade to *nothing*. Dithering that is a band of speckle
     crawling across the ground at a fixed distance, which is far more visible
     than the popping it was meant to hide. Shrinking instead is free — the
     models are all built with their base at y = 0, so scaling toward the origin
     sinks the object into the ground as it goes — and it reads as the object
     simply being small rather than as a rendering artefact. */
  transformed *= mix(1.0, sFadeOut, uShrink);

  float sPhase = uTime * uWind.x + dot(sOrigin.xz, uWindDir) * uWind.z;
  float sSway = sin(sPhase) + 0.45 * sin(sPhase * 2.31 + 1.7);
  float sGust = 1.0 + uWind.w * (sScatterNoise(sOrigin.xz * 0.012 + uTime * 0.035) - 0.5) * 2.0;
  transformed.xz += uWindDir * (sSway * uWind.y * sGust * aFlex);

  vScatterFade = vec2(sFadeIn, mix(sFadeOut, 1.0, uShrink));

  float sGrove = sScatterNoise(sOrigin.xz * 0.012);
  float sJit = sScatterHash(floor(sOrigin.xz * 2.0) + 0.5);
  vScatterTint = mix(uTintA, uTintB, sGrove) * (0.86 + 0.3 * sJit);
}
`;

const FRAG_PARS = /* glsl */ `
varying vec2 vScatterFade;
varying vec3 vScatterTint;
${GLSL_COMMON}
`;

const FRAG_BODY = /* glsl */ `
if (sScatterDissolve(vScatterFade, gl_FragCoord.xy)) discard;
diffuseColor.rgb *= vScatterTint;
`;

interface ScatterUniforms {
  uTime: { value: number };
  uWindDir: { value: THREE.Vector2 };
  uWind: { value: THREE.Vector4 };
  uFade: { value: THREE.Vector4 };
  uTintA: { value: THREE.Color };
  uTintB: { value: THREE.Color };
  /** 1 on a bucket whose far edge is the cull distance; see VERT_BODY. */
  uShrink: { value: number };
}

/**
 * One shared function object for every scatter material, which is what lets
 * three reuse a single compiled program across all of them — its default
 * program cache key is `onBeforeCompile.toString()`.
 */
function scatterOnBeforeCompile(this: THREE.Material, shader: THREE.WebGLProgramParametersWithUniforms): void {
  const u = (this.userData as { scatterUniforms?: ScatterUniforms }).scatterUniforms;
  if (u) Object.assign(shader.uniforms, u);
  shader.vertexShader = shader.vertexShader
    .replace('#include <common>', `#include <common>\n${VERT_PARS}`)
    .replace('#include <begin_vertex>', `#include <begin_vertex>\n${VERT_BODY}`);
  shader.fragmentShader = shader.fragmentShader
    .replace('#include <common>', `#include <common>\n${FRAG_PARS}`)
    .replace('#include <alphatest_fragment>', `${FRAG_BODY}\n#include <alphatest_fragment>`);
}

const IMPOSTOR_VERT = /* glsl */ `
attribute vec3 aImp;
uniform float uTime;
uniform vec2 uWindDir;
uniform vec4 uWind;
uniform vec4 uFade;
uniform vec3 uTintA;
uniform vec3 uTintB;
uniform float uAngles;
uniform float uRows;
varying vec2 vUv;
varying vec2 vScatterFade;
varying vec3 vScatterTint;
#include <common>
#include <fog_pars_vertex>
${GLSL_COMMON}

void main() {
  vec3 centre = (modelMatrix * instanceMatrix * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
  float sc = length(instanceMatrix[0].xyz);
  vec2 half2 = vec2(aImp.y, aImp.z) * sc;

  /* Which of the baked angles faces the camera, measured in the instance's
     own frame so neighbouring trees do not all flip at the same moment. */
  vec3 fwd = normalize(vec3(instanceMatrix[2].x, 0.0, instanceMatrix[2].z) + vec3(1e-5, 0.0, 0.0));
  vec3 rgt = vec3(fwd.z, 0.0, -fwd.x);
  vec3 toCam = cameraPosition - centre;
  float ang = atan(dot(toCam, rgt), dot(toCam, fwd));
  float slot = floor(mod(ang / 6.2831853 * uAngles + uAngles + 0.5, uAngles));

  vec3 camRight = normalize(vec3(viewMatrix[0][0], viewMatrix[1][0], viewMatrix[2][0]));
  vec3 world = centre + camRight * (position.x * half2.x) + vec3(0.0, position.y * half2.y, 0.0);

  float sPhase = uTime * uWind.x + dot(centre.xz, uWindDir) * uWind.z;
  float sSway = sin(sPhase) + 0.45 * sin(sPhase * 2.31 + 1.7);
  world.xz += uWindDir * (sSway * uWind.y * position.y * position.y);

  vUv = vec2((slot + uv.x) / uAngles, (aImp.x + uv.y) / uRows);

  float sD = distance(cameraPosition, centre);
  vScatterFade = vec2(smoothstep(uFade.x, uFade.y, sD), 1.0 - smoothstep(uFade.z, uFade.w, sD));
  float sGrove = sScatterNoise(centre.xz * 0.012);
  float sJit = sScatterHash(floor(centre.xz * 2.0) + 0.5);
  vScatterTint = mix(uTintA, uTintB, sGrove) * (0.86 + 0.3 * sJit);

  vec4 mvPosition = viewMatrix * vec4(world, 1.0);
  gl_Position = projectionMatrix * mvPosition;
  #include <fog_vertex>
}
`;

const IMPOSTOR_FRAG = /* glsl */ `
uniform sampler2D uAtlas;
varying vec2 vUv;
varying vec2 vScatterFade;
varying vec3 vScatterTint;
#include <common>
#include <fog_pars_fragment>
${GLSL_COMMON}

void main() {
  vec4 texel = texture2D(uAtlas, vUv);
  if (texel.a < 0.45) discard;
  if (sScatterDissolve(vScatterFade, gl_FragCoord.xy)) discard;
  gl_FragColor = vec4(texel.rgb * vScatterTint, 1.0);
  #include <fog_fragment>
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}
`;

/* -------------------------------------------------------------------------- */
/* Scatter                                                                    */
/* -------------------------------------------------------------------------- */

export class Scatter {
  readonly object3d: THREE.Object3D;

  private readonly terrain: TerrainSampler;
  private readonly ctx: PhysicsContext;
  private readonly seed: number;
  private readonly density: number;

  private readonly layers: Layer[] = [];
  private readonly materials: THREE.Material[] = [];
  private readonly uniforms: ScatterUniforms[] = [];
  private readonly ownedGeometry: THREE.BufferGeometry[] = [];
  private readonly ownedTextures: THREE.Texture[] = [];

  /** World grid used for culling and collider queries. */
  private gridN = 0;
  private gridOrigin = 0;

  private camera: THREE.Camera | null = null;
  private readonly lastBuildPos = new THREE.Vector3(1e9, 1e9, 1e9);
  private readonly lastBuildQuat = new THREE.Quaternion(0, 0, 0, 1);
  private readonly lastColliderPos = new THREE.Vector3(1e9, 1e9, 1e9);
  private readonly lastGrassPos = new THREE.Vector3(1e9, 1e9, 1e9);
  private needsRebuild = true;

  private colliders = new Map<number, RAPIER.Collider>();

  private grassMeshes: THREE.InstancedMesh[] = [];
  private grassArrays: Float32Array[] = [];
  private grassCount = 0;

  private readonly startTime = nowMs();
  private stats: ScatterStats;
  private lastRebuildMs = 0;
  private buildMs = 0;
  private impostorMs = 0;

  /** Per-layer visibility toggle, for the preview. */
  private readonly enabled = new Map<string, boolean>();

  constructor(terrain: TerrainSampler, ctx: PhysicsContext, opts: ScatterOptions = {}) {
    this.terrain = terrain;
    this.ctx = ctx;
    this.seed = opts.seed ?? 0x5ca77e2;
    this.density = opts.density ?? 1;

    this.object3d = new THREE.Group();
    this.object3d.name = 'scatter';

    const t0 = nowMs();
    const tex = getScatterTextures();

    const models = {
      conifer: buildConiferModels(this.seed + 11),
      broadleaf: buildBroadleafModels(this.seed + 23),
      bush: buildBushModels(this.seed + 37),
      boulder: buildRockModels('boulder', this.seed + 41),
      stone: buildRockModels('stone', this.seed + 53),
      scree: buildRockModels('scree', this.seed + 67),
      deadwood: buildDeadwoodModels(this.seed + 71),
      prop: buildPropModels(this.seed + 83) as ScatterModel[],
    };
    for (const list of Object.values(models)) {
      for (const m of list) for (const g of m.lods) this.ownedGeometry.push(g);
    }

    const specs = makeLayerSpecs();
    for (const spec of specs) {
      const list = models[spec.id as keyof typeof models];
      this.layers.push(new Layer(spec, list));
      this.enabled.set(spec.id, true);
    }

    /* -- placement ------------------------------------------------------- */
    const collectors = this.placeAll(models.prop as PropModel[]);

    /* -- GPU resources --------------------------------------------------- */
    const impostorModels: { layer: Layer; models: ScatterModel[] }[] = [];
    for (let i = 0; i < this.layers.length; i++) {
      const layer = this.layers[i]!;
      this.pack(layer, collectors[i]!);
      this.buildBuckets(layer, tex);
      if (layer.spec.impostor) impostorModels.push({ layer, models: layer.models });
    }

    if (opts.impostors !== false && impostorModels.length > 0) {
      const ti = nowMs();
      this.bakeImpostors(impostorModels, opts.renderer);
      this.impostorMs = nowMs() - ti;
    }

    this.buildGrass(tex);

    /* A zero-cost sentinel whose only job is to hand us the camera three is
       actually rendering with, so per-instance frustum culling works through
       the `update(position)` API the game uses. */
    const probeGeo = new THREE.BufferGeometry();
    probeGeo.setAttribute('position', new THREE.Float32BufferAttribute([0, 0, 0], 3));
    const probeMat = new THREE.PointsMaterial({ size: 0, colorWrite: false, depthWrite: false, depthTest: false });
    const probe = new THREE.Points(probeGeo, probeMat);
    probe.frustumCulled = false;
    probe.renderOrder = -1000;
    probe.onBeforeRender = (_r, _s, cam) => {
      this.camera = cam;
    };
    this.ownedGeometry.push(probeGeo);
    this.materials.push(probeMat);
    this.object3d.add(probe);

    this.buildMs = nowMs() - t0;
    this.stats = this.collectStats();
  }

  /* ---------------------------------------------------------------------- */
  /* Frame                                                                  */
  /* ---------------------------------------------------------------------- */

  /**
   * Repositions everything that depends on where the camera is. Cheap by
   * design: the heavy work is gated on dirty checks, so the common frame does
   * nothing but advance the wind clock.
   */
  update(cameraPosition: THREE.Vector3, camera?: THREE.Camera): void {
    if (camera) this.camera = camera;

    const t = (nowMs() - this.startTime) * 0.001;
    for (const u of this.uniforms) u.uTime.value = t;

    const moved = this.lastBuildPos.distanceToSquared(cameraPosition);
    let turned = 0;
    if (this.camera) {
      const d = this.camera.quaternion.dot(this.lastBuildQuat);
      turned = 2 * Math.acos(Math.min(1, Math.abs(d)));
    }
    if (this.needsRebuild || moved > REBUILD_MOVE * REBUILD_MOVE || turned > REBUILD_TURN) {
      const t0 = nowMs();
      this.rebuildInstances(cameraPosition);
      this.lastRebuildMs = nowMs() - t0;
      this.lastBuildPos.copy(cameraPosition);
      if (this.camera) this.lastBuildQuat.copy(this.camera.quaternion);
      this.needsRebuild = false;
    }

    if (this.lastGrassPos.distanceToSquared(cameraPosition) > GRASS_MOVE * GRASS_MOVE) {
      this.rebuildGrass(cameraPosition);
      this.lastGrassPos.copy(cameraPosition);
    }

    if (this.lastColliderPos.distanceToSquared(cameraPosition) > COLLIDER_MOVE * COLLIDER_MOVE) {
      this.syncColliders(cameraPosition);
      this.lastColliderPos.copy(cameraPosition);
    }

    this.stats = this.collectStats();
  }

  getStats(): ScatterStats {
    return this.stats;
  }

  /** Preview hook: show or hide a whole layer by id (or `grass`). */
  setLayerEnabled(id: string, on: boolean): void {
    this.enabled.set(id, on);
    if (id === 'grass') {
      for (const m of this.grassMeshes) m.visible = on;
      return;
    }
    const layer = this.layers.find((l) => l.spec.id === id);
    if (!layer) return;
    for (const row of layer.buckets) for (const b of row) b.mesh.visible = on;
    if (layer.impostor) layer.impostor.mesh.visible = on;
  }

  /** Preview hook: force a LOD level, or -1 for automatic. */
  forceLod(level: number): void {
    this.forcedLod = level;
    this.needsRebuild = true;
  }

  private forcedLod = -1;

  dispose(): void {
    for (const c of this.colliders.values()) this.ctx.world.removeCollider(c, false);
    this.colliders.clear();
    for (const g of this.ownedGeometry) g.dispose();
    this.ownedGeometry.length = 0;
    for (const m of this.materials) m.dispose();
    this.materials.length = 0;
    for (const t of this.ownedTextures) t.dispose();
    this.ownedTextures.length = 0;
    for (const layer of this.layers) {
      layer.buckets.length = 0;
      layer.impostor = null;
    }
    this.grassMeshes.length = 0;
    this.object3d.clear();
  }

  /* ====================================================================== */
  /* Placement                                                              */
  /* ====================================================================== */

  /**
   * One sweep over the map. Every candidate is evaluated once and the layers
   * bid for it; the winner is whichever bid the roulette lands on, which makes
   * the layers mutually exclusive without any explicit conflict resolution.
   */
  private placeAll(props: PropModel[]): Collector[] {
    const half = this.terrain.halfSize;
    const worldScale = (half * 2) / DESIGN_SIZE;
    const rng = new Rng(this.seed ^ 0x9e3779b9);
    const collectors = this.layers.map(() => new Collector());

    this.gridN = Math.max(1, Math.ceil((half * 2) / CELL));
    this.gridOrigin = -half;

    /* -- occupancy grid (Poisson-disc-ish spacing) ----------------------- */
    const occCell = 8 * worldScale;
    const occN = Math.max(1, Math.ceil((half * 2) / occCell));
    const OCC_K = 8;
    const occCount = new Uint8Array(occN * occN);
    const occData = new Float32Array(occN * occN * OCC_K * 3);

    const occIndex = (x: number, z: number): number => {
      const cx = Math.min(occN - 1, Math.max(0, Math.floor((x + half) / occCell)));
      const cz = Math.min(occN - 1, Math.max(0, Math.floor((z + half) / occCell)));
      return cz * occN + cx;
    };
    const occFree = (x: number, z: number, r: number): boolean => {
      const cx = Math.floor((x + half) / occCell);
      const cz = Math.floor((z + half) / occCell);
      for (let dz = -1; dz <= 1; dz++) {
        const jz = cz + dz;
        if (jz < 0 || jz >= occN) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const jx = cx + dx;
          if (jx < 0 || jx >= occN) continue;
          const ci = jz * occN + jx;
          const n = occCount[ci]!;
          const base = ci * OCC_K * 3;
          for (let k = 0; k < n; k++) {
            const ox = occData[base + k * 3]!;
            const oz = occData[base + k * 3 + 1]!;
            const orr = occData[base + k * 3 + 2]!;
            const dxx = ox - x;
            const dzz = oz - z;
            const rr = orr + r;
            if (dxx * dxx + dzz * dzz < rr * rr) return false;
          }
        }
      }
      return true;
    };
    const occAdd = (x: number, z: number, r: number): void => {
      const ci = occIndex(x, z);
      const n = occCount[ci]!;
      if (n >= OCC_K) return;
      const base = ci * OCC_K * 3 + n * 3;
      occData[base] = x;
      occData[base + 1] = z;
      occData[base + 2] = r;
      occCount[ci] = n + 1;
    };

    /* -- coarse ecology fields ------------------------------------------ */
    const cg = 128;
    const forestField = new Float32Array(cg * cg);
    const moistField = new Float32Array(cg * cg);
    const rockField = new Float32Array(cg * cg);
    for (let j = 0; j < cg; j++) {
      const z = -half + ((j + 0.5) / cg) * half * 2;
      for (let i = 0; i < cg; i++) {
        const x = -half + ((i + 0.5) / cg) * half * 2;
        const k = j * cg + i;
        forestField[k] = fbm2(x / (150 * worldScale), z / (150 * worldScale), this.seed + 5, 4);
        moistField[k] = fbm2(x / (260 * worldScale), z / (260 * worldScale), this.seed + 9, 3);
        rockField[k] = ridged2(x / (95 * worldScale), z / (95 * worldScale), this.seed + 13, 3);
      }
    }
    const sampleField = (f: Float32Array, x: number, z: number): number => {
      const fx = Math.min(cg - 1.001, Math.max(0, ((x + half) / (half * 2)) * cg - 0.5));
      const fz = Math.min(cg - 1.001, Math.max(0, ((z + half) / (half * 2)) * cg - 0.5));
      const i0 = Math.floor(fx);
      const j0 = Math.floor(fz);
      const tx = fx - i0;
      const tz = fz - j0;
      const i1 = Math.min(cg - 1, i0 + 1);
      const j1 = Math.min(cg - 1, j0 + 1);
      const a = f[j0 * cg + i0]! * (1 - tx) + f[j0 * cg + i1]! * tx;
      const b = f[j1 * cg + i0]! * (1 - tx) + f[j1 * cg + i1]! * tx;
      return a * (1 - tz) + b * tz;
    };

    /* -- trail furniture first: it owns the ground it stands on ---------- */
    const propIdx = this.layers.findIndex((l) => l.spec.id === 'prop');
    if (propIdx >= 0) {
      this.placeProps(props, collectors[propIdx]!, rng, worldScale, occFree, occAdd);
    }

    /* -- the sweep -------------------------------------------------------- */
    const step = CANDIDATE_STEP * worldScale;
    const jitter = step * 0.45;
    const routeAt = this.terrain.routeAt?.bind(this.terrain);
    const edge = half - step * 2;

    const ids = this.layers.map((l) => l.spec.id);
    const iConifer = ids.indexOf('conifer');
    const iBroad = ids.indexOf('broadleaf');
    const iBush = ids.indexOf('bush');
    const iBoulder = ids.indexOf('boulder');
    const iStone = ids.indexOf('stone');
    const iScree = ids.indexOf('scree');
    const iDead = ids.indexOf('deadwood');

    const w: number[] = new Array(this.layers.length).fill(0);
    const dens = this.density;

    for (let gz = -edge; gz <= edge; gz += step) {
      for (let gx = -edge; gx <= edge; gx += step) {
        const x = gx + rng.range(-jitter, jitter);
        const z = gz + rng.range(-jitter, jitter);
        if (Math.abs(x) > edge || Math.abs(z) > edge) continue;

        const h = this.terrain.heightAt(x, z);
        // Slope from a one-cell central difference; cheaper than normalAt and
        // stable enough for a placement decision.
        const e = 1.6 * worldScale;
        const dhx = this.terrain.heightAt(x + e, z) - this.terrain.heightAt(x - e, z);
        const dhz = this.terrain.heightAt(x, z + e) - this.terrain.heightAt(x, z - e);
        const grad = Math.hypot(dhx, dhz) / (2 * e);
        const slopeDeg = Math.atan(grad) * (180 / Math.PI);
        if (slopeDeg > 52) continue;

        const alt = h / worldScale;
        const surf = this.terrain.surfaceAt(x, z);
        const route = routeAt ? routeAt(x, z) : 0;

        const forest = clamp01(sampleField(forestField, x, z) * 1.5 - 0.22);
        const moist = clamp01(sampleField(moistField, x, z) * 1.4 - 0.15);
        const rocky = clamp01(sampleField(rockField, x, z) * 1.35 - 0.1);

        // Distance to the wash centreline, in design metres — broadleaf and
        // scrub follow water, exactly as they do on the reference photos.
        const cx = washCenterX(z / worldScale) * worldScale;
        const washD = Math.abs(x - cx) / worldScale;
        const washTaper = 1 - smoothstep(340, 460, Math.abs(z) / worldScale);
        const riparian = (1 - smoothstep(22, 110, washD)) * washTaper;

        const flat = 1 - smoothstep(24, 37, slopeDeg);
        const steep = smoothstep(26, 44, slopeDeg);
        const grove = smoothstep(0.3, 0.72, forest);
        const open = 1 - grove;

        const soilConifer = surf === 'rock' ? 0.14 : surf === 'sand' ? 0.04 : surf === 'gravel' ? 0.5 : 1;
        const soilGreen = surf === 'rock' ? 0.1 : surf === 'sand' ? 0.15 : surf === 'gravel' ? 0.45 : 1;
        const soilDry = surf === 'sand' || surf === 'gravel' ? 1.4 : surf === 'rock' ? 0.7 : 0.75;

        // Route thinning. Not a hard clear: a handful of trunks are left close
        // enough to the line to be genuinely dangerous, which is the point.
        const clearBig = 1 - smoothstep(0.22, 0.62, route) * 0.94;
        const clearSmall = 1 - smoothstep(0.5, 0.92, route) * 0.8;

        /* The mix here is the art direction, in numbers.
         *
         * Screamer 4x4 is open country: rolling olive grassland with individual
         * rounded trees standing alone or in loose threes and fours, and a lot
         * of empty ground between them. It is emphatically not a closed forest.
         * So broadleaf outbids conifer nearly everywhere, conifer survives only
         * as a ridgeline accent on the higher, damper ground, and every weight
         * is low enough that most candidates win nothing at all — the gaps are
         * the point, and they are also what keeps the frame affordable. */
        w.fill(0);
        if (iConifer >= 0) {
          w[iConifer] =
            0.1 * grove * band(alt, 78, 150, 28) * flat * soilConifer * (0.4 + 0.9 * moist) * clearBig;
        }
        if (iBroad >= 0) {
          // Weighted *toward* the grove mask rather than away from it. A low
          // uniform probability gives evenly spaced trees, which reads as an
          // orchard; the same number of trees biased by a slow noise field
          // gives loose groups with real clearings between them, which is what
          // the reference actually looks like.
          w[iBroad] =
            0.18 *
            band(alt, 10, 118, 30) *
            flat *
            soilGreen *
            (0.3 + 1.15 * grove) *
            (0.6 + 0.7 * riparian) *
            (0.45 + 0.75 * moist) *
            clearBig;
        }
        if (iBush >= 0) {
          w[iBush] =
            0.26 *
            band(alt, 8, 156, 34) *
            (1 - smoothstep(32, 46, slopeDeg)) *
            (0.3 + 0.85 * open) *
            (soilDry * 0.55 + 0.55 * moist) *
            clearSmall;
        }
        // Rock belongs in the valleys and on the broken ground, not strewn over
        // the grassland — the `rocky` ridge field and the slope term do the
        // sorting, so both weights lean hard on them and start low.
        if (iBoulder >= 0) {
          w[iBoulder] = 0.11 * (0.1 + 1.7 * rocky) * (0.15 + steep) * (surf === 'rock' || surf === 'gravel' ? 1.7 : 0.35) * clearBig;
        }
        if (iStone >= 0) {
          w[iStone] = 0.3 * (0.1 + 1.35 * rocky) * (0.2 + steep) * (surf === 'rock' || surf === 'gravel' ? 1.5 : 0.4) * clearSmall;
        }
        if (iScree >= 0) {
          w[iScree] = 1.1 * smoothstep(26, 46, slopeDeg) * (0.35 + 1.1 * rocky) * (surf === 'rock' ? 1.8 : 0.6);
        }
        if (iDead >= 0) {
          w[iDead] = 0.07 * grove * flat * (0.4 + 0.9 * moist) * clearBig;
        }

        let acc = 0;
        const r = rng.next();
        let chosen = -1;
        for (let li = 0; li < w.length; li++) {
          acc += w[li]! * dens;
          if (r < acc) {
            chosen = li;
            break;
          }
        }
        if (chosen < 0) continue;

        const layer = this.layers[chosen]!;
        const spec = layer.spec;
        const vi = rng.int(layer.models.length);
        const model = layer.models[vi]!;
        const s = rng.range(spec.scaleMin, spec.scaleMax) * (0.85 + 0.3 * forest);
        const radius = model.spacing * s;

        if (spec.respects > 0 && !occFree(x, z, radius * spec.respects)) continue;
        if (spec.occupies) occAdd(x, z, radius);

        this.emit(collectors[chosen]!, spec, model, x, z, s, vi, rng);
      }
    }

    return collectors;
  }

  /** Writes one placement, including ground alignment, lean and burial. */
  private emit(
    into: Collector,
    spec: LayerSpec,
    model: ScatterModel,
    x: number,
    z: number,
    s: number,
    variant: number,
    rng: Rng,
  ): void {
    const y = this.terrain.heightAt(x, z) - model.height * s * spec.sink;
    _q.setFromAxisAngle(_up, rng.next() * Math.PI * 2);
    if (spec.align > 0) {
      this.terrain.normalAt(x, z, _v2);
      _v2.lerp(_up, 1 - spec.align).normalize();
      _q2.setFromUnitVectors(_up, _v2);
      _q.premultiply(_q2);
    }
    if (spec.lean > 0) {
      _v2.set(rng.gauss(), 0, rng.gauss()).normalize();
      _q2.setFromAxisAngle(_v2, rng.range(0, spec.lean));
      _q.premultiply(_q2);
    }
    _v.set(x, y, z);
    into.push(_v, _q, s, variant);
  }

  /**
   * Trail furniture along the authored routes.
   *
   * Walking the actual corridor polylines (rather than fishing for a high
   * route-mask value) is what makes markers march evenly down the road instead
   * of clumping wherever the mask happens to peak.
   */
  private placeProps(
    models: PropModel[],
    into: Collector,
    rng: Rng,
    worldScale: number,
    occFree: (x: number, z: number, r: number) => boolean,
    occAdd: (x: number, z: number, r: number) => void,
  ): void {
    const byRole = new Map<PropRole, number[]>();
    models.forEach((m, i) => {
      const list = byRole.get(m.role) ?? [];
      list.push(i);
      byRole.set(m.role, list);
    });

    const lines: { pts: [number, number][]; halfWidth: number }[] = [];

    // The dry wash — the main road through the level.
    {
      const pts: [number, number][] = [];
      for (let z = -455; z <= 455; z += 10) pts.push([washCenterX(z) * worldScale, z * worldScale]);
      lines.push({ pts, halfWidth: 17 * worldScale });
    }
    for (const c of FEATURES.corridors) {
      lines.push({
        pts: c.points.map((p) => [p[0] * worldScale, p[1] * worldScale] as [number, number]),
        halfWidth: c.halfWidth * worldScale,
      });
    }

    let side = 1;
    let sinceWreck = 0;

    for (const line of lines) {
      // Resample to a constant step so spacing is uniform along the whole run.
      const dense: [number, number][] = [];
      const stride = 3 * worldScale;
      for (let i = 0; i < line.pts.length - 1; i++) {
        const a = line.pts[i]!;
        const b = line.pts[i + 1]!;
        const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
        const n = Math.max(1, Math.round(len / stride));
        for (let k = 0; k < n; k++) {
          const t = k / n;
          dense.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]);
        }
      }
      dense.push(line.pts[line.pts.length - 1]!);

      let nextAt = 0;
      for (let i = 1; i < dense.length - 1; i++) {
        const p = dense[i]!;
        const prev = dense[i - 1]!;
        const next = dense[i + 1]!;
        const dx = next[0] - prev[0];
        const dz = next[1] - prev[1];
        const dl = Math.hypot(dx, dz) || 1;
        const tx = dx / dl;
        const tz = dz / dl;
        // Corner sharpness, used to bunch furniture where the driver needs it.
        const px = dense[Math.max(0, i - 6)]!;
        const nx = dense[Math.min(dense.length - 1, i + 6)]!;
        const curve = 1 - (tx * (nx[0] - px[0]) + tz * (nx[1] - px[1])) / (Math.hypot(nx[0] - px[0], nx[1] - px[1]) || 1);

        nextAt -= stride;
        if (nextAt > 0) continue;
        const spacing = (curve > 0.02 ? 5 : 8) * worldScale;
        nextAt = spacing;
        side = -side;
        sinceWreck += spacing;

        let role: PropRole = 'marker';
        const roll = rng.next();
        if (curve > 0.03 && roll < 0.34) role = rng.bool(0.5) ? 'tyres' : 'bale';
        else if (roll < 0.62) role = 'marker';
        else if (roll < 0.75) role = 'stripe';
        else if (roll < 0.81) role = 'drum';
        else if (roll < 0.87) role = 'fence';
        else if (roll < 0.91) role = 'barrier';
        else if (roll < 0.94) role = 'sign';
        else role = 'marker';

        if (sinceWreck > 420 * worldScale && rng.bool(0.35)) {
          role = 'wreck';
          sinceWreck = 0;
        }

        const pool = byRole.get(role) ?? byRole.get('marker')!;
        const vi = pool[rng.int(pool.length)]!;
        const model = models[vi]!;

        const margin = (role === 'wreck' ? rng.range(6, 13) : rng.range(1.2, 3.6)) * worldScale;
        const off = (line.halfWidth + margin) * side;
        const x = p[0] - tz * off;
        const z = p[1] + tx * off;
        if (Math.abs(x) > this.terrain.halfSize - 8 || Math.abs(z) > this.terrain.halfSize - 8) continue;

        const e = 1.6 * worldScale;
        const dhx = this.terrain.heightAt(x + e, z) - this.terrain.heightAt(x - e, z);
        const dhz = this.terrain.heightAt(x, z + e) - this.terrain.heightAt(x, z - e);
        if (Math.hypot(dhx, dhz) / (2 * e) > 0.62) continue;

        const s = rng.range(0.92, 1.1);
        if (!occFree(x, z, model.spacing * s)) continue;
        occAdd(x, z, model.spacing * s * 1.15);

        const y = this.terrain.heightAt(x, z) - model.height * s * 0.02;
        // Face along the road, so fences and barriers run with it.
        const yaw = Math.atan2(tx, tz) + (role === 'marker' || role === 'drum' ? rng.range(-0.6, 0.6) : rng.range(-0.12, 0.12));
        _q.setFromAxisAngle(_up, yaw);
        if (model.align > 0) {
          this.terrain.normalAt(x, z, _v2);
          _v2.lerp(_up, 1 - model.align).normalize();
          _q2.setFromUnitVectors(_up, _v2);
          _q.premultiply(_q2);
        }
        _v.set(x, y, z);
        into.push(_v, _q, s, vi);
      }
    }
  }

  /* ====================================================================== */
  /* Packing                                                                */
  /* ====================================================================== */

  /** Sorts placements into grid-cell order and bakes their instance matrices. */
  private pack(layer: Layer, c: Collector): void {
    const n = c.length;
    layer.n = n;
    const cells = this.gridN * this.gridN;
    layer.cellStart = new Int32Array(cells + 1);
    layer.cellSphere = new Float32Array(cells * 4);
    layer.matrices = new Float32Array(n * 16);
    layer.pos = new Float32Array(n * 3);
    layer.scale = new Float32Array(n);
    layer.variant = new Uint8Array(n);
    if (layer.hasColliders) layer.quat = new Float32Array(n * 4);
    if (n === 0) return;

    const cellOf = new Int32Array(n);
    const counts = new Int32Array(cells);
    for (let i = 0; i < n; i++) {
      const cx = Math.min(this.gridN - 1, Math.max(0, Math.floor((c.x[i]! - this.gridOrigin) / CELL)));
      const cz = Math.min(this.gridN - 1, Math.max(0, Math.floor((c.z[i]! - this.gridOrigin) / CELL)));
      const ci = cz * this.gridN + cx;
      cellOf[i] = ci;
      counts[ci] = counts[ci]! + 1;
    }
    let run = 0;
    for (let ci = 0; ci < cells; ci++) {
      layer.cellStart[ci] = run;
      run += counts[ci]!;
    }
    layer.cellStart[cells] = run;
    const cursor = Int32Array.from(layer.cellStart.subarray(0, cells));

    const cellMinY = new Float32Array(cells).fill(Infinity);
    const cellMaxY = new Float32Array(cells).fill(-Infinity);
    let maxModelR = 0;
    for (const m of layer.models) maxModelR = Math.max(maxModelR, m.radius * layer.spec.scaleMax);

    for (let i = 0; i < n; i++) {
      const ci = cellOf[i]!;
      const d = cursor[ci]!;
      cursor[ci] = d + 1;

      const s = c.s[i]!;
      _v.set(c.x[i]!, c.y[i]!, c.z[i]!);
      _q.set(c.qx[i]!, c.qy[i]!, c.qz[i]!, c.qw[i]!);
      _scaleVec.set(s, s, s);
      _m.compose(_v, _q, _scaleVec);
      _m.toArray(layer.matrices, d * 16);

      layer.pos[d * 3] = _v.x;
      layer.pos[d * 3 + 1] = _v.y;
      layer.pos[d * 3 + 2] = _v.z;
      layer.scale[d] = s;
      layer.variant[d] = c.v[i]!;
      if (layer.hasColliders) {
        layer.quat[d * 4] = _q.x;
        layer.quat[d * 4 + 1] = _q.y;
        layer.quat[d * 4 + 2] = _q.z;
        layer.quat[d * 4 + 3] = _q.w;
      }

      const top = _v.y + layer.models[c.v[i]!]!.height * s;
      if (_v.y < cellMinY[ci]!) cellMinY[ci] = _v.y;
      if (top > cellMaxY[ci]!) cellMaxY[ci] = top;
    }

    // One bounding sphere per cell: the whole culling scheme rests on it.
    for (let cz = 0; cz < this.gridN; cz++) {
      for (let cx = 0; cx < this.gridN; cx++) {
        const ci = cz * this.gridN + cx;
        if (layer.cellStart[ci + 1]! === layer.cellStart[ci]!) continue;
        const x0 = this.gridOrigin + cx * CELL;
        const z0 = this.gridOrigin + cz * CELL;
        const y0 = cellMinY[ci]!;
        const y1 = cellMaxY[ci]!;
        const cxw = x0 + CELL * 0.5;
        const czw = z0 + CELL * 0.5;
        const cyw = (y0 + y1) * 0.5;
        const r = Math.hypot(CELL * 0.5 + maxModelR, (y1 - y0) * 0.5 + maxModelR);
        layer.cellSphere[ci * 4] = cxw;
        layer.cellSphere[ci * 4 + 1] = cyw;
        layer.cellSphere[ci * 4 + 2] = czw;
        layer.cellSphere[ci * 4 + 3] = r;
      }
    }
  }

  /* ====================================================================== */
  /* GPU resources                                                          */
  /* ====================================================================== */

  private makeUniforms(spec: LayerSpec, fade: THREE.Vector4, shrink = false): ScatterUniforms {
    const u: ScatterUniforms = {
      uTime: { value: 0 },
      uWindDir: { value: new THREE.Vector2(0.82, 0.57) },
      uWind: { value: new THREE.Vector4(spec.wind[0], spec.wind[1], spec.wind[2], spec.wind[3]) },
      uFade: { value: fade },
      uTintA: { value: new THREE.Color().setScalar(spec.tintA) },
      uTintB: { value: new THREE.Color().setScalar(spec.tintB) },
      uShrink: { value: shrink ? 1 : 0 },
    };
    this.uniforms.push(u);
    return u;
  }

  private makeMaterial(
    spec: LayerSpec,
    map: THREE.Texture,
    fade: THREE.Vector4,
    shrink = false,
  ): THREE.MeshLambertMaterial {
    const mat = new THREE.MeshLambertMaterial({
      map,
      vertexColors: true,
      side: THREE.FrontSide,
    });
    mat.userData.scatterUniforms = this.makeUniforms(spec, fade, shrink);
    mat.onBeforeCompile = scatterOnBeforeCompile;
    this.materials.push(mat);
    return mat;
  }

  private buildBuckets(layer: Layer, tex: ReturnType<typeof getScatterTextures>): void {
    const spec = layer.spec;
    const map = spec.material === 'rock' ? tex.rock : spec.material === 'prop' ? tex.props : tex.canopy;
    const solidLods = spec.impostor ? spec.lodRanges.length - 1 : spec.lodRanges.length;

    const perVariant = new Int32Array(layer.models.length);
    for (let i = 0; i < layer.n; i++) perVariant[layer.variant[i]!] = perVariant[layer.variant[i]!]! + 1;

    for (let lod = 0; lod < solidLods; lod++) {
      const inStart = lod === 0 ? -2 : spec.lodRanges[lod - 1]! - FADE;
      const inEnd = lod === 0 ? -1 : spec.lodRanges[lod - 1]!;
      const outStart = spec.lodRanges[lod]! - FADE;
      const outEnd = spec.lodRanges[lod]!;
      const fade = new THREE.Vector4(inStart, inEnd, outStart, outEnd);
      // Only the outermost bucket shrinks: every other one hands off to the
      // next LOD, which needs the complementary half of the dither pattern.
      const mat = this.makeMaterial(spec, map, fade, !spec.impostor && lod === solidLods - 1);
      const row: Bucket[] = [];
      for (let v = 0; v < layer.models.length; v++) {
        const model = layer.models[v]!;
        const geo = model.lods[Math.min(lod, model.lods.length - 1)]!;
        const cap = Math.max(1, perVariant[v]!);
        const mesh = new THREE.InstancedMesh(geo, mat, cap);
        mesh.name = `scatter.${spec.id}.lod${lod}.${model.name}`;
        mesh.count = 0;
        mesh.frustumCulled = false;
        mesh.castShadow = spec.castShadow && lod === 0;
        mesh.receiveShadow = lod === 0;
        mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
        this.object3d.add(mesh);
        row.push({
          mesh,
          array: mesh.instanceMatrix.array as Float32Array,
          attr: mesh.instanceMatrix,
          n: 0,
        });
      }
      layer.buckets.push(row);
    }
  }

  /**
   * Renders every impostor-using model from 8 directions into one atlas, then
   * hands each layer a single billboard InstancedMesh that reads it.
   *
   * This is the whole reason a forest is affordable. A flat cutout of a cone
   * would cost the same and look like a cardboard stand-up; a render of the
   * actual tree keeps branch structure, self-shading and silhouette, and at
   * 100 m nobody can tell it is two triangles.
   */
  private bakeImpostors(
    entries: { layer: Layer; models: ScatterModel[] }[],
    provided?: THREE.WebGLRenderer,
  ): void {
    let renderer = provided;
    let owned = false;
    if (!renderer) {
      try {
        renderer = new THREE.WebGLRenderer({ alpha: true, antialias: false, powerPreference: 'low-power' });
        owned = true;
      } catch {
        return; // No WebGL (headless tooling) — layers fall back to their solid LODs.
      }
    }

    const prevTarget = renderer.getRenderTarget();
    const prevAutoClear = renderer.autoClear;
    const prevClear = new THREE.Color();
    renderer.getClearColor(prevClear);
    const prevAlpha = renderer.getClearAlpha();
    const prevViewport = new THREE.Vector4();
    renderer.getViewport(prevViewport);

    const scene = new THREE.Scene();
    /* A high, soft key with a strong fill: the impostor's baked lighting cannot
     * follow the instance's yaw, so anything low and directional would read as
     * obviously wrong the moment the tree rotated.
     *
     * The absolute levels have to track `Lighting`'s midday rig (sun ~2.0,
     * hemisphere ~0.55), because the impostor fragment shader does no lighting
     * at all — it blits the baked texel. Bake brighter than the scene and the
     * far trees glow; bake darker and the LOD change is a visible dimming as
     * you drive toward the treeline. */
    const key = new THREE.DirectionalLight(0xfff2dd, 1.95);
    key.position.set(0.45, 1.5, 0.75);
    const fill = new THREE.HemisphereLight(0xc6dcff, 0x8a7a5e, 0.75);
    scene.add(key, fill);
    const holder = new THREE.Group();
    scene.add(holder);

    for (const entry of entries) {
      const rows = entry.models.length;
      const W = IMPOSTOR_TILE_W * IMPOSTOR_ANGLES;
      const H = IMPOSTOR_TILE_H * rows;
      const rt = new THREE.WebGLRenderTarget(W, H, {
        minFilter: THREE.LinearFilter,
        magFilter: THREE.LinearFilter,
        format: THREE.RGBAFormat,
        type: THREE.UnsignedByteType,
        depthBuffer: true,
      });
      rt.texture.colorSpace = THREE.SRGBColorSpace;

      // Viewport and scissor are driven from the *render target*, not the
      // renderer, so nothing here depends on the device pixel ratio of a
      // renderer we may have been handed.
      rt.scissorTest = false;
      rt.viewport.set(0, 0, W, H);
      rt.scissor.set(0, 0, W, H);
      renderer.setRenderTarget(rt);
      renderer.setClearColor(0x000000, 0);
      renderer.autoClear = false;
      renderer.clear(true, true, false);

      const sizes: { w: number; h: number }[] = [];
      const cam = new THREE.OrthographicCamera();

      for (let r = 0; r < rows; r++) {
        const model = entry.models[r]!;
        const mat = new THREE.MeshLambertMaterial({
          map: getScatterTextures().canopy,
          vertexColors: true,
        });
        const mesh = new THREE.Mesh(model.lods[0]!, mat);
        holder.clear();
        holder.add(mesh);

        const w = model.radius * 2 * 1.08;
        const h = model.height * 1.05;
        // The quad's local x spans exactly 1.0, so the stored width is the
        // full tile width, not a half-extent.
        sizes.push({ w, h });

        // Inset by 3 px so mip generation blurs into empty padding rather than
        // into the neighbouring tile.
        const pad = 3;
        for (let a = 0; a < IMPOSTOR_ANGLES; a++) {
          const ang = (a / IMPOSTOR_ANGLES) * Math.PI * 2;
          const d = Math.max(w, h) * 2 + 10;
          cam.left = -w * 0.5;
          cam.right = w * 0.5;
          cam.top = h * 0.5;
          cam.bottom = -h * 0.5;
          cam.near = 0.01;
          cam.far = d * 3;
          cam.position.set(Math.sin(ang) * d, h * 0.5, Math.cos(ang) * d);
          cam.up.set(0, 1, 0);
          cam.lookAt(0, h * 0.5, 0);
          cam.updateProjectionMatrix();

          const vx = a * IMPOSTOR_TILE_W + pad;
          const vy = r * IMPOSTOR_TILE_H + pad;
          const vw = IMPOSTOR_TILE_W - pad * 2;
          const vh = IMPOSTOR_TILE_H - pad * 2;
          rt.viewport.set(vx, vy, vw, vh);
          rt.scissor.set(vx, vy, vw, vh);
          rt.scissorTest = true;
          renderer.setRenderTarget(rt);
          renderer.render(scene, cam);
        }
        mat.dispose();
      }

      const pixels = new Uint8Array(W * H * 4);
      renderer.readRenderTargetPixels(rt, 0, 0, W, H, pixels);
      rt.dispose();
      dilateAlpha(pixels, W, H, 3);

      const atlas = new THREE.DataTexture(pixels, W, H, THREE.RGBAFormat, THREE.UnsignedByteType);
      atlas.colorSpace = THREE.SRGBColorSpace;
      atlas.minFilter = THREE.LinearMipmapLinearFilter;
      atlas.magFilter = THREE.LinearFilter;
      atlas.generateMipmaps = true;
      atlas.anisotropy = 4;
      atlas.needsUpdate = true;
      atlas.name = `scatter.impostor.${entry.layer.spec.id}`;
      this.ownedTextures.push(atlas);

      this.attachImpostor(entry.layer, atlas, sizes);
    }

    holder.clear();
    key.dispose();
    fill.dispose();

    renderer.setScissorTest(false);
    renderer.setRenderTarget(prevTarget);
    renderer.autoClear = prevAutoClear;
    renderer.setClearColor(prevClear, prevAlpha);
    renderer.setViewport(prevViewport);
    if (owned) renderer.dispose();
  }

  private attachImpostor(layer: Layer, atlas: THREE.Texture, sizes: { w: number; h: number }[]): void {
    const spec = layer.spec;
    const lod = spec.lodRanges.length - 1;
    const cap = Math.max(1, layer.n);

    const geo = new THREE.BufferGeometry();
    geo.setAttribute(
      'position',
      new THREE.Float32BufferAttribute([-0.5, 0, 0, 0.5, 0, 0, 0.5, 1, 0, -0.5, 1, 0], 3),
    );
    geo.setAttribute('uv', new THREE.Float32BufferAttribute([0, 0, 1, 0, 1, 1, 0, 1], 2));
    geo.setAttribute(
      'normal',
      new THREE.Float32BufferAttribute([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], 3),
    );
    geo.setIndex([0, 1, 2, 0, 2, 3]);
    const impArr = new Float32Array(cap * 3);
    const impAttr = new THREE.InstancedBufferAttribute(impArr, 3);
    impAttr.setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute('aImp', impAttr);
    this.ownedGeometry.push(geo);

    const fade = new THREE.Vector4(
      spec.lodRanges[lod - 1]! - FADE,
      spec.lodRanges[lod - 1]!,
      spec.lodRanges[lod]! - FADE * 2,
      spec.lodRanges[lod]!,
    );
    const u = this.makeUniforms(spec, fade);
    const mat = new THREE.ShaderMaterial({
      uniforms: THREE.UniformsUtils.merge([
        THREE.UniformsLib.fog,
        {
          uAtlas: { value: null },
          uAngles: { value: IMPOSTOR_ANGLES },
          uRows: { value: sizes.length },
        },
      ]),
      vertexShader: IMPOSTOR_VERT,
      fragmentShader: IMPOSTOR_FRAG,
      fog: true,
    });
    Object.assign(mat.uniforms, u);
    mat.uniforms.uAtlas = { value: atlas };
    this.materials.push(mat);

    const mesh = new THREE.InstancedMesh(geo, mat, cap);
    mesh.name = `scatter.${spec.id}.impostor`;
    mesh.count = 0;
    mesh.frustumCulled = false;
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.object3d.add(mesh);

    layer.impostor = {
      mesh,
      matrix: mesh.instanceMatrix.array as Float32Array,
      imp: impArr,
      matrixAttr: mesh.instanceMatrix,
      impAttr,
      n: 0,
    };
    // Cached so the rebuild loop never touches the model list.
    layer.impostorSizes = sizes;
  }

  /* ====================================================================== */
  /* Rebuild                                                                */
  /* ====================================================================== */

  private rebuildInstances(camPos: THREE.Vector3): void {
    const cam = this.camera;
    let useFrustum = false;
    if (cam) {
      cam.updateMatrixWorld();
      _projScreen.multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse);
      _frustum.setFromProjectionMatrix(_projScreen);
      useFrustum = true;
    }
    // Pad the cull sphere by the rebuild threshold so nothing pops in at the
    // screen edge between rebuilds.
    const pad = REBUILD_MOVE + 6;

    for (const layer of this.layers) {
      for (const row of layer.buckets) for (const b of row) b.n = 0;
      if (layer.impostor) layer.impostor.n = 0;
      layer.drawn = 0;
      if (layer.n === 0) continue;

      const spec = layer.spec;
      const ranges = spec.lodRanges;
      const cull = ranges[ranges.length - 1]!;
      const cull2 = cull * cull;
      const solidLods = layer.buckets.length;
      const sizes = layer.impostorSizes;

      const cx0 = Math.max(0, Math.floor((camPos.x - cull - this.gridOrigin) / CELL));
      const cx1 = Math.min(this.gridN - 1, Math.floor((camPos.x + cull - this.gridOrigin) / CELL));
      const cz0 = Math.max(0, Math.floor((camPos.z - cull - this.gridOrigin) / CELL));
      const cz1 = Math.min(this.gridN - 1, Math.floor((camPos.z + cull - this.gridOrigin) / CELL));

      for (let cz = cz0; cz <= cz1; cz++) {
        for (let cx = cx0; cx <= cx1; cx++) {
          const ci = cz * this.gridN + cx;
          const start = layer.cellStart[ci]!;
          const end = layer.cellStart[ci + 1]!;
          if (end === start) continue;

          const sr = layer.cellSphere[ci * 4 + 3]!;
          const sx = layer.cellSphere[ci * 4]!;
          const sy = layer.cellSphere[ci * 4 + 1]!;
          const sz = layer.cellSphere[ci * 4 + 2]!;
          const ddx = sx - camPos.x;
          const ddz = sz - camPos.z;
          if (ddx * ddx + ddz * ddz > (cull + sr) * (cull + sr)) continue;
          if (useFrustum) {
            _sphere.center.set(sx, sy, sz);
            _sphere.radius = sr + pad;
            if (!_frustum.intersectsSphere(_sphere)) continue;
          }

          for (let i = start; i < end; i++) {
            const px = layer.pos[i * 3]! - camPos.x;
            const py = layer.pos[i * 3 + 1]! - camPos.y;
            const pz = layer.pos[i * 3 + 2]! - camPos.z;
            const d2 = px * px + py * py + pz * pz;
            if (d2 > cull2) continue;
            const d = Math.sqrt(d2);
            const v = layer.variant[i]!;

            if (this.forcedLod >= 0) {
              const l = Math.min(this.forcedLod, ranges.length - 1);
              if (l < solidLods) this.writeSolid(layer, l, v, i);
              else this.writeImpostor(layer, sizes, v, i);
              layer.drawn++;
              continue;
            }

            for (let l = 0; l < ranges.length; l++) {
              const lo = l === 0 ? -1 : ranges[l - 1]! - FADE;
              if (d < lo || d >= ranges[l]!) continue;
              if (l < solidLods) this.writeSolid(layer, l, v, i);
              else this.writeImpostor(layer, sizes, v, i);
              layer.drawn++;
            }
          }
        }
      }

      for (const row of layer.buckets) {
        for (const b of row) {
          b.mesh.count = b.n;
          if (b.n > 0) markRange(b.attr, b.n * 16);
        }
      }
      if (layer.impostor) {
        const imp = layer.impostor;
        imp.mesh.count = imp.n;
        if (imp.n > 0) {
          markRange(imp.matrixAttr, imp.n * 16);
          markRange(imp.impAttr, imp.n * 3);
        }
      }
    }
  }

  private writeSolid(layer: Layer, lod: number, variant: number, i: number): void {
    const b = layer.buckets[lod]![variant]!;
    if (b.n >= b.mesh.instanceMatrix.count) return;
    const dst = b.array;
    const src = layer.matrices;
    const di = b.n * 16;
    const si = i * 16;
    for (let k = 0; k < 16; k++) dst[di + k] = src[si + k]!;
    b.n++;
  }

  private writeImpostor(layer: Layer, sizes: { w: number; h: number }[] | undefined, variant: number, i: number): void {
    const imp = layer.impostor;
    if (!imp || !sizes) return;
    if (imp.n >= imp.mesh.instanceMatrix.count) return;
    const di = imp.n * 16;
    const si = i * 16;
    for (let k = 0; k < 16; k++) imp.matrix[di + k] = layer.matrices[si + k]!;
    const s = sizes[variant]!;
    imp.imp[imp.n * 3] = variant;
    imp.imp[imp.n * 3 + 1] = s.w;
    imp.imp[imp.n * 3 + 2] = s.h;
    imp.n++;
  }

  /* ====================================================================== */
  /* Grass                                                                  */
  /* ====================================================================== */

  private buildGrass(tex: ReturnType<typeof getScatterTextures>): void {
    const geoms = buildGrassVariants(this.seed + 97);
    for (const g of geoms) this.ownedGeometry.push(g);

    const spec: LayerSpec = {
      id: 'grass',
      material: 'canopy',
      lodRanges: [GRASS_RADIUS],
      impostor: false,
      castShadow: false,
      wind: [2.1, 0.16, 0.09, 0.9],
      tintA: 0.72,
      tintB: 1.15,
      align: 1,
      scaleMin: 1,
      scaleMax: 1,
      lean: 0,
      sink: 0,
      occupies: false,
      respects: 0,
    };
    const fade = new THREE.Vector4(-2, -1, GRASS_RADIUS - 10, GRASS_RADIUS);
    const mat = new THREE.MeshLambertMaterial({
      map: tex.grass,
      vertexColors: true,
      // FrontSide, not DoubleSide: the tuft geometry already carries each card
      // wound both ways so that both faces keep the upward normal the shading
      // depends on. See `buildGrassVariants`.
      side: THREE.FrontSide,
      alphaTest: 0.5,
      transparent: false,
    });
    // Grass has one bucket and a hard radius, so it shrinks out at the edge
    // rather than dithering — a dissolving ring of speckle travelling with the
    // camera is about the most conspicuous artefact a ground layer can have.
    mat.userData.scatterUniforms = this.makeUniforms(spec, fade, true);
    mat.onBeforeCompile = scatterOnBeforeCompile;
    this.materials.push(mat);

    for (const g of geoms) {
      const mesh = new THREE.InstancedMesh(g, mat, GRASS_PER_VARIANT);
      mesh.name = 'scatter.grass';
      mesh.count = 0;
      mesh.frustumCulled = false;
      mesh.castShadow = false;
      // Grass casts nothing — a tuft's own shadow is smaller than a shadow-map
      // texel — but it must *receive*, or every tuft under a tree stays lit and
      // the cover reads as a layer floating above the ground instead of part of
      // it. That mismatch is especially loud because the tufts are shaded with
      // upward normals, so nothing else darkens them.
      mesh.receiveShadow = true;
      mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      this.object3d.add(mesh);
      this.grassMeshes.push(mesh);
      this.grassArrays.push(mesh.instanceMatrix.array as Float32Array);
    }
  }

  /**
   * Grass exists only within a radius of the camera and is regenerated as it
   * moves, so the field appears infinite for the cost of a few thousand tufts.
   *
   * Positions are hashed from the world cell, never from the camera, so tufts
   * stay exactly where they were the last time you drove past instead of
   * swimming around underneath you.
   */
  private rebuildGrass(camPos: THREE.Vector3): void {
    const counts = new Array(this.grassMeshes.length).fill(0) as number[];
    const caps = this.grassMeshes.map((m) => m.instanceMatrix.count);
    const routeAt = this.terrain.routeAt?.bind(this.terrain);
    const half = this.terrain.halfSize;

    const cam = this.camera;
    let useFrustum = false;
    if (cam) {
      cam.updateMatrixWorld();
      _projScreen.multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse);
      _frustum.setFromProjectionMatrix(_projScreen);
      useFrustum = true;
    }

    const c0x = Math.floor((camPos.x - GRASS_RADIUS) / GRASS_CELL);
    const c1x = Math.floor((camPos.x + GRASS_RADIUS) / GRASS_CELL);
    const c0z = Math.floor((camPos.z - GRASS_RADIUS) / GRASS_CELL);
    const c1z = Math.floor((camPos.z + GRASS_RADIUS) / GRASS_CELL);
    const r2 = GRASS_RADIUS * GRASS_RADIUS;
    const nv = this.grassMeshes.length;

    for (let cz = c0z; cz <= c1z; cz++) {
      for (let cx = c0x; cx <= c1x; cx++) {
        // Deterministic hash of the cell — the same tuft every time.
        let hsh = (Math.imul(cx, 0x27d4eb2d) ^ Math.imul(cz, 0x165667b1)) >>> 0;
        hsh = (hsh ^ (hsh >>> 15)) >>> 0;
        const jx = ((hsh & 0xffff) / 65535 - 0.5) * GRASS_CELL;
        const jz = (((hsh >>> 16) & 0xffff) / 65535 - 0.5) * GRASS_CELL;
        const x = (cx + 0.5) * GRASS_CELL + jx;
        const z = (cz + 0.5) * GRASS_CELL + jz;
        const ddx = x - camPos.x;
        const ddz = z - camPos.z;
        if (ddx * ddx + ddz * ddz > r2) continue;
        if (Math.abs(x) > half - 2 || Math.abs(z) > half - 2) continue;

        const surf = this.terrain.surfaceAt(x, z);
        let prob: number;
        let dry = false;
        // Dirt and gravel carry real cover too. Grassland this dry does not stop
        // at a material boundary — thinning out across one is what sells the
        // boundary, whereas cutting to nothing draws it.
        if (surf === 'grass') prob = 0.96;
        else if (surf === 'dirt') { prob = 0.55; dry = true; }
        else if (surf === 'gravel') { prob = 0.3; dry = true; }
        else if (surf === 'sand') { prob = 0.13; dry = true; }
        else prob = 0.06;

        const h2 = (Math.imul(hsh ^ 0x9e3779b9, 0x85ebca6b) >>> 8) / 0xffffff;
        if (h2 > prob) continue;
        if (routeAt && routeAt(x, z) > 0.72) continue;

        const y = this.terrain.heightAt(x, z);
        if (useFrustum) {
          _sphere.center.set(x, y + 0.4, z);
          _sphere.radius = 1.0;
          if (!_frustum.intersectsSphere(_sphere)) continue;
        }

        // Dry ground gets the pale variants (index >= 2), damp ground the lush.
        const h3 = ((hsh >>> 7) & 0xff) / 255;
        const vi = dry ? 2 + ((hsh >>> 3) % Math.max(1, nv - 2)) : (hsh >>> 3) % Math.min(2, nv);
        const slot = counts[vi]!;
        if (slot >= caps[vi]!) continue;

        const s = 0.8 + h3 * 0.5;
        this.terrain.normalAt(x, z, _v2);
        _v2.lerp(_up, 0.35).normalize();
        _q.setFromUnitVectors(_up, _v2);
        _q2.setFromAxisAngle(_up, h3 * Math.PI * 2);
        _q.multiply(_q2);
        _v.set(x, y - 0.04, z);
        _scaleVec.set(s, s * (0.85 + h3 * 0.4), s);
        _m.compose(_v, _q, _scaleVec);
        _m.toArray(this.grassArrays[vi]!, slot * 16);
        counts[vi] = slot + 1;
      }
    }

    this.grassCount = 0;
    for (let i = 0; i < nv; i++) {
      const mesh = this.grassMeshes[i]!;
      mesh.count = counts[i]!;
      this.grassCount += counts[i]!;
      if (counts[i]! > 0) markRange(mesh.instanceMatrix, counts[i]! * 16);
    }
  }

  /* ====================================================================== */
  /* Colliders                                                              */
  /* ====================================================================== */

  /**
   * Static colliders exist only near the player. Ten thousand permanent fixed
   * shapes would cost Rapier a broad-phase update it does not need to do; a
   * few dozen, created and destroyed as you drive, cost nothing measurable and
   * feel identical.
   */
  private syncColliders(camPos: THREE.Vector3): void {
    const { world, rapier } = this.ctx;
    const wanted = new Set<number>();
    const r2 = COLLIDER_RADIUS * COLLIDER_RADIUS;
    const groups = interactionGroups(GROUP.PROP, GROUP.CHASSIS | GROUP.WHEEL_RAY);

    for (let li = 0; li < this.layers.length; li++) {
      const layer = this.layers[li]!;
      if (!layer.hasColliders || layer.n === 0) continue;

      const cx0 = Math.max(0, Math.floor((camPos.x - COLLIDER_RADIUS - this.gridOrigin) / CELL));
      const cx1 = Math.min(this.gridN - 1, Math.floor((camPos.x + COLLIDER_RADIUS - this.gridOrigin) / CELL));
      const cz0 = Math.max(0, Math.floor((camPos.z - COLLIDER_RADIUS - this.gridOrigin) / CELL));
      const cz1 = Math.min(this.gridN - 1, Math.floor((camPos.z + COLLIDER_RADIUS - this.gridOrigin) / CELL));

      for (let cz = cz0; cz <= cz1; cz++) {
        for (let cx = cx0; cx <= cx1; cx++) {
          const ci = cz * this.gridN + cx;
          const start = layer.cellStart[ci]!;
          const end = layer.cellStart[ci + 1]!;
          for (let i = start; i < end; i++) {
            const model = layer.models[layer.variant[i]!]!;
            const shape = model.collider;
            if (!shape) continue;
            const dx = layer.pos[i * 3]! - camPos.x;
            const dz = layer.pos[i * 3 + 2]! - camPos.z;
            if (dx * dx + dz * dz > r2) continue;

            const key = li * 0x100000 + i;
            wanted.add(key);
            if (this.colliders.has(key)) continue;

            const s = layer.scale[i]!;
            const qx = layer.quat[i * 4]!;
            const qy = layer.quat[i * 4 + 1]!;
            const qz = layer.quat[i * 4 + 2]!;
            const qw = layer.quat[i * 4 + 3]!;
            const px = layer.pos[i * 3]!;
            const py = layer.pos[i * 3 + 1]!;
            const pz = layer.pos[i * 3 + 2]!;

            const desc = makeColliderDesc(rapier, shape, s);
            if (!desc) continue;
            // The shape's local offset has to be rotated with the instance or
            // a leaning tree's trunk ends up somewhere it visibly is not.
            const off = shapeOffset(shape) * s;
            _v.set(0, off, 0).applyQuaternion(_q.set(qx, qy, qz, qw));
            desc
              .setTranslation(px + _v.x, py + _v.y, pz + _v.z)
              .setRotation({ x: qx, y: qy, z: qz, w: qw })
              .setFriction(0.85)
              .setRestitution(0.05)
              .setCollisionGroups(groups);
            this.colliders.set(key, world.createCollider(desc));
          }
        }
      }
    }

    for (const [key, col] of this.colliders) {
      if (wanted.has(key)) continue;
      world.removeCollider(col, false);
      this.colliders.delete(key);
    }
  }

  /* ====================================================================== */
  /* Stats                                                                  */
  /* ====================================================================== */

  private collectStats(): ScatterStats {
    let total = 0;
    let drawn = 0;
    let meshes = 0;
    const perLayer: { id: string; total: number; drawn: number; perLod: number[] }[] = [];
    for (const layer of this.layers) {
      total += layer.n;
      drawn += layer.drawn;
      const perLod: number[] = [];
      for (const row of layer.buckets) {
        let c = 0;
        for (const b of row) {
          c += b.n;
          if (b.n > 0 && b.mesh.visible) meshes++;
        }
        perLod.push(c);
      }
      if (layer.impostor) {
        perLod.push(layer.impostor.n);
        if (layer.impostor.n > 0 && layer.impostor.mesh.visible) meshes++;
      }
      perLayer.push({ id: layer.spec.id, total: layer.n, drawn: layer.drawn, perLod });
    }
    for (const m of this.grassMeshes) if (m.count > 0 && m.visible) meshes++;
    return {
      totalInstances: total,
      drawnInstances: drawn + this.grassCount,
      drawnMeshes: meshes,
      colliders: this.colliders.size,
      grass: this.grassCount,
      buildMs: this.buildMs,
      impostorMs: this.impostorMs,
      lastRebuildMs: this.lastRebuildMs,
      perLayer,
    };
  }
}

/* -------------------------------------------------------------------------- */
/* Layer table                                                                */
/* -------------------------------------------------------------------------- */

function makeLayerSpecs(): LayerSpec[] {
  return [
    {
      id: 'conifer',
      material: 'canopy',
      lodRanges: [46, 125, 620],
      impostor: true,
      castShadow: true,
      wind: [0.85, 0.16, 0.006, 0.7],
      tintA: 0.78,
      tintB: 1.16,
      align: 0.12,
      scaleMin: 0.62,
      scaleMax: 1.35,
      lean: 0.06,
      sink: 0.012,
      occupies: true,
      // A tree claims two and a bit crown radii of ground. At 1 they could sit
      // crown-to-crown, which is a hedge; the gaps are the art direction here.
      respects: 2.2,
    },
    {
      id: 'broadleaf',
      material: 'canopy',
      lodRanges: [42, 112, 540],
      impostor: true,
      castShadow: true,
      wind: [1.05, 0.22, 0.006, 0.8],
      tintA: 0.8,
      tintB: 1.2,
      align: 0.15,
      scaleMin: 0.6,
      scaleMax: 1.3,
      lean: 0.09,
      sink: 0.012,
      occupies: true,
      respects: 2.2,
    },
    {
      id: 'bush',
      material: 'canopy',
      lodRanges: [34, 125],
      impostor: false,
      castShadow: false,
      wind: [1.5, 0.09, 0.02, 0.9],
      tintA: 0.74,
      tintB: 1.22,
      align: 0.55,
      scaleMin: 0.65,
      scaleMax: 1.15,
      lean: 0.12,
      sink: 0.06,
      occupies: false,
      respects: 0.45,
    },
    {
      id: 'boulder',
      material: 'rock',
      lodRanges: [70, 260],
      impostor: false,
      castShadow: true,
      wind: [0, 0, 0, 0],
      tintA: 0.82,
      tintB: 1.14,
      align: 0.4,
      scaleMin: 0.55,
      scaleMax: 1.6,
      lean: 0.3,
      sink: 0.16,
      occupies: true,
      respects: 1,
    },
    {
      id: 'stone',
      material: 'rock',
      lodRanges: [85],
      impostor: false,
      castShadow: false,
      wind: [0, 0, 0, 0],
      tintA: 0.84,
      tintB: 1.16,
      align: 0.75,
      scaleMin: 0.55,
      scaleMax: 1.7,
      lean: 0.5,
      sink: 0.22,
      occupies: false,
      respects: 0.3,
    },
    {
      id: 'scree',
      material: 'rock',
      lodRanges: [55],
      impostor: false,
      castShadow: false,
      wind: [0, 0, 0, 0],
      tintA: 0.86,
      tintB: 1.12,
      align: 0.9,
      scaleMin: 0.6,
      scaleMax: 1.8,
      lean: 0.7,
      sink: 0.3,
      occupies: false,
      respects: 0,
    },
    {
      id: 'deadwood',
      material: 'canopy',
      lodRanges: [110],
      impostor: false,
      castShadow: false,
      wind: [0, 0, 0, 0],
      tintA: 0.8,
      tintB: 1.15,
      align: 0.7,
      scaleMin: 0.7,
      scaleMax: 1.35,
      lean: 0.12,
      sink: 0.05,
      occupies: true,
      respects: 1,
    },
    {
      id: 'prop',
      material: 'prop',
      lodRanges: [230],
      impostor: false,
      castShadow: true,
      wind: [0, 0, 0, 0],
      tintA: 0.88,
      tintB: 1.1,
      align: 0.7,
      scaleMin: 0.95,
      scaleMax: 1.05,
      lean: 0,
      sink: 0.01,
      occupies: true,
      respects: 1,
    },
  ];
}

/* -------------------------------------------------------------------------- */
/* Helpers                                                                    */
/* -------------------------------------------------------------------------- */

function shapeOffset(shape: ColliderShape): number {
  return shape.kind === 'hull' ? 0 : shape.y;
}

function makeColliderDesc(
  rapier: PhysicsContext['rapier'],
  shape: ColliderShape,
  s: number,
): RAPIER.ColliderDesc | null {
  switch (shape.kind) {
    case 'cylinder':
      return rapier.ColliderDesc.cylinder(shape.halfHeight * s, shape.radius * s);
    case 'ball':
      return rapier.ColliderDesc.ball(shape.radius * s);
    case 'cuboid':
      return rapier.ColliderDesc.cuboid(shape.hx * s, shape.hy * s, shape.hz * s);
    case 'hull': {
      const pts = new Float32Array(shape.points.length);
      for (let i = 0; i < pts.length; i++) pts[i] = shape.points[i]! * s;
      return rapier.ColliderDesc.convexHull(pts);
    }
    default:
      return null;
  }
}

/**
 * Flood colour outward into transparent texels.
 *
 * Without this, mip generation averages the tree's edge pixels against the
 * black transparent background and every distant trunk grows a dark halo. Two
 * or three passes is enough to cover the mip levels an impostor is ever seen
 * at.
 */
function dilateAlpha(px: Uint8Array, w: number, h: number, passes: number): void {
  const solid = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) solid[i] = px[i * 4 + 3]! > 8 ? 1 : 0;
  const next = new Uint8Array(solid);

  for (let p = 0; p < passes; p++) {
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        if (solid[i]) continue;
        let r = 0;
        let g = 0;
        let b = 0;
        let n = 0;
        for (let dy = -1; dy <= 1; dy++) {
          const jy = y + dy;
          if (jy < 0 || jy >= h) continue;
          for (let dx = -1; dx <= 1; dx++) {
            const jx = x + dx;
            if (jx < 0 || jx >= w) continue;
            const j = jy * w + jx;
            if (!solid[j]) continue;
            r += px[j * 4]!;
            g += px[j * 4 + 1]!;
            b += px[j * 4 + 2]!;
            n++;
          }
        }
        if (n === 0) continue;
        px[i * 4] = (r / n) | 0;
        px[i * 4 + 1] = (g / n) | 0;
        px[i * 4 + 2] = (b / n) | 0;
        next[i] = 1;
      }
    }
    solid.set(next);
  }
}
