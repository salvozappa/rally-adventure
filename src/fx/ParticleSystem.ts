/**
 * ParticleSystem.ts — one GPU-simulated instanced particle pool.
 *
 * The CPU only ever *spawns*: it writes a particle's initial state into a ring
 * buffer of per-instance attributes and never touches it again. The vertex
 * shader integrates position analytically from `age = uTime - spawnTime`, so a
 * pool of several thousand long-lived particles costs one draw call and a few
 * hundred bytes of upload per frame.
 *
 * Three integration modes cover everything the drive FX needs:
 *
 *   BILLOW    linear drag towards a terminal drift, plus a slow swirl. Dust and
 *             smoke: launched fast, stopped by air in half a second, then
 *             hanging and expanding. Closed form:
 *                 v(t) = v_inf + (v0 - v_inf) e^(-kt),  v_inf = a/k
 *                 x(t) = x0 + v_inf t + (v0 - v_inf)(1 - e^(-kt))/k
 *
 *   BALLISTIC parabola under gravity with a single analytic bounce off a
 *             horizontal plane at `groundY`. Clods and splash droplets.
 *
 *   STREAK    ballistic, but the quad is stretched along the screen-space
 *             velocity instead of being square. Sparks.
 *
 * Soft particles: if a scene depth texture is handed over with
 * `setDepthTexture()` the fragment shader fades where the quad approaches
 * geometry, which is the real fix. Without one it falls back to a vertical
 * fade over the last `softFade` metres above the particle's own ground plane —
 * cheaper, and it kills the one artefact that actually reads, namely the hard
 * line where a big dust quad slices into the terrain.
 */

import * as THREE from 'three';
import { getFxTextures, ATLAS_GRID } from './fxTextures';

/* ------------------------------------------------------------------ *
 * Public shapes
 * ------------------------------------------------------------------ */

export const MODE_BILLOW = 0;
export const MODE_BALLISTIC = 1;
export const MODE_STREAK = 2;

/**
 * Spawn parameters. Callers keep one of these around and mutate it — the
 * system copies out of it immediately, so nothing here is retained and the
 * emitter allocates nothing per particle.
 */
export interface ParticleDesc {
  /** Spawn position, world space. */
  x: number;
  y: number;
  z: number;
  /** Spawn velocity, world space, m/s. */
  vx: number;
  vy: number;
  vz: number;
  /** Linear colour 0..1 (already converted out of sRGB). */
  r: number;
  g: number;
  b: number;
  /** Atlas cell index; see fxTextures. */
  cell: number;
  /** Seconds. */
  life: number;
  /** Quad half-extent in metres at birth and at death. */
  size0: number;
  size1: number;
  /** Radians, and radians/second. */
  rot: number;
  rotSpeed: number;
  /** BILLOW: drag coefficient 1/s. Ignored otherwise. */
  drag: number;
  /**
   * BILLOW: vertical acceleration, m/s^2 (positive = buoyant).
   * BALLISTIC/STREAK: downward gravity magnitude, m/s^2 (positive).
   */
  gravity: number;
  /** Peak alpha, before the pool's global opacity scale. */
  opacity: number;
  /** Plane the BALLISTIC bounce happens on, and the vertical-fade reference. */
  groundY: number;
  mode: number;
}

export interface ParticleSystemOptions {
  /** Ring buffer size. Every slot costs 88 bytes of CPU + GPU memory. */
  capacity?: number;
  /** `THREE.NormalBlending` for dust, `THREE.AdditiveBlending` for sparks. */
  blending?: THREE.Blending;
  /** Swirl strength applied to BILLOW particles. */
  turbulence?: number;
  /** Fraction of lifetime spent fading in (BILLOW only). */
  fadeIn?: number;
  /** Metres above `groundY` over which the fallback soft fade ramps in. */
  softFade?: number;
  /** Depth difference in metres over which the depth-based soft fade ramps. */
  softDepth?: number;
  /** Particles closer than `.x` are invisible, fully visible past `.y`. */
  nearFade?: [number, number];
  /** BALLISTIC restitution and horizontal retention on the single bounce. */
  bounce?: number;
  friction?: number;
  /** Multiplier on the streak length for MODE_STREAK. */
  streak?: number;
  /** Render order; particles should draw after opaque scene geometry. */
  renderOrder?: number;
  name?: string;
}

/* ------------------------------------------------------------------ *
 * Shaders
 * ------------------------------------------------------------------ */

const VERT = /* glsl */ `
precision highp float;

attribute vec4 iCol;   // rgb, atlas cell
attribute vec4 iPos;   // xyz spawn position, w spawn time
attribute vec4 iVel;   // xyz spawn velocity, w lifetime
attribute vec4 iSize;  // size0, size1, rot0, rotSpeed
attribute vec4 iPhys;  // drag, gravity, opacity, groundY
attribute vec2 iMisc;  // seed 0..1, mode

uniform float uTime;
uniform float uSizeScale;
uniform float uOpacityScale;
uniform float uTurbulence;
uniform float uFadeIn;
uniform float uBounce;
uniform float uFriction;
uniform float uStreak;
uniform vec2  uNearFade;
uniform float uInvGrid;

varying vec2  vUv;
varying vec3  vCol;
varying float vAlpha;
varying float vHeight;   // metres above this particle's ground plane
varying float vViewZ;    // positive distance in front of the camera
varying vec4  vClip;

#include <fog_pars_vertex>

void main() {
  float life = iVel.w;
  float age  = uTime - iPos.w;

  if (life <= 0.0 || age < 0.0 || age >= life) {
    // Cull: collapse to a point outside the clip volume. Four identical
    // vertices produce no fragments and the primitive is trivially rejected.
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    vUv = vec2(0.0);
    vCol = vec3(0.0);
    vAlpha = 0.0;
    vHeight = 1.0;
    vViewZ = 1.0;
    vClip = vec4(0.0, 0.0, 1.0, 1.0);
    #ifdef USE_FOG
      vFogDepth = 0.0;
    #endif
    return;
  }

  float n    = age / life;                 // 0..1 normalised age
  float mode = iMisc.y;
  float seed = iMisc.x;

  vec3 p;
  vec3 vel;

  if (mode < 0.5) {
    // ---- BILLOW ------------------------------------------------------
    float k = max(iPhys.x, 0.05);
    vec3 vinf = vec3(0.0, iPhys.y, 0.0) / k;
    float e = exp(-k * age);
    p   = iPos.xyz + vinf * age + (iVel.xyz - vinf) * (1.0 - e) / k;
    vel = vinf + (iVel.xyz - vinf) * e;

    // Slow swirl, amplitude growing with age so young particles stay tight
    // to the tyre and old ones drift apart. Scaled by size so big lazy
    // puffs wander more than small ones.
    float s = seed * 6.2831853;
    float amp = uTurbulence * iSize.x * age;
    p += vec3(sin(age * 0.83 + s),
              sin(age * 0.61 + s * 1.7) * 0.55,
              cos(age * 1.07 + s * 2.3)) * amp;
  } else {
    // ---- BALLISTIC / STREAK -------------------------------------------
    float g = max(iPhys.y, 0.0);
    float h = iPos.y - iPhys.w;                     // height above the plane
    float disc = iVel.y * iVel.y + 2.0 * g * max(h, 0.0);
    float tHit = (g > 0.001 && disc > 0.0) ? (iVel.y + sqrt(disc)) / g : 1.0e9;

    if (mode > 1.5 || age < tHit) {
      // Before the bounce (streaks never bounce).
      p = iPos.xyz + iVel.xyz * age - vec3(0.0, 0.5 * g * age * age, 0.0);
      vel = iVel.xyz - vec3(0.0, g * age, 0.0);
    } else {
      vec3 hit = iPos.xyz + iVel.xyz * tHit - vec3(0.0, 0.5 * g * tHit * tHit, 0.0);
      hit.y = iPhys.w;
      float vyHit = iVel.y - g * tHit;
      // Vary the bounce per particle so a shower of clods doesn't move as
      // one body.
      float rest = uBounce * (0.55 + 0.9 * seed);
      vec3 v2 = vec3(iVel.x * uFriction, -vyHit * rest, iVel.z * uFriction);
      float t2 = age - tHit;
      p = hit + v2 * t2 - vec3(0.0, 0.5 * g * t2 * t2, 0.0);
      vel = v2 - vec3(0.0, g * t2, 0.0);
      // Second contact: settle on the surface rather than sinking through it.
      if (p.y < iPhys.w) { p.y = iPhys.w; vel = vec3(0.0); }
    }
  }

  // ---- size, rotation, alpha ------------------------------------------
  // Billows expand quickly then ease off; solids hold their size.
  float grow = (mode < 0.5) ? pow(n, 0.55) : n;
  float size = mix(iSize.x, iSize.y, grow) * uSizeScale;

  float fadeInWindow = (mode < 0.5) ? uFadeIn : 0.02;
  float fadeIn  = smoothstep(0.0, fadeInWindow, n);
  float fadeOut = 1.0 - smoothstep((mode < 0.5) ? 0.45 : 0.7, 1.0, n);
  float alpha = iPhys.z * uOpacityScale * fadeIn * fadeOut;

  float rot = iSize.z + iSize.w * age;

  // ---- billboard --------------------------------------------------------
  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  vec2 corner = position.xy;   // unit quad, -0.5 .. 0.5
  vec2 off;

  if (mode > 1.5) {
    // Stretch along the screen projection of the velocity.
    vec3 vv = (modelViewMatrix * vec4(vel, 0.0)).xyz;
    float len = length(vv.xy);
    vec2 dir = len > 1e-4 ? vv.xy / len : vec2(0.0, 1.0);
    vec2 perp = vec2(-dir.y, dir.x);
    float stretch = 1.0 + uStreak * min(length(vel) * 0.06, 4.0);
    off = dir * (corner.y * size * stretch) + perp * (corner.x * size);
  } else {
    float c = cos(rot), s = sin(rot);
    off = vec2(corner.x * c - corner.y * s, corner.x * s + corner.y * c) * size;
  }

  mv.xy += off;

  // viewMatrix's rotation is orthonormal, so v * M is transpose(M) * v and
  // gives us the world-space form of the view-space corner offset. We only
  // need its height, for the fallback soft fade.
  vec3 worldOff = vec3(off, 0.0) * mat3(viewMatrix);

  vViewZ = -mv.z;
  vHeight = (p.y + worldOff.y) - iPhys.w;

  // Fade out anything about to clip through the near plane — a dust puff
  // filling the frame because the chase camera swung into the plume is the
  // ugliest failure this system has.
  alpha *= smoothstep(uNearFade.x, uNearFade.y, vViewZ);

  // ---- atlas uv ---------------------------------------------------------
  float cell = iCol.w;
  float grid = 1.0 / uInvGrid;
  float cx = mod(cell, grid);
  float cy = floor(cell * uInvGrid);
  // 1.5% inset so coarse mip levels can't bleed a neighbouring cell in.
  vec2 local = (corner + 0.5) * 0.97 + 0.015;
  vUv = (local + vec2(cx, cy)) * uInvGrid;

  vCol = iCol.rgb;
  vAlpha = alpha;

  gl_Position = projectionMatrix * mv;
  vClip = gl_Position;

  #ifdef USE_FOG
    vec4 mvPosition = mv;
    #include <fog_vertex>
  #endif
}
`;

const FRAG = /* glsl */ `
precision highp float;

uniform sampler2D uMap;
uniform sampler2D uDepth;
uniform float uHasDepth;
uniform vec2  uDepthRange;   // camera near, far
uniform float uSoftDepth;    // metres of depth-based soft fade
uniform float uSoftFade;     // metres of vertical fallback fade

varying vec2  vUv;
varying vec3  vCol;
varying float vAlpha;
varying float vHeight;
varying float vViewZ;
varying vec4  vClip;

#include <fog_pars_fragment>

void main() {
  if (vAlpha <= 0.0) discard;

  vec4 tex = texture2D(uMap, vUv);
  float a = tex.a * vAlpha;

  // ---- soft particles ---------------------------------------------------
  if (uHasDepth > 0.5) {
    vec2 sc = (vClip.xy / vClip.w) * 0.5 + 0.5;
    float dz = texture2D(uDepth, sc).x;
    float near = uDepthRange.x, far = uDepthRange.y;
    // Window depth -> eye distance, perspective.
    float sceneZ = (2.0 * near * far) / (far + near - (dz * 2.0 - 1.0) * (far - near));
    a *= clamp((sceneZ - vViewZ) / uSoftDepth, 0.0, 1.0);
  } else {
    // No depth buffer: fade over the last metre above the plane the particle
    // was spawned on. Keeps the quad from drawing a hard chord across the
    // ground, which is the artefact people actually notice.
    a *= smoothstep(0.0, uSoftFade, vHeight + uSoftFade * 0.35);
  }

  if (a <= 0.002) discard;

  gl_FragColor = vec4(vCol * tex.rgb, a);

  #include <colorspace_fragment>

  #ifdef USE_FOG
    #ifdef FOG_EXP2
      float fogFactor = 1.0 - exp(-fogDensity * fogDensity * vFogDepth * vFogDepth);
    #else
      float fogFactor = smoothstep(fogNear, fogFar, vFogDepth);
    #endif
    #ifdef FX_ADDITIVE
      // Additive light must fade *out* with distance; mixing towards the fog
      // colour would make sparks brighten into the haze.
      gl_FragColor.rgb *= (1.0 - fogFactor);
    #else
      gl_FragColor.rgb = mix(gl_FragColor.rgb, fogColor, fogFactor);
    #endif
  #endif
}
`;

/* ------------------------------------------------------------------ *
 * The pool
 * ------------------------------------------------------------------ */

/** Floats per instance across all attributes — for the memory accounting. */
const FLOATS_PER_PARTICLE = 4 + 4 + 4 + 4 + 2;

export class ParticleSystem {
  /** Add this to the scene. */
  readonly object3d: THREE.InstancedMesh;

  readonly capacity: number;

  private readonly geometry: THREE.InstancedBufferGeometry;
  private readonly material: THREE.ShaderMaterial;

  private readonly aCol: THREE.InstancedBufferAttribute;
  private readonly aPos: THREE.InstancedBufferAttribute;
  private readonly aVel: THREE.InstancedBufferAttribute;
  private readonly aSize: THREE.InstancedBufferAttribute;
  private readonly aPhys: THREE.InstancedBufferAttribute;
  private readonly aMisc: THREE.InstancedBufferAttribute;

  /** Expiry time per slot, so `alive` is a cheap scan and not a GPU readback. */
  private readonly expiry: Float32Array;

  private head = 0;
  private time = 0;
  private dirtyLo = Infinity;
  private dirtyHi = -Infinity;
  private wrapped = false;
  private aliveCount = 0;
  private aliveDirty = true;
  private emittedThisFrame = 0;

  constructor(opts: ParticleSystemOptions = {}) {
    this.capacity = Math.max(16, opts.capacity ?? 2048);
    const n = this.capacity;

    const additive = opts.blending === THREE.AdditiveBlending;

    const base = new THREE.PlaneGeometry(1, 1);
    const geo = new THREE.InstancedBufferGeometry();
    geo.index = base.index;
    geo.attributes.position = base.attributes.position;
    geo.attributes.uv = base.attributes.uv;
    geo.instanceCount = n;
    base.dispose();

    const mk = (items: number): THREE.InstancedBufferAttribute => {
      const a = new THREE.InstancedBufferAttribute(new Float32Array(n * items), items);
      a.setUsage(THREE.DynamicDrawUsage);
      return a;
    };

    this.aCol = mk(4);
    this.aPos = mk(4);
    this.aVel = mk(4);
    this.aSize = mk(4);
    this.aPhys = mk(4);
    this.aMisc = mk(2);

    geo.setAttribute('iCol', this.aCol);
    geo.setAttribute('iPos', this.aPos);
    geo.setAttribute('iVel', this.aVel);
    geo.setAttribute('iSize', this.aSize);
    geo.setAttribute('iPhys', this.aPhys);
    geo.setAttribute('iMisc', this.aMisc);

    // Everything is positioned in the vertex shader, so three's own culling
    // has nothing meaningful to test against.
    geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e6);

    this.geometry = geo;
    this.expiry = new Float32Array(n);

    const tex = getFxTextures();
    const nearFade = opts.nearFade ?? [0.35, 1.6];

    this.material = new THREE.ShaderMaterial({
      uniforms: THREE.UniformsUtils.merge([
        THREE.UniformsLib.fog,
        {
          uMap: { value: tex.atlas },
          uDepth: { value: null },
          uHasDepth: { value: 0 },
          uDepthRange: { value: new THREE.Vector2(0.1, 1000) },
          uSoftDepth: { value: 1.2 },
          uSoftFade: { value: opts.softFade ?? 0.9 },
          uTime: { value: 0 },
          uSizeScale: { value: 1 },
          uOpacityScale: { value: 1 },
          uTurbulence: { value: opts.turbulence ?? 0.25 },
          uFadeIn: { value: opts.fadeIn ?? 0.14 },
          uBounce: { value: opts.bounce ?? 0.3 },
          uFriction: { value: opts.friction ?? 0.55 },
          uStreak: { value: opts.streak ?? 1 },
          uNearFade: { value: new THREE.Vector2(nearFade[0], nearFade[1]) },
          uInvGrid: { value: 1 / ATLAS_GRID },
        },
      ]),
      vertexShader: VERT,
      fragmentShader: FRAG,
      defines: additive ? { FX_ADDITIVE: '' } : {},
      transparent: true,
      depthTest: true,
      depthWrite: false,
      blending: opts.blending ?? THREE.NormalBlending,
      side: THREE.DoubleSide,
      fog: true,
      toneMapped: true,
    });
    // UniformsUtils.merge clones values, so re-point the texture (clone()
    // on a Texture would otherwise cost us a second GPU upload).
    this.material.uniforms.uMap!.value = tex.atlas;
    this.material.uniforms.uSoftDepth!.value = opts.softDepth ?? 1.2;

    const mesh = new THREE.InstancedMesh(this.geometry, this.material, n);
    mesh.frustumCulled = false;
    mesh.renderOrder = opts.renderOrder ?? 10;
    mesh.name = opts.name ?? 'fx.particles';
    // InstancedMesh allocates an instanceMatrix we never use; drop it so the
    // renderer doesn't upload 64 bytes per particle every frame.
    mesh.instanceMatrix.setUsage(THREE.StaticDrawUsage);
    this.object3d = mesh;
  }

  /* ---------------- emission ---------------- */

  /** Room left before the ring buffer starts recycling live particles. */
  get free(): number {
    return this.capacity - this.alive;
  }

  /** Live particle count. Recomputed at most once per `update()`. */
  get alive(): number {
    if (this.aliveDirty) {
      let c = 0;
      const e = this.expiry;
      const t = this.time;
      for (let i = 0; i < e.length; i++) if (e[i]! > t) c++;
      this.aliveCount = c;
      this.aliveDirty = false;
    }
    return this.aliveCount;
  }

  /**
   * Write one particle into the ring buffer. Overwrites the oldest slot when
   * full, which is exactly the behaviour you want under load: the newest,
   * closest-to-the-action particles win.
   */
  emit(d: ParticleDesc): void {
    const i = this.head;
    this.head = (this.head + 1) % this.capacity;
    if (this.head === 0) this.wrapped = true;

    const c = i * 4;
    const co = this.aCol.array as Float32Array;
    co[c] = d.r;
    co[c + 1] = d.g;
    co[c + 2] = d.b;
    co[c + 3] = d.cell;

    const po = this.aPos.array as Float32Array;
    po[c] = d.x;
    po[c + 1] = d.y;
    po[c + 2] = d.z;
    po[c + 3] = this.time;

    const ve = this.aVel.array as Float32Array;
    ve[c] = d.vx;
    ve[c + 1] = d.vy;
    ve[c + 2] = d.vz;
    ve[c + 3] = d.life;

    const si = this.aSize.array as Float32Array;
    si[c] = d.size0;
    si[c + 1] = d.size1;
    si[c + 2] = d.rot;
    si[c + 3] = d.rotSpeed;

    const ph = this.aPhys.array as Float32Array;
    ph[c] = d.drag;
    ph[c + 1] = d.gravity;
    ph[c + 2] = d.opacity;
    ph[c + 3] = d.groundY;

    const mi = this.aMisc.array as Float32Array;
    mi[i * 2] = Math.random();
    mi[i * 2 + 1] = d.mode;

    this.expiry[i] = this.time + d.life;

    if (i < this.dirtyLo) this.dirtyLo = i;
    if (i > this.dirtyHi) this.dirtyHi = i;
    this.emittedThisFrame++;
  }

  /* ---------------- frame ---------------- */

  /** Advance the clock and flush freshly spawned particles to the GPU. */
  update(dt: number): void {
    this.time += dt;
    this.material.uniforms.uTime!.value = this.time;

    if (this.emittedThisFrame > 0) {
      const lo = this.wrapped ? 0 : this.dirtyLo;
      const hi = this.wrapped ? this.capacity - 1 : this.dirtyHi;
      this.flush(this.aCol, lo, hi);
      this.flush(this.aPos, lo, hi);
      this.flush(this.aVel, lo, hi);
      this.flush(this.aSize, lo, hi);
      this.flush(this.aPhys, lo, hi);
      this.flush(this.aMisc, lo, hi);
      this.aliveDirty = true;
    }
    this.dirtyLo = Infinity;
    this.dirtyHi = -Infinity;
    this.wrapped = false;
    this.emittedThisFrame = 0;
    this.aliveDirty = true;
  }

  /**
   * Queue a byte range for upload.
   *
   * The ranges must *accumulate*: the renderer is what clears them, once it has
   * actually uploaded. Wiping them here would silently drop every batch but the
   * last whenever the pool is stepped more than once between two renders —
   * which is exactly what a fixed-timestep caller, or a scripted fast-forward,
   * does. Emission is sequential in the ring buffer, so consecutive batches
   * merge into one range and the list stays at one or two entries; if it ever
   * fragments badly, fall back to uploading the whole attribute.
   */
  private flush(attr: THREE.InstancedBufferAttribute, lo: number, hi: number): void {
    const ranges = attr.updateRanges;
    const start = lo * attr.itemSize;
    const count = (hi - lo + 1) * attr.itemSize;
    const last = ranges.length > 0 ? ranges[ranges.length - 1]! : null;

    if (last && start <= last.start + last.count && start + count >= last.start) {
      const end = Math.max(last.start + last.count, start + count);
      last.start = Math.min(last.start, start);
      last.count = end - last.start;
    } else if (ranges.length >= 32) {
      attr.clearUpdateRanges();
      attr.addUpdateRange(0, attr.array.length);
    } else {
      attr.addUpdateRange(start, count);
    }
    attr.needsUpdate = true;
  }

  /* ---------------- tuning ---------------- */

  /** Global multiplier on every particle's size. Debug / quality knob. */
  setSizeScale(v: number): void {
    this.material.uniforms.uSizeScale!.value = v;
  }

  /** Global multiplier on every particle's alpha. */
  setOpacityScale(v: number): void {
    this.material.uniforms.uOpacityScale!.value = v;
  }

  setTurbulence(v: number): void {
    this.material.uniforms.uTurbulence!.value = v;
  }

  setSoftFade(metres: number): void {
    this.material.uniforms.uSoftFade!.value = metres;
  }

  /**
   * Hand over the scene's depth buffer to enable true soft particles. Pass
   * `null` to go back to the vertical fade. `near`/`far` must match the camera
   * the depth was rendered with.
   */
  setDepthTexture(depth: THREE.Texture | null, near = 0.1, far = 1000): void {
    const u = this.material.uniforms;
    u.uDepth!.value = depth;
    u.uHasDepth!.value = depth ? 1 : 0;
    (u.uDepthRange!.value as THREE.Vector2).set(near, far);
    this.material.needsUpdate = true;
  }

  setSoftDepth(metres: number): void {
    this.material.uniforms.uSoftDepth!.value = metres;
  }

  get visible(): boolean {
    return this.object3d.visible;
  }

  set visible(v: boolean) {
    this.object3d.visible = v;
  }

  /** Kill everything immediately, e.g. on a respawn. */
  clear(): void {
    this.expiry.fill(0);
    (this.aVel.array as Float32Array).fill(0); // lifetime 0 == dead
    this.aVel.clearUpdateRanges();
    this.aVel.needsUpdate = true;
    this.head = 0;
    this.aliveDirty = true;
  }

  /** Bytes of instance data held on the CPU (and mirrored on the GPU). */
  get byteSize(): number {
    return this.capacity * FLOATS_PER_PARTICLE * 4;
  }

  dispose(): void {
    this.object3d.removeFromParent();
    this.geometry.dispose();
    this.material.dispose();
    (this.object3d as THREE.InstancedMesh).dispose();
  }
}
