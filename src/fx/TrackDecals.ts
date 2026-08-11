/**
 * TrackDecals.ts — tyre tracks and skid marks.
 *
 * Each wheel trails a continuous ribbon of quads laid onto the terrain. The
 * ribbon lives in a fixed ring buffer, so the total amount of track on screen
 * is capped no matter how long the session runs; the oldest segment is simply
 * overwritten by the newest.
 *
 * Continuity comes from sharing edges: segment N's trailing edge is byte-for-
 * byte segment N-1's leading edge, so the ribbon has no seams and no gaps
 * around corners. When a wheel leaves the ground the strip is broken and a new
 * one is anchored on the next contact.
 *
 * Fading is done on the GPU from a per-vertex birth stamp against a `uTime`
 * uniform, which means an idle car costs one uniform write per frame and no
 * vertex traffic at all.
 *
 * Z-fighting is fought on two fronts: the ribbon is lifted a few centimetres
 * along the terrain normal, *and* the material carries a polygon offset. The
 * lift alone fails on steep ground viewed edge-on; the offset alone fails at
 * distance where depth precision collapses.
 */

import * as THREE from 'three';
import type { SurfaceKind, TerrainSampler, WheelState } from '../types';
import { getFxTextures } from './fxTextures';

/* ------------------------------------------------------------------ *
 * Per-surface look
 * ------------------------------------------------------------------ */

export interface TrackSurfaceLook {
  /** Linear colour of a rolling track on this surface. */
  color: THREE.Color;
  /** Opacity of a track laid by a freely rolling, loaded wheel. */
  roll: number;
  /** Extra opacity at full slip. */
  skid: number;
  /**
   * How much a skidding wheel darkens the mark towards scrubbed rubber. Hard
   * surfaces go almost black; soft ones just churn more soil.
   */
  rubber: number;
}

/** Rubber laid on a hard surface. Linear-space. */
const RUBBER = new THREE.Color(0x14100e);

function look(hex: number, roll: number, skid: number, rubber: number): TrackSurfaceLook {
  return { color: new THREE.Color(hex), roll, skid, rubber };
}

const TRACK_LOOK: Record<SurfaceKind, TrackSurfaceLook> = {
  // Soil turns over dark and wet-looking under a tyre.
  dirt: look(0x3a2a1a, 0.5, 0.42, 0.35),
  // Grass bruises rather than marks; it takes a hard scrub to show.
  grass: look(0x2c3a1a, 0.34, 0.5, 0.3),
  // Nothing rolls a mark into rock — only rubber does.
  rock: look(0x2a2724, 0.0, 0.62, 0.9),
  gravel: look(0x453f36, 0.34, 0.4, 0.3),
  sand: look(0x8a6b3d, 0.6, 0.42, 0.12),
  mud: look(0x1a1109, 0.78, 0.5, 0.2),
  snow: look(0x6d7d99, 0.72, 0.44, 0.1),
};

/* ------------------------------------------------------------------ *
 * Options
 * ------------------------------------------------------------------ */

export interface TrackDecalOptions {
  terrain?: TerrainSampler;
  /** How many wheels get a ribbon. */
  wheels?: number;
  /** Ring buffer depth per wheel. Each segment is one quad. */
  segmentsPerWheel?: number;
  /** Ribbon width, metres. Should be a touch wider than the tyre. */
  width?: number;
  /** Shortest segment, metres. Grows with speed up to `maxStep`. */
  step?: number;
  maxStep?: number;
  /** Metres of travel per repeat of the tread pattern. */
  tile?: number;
  /** Seconds from laid to gone. */
  life?: number;
  /** Metres lifted along the terrain normal. */
  lift?: number;
  /** Global opacity multiplier. */
  opacity?: number;
  renderOrder?: number;
}

/* ------------------------------------------------------------------ *
 * Shaders
 * ------------------------------------------------------------------ */

const VERT = /* glsl */ `
precision highp float;

attribute vec3 aColor;
attribute vec2 aData;   // x = birth time, y = opacity

uniform float uTime;
uniform float uLife;
uniform float uOpacity;

varying vec2  vUv;
varying vec3  vCol;
varying float vAlpha;

#include <fog_pars_vertex>

void main() {
  float age = uTime - aData.x;
  // Hold at full strength for the first 45% of the life, then ease away. A
  // track that starts dimming the instant it is laid never reads as a mark in
  // the ground, it reads as a fading decal.
  float fade = 1.0 - smoothstep(uLife * 0.45, uLife, age);
  vAlpha = aData.y * uOpacity * fade;

  if (vAlpha <= 0.002 || age < 0.0) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    vUv = vec2(0.0);
    vCol = vec3(0.0);
    vAlpha = 0.0;
    #ifdef USE_FOG
      vFogDepth = 0.0;
    #endif
    return;
  }

  vUv = uv;
  vCol = aColor;

  vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mvPosition;

  #include <fog_vertex>
}
`;

const FRAG = /* glsl */ `
precision highp float;

uniform sampler2D uMap;

varying vec2  vUv;
varying vec3  vCol;
varying float vAlpha;

#include <fog_pars_fragment>

void main() {
  if (vAlpha <= 0.0) discard;

  vec4 t = texture2D(uMap, vUv);
  float a = t.a * vAlpha;
  if (a <= 0.004) discard;

  // The texture's luminance is relief, not colour: the rut floor is darker
  // than the ridge between the lugs. Multiplying keeps the mark reading as an
  // impression in the ground rather than paint on top of it.
  gl_FragColor = vec4(vCol * (0.55 + t.r * 0.75), a);

  #include <colorspace_fragment>

  #ifdef USE_FOG
    #ifdef FOG_EXP2
      float fogFactor = 1.0 - exp(-fogDensity * fogDensity * vFogDepth * vFogDepth);
    #else
      float fogFactor = smoothstep(fogNear, fogFar, vFogDepth);
    #endif
    gl_FragColor.rgb = mix(gl_FragColor.rgb, fogColor, fogFactor);
  #endif
}
`;

/* ------------------------------------------------------------------ *
 * Per-wheel strip cursor
 * ------------------------------------------------------------------ */

interface Strip {
  /** true once an anchor exists and the next move can close a quad. */
  active: boolean;
  /** Ring cursor within this wheel's slice of the buffer. */
  head: number;
  /** Contact point the current leading edge sits on. */
  anchor: THREE.Vector3;
  /** Leading edge corners, reused as the next segment's trailing edge. */
  left: THREE.Vector3;
  right: THREE.Vector3;
  /** Texture V at the leading edge, so the tread tiles continuously. */
  v: number;
  /** Horizontal travel direction of the last emitted segment. */
  dir: THREE.Vector3;
  haveDir: boolean;
  /** Opacity of the leading edge, so segments blend into each other. */
  alpha: number;
  color: THREE.Color;
}

const VERTS_PER_SEG = 4;
const INDICES_PER_SEG = 6;

/* ------------------------------------------------------------------ */

export class TrackDecals {
  readonly object3d: THREE.Mesh;

  private readonly terrain: TerrainSampler | undefined;
  private readonly segments: number;
  private readonly wheelCount: number;
  private readonly halfWidth: number;
  private readonly step: number;
  private readonly maxStep: number;
  private readonly tile: number;
  private readonly lift: number;

  private readonly geometry: THREE.BufferGeometry;
  private readonly material: THREE.ShaderMaterial;

  private readonly aPos: THREE.BufferAttribute;
  private readonly aUv: THREE.BufferAttribute;
  private readonly aColor: THREE.BufferAttribute;
  private readonly aData: THREE.BufferAttribute;

  private readonly strips: Strip[] = [];
  private time = 0;
  private dirty = false;

  // Scratch. Nothing in the hot path allocates.
  private readonly tmpB = new THREE.Vector3();
  private readonly tmpDir = new THREE.Vector3();
  private readonly tmpNrm = new THREE.Vector3();
  private readonly tmpLat = new THREE.Vector3();
  private readonly tmpL = new THREE.Vector3();
  private readonly tmpR = new THREE.Vector3();
  private readonly tmpCol = new THREE.Color();

  constructor(opts: TrackDecalOptions = {}) {
    this.terrain = opts.terrain;
    this.wheelCount = Math.max(1, opts.wheels ?? 4);
    this.segments = Math.max(8, opts.segmentsPerWheel ?? 320);
    this.halfWidth = (opts.width ?? 0.34) * 0.5;
    this.step = opts.step ?? 0.42;
    this.maxStep = Math.max(this.step, opts.maxStep ?? 1.25);
    this.tile = opts.tile ?? 0.75;
    this.lift = opts.lift ?? 0.035;

    const quads = this.wheelCount * this.segments;
    const vertCount = quads * VERTS_PER_SEG;

    const geo = new THREE.BufferGeometry();
    this.aPos = new THREE.BufferAttribute(new Float32Array(vertCount * 3), 3);
    this.aUv = new THREE.BufferAttribute(new Float32Array(vertCount * 2), 2);
    this.aColor = new THREE.BufferAttribute(new Float32Array(vertCount * 3), 3);
    this.aData = new THREE.BufferAttribute(new Float32Array(vertCount * 2), 2);
    for (const a of [this.aPos, this.aUv, this.aColor, this.aData]) {
      a.setUsage(THREE.DynamicDrawUsage);
    }
    geo.setAttribute('position', this.aPos);
    geo.setAttribute('uv', this.aUv);
    geo.setAttribute('aColor', this.aColor);
    geo.setAttribute('aData', this.aData);

    // Static index buffer: two triangles per quad, corners ordered
    // trailing-left, trailing-right, leading-left, leading-right.
    const idx = vertCount > 65535 ? new Uint32Array(quads * INDICES_PER_SEG)
                                  : new Uint16Array(quads * INDICES_PER_SEG);
    for (let q = 0; q < quads; q++) {
      const v = q * VERTS_PER_SEG;
      const o = q * INDICES_PER_SEG;
      idx[o] = v;
      idx[o + 1] = v + 1;
      idx[o + 2] = v + 2;
      idx[o + 3] = v + 2;
      idx[o + 4] = v + 1;
      idx[o + 5] = v + 3;
    }
    geo.setIndex(new THREE.BufferAttribute(idx, 1));
    // Vertices move constantly and are placed in world space, so three's own
    // culling has nothing stable to test.
    geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e6);
    this.geometry = geo;

    this.material = new THREE.ShaderMaterial({
      uniforms: THREE.UniformsUtils.merge([
        THREE.UniformsLib.fog,
        {
          uMap: { value: null },
          uTime: { value: 0 },
          uLife: { value: opts.life ?? 42 },
          uOpacity: { value: opts.opacity ?? 1 },
        },
      ]),
      vertexShader: VERT,
      fragmentShader: FRAG,
      transparent: true,
      depthTest: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      blending: THREE.NormalBlending,
      fog: true,
      toneMapped: true,
      // Pull the ribbon towards the camera in depth. Combined with the normal
      // lift this survives both grazing angles and long view distances.
      polygonOffset: true,
      polygonOffsetFactor: -4,
      polygonOffsetUnits: -8,
    });
    this.material.uniforms.uMap!.value = getFxTextures().track;

    const mesh = new THREE.Mesh(this.geometry, this.material);
    mesh.frustumCulled = false;
    mesh.renderOrder = opts.renderOrder ?? 5;
    mesh.name = 'fx.tracks';
    mesh.matrixAutoUpdate = false;
    this.object3d = mesh;

    for (let i = 0; i < this.wheelCount; i++) {
      this.strips.push({
        active: false,
        head: 0,
        anchor: new THREE.Vector3(),
        left: new THREE.Vector3(),
        right: new THREE.Vector3(),
        v: 0,
        dir: new THREE.Vector3(0, 0, 1),
        haveDir: false,
        alpha: 0,
        color: new THREE.Color(),
      });
    }
  }

  /** Seconds a mark survives. */
  get life(): number {
    return this.material.uniforms.uLife!.value as number;
  }

  set life(v: number) {
    this.material.uniforms.uLife!.value = v;
  }

  setOpacity(v: number): void {
    this.material.uniforms.uOpacity!.value = v;
  }

  get visible(): boolean {
    return this.object3d.visible;
  }

  set visible(v: boolean) {
    this.object3d.visible = v;
  }

  /** Quads currently holding a mark young enough to be drawn. */
  get liveSegments(): number {
    let n = 0;
    const d = this.aData.array as Float32Array;
    const life = this.life;
    for (let q = 0; q < this.wheelCount * this.segments; q++) {
      const i = q * VERTS_PER_SEG * 2;
      if (d[i + 1]! > 0 && this.time - d[i]! < life) n++;
    }
    return n;
  }

  /**
   * Advance and lay new segments.
   *
   * `contactY` is the ground height under each wheel; pass the value the FX
   * layer already computed rather than making this class re-derive it.
   */
  update(
    dt: number,
    wheels: readonly WheelState[],
    contactY: readonly number[],
    speed: number,
  ): void {
    this.time += dt;
    this.material.uniforms.uTime!.value = this.time;

    const n = Math.min(wheels.length, this.wheelCount);
    // Longer segments at speed: the ribbon holds far more history for the same
    // buffer, and at 25 m/s a 1.2 m quad is still under two frames of travel.
    const stride = THREE.MathUtils.clamp(
      this.step + speed * 0.032,
      this.step,
      this.maxStep,
    );

    for (let i = 0; i < n; i++) {
      this.updateWheel(i, wheels[i]!, contactY[i] ?? 0, stride);
    }

    if (this.dirty) {
      this.aPos.needsUpdate = true;
      this.aUv.needsUpdate = true;
      this.aColor.needsUpdate = true;
      this.aData.needsUpdate = true;
      this.dirty = false;
    }
  }

  private updateWheel(i: number, w: WheelState, groundY: number, stride: number): void {
    const s = this.strips[i]!;

    if (!w.grounded) {
      s.active = false;
      s.haveDir = false;
      return;
    }

    // `intensity` writes the mark colour into `tmpCol`; `s.color` holds the
    // colour of the strip's current leading edge, i.e. the previous sample.
    const alpha = this.intensity(w, this.tmpCol);
    const cx = w.position.x;
    const cz = w.position.z;

    if (!s.active) {
      // Anchor a fresh strip. No quad yet — the first one closes on the next
      // move, which is what stops a stationary wheel stamping a mark.
      s.color.copy(this.tmpCol);
      this.anchorAt(s, cx, groundY, cz, this.tmpDir.set(0, 0, 0), alpha);
      s.active = true;
      s.haveDir = false;
      return;
    }

    this.tmpDir.set(cx - s.anchor.x, 0, cz - s.anchor.z);
    const dist = this.tmpDir.length();
    if (dist < 1e-5) return;
    this.tmpDir.multiplyScalar(1 / dist);

    // Emit early through a corner so the ribbon does not cut it.
    const turned = s.haveDir && this.tmpDir.dot(s.dir) < 0.985;
    const trigger = turned ? Math.min(stride, 0.35) : stride;
    if (dist < trigger) return;

    this.emit(i, s, cx, groundY, cz, dist, alpha, this.tmpCol);
  }

  /** Opacity this wheel should be laying right now, and its colour. */
  private intensity(w: WheelState, out: THREE.Color): number {
    const lk = TRACK_LOOK[w.surface] ?? TRACK_LOOK.dirt;
    const long = Math.abs(w.slipRatio);
    const lat = Math.abs(Math.sin(w.slipAngle));
    const slip = THREE.MathUtils.clamp(Math.hypot(long * 0.85, lat * 1.35), 0, 1);
    // Load relative to a quarter of a laden 4x4. A wheel that has gone light
    // over a crest should barely mark.
    const loadF = THREE.MathUtils.clamp(w.load / 4200, 0, 1.5);

    out.copy(lk.color).lerp(RUBBER, lk.rubber * slip);
    const a = (lk.roll + lk.skid * slip * slip) * THREE.MathUtils.clamp(loadF, 0, 1.25);
    return THREE.MathUtils.clamp(a, 0, 0.92);
  }

  /** Place the leading edge without emitting geometry. */
  private anchorAt(
    s: Strip,
    x: number,
    y: number,
    z: number,
    dir: THREE.Vector3,
    alpha: number,
  ): void {
    s.anchor.set(x, y, z);
    this.groundNormal(x, z, this.tmpNrm);
    if (dir.lengthSq() < 1e-6) {
      // No travel direction yet: any axis perpendicular to the normal will do,
      // and the first real segment will re-derive it.
      this.tmpLat.set(1, 0, 0).cross(this.tmpNrm);
      if (this.tmpLat.lengthSq() < 1e-6) this.tmpLat.set(0, 0, 1).cross(this.tmpNrm);
      this.tmpLat.normalize();
    } else {
      this.tmpLat.copy(this.tmpNrm).cross(dir).normalize();
    }
    this.corner(s.left, x, y, z, -this.halfWidth);
    this.corner(s.right, x, y, z, this.halfWidth);
    s.alpha = alpha;
  }

  /**
   * Corner position, snapped to the terrain and lifted along its normal. The
   * per-corner height query is what makes the ribbon sit in a rut or bank over
   * a camber instead of hovering as a flat plate.
   */
  private corner(
    out: THREE.Vector3,
    cx: number,
    cy: number,
    cz: number,
    offset: number,
  ): void {
    const x = cx + this.tmpLat.x * offset;
    const z = cz + this.tmpLat.z * offset;
    const t = this.terrain;
    if (t) {
      t.normalAt(x, z, this.tmpB);
      out.set(
        x + this.tmpB.x * this.lift,
        t.heightAt(x, z) + this.tmpB.y * this.lift,
        z + this.tmpB.z * this.lift,
      );
    } else {
      // No sampler: the wheel's own contact height is the best guess there is.
      out.set(x, cy + this.lift, z);
    }
  }

  private groundNormal(x: number, z: number, out: THREE.Vector3): THREE.Vector3 {
    if (this.terrain) return this.terrain.normalAt(x, z, out);
    return out.set(0, 1, 0);
  }

  private emit(
    wheel: number,
    s: Strip,
    cx: number,
    cy: number,
    cz: number,
    dist: number,
    alpha: number,
    color: THREE.Color,
  ): void {
    // The strip's current leading edge becomes this quad's trailing edge, so
    // consecutive segments share vertices exactly and the ribbon is watertight.
    this.tmpL.copy(s.left);
    this.tmpR.copy(s.right);
    const v0 = s.v;
    const a0 = s.alpha;

    this.groundNormal(cx, cz, this.tmpNrm);
    this.tmpLat.copy(this.tmpNrm).cross(this.tmpDir);
    if (this.tmpLat.lengthSq() < 1e-8) this.tmpLat.set(1, 0, 0);
    this.tmpLat.normalize();

    s.anchor.set(cx, cy, cz);
    this.corner(s.left, cx, cy, cz, -this.halfWidth);
    this.corner(s.right, cx, cy, cz, this.halfWidth);
    s.v += dist / this.tile;
    s.dir.copy(this.tmpDir);
    s.haveDir = true;

    const q = wheel * this.segments + s.head;
    s.head = (s.head + 1) % this.segments;

    const base = q * VERTS_PER_SEG;
    const p = this.aPos.array as Float32Array;
    const u = this.aUv.array as Float32Array;
    const c = this.aColor.array as Float32Array;
    const d = this.aData.array as Float32Array;

    const write = (
      slot: number,
      pos: THREE.Vector3,
      uu: number,
      vv: number,
      col: THREE.Color,
      a: number,
    ): void => {
      const vi = base + slot;
      p[vi * 3] = pos.x;
      p[vi * 3 + 1] = pos.y;
      p[vi * 3 + 2] = pos.z;
      u[vi * 2] = uu;
      u[vi * 2 + 1] = vv;
      c[vi * 3] = col.r;
      c[vi * 3 + 1] = col.g;
      c[vi * 3 + 2] = col.b;
      d[vi * 2] = this.time;
      d[vi * 2 + 1] = a;
    };

    // Trailing edge inherits the previous sample's colour and opacity, so a
    // mark that ramps from rolling to full lock does so smoothly along its
    // length rather than in visible steps.
    write(0, this.tmpL, 0, v0, s.color, a0);
    write(1, this.tmpR, 1, v0, s.color, a0);
    s.color.copy(color);
    s.alpha = alpha;
    write(2, s.left, 0, s.v, s.color, alpha);
    write(3, s.right, 1, s.v, s.color, alpha);

    this.aPos.addUpdateRange(base * 3, VERTS_PER_SEG * 3);
    this.aUv.addUpdateRange(base * 2, VERTS_PER_SEG * 2);
    this.aColor.addUpdateRange(base * 3, VERTS_PER_SEG * 3);
    this.aData.addUpdateRange(base * 2, VERTS_PER_SEG * 2);
    this.dirty = true;
  }

  /** Wipe every mark, e.g. on a respawn. */
  clear(): void {
    (this.aData.array as Float32Array).fill(0);
    this.aData.clearUpdateRanges();
    this.aData.needsUpdate = true;
    for (const s of this.strips) {
      s.active = false;
      s.haveDir = false;
      s.head = 0;
      s.v = 0;
    }
  }

  /** Bytes of vertex data held on the CPU and mirrored on the GPU. */
  get byteSize(): number {
    const verts = this.wheelCount * this.segments * VERTS_PER_SEG;
    return verts * (3 + 2 + 3 + 2) * 4 + verts * 1.5 * (verts > 65535 ? 4 : 2);
  }

  dispose(): void {
    this.object3d.removeFromParent();
    this.geometry.dispose();
    this.material.dispose();
  }
}
