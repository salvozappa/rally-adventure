/**
 * Geometry construction helpers for the Jeep.
 *
 * Everything here is pure geometry — no materials, no scene graph. The rule
 * that drives most of it: a procedural vehicle looks cheap the instant you
 * leave a hard 90 degree edge on a body panel, so the workhorse primitive is a
 * *chamfered* box built from a bevelled extrusion rather than `BoxGeometry`.
 */
import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

export type Vec3 = [number, number, number];

export interface Placement {
  pos?: Vec3;
  /** Euler XYZ, radians. */
  rot?: Vec3;
  scale?: Vec3 | number;
  /** Applied after rot/pos: mirrors across X (for left/right handed parts). */
  mirrorX?: boolean;
}

const _m = new THREE.Matrix4();
const _q = new THREE.Quaternion();
const _e = new THREE.Euler();
const _v = new THREE.Vector3();
const _s = new THREE.Vector3();

const _mirror = new THREE.Matrix4().makeScale(-1, 1, 1);

/**
 * Bakes a placement into a geometry (mutates and returns it). `mirrorX` mirrors
 * the *placed* part across the YZ plane, so a right-hand part becomes its
 * left-hand twin without recomputing any coordinates by hand.
 */
export function place(geo: THREE.BufferGeometry, p: Placement): THREE.BufferGeometry {
  const s = p.scale ?? 1;
  _s.set(...(typeof s === 'number' ? ([s, s, s] as Vec3) : s));
  _e.set(...(p.rot ?? [0, 0, 0]));
  _q.setFromEuler(_e);
  _v.set(...(p.pos ?? [0, 0, 0]));
  geo.applyMatrix4(_m.compose(_v, _q, _s));
  if (p.mirrorX) {
    geo.applyMatrix4(_mirror);
    // Un-mirroring the winding: negative determinant flips faces, so rebuild
    // normals from the mirrored positions instead of trusting the old ones.
    const idx = geo.getIndex();
    if (idx) {
      const a = idx.array as Uint32Array | Uint16Array;
      for (let i = 0; i < a.length; i += 3) {
        const t = a[i]!;
        a[i] = a[i + 2]!;
        a[i + 2] = t;
      }
      idx.needsUpdate = true;
    } else {
      flipWinding(geo);
    }
    geo.computeVertexNormals();
  }
  return geo;
}

function flipWinding(geo: THREE.BufferGeometry): void {
  for (const name of Object.keys(geo.attributes)) {
    const attr = geo.attributes[name] as THREE.BufferAttribute;
    const arr = attr.array as Float32Array;
    const n = attr.itemSize;
    for (let i = 0; i < arr.length; i += n * 3) {
      for (let k = 0; k < n; k++) {
        const t = arr[i + k]!;
        arr[i + k] = arr[i + 2 * n + k]!;
        arr[i + 2 * n + k] = t;
      }
    }
    attr.needsUpdate = true;
  }
}

/** Number of triangles in a geometry. */
export function triCount(geo: THREE.BufferGeometry): number {
  const idx = geo.getIndex();
  return (idx ? idx.count : geo.getAttribute('position').count) / 3;
}

/** Rendered triangles in a subtree, counting each instance of shared geometry. */
export function countTriangles(root: THREE.Object3D): number {
  let n = 0;
  root.traverse((o) => {
    const mesh = o as THREE.Mesh;
    if (mesh.isMesh && mesh.geometry) n += triCount(mesh.geometry);
  });
  return n;
}

// ---------------------------------------------------------------------------
// shapes
// ---------------------------------------------------------------------------

/**
 * Rounded rectangle centred on the origin. `seg` = segments per corner.
 *
 * The corners are emitted as explicit line segments rather than arcs: a
 * `THREE.Shape` carries no per-shape tessellation setting (`curveSegments`
 * lives on the *ExtrudeGeometry options*), so baking the resolution into the
 * point list is the only way `seg` can mean anything. It also keeps the
 * extrusion exact — with `curveSegments: 1` every emitted point survives and
 * nothing extra is inserted.
 */
export function roundedRectShape(w: number, h: number, r: number, seg = 1): THREE.Shape {
  const x = w / 2;
  const y = h / 2;
  const rr = Math.max(0.0001, Math.min(r, x, y));
  const n = Math.max(1, Math.round(seg));
  const s = new THREE.Shape();
  s.moveTo(-x + rr, -y);
  s.lineTo(x - rr, -y);
  arcTo(s, x - rr, -y + rr, rr, -Math.PI / 2, 0, n);
  s.lineTo(x, y - rr);
  arcTo(s, x - rr, y - rr, rr, 0, Math.PI / 2, n);
  s.lineTo(-x + rr, y);
  arcTo(s, -x + rr, y - rr, rr, Math.PI / 2, Math.PI, n);
  s.lineTo(-x, -y + rr);
  arcTo(s, -x + rr, -y + rr, rr, Math.PI, Math.PI * 1.5, n);
  s.closePath();
  return s;
}

/** Appends an arc as `n` straight segments (the start point is assumed current). */
function arcTo(
  s: THREE.Shape,
  cx: number,
  cy: number,
  r: number,
  a0: number,
  a1: number,
  n: number,
): void {
  for (let i = 1; i <= n; i++) {
    const a = a0 + ((a1 - a0) * i) / n;
    s.lineTo(cx + Math.cos(a) * r, cy + Math.sin(a) * r);
  }
}

/** Closed polygon from 2D points. */
export function polyShape(pts: Array<[number, number]>): THREE.Shape {
  const s = new THREE.Shape();
  s.moveTo(pts[0]![0], pts[0]![1]);
  for (let i = 1; i < pts.length; i++) s.lineTo(pts[i]![0], pts[i]![1]);
  s.closePath();
  return s;
}

export interface ExtrudeOpts {
  /** Bevel size and depth; 0 disables the bevel. */
  chamfer?: number;
  /** Bevel subdivisions — 1 gives a flat 45 degree chamfer, which is what we want. */
  bevelSegments?: number;
  curveSegments?: number;
  /** Centre the result on the extrusion axis. */
  centered?: boolean;
  /** Scales the generated UVs (extrusion UVs are in shape units). */
  uvScale?: number;
}

/**
 * Extrudes a shape along +Z with a chamfer on both caps. Total depth is exactly
 * `depth`; the chamfer eats into it rather than growing it.
 */
export function extrudeShape(
  shape: THREE.Shape | THREE.Shape[],
  depth: number,
  opts: ExtrudeOpts = {},
): THREE.BufferGeometry {
  const c = Math.min(opts.chamfer ?? 0, depth * 0.45);
  const geo = new THREE.ExtrudeGeometry(shape, {
    depth: depth - 2 * c,
    bevelEnabled: c > 0,
    bevelThickness: c,
    bevelSize: c,
    bevelOffset: 0,
    bevelSegments: opts.bevelSegments ?? 1,
    curveSegments: opts.curveSegments ?? 1,
    steps: 1,
    UVGenerator: uvGenerator(opts.uvScale ?? 1),
  });
  if (opts.centered !== false) geo.translate(0, 0, -(depth / 2 - c));
  return geo;
}

/**
 * World-scale UVs for extrusions. The stock generator maps side walls to the
 * raw shape coordinates, which stretches paint texture badly on long panels;
 * this one uses arc-length across and depth along.
 */
function uvGenerator(scale: number): THREE.UVGenerator {
  return {
    generateTopUV(_g, v, a, b, c) {
      return [
        new THREE.Vector2(v[a * 3]! * scale, v[a * 3 + 1]! * scale),
        new THREE.Vector2(v[b * 3]! * scale, v[b * 3 + 1]! * scale),
        new THREE.Vector2(v[c * 3]! * scale, v[c * 3 + 1]! * scale),
      ];
    },
    generateSideWallUV(_g, v, a, b, c, d) {
      const uv = (i: number): THREE.Vector2 => {
        const x = v[i * 3]!;
        const y = v[i * 3 + 1]!;
        const z = v[i * 3 + 2]!;
        return new THREE.Vector2((Math.abs(x) + Math.abs(y)) * scale, z * scale);
      };
      return [uv(a), uv(b), uv(c), uv(d)];
    },
  };
}

/**
 * A box with every edge chamfered. `corner` softens the four edges parallel to
 * Z as well, which is what stops procedural bodywork looking like Lego.
 */
export function chamferBox(
  w: number,
  h: number,
  d: number,
  chamfer = Math.min(w, h, d) * 0.12,
  corner = chamfer,
): THREE.BufferGeometry {
  const c = Math.min(chamfer, w * 0.45, h * 0.45, d * 0.45);
  return extrudeShape(roundedRectShape(w, h, Math.max(corner, c * 1.0001), 1), d, {
    chamfer: c,
    uvScale: 1,
  });
}

/** Chamfered box aligned to an arbitrary axis: 'x' | 'y' | 'z' is the depth axis. */
export function chamferBoxOn(
  axis: 'x' | 'y' | 'z',
  sx: number,
  sy: number,
  sz: number,
  chamfer?: number,
  corner?: number,
): THREE.BufferGeometry {
  if (axis === 'z') return chamferBox(sx, sy, sz, chamfer, corner);
  if (axis === 'x') {
    const g = chamferBox(sz, sy, sx, chamfer, corner);
    g.rotateY(Math.PI / 2);
    return g;
  }
  const g = chamferBox(sx, sz, sy, chamfer, corner);
  g.rotateX(-Math.PI / 2);
  return g;
}

// ---------------------------------------------------------------------------
// tubes — roll cage, bull bar, sliders, exhaust
// ---------------------------------------------------------------------------

/** Polyline with radiused corners, as a single continuous curve. */
export function roundedPath(points: THREE.Vector3[], radius: number): THREE.CurvePath<THREE.Vector3> {
  const path = new THREE.CurvePath<THREE.Vector3>();
  if (points.length < 2) throw new Error('roundedPath needs >= 2 points');
  let from = points[0]!.clone();
  for (let i = 1; i < points.length - 1; i++) {
    const p = points[i]!;
    const next = points[i + 1]!;
    const inDir = new THREE.Vector3().subVectors(p, from);
    const outDir = new THREE.Vector3().subVectors(next, p);
    const inLen = inDir.length();
    const outLen = outDir.length();
    if (inLen < 1e-5 || outLen < 1e-5) continue;
    inDir.divideScalar(inLen);
    outDir.divideScalar(outLen);
    const r = Math.min(radius, inLen * 0.5, outLen * 0.5);
    const a = new THREE.Vector3().copy(p).addScaledVector(inDir, -r);
    const b = new THREE.Vector3().copy(p).addScaledVector(outDir, r);
    if (a.distanceTo(from) > 1e-4) path.add(new THREE.LineCurve3(from.clone(), a));
    path.add(new THREE.QuadraticBezierCurve3(a, p.clone(), b));
    from = b;
  }
  const last = points[points.length - 1]!;
  if (last.distanceTo(from) > 1e-4) path.add(new THREE.LineCurve3(from.clone(), last.clone()));
  return path;
}

export interface TubeOpts {
  cornerRadius?: number;
  radialSegments?: number;
  /** Tubular segments per metre of path. */
  density?: number;
  caps?: boolean;
}

/** Round tube following a polyline, with radiused bends. */
export function tube(pts: Vec3[], radius: number, opts: TubeOpts = {}): THREE.BufferGeometry {
  const v = pts.map((p) => new THREE.Vector3(...p));
  const curve = roundedPath(v, opts.cornerRadius ?? radius * 3);
  const len = curve.getLength();
  const segs = Math.max(2, Math.round(len * (opts.density ?? 7)));
  const radial = opts.radialSegments ?? 6;
  const geo: THREE.BufferGeometry = new THREE.TubeGeometry(curve, segs, radius, radial, false);
  if (opts.caps === false) return geo;
  const parts: THREE.BufferGeometry[] = [geo];
  for (const t of [0, 1]) {
    const cap = new THREE.CircleGeometry(radius, radial);
    const p = curve.getPoint(t);
    const tan = curve.getTangent(t);
    cap.lookAt(tan);
    if (t === 0) cap.rotateX(Math.PI); // face outward
    cap.translate(p.x, p.y, p.z);
    parts.push(cap);
  }
  return mergeAll(parts);
}

/**
 * Helical coil spring, built along +Y from y=0 to y=height with the bottom coil
 * flattened into a seat. Compression is done by scaling Y, which brings the
 * turns together exactly the way a real coil does.
 */
export function coilSpring(
  height: number,
  coilRadius: number,
  wireRadius: number,
  turns = 5,
  segmentsPerTurn = 7,
  radialSegments = 5,
): THREE.BufferGeometry {
  const n = Math.round(turns * segmentsPerTurn);
  const pts: THREE.Vector3[] = [];
  for (let i = 0; i <= n; i++) {
    const t = i / n;
    const a = t * turns * Math.PI * 2;
    // Ease the pitch at both ends so the end coils sit flat, like a real seat.
    const ease = Math.min(1, t * turns * 1.2, (1 - t) * turns * 1.2);
    const y = height * smoothstepRamp(t, turns);
    const r = coilRadius * (1 - 0.06 * (1 - ease));
    pts.push(new THREE.Vector3(Math.cos(a) * r, y, Math.sin(a) * r));
  }
  const curve = new THREE.CatmullRomCurve3(pts, false, 'centripetal', 0.5);
  return new THREE.TubeGeometry(curve, n, wireRadius, radialSegments, false);
}

function smoothstepRamp(t: number, turns: number): number {
  // Linear in the middle, compressed at the ends: the dead coils occupy less
  // height than the active ones.
  const dead = Math.min(0.4, 1 / turns);
  if (t < dead) return (t / dead) * dead * 0.45;
  if (t > 1 - dead) return 1 - ((1 - t) / dead) * dead * 0.45;
  const inner = (t - dead) / (1 - 2 * dead);
  return dead * 0.45 + inner * (1 - 2 * dead * 0.45);
}

/** Surface of revolution about +Y from (radius, y) profile pairs. */
export function lathe(profile: Array<[number, number]>, segments = 16, phiStart = 0, phiLength = Math.PI * 2): THREE.BufferGeometry {
  return new THREE.LatheGeometry(
    profile.map(([r, y]) => new THREE.Vector2(r, y)),
    segments,
    phiStart,
    phiLength,
  );
}

// ---------------------------------------------------------------------------
// wheel
// ---------------------------------------------------------------------------

export interface TyreSpec {
  /** Outer radius over the tread blocks. */
  radius: number;
  /** Overall section width. */
  width: number;
  /** Rim (bead seat) radius. */
  rimRadius: number;
  /** How far the tread blocks stand proud of the carcass. */
  treadDepth: number;
  /** Tread blocks around the circumference. */
  treadCount: number;
  radialSegments: number;
}

export const DEFAULT_TYRE: TyreSpec = {
  radius: 0.42,
  width: 0.33,
  rimRadius: 0.205,
  treadDepth: 0.028,
  treadCount: 15,
  radialSegments: 16,
};

export interface WheelGeometries {
  /** Carcass + sidewalls (rubber, sidewall texture). */
  carcass: THREE.BufferGeometry;
  /** Tread blocks (rubber). */
  tread: THREE.BufferGeometry;
  /** Wheel rim (painted steel/alloy). */
  rim: THREE.BufferGeometry;
  /** Hub cap and lug nuts (bright trim). */
  hub: THREE.BufferGeometry;
  /** Brake disc + caliper, merged. Mounted behind the rim, so it spins with it. */
  brake: THREE.BufferGeometry;
}

/**
 * A mud-terrain tyre on a beadlock-style wheel, axis along +X.
 *
 * The tread is real geometry rather than a texture: on an off-roader you stare
 * at the wheels constantly and moulded blocks are the single loudest quality
 * signal the model has.
 */
export function buildWheel(spec: TyreSpec = DEFAULT_TYRE): WheelGeometries {
  const { radius: R, width: W, rimRadius: RR, treadDepth: TD, treadCount: N, radialSegments: S } = spec;
  const hw = W / 2;
  const carcassR = R - TD;

  // --- carcass -------------------------------------------------------------
  // Profile runs inner bead -> sidewall bulge -> shoulder -> crown and back.
  const prof: Array<[number, number]> = [
    [RR, -hw * 0.86],
    [RR + 0.035, -hw * 0.92],
    [carcassR * 0.79, -hw * 1.0],
    [carcassR * 0.955, -hw * 0.86],
    [carcassR, -hw * 0.62],
    [carcassR, hw * 0.62],
    [carcassR * 0.955, hw * 0.86],
    [carcassR * 0.79, hw * 1.0],
    [RR + 0.035, hw * 0.92],
    [RR, hw * 0.86],
  ];
  const carcass = lathe(prof, S);
  carcass.rotateZ(-Math.PI / 2); // lathe axis Y -> wheel axis X

  // --- tread ---------------------------------------------------------------
  const blocks: THREE.BufferGeometry[] = [];
  const shoulderX = hw * 0.74;
  for (let i = 0; i < N; i++) {
    const a = (i / N) * Math.PI * 2;
    const odd = i % 2 === 1;
    // Centre lugs: two staggered blocks straddling the crown.
    for (const side of [-1, 1]) {
      const g = new THREE.BoxGeometry(W * 0.24, TD * 2.1, R * 0.3);
      g.translate(side * W * 0.15 + (odd ? W * 0.05 : -W * 0.05), carcassR + TD * 0.15, 0);
      g.rotateZ(odd ? 0.04 : -0.04);
      g.rotateX(a + (side > 0 ? 0.06 : -0.06));
      blocks.push(g);
    }
    // Shoulder lugs wrap over the corner and out onto the sidewall.
    for (const side of [-1, 1]) {
      const g = new THREE.BoxGeometry(W * 0.30, TD * 2.4, R * 0.20);
      g.translate(0, carcassR * 0.965, 0);
      g.rotateZ(side * 0.34);
      g.translate(side * shoulderX, 0, 0);
      g.rotateX(a + (odd ? Math.PI / N : 0));
      blocks.push(g);
    }
  }
  const tread = mergeAll(blocks);

  // --- rim -----------------------------------------------------------------
  const rimParts: THREE.BufferGeometry[] = [];
  // barrel
  const barrel = new THREE.CylinderGeometry(RR * 0.98, RR * 0.98, W * 0.82, S, 1, true);
  barrel.rotateZ(Math.PI / 2);
  rimParts.push(barrel);
  // outer bead ring (beadlock look): a stepped lathe ring on the outboard face
  const ring = lathe(
    [
      [RR * 0.86, hw * 0.5],
      [RR * 0.99, hw * 0.5],
      [RR * 1.02, hw * 0.62],
      [RR * 1.02, hw * 0.78],
      [RR * 0.9, hw * 0.78],
      [RR * 0.86, hw * 0.66],
    ],
    S,
  );
  ring.rotateZ(-Math.PI / 2);
  rimParts.push(ring);
  // dished face
  const face = lathe(
    [
      [0.0, hw * 0.40],
      [0.075, hw * 0.44],
      [0.1, hw * 0.40],
      [RR * 0.86, hw * 0.5],
    ],
    S,
  );
  face.rotateZ(-Math.PI / 2);
  rimParts.push(face);
  // spokes standing proud of the face
  for (let i = 0; i < 5; i++) {
    const a = (i / 5) * Math.PI * 2;
    const sp = new THREE.BoxGeometry(0.026, 0.115, 0.062);
    sp.translate(0, RR * 0.52, 0);
    sp.rotateX(a);
    sp.translate(hw * 0.52, 0, 0);
    rimParts.push(sp);
  }
  const rim = mergeAll(rimParts);

  // --- hub cap + lug nuts --------------------------------------------------
  const hubParts: THREE.BufferGeometry[] = [];
  const cap = new THREE.CylinderGeometry(0.072, 0.062, 0.045, 12, 1, false);
  cap.rotateZ(Math.PI / 2);
  cap.translate(hw * 0.60, 0, 0);
  hubParts.push(cap);
  for (let i = 0; i < 5; i++) {
    const a = (i / 5) * Math.PI * 2 + 0.3;
    const nut = new THREE.CylinderGeometry(0.017, 0.017, 0.028, 6, 1, false);
    nut.rotateZ(Math.PI / 2);
    nut.translate(hw * 0.53, 0.104, 0);
    nut.rotateX(a);
    hubParts.push(nut);
  }
  // Beadlock ring bolts around the outer lip.
  for (let i = 0; i < 8; i++) {
    const a = (i / 8) * Math.PI * 2 + 0.2;
    const bolt = new THREE.CylinderGeometry(0.011, 0.011, 0.02, 5, 1, false);
    bolt.rotateZ(Math.PI / 2);
    bolt.translate(hw * 0.78, RR * 0.94, 0);
    bolt.rotateX(a);
    hubParts.push(bolt);
  }
  const hub = mergeAll(hubParts);

  // --- brake ---------------------------------------------------------------
  const disc = new THREE.CylinderGeometry(RR * 0.72, RR * 0.72, 0.026, 14);
  disc.rotateZ(Math.PI / 2);
  disc.translate(-hw * 0.1, 0, 0);
  const caliper = chamferBox(0.05, 0.09, 0.11, 0.012);
  caliper.translate(-hw * 0.12, RR * 0.62, -0.02);
  const brake = mergeAll([disc, caliper]);

  return { carcass, tread, rim, hub, brake };
}

// ---------------------------------------------------------------------------
// merging
// ---------------------------------------------------------------------------

/** Merge geometries that may differ in indexed-ness or attribute order. */
export function mergeAll(geos: THREE.BufferGeometry[]): THREE.BufferGeometry {
  if (geos.length === 0) return new THREE.BufferGeometry();
  const norm = geos.map((g) => {
    let out = g.index ? g.toNonIndexed() : g;
    if (out !== g) g.dispose();
    if (!out.getAttribute('uv')) {
      const count = out.getAttribute('position').count;
      out.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(count * 2), 2));
    }
    if (!out.getAttribute('normal')) out.computeVertexNormals();
    // Drop anything exotic so the merge sees a uniform attribute set.
    for (const key of Object.keys(out.attributes)) {
      if (key !== 'position' && key !== 'normal' && key !== 'uv') out.deleteAttribute(key);
    }
    out.morphAttributes = {};
    return out;
  });
  if (norm.length === 1) return norm[0]!;
  const merged = mergeGeometries(norm, false);
  if (!merged) throw new Error('jeepGeometry: geometry merge failed');
  for (const g of norm) g.dispose();
  return merged;
}

/**
 * Collects geometry per material slot and emits one merged mesh per slot. Keeps
 * the static bodywork to a handful of draw calls without giving up per-part
 * materials.
 */
export class Assembler {
  private buckets = new Map<string, THREE.BufferGeometry[]>();

  add(material: string, geo: THREE.BufferGeometry, p?: Placement): this {
    if (p) place(geo, p);
    let list = this.buckets.get(material);
    if (!list) this.buckets.set(material, (list = []));
    list.push(geo);
    return this;
  }

  /** Adds the same geometry twice, mirrored about X. Consumes `geo`. */
  addPair(material: string, geo: THREE.BufferGeometry, p?: Placement): this {
    const right = geo;
    const left = geo.clone();
    this.add(material, right, p);
    this.add(material, left, { ...p, mirrorX: true });
    return this;
  }

  /** Merged geometry per material slot; the assembler is left empty. */
  build(): Map<string, THREE.BufferGeometry> {
    const out = new Map<string, THREE.BufferGeometry>();
    for (const [k, list] of this.buckets) out.set(k, mergeAll(list));
    this.buckets.clear();
    return out;
  }

  /** Builds meshes into `target` using the supplied material table. */
  emit(target: THREE.Object3D, materials: Record<string, THREE.Material>, namePrefix = ''): THREE.Mesh[] {
    const meshes: THREE.Mesh[] = [];
    for (const [k, geo] of this.build()) {
      const mat = materials[k];
      if (!mat) throw new Error(`jeepGeometry: no material named "${k}"`);
      const mesh = new THREE.Mesh(geo, mat);
      mesh.name = `${namePrefix}${k}`;
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      target.add(mesh);
      meshes.push(mesh);
    }
    return meshes;
  }
}
