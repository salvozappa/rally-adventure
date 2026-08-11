/**
 * The world surface: one height field, three consumers.
 *
 * `generateTerrain` produces the numbers; this file turns them into
 *
 *   - a chunked, LOD'd visual mesh (frustum culling + vertex-density falloff),
 *   - a Rapier heightfield collider built from the *same* Float32Array, and
 *   - the `TerrainSampler` everything else in the game queries.
 *
 * ── Why the collider and the mesh cannot drift ──────────────────────────────
 * Rapier's heightfield stores a column-major matrix indexed `(row, col)` where
 * the row is the **z** sample and the column is the **x** sample, and it splits
 * every cell along the anti-diagonal joining (ix+1, iz) to (ix, iz+1). Our grid
 * is indexed `ix * samples + iz`, which is exactly that column-major layout, so
 * the array can be handed to Rapier verbatim with scale `(size, 1, size)`.
 * `sampleGridTriangulated` reproduces the same anti-diagonal split, and the
 * mesh below emits its two triangles across the same diagonal. All three agree
 * to float precision — `terrain.test.ts` raycasts 1000 random points to prove
 * it, because a few centimetres of disagreement here is the difference between
 * a car that drives and a car that hovers.
 *
 * ── Chunking ───────────────────────────────────────────────────────────────
 * The map is split into a grid of chunks, each one a self-contained
 * BufferGeometry so the renderer can cull it. Every chunk carries its *full*
 * vertex set; the LOD levels are three pre-built **index** buffers, shared by
 * every chunk, that stride the same vertices at 1, 2 and 4. Swapping LOD is
 * therefore a pointer assignment, not a re-upload, and the whole scheme costs
 * one vertex buffer per chunk instead of three.
 *
 * Chunks at different LODs would leave hairline cracks along their shared edge,
 * so each chunk also carries a downward skirt around its perimeter. It is a few
 * percent more triangles and it removes the problem entirely.
 */

import * as THREE from 'three';
import type RAPIER from '@dimforge/rapier3d-compat';
import {
  generateTerrain,
  sampleGridTriangulated,
  sampleSurface,
  type TerrainField,
} from './heightfield';
import { TerrainMaterial } from './TerrainMaterial';
import { clamp } from './noise';
import {
  GROUP,
  interactionGroups,
  type PhysicsContext,
  type SurfaceKind,
  type TerrainSampler,
} from '../types';

/* -------------------------------------------------------------------------- */
/* Tuning                                                                     */
/* -------------------------------------------------------------------------- */

/** Vertex strides for LOD 0/1/2. Must all divide the chunk's cell count. */
const LOD_STRIDES = [1, 2, 4] as const;

/** Distance (metres, to the nearest point of the chunk) at which LOD steps up. */
const LOD_DISTANCE = [190, 420];
/** Hysteresis so a chunk straddling a threshold does not flicker. */
const LOD_HYSTERESIS = 30;

/** How far the crack-hiding skirt hangs below the chunk edge, in cells. */
const SKIRT_CELLS = 4;

export interface TerrainOptions {
  seed?: number;
  /** Side length of the square world, metres. */
  size?: number;
  /** Height samples per side minus one. Also the collider resolution. */
  resolution?: number;
  /** Chunks per side for the visual mesh. Snapped to a divisor of `resolution`. */
  chunks?: number;
  /**
   * Build the Three.js mesh. Off for headless tooling and tests, which only
   * want the field and the collider.
   */
  visual?: boolean;
}

interface Chunk {
  mesh: THREE.Mesh;
  /** Chunk centre in world XZ. */
  cx: number;
  cz: number;
  lod: number;
}

/* -------------------------------------------------------------------------- */
/* Terrain                                                                    */
/* -------------------------------------------------------------------------- */

export class Terrain implements TerrainSampler {
  readonly object3d: THREE.Object3D;
  readonly halfSize: number;
  /** The raw generated field. Scatter, AI and tooling read this directly. */
  readonly field: TerrainField;
  /** Null when constructed with `visual: false`. */
  readonly material: TerrainMaterial | null = null;

  /** Triangles in the visual mesh with every chunk at LOD 0. */
  readonly triangleCount: number = 0;
  /** Wall-clock cost of `generateTerrain`, milliseconds. */
  readonly generationMs: number;
  /** Wall-clock cost of building geometry + collider, milliseconds. */
  readonly buildMs: number = 0;

  private readonly ctx: PhysicsContext;
  private collider: RAPIER.Collider | null = null;
  private readonly chunks: Chunk[] = [];
  private readonly lodIndices: THREE.BufferAttribute[] = [];
  private chunkRadius = 0;

  private readonly tmpVec = new THREE.Vector3();

  constructor(ctx: PhysicsContext, opts: TerrainOptions = {}) {
    this.ctx = ctx;

    const t0 = nowMs();
    this.field = generateTerrain({
      seed: opts.seed,
      size: opts.size,
      resolution: opts.resolution,
    });
    this.generationMs = nowMs() - t0;
    this.halfSize = this.field.halfSize;

    this.object3d = new THREE.Group();
    this.object3d.name = 'terrain';

    const t1 = nowMs();
    if (opts.visual !== false) {
      this.material = new TerrainMaterial({
        control: this.field.control,
        controlSize: this.field.controlSize,
        worldSize: this.field.size,
        seed: this.field.seed,
      });
      this.triangleCount = this.buildChunks(opts.chunks ?? 16);
    }
    this.buildCollider();
    this.buildMs = nowMs() - t1;
  }

  /* ---------------------------------------------------------------------- */
  /* TerrainSampler                                                         */
  /* ---------------------------------------------------------------------- */

  /**
   * Ground height at world (x, z), using the collider's own triangulation.
   * Outside the world square this clamps to the edge rather than extrapolating.
   */
  heightAt(x: number, z: number): number {
    const f = this.field;
    return sampleGridTriangulated(
      f.heights,
      f.samples,
      (x + f.halfSize) / f.spacing,
      (z + f.halfSize) / f.spacing,
    );
  }

  /**
   * Surface normal at world (x, z), by central difference one grid cell wide.
   *
   * One cell rather than something smaller on purpose: the surface is piecewise
   * linear, so a tighter epsilon just reports the facet you happen to be
   * standing on and makes camera and AI queries jitter as they cross triangles.
   */
  normalAt(x: number, z: number, out?: THREE.Vector3): THREE.Vector3 {
    const e = this.field.spacing;
    const hL = this.heightAt(x - e, z);
    const hR = this.heightAt(x + e, z);
    const hD = this.heightAt(x, z - e);
    const hU = this.heightAt(x, z + e);
    const v = out ?? new THREE.Vector3();
    return v.set(-(hR - hL) / (2 * e), 1, -(hU - hD) / (2 * e)).normalize();
  }

  /** Dominant splat layer at world (x, z) — the same answer the shader draws. */
  surfaceAt(x: number, z: number): SurfaceKind {
    const f = this.field;
    return sampleSurface(f.control, f.controlSize, f.size, x, z);
  }

  /** 0..1 "this is an intended driving route" mask. Used by Scatter and AI. */
  routeAt(x: number, z: number): number {
    const f = this.field;
    return sampleGridTriangulated(
      f.routeMask,
      f.samples,
      (x + f.halfSize) / f.spacing,
      (z + f.halfSize) / f.spacing,
    );
  }

  /**
   * A flat, on-route start position at the south end of the dry wash, facing
   * up the valley toward the jump line and the climb.
   *
   * `heading` is a yaw about +Y in the Three.js sense: the vehicle's forward
   * axis is -Z at heading 0, so `forward = (-sin h, 0, -cos h)`.
   */
  getSpawnPoint(): { position: THREE.Vector3; heading: number } {
    const f = this.field;
    const pan = f.layout.pan;

    // Search the playa for the flattest square metre that is still on-route.
    // Scoring rather than picking the centre outright, because erosion and the
    // droplet pass can leave a gully running right through the middle of it.
    const reach = pan.radius * 0.7;
    const step = Math.max(f.spacing, reach / 24);
    let bestScore = -Infinity;
    let bestX = pan.x;
    let bestZ = pan.z;

    for (let dz = -reach; dz <= reach; dz += step) {
      for (let dx = -reach; dx <= reach; dx += step) {
        if (dx * dx + dz * dz > reach * reach) continue;
        const x = pan.x + dx;
        const z = pan.z + dz;
        const n = this.normalAt(x, z, this.tmpVec);
        const slope = 1 - clamp(n.y, 0, 1);
        const route = this.routeAt(x, z);
        const centreBias = Math.hypot(dx, dz) / reach;
        const score = route * 2 - slope * 14 - centreBias * 0.5;
        if (score > bestScore) {
          bestScore = score;
          bestX = x;
          bestZ = z;
        }
      }
    }

    // Face along the wash. Its centreline meanders, so take the local tangent
    // rather than assuming due north.
    //
    // Chassis forward is local +Z (VehicleTuning's header), so a yaw of
    // `atan2(dirX, dirZ)` turns +Z onto the tangent directly. Negating both
    // components — the old -Z-forward convention — aims the car down the
    // valley's *reverse*, which here means spawning 132 m from the southern
    // boundary range and staring straight into a 570 m wall of rock.
    const ahead = 24;
    const dirX = f.layout.washCenterX(bestZ + ahead) - f.layout.washCenterX(bestZ);
    const dirZ = ahead;
    const len = Math.hypot(dirX, dirZ);
    const heading = Math.atan2(dirX / len, dirZ / len);

    return {
      position: new THREE.Vector3(bestX, this.heightAt(bestX, bestZ) + 1.0, bestZ),
      heading,
    };
  }

  /* ---------------------------------------------------------------------- */
  /* Frame hooks                                                            */
  /* ---------------------------------------------------------------------- */

  /** Distance colour and density for the horizon. Driven by the sky. */
  setFog(color: THREE.Color, density: number): void {
    this.material?.setFog(color, density);
  }

  /** Picks a LOD per chunk. Cheap enough to run every frame. */
  update(cameraPosition: THREE.Vector3): void {
    const maxLod = this.lodIndices.length - 1;
    if (maxLod < 1) return;

    for (const c of this.chunks) {
      const d = Math.max(
        0,
        Math.hypot(cameraPosition.x - c.cx, cameraPosition.z - c.cz) - this.chunkRadius,
      );
      // Push the thresholds outward for a chunk that is already coarse, so a
      // camera hovering on a boundary does not swap buffers every frame.
      const near = LOD_DISTANCE[0]! + (c.lod > 0 ? LOD_HYSTERESIS : 0);
      const far = LOD_DISTANCE[1]! + (c.lod > 1 ? LOD_HYSTERESIS : 0);
      const want = Math.min(maxLod, d < near ? 0 : d < far ? 1 : 2);
      if (want !== c.lod) {
        c.lod = want;
        c.mesh.geometry.setIndex(this.lodIndices[want]!);
      }
    }
  }

  dispose(): void {
    if (this.collider) {
      this.ctx.world.removeCollider(this.collider, false);
      this.collider = null;
    }
    for (const c of this.chunks) c.mesh.geometry.dispose();
    this.chunks.length = 0;
    this.lodIndices.length = 0;
    this.object3d.clear();
    this.material?.dispose();
  }

  /* ---------------------------------------------------------------------- */
  /* Collider                                                               */
  /* ---------------------------------------------------------------------- */

  private buildCollider(): void {
    const { rapier, world } = this.ctx;
    const f = this.field;

    // nrows = z cells, ncols = x cells; our `ix * samples + iz` indexing is
    // already the column-major (row = z, col = x) matrix Rapier wants.
    const desc = rapier.ColliderDesc.heightfield(f.cells, f.cells, f.heights, {
      x: f.size,
      y: 1,
      z: f.size,
    })
      .setFriction(1.0)
      // Zero bounce: the tyre model supplies compliance through the suspension,
      // and any restitution here turns a hard landing into a pogo stick.
      .setRestitution(0.0)
      .setCollisionGroups(
        interactionGroups(GROUP.TERRAIN, GROUP.CHASSIS | GROUP.PROP | GROUP.WHEEL_RAY),
      );

    this.collider = world.createCollider(desc);
  }

  /* ---------------------------------------------------------------------- */
  /* Mesh                                                                   */
  /* ---------------------------------------------------------------------- */

  private buildChunks(requestedChunks: number): number {
    const f = this.field;
    const perSide = pickChunkCount(f.cells, requestedChunks);
    const cc = f.cells / perSide;
    const strides = LOD_STRIDES.filter((s) => cc % s === 0);

    const vertsPerSide = cc + 1;
    const gridVerts = vertsPerSide * vertsPerSide;
    const ringVerts = 4 * vertsPerSide;
    const totalVerts = gridVerts + ringVerts;

    for (const s of strides) this.lodIndices.push(buildIndexBuffer(cc, s, totalVerts));

    const chunkSize = cc * f.spacing;
    this.chunkRadius = chunkSize * Math.SQRT1_2;
    const skirtDepth = SKIRT_CELLS * f.spacing + 1;

    for (let czi = 0; czi < perSide; czi++) {
      for (let cxi = 0; cxi < perSide; cxi++) {
        const geom = this.buildChunkGeometry(cxi * cc, czi * cc, cc, totalVerts, skirtDepth);
        geom.setIndex(this.lodIndices[0]!);

        const mesh = new THREE.Mesh(geom, this.material!);
        mesh.name = `terrain-${cxi}-${czi}`;
        mesh.matrixAutoUpdate = false;
        mesh.receiveShadow = true;
        // A 1 km heightfield in the shadow map would spend the whole depth
        // range on terrain and leave nothing for the car. Baked AO covers the
        // macro occlusion instead.
        mesh.castShadow = false;
        this.object3d.add(mesh);

        this.chunks.push({
          mesh,
          cx: (cxi * cc + cc * 0.5) * f.spacing - f.halfSize,
          cz: (czi * cc + cc * 0.5) * f.spacing - f.halfSize,
          lod: 0,
        });
      }
    }

    const idx = this.lodIndices[0]!;
    return (idx.count / 3) * this.chunks.length;
  }

  private buildChunkGeometry(
    ix0: number,
    iz0: number,
    cc: number,
    totalVerts: number,
    skirtDepth: number,
  ): THREE.BufferGeometry {
    const f = this.field;
    const n = f.samples;
    const heights = f.heights;
    const occlusion = f.occlusion;
    const spacing = f.spacing;
    const half = f.halfSize;
    const vps = cc + 1;

    const position = new Float32Array(totalVerts * 3);
    // Int16 normalised: 1/32767 of a unit is far below what Lambert shading can
    // show, and it halves the attribute against Float32.
    const normal = new Int16Array(totalVerts * 3);
    const color = new Uint8Array(totalVerts * 3);

    const write = (v: number, ix: number, iz: number, yOffset: number): void => {
      const i = ix * n + iz;
      const x = ix * spacing - half;
      const z = iz * spacing - half;
      const y = heights[i]!;

      const o3 = v * 3;
      position[o3] = x;
      position[o3 + 1] = y + yOffset;
      position[o3 + 2] = z;

      // Central difference on the shared grid, so normals match exactly across
      // chunk seams — recomputing per chunk from local data would not.
      const xl = ix > 0 ? ix - 1 : 0;
      const xr = ix < n - 1 ? ix + 1 : n - 1;
      const zl = iz > 0 ? iz - 1 : 0;
      const zr = iz < n - 1 ? iz + 1 : n - 1;
      const dhx = (heights[xr * n + iz]! - heights[xl * n + iz]!) / ((xr - xl) * spacing);
      const dhz = (heights[ix * n + zr]! - heights[ix * n + zl]!) / ((zr - zl) * spacing);
      const inv = 1 / Math.sqrt(dhx * dhx + 1 + dhz * dhz);
      normal[o3] = Math.round(clamp(-dhx * inv, -1, 1) * 32767);
      normal[o3 + 1] = Math.round(inv * 32767);
      normal[o3 + 2] = Math.round(clamp(-dhz * inv, -1, 1) * 32767);

      const ao = Math.round(clamp(occlusion[i]!, 0, 1) * 255);
      color[o3] = ao;
      color[o3 + 1] = ao;
      color[o3 + 2] = ao;
    };

    for (let a = 0; a <= cc; a++) {
      for (let b = 0; b <= cc; b++) {
        write(a * vps + b, ix0 + a, iz0 + b, 0);
      }
    }

    // Perimeter skirt, in the edge order the shared index buffers expect:
    // 0 = -X, 1 = +X, 2 = -Z, 3 = +Z.
    const ringBase = vps * vps;
    for (let j = 0; j <= cc; j++) {
      write(ringBase + 0 * vps + j, ix0, iz0 + j, -skirtDepth);
      write(ringBase + 1 * vps + j, ix0 + cc, iz0 + j, -skirtDepth);
      write(ringBase + 2 * vps + j, ix0 + j, iz0, -skirtDepth);
      write(ringBase + 3 * vps + j, ix0 + j, iz0 + cc, -skirtDepth);
    }

    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(position, 3));
    geom.setAttribute('normal', new THREE.BufferAttribute(normal, 3, true));
    geom.setAttribute('color', new THREE.BufferAttribute(color, 3, true));
    geom.computeBoundingSphere();
    geom.computeBoundingBox();
    return geom;
  }
}

/* -------------------------------------------------------------------------- */
/* Geometry helpers                                                           */
/* -------------------------------------------------------------------------- */

/**
 * Largest chunk count no bigger than `requested` that divides the grid evenly
 * and leaves a cell count every LOD stride divides.
 */
function pickChunkCount(cells: number, requested: number): number {
  const maxStride = LOD_STRIDES[LOD_STRIDES.length - 1]!;
  for (let c = Math.max(1, Math.min(requested, cells)); c >= 1; c--) {
    if (cells % c === 0 && cells / c % maxStride === 0) return c;
  }
  return 1;
}

/**
 * Index buffer for one LOD level, shared by every chunk.
 *
 * The grid quads are split along the anti-diagonal (a+s, b) — (a, b+s), which
 * is the diagonal Rapier's heightfield uses. At LOD 0 the rendered surface is
 * therefore the collider surface, triangle for triangle.
 */
function buildIndexBuffer(cc: number, stride: number, totalVerts: number): THREE.BufferAttribute {
  const vps = cc + 1;
  const q = cc / stride;
  const ringBase = vps * vps;
  const count = q * q * 6 + 4 * q * 6;
  const arr = totalVerts > 65535 ? new Uint32Array(count) : new Uint16Array(count);
  let w = 0;

  const gi = (a: number, b: number): number => a * vps + b;

  for (let a = 0; a < cc; a += stride) {
    for (let b = 0; b < cc; b += stride) {
      const v00 = gi(a, b);
      const v01 = gi(a, b + stride);
      const v10 = gi(a + stride, b);
      const v11 = gi(a + stride, b + stride);
      // Wound counter-clockwise seen from +Y, i.e. facing up.
      arr[w++] = v00;
      arr[w++] = v01;
      arr[w++] = v10;
      arr[w++] = v10;
      arr[w++] = v01;
      arr[w++] = v11;
    }
  }

  // Skirt. Edges 0 (-X) and 3 (+Z) wind one way, 1 (+X) and 2 (-Z) the other,
  // so every skirt quad faces outward and survives back-face culling.
  const topOf = (edge: number, j: number): number =>
    edge === 0 ? gi(0, j) : edge === 1 ? gi(cc, j) : edge === 2 ? gi(j, 0) : gi(j, cc);

  for (let edge = 0; edge < 4; edge++) {
    const flip = edge === 1 || edge === 2;
    for (let j = 0; j < cc; j += stride) {
      const t0 = topOf(edge, j);
      const t1 = topOf(edge, j + stride);
      const b0 = ringBase + edge * vps + j;
      const b1 = ringBase + edge * vps + j + stride;
      if (flip) {
        arr[w++] = t0;
        arr[w++] = t1;
        arr[w++] = b0;
        arr[w++] = t1;
        arr[w++] = b1;
        arr[w++] = b0;
      } else {
        arr[w++] = t0;
        arr[w++] = b0;
        arr[w++] = t1;
        arr[w++] = t1;
        arr[w++] = b0;
        arr[w++] = b1;
      }
    }
  }

  return new THREE.BufferAttribute(arr, 1);
}

function nowMs(): number {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}
