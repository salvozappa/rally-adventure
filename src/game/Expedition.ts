import * as THREE from 'three';
import type { TerrainSampler, VehicleState } from '../types';

export interface Waypoint {
  id: number;
  name: string;
  position: THREE.Vector3;
  radius: number;
  found: boolean;
}

export interface ExpeditionStats {
  found: number;
  total: number;
  distanceKm: number;
  bestAirtime: number;
  totalAirtime: number;
  bestSpeedKmh: number;
  longestDriftMeters: number;
  elapsed: number;
}

/**
 * A light objective layer: scattered viewpoints to discover across the map,
 * plus the driving stats that make free-roam feel like it's keeping score.
 *
 * Deliberately not a race. The brief is an off-road *adventure*, so the reward
 * loop is exploration and stunt-driving rather than lap times — and it stays
 * out of the way of proper level design later.
 */
export class Expedition {
  readonly waypoints: Waypoint[] = [];
  readonly object3d = new THREE.Group();

  private stats: ExpeditionStats = {
    found: 0,
    total: 0,
    distanceKm: 0,
    bestAirtime: 0,
    totalAirtime: 0,
    bestSpeedKmh: 0,
    longestDriftMeters: 0,
    elapsed: 0,
  };

  private lastPos = new THREE.Vector3();
  private hasLastPos = false;
  private airborneTime = 0;
  private driftDistance = 0;
  private markers: THREE.Mesh[] = [];
  private beacons: THREE.Object3D[] = [];

  /** Fired when a waypoint is discovered. */
  onDiscover: ((w: Waypoint, stats: ExpeditionStats) => void) | null = null;
  /** Fired when a notable stunt completes. */
  onStunt: ((kind: string, value: number) => void) | null = null;

  constructor(
    private terrain: TerrainSampler,
    opts: { count?: number; seed?: number } = {},
  ) {
    const count = opts.count ?? 8;
    const rng = mulberry32(opts.seed ?? 0xC0FFEE);
    this.object3d.name = 'expedition';

    // Place waypoints on high, visible ground spread around the map, so they
    // double as landmarks you can navigate by from a distance.
    const half = this.terrain.halfSize * 0.78;
    const names = [
      'RIDGE LOOKOUT', 'DRY FORD', 'BOULDER FIELD', 'HIGH SADDLE',
      'CANYON MOUTH', 'BURNT PINES', 'THE STAIRCASE', 'FAR BEACON',
      'OLD QUARRY', 'WIND GAP', 'SUNKEN TRAIL', 'EAGLE POINT',
    ];

    for (let i = 0; i < count; i++) {
      // Golden-angle spiral gives good angular spread without clumping.
      const ang = i * 2.399963 + rng() * 0.6;
      const rad = half * (0.32 + 0.68 * Math.sqrt((i + rng()) / count));
      const p = this.pickVisibleSpot(Math.cos(ang) * rad, Math.sin(ang) * rad, rng);
      this.waypoints.push({
        id: i,
        name: names[i % names.length],
        position: p,
        radius: 11,
        found: false,
      });
      this.buildMarker(p, i);
    }
    this.stats.total = this.waypoints.length;
  }

  /** Nudge a candidate position toward locally high, gently-sloped ground. */
  private pickVisibleSpot(x: number, z: number, rng: () => number): THREE.Vector3 {
    let bx = x;
    let bz = z;
    let best = -Infinity;
    const n = new THREE.Vector3();
    for (let i = 0; i < 48; i++) {
      const cx = x + (rng() - 0.5) * 90;
      const cz = z + (rng() - 0.5) * 90;
      if (Math.abs(cx) > this.terrain.halfSize - 30 || Math.abs(cz) > this.terrain.halfSize - 30) continue;
      const h = this.terrain.heightAt(cx, cz);
      this.terrain.normalAt(cx, cz, n);
      // Reward height, punish steepness — a lookout you can't park on is no use.
      const flatness = THREE.MathUtils.clamp(n.y, 0, 1);
      const score = h + flatness * 26;
      if (flatness > 0.86 && score > best) {
        best = score;
        bx = cx;
        bz = cz;
      }
    }
    return new THREE.Vector3(bx, this.terrain.heightAt(bx, bz), bz);
  }

  /** A striped pole with a floating ring — readable from a long way off. */
  private buildMarker(p: THREE.Vector3, id: number): void {
    const group = new THREE.Group();
    group.position.copy(p);

    const poleH = 6;
    const bands = 6;
    for (let i = 0; i < bands; i++) {
      const seg = new THREE.Mesh(
        new THREE.CylinderGeometry(0.11, 0.12, poleH / bands, 6),
        new THREE.MeshLambertMaterial({ color: i % 2 ? 0xdedad0 : 0xc4432a }),
      );
      seg.position.y = (i + 0.5) * (poleH / bands);
      seg.castShadow = true;
      group.add(seg);
    }

    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(1.5, 0.14, 6, 20),
      new THREE.MeshBasicMaterial({ color: 0xffc94a, fog: true }),
    );
    ring.position.y = poleH + 1.4;
    ring.rotation.x = Math.PI / 2;
    ring.userData.spin = true;
    group.add(ring);
    this.beacons.push(ring);

    group.userData.waypointId = id;
    this.markers.push(ring);
    this.object3d.add(group);
  }

  update(dt: number, s: VehicleState): void {
    this.stats.elapsed += dt;

    // --- odometer ---
    if (this.hasLastPos) {
      this.stats.distanceKm += this.lastPos.distanceTo(s.position) / 1000;
    }
    this.lastPos.copy(s.position);
    this.hasLastPos = true;

    const kmh = s.speed * 3.6;
    if (kmh > this.stats.bestSpeedKmh) this.stats.bestSpeedKmh = kmh;

    // --- airtime ---
    if (s.airborne) {
      this.airborneTime += dt;
    } else if (this.airborneTime > 0) {
      this.stats.totalAirtime += this.airborneTime;
      if (this.airborneTime > this.stats.bestAirtime) {
        this.stats.bestAirtime = this.airborneTime;
        if (this.airborneTime > 1.0) this.onStunt?.('AIRTIME', this.airborneTime);
      }
      this.airborneTime = 0;
    }

    // --- drift: sustained lateral slip while moving at pace ---
    const drifting = !s.airborne && s.speed > 7 && this.slipAngleOf(s) > 0.28;
    if (drifting) {
      this.driftDistance += s.speed * dt;
    } else if (this.driftDistance > 0) {
      if (this.driftDistance > this.stats.longestDriftMeters) {
        this.stats.longestDriftMeters = this.driftDistance;
        if (this.driftDistance > 25) this.onStunt?.('DRIFT', this.driftDistance);
      }
      this.driftDistance = 0;
    }

    // --- discovery ---
    for (const w of this.waypoints) {
      if (w.found) continue;
      const dx = s.position.x - w.position.x;
      const dz = s.position.z - w.position.z;
      if (dx * dx + dz * dz < w.radius * w.radius) {
        w.found = true;
        this.stats.found++;
        this.onDiscover?.(w, this.stats);
        this.retireMarker(w.id);
      }
    }

    // Slow spin on the beacon rings, so they catch the eye at distance.
    for (const b of this.beacons) b.rotation.z += dt * 0.9;
  }

  private retireMarker(id: number): void {
    const grp = this.object3d.children.find((c) => c.userData.waypointId === id);
    if (!grp) return;
    grp.traverse((o) => {
      const m = (o as THREE.Mesh).material as THREE.MeshLambertMaterial | undefined;
      if (m && 'color' in m) m.color.multiplyScalar(0.45);
    });
  }

  /** Angle between where the car points and where it's actually going. */
  private slipAngleOf(s: VehicleState): number {
    if (s.speed < 1) return 0;
    const fwd = new THREE.Vector3(0, 0, 1).applyQuaternion(s.quaternion);
    fwd.y = 0;
    const vel = s.velocity.clone();
    vel.y = 0;
    if (vel.lengthSq() < 1e-4 || fwd.lengthSq() < 1e-6) return 0;
    fwd.normalize();
    vel.normalize();
    return Math.acos(THREE.MathUtils.clamp(fwd.dot(vel), -1, 1));
  }

  /** Bearing and distance to the nearest undiscovered waypoint, for the HUD compass. */
  nearestTarget(from: THREE.Vector3): { waypoint: Waypoint; distance: number } | null {
    let best: Waypoint | null = null;
    let bestD = Infinity;
    for (const w of this.waypoints) {
      if (w.found) continue;
      const d = from.distanceTo(w.position);
      if (d < bestD) {
        bestD = d;
        best = w;
      }
    }
    return best ? { waypoint: best, distance: bestD } : null;
  }

  getStats(): Readonly<ExpeditionStats> {
    return this.stats;
  }

  dispose(): void {
    this.object3d.traverse((o) => {
      const m = o as THREE.Mesh;
      m.geometry?.dispose();
      const mat = m.material;
      if (Array.isArray(mat)) mat.forEach((x) => x.dispose());
      else mat?.dispose();
    });
    this.object3d.clear();
  }
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
