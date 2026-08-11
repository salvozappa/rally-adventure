/**
 * DriveFx.ts — every particle the vehicle throws at the world.
 *
 * One orchestrator over three `ParticleSystem` pools and one `TrackDecals`
 * ribbon set. Three pools rather than one because blending mode and integration
 * mode are baked into the material:
 *
 *   air     alpha-blended BILLOW   dust, tyre smoke, rooster tail, exhaust,
 *                                  landing puffs, water mist
 *   debris  alpha-blended BALLISTIC clods, grass clippings, splash droplets
 *   sparks  additive      STREAK    rock scrapes
 *
 * Everything is driven from the per-wheel telemetry in `VehicleState`. Nothing
 * here reads the input device or the physics internals: if the tyre model says
 * a wheel is spinning on sand under 6 kN of load, that — and only that — is
 * what decides how much pale dust comes off it. The result is that the picture
 * always agrees with the handling, which is the whole point.
 *
 * The tuning bias throughout is *under*-doing it. Dust reads as speed because
 * it is long-lived and low-contrast, not because it is dense; the failure mode
 * of vehicle FX is a smoke machine following the car around.
 */

import * as THREE from 'three';
import {
  ParticleSystem,
  MODE_BILLOW,
  MODE_BALLISTIC,
  MODE_STREAK,
  type ParticleDesc,
} from './ParticleSystem';
import { ATLAS_DUST, ATLAS_SMOKE, ATLAS_CLOD, ATLAS_SPARK } from './fxTextures';
import { TrackDecals, type TrackDecalOptions } from './TrackDecals';
import type { SurfaceKind, TerrainSampler, VehicleState, WheelState } from '../types';

/* ------------------------------------------------------------------ *
 * Surfaces
 * ------------------------------------------------------------------ */

/** How a surface behaves when a tyre works it. */
interface SurfaceFx {
  /** Airborne colour. Linear space — the shader does no conversion. */
  dust: THREE.Color;
  /** Colour of the solid bits that get thrown. */
  chunk: THREE.Color;
  /** Relative volume of airborne dust, 0..1.3. */
  dustAmount: number;
  /** Relative volume of ballistic debris, 0..1.4. */
  chunkAmount: number;
  /** Tendency to make grey rubber smoke instead of dust. */
  smoke: number;
  /** Air drag on the plume, 1/s. High = the dust drops out quickly. */
  drag: number;
  /** Buoyancy, m/s^2. */
  rise: number;
  /** Base plume lifetime, seconds. */
  life: number;
  /** Base sprite size at death, metres. */
  spread: number;
}

/**
 * Palette entry. `new THREE.Color(hex)` already interprets the literal as sRGB
 * and lands in the renderer's linear working space, which is exactly what the
 * particle shader wants — converting a second time would darken every colour
 * here by a factor of five.
 */
function srgb(hex: number): THREE.Color {
  return new THREE.Color(hex);
}

const SURFACE: Record<SurfaceKind, SurfaceFx> = {
  // The default off-road surface: warm ochre, hangs for ages, throws lumps.
  dirt: {
    dust: srgb(0xc7a87e), chunk: srgb(0x4d3720),
    dustAmount: 1.0, chunkAmount: 0.85, smoke: 0.05,
    drag: 0.9, rise: 1.5, life: 3.4, spread: 5.4,
  },
  // Loose stone: less fine material in the air, far more of it on the ground.
  gravel: {
    dust: srgb(0xaea395), chunk: srgb(0x6a6257),
    dustAmount: 0.6, chunkAmount: 1.05, smoke: 0.14,
    drag: 1.2, rise: 1.0, life: 2.6, spread: 4.4,
  },
  // Sand is the biggest, palest, longest-hanging plume in the game.
  sand: {
    dust: srgb(0xe4d2a6), chunk: srgb(0xc2a469),
    dustAmount: 1.25, chunkAmount: 0.5, smoke: 0.02,
    drag: 0.78, rise: 1.8, life: 4.2, spread: 6.2,
  },
  // Grass barely dusts; what it does is fling clippings.
  grass: {
    dust: srgb(0x8a9463), chunk: srgb(0x4f7d27),
    dustAmount: 0.3, chunkAmount: 0.95, smoke: 0.03,
    drag: 1.4, rise: 0.7, life: 2.0, spread: 2.8,
  },
  // Wet: almost nothing airborne, everything thrown.
  mud: {
    dust: srgb(0x6d5840), chunk: srgb(0x2a1e11),
    dustAmount: 0.2, chunkAmount: 1.4, smoke: 0.0,
    drag: 1.65, rise: 0.4, life: 1.6, spread: 2.2,
  },
  // Rock makes rubber smoke, not dust.
  rock: {
    dust: srgb(0xa39b91), chunk: srgb(0x615b52),
    dustAmount: 0.18, chunkAmount: 0.25, smoke: 1.0,
    drag: 1.25, rise: 1.1, life: 2.4, spread: 3.6,
  },
  snow: {
    dust: srgb(0xe8effa), chunk: srgb(0xcfe0f2),
    dustAmount: 0.95, chunkAmount: 0.7, smoke: 0.02,
    drag: 1.05, rise: 1.3, life: 3.1, spread: 5.0,
  },
};

/** Hot rubber. Deliberately neutral grey — brown smoke reads as dust. */
const TYRE_SMOKE = srgb(0xb8b5b0);
/** Cold diesel. Slightly blue so it separates from the dust behind it. */
const EXHAUST_COLD = srgb(0x8e959c);
/** A loaded engine soots up. */
const EXHAUST_HOT = srgb(0x5a5854);
/** Aerated water. */
const SPLASH = srgb(0xdfe9ea);
const SPARK = srgb(0xffb055);

/* ------------------------------------------------------------------ *
 * Options
 * ------------------------------------------------------------------ */

export interface DriveFxOptions {
  /** Ground queries for decals, splashes and plume ground planes. */
  terrain?: TerrainSampler;
  /** Total particle budget across all three pools. */
  maxParticles?: number;
  /** World Y of the water plane. Contacts below it splash instead of dust. */
  waterLevel?: number;
  /** Rolling radius, metres. Only used when there is no terrain sampler. */
  wheelRadius?: number;
  /** Exhaust tip in chassis-local space. */
  exhaustOffset?: THREE.Vector3;
  /** Global volume knob, 0..2. 1 is the tuned default. */
  intensity?: number;
  /** `false` to skip the ribbon meshes entirely. */
  tracks?: boolean | TrackDecalOptions;
}

/** Everything that can be switched off independently, for the preview rig. */
export type FxChannel =
  | 'dust'
  | 'clods'
  | 'rooster'
  | 'smoke'
  | 'tracks'
  | 'landing'
  | 'exhaust'
  | 'splash'
  | 'sparks';

const CHANNELS: FxChannel[] = [
  'dust', 'clods', 'rooster', 'smoke', 'tracks', 'landing', 'exhaust', 'splash', 'sparks',
];

/* ------------------------------------------------------------------ *
 * Per-wheel bookkeeping
 * ------------------------------------------------------------------ */

interface WheelFx {
  /** Fractional particle carry, so low rates still emit at the right average. */
  dustAcc: number;
  chunkAcc: number;
  smokeAcc: number;
  splashAcc: number;
  prevGrounded: boolean;
  prevCompression: number;
  /** Seconds until this wheel may fire another landing puff. */
  thumpCooldown: number;
}

const NOMINAL_LOAD = 4200; // N — a quarter of a laden 4x4, near enough.

/* ------------------------------------------------------------------ */

export class DriveFx {
  readonly tracks: TrackDecals | null;

  private readonly scene: THREE.Scene;
  private readonly terrain: TerrainSampler | undefined;
  private readonly waterLevel: number;
  private readonly wheelRadius: number;
  private readonly exhaustOffset: THREE.Vector3;

  private readonly air: ParticleSystem;
  private readonly debris: ParticleSystem;
  private readonly sparks: ParticleSystem;

  private readonly enabled = new Set<FxChannel>(CHANNELS);
  private intensity: number;

  private readonly wheels: WheelFx[] = [];
  private readonly contactY: number[] = [];

  private time = 0;
  private prevAirborne = false;
  private prevVelY = 0;
  private prevGear = 1;
  private prevRpm = 0;
  private exhaustAcc = 0;
  private shiftPuff = 0;
  private lastEmitted = 0;

  // Scratch — the update path allocates nothing.
  private readonly d: ParticleDesc = {
    x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0,
    r: 1, g: 1, b: 1, cell: ATLAS_DUST,
    life: 1, size0: 0.2, size1: 1, rot: 0, rotSpeed: 0,
    drag: 1, gravity: 0, opacity: 0.2, groundY: 0, mode: MODE_BILLOW,
  };
  private readonly fwd = new THREE.Vector3(0, 0, 1);
  private readonly travel = new THREE.Vector3(0, 0, 1);
  private readonly lateral = new THREE.Vector3(1, 0, 0);
  private readonly outward = new THREE.Vector3(1, 0, 0);
  private readonly tmp = new THREE.Vector3();
  private readonly col = new THREE.Color();

  constructor(scene: THREE.Scene, opts: DriveFxOptions = {}) {
    this.scene = scene;
    this.terrain = opts.terrain;
    this.waterLevel = opts.waterLevel ?? -Infinity;
    this.wheelRadius = opts.wheelRadius ?? 0.4;
    this.exhaustOffset = (opts.exhaustOffset ?? new THREE.Vector3(0.46, -0.26, -2.05)).clone();
    this.intensity = opts.intensity ?? 1;

    const budget = Math.max(256, opts.maxParticles ?? 3400);

    // Dust dominates: it is the long-lived pool and the one that carries the
    // whole read. Debris is short-lived, sparks are rare.
    this.air = new ParticleSystem({
      name: 'fx.air',
      capacity: Math.round(budget * 0.66),
      blending: THREE.NormalBlending,
      turbulence: 0.34,
      fadeIn: 0.05,
      softFade: 1.15,
      softDepth: 1.4,
      nearFade: [0.4, 2.2],
      renderOrder: 12,
    });

    this.debris = new ParticleSystem({
      name: 'fx.debris',
      capacity: Math.round(budget * 0.26),
      blending: THREE.NormalBlending,
      turbulence: 0,
      fadeIn: 0.02,
      softFade: 0.12,
      softDepth: 0.25,
      nearFade: [0.2, 0.7],
      bounce: 0.34,
      friction: 0.5,
      renderOrder: 13,
    });

    this.sparks = new ParticleSystem({
      name: 'fx.sparks',
      capacity: Math.max(64, Math.round(budget * 0.08)),
      blending: THREE.AdditiveBlending,
      turbulence: 0,
      fadeIn: 0.02,
      softFade: 0.05,
      softDepth: 0.2,
      nearFade: [0.2, 0.7],
      streak: 1.6,
      renderOrder: 14,
    });

    scene.add(this.air.object3d, this.debris.object3d, this.sparks.object3d);

    const trackOpts = opts.tracks;
    if (trackOpts === false) {
      this.tracks = null;
    } else {
      this.tracks = new TrackDecals({
        terrain: opts.terrain,
        ...(typeof trackOpts === 'object' ? trackOpts : {}),
      });
      scene.add(this.tracks.object3d);
    }
  }

  /* ---------------- public API ---------------- */

  /**
   * Advance every effect one rendered frame.
   *
   * Safe to call with a paused or absent vehicle: a zero `dt` is a no-op and a
   * wheel list of any length is tolerated.
   */
  update(dt: number, state: VehicleState): void {
    if (!(dt > 0)) return;
    // A frame hitch must not dump a second's worth of particles in one go.
    const step = Math.min(dt, 0.05);
    this.time += step;

    const n = state.wheels.length;
    while (this.wheels.length < n) {
      this.wheels.push({
        dustAcc: 0, chunkAcc: 0, smokeAcc: 0, splashAcc: 0,
        prevGrounded: true, prevCompression: 0, thumpCooldown: 0,
      });
      this.contactY.push(0);
    }

    this.frame(step, state);

    this.air.update(step);
    this.debris.update(step);
    this.sparks.update(step);
  }

  /**
   * A one-off hit: landing, a wheel dropping into a rut, a body panel finding a
   * rock. `strength` is roughly the impact speed in m/s over 10, clamped to a
   * sane range internally; 0.3 is a firm bump, 1 is a heavy landing.
   */
  impact(position: THREE.Vector3, strength: number, surface: SurfaceKind): void {
    const s = THREE.MathUtils.clamp(strength, 0, 2.5);
    if (s < 0.05) return;
    const sf = SURFACE[surface] ?? SURFACE.dirt;
    const groundY = this.groundAt(position.x, position.z, position.y);

    if (position.y <= this.waterLevel + 0.25 && this.waterLevel > -Infinity) {
      this.splashBurst(position.x, this.waterLevel, position.z, s * 2.2, groundY);
      return;
    }

    if (this.enabled.has('landing')) {
      // A landing puff is a flat radial ring, not a ball: the air has nowhere
      // to go but sideways when a tyre slams into the ground.
      const count = Math.round((5 + 13 * s) * sf.dustAmount * this.intensity);
      for (let i = 0; i < count; i++) {
        const a = Math.random() * Math.PI * 2;
        const speed = (1.4 + 3.4 * s) * (0.45 + Math.random() * 0.85);
        const d = this.d;
        d.mode = MODE_BILLOW;
        d.cell = ATLAS_DUST;
        this.tint(sf.dust, 0.14);
        d.x = position.x + Math.cos(a) * 0.22;
        d.y = groundY + 0.1 + Math.random() * 0.16;
        d.z = position.z + Math.sin(a) * 0.22;
        d.vx = Math.cos(a) * speed;
        d.vz = Math.sin(a) * speed;
        d.vy = 0.5 + Math.random() * 1.5 * s;
        d.life = sf.life * (0.55 + Math.random() * 0.5);
        d.size0 = 0.25 + Math.random() * 0.3;
        d.size1 = sf.spread * (0.5 + 0.55 * s) * (0.7 + Math.random() * 0.7);
        d.rot = Math.random() * 6.283;
        d.rotSpeed = (Math.random() - 0.5) * 0.6;
        d.drag = sf.drag * 1.15;
        d.gravity = sf.rise;
        d.opacity = (0.28 + 0.2 * s) * this.intensity;
        d.groundY = groundY;
        this.air.emit(d);
        this.lastEmitted++;
      }
    }

    if (this.enabled.has('clods') && sf.chunkAmount > 0.2) {
      const count = Math.round(4 + 14 * s * sf.chunkAmount);
      for (let i = 0; i < count; i++) {
        const a = Math.random() * Math.PI * 2;
        const speed = (1.6 + 4.5 * s) * (0.35 + Math.random());
        this.emitChunk(
          sf,
          position.x, groundY + 0.08, position.z,
          Math.cos(a) * speed, 1.6 + Math.random() * 4.2 * s, Math.sin(a) * speed,
          groundY,
          0.65 + Math.random() * 0.5,
        );
      }
    }

    // Steel on stone. Only rock, only a real hit.
    if (this.enabled.has('sparks') && surface === 'rock' && s > 0.45) {
      this.sparkBurst(position.x, groundY + 0.06, position.z, s, null);
    }
  }

  /**
   * Hand over the scene depth buffer for true soft particles. Optional — the
   * pools fall back to a vertical fade, which handles the artefact that
   * actually reads (a quad slicing into the ground) at zero cost.
   */
  setDepthTexture(depth: THREE.Texture | null, near = 0.1, far = 1000): void {
    this.air.setDepthTexture(depth, near, far);
    this.debris.setDepthTexture(depth, near, far);
    this.sparks.setDepthTexture(depth, near, far);
  }

  /** Global volume, 0..2. */
  setIntensity(v: number): void {
    this.intensity = THREE.MathUtils.clamp(v, 0, 2);
  }

  setChannel(channel: FxChannel, on: boolean): void {
    if (on) this.enabled.add(channel);
    else this.enabled.delete(channel);
    if (channel === 'tracks' && this.tracks) this.tracks.visible = on;
  }

  isChannelOn(channel: FxChannel): boolean {
    return this.enabled.has(channel);
  }

  /** Live particles, per pool and total. For the HUD and the budget report. */
  get stats(): {
    air: number; debris: number; sparks: number; particles: number;
    capacity: number; trackSegments: number; emittedLastFrame: number;
  } {
    const a = this.air.alive;
    const d = this.debris.alive;
    const s = this.sparks.alive;
    return {
      air: a, debris: d, sparks: s, particles: a + d + s,
      capacity: this.air.capacity + this.debris.capacity + this.sparks.capacity,
      trackSegments: this.tracks ? this.tracks.liveSegments : 0,
      emittedLastFrame: this.lastEmitted,
    };
  }

  /** Wipe everything. Call on respawn so old dust doesn't teleport with you. */
  clear(): void {
    this.air.clear();
    this.debris.clear();
    this.sparks.clear();
    this.tracks?.clear();
  }

  dispose(): void {
    this.air.dispose();
    this.debris.dispose();
    this.sparks.dispose();
    this.tracks?.dispose();
    // The pools remove themselves; this keeps the scene reference from
    // outliving us if a caller held on to it.
    void this.scene;
  }

  /* ---------------- per-frame ---------------- */

  private frame(dt: number, state: VehicleState): void {
    this.lastEmitted = 0;

    const speed = state.speed;
    const vel = state.velocity;

    // Convention-free frame: at speed the direction of travel is the honest
    // reference; below walking pace fall back to the chassis axis so a burnout
    // from rest still throws its rooster tail the right way.
    this.fwd.set(0, 0, 1).applyQuaternion(state.quaternion);
    const horiz = Math.hypot(vel.x, vel.z);
    if (horiz > 0.6) {
      this.travel.set(vel.x / horiz, 0, vel.z / horiz);
    } else {
      this.travel.set(this.fwd.x, 0, this.fwd.z);
      if (this.travel.lengthSq() < 1e-6) this.travel.set(0, 0, 1);
      this.travel.normalize();
    }
    this.lateral.set(this.travel.z, 0, -this.travel.x);

    for (let i = 0; i < state.wheels.length; i++) {
      const w = state.wheels[i]!;
      const fx = this.wheels[i]!;
      const groundY = this.groundAt(w.position.x, w.position.z, w.position.y - this.wheelRadius);
      this.contactY[i] = groundY;

      if (fx.thumpCooldown > 0) fx.thumpCooldown -= dt;

      if (w.grounded) {
        this.wheelEmission(dt, i, w, fx, state, speed, groundY);
        this.thump(dt, w, fx, groundY);
      } else {
        fx.dustAcc = 0;
        fx.chunkAcc = 0;
        fx.smokeAcc = 0;
        fx.splashAcc = 0;
      }
      fx.prevGrounded = w.grounded;
      fx.prevCompression = w.compression;
    }

    if (this.tracks && this.enabled.has('tracks')) {
      this.tracks.update(dt, state.wheels, this.contactY, speed);
    }

    this.exhaust(dt, state);

    // Whole-vehicle landing: kick a puff under every wheel that just found
    // ground after real air time, scaled by the vertical speed we killed.
    if (this.prevAirborne && !state.airborne) {
      const strength = THREE.MathUtils.clamp(-this.prevVelY / 11, 0, 1.6);
      if (strength > 0.12) {
        for (let i = 0; i < state.wheels.length; i++) {
          const w = state.wheels[i]!;
          if (!w.grounded) continue;
          this.tmp.set(w.position.x, this.contactY[i]!, w.position.z);
          this.impact(this.tmp, strength, w.surface);
          this.wheels[i]!.thumpCooldown = 0.4;
        }
      }
    }
    this.prevAirborne = state.airborne;
    this.prevVelY = state.velocity.y;
  }

  /* ---------------- wheel emission ---------------- */

  private wheelEmission(
    dt: number,
    index: number,
    w: WheelState,
    fx: WheelFx,
    state: VehicleState,
    speed: number,
    groundY: number,
  ): void {
    const sf = SURFACE[w.surface] ?? SURFACE.dirt;

    // --- how hard is this tyre working -------------------------------------
    const long = Math.abs(w.slipRatio);
    const lat = Math.abs(Math.sin(w.slipAngle));
    const slip = THREE.MathUtils.clamp(Math.hypot(long * 0.9, lat * 1.3), 0, 1.6);
    const loadF = THREE.MathUtils.clamp(w.load / NOMINAL_LOAD, 0.1, 2.0);
    // Rolling dust: nothing below walking pace, saturating by motorway speed.
    const roll = THREE.MathUtils.smoothstep(speed, 1.8, 13);
    const fast = THREE.MathUtils.clamp(speed / 24, 0, 1.3);

    // Under water there is no dust, only splash.
    const submerged = this.waterLevel > -Infinity && groundY < this.waterLevel - 0.02;

    // --- geometry ----------------------------------------------------------
    // Outward = away from the chassis centreline, so each wheel throws its
    // plume to its own side instead of all four stacking in the middle.
    this.outward.set(w.position.x - state.position.x, 0, w.position.z - state.position.z);
    if (this.outward.lengthSq() < 1e-6) this.outward.copy(this.lateral);
    else this.outward.normalize();

    // A spinning wheel throws material backwards; a locked one ploughs it
    // forwards. That sign flip is a surprisingly strong readability cue.
    const eject = w.slipRatio >= -0.02 ? -1 : 1;

    const cx = w.position.x;
    const cz = w.position.z;

    if (submerged) {
      this.waterWheel(dt, w, fx, speed, loadF);
      return;
    }

    // --- airborne dust -----------------------------------------------------
    // Grey rubber smoke displaces brown dust on hard surfaces; the two share a
    // budget so a wheel never emits both at full rate.
    const smokeMix = THREE.MathUtils.clamp(sf.smoke * THREE.MathUtils.clamp(slip * 1.4 - 0.25, 0, 1), 0, 1);

    if (this.enabled.has('dust') && sf.dustAmount > 0.02) {
      const rate =
        24 * sf.dustAmount * loadF * (0.9 * roll + 1.35 * Math.min(slip, 1.25)) *
        (0.6 + 0.6 * fast) * (1 - smokeMix * 0.75) * this.intensity;
      fx.dustAcc += Math.min(rate, 44) * dt;
      const count = Math.floor(fx.dustAcc);
      fx.dustAcc -= count;
      for (let i = 0; i < count; i++) {
        this.emitDust(sf, cx, cz, groundY, speed, slip, roll, fast, eject);
      }
    }

    // --- tyre smoke --------------------------------------------------------
    if (this.enabled.has('smoke') && smokeMix > 0.02) {
      const rate = 11 * smokeMix * loadF * (0.35 + 0.9 * Math.min(slip, 1.3)) * this.intensity;
      fx.smokeAcc += Math.min(rate, 20) * dt;
      const count = Math.floor(fx.smokeAcc);
      fx.smokeAcc -= count;
      for (let i = 0; i < count; i++) {
        const d = this.d;
        d.mode = MODE_BILLOW;
        d.cell = ATLAS_SMOKE;
        this.tint(TYRE_SMOKE, 0.09);
        d.x = cx + (Math.random() - 0.5) * 0.3;
        d.y = groundY + 0.12 + Math.random() * 0.25;
        d.z = cz + (Math.random() - 0.5) * 0.3;
        // Smoke boils off the contact patch nearly straight up and gets left
        // behind; it does not travel with the car the way thrown dust does.
        d.vx = this.travel.x * eject * (0.4 + Math.random() * 0.9) + (Math.random() - 0.5) * 0.7;
        d.vz = this.travel.z * eject * (0.4 + Math.random() * 0.9) + (Math.random() - 0.5) * 0.7;
        d.vy = 1.1 + Math.random() * 1.6;
        d.life = 2.2 + Math.random() * 1.6;
        d.size0 = 0.22 + Math.random() * 0.22;
        d.size1 = 1.7 + Math.random() * 1.5;
        d.rot = Math.random() * 6.283;
        d.rotSpeed = (Math.random() - 0.5) * 0.7;
        d.drag = 1.5;
        d.gravity = 0.75;
        d.opacity = 0.26 * this.intensity;
        d.groundY = groundY;
        this.air.emit(d);
        this.lastEmitted++;
      }
    }

    // --- rooster tail ------------------------------------------------------
    // Hard wheelspin only. Dense, directed, short-lived: it is a jet, and the
    // general dust behind it is what lingers.
    const spinExcess = THREE.MathUtils.clamp((w.slipRatio - 0.28) / 0.8, 0, 1);
    if (this.enabled.has('rooster') && spinExcess > 0.02) {
      const rate = 22 * spinExcess * loadF * (0.35 + sf.dustAmount * 0.8) * this.intensity;
      const count = this.take(rate * dt, index);
      const throw_ = 2.6 + spinExcess * 5.5 + speed * 0.22;
      for (let i = 0; i < count; i++) {
        const d = this.d;
        d.mode = MODE_BILLOW;
        d.cell = ATLAS_DUST;
        this.tint(sf.dust, 0.1);
        d.x = cx + (Math.random() - 0.5) * 0.22;
        d.y = groundY + 0.08 + Math.random() * 0.2;
        d.z = cz + (Math.random() - 0.5) * 0.22;
        const spread = 0.35;
        d.vx = -this.travel.x * throw_ * (0.7 + Math.random() * 0.6)
             + this.lateral.x * (Math.random() - 0.5) * throw_ * spread;
        d.vz = -this.travel.z * throw_ * (0.7 + Math.random() * 0.6)
             + this.lateral.z * (Math.random() - 0.5) * throw_ * spread;
        d.vy = 1.6 + Math.random() * 3.2;
        d.life = 1.3 + Math.random() * 1.3;
        d.size0 = 0.16 + Math.random() * 0.18;
        d.size1 = sf.spread * (0.4 + Math.random() * 0.4);
        d.rot = Math.random() * 6.283;
        d.rotSpeed = (Math.random() - 0.5) * 1.4;
        d.drag = sf.drag * 1.5;
        d.gravity = sf.rise * 0.5;
        d.opacity = 0.34 * this.intensity;
        d.groundY = groundY;
        this.air.emit(d);
        this.lastEmitted++;
      }
    }

    // --- clods / clippings -------------------------------------------------
    if (this.enabled.has('clods') && sf.chunkAmount > 0.05) {
      const bite = THREE.MathUtils.clamp((slip - 0.16) / 0.7, 0, 1);
      const rate = 17 * sf.chunkAmount * loadF * bite * (0.4 + 0.7 * fast) * this.intensity;
      fx.chunkAcc += Math.min(rate, 34) * dt;
      const count = Math.floor(fx.chunkAcc);
      fx.chunkAcc -= count;
      const throw_ = 2.2 + slip * 3.4 + speed * 0.3;
      for (let i = 0; i < count; i++) {
        this.emitChunk(
          sf,
          cx + (Math.random() - 0.5) * 0.25,
          groundY + 0.1,
          cz + (Math.random() - 0.5) * 0.25,
          this.travel.x * eject * throw_ * (0.5 + Math.random() * 0.9)
            + this.outward.x * (Math.random() - 0.2) * 1.9
            + (Math.random() - 0.5) * 1.2,
          2.2 + Math.random() * 4.4,
          this.travel.z * eject * throw_ * (0.5 + Math.random() * 0.9)
            + this.outward.z * (Math.random() - 0.2) * 1.9
            + (Math.random() - 0.5) * 1.2,
          groundY,
          1,
        );
      }
    }
  }

  /** One dust billow off a working tyre. */
  private emitDust(
    sf: SurfaceFx,
    cx: number,
    cz: number,
    groundY: number,
    speed: number,
    slip: number,
    roll: number,
    fast: number,
    eject: number,
  ): void {
    const d = this.d;
    const r = Math.random();
    d.mode = MODE_BILLOW;
    d.cell = ATLAS_DUST;
    this.tint(sf.dust, 0.11);

    d.x = cx + (Math.random() - 0.5) * 0.42;
    d.y = groundY + 0.1 + Math.random() * 0.3;
    d.z = cz + (Math.random() - 0.5) * 0.42;

    // Behind, outward and a little up. The backward component is only a
    // fraction of road speed: the plume must be *left* in the world, not towed.
    const back = 0.7 + speed * (0.1 + 0.06 * slip) + r * 1.1;
    const out = (0.35 + Math.random() * 1.15) * (0.5 + slip * 0.9);
    d.vx = this.travel.x * eject * back + this.outward.x * out + (Math.random() - 0.5) * 0.6;
    d.vz = this.travel.z * eject * back + this.outward.z * out + (Math.random() - 0.5) * 0.6;
    d.vy = 1.1 + Math.random() * (1.6 + slip * 2.2);

    // Long-lived is the whole trick: a plume that hangs for three seconds is
    // the single clearest speed cue the game has.
    d.life = sf.life * (0.7 + Math.random() * 0.6) * (0.85 + 0.3 * roll);
    // Size and opacity are anti-correlated: the same puff of material spread
    // over a bigger quad has to be thinner. Rolling them independently is what
    // makes a trail read as a string of beads, because the few big *and* bright
    // sprites stand out from their neighbours.
    const big = Math.random();
    d.size0 = 0.7 + big * 1.3;
    d.size1 = sf.spread * (0.55 + big * 1.0) * (0.88 + fast * 0.5);
    d.rot = Math.random() * 6.283;
    d.rotSpeed = (Math.random() - 0.5) * 0.45;
    d.drag = sf.drag * (0.8 + Math.random() * 0.5);
    d.gravity = sf.rise * (0.6 + Math.random() * 0.9);
    // Individually almost invisible. Density comes from overlap, which is what
    // makes a plume look like a volume rather than a stack of cards.
    d.opacity = (0.34 - big * 0.15) * this.intensity;
    d.groundY = groundY;
    this.air.emit(d);
    this.lastEmitted++;
  }

  /** One ballistic lump: soil, gravel or a tuft of grass. */
  private emitChunk(
    sf: SurfaceFx,
    x: number,
    y: number,
    z: number,
    vx: number,
    vy: number,
    vz: number,
    groundY: number,
    scale: number,
  ): void {
    const d = this.d;
    d.mode = MODE_BALLISTIC;
    d.cell = ATLAS_CLOD;
    this.tint(sf.chunk, 0.16);
    d.x = x;
    d.y = y;
    d.z = z;
    d.vx = vx;
    d.vy = vy;
    d.vz = vz;
    d.life = 1.3 + Math.random() * 1.1;
    // Small: a clod that reads as a boulder is worse than no clod at all.
    const s = (0.075 + Math.random() * 0.11) * scale;
    d.size0 = s;
    d.size1 = s;
    d.rot = Math.random() * 6.283;
    d.rotSpeed = (Math.random() - 0.5) * 12;
    d.drag = 0;
    d.gravity = 15;
    d.opacity = 0.95;
    d.groundY = groundY;
    this.debris.emit(d);
    this.lastEmitted++;
  }

  /* ---------------- water ---------------- */

  private waterWheel(
    dt: number,
    w: WheelState,
    fx: WheelFx,
    speed: number,
    loadF: number,
  ): void {
    if (!this.enabled.has('splash')) return;
    const wl = this.waterLevel;
    const drive = THREE.MathUtils.smoothstep(speed, 0.4, 9);
    if (drive < 0.02) return;
    const rate = 26 * drive * THREE.MathUtils.clamp(loadF, 0.3, 1.4) * this.intensity;
    fx.splashAcc += Math.min(rate, 34) * dt;
    const count = Math.floor(fx.splashAcc);
    fx.splashAcc -= count;
    for (let i = 0; i < count; i++) {
      const d = this.d;
      d.mode = MODE_BALLISTIC;
      d.cell = ATLAS_DUST;
      this.tint(SPLASH, 0.05);
      d.x = w.position.x + (Math.random() - 0.5) * 0.35;
      d.y = wl + 0.04;
      d.z = w.position.z + (Math.random() - 0.5) * 0.35;
      const throw_ = 1.4 + speed * 0.42;
      d.vx = -this.travel.x * throw_ * Math.random() + this.outward.x * (0.6 + Math.random() * 2.2);
      d.vz = -this.travel.z * throw_ * Math.random() + this.outward.z * (0.6 + Math.random() * 2.2);
      d.vy = 2.2 + Math.random() * 3.6;
      d.life = 0.75 + Math.random() * 0.7;
      d.size0 = 0.05 + Math.random() * 0.11;
      d.size1 = 0.12 + Math.random() * 0.2;
      d.rot = Math.random() * 6.283;
      d.rotSpeed = 0;
      d.drag = 0;
      d.gravity = 13;
      d.opacity = 0.5 + Math.random() * 0.3;
      d.groundY = wl;
      this.debris.emit(d);
      this.lastEmitted++;
    }

    // Aerated mist above the sheet. Short-lived and very faint.
    if (Math.random() < drive * 0.55) {
      const d = this.d;
      d.mode = MODE_BILLOW;
      d.cell = ATLAS_SMOKE;
      this.tint(SPLASH, 0.04);
      d.x = w.position.x + (Math.random() - 0.5) * 0.4;
      d.y = wl + 0.2 + Math.random() * 0.3;
      d.z = w.position.z + (Math.random() - 0.5) * 0.4;
      d.vx = this.outward.x * (0.5 + Math.random()) - this.travel.x * speed * 0.1;
      d.vz = this.outward.z * (0.5 + Math.random()) - this.travel.z * speed * 0.1;
      d.vy = 0.9 + Math.random() * 1.3;
      d.life = 0.9 + Math.random() * 0.8;
      d.size0 = 0.2 + Math.random() * 0.25;
      d.size1 = 0.9 + Math.random() * 0.9;
      d.rot = Math.random() * 6.283;
      d.rotSpeed = (Math.random() - 0.5) * 0.9;
      d.drag = 2.4;
      d.gravity = 0.2;
      d.opacity = 0.24 * this.intensity;
      d.groundY = wl;
      this.air.emit(d);
      this.lastEmitted++;
    }
  }

  private splashBurst(x: number, y: number, z: number, s: number, groundY: number): void {
    if (!this.enabled.has('splash')) return;
    void groundY;
    const count = Math.round(10 + 26 * THREE.MathUtils.clamp(s, 0, 1.5));
    for (let i = 0; i < count; i++) {
      const a = Math.random() * Math.PI * 2;
      const speed = (1.6 + 3.2 * s) * (0.35 + Math.random());
      const d = this.d;
      d.mode = MODE_BALLISTIC;
      d.cell = ATLAS_DUST;
      this.tint(SPLASH, 0.05);
      d.x = x + Math.cos(a) * 0.15;
      d.y = y + 0.05;
      d.z = z + Math.sin(a) * 0.15;
      d.vx = Math.cos(a) * speed;
      d.vz = Math.sin(a) * speed;
      d.vy = 2.5 + Math.random() * 4.5 * s;
      d.life = 0.8 + Math.random() * 0.8;
      d.size0 = 0.06 + Math.random() * 0.12;
      d.size1 = 0.15 + Math.random() * 0.24;
      d.rot = 0;
      d.rotSpeed = 0;
      d.drag = 0;
      d.gravity = 13;
      d.opacity = 0.55 + Math.random() * 0.3;
      d.groundY = y;
      this.debris.emit(d);
      this.lastEmitted++;
    }
  }

  /* ---------------- suspension thump ---------------- */

  /**
   * A wheel that slams through its travel over a bump throws a small puff even
   * without a full airborne landing. This is what stops rough ground from
   * looking dead between jumps.
   */
  private thump(dt: number, w: WheelState, fx: WheelFx, groundY: number): void {
    if (!this.enabled.has('landing') || fx.thumpCooldown > 0) return;
    const rate = (w.compression - fx.prevCompression) / dt;
    if (w.compression < 0.72 || rate < 4.5) return;
    const sf = SURFACE[w.surface] ?? SURFACE.dirt;
    const s = THREE.MathUtils.clamp((rate - 4.5) / 22, 0.08, 0.5) * (0.4 + sf.dustAmount * 0.6);
    fx.thumpCooldown = 0.22;
    this.tmp.set(w.position.x, groundY, w.position.z);
    this.impact(this.tmp, s, w.surface);
  }

  /* ---------------- exhaust ---------------- */

  /**
   * Exhaust reads engine work, not the throttle pedal — the pedal is not in
   * `VehicleState` unless the vehicle chooses to publish it. Rising revs are a
   * blip, a gear change is a shove of unburnt fuel, and idle is a thin wisp.
   */
  private exhaust(dt: number, state: VehicleState): void {
    if (!this.enabled.has('exhaust')) {
      this.prevGear = state.gear;
      this.prevRpm = state.engineRpm;
      return;
    }

    const rpm = state.engineRpm;
    const dRpm = (rpm - this.prevRpm) / dt;
    this.prevRpm = rpm;

    if (state.gear !== this.prevGear) {
      this.shiftPuff = 1;
      this.prevGear = state.gear;
    }
    this.shiftPuff = Math.max(0, this.shiftPuff - dt * 5.5);

    // `throttle` is optional on VehicleState; the rev derivative stands in for
    // it when the vehicle doesn't publish one.
    const pedal = state.throttle;
    const load = pedal !== undefined
      ? THREE.MathUtils.clamp(pedal, 0, 1)
      : THREE.MathUtils.clamp(dRpm / 2600, 0, 1);

    const idle = 0.55;
    const rate = (idle + load * 7 + this.shiftPuff * 11) * this.intensity;
    this.exhaustAcc += rate * dt;
    const count = Math.floor(this.exhaustAcc);
    this.exhaustAcc -= count;
    if (count <= 0) return;

    this.tmp.copy(this.exhaustOffset).applyQuaternion(state.quaternion).add(state.position);
    const back = this.fwd;
    const groundY = this.groundAt(this.tmp.x, this.tmp.z, this.tmp.y - 0.5);

    for (let i = 0; i < count; i++) {
      const d = this.d;
      d.mode = MODE_BILLOW;
      d.cell = ATLAS_SMOKE;
      this.col.copy(EXHAUST_COLD).lerp(EXHAUST_HOT, Math.min(1, load + this.shiftPuff));
      this.tint(this.col, 0.05);
      d.x = this.tmp.x + (Math.random() - 0.5) * 0.09;
      d.y = this.tmp.y + (Math.random() - 0.5) * 0.09;
      d.z = this.tmp.z + (Math.random() - 0.5) * 0.09;
      const push = 1.1 + load * 3.4 + this.shiftPuff * 3.2;
      d.vx = -back.x * push + state.velocity.x * 0.4 + (Math.random() - 0.5) * 0.4;
      d.vy = -back.y * push + state.velocity.y * 0.4 + 0.25 + Math.random() * 0.4;
      d.vz = -back.z * push + state.velocity.z * 0.4 + (Math.random() - 0.5) * 0.4;
      d.life = 0.9 + Math.random() * 1.1 + this.shiftPuff * 0.6;
      d.size0 = 0.07 + Math.random() * 0.06;
      d.size1 = 0.55 + Math.random() * 0.6 + this.shiftPuff * 0.5;
      d.rot = Math.random() * 6.283;
      d.rotSpeed = (Math.random() - 0.5) * 1.6;
      d.drag = 2.6;
      d.gravity = 0.9;
      d.opacity = (0.12 + load * 0.12 + this.shiftPuff * 0.22) * this.intensity;
      d.groundY = groundY;
      this.air.emit(d);
      this.lastEmitted++;
    }
  }

  /* ---------------- sparks ---------------- */

  private sparkBurst(
    x: number,
    y: number,
    z: number,
    s: number,
    dir: THREE.Vector3 | null,
  ): void {
    const count = Math.round(6 + 18 * THREE.MathUtils.clamp(s, 0, 1.5));
    for (let i = 0; i < count; i++) {
      const d = this.d;
      d.mode = MODE_STREAK;
      d.cell = ATLAS_SPARK;
      // Sparks cool from white through orange; vary the birth colour rather
      // than animating it, which is free and reads the same in motion.
      this.col.copy(SPARK).multiplyScalar(0.7 + Math.random() * 0.9);
      d.r = this.col.r;
      d.g = this.col.g;
      d.b = this.col.b;
      const a = Math.random() * Math.PI * 2;
      const spread = 1.4 + Math.random() * 4.5 * s;
      const bx = dir ? dir.x : Math.cos(a);
      const bz = dir ? dir.z : Math.sin(a);
      d.x = x;
      d.y = y;
      d.z = z;
      d.vx = bx * spread + (Math.random() - 0.5) * 2.4;
      d.vz = bz * spread + (Math.random() - 0.5) * 2.4;
      d.vy = 1.2 + Math.random() * 3.4;
      d.life = 0.28 + Math.random() * 0.45;
      d.size0 = 0.06 + Math.random() * 0.05;
      d.size1 = 0.012;
      d.rot = 0;
      d.rotSpeed = 0;
      d.drag = 0;
      d.gravity = 11;
      d.opacity = 0.9;
      d.groundY = y - 0.02;
      this.sparks.emit(d);
      this.lastEmitted++;
    }
  }

  /* ---------------- helpers ---------------- */

  /** Ground height, from the sampler when there is one, else the fallback. */
  private groundAt(x: number, z: number, fallback: number): number {
    return this.terrain ? this.terrain.heightAt(x, z) : fallback;
  }

  /**
   * Write a colour into the scratch desc with a little per-particle variance,
   * so a plume is not one flat wash of a single hue.
   */
  private tint(base: THREE.Color, jitter: number): void {
    const k = 1 + (Math.random() - 0.5) * 2 * jitter;
    this.d.r = base.r * k;
    this.d.g = base.g * k;
    this.d.b = base.b * k;
  }

  /**
   * Integer particle count from a fractional budget, with the remainder
   * dithered rather than carried — used where a per-wheel accumulator would be
   * more state than the effect is worth.
   */
  private take(v: number, salt: number): number {
    const n = Math.floor(v);
    void salt;
    return n + (Math.random() < v - n ? 1 : 0);
  }
}
