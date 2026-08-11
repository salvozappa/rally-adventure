/**
 * ============================================================================
 *  VEHICLE VERIFICATION HARNESS — headless, no renderer, no real terrain.
 * ============================================================================
 *
 *   npx esbuild src/physics/vehicle.test.ts --bundle --platform=node \
 *     --format=esm --outfile=/tmp/vt.mjs --log-level=error && node /tmp/vt.mjs
 *
 * Everything the car drives on here is a purpose-built Rapier heightfield, so
 * the numbers below are about the *vehicle* and nothing else: a flat plain at
 * exactly y = 0, one 10-degree kicker, and one plain with a pit under the
 * rear-left wheel. See the note on `GroundSpec` for why it is a heightfield
 * and not a cuboid — that choice is load-bearing.
 *
 * The measurements are the deliverable. Every check prints the value it
 * measured next to the verdict, so a regression shows up as a number that
 * moved, not just as a red line.
 */

import RAPIER from '@dimforge/rapier3d-compat';
import * as THREE from 'three';
import { Vehicle, resetVehicleWarnings } from './Vehicle';
import { PHYSICS_DT } from './PhysicsWorld';
import { DESIGN_RIDE_HEIGHT, JEEP_TUNING } from './VehicleTuning';
import { GROUP, interactionGroups } from '../types';
import type { DriveInput, PhysicsContext, SurfaceKind, VehicleState } from '../types';

/* -------------------------------------------------------------------------- */
/* Harness                                                                    */
/* -------------------------------------------------------------------------- */

let failures = 0;
let checks = 0;

function check(ok: boolean, label: string, detail = ''): void {
  checks++;
  if (!ok) failures++;
  const mark = ok ? 'PASS' : 'FAIL';
  console.log(`  [${mark}] ${label}${detail ? `  ${detail}` : ''}`);
}

function section(title: string): void {
  console.log(`\n${title}`);
  console.log('-'.repeat(title.length));
}

function report(label: string, value: string | number): void {
  const v = typeof value === 'number' ? fmt(value) : value;
  console.log(`  ${label.padEnd(38, '.')} ${v}`);
}

function fmt(v: number): string {
  if (!Number.isFinite(v)) return String(v);
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e6)) return v.toExponential(3);
  return v.toFixed(Math.abs(v) >= 100 ? 1 : 4).replace(/\.?0+$/, '') || '0';
}

const DEG = 180 / Math.PI;
const KMH = 3.6;

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

/** Deterministic RNG so the randomised-input soak is reproducible. */
function mulberry(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* --- capture anything the vehicle's warnOnce prints ------------------------ */

const vehicleWarnings: string[] = [];
{
  const realWarn = console.warn.bind(console);
  console.warn = (...args: unknown[]): void => {
    const msg = args.map((a) => String(a)).join(' ');
    if (msg.includes('[Vehicle]')) vehicleWarnings.push(msg);
    realWarn(...args);
  };
}
function warningsSince(mark: number): string[] {
  return vehicleWarnings.slice(mark);
}

/* -------------------------------------------------------------------------- */
/* World construction                                                         */
/* -------------------------------------------------------------------------- */

await RAPIER.init();

const DT = PHYSICS_DT;
const TUNE = JEEP_TUNING;
const MASS = TUNE.chassis.mass;
const WEIGHT = MASS * 9.81;
const GROUND_GROUPS = interactionGroups(GROUP.TERRAIN, GROUP.CHASSIS | GROUP.WHEEL_RAY | GROUP.PROP);

/** Chassis collider corner offsets, for the ground-penetration probe. */
const CHASSIS_CORNERS: THREE.Vector3[] = [];
for (const sx of [-1, 1]) {
  for (const sy of [-1, 1]) {
    for (const sz of [-1, 1]) {
      CHASSIS_CORNERS.push(
        new THREE.Vector3(
          TUNE.chassis.colliderOffset.x + sx * TUNE.chassis.halfExtents.x,
          TUNE.chassis.colliderOffset.y + sy * TUNE.chassis.halfExtents.y,
          TUNE.chassis.colliderOffset.z + sz * TUNE.chassis.halfExtents.z,
        ),
      );
    }
  }
}

function makeInput(p: Partial<DriveInput> = {}): DriveInput {
  return {
    steer: 0,
    throttle: 0,
    brake: 0,
    handbrake: 0,
    shiftUp: false,
    shiftDown: false,
    recover: false,
    ...p,
  };
}

/**
 * Test ground is a Rapier HEIGHTFIELD, not a big cuboid, and that is not a
 * detail — it is the first thing this harness found.
 *
 * A sphere cast against a single 1 km cuboid comes back with the time of
 * impact wrong by up to 114 mm, because the convex cast solves in the box's
 * own huge coordinate range. Against a heightfield the cast descends a BVH and
 * finishes against one 2 m cell, and the answer is good to 0.04 mm. A 100 mm
 * error per wheel is roughly 3 kN of phantom spring force, which is why the
 * car appeared to buzz on a flat cuboid floor and sits perfectly still here.
 * The shipping terrain is a heightfield, so this matches production — but any
 * *prop* collider big enough to matter would hit the same wall.
 */
interface GroundSpec {
  /** Extent of the field, m. */
  sizeX: number;
  sizeZ: number;
  /** Grid spacing, m. The shipping terrain runs ~2 m cells. */
  cell: number;
  height: (x: number, z: number) => number;
}

function flatGround(sizeZ = 4000): GroundSpec {
  return { sizeX: 400, sizeZ, cell: 2, height: () => 0 };
}

/** Flat, plus a constant-grade kicker from z0 to z1 with a clean lip at z1. */
function rampGround(z0: number, z1: number, deg: number, halfWidth = 8): GroundSpec {
  const slope = Math.tan((deg * Math.PI) / 180);
  return {
    sizeX: 400,
    sizeZ: 400,
    cell: 2,
    height: (x, z) => (Math.abs(x) <= halfWidth && z >= z0 && z <= z1 ? slope * (z - z0) : 0),
  };
}

/** Flat, with one square pit deep enough that a wheel over it finds no ground. */
function pitGround(hole: { x0: number; x1: number; z0: number; z1: number }): GroundSpec {
  return {
    sizeX: 60,
    sizeZ: 200,
    cell: 0.25,
    height: (x, z) => (x > hole.x0 && x < hole.x1 && z > hole.z0 && z < hole.z1 ? -2 : 0),
  };
}

interface RigOptions {
  spawn?: THREE.Vector3;
  heading?: number;
  surface?: SurfaceKind;
  ground?: GroundSpec;
}

/** One Rapier world holding one car and whatever geometry the test needs. */
class Rig {
  readonly ctx: PhysicsContext;
  readonly world: RAPIER.World;
  readonly vehicle: Vehicle;
  readonly surfaceAt: (x: number, z: number) => SurfaceKind;
  t = 0;
  steps = 0;

  constructor(opts: RigOptions = {}) {
    const world = new RAPIER.World({ x: 0, y: -9.81, z: 0 });
    world.timestep = DT;
    world.numSolverIterations = 8;
    this.world = world;
    this.ctx = { world, rapier: RAPIER, dt: DT };

    // Rapier's heightfield is a column-major (row = z, col = x) matrix, which
    // is exactly the `ix * (rows + 1) + iz` layout the shipping terrain uses.
    const g = opts.ground ?? flatGround();
    const nx = Math.round(g.sizeX / g.cell);
    const nz = Math.round(g.sizeZ / g.cell);
    const heights = new Float32Array((nx + 1) * (nz + 1));
    for (let ix = 0; ix <= nx; ix++) {
      const x = (ix / nx - 0.5) * g.sizeX;
      for (let iz = 0; iz <= nz; iz++) {
        const z = (iz / nz - 0.5) * g.sizeZ;
        heights[ix * (nz + 1) + iz] = g.height(x, z);
      }
    }
    const groundBody = world.createRigidBody(RAPIER.RigidBodyDesc.fixed());
    world.createCollider(
      RAPIER.ColliderDesc.heightfield(nz, nx, heights, { x: g.sizeX, y: 1, z: g.sizeZ })
        .setFriction(1)
        .setRestitution(0)
        .setCollisionGroups(GROUND_GROUPS),
      groundBody,
    );

    const surface = opts.surface ?? 'dirt';
    this.surfaceAt = () => surface;

    // The query pipeline is only populated after a step.
    world.step();

    this.vehicle = new Vehicle(this.ctx, TUNE, {
      position: opts.spawn ?? new THREE.Vector3(0, 0.6, 0),
      heading: opts.heading ?? 0,
    });
  }

  get state(): VehicleState {
    return this.vehicle.state;
  }

  step(input: DriveInput): void {
    this.vehicle.fixedUpdate(DT, input, this.surfaceAt);
    this.world.step();
    this.t += DT;
    this.steps++;
  }

  /**
   * Run for `seconds`, driving from `control` and sampling every step. The `t`
   * handed to both callbacks is elapsed time *within this call*, so a
   * measurement window never has to know how long the car spent settling.
   */
  run(
    seconds: number,
    control: (t: number, st: VehicleState) => DriveInput,
    sample?: (t: number, st: VehicleState) => void,
  ): void {
    const n = Math.round(seconds / DT);
    for (let i = 0; i < n; i++) {
      this.step(control(i * DT, this.state));
      sample?.((i + 1) * DT, this.state);
    }
  }

  /** Drop the car and let it settle; returns once it is quiet or time runs out. */
  settle(seconds = 3): void {
    this.run(seconds, () => makeInput());
  }

  dispose(): void {
    this.vehicle.dispose();
    this.world.free();
  }
}

/* --- shared measurements --------------------------------------------------- */

const _v = new THREE.Vector3();
const _up = new THREE.Vector3();

/** Lowest world-space corner of the chassis collider. */
function chassisLowestY(st: VehicleState): number {
  let lo = Infinity;
  for (const c of CHASSIS_CORNERS) {
    _v.copy(c).applyQuaternion(st.quaternion).add(st.position);
    if (_v.y < lo) lo = _v.y;
  }
  return lo;
}

/** Underbody clearance above flat ground — the thing DESIGN_RIDE_HEIGHT names. */
function rideHeight(st: VehicleState): number {
  return st.position.y + TUNE.chassis.colliderOffset.y - TUNE.chassis.halfExtents.y;
}

function uprightness(st: VehicleState): number {
  return _up.set(0, 1, 0).applyQuaternion(st.quaternion).y;
}

/** Roll and pitch of the chassis in degrees. */
function attitude(st: VehicleState): { roll: number; pitch: number } {
  const e = new THREE.Euler().setFromQuaternion(st.quaternion, 'YXZ');
  return { roll: e.z * DEG, pitch: e.x * DEG };
}

function stateFinite(st: VehicleState): boolean {
  const nums = [
    st.position.x, st.position.y, st.position.z,
    st.quaternion.x, st.quaternion.y, st.quaternion.z, st.quaternion.w,
    st.velocity.x, st.velocity.y, st.velocity.z,
    st.angularVelocity.x, st.angularVelocity.y, st.angularVelocity.z,
    st.forwardSpeed, st.speed, st.engineRpm, st.clutch,
    st.localAccel.x, st.localAccel.y, st.localAccel.z,
  ];
  for (const w of st.wheels) {
    nums.push(w.compression, w.slipRatio, w.slipAngle, w.load, w.spin, w.steerAngle);
    nums.push(w.position.x, w.position.y, w.position.z);
  }
  for (const n of nums) if (!Number.isFinite(n)) return false;
  return true;
}

function stats(a: readonly number[]): { min: number; max: number; mean: number; sd: number } {
  if (a.length === 0) return { min: 0, max: 0, mean: 0, sd: 0 };
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  for (const v of a) {
    if (v < min) min = v;
    if (v > max) max = v;
    sum += v;
  }
  const mean = sum / a.length;
  let acc = 0;
  for (const v of a) acc += (v - mean) * (v - mean);
  return { min, max, mean, sd: Math.sqrt(acc / a.length) };
}

/* -------------------------------------------------------------------------- */

console.log('Rally vehicle verification');
console.log(`physics ${Math.round(1 / DT)} Hz   mass ${MASS} kg   design ride height ${fmt(DESIGN_RIDE_HEIGHT)} m`);

/* ========================================================================== */
/* 0. Static budget                                                           */
/* ========================================================================== */

section('0. Static budget');
{
  const wb = TUNE.steering.wheelbase;
  const comZ = TUNE.chassis.centreOfMass.z;
  const frontFrac = (comZ + wb / 2) / wb;
  const front = WEIGHT * frontFrac;
  const rear = WEIGHT * (1 - frontFrac);
  report('kerb weight', `${fmt(WEIGHT)} N`);
  report('static front / rear', `${fmt(frontFrac * 100)} / ${fmt((1 - frontFrac) * 100)} %`);
  report('static corner load F / R', `${fmt(front / 2)} / ${fmt(rear / 2)} N`);
  const sagF = front / 2 / TUNE.suspension.front.springRate;
  const sagR = rear / 2 / TUNE.suspension.rear.springRate;
  report('predicted sag F / R', `${fmt(sagF * 1000)} / ${fmt(sagR * 1000)} mm of ${fmt(TUNE.suspension.travel * 1000)} mm`);
  report('sag as fraction of travel', `${fmt((sagF / TUNE.suspension.travel) * 100)} / ${fmt((sagR / TUNE.suspension.travel) * 100)} %`);
  const fnF = Math.sqrt(TUNE.suspension.front.springRate / (front / 2 / 9.81)) / (2 * Math.PI);
  const fnR = Math.sqrt(TUNE.suspension.rear.springRate / (rear / 2 / 9.81)) / (2 * Math.PI);
  report('ride frequency F / R', `${fmt(fnF)} / ${fmt(fnR)} Hz`);
  const ccF = 2 * Math.sqrt(TUNE.suspension.front.springRate * (front / 2 / 9.81));
  const ccR = 2 * Math.sqrt(TUNE.suspension.rear.springRate * (rear / 2 / 9.81));
  report('front damping ratio bump/reb', `${fmt(TUNE.suspension.front.bumpDamping / ccF)} / ${fmt(TUNE.suspension.front.reboundDamping / ccF)}`);
  report('rear damping ratio bump/reb', `${fmt(TUNE.suspension.rear.bumpDamping / ccR)} / ${fmt(TUNE.suspension.rear.reboundDamping / ccR)}`);
  const rollover = TUNE.steering.frontTrack / 2 / TUNE.chassis.centreOfMass.y;
  report('static rollover threshold', `${fmt(rollover)} g`);
  report('dirt peak mu', TUNE.surfaces.dirt.friction);

  check(fnF > 1.0 && fnF < 1.8, 'front ride frequency is off-road soft (1.0-1.8 Hz)', `${fmt(fnF)} Hz`);
  check(fnR > 1.0 && fnR < 1.8, 'rear ride frequency is off-road soft (1.0-1.8 Hz)', `${fmt(fnR)} Hz`);
  check(
    TUNE.suspension.front.reboundDamping / ccF > 0.4 && TUNE.suspension.front.reboundDamping / ccF < 0.8,
    'front rebound damping in the no-pogo band (0.4-0.8 crit)',
    `${fmt(TUNE.suspension.front.reboundDamping / ccF)}`,
  );
  check(
    rollover > 1.15,
    'static rollover threshold well above tyre grip',
    `${fmt(rollover)} g vs ${fmt(TUNE.surfaces.rock.friction)} g of rock grip`,
  );
}

/* ========================================================================== */
/* 1. Rest                                                                    */
/* ========================================================================== */

section('1. Rest — dropped onto flat dirt');
{
  const mark = vehicleWarnings.length;
  resetVehicleWarnings();
  const rig = new Rig({ spawn: new THREE.Vector3(0, 1.0, 0) });

  const yLate: number[] = [];
  const vLate: number[] = [];
  const wLate: number[] = [];
  rig.run(
    5,
    () => makeInput(),
    (t, st) => {
      if (t > 3.5) {
        yLate.push(st.position.y);
        vLate.push(st.velocity.length());
        wLate.push(st.angularVelocity.length());
      }
    },
  );

  const st = rig.state;
  const ride = rideHeight(st);
  const err = Math.abs(ride - DESIGN_RIDE_HEIGHT) / DESIGN_RIDE_HEIGHT;
  const y = stats(yLate);
  const v = stats(vLate);
  const w = stats(wLate);
  const loads = st.wheels.map((x) => x.load);
  const comp = st.wheels.map((x) => x.compression);
  const att = attitude(st);

  report('settle time budget', '5 s');
  report('ride height (underbody)', `${fmt(ride * 1000)} mm  (design ${fmt(DESIGN_RIDE_HEIGHT * 1000)} mm)`);
  report('ride-height error', `${fmt(err * 100)} %`);
  report('body origin y', `${fmt(st.position.y * 1000)} mm`);
  report('speed after settling', `${fmt(v.max)} m/s max, ${fmt(v.mean)} mean`);
  report('angular speed after settling', `${fmt(w.max)} rad/s max`);
  report('height jitter (last 1.5 s)', `${fmt((y.max - y.min) * 1e6)} um p-p`);
  report('wheel loads FL FR RL RR', loads.map((l) => `${fmt(l)}`).join('  ') + ' N');
  report('sum of loads / weight', `${fmt(loads.reduce((a, b) => a + b, 0) / WEIGHT)}`);
  report('front / rear load split', `${fmt(((loads[0]! + loads[1]!) / WEIGHT) * 100)} / ${fmt(((loads[2]! + loads[3]!) / WEIGHT) * 100)} %`);
  report('suspension compression', comp.map((c) => `${fmt(c * 100)}%`).join('  '));
  report('attitude', `roll ${fmt(att.roll)} deg, pitch ${fmt(att.pitch)} deg`);

  check(err < 0.02, 'ride height within 2% of DESIGN_RIDE_HEIGHT', `${fmt(err * 100)} %`);
  check(v.max < 0.01, 'velocity below 10 mm/s after 3 s', `${fmt(v.max * 1000)} mm/s`);
  check(w.max < 0.01, 'no residual tumble', `${fmt(w.max)} rad/s`);
  check(y.max - y.min < 1e-4, 'no standstill jitter', `${fmt((y.max - y.min) * 1e6)} um p-p`);
  check(st.wheels.every((x) => x.grounded), 'all four wheels grounded');
  const lo = Math.min(...loads);
  const hi = Math.max(...loads);
  check(hi / lo < 1.25, 'load spread across the four corners under 25%', `${fmt(hi / lo)}x`);
  check(Math.abs(loads[0]! - loads[1]!) < 20 && Math.abs(loads[2]! - loads[3]!) < 20, 'left/right loads symmetric', `${fmt(Math.abs(loads[0]! - loads[1]!))} / ${fmt(Math.abs(loads[2]! - loads[3]!))} N`);
  check(
    Math.abs(loads.reduce((a, b) => a + b, 0) - WEIGHT) / WEIGHT < 0.01,
    'suspension carries exactly the vehicle weight',
    `${fmt(loads.reduce((a, b) => a + b, 0))} N vs ${fmt(WEIGHT)} N`,
  );
  check(Math.abs(att.roll) < 0.2 && Math.abs(att.pitch) < 0.6, 'sits level', `roll ${fmt(att.roll)}, pitch ${fmt(att.pitch)} deg`);
  check(warningsSince(mark).length === 0, 'no vehicle warnings', warningsSince(mark).join(' | '));
  rig.dispose();
}

/* ========================================================================== */
/* 2. Straight line                                                           */
/* ========================================================================== */

section('2. Straight line — full throttle from rest on dirt');
let accelSummary = { t100: 0, top: 0 };
{
  const mark = vehicleWarnings.length;
  resetVehicleWarnings();
  const rig = new Rig({ spawn: new THREE.Vector3(0, 0.45, 0) });
  rig.settle(1.5);
  rig.vehicle.respawn(new THREE.Vector3(0, rig.state.position.y, 0), 0);
  rig.settle(0.5);

  const z0 = rig.state.position.z;
  let t60 = -1;
  let t100 = -1;
  let prevV = 0;
  let prevT = 0;
  let top = 0;
  let drift = 0;
  let launchDrift = 0;
  let maxHeading = 0;
  let maxSlip = 0;
  const heading = new THREE.Vector3();
  const rpms: number[] = [];
  const gears = new Set<number>();
  const gearSeq: number[] = [];
  let lastGear = -99;
  let distance = 0;

  rig.run(
    60,
    () => makeInput({ throttle: 1 }),
    (t, st) => {
      const v = st.forwardSpeed;
      const target60 = 60 / KMH;
      const target100 = 100 / KMH;
      if (t60 < 0 && v >= target60) t60 = prevT + ((target60 - prevV) / (v - prevV)) * DT;
      if (t100 < 0 && v >= target100) t100 = prevT + ((target100 - prevV) / (v - prevV)) * DT;
      prevV = v;
      prevT = t;
      top = Math.max(top, v);
      drift = Math.max(drift, Math.abs(st.position.x));
      if (st.position.z - z0 < 200) launchDrift = Math.max(launchDrift, Math.abs(st.position.x));
      heading.set(0, 0, 1).applyQuaternion(st.quaternion);
      maxHeading = Math.max(maxHeading, Math.abs(Math.atan2(heading.x, heading.z)));
      rpms.push(st.engineRpm);
      if (st.gear !== lastGear) {
        gearSeq.push(st.gear);
        lastGear = st.gear;
      }
      gears.add(st.gear);
      for (const w of st.wheels) maxSlip = Math.max(maxSlip, w.slipRatio);
      distance = st.position.z - z0;
    },
  );

  const r = stats(rpms);
  accelSummary = { t100, top };
  report('0-60 km/h', `${fmt(t60)} s`);
  report('0-100 km/h', `${fmt(t100)} s`);
  report('top speed', `${fmt(top)} m/s = ${fmt(top * KMH)} km/h`);
  report('distance covered in 60 s', `${fmt(distance)} m`);
  report('lateral drift over first 200 m', `${fmt(launchDrift * 1000)} mm`);
  report('lateral drift over the whole run', `${fmt(drift)} m in ${fmt(distance)} m`);
  report('worst heading error', `${fmt(maxHeading * DEG)} deg`);
  report('peak wheelspin slip ratio', fmt(maxSlip));
  report('engine rpm range', `${fmt(r.min)} .. ${fmt(r.max)}`);
  report('gear sequence', gearSeq.join(' -> '));

  check(t100 > 0 && t100 >= 8 && t100 <= 16, '0-100 km/h plausible for a 1900 kg off-roader', `${fmt(t100)} s`);
  check(top * KMH >= 130 && top * KMH <= 175, 'top speed 130-175 km/h', `${fmt(top * KMH)} km/h`);
  // The car tracks straight; it does not track *perfectly* straight, and it
  // should not. Limited-slip diffs amplify float-level asymmetry at a standing
  // start into a fraction of a degree of pull, exactly as a real one amplifies
  // an uneven surface. What matters is that the pull is a fixed heading offset
  // and not a divergence: hands off at 150 km/h the car holds a line.
  check(launchDrift < 0.5, 'lateral drift under 0.5 m over the first 200 m', `${fmt(launchDrift * 1000)} mm`);
  check(maxHeading * DEG < 2, 'heading error stays under 2 deg with no steering input', `${fmt(maxHeading * DEG)} deg`);
  check(r.min >= TUNE.drivetrain.engine.stallRpm, 'engine never stalls', `${fmt(r.min)} rpm`);
  check(
    r.max <= TUNE.drivetrain.engine.redlineRpm + TUNE.drivetrain.engine.limiterBandRpm + 60,
    'rev limiter holds the engine at the redline',
    `${fmt(r.max)} rpm vs redline ${TUNE.drivetrain.engine.redlineRpm}`,
  );
  check(gears.size >= 4, 'the gearbox works up through the ratios', `${gears.size} gears used`);
  check(maxSlip < 1.0, 'traction hooks up rather than spinning forever', `peak kappa ${fmt(maxSlip)}`);
  check(warningsSince(mark).length === 0, 'no vehicle warnings', warningsSince(mark).join(' | '));
  rig.dispose();
}

/* ========================================================================== */
/* 3. Braking                                                                 */
/* ========================================================================== */

section('3. Braking — 80 km/h to a stop on dirt');
{
  const mark = vehicleWarnings.length;
  resetVehicleWarnings();
  const rig = new Rig({ spawn: new THREE.Vector3(0, 0.45, 0) });
  rig.settle(1.5);

  const target = 80 / KMH;
  let armed = false;
  for (let i = 0; i < 120 / DT && !armed; i++) {
    rig.step(makeInput({ throttle: 1 }));
    if (rig.state.forwardSpeed >= target) armed = true;
  }
  const vEntry = rig.state.forwardSpeed;
  const z0 = rig.state.position.z;
  const tStart = rig.t;

  let minSlip = 0;
  let peakDecel = 0;
  let prevV = vEntry;
  let stopped = false;
  let dist = 0;
  let tStop = 0;
  const pitches: number[] = [];

  for (let i = 0; i < Math.round(12 / DT); i++) {
    rig.step(makeInput({ brake: 1 }));
    const st = rig.state;
    for (const w of st.wheels) minSlip = Math.min(minSlip, w.slipRatio);
    const a = (prevV - st.forwardSpeed) / DT;
    peakDecel = Math.max(peakDecel, a);
    prevV = st.forwardSpeed;
    pitches.push(attitude(st).pitch);
    if (!stopped && st.forwardSpeed < 0.3) {
      stopped = true;
      dist = st.position.z - z0;
      tStop = rig.t - tStart;
      break;
    }
  }
  if (!stopped) dist = rig.state.position.z - z0;

  const meanG = vEntry ** 2 / (2 * Math.max(1e-3, dist)) / 9.81;
  const p = stats(pitches);
  report('entry speed', `${fmt(vEntry)} m/s = ${fmt(vEntry * KMH)} km/h`);
  report('stopping distance', `${fmt(dist)} m`);
  report('stopping time', `${fmt(tStop)} s`);
  report('mean deceleration', `${fmt(meanG)} g`);
  report('peak deceleration', `${fmt(peakDecel / 9.81)} g`);
  report('most negative slip ratio', fmt(minSlip));
  report('nose dive (peak pitch)', `${fmt(p.min)} .. ${fmt(p.max)} deg`);

  check(stopped, 'the car actually stops');
  check(dist >= 28 && dist <= 45, 'stopping distance 28-45 m on dirt', `${fmt(dist)} m`);
  check(minSlip < -0.85, 'wheels lock (slip ratio approaches -1)', `${fmt(minSlip)}`);
  check(meanG > 0.6 && meanG < 1.0, 'mean retardation is tyre-limited, not brake-limited', `${fmt(meanG)} g`);
  check(Math.abs(p.min) < 12, 'nose dive stays controlled', `${fmt(p.min)} deg`);
  check(warningsSince(mark).length === 0, 'no vehicle warnings', warningsSince(mark).join(' | '));
  rig.dispose();
}

/* ========================================================================== */
/* 4. Cornering                                                               */
/* ========================================================================== */

section('4. Cornering — steady-state skidpad on dirt');
{
  const mark = vehicleWarnings.length;
  resetVehicleWarnings();

  /** Hold a target speed with the throttle; returns the steady-state circle. */
  function skidpad(steer: number, targetSpeed: number, seconds = 26) {
    const rig = new Rig({ spawn: new THREE.Vector3(0, 0.45, 0) });
    rig.settle(1.5);
    const lat: number[] = [];
    const yaw: number[] = [];
    const spd: number[] = [];
    const rolls: number[] = [];
    let minUp = 1;
    let maxSlipAngle = 0;
    let ok = true;
    rig.run(
      seconds,
      (t, st) => {
        const e = targetSpeed - st.forwardSpeed;
        // Ease the steering in so the entry transient does not dominate.
        const s = steer * clamp(t / 2.5, 0, 1);
        return makeInput({ steer: s, throttle: clamp(e * 0.5, 0, 1), brake: clamp(-e * 0.25, 0, 0.4) });
      },
      (t, st) => {
        if (!stateFinite(st)) ok = false;
        minUp = Math.min(minUp, uprightness(st));
        if (t > seconds * 0.55) {
          const w = st.angularVelocity.y;
          lat.push((Math.abs(w) * st.speed) / 9.81);
          yaw.push(w);
          spd.push(st.speed);
          rolls.push(attitude(st).roll);
          for (const wh of st.wheels) maxSlipAngle = Math.max(maxSlipAngle, Math.abs(wh.slipAngle));
        }
      },
    );
    const g = stats(lat);
    const y = stats(yaw);
    const v = stats(spd);
    const roll = stats(rolls);
    rig.dispose();
    return {
      g,
      yaw: y,
      speed: v,
      radius: v.mean / Math.max(1e-6, Math.abs(y.mean)),
      roll,
      minUp,
      maxSlipAngle,
      ok,
    };
  }

  const mid = skidpad(0.45, 13);
  report('steer input / target speed', `0.45 / 13 m/s`);
  report('steady speed', `${fmt(mid.speed.mean)} m/s (sd ${fmt(mid.speed.sd)})`);
  report('steady yaw rate', `${fmt(mid.yaw.mean)} rad/s (sd ${fmt(mid.yaw.sd)})`);
  report('steady-state radius', `${fmt(mid.radius)} m`);
  report('lateral acceleration', `${fmt(mid.g.mean)} g (peak ${fmt(mid.g.max)})`);
  report('body roll', `${fmt(mid.roll.mean)} deg (max ${fmt(Math.max(Math.abs(mid.roll.min), Math.abs(mid.roll.max)))})`);
  report('peak slip angle', `${fmt(mid.maxSlipAngle * DEG)} deg`);
  report('min uprightness', fmt(mid.minUp));

  check(mid.ok, 'state stays finite through the corner');
  check(mid.yaw.sd / Math.abs(mid.yaw.mean) < 0.06, 'the circle is steady, not a spin or a wobble', `yaw sd/mean ${fmt(mid.yaw.sd / Math.abs(mid.yaw.mean))}`);
  check(mid.minUp > 0.9, 'never close to a rollover', `min up.y ${fmt(mid.minUp)}`);
  check(Math.abs(mid.roll.mean) > 0.8 && Math.abs(mid.roll.mean) < 8, 'body rolls visibly but not alarmingly', `${fmt(mid.roll.mean)} deg`);

  // Push it to the grip limit.
  console.log('');
  const limit = skidpad(0.45, 19, 30);
  report('limit run target speed', '19 m/s');
  report('limit steady speed', `${fmt(limit.speed.mean)} m/s`);
  report('limit radius', `${fmt(limit.radius)} m`);
  report('limit lateral acceleration', `${fmt(limit.g.mean)} g (peak ${fmt(limit.g.max)})`);
  report('limit body roll', `${fmt(limit.roll.mean)} deg`);
  report('limit peak slip angle', `${fmt(limit.maxSlipAngle * DEG)} deg`);
  report('limit min uprightness', fmt(limit.minUp));

  check(limit.g.mean >= 0.6 && limit.g.mean <= 0.95, 'limit lateral grip 0.6-0.95 g on dirt', `${fmt(limit.g.mean)} g`);
  check(limit.minUp > 0.85, 'no rollover at the grip limit', `min up.y ${fmt(limit.minUp)}`);
  check(limit.yaw.sd / Math.abs(limit.yaw.mean) < 0.25, 'the slide is progressive, not a spin', `yaw sd/mean ${fmt(limit.yaw.sd / Math.abs(limit.yaw.mean))}`);
  check(limit.maxSlipAngle * DEG > 4, 'the car really is sliding at the limit', `${fmt(limit.maxSlipAngle * DEG)} deg of slip angle`);
  check(warningsSince(mark).length === 0, 'no vehicle warnings', warningsSince(mark).join(' | '));
}

/* ========================================================================== */
/* 5. Jump                                                                    */
/* ========================================================================== */

section('5. Jump — 10 degree kicker');
{
  const mark = vehicleWarnings.length;
  resetVehicleWarnings();
  const Z0 = 30;
  const Z1 = 40;
  const LIP = Math.tan((10 * Math.PI) / 180) * (Z1 - Z0);

  interface JumpResult {
    airtime: number;
    apex: number;
    distance: number;
    impactVy: number;
    launchSpeed: number;
    maxComp: number;
    peakLoad: number;
    peakDecel: number;
    landingPitch: number;
    minClear: number;
    bounces: number;
    settle: number;
    upright: number;
    finite: boolean;
  }

  /** Approach the kicker holding `target` m/s, fly, land, settle. */
  function jump(target: number): JumpResult {
    const rig = new Rig({ spawn: new THREE.Vector3(0, 0.45, -60), ground: rampGround(Z0, Z1, 10) });
    rig.settle(1.5);

    const r: JumpResult = {
      airtime: -1, apex: -Infinity, distance: 0, impactVy: 0, launchSpeed: 0,
      maxComp: 0, peakLoad: 0, peakDecel: 0, landingPitch: 0, minClear: Infinity,
      bounces: 0, settle: 0, upright: 0, finite: true,
    };
    let airStart = -1;
    let airEnd = -1;
    let wasAir = false;
    let landed = false;
    let prevVy = 0;
    const postLandY: number[] = [];

    rig.run(
      16,
      (t, st) => {
        const e = target - st.forwardSpeed;
        return makeInput({ throttle: clamp(e * 0.5, 0, 1), brake: clamp(-e * 0.2, 0, 0.3) });
      },
      (t, st) => {
        if (!stateFinite(st)) r.finite = false;
        const air = st.airborne;
        // Only the flight off the lip counts; the little skip over the kink at
        // the top of the ramp does not.
        if (air && !wasAir && st.position.z > Z1 - 0.5) {
          if (airStart < 0) {
            airStart = t;
            r.launchSpeed = st.speed;
          } else if (landed) {
            r.bounces++;
          }
        }
        if (!air && wasAir && airStart > 0 && airEnd < 0) {
          airEnd = t;
          landed = true;
          r.distance = st.position.z - Z1;
          r.impactVy = Math.abs(prevVy);
          r.landingPitch = attitude(st).pitch;
        }
        wasAir = air;
        if (airStart > 0) {
          r.apex = Math.max(r.apex, st.position.y);
          r.minClear = Math.min(r.minClear, chassisLowestY(st));
          if (landed) {
            for (const w of st.wheels) {
              r.maxComp = Math.max(r.maxComp, w.compression);
              r.peakLoad = Math.max(r.peakLoad, w.load);
            }
            r.peakDecel = Math.max(r.peakDecel, (st.velocity.y - prevVy) / DT);
            postLandY.push(st.position.y);
          }
        }
        prevVy = st.velocity.y;
      },
    );

    r.airtime = airEnd > 0 ? airEnd - airStart : -1;
    const tail = stats(postLandY.slice(-Math.round(1.5 / DT)));
    r.settle = tail.max - tail.min;
    r.upright = uprightness(rig.state);
    rig.dispose();
    return r;
  }

  report('kicker', `${Z1 - Z0} m at 10 deg, lip ${fmt(LIP)} m high`);

  for (const [label, target] of [['moderate', 12] as const, ['big', 18] as const]) {
    console.log('');
    const r = jump(target);
    report(`${label}: launch speed`, `${fmt(r.launchSpeed)} m/s = ${fmt(r.launchSpeed * KMH)} km/h`);
    report(`${label}: airtime`, `${fmt(r.airtime)} s`);
    report(`${label}: apex (body origin)`, `${fmt(r.apex)} m`);
    report(`${label}: distance past the lip`, `${fmt(r.distance)} m`);
    report(`${label}: vertical impact speed`, `${fmt(r.impactVy)} m/s`);
    report(`${label}: pitch at touchdown`, `${fmt(r.landingPitch)} deg`);
    report(`${label}: peak suspension travel used`, `${fmt(r.maxComp * 100)} % of ${fmt(TUNE.suspension.travel * 1000)} mm`);
    report(`${label}: peak corner load`, `${fmt(r.peakLoad / 1000)} kN = ${fmt(r.peakLoad / (WEIGHT / 4))}x static`);
    report(`${label}: peak vertical deceleration`, `${fmt(r.peakDecel / 9.81)} g`);
    report(`${label}: min chassis clearance`, `${fmt(r.minClear * 1000)} mm`);
    report(`${label}: re-bounces after touchdown`, r.bounces);
    report(`${label}: height spread 1.5 s later`, `${fmt(r.settle * 1000)} mm p-p`);

    check(r.finite, `${label}: no NaN anywhere through the jump`);
    check(r.airtime > 0.3 && r.airtime < 2.5, `${label}: a real jump happened`, `${fmt(r.airtime)} s of airtime`);
    check(r.upright > 0.95, `${label}: lands and stays on its wheels`, `up.y ${fmt(r.upright)}`);
    check(r.minClear > 0.02, `${label}: chassis never touches the ground`, `${fmt(r.minClear * 1000)} mm`);
    check(Math.abs(r.landingPitch) < 6, `${label}: lands close to flat`, `${fmt(r.landingPitch)} deg of pitch`);
    check(r.bounces === 0, `${label}: no pogo after the landing`, `${r.bounces} re-bounces`);
    check(r.settle < 0.01, `${label}: settled flat 1.5 s after landing`, `${fmt(r.settle * 1000)} mm p-p`);

    if (target === 12) {
      // A jump a player takes without thinking about it. The suspension should
      // swallow it whole and still have travel in reserve.
      check(r.maxComp < 0.95, 'moderate: suspension has travel left in reserve', `${fmt(r.maxComp * 100)} % used`);
      check(r.peakLoad < 10 * (WEIGHT / 4), 'moderate: landing load stays modest', `${fmt(r.peakLoad / (WEIGHT / 4))}x static`);
    } else {
      // 6 m/s of vertical impact is a 1.9 m free fall. No 300 mm compression
      // stroke absorbs that without reaching the stops — 6.1^2/(2*0.30) is
      // already 6.3 g of *average* deceleration, so a peak near twice that is
      // the floor, not a tuning failure. What must hold is that the stop is a
      // progressive squash the car drives away from.
      check(r.maxComp > 0.85, 'big: the suspension takes the hit, not the chassis', `${fmt(r.maxComp * 100)} % used`);
      check(r.peakLoad < 20 * (WEIGHT / 4), 'big: the bottom-out is a squash, not a hammer', `${fmt(r.peakLoad / (WEIGHT / 4))}x static`);
      check(r.peakDecel / 9.81 < 16, 'big: landing deceleration stays survivable', `${fmt(r.peakDecel / 9.81)} g`);
    }
  }
  check(warningsSince(mark).length === 0, 'no vehicle warnings', warningsSince(mark).join(' | '));
}


/* ========================================================================== */
/* 6. Stability soak                                                          */
/* ========================================================================== */

section('6. Stability — 60 s of randomised input');
{
  const mark = vehicleWarnings.length;
  resetVehicleWarnings();
  const rig = new Rig({ spawn: new THREE.Vector3(0, 0.45, 0) });
  rig.settle(1.0);

  const rnd = mulberry(0x5eed1234);
  let steer = 0;
  let throttle = 0;
  let brake = 0;
  let handbrake = 0;
  let hold = 0;
  let finite = true;
  let maxSpeed = 0;
  let maxOmega = 0;
  let maxEnergy = 0;
  let maxDist = 0;
  let flips = 0;
  let wasFlipped = false;
  const I = TUNE.chassis.inertia;

  rig.run(
    60,
    (t, st) => {
      hold -= DT;
      if (hold <= 0) {
        hold = 0.3 + rnd() * 1.2;
        steer = (rnd() * 2 - 1) * (rnd() < 0.3 ? 1 : 0.6);
        throttle = rnd() < 0.65 ? rnd() : 0;
        brake = rnd() < 0.25 ? rnd() : 0;
        handbrake = rnd() < 0.12 ? 1 : 0;
      }
      return makeInput({ steer, throttle, brake, handbrake, recover: rnd() < 0.002 });
    },
    (t, st) => {
      if (!stateFinite(st)) finite = false;
      maxSpeed = Math.max(maxSpeed, st.speed);
      maxOmega = Math.max(maxOmega, st.angularVelocity.length());
      const w = st.angularVelocity;
      const e = 0.5 * MASS * st.speed ** 2 + 0.5 * (I.x * w.x ** 2 + I.y * w.y ** 2 + I.z * w.z ** 2);
      maxEnergy = Math.max(maxEnergy, e);
      maxDist = Math.max(maxDist, Math.hypot(st.position.x, st.position.z));
      const flipped = uprightness(st) < TUNE.assists.flippedThreshold;
      if (flipped && !wasFlipped) flips++;
      wasFlipped = flipped;
    },
  );

  const st = rig.state;
  report('max speed reached', `${fmt(maxSpeed)} m/s = ${fmt(maxSpeed * KMH)} km/h`);
  report('max angular speed', `${fmt(maxOmega)} rad/s`);
  report('max kinetic energy', `${fmt(maxEnergy / 1000)} kJ`);
  report('max distance from origin', `${fmt(maxDist)} m`);
  report('times it ended up flipped', flips);
  report('final position', `x=${fmt(st.position.x)} y=${fmt(st.position.y)} z=${fmt(st.position.z)}`);
  report('final ride height', `${fmt(rideHeight(st) * 1000)} mm`);
  report('vehicle warnings fired', warningsSince(mark).length);

  check(finite, 'no NaN in 7200 steps of abuse');
  check(maxSpeed < 60, 'no energy explosion (speed stays physical)', `${fmt(maxSpeed)} m/s`);
  check(maxOmega < 12, 'no angular explosion', `${fmt(maxOmega)} rad/s`);
  check(maxDist < 480, 'stays inside the test world', `${fmt(maxDist)} m`);
  check(Math.abs(st.position.y) < 5, 'never falls through the floor', `y=${fmt(st.position.y)} m`);
  check(warningsSince(mark).length === 0, 'resetVehicleWarnings() reports no warnings fired', warningsSince(mark).join(' | '));
  rig.dispose();
}

/* ========================================================================== */
/* 7. Drivetrain                                                              */
/* ========================================================================== */

section('7. Drivetrain — shifts, rev range, LSD with a wheel in the air');
{
  const mark = vehicleWarnings.length;
  resetVehicleWarnings();

  // --- a. up and down through the box --------------------------------------
  {
    const rig = new Rig({ spawn: new THREE.Vector3(0, 0.45, 0) });
    rig.settle(1.5);
    const seq: number[] = [];
    let last = -99;
    const rpms: number[] = [];
    let idleRpm = 0;
    rig.run(
      36,
      (t) => {
        // 18 s hard acceleration, then coast and brake so it must downshift.
        if (t < 18) return makeInput({ throttle: 1 });
        if (t < 30) return makeInput({ brake: 0.35 });
        return makeInput();
      },
      (t, st) => {
        if (st.gear !== last) {
          seq.push(st.gear);
          last = st.gear;
        }
        rpms.push(st.engineRpm);
        if (t > 34) idleRpm = st.engineRpm;
      },
    );
    const r = stats(rpms);
    let up = 0;
    let down = 0;
    for (let i = 1; i < seq.length; i++) {
      if (seq[i]! > seq[i - 1]!) up++;
      else down++;
    }
    report('gear sequence', seq.join(' -> '));
    report('upshifts / downshifts', `${up} / ${down}`);
    report('rpm min / max', `${fmt(r.min)} / ${fmt(r.max)}`);
    report('rpm back at rest', fmt(idleRpm));
    check(up >= 3, 'shifts up through at least four gears', `${up} upshifts`);
    check(down >= 2, 'shifts back down under braking', `${down} downshifts`);
    check(r.min >= TUNE.drivetrain.engine.stallRpm, 'rpm never drops below the stall line', `${fmt(r.min)} rpm`);
    check(
      r.max <= TUNE.drivetrain.engine.redlineRpm + TUNE.drivetrain.engine.limiterBandRpm + 60,
      'rpm never exceeds the limiter band',
      `${fmt(r.max)} rpm`,
    );
    check(
      Math.abs(idleRpm - TUNE.drivetrain.engine.idleRpm) < 140,
      'returns to idle when stopped',
      `${fmt(idleRpm)} vs ${TUNE.drivetrain.engine.idleRpm} rpm`,
    );
    rig.dispose();
  }

  // --- b. low range ---------------------------------------------------------
  console.log('');
  {
    const rig = new Rig({ spawn: new THREE.Vector3(0, 0.45, 0) });
    rig.settle(1.5);
    const engaged = rig.vehicle.setLowRange(true);
    let d0 = rig.state.position.z;
    let crawlSpeed = 0;
    rig.run(6, () => makeInput({ throttle: 0.25 }), (t, st) => {
      if (t > 4) crawlSpeed = st.forwardSpeed;
    });
    report('low range engaged at rest', engaged ? 'yes' : 'NO');
    report('crawl speed at 25% throttle', `${fmt(crawlSpeed)} m/s`);
    report('distance in 6 s', `${fmt(rig.state.position.z - d0)} m`);
    check(engaged, 'low range engages at a standstill');
    check(rig.vehicle.isLowRange(), 'low range reports engaged');
    check(crawlSpeed > 0.5 && crawlSpeed < 12, 'low range crawls rather than bolts', `${fmt(crawlSpeed)} m/s`);
    const refused = rig.vehicle.setLowRange(false);
    report('low range change while moving', refused ? 'allowed' : 'refused');
    rig.dispose();
  }

  // --- c. LSD with the rear-left wheel over a hole --------------------------
  console.log('');
  {
    const hole = { x0: -1.55, x1: -0.05, z0: -2.05, z1: -0.55 };
    const rig = new Rig({ spawn: new THREE.Vector3(0, 0.45, 0), ground: pitGround(hole) });
    rig.settle(2.0);

    const st0 = rig.state;
    const grounded = st0.wheels.map((w) => w.grounded);
    const z0 = st0.position.z;
    report('wheels grounded at rest FL FR RL RR', grounded.map((g) => (g ? 'Y' : '-')).join(' '));
    report('loads at rest', st0.wheels.map((w) => fmt(w.load)).join('  ') + ' N');
    check(!grounded[2], 'rear-left wheel really is hanging in the hole');
    check(grounded[0]! && grounded[1]! && grounded[3]!, 'the other three carry the car');

    let maxTorqueSpread = 0;
    let liftedSpin = 0;
    let loadedSpin = 0;
    rig.run(
      6,
      () => makeInput({ throttle: 1 }),
      (t, st) => {
        if (t > 0.2 && t < 2.0) {
          liftedSpin = Math.max(liftedSpin, Math.abs(st.wheels[2]!.spin));
          loadedSpin = Math.max(loadedSpin, Math.abs(st.wheels[3]!.spin));
          maxTorqueSpread = Math.max(maxTorqueSpread, Math.abs(st.wheels[2]!.spin - st.wheels[3]!.spin));
        }
      },
    );
    const travelled = rig.state.position.z - z0;
    report('distance crawled in 6 s', `${fmt(travelled)} m`);
    report('lifted-wheel peak spin', `${fmt(liftedSpin)} rad/s`);
    report('loaded-wheel peak spin', `${fmt(loadedSpin)} rad/s`);
    report('final speed', `${fmt(rig.state.forwardSpeed)} m/s`);
    check(travelled > 2, 'the LSD drives the loaded wheel and the car crawls out', `${fmt(travelled)} m`);
    check(loadedSpin > 1, 'the loaded wheel receives real torque', `${fmt(loadedSpin)} rad/s`);
    check(liftedSpin < TUNE.tire.maxSpin, 'the free wheel does not run away', `${fmt(liftedSpin)} rad/s`);
    rig.dispose();
  }

  check(warningsSince(mark).length === 0, 'no vehicle warnings', warningsSince(mark).join(' | '));
}

/* ========================================================================== */
/* 8. Determinism                                                             */
/* ========================================================================== */

section('8. Determinism — identical inputs, identical spawn');
{
  const mark = vehicleWarnings.length;
  resetVehicleWarnings();

  function trace(): number[] {
    const rig = new Rig({ spawn: new THREE.Vector3(0, 0.55, 0) });
    const rnd = mulberry(0xd37e12);
    let steer = 0;
    let throttle = 0;
    let hold = 0;
    rig.run(10, () => {
      hold -= DT;
      if (hold <= 0) {
        hold = 0.4 + rnd() * 0.8;
        steer = (rnd() * 2 - 1) * 0.7;
        throttle = rnd();
      }
      return makeInput({ steer, throttle });
    });
    const st = rig.state;
    const out = [
      st.position.x, st.position.y, st.position.z,
      st.quaternion.x, st.quaternion.y, st.quaternion.z, st.quaternion.w,
      st.velocity.x, st.velocity.y, st.velocity.z,
      st.angularVelocity.x, st.angularVelocity.y, st.angularVelocity.z,
      st.engineRpm, st.gear, st.clutch,
      ...st.wheels.flatMap((w) => [w.spin, w.slipRatio, w.slipAngle, w.load, w.compression]),
    ];
    rig.dispose();
    return out;
  }

  const a = trace();
  const b = trace();
  let worst = 0;
  let worstIdx = -1;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i]! - b[i]!);
    if (d > worst) {
      worst = d;
      worstIdx = i;
    }
  }
  report('state values compared', a.length);
  report('final position (run A)', `x=${fmt(a[0]!)} y=${fmt(a[1]!)} z=${fmt(a[2]!)}`);
  report('final position (run B)', `x=${fmt(b[0]!)} y=${fmt(b[1]!)} z=${fmt(b[2]!)}`);
  report('max |A - B|', worst === 0 ? '0 (bit identical)' : `${worst} (index ${worstIdx})`);
  check(worst === 0, 'two runs from the same spawn produce identical state after 10 s', `max delta ${worst}`);
  check(warningsSince(mark).length === 0, 'no vehicle warnings', warningsSince(mark).join(' | '));
}

/* ========================================================================== */

section('Summary');
report('0-100 km/h', `${fmt(accelSummary.t100)} s`);
report('top speed', `${fmt(accelSummary.top * KMH)} km/h`);
report('total vehicle warnings across all tests', vehicleWarnings.length);
check(vehicleWarnings.length === 0, 'the whole suite ran without a single vehicle warning');

console.log(`\n${failures === 0 ? 'OK' : 'FAILED'} — ${checks - failures}/${checks} checks passed`);
process.exit(failures === 0 ? 0 : 1);
