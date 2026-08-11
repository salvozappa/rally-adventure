/**
 * fx.preview.ts — visual test rig for the drive FX.
 *
 * There is no physics engine here and no real vehicle. A kinematic stand-in
 * publishes a fully-formed `VehicleState` — grounded flags, slip ratios, slip
 * angles, suspension compression, per-wheel load and surface — and DriveFx is
 * fed exactly the same data it gets in the game. A scripted repertoire drives
 * that stand-in through the cases worth judging: cruising, a launch with
 * wheelspin, a handbrake slide, a jump and its landing, every surface in turn,
 * a water crossing and a rock scrape.
 *
 * Everything is deterministic and seek-able, so a screenshot taken at t=4.0 of
 * the slide is the same picture every time and A/B comparisons between tuning
 * passes are honest.
 */

import * as THREE from 'three';
import { DriveFx, type FxChannel } from './DriveFx';
import type { SurfaceKind, TerrainSampler, VehicleState, WheelState } from '../types';
import { clamp, smoothstep, lerp } from '../world/noise';

/* ================================================================== *
 * Renderer
 * ================================================================== */

const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
// Deliberately 1:1. The game renders its world at 360-540p inside the retro
// pipeline, so measuring particle fill cost at retina DPR would be measuring a
// resolution the game never uses.
renderer.setPixelRatio(1);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
// The game runs an untonemapped pipeline; matching it here means the FX are
// judged at the contrast they will actually ship at.
renderer.toneMapping = THREE.NoToneMapping;
renderer.toneMappingExposure = 1;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFShadowMap;
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(58, 1, 0.15, 2500);

/** Warm afternoon haze. The particles must sink into this, not glow through it. */
const FOG_COLOR = new THREE.Color(0xbfc6c0);
scene.fog = new THREE.FogExp2(FOG_COLOR.getHex(), 0.0042);
scene.fog.color.copy(FOG_COLOR);
scene.background = FOG_COLOR.clone();

// Three's lighting is energy-normalised: a directional light needs roughly a
// factor of pi more intensity than the old legacy-lights numbers to read as
// full daylight.
const sun = new THREE.DirectionalLight(0xfff0d8, 4.0);
sun.position.set(-42, 46, -30);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.camera.near = 1;
sun.shadow.camera.far = 220;
sun.shadow.camera.left = -60;
sun.shadow.camera.right = 60;
sun.shadow.camera.top = 60;
sun.shadow.camera.bottom = -60;
sun.shadow.bias = -0.0008;
sun.shadow.normalBias = 0.06;
scene.add(sun, sun.target);
scene.add(new THREE.HemisphereLight(0xa9c4e0, 0x54432c, 1.1));

window.addEventListener('resize', () => {
  renderer.setSize(window.innerWidth, window.innerHeight);
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
});
camera.aspect = window.innerWidth / window.innerHeight;
camera.updateProjectionMatrix();

/* ================================================================== *
 * Terrain
 * ================================================================== */

const WATER_LEVEL = -1.35;
const POND = { x: 0, z: 118, r: 38, floor: -1.95 };
const RAMP = { z: 42, up: 8, down: 2.6, half: 14, height: 2.7 };

const SURFACE_ORDER: SurfaceKind[] = ['snow', 'rock', 'mud', 'dirt', 'gravel', 'sand', 'grass'];
/** Metres per surface band, laid out along X with dirt on the centreline. */
const BAND = 32;

const GROUND_ALBEDO: Record<SurfaceKind, THREE.Color> = {
  dirt: new THREE.Color(0x6d5133),
  gravel: new THREE.Color(0x6f685c),
  sand: new THREE.Color(0xc0a76f),
  grass: new THREE.Color(0x4c6b29),
  mud: new THREE.Color(0x342718),
  rock: new THREE.Color(0x6a635a),
  snow: new THREE.Color(0xb6c4d6),
};

function baseHeight(x: number, z: number): number {
  return (
    1.15 * Math.sin(x * 0.031) * Math.cos(z * 0.026) +
    0.55 * Math.sin(x * 0.083 + 1.7) +
    0.42 * Math.cos(z * 0.071 - 0.6) +
    0.16 * Math.sin((x + z) * 0.21) +
    0.09 * Math.sin(x * 0.44 - z * 0.37)
  );
}

class PreviewTerrain implements TerrainSampler {
  readonly halfSize = 250;

  heightAt(x: number, z: number): number {
    let h = baseHeight(x, z);

    // Take-off ramp: a long smooth rise onto a crest with the far side cut
    // away, so a car at speed simply leaves it.
    const lat = 1 - smoothstep(0.62, 1, Math.abs(x) / RAMP.half);
    if (lat > 0) {
      if (z > RAMP.z - RAMP.up && z <= RAMP.z) {
        h += RAMP.height * lat * smoothstep(0, 1, (z - (RAMP.z - RAMP.up)) / RAMP.up);
      } else if (z > RAMP.z && z < RAMP.z + RAMP.down) {
        h += RAMP.height * lat * (1 - smoothstep(0, 1, (z - RAMP.z) / RAMP.down));
      }
    }

    // Pond basin.
    const d = Math.hypot(x - POND.x, z - POND.z) / POND.r;
    if (d < 1) {
      const w = 1 - smoothstep(0.68, 1, d);
      h = lerp(h, POND.floor, w);
    }
    return h;
  }

  normalAt(x: number, z: number, out?: THREE.Vector3): THREE.Vector3 {
    const v = out ?? new THREE.Vector3();
    const e = 0.6;
    const hl = this.heightAt(x - e, z);
    const hr = this.heightAt(x + e, z);
    const hd = this.heightAt(x, z - e);
    const hu = this.heightAt(x, z + e);
    return v.set(hl - hr, 2 * e, hd - hu).normalize();
  }

  surfaceAt(x: number, z: number): SurfaceKind {
    // A slow wobble keeps the band edges from reading as drawn-on stripes.
    const w = x + Math.sin(z * 0.055) * 5.5 + Math.sin(z * 0.017) * 3;
    const i = clamp(Math.round(w / BAND), -3, 3);
    return SURFACE_ORDER[i + 3]!;
  }
}

const terrain = new PreviewTerrain();

/* ---- ground mesh --------------------------------------------------- */

function buildGround(): THREE.Mesh {
  const SIZE = 520;
  const SEG = 384;
  const geo = new THREE.PlaneGeometry(SIZE, SIZE, SEG, SEG);
  geo.rotateX(-Math.PI / 2);
  const pos = geo.attributes.position as THREE.BufferAttribute;
  const colors = new Float32Array(pos.count * 3);
  const c = new THREE.Color();

  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i);
    const z = pos.getZ(i);
    pos.setY(i, terrain.heightAt(x, z));
    c.copy(GROUND_ALBEDO[terrain.surfaceAt(x, z)]!);
    // Grain, so the flat-shaded plane has something for the light to catch.
    const n = 0.86 + 0.28 * (Math.sin(x * 1.7) * Math.sin(z * 1.31) * 0.5 + 0.5);
    colors[i * 3] = c.r * n;
    colors[i * 3 + 1] = c.g * n;
    colors[i * 3 + 2] = c.b * n;
  }
  geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geo.computeVertexNormals();

  const mesh = new THREE.Mesh(
    geo,
    new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 1, metalness: 0 }),
  );
  mesh.receiveShadow = true;
  mesh.name = 'ground';
  return mesh;
}

scene.add(buildGround());

const water = new THREE.Mesh(
  new THREE.CircleGeometry(POND.r * 0.98, 64).rotateX(-Math.PI / 2),
  new THREE.MeshStandardMaterial({
    color: 0x3f6b74,
    roughness: 0.12,
    metalness: 0.2,
    transparent: true,
    opacity: 0.82,
  }),
);
water.position.set(POND.x, WATER_LEVEL, POND.z);
scene.add(water);

/* ---- scenery: something for the plumes to pass behind --------------- */

{
  const rockGeo = new THREE.IcosahedronGeometry(1, 0);
  const rockMat = new THREE.MeshStandardMaterial({ color: 0x6d675e, roughness: 0.95, flatShading: true });
  const rocks = new THREE.InstancedMesh(rockGeo, rockMat, 140);
  rocks.castShadow = true;
  rocks.receiveShadow = true;
  const m = new THREE.Matrix4();
  const q = new THREE.Quaternion();
  const s = new THREE.Vector3();
  const p = new THREE.Vector3();
  let seed = 1337;
  const rnd = (): number => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 4294967296;
  };
  for (let i = 0; i < 140; i++) {
    const x = (rnd() - 0.5) * 420;
    const z = (rnd() - 0.5) * 420;
    // Keep the driving lanes clear.
    const near = Math.abs(x % BAND) < 6 && Math.abs(z) < 150;
    const sc = 0.5 + rnd() * 2.4;
    p.set(x, terrain.heightAt(x, z) + sc * 0.35, z);
    q.setFromEuler(new THREE.Euler(rnd() * 3, rnd() * 6, rnd() * 3));
    s.set(sc, sc * (0.5 + rnd() * 0.5), sc);
    if (near) s.multiplyScalar(0.001);
    rocks.setMatrixAt(i, m.compose(p, q, s));
  }
  rocks.instanceMatrix.needsUpdate = true;
  scene.add(rocks);
}

/* ================================================================== *
 * Stand-in vehicle
 * ================================================================== */

const WHEEL_R = 0.4;
const SUSP_REST = 0.42;
const SUSP_TRAVEL = 0.42;
const RIDE = 0.673; // puts a static wheel at ~0.35 compression
const TRACK_HALF = 0.82;
const BASE_HALF = 1.35;

/** FL, FR, RL, RR. Forward is +Z in chassis space, matching Vehicle.ts. */
const MOUNTS: THREE.Vector3[] = [
  new THREE.Vector3(-TRACK_HALF, 0, BASE_HALF),
  new THREE.Vector3(TRACK_HALF, 0, BASE_HALF),
  new THREE.Vector3(-TRACK_HALF, 0, -BASE_HALF),
  new THREE.Vector3(TRACK_HALF, 0, -BASE_HALF),
];

interface Intent {
  speedTarget: number;
  accel: number;
  steer: number;
  slipRear: number;
  slipFront: number;
  slipLat: number;
  throttle: number;
}

function blankWheel(): WheelState {
  return {
    position: new THREE.Vector3(),
    quaternion: new THREE.Quaternion(),
    compression: 0.35,
    grounded: true,
    slipRatio: 0,
    slipAngle: 0,
    load: 4400,
    surface: 'dirt',
    spin: 0,
    steerAngle: 0,
  };
}

class Rig {
  readonly state: VehicleState = {
    position: new THREE.Vector3(),
    quaternion: new THREE.Quaternion(),
    velocity: new THREE.Vector3(),
    angularVelocity: new THREE.Vector3(),
    forwardSpeed: 0,
    speed: 0,
    engineRpm: 900,
    gear: 1,
    clutch: 1,
    throttle: 0,
    wheels: [blankWheel(), blankWheel(), blankWheel(), blankWheel()],
    airborne: false,
    localAccel: new THREE.Vector3(),
  };

  readonly intent: Intent = {
    speedTarget: 0, accel: 4, steer: 0, slipRear: 0, slipFront: 0, slipLat: 0, throttle: 0,
  };

  yaw = 0;
  speed = 0;
  slipLat = 0;
  y = 0;
  vy = 0;
  airborne = false;
  jumped = false;
  scrape = 0;
  pitch = 0;
  roll = 0;
  landing = 0;

  private readonly euler = new THREE.Euler(0, 0, 0, 'YXZ');
  private readonly mountWorld = [
    new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3(),
  ];
  private prevSpeed = 0;

  reset(x: number, z: number, yaw: number, speed: number): void {
    this.yaw = yaw;
    this.speed = speed;
    this.prevSpeed = speed;
    this.slipLat = 0;
    this.airborne = false;
    this.jumped = false;
    this.vy = 0;
    this.pitch = 0;
    this.roll = 0;
    this.landing = 0;
    this.scrape = 0;
    this.state.position.set(x, 0, z);
    this.y = terrain.heightAt(x, z) + RIDE;
    this.state.position.y = this.y;
    this.state.gear = 1;
    this.state.engineRpm = 900;
  }

  /** Leave the ground with the given upward velocity. */
  launch(vy: number): void {
    this.airborne = true;
    this.vy = vy;
  }

  step(dt: number): void {
    const it = this.intent;
    const st = this.state;

    // ---- planar motion ---------------------------------------------------
    const dv = clamp(it.speedTarget - this.speed, -it.accel * dt * 2.2, it.accel * dt);
    this.speed += dv;

    // Yaw authority grows with speed and again once the car is sideways —
    // enough to make a handbrake slide actually rotate.
    const grip = clamp(this.speed * 0.062, 0, 1.15);
    const yawRate = it.steer * grip * (1 + Math.abs(this.slipLat) * 1.7);
    this.yaw += yawRate * dt;
    this.slipLat += (it.slipLat - this.slipLat) * Math.min(1, dt * 4);

    const travelYaw = this.yaw - this.slipLat;
    // The vertical component matters: DriveFx sizes the landing puff from the
    // downward speed the suspension has to kill.
    st.velocity.set(Math.sin(travelYaw) * this.speed, this.vy, Math.cos(travelYaw) * this.speed);
    st.position.x += st.velocity.x * dt;
    st.position.z += st.velocity.z * dt;

    // ---- attitude --------------------------------------------------------
    const accel = (this.speed - this.prevSpeed) / Math.max(dt, 1e-4);
    this.prevSpeed = this.speed;
    const latAccel = yawRate * this.speed;

    // ---- vertical --------------------------------------------------------
    this.euler.set(0, this.yaw, 0);
    st.quaternion.setFromEuler(this.euler);

    let gSum = 0;
    let gFront = 0;
    let gRear = 0;
    let gLeft = 0;
    let gRight = 0;
    for (let i = 0; i < 4; i++) {
      const mw = this.mountWorld[i]!.copy(MOUNTS[i]!).applyQuaternion(st.quaternion);
      mw.x += st.position.x;
      mw.z += st.position.z;
      const g = terrain.heightAt(mw.x, mw.z);
      gSum += g;
      if (i < 2) gFront += g * 0.5; else gRear += g * 0.5;
      if (i % 2 === 0) gLeft += g * 0.5; else gRight += g * 0.5;
    }
    const gAvg = gSum * 0.25;
    const targetY = gAvg + RIDE;

    this.landing = Math.max(0, this.landing - dt * 3);
    if (this.airborne) {
      this.vy -= 9.81 * dt;
      this.y += this.vy * dt;
      if (this.y <= targetY) {
        this.landing = clamp(-this.vy / 9, 0, 1.6);
        this.y = targetY;
        this.vy = -this.vy * 0.12;
        this.airborne = false;
      }
    } else {
      // Critically damped follow: bumps compress the suspension visibly, which
      // is what the landing puff keys off.
      this.vy += ((targetY - this.y) * 46 - this.vy * 11) * dt;
      this.y += this.vy * dt;
      // Crest of the ramp: the ground drops away faster than the body can.
      if (this.y - targetY > 0.55 && this.vy > 1.0) this.airborne = true;
    }
    st.position.y = this.y;

    // Terrain-following pitch and roll plus weight transfer.
    const pitchT = Math.atan2(gRear - gFront, BASE_HALF * 2) - clamp(accel * 0.012, -0.09, 0.09);
    const rollT = Math.atan2(gRight - gLeft, TRACK_HALF * 2) + clamp(latAccel * 0.012, -0.16, 0.16);
    this.pitch += (pitchT - this.pitch) * Math.min(1, dt * 7);
    this.roll += (rollT - this.roll) * Math.min(1, dt * 7);
    this.euler.set(this.pitch, this.yaw, this.roll);
    st.quaternion.setFromEuler(this.euler);

    // ---- wheels ----------------------------------------------------------
    for (let i = 0; i < 4; i++) {
      const w = st.wheels[i]!;
      const mw = this.mountWorld[i]!.copy(MOUNTS[i]!).applyQuaternion(st.quaternion).add(st.position);
      const g = terrain.heightAt(mw.x, mw.z);
      const free = mw.y - SUSP_REST;
      const rest = g + WHEEL_R;
      const delta = rest - free;
      const grounded = !this.airborne && delta > -0.02;
      w.grounded = grounded;
      w.compression = clamp(delta / SUSP_TRAVEL, 0, 1);
      w.position.set(mw.x, grounded ? rest : free, mw.z);
      w.quaternion.copy(st.quaternion);

      const front = i < 2;
      // A little per-wheel decorrelation so the four plumes are not in lockstep.
      const jitter = 1 + (i - 1.5) * 0.06;
      w.slipRatio = grounded ? (front ? it.slipFront : it.slipRear) * jitter : 0;
      w.slipAngle = grounded ? this.slipLat + (front ? it.steer * 0.07 : 0) : 0;
      w.steerAngle = front ? it.steer * 0.5 : 0;
      w.spin = (this.speed / WHEEL_R) * (1 + w.slipRatio);
      w.surface = surfaceOverride ?? terrain.surfaceAt(mw.x, mw.z);

      // Load tracks compression, with a transfer term so the outside wheels
      // in a slide really are the ones digging in.
      const side = i % 2 === 0 ? -1 : 1;
      const transfer = clamp(1 + side * latAccel * 0.045 - (front ? 1 : -1) * accel * 0.02, 0.15, 2.2);
      w.load = grounded ? 4400 * (0.3 + w.compression * 2.0) * transfer * (1 + this.landing * 1.4) : 0;
    }

    // ---- drivetrain readout ---------------------------------------------
    st.speed = Math.abs(this.speed);
    st.forwardSpeed = this.speed;
    st.airborne = this.airborne;
    st.throttle = it.throttle;
    st.angularVelocity.set(0, yawRate, 0);
    st.localAccel.set(latAccel, 0, accel);

    const ratios = [0, 9, 17, 25, 34, 46];
    let gear = 1;
    for (let i = 1; i < ratios.length - 1; i++) if (st.speed >= ratios[i]!) gear = i + 1;
    st.gear = gear;
    const lo = ratios[gear - 1]!;
    const hi = ratios[gear]!;
    const frac = clamp((st.speed - lo) / Math.max(hi - lo, 1), 0, 1);
    const target = 1000 + frac * 4200 + Math.max(0, it.slipRear) * 1600 + it.throttle * 350;
    st.engineRpm += (target - st.engineRpm) * Math.min(1, dt * 6);
    st.clutch = st.speed < 1.5 && it.throttle > 0.5 ? 0.4 : 1;
  }
}

const rig = new Rig();

/* ---- car mesh ------------------------------------------------------- */

const carGroup = new THREE.Group();
{
  const paint = new THREE.MeshStandardMaterial({ color: 0x9b5a2c, roughness: 0.65, metalness: 0.1 });
  const dark = new THREE.MeshStandardMaterial({ color: 0x1d1f22, roughness: 0.8 });
  const glass = new THREE.MeshStandardMaterial({
    color: 0x27343a, roughness: 0.15, metalness: 0.3, transparent: true, opacity: 0.75,
  });

  const body = new THREE.Mesh(new THREE.BoxGeometry(1.86, 0.78, 3.9), paint);
  body.position.y = 0.62;
  const cab = new THREE.Mesh(new THREE.BoxGeometry(1.7, 0.72, 1.9), glass);
  cab.position.set(0, 1.28, -0.12);
  const roof = new THREE.Mesh(new THREE.BoxGeometry(1.74, 0.1, 1.94), paint);
  roof.position.set(0, 1.68, -0.12);
  const bar = new THREE.Mesh(new THREE.BoxGeometry(1.9, 0.22, 0.18), dark);
  bar.position.set(0, 0.72, 2.02);
  for (const m of [body, cab, roof, bar]) {
    m.castShadow = true;
    carGroup.add(m);
  }
}
scene.add(carGroup);

const wheelMeshes: THREE.Mesh[] = [];
{
  const geo = new THREE.CylinderGeometry(WHEEL_R, WHEEL_R, 0.34, 18);
  geo.rotateZ(Math.PI / 2);
  const mat = new THREE.MeshStandardMaterial({ color: 0x17181a, roughness: 0.95 });
  for (let i = 0; i < 4; i++) {
    const m = new THREE.Mesh(geo, mat);
    m.castShadow = true;
    wheelMeshes.push(m);
    scene.add(m);
  }
}

let wheelRoll = 0;
const AXIS_X = new THREE.Vector3(1, 0, 0);
const AXIS_Y = new THREE.Vector3(0, 1, 0);
const qSteer = new THREE.Quaternion();
const qSpin = new THREE.Quaternion();

function syncCarMesh(dt: number): void {
  carGroup.position.copy(rig.state.position);
  carGroup.quaternion.copy(rig.state.quaternion);
  wheelRoll += (rig.state.wheels[2]!.spin ?? 0) * dt;
  for (let i = 0; i < 4; i++) {
    const w = rig.state.wheels[i]!;
    const m = wheelMeshes[i]!;
    m.position.copy(w.position);
    // Chassis attitude, then steer about the suspension axis, then roll about
    // the (already steered) hub axis.
    m.quaternion.copy(rig.state.quaternion);
    if (w.steerAngle !== 0) m.quaternion.multiply(qSteer.setFromAxisAngle(AXIS_Y, w.steerAngle));
    m.quaternion.multiply(qSpin.setFromAxisAngle(AXIS_X, wheelRoll));
  }
}

/* ================================================================== *
 * Scenarios
 * ================================================================== */

interface Scenario {
  id: string;
  label: string;
  duration: number;
  start: { x: number; z: number; yaw: number; speed: number };
  drive(t: number, it: Intent, r: Rig): void;
  /** Camera framing that shows this scenario off. */
  camera?: 'chase' | 'broadside' | 'low';
}

const keys = new Set<string>();
window.addEventListener('keydown', (e) => keys.add(e.code));
window.addEventListener('keyup', (e) => keys.delete(e.code));

function manualDrive(_t: number, it: Intent, r: Rig): void {
  const fwd = keys.has('KeyW');
  const back = keys.has('KeyS');
  const hand = keys.has('ShiftLeft') || keys.has('ShiftRight');
  it.steer = (keys.has('KeyA') ? -0.6 : 0) + (keys.has('KeyD') ? 0.6 : 0);
  it.throttle = fwd ? 1 : 0;
  it.speedTarget = fwd ? 26 : back ? -7 : 0;
  it.accel = fwd ? 6 : back ? 7 : 3.5;
  it.slipFront = back ? -0.35 : 0.02;
  if (hand) {
    it.slipRear = -0.85;
    it.slipLat = it.steer * 1.05;
  } else {
    it.slipRear = fwd ? Math.max(0.04, 1.25 - r.speed * 0.085) : back ? -0.45 : 0.03;
    it.slipLat = it.steer * 0.12;
  }
}

const SCENARIOS: Scenario[] = [
  {
    id: 'cruise',
    label: 'Cruise · dirt',
    duration: 11,
    start: { x: 0, z: -95, yaw: 0, speed: 14 },
    camera: 'chase',
    drive(t, it) {
      it.speedTarget = 16;
      it.accel = 4;
      it.steer = Math.cos(t * 0.42) * 0.2;
      it.throttle = 0.34;
      it.slipRear = 0.08;
      it.slipFront = 0.02;
      it.slipLat = Math.cos(t * 0.42) * 0.05;
    },
  },
  {
    id: 'launch',
    label: 'Launch · wheelspin',
    duration: 10,
    start: { x: 0, z: -95, yaw: 0, speed: 0 },
    camera: 'broadside',
    drive(t, it, r) {
      const go = t > 0.7;
      it.throttle = go ? 1 : 0.15;
      it.speedTarget = go ? 27 : 0;
      it.accel = 5.5;
      it.steer = 0;
      // A big slip at zero road speed, hooking up as the car gets going.
      it.slipRear = go ? Math.max(0.05, 1.45 - r.speed * 0.075) : 0;
      it.slipFront = 0.02;
      it.slipLat = 0;
    },
  },
  {
    id: 'slide',
    label: 'Handbrake slide',
    duration: 13,
    start: { x: -12, z: -80, yaw: 0.2, speed: 19 },
    camera: 'broadside',
    drive(t, it) {
      if (t < 2.2) {
        it.speedTarget = 19; it.accel = 4; it.steer = 0.04; it.throttle = 0.4;
        it.slipRear = 0.06; it.slipFront = 0.02; it.slipLat = 0.03;
      } else {
        const k = clamp((t - 2.2) / 1.1, 0, 1);
        it.speedTarget = 13; it.accel = 3;
        it.steer = 0.5 * k + 0.05;
        it.throttle = 0.6;
        it.slipRear = -0.78 * k;
        it.slipFront = -0.12 * k;
        it.slipLat = 0.66 * k;
      }
    },
  },
  {
    id: 'jump',
    label: 'Jump + landing',
    duration: 12,
    start: { x: 0, z: 10, yaw: 0, speed: 20 },
    camera: 'broadside',
    drive(t, it, r) {
      it.speedTarget = 22; it.accel = 5; it.steer = 0; it.throttle = 0.75;
      it.slipRear = r.airborne ? 0.5 : 0.1;
      it.slipFront = 0.02;
      it.slipLat = 0;
      void t;
    },
  },
  {
    id: 'surfaces',
    label: 'Surface tour',
    duration: 15,
    start: { x: -112, z: -40, yaw: Math.PI / 2, speed: 17 },
    camera: 'chase',
    drive(t, it) {
      it.speedTarget = 17; it.accel = 5; it.steer = 0; it.throttle = 0.55;
      // Held at a constant moderate slip so the only thing changing across the
      // bands is the surface itself.
      it.slipRear = 0.34;
      it.slipFront = 0.05;
      it.slipLat = 0.17;
      void t;
    },
  },
  {
    id: 'water',
    label: 'Water crossing',
    duration: 12,
    start: { x: 0, z: 130, yaw: Math.PI / 2, speed: 7 },
    camera: 'broadside',
    drive(t, it) {
      it.speedTarget = 7.5; it.accel = 4; it.throttle = 0.5;
      // Circles inside the basin at a radius the pond can hold.
      it.steer = 0.95;
      it.slipRear = 0.16; it.slipFront = 0.04; it.slipLat = 0.1;
      void t;
    },
  },
  {
    id: 'rock',
    label: 'Rock · smoke + sparks',
    duration: 12,
    start: { x: -64, z: -80, yaw: 0, speed: 16 },
    camera: 'broadside',
    drive(t, it, r) {
      it.speedTarget = 15; it.accel = 4; it.throttle = 0.65;
      it.steer = Math.sin(t * 0.85) * 0.5;
      it.slipRear = 0.55 + Math.sin(t * 2.1) * 0.32;
      it.slipFront = -0.22;
      it.slipLat = Math.sin(t * 0.85) * 0.34;
      // Belly-pan scrapes on the way through.
      if (Math.floor(t / 1.7) !== Math.floor((t - 0.02) / 1.7)) r.scrape = 1;
    },
  },
  {
    id: 'manual',
    label: 'Manual (WASD)',
    duration: Infinity,
    start: { x: 0, z: -60, yaw: 0, speed: 0 },
    camera: 'chase',
    drive: manualDrive,
  },
];

/* ================================================================== *
 * FX
 * ================================================================== */

const fx = new DriveFx(scene, {
  terrain,
  maxParticles: 3400,
  waterLevel: WATER_LEVEL,
  wheelRadius: WHEEL_R,
});

let surfaceOverride: SurfaceKind | null = null;

/* ================================================================== *
 * Simulation driver
 * ================================================================== */

const SIM_DT = 1 / 120;

let scenarioIndex = 0;
let scenarioTime = 0;
let paused = false;
let loopScenarios = true;
let simAccum = 0;

function scenario(): Scenario {
  return SCENARIOS[scenarioIndex]!;
}

function restart(): void {
  const s = scenario();
  rig.reset(s.start.x, s.start.z, s.start.yaw, s.start.speed);
  const it = rig.intent;
  it.speedTarget = s.start.speed;
  it.accel = 5;
  it.steer = 0;
  it.slipRear = 0;
  it.slipFront = 0;
  it.slipLat = 0;
  it.throttle = 0;
  scenarioTime = 0;
  simAccum = 0;
  fx.clear();
  camMode = s.camera ?? 'chase';
  camFirst = true;
  syncCarMesh(0);
  updateBanner();
}

function setScenario(id: string): void {
  const i = SCENARIOS.findIndex((s) => s.id === id);
  if (i < 0) return;
  scenarioIndex = i;
  restart();
  markScenarioButtons();
}

function nextScenario(): void {
  scenarioIndex = (scenarioIndex + 1) % SCENARIOS.length;
  // Manual is opt-in; the auto rotation skips it.
  if (scenario().id === 'manual') scenarioIndex = 0;
  restart();
  markScenarioButtons();
}

const scratch = new THREE.Vector3();

/**
 * Advance one *rendered* frame's worth of time.
 *
 * The stand-in vehicle is integrated at a fixed rate for stability, but DriveFx
 * is ticked exactly once per frame with the whole frame's dt — the same way
 * Game.ts drives it. Returns the FX CPU cost in ms.
 */
function advance(dt: number): number {
  const s = scenario();
  simAccum += dt;
  let steps = 0;
  while (simAccum >= SIM_DT && steps < 16) {
    s.drive(scenarioTime, rig.intent, rig);
    rig.step(SIM_DT);
    scenarioTime += SIM_DT;
    simAccum -= SIM_DT;
    steps++;
  }

  const t0 = performance.now();
  fx.update(dt, rig.state);
  if (rig.scrape > 0) {
    // Chassis dragging over stone: a burst from under the nose.
    scratch.copy(rig.state.position);
    scratch.y = terrain.heightAt(scratch.x, scratch.z) + 0.05;
    fx.impact(scratch, 0.8, 'rock');
    rig.scrape = 0;
  }
  const cost = performance.now() - t0;

  if (scenarioTime > s.duration) {
    if (loopScenarios) nextScenario();
    else restart();
  }
  return cost;
}

/**
 * Run the current scenario deterministically from its start to `t`. This is
 * what makes screenshots reproducible: the particle field at t=4.0 is always
 * the same field.
 */
function seek(t: number): void {
  const wasLoop = loopScenarios;
  loopScenarios = false;
  restart();
  const frames = Math.min(Math.round(t * 60), 6000);
  for (let i = 0; i < frames; i++) advance(1 / 60);
  loopScenarios = wasLoop;
  syncCarMesh(SIM_DT);
  updateCamera(1, true);
}

/* ================================================================== *
 * Camera
 * ================================================================== */

type CamMode = 'chase' | 'broadside' | 'low';
let camMode: CamMode = 'chase';
let camFirst = true;
let orbitYaw = 0;
let orbitPitch = 0;
let camDist = 1;

const camPos = new THREE.Vector3();
const camTarget = new THREE.Vector3();
const camWant = new THREE.Vector3();

function updateCamera(dt: number, snap = false): void {
  const p = rig.state.position;
  camWant.set(p.x, p.y + 1.05, p.z);
  if (snap || camFirst) camTarget.copy(camWant);
  else camTarget.lerp(camWant, Math.min(1, dt * 5));

  let dist = 9.5;
  let height = 3.4;
  let yaw = rig.yaw + Math.PI + 0.5;
  if (camMode === 'broadside') {
    dist = 13;
    height = 3.2;
    yaw = rig.yaw + Math.PI * 0.5 + 0.35;
  } else if (camMode === 'low') {
    dist = 7.5;
    height = 1.0;
    yaw = rig.yaw + Math.PI - 0.5;
  }
  yaw += orbitYaw;
  dist *= camDist;
  height = height * camDist + orbitPitch * dist;

  camWant.set(
    camTarget.x + Math.sin(yaw) * dist,
    camTarget.y + height,
    camTarget.z + Math.cos(yaw) * dist,
  );
  // Never let the camera drop through the ground.
  const floor = terrain.heightAt(camWant.x, camWant.z) + 0.7;
  if (camWant.y < floor) camWant.y = floor;

  if (snap || camFirst) camPos.copy(camWant);
  else camPos.lerp(camWant, Math.min(1, dt * 4.5));
  camFirst = false;

  camera.position.copy(camPos);
  camera.lookAt(camTarget);

  // Follow the car with the shadow frustum.
  sun.position.set(p.x - 42, p.y + 46, p.z - 30);
  sun.target.position.copy(p);
  sun.target.updateMatrixWorld();
}

{
  let dragging = false;
  let lx = 0;
  let ly = 0;
  renderer.domElement.addEventListener('pointerdown', (e) => {
    dragging = true;
    lx = e.clientX;
    ly = e.clientY;
    renderer.domElement.setPointerCapture(e.pointerId);
  });
  renderer.domElement.addEventListener('pointerup', (e) => {
    dragging = false;
    renderer.domElement.releasePointerCapture(e.pointerId);
  });
  renderer.domElement.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    orbitYaw -= (e.clientX - lx) * 0.006;
    orbitPitch = clamp(orbitPitch - (e.clientY - ly) * 0.004, -0.25, 1.1);
    lx = e.clientX;
    ly = e.clientY;
  });
  renderer.domElement.addEventListener(
    'wheel',
    (e) => {
      camDist = clamp(camDist * (1 + e.deltaY * 0.0012), 0.3, 4);
      e.preventDefault();
    },
    { passive: false },
  );
}

/* ================================================================== *
 * GPU timer
 * ================================================================== */

/**
 * Real GPU time when the driver exposes timer queries. The rAF interval is
 * vsync-locked and useless as a cost measure on its own.
 */
class GpuTimer {
  private readonly gl: WebGL2RenderingContext;
  private ext: { TIME_ELAPSED_EXT: number; GPU_DISJOINT_EXT: number } | null = null;
  private pending: WebGLQuery[] = [];
  private active: WebGLQuery | null = null;
  ms = 0;
  supported = false;

  constructor(r: THREE.WebGLRenderer) {
    this.gl = r.getContext() as WebGL2RenderingContext;
    const ext = this.gl.getExtension('EXT_disjoint_timer_query_webgl2');
    if (ext) {
      this.ext = ext as unknown as { TIME_ELAPSED_EXT: number; GPU_DISJOINT_EXT: number };
      this.supported = true;
    }
  }

  begin(): void {
    if (!this.ext || this.active) return;
    const q = this.gl.createQuery();
    if (!q) return;
    this.gl.beginQuery(this.ext.TIME_ELAPSED_EXT, q);
    this.active = q;
  }

  end(): void {
    if (!this.ext || !this.active) return;
    this.gl.endQuery(this.ext.TIME_ELAPSED_EXT);
    this.pending.push(this.active);
    this.active = null;
    this.poll();
  }

  private poll(): void {
    if (!this.ext) return;
    const gl = this.gl;
    if (gl.getParameter(this.ext.GPU_DISJOINT_EXT)) {
      for (const q of this.pending) gl.deleteQuery(q);
      this.pending = [];
      return;
    }
    while (this.pending.length) {
      const q = this.pending[0]!;
      if (!gl.getQueryParameter(q, gl.QUERY_RESULT_AVAILABLE)) break;
      const ns = gl.getQueryParameter(q, gl.QUERY_RESULT) as number;
      this.ms = this.ms === 0 ? ns / 1e6 : this.ms * 0.88 + (ns / 1e6) * 0.12;
      gl.deleteQuery(q);
      this.pending.shift();
    }
  }
}

const gpuTimer = new GpuTimer(renderer);

/* ================================================================== *
 * UI
 * ================================================================== */

const ui = document.getElementById('ui') as HTMLDivElement;
const statsEl = document.getElementById('stats') as HTMLDivElement;
const bannerEl = document.getElementById('banner') as HTMLDivElement;
const hintEl = document.getElementById('hint') as HTMLDivElement;

const CHANNEL_LABELS: Array<[FxChannel, string]> = [
  ['dust', 'dust plumes'],
  ['rooster', 'rooster tail'],
  ['clods', 'clods / clippings'],
  ['smoke', 'tyre smoke'],
  ['tracks', 'tyre tracks'],
  ['landing', 'landing puffs'],
  ['exhaust', 'exhaust'],
  ['splash', 'water splash'],
  ['sparks', 'sparks'],
];

const scenarioButtons: HTMLButtonElement[] = [];
const surfaceButtons: HTMLButtonElement[] = [];
const channelBoxes = new Map<FxChannel, HTMLInputElement>();

function section(title: string): void {
  const h = document.createElement('h4');
  h.textContent = title;
  ui.appendChild(h);
}

function buttonRow(): HTMLDivElement {
  const d = document.createElement('div');
  d.className = 'btns';
  ui.appendChild(d);
  return d;
}

function slider(
  label: string,
  min: number,
  max: number,
  step: number,
  get: () => number,
  set: (v: number) => void,
): void {
  const wrap = document.createElement('div');
  wrap.className = 'slider';
  const head = document.createElement('div');
  const name = document.createElement('span');
  name.textContent = label;
  const val = document.createElement('span');
  const input = document.createElement('input');
  input.type = 'range';
  input.min = String(min);
  input.max = String(max);
  input.step = String(step);
  input.value = String(get());
  val.textContent = input.value;
  input.addEventListener('input', () => {
    const v = Number(input.value);
    set(v);
    val.textContent = v.toFixed(step < 1 ? 2 : 0);
  });
  head.append(name, val);
  wrap.append(head, input);
  ui.appendChild(wrap);
}

function markScenarioButtons(): void {
  scenarioButtons.forEach((b, i) => b.classList.toggle('on', i === scenarioIndex));
}

function markSurfaceButtons(): void {
  surfaceButtons.forEach((b) =>
    b.classList.toggle('on', (b.dataset.surface ?? '') === (surfaceOverride ?? '')),
  );
}

function updateBanner(): void {
  bannerEl.textContent = scenario().label;
}

section('Scenario');
{
  const box = buttonRow();
  SCENARIOS.forEach((s, i) => {
    const b = document.createElement('button');
    b.textContent = s.id;
    b.addEventListener('click', () => {
      scenarioIndex = i;
      restart();
      markScenarioButtons();
    });
    scenarioButtons.push(b);
    box.appendChild(b);
  });
  const box2 = buttonRow();
  const replay = document.createElement('button');
  replay.textContent = 'replay';
  replay.addEventListener('click', () => restart());
  const loopBtn = document.createElement('button');
  loopBtn.textContent = 'auto-advance';
  loopBtn.classList.add('on');
  loopBtn.addEventListener('click', () => {
    loopScenarios = !loopScenarios;
    loopBtn.classList.toggle('on', loopScenarios);
  });
  box2.append(replay, loopBtn);
  markScenarioButtons();
}

section('Effects');
for (const [id, label] of CHANNEL_LABELS) {
  const row = document.createElement('label');
  row.className = 'row';
  const box = document.createElement('input');
  box.type = 'checkbox';
  box.checked = fx.isChannelOn(id);
  const txt = document.createElement('span');
  txt.textContent = label;
  row.append(box, txt);
  box.addEventListener('change', () => fx.setChannel(id, box.checked));
  channelBoxes.set(id, box);
  ui.appendChild(row);
}
{
  const box = buttonRow();
  const all = document.createElement('button');
  all.textContent = 'all on';
  all.addEventListener('click', () => setAllChannels(true));
  const none = document.createElement('button');
  none.textContent = 'all off';
  none.addEventListener('click', () => setAllChannels(false));
  box.append(all, none);
}

section('Tuning');
slider('intensity', 0, 2, 0.05, () => 1, (v) => fx.setIntensity(v));
slider('track life (s)', 4, 90, 1, () => (fx.tracks ? fx.tracks.life : 42), (v) => {
  if (fx.tracks) fx.tracks.life = v;
});
slider('exposure', 0.5, 2, 0.01, () => renderer.toneMappingExposure, (v) => {
  renderer.toneMappingExposure = v;
});
slider('fog density', 0, 0.02, 0.0002, () => (scene.fog as THREE.FogExp2).density, (v) => {
  (scene.fog as THREE.FogExp2).density = v;
});

section('Force surface');
{
  const box = buttonRow();
  const mk = (label: string, s: SurfaceKind | null): void => {
    const b = document.createElement('button');
    b.textContent = label;
    b.dataset.surface = s ?? '';
    b.addEventListener('click', () => {
      surfaceOverride = s;
      markSurfaceButtons();
    });
    surfaceButtons.push(b);
    box.appendChild(b);
  };
  mk('auto', null);
  for (const s of ['dirt', 'sand', 'gravel', 'grass', 'mud', 'rock', 'snow'] as SurfaceKind[]) {
    mk(s, s);
  }
  markSurfaceButtons();
}

section('Camera');
{
  const box = buttonRow();
  for (const m of ['chase', 'broadside', 'low'] as CamMode[]) {
    const b = document.createElement('button');
    b.textContent = m;
    b.addEventListener('click', () => {
      camMode = m;
      orbitYaw = 0;
      orbitPitch = 0;
    });
    box.appendChild(b);
  }
  const box2 = buttonRow();
  const reset = document.createElement('button');
  reset.textContent = 'reset view';
  reset.addEventListener('click', () => {
    orbitYaw = 0;
    orbitPitch = 0;
    camDist = 1;
  });
  const measureBtn = document.createElement('button');
  measureBtn.textContent = 'measure cost';
  measureBtn.addEventListener('click', () => {
    void measure().then((r) => {
      measureResult = `gpu +${r.gpuMs.toFixed(2)} ms  cpu ${r.cpuMs.toFixed(2)} ms`;
    });
  });
  box2.append(reset, measureBtn);
}

function setAllChannels(on: boolean): void {
  for (const [id] of CHANNEL_LABELS) {
    fx.setChannel(id, on);
    const b = channelBoxes.get(id);
    if (b) b.checked = on;
  }
}

window.addEventListener('keydown', (e) => {
  switch (e.code) {
    case 'Space':
      paused = !paused;
      e.preventDefault();
      break;
    case 'KeyC':
      camMode = camMode === 'chase' ? 'broadside' : camMode === 'broadside' ? 'low' : 'chase';
      orbitYaw = 0;
      orbitPitch = 0;
      break;
    case 'KeyN':
      nextScenario();
      break;
    case 'KeyR':
      restart();
      break;
    case 'KeyH':
      hideUi(ui.style.display !== 'none');
      break;
    default:
      break;
  }
});

function hideUi(v: boolean): void {
  ui.style.display = v ? 'none' : '';
  statsEl.style.display = v ? 'none' : '';
  hintEl.style.display = v ? 'none' : '';
  bannerEl.style.display = v ? 'none' : '';
}

/* ================================================================== *
 * Cost measurement
 * ================================================================== */

let measureResult = '';

/**
 * Honest A/B: sample the GPU time with every effect off, then with everything
 * on, and report the difference. A single absolute number would mostly be
 * measuring the ground plane.
 */
function measure(): Promise<{ gpuMs: number; cpuMs: number }> {
  return new Promise((resolve) => {
    const restore = CHANNEL_LABELS.filter(([id]) => fx.isChannelOn(id)).map(([id]) => id);
    const settle = 55;
    let phase = 0;
    let n = 0;
    let offMs = 0;
    let onMs = 0;
    setAllChannels(false);

    const tick = (): void => {
      n++;
      if (phase === 0 && n > settle) {
        offMs = gpuTimer.ms;
        setAllChannels(true);
        phase = 1;
        n = 0;
      } else if (phase === 1 && n > settle) {
        onMs = gpuTimer.ms;
        setAllChannels(false);
        for (const id of restore) {
          fx.setChannel(id, true);
          const b = channelBoxes.get(id);
          if (b) b.checked = true;
        }
        measureHook = null;
        resolve({ gpuMs: Math.max(0, onMs - offMs), cpuMs: fxCpuMs });
        return;
      }
    };
    measureHook = tick;
  });
}

let measureHook: (() => void) | null = null;

/**
 * Timer-query benchmark that does not depend on requestAnimationFrame.
 *
 * A background tab has rAF throttled to nothing, so any measurement built on
 * the frame loop silently reports whatever it last saw. This submits N renders
 * back to back inside one GPU timer query and polls the result off a timeout,
 * which works whether or not the tab is on screen. It measures the scene twice
 * — once with every effect suppressed, once with them all on — because the
 * absolute number is mostly ground plane.
 */
async function benchGpu(frames = 80): Promise<{
  offMs: number; onMs: number; fxMs: number; cpuMs: number; particles: number; drawCalls: number;
}> {
  const gl = renderer.getContext() as WebGL2RenderingContext;
  const ext = gl.getExtension('EXT_disjoint_timer_query_webgl2') as
    | { TIME_ELAPSED_EXT: number }
    | null;

  const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));

  const run = async (): Promise<number> => {
    if (!ext) {
      const t0 = performance.now();
      for (let i = 0; i < frames; i++) renderer.render(scene, camera);
      gl.finish();
      return (performance.now() - t0) / frames;
    }
    const q = gl.createQuery()!;
    gl.beginQuery(ext.TIME_ELAPSED_EXT, q);
    for (let i = 0; i < frames; i++) renderer.render(scene, camera);
    gl.endQuery(ext.TIME_ELAPSED_EXT);
    gl.flush();
    for (let tries = 0; tries < 200; tries++) {
      if (gl.getQueryParameter(q, gl.QUERY_RESULT_AVAILABLE)) {
        const ns = gl.getQueryParameter(q, gl.QUERY_RESULT) as number;
        gl.deleteQuery(q);
        return ns / 1e6 / frames;
      }
      await sleep(10);
    }
    gl.deleteQuery(q);
    return 0;
  };

  const restore = CHANNEL_LABELS.filter(([id]) => fx.isChannelOn(id)).map(([id]) => id);
  const particles = fx.stats.particles;
  const drawCalls = renderer.info.render.calls;

  setAllChannels(false);
  const wasVisible = { air: true };
  void wasVisible;
  // Suppressing emission does not remove the particles already in flight, so
  // hide the pools outright for the baseline.
  for (const name of ['fx.air', 'fx.debris', 'fx.sparks', 'fx.tracks']) {
    const o = scene.getObjectByName(name);
    if (o) o.visible = false;
  }
  const offMs = await run();
  for (const name of ['fx.air', 'fx.debris', 'fx.sparks', 'fx.tracks']) {
    const o = scene.getObjectByName(name);
    if (o) o.visible = true;
  }
  const onMs = await run();

  setAllChannels(false);
  for (const id of restore) {
    fx.setChannel(id, true);
    const b = channelBoxes.get(id);
    if (b) b.checked = true;
  }

  // CPU: time the emission path on its own.
  const t0 = performance.now();
  for (let i = 0; i < 60; i++) fx.update(1 / 60, rig.state);
  const cpuMs = (performance.now() - t0) / 60;

  return { offMs, onMs, fxMs: Math.max(0, onMs - offMs), cpuMs, particles, drawCalls };
}

/* ================================================================== *
 * Loop
 * ================================================================== */

let last = performance.now();
let frameMs = 16.7;
let frameAcc = 0;
let frameCount = 0;
let fxCpuMs = 0;

function updateStats(): void {
  const s = fx.stats;
  const gpu = gpuTimer.supported ? `${gpuTimer.ms.toFixed(2)} ms` : 'n/a';
  const info = renderer.info.render;
  statsEl.innerHTML =
    `<b>scenario</b> ${scenario().id}  t=${scenarioTime.toFixed(1)}s\n` +
    `<b>speed</b>    ${(rig.state.speed * 3.6).toFixed(0)} km/h   rpm ${rig.state.engineRpm.toFixed(0)}\n` +
    `<b>slip</b>     R ${rig.state.wheels[2]!.slipRatio.toFixed(2)}  lat ${rig.state.wheels[2]!.slipAngle.toFixed(2)}\n` +
    `<b>surface</b>  ${rig.state.wheels[2]!.surface}${surfaceOverride ? ' (forced)' : ''}\n` +
    `\n` +
    `<b>air</b>      ${s.air}\n` +
    `<b>debris</b>   ${s.debris}\n` +
    `<b>sparks</b>   ${s.sparks}\n` +
    `<b>total</b>    ${s.particles} / ${s.capacity}\n` +
    `<b>tracks</b>   ${s.trackSegments} quads\n` +
    `\n` +
    `<b>fx cpu</b>   ${fxCpuMs.toFixed(3)} ms\n` +
    `<b>gpu</b>      ${gpu}\n` +
    `<b>frame</b>    ${frameMs.toFixed(2)} ms (${(1000 / Math.max(frameMs, 0.01)).toFixed(0)} fps)\n` +
    `<b>calls</b>    ${info.calls}  tris ${info.triangles.toLocaleString()}\n` +
    (measureResult ? `\n<b>fx cost</b>  ${measureResult}` : '');
}

function frame(now: number): void {
  requestAnimationFrame(frame);
  const raw = (now - last) / 1000;
  last = now;
  const dt = Math.min(raw, 0.1);

  if (!paused) {
    const cost = advance(dt);
    fxCpuMs = fxCpuMs === 0 ? cost : fxCpuMs * 0.9 + cost * 0.1;
    syncCarMesh(dt);
  }

  updateCamera(paused ? 1 : dt);

  gpuTimer.begin();
  renderer.render(scene, camera);
  gpuTimer.end();

  if (measureHook) measureHook();

  frameAcc += raw * 1000;
  frameCount++;
  if (frameCount >= 20) {
    frameMs = frameAcc / frameCount;
    frameAcc = 0;
    frameCount = 0;
    updateStats();
  }
}

restart();
requestAnimationFrame(frame);

/* ================================================================== *
 * Console / automation hook
 * ================================================================== */

declare global {
  interface Window {
    fxPreview: {
      fx: DriveFx;
      rig: Rig;
      scene: THREE.Scene;
      scenarios: string[];
      setScenario(id: string): void;
      seek(t: number): void;
      setPaused(v: boolean): void;
      setCamera(m: CamMode, dist?: number, yaw?: number, pitch?: number): void;
      setSurface(s: SurfaceKind | null): void;
      setChannel(id: FxChannel, on: boolean): void;
      setIntensity(v: number): void;
      hideUi(v: boolean): void;
      stats(): Record<string, number | string>;
      measure(): Promise<{ gpuMs: number; cpuMs: number }>;
      benchGpu(frames?: number): Promise<{
        offMs: number; onMs: number; fxMs: number; cpuMs: number;
        particles: number; drawCalls: number;
      }>;
    };
  }
}

window.fxPreview = {
  fx,
  rig,
  scene,
  scenarios: SCENARIOS.map((s) => s.id),
  setScenario,
  seek,
  setPaused: (v: boolean) => {
    paused = v;
  },
  setCamera: (m: CamMode, dist?: number, yaw?: number, pitch?: number) => {
    camMode = m;
    if (dist !== undefined) camDist = dist;
    if (yaw !== undefined) orbitYaw = yaw;
    if (pitch !== undefined) orbitPitch = pitch;
    updateCamera(1, true);
  },
  setSurface: (s: SurfaceKind | null) => {
    surfaceOverride = s;
    markSurfaceButtons();
  },
  setChannel: (id: FxChannel, on: boolean) => {
    fx.setChannel(id, on);
    const b = channelBoxes.get(id);
    if (b) b.checked = on;
  },
  setIntensity: (v: number) => fx.setIntensity(v),
  hideUi,
  stats: () => {
    const s = fx.stats;
    return {
      scenario: scenario().id,
      t: Number(scenarioTime.toFixed(2)),
      speedKmh: Number((rig.state.speed * 3.6).toFixed(1)),
      air: s.air,
      debris: s.debris,
      sparks: s.sparks,
      particles: s.particles,
      capacity: s.capacity,
      trackQuads: s.trackSegments,
      fxCpuMs: Number(fxCpuMs.toFixed(3)),
      gpuMs: Number(gpuTimer.ms.toFixed(3)),
      frameMs: Number(frameMs.toFixed(2)),
      drawCalls: renderer.info.render.calls,
    };
  },
  measure,
  benchGpu,
};
