/**
 * Turntable preview for `JeepModel`.
 *
 * This exists to be *looked at*. It drives the model through a synthetic
 * `VehicleState` built by the same arithmetic the physics uses to publish wheel
 * poses (attach point + chassis-down * suspension length), so what you see here
 * is exactly what the game will see — including the failure modes. The
 * articulation and steering sweeps are the point: a static hero shot hides
 * every rigging bug there is.
 *
 * Run:  npx vite --port 5190
 *       http://127.0.0.1:5190/src/vehicle/jeep.preview.html
 */
import * as THREE from 'three';

import type { VehicleState, WheelState } from '../types';
import { JeepModel } from './JeepModel';
import { countTriangles } from './jeepGeometry';
import { getMap } from '../render/textures';
import { JEEP_TUNING } from '../physics/VehicleTuning';

// --- the numbers the physics uses ---
// Read, not retyped. These were hard-coded once and silently went stale the
// first time the suspension was retuned, which made the preview show an
// articulation range the car does not have — precisely the class of bug this
// page exists to catch.
const REST_LENGTH = JEEP_TUNING.suspension.restLength;
const TRAVEL = JEEP_TUNING.suspension.travel;
const WHEEL_R = JEEP_TUNING.tire.radius;
/** Static sag, derived the same way the tuning derives the spring rates. */
const STATIC_SAG = REST_LENGTH - (JEEP_TUNING.wheels[0]!.attach.y - WHEEL_R);

const COLORS = ['#7a2f22', '#2f5d3a', '#b8912f', '#26405e', '#3a3d40', '#8a4a1c'];

type Mode = 'static' | 'articulate' | 'steer' | 'both' | 'drive';
const MODES: Mode[] = ['static', 'articulate', 'steer', 'both', 'drive'];

interface View {
  name: string;
  az: number;
  el: number;
  dist: number;
  height: number;
}

const VIEWS: View[] = [
  { name: 'front', az: 0, el: 0.1, dist: 6.6, height: 1.0 },
  { name: 'side', az: Math.PI / 2, el: 0.08, dist: 7.2, height: 1.0 },
  { name: 'rear', az: Math.PI, el: 0.1, dist: 6.6, height: 1.0 },
  { name: 'front 3/4', az: -0.72, el: 0.2, dist: 7.0, height: 1.05 },
  { name: 'rear 3/4', az: Math.PI + 0.72, el: 0.2, dist: 7.0, height: 1.05 },
  { name: 'low front 3/4', az: -0.6, el: -0.02, dist: 5.6, height: 0.55 },
  { name: 'high 3/4', az: -0.9, el: 0.55, dist: 7.4, height: 1.2 },
  { name: 'underside', az: -0.5, el: -0.42, dist: 6.0, height: 0.9 },
];

// ---------------------------------------------------------------------------
// renderer / scene
// ---------------------------------------------------------------------------

const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(2, devicePixelRatio));
renderer.setSize(innerWidth, innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.NoToneMapping;
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x5b7590);
scene.fog = new THREE.Fog(0x6d86a0, 26, 80);

const camera = new THREE.PerspectiveCamera(38, innerWidth / innerHeight, 0.1, 300);

// Warm key sun, cool sky fill: enough separation that the chamfers on the
// bodywork actually catch a different value on each face, which is the whole
// reason for chamfering them.
const sun = new THREE.DirectionalLight(0xfff0d2, 2.6);
sun.position.set(-5.5, 8.5, 6.5);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.camera.near = 1;
sun.shadow.camera.far = 30;
sun.shadow.camera.left = -5;
sun.shadow.camera.right = 5;
sun.shadow.camera.top = 5;
sun.shadow.camera.bottom = -5;
sun.shadow.bias = -0.0009;
sun.shadow.normalBias = 0.02;
scene.add(sun);
scene.add(sun.target);

const rim = new THREE.DirectionalLight(0xa8c4e8, 0.85);
rim.position.set(7, 4.5, -6.5);
scene.add(rim);

scene.add(new THREE.HemisphereLight(0xbcd6ee, 0x6a5540, 1.15));
scene.add(new THREE.AmbientLight(0xffffff, 0.22));

// Ground: real dirt so the tyres have something to read against.
const ground = new THREE.Mesh(
  new THREE.CircleGeometry(26, 48),
  new THREE.MeshLambertMaterial({ map: getMap('dirt', { repeat: 14 }), color: 0xc4b49c }),
);
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
scene.add(ground);

const grid = new THREE.GridHelper(20, 40, 0x555f66, 0x333b41);
(grid.material as THREE.Material).transparent = true;
(grid.material as THREE.Material).opacity = 0.28;
grid.position.y = 0.002;
scene.add(grid);

// ---------------------------------------------------------------------------
// model
// ---------------------------------------------------------------------------

let colorIndex = 0;
let jeep = new JeepModel({ color: COLORS[colorIndex]! });
scene.add(jeep.object3d);

const turntable = new THREE.Group();

function rebuild(): void {
  scene.remove(jeep.object3d);
  jeep.dispose();
  jeep = new JeepModel({ color: COLORS[colorIndex]! });
  scene.add(jeep.object3d);
  applyWireframe();
}

// ---------------------------------------------------------------------------
// synthetic vehicle state
// ---------------------------------------------------------------------------

function makeWheel(): WheelState {
  return {
    position: new THREE.Vector3(),
    quaternion: new THREE.Quaternion(),
    compression: 0.48,
    grounded: true,
    slipRatio: 0,
    slipAngle: 0,
    load: 4600,
    surface: 'dirt',
    spin: 0,
    steerAngle: 0,
  };
}

const state: VehicleState = {
  position: new THREE.Vector3(0, 0, 0),
  quaternion: new THREE.Quaternion(),
  velocity: new THREE.Vector3(),
  angularVelocity: new THREE.Vector3(),
  forwardSpeed: 0,
  speed: 0,
  engineRpm: 780,
  gear: 1,
  clutch: 1,
  wheels: [makeWheel(), makeWheel(), makeWheel(), makeWheel()],
  airborne: false,
  localAccel: new THREE.Vector3(),
};

/**
 * Attach points, body-local, straight out of the tuning. Order is fixed there:
 * index 0 and 1 are the FRONT pair (positive Z, the steered axle), 2 and 3 the
 * rear — which is the mapping `driveState` relies on when it steers `i < 2`.
 */
const ATTACH: THREE.Vector3[] = JEEP_TUNING.wheels.map(
  (w) => new THREE.Vector3(w.attach.x, w.attach.y, w.attach.z),
);

const _v = new THREE.Vector3();
const _down = new THREE.Vector3();
const _qs = new THREE.Quaternion();
const _qp = new THREE.Quaternion();
const AXIS_X = new THREE.Vector3(1, 0, 0);
const AXIS_Y = new THREE.Vector3(0, 1, 0);

const spinAngle = [0, 0, 0, 0];

/**
 * Reproduces `Vehicle.updateWheelVisuals`: hub = attach + chassisDown * length,
 * quaternion = chassis * steer * spin. If the model looks right against this it
 * will look right in the game.
 */
function driveState(t: number, mode: Mode, dt: number): void {
  // Suspension length per wheel, 0 = fully compressed .. TRAVEL = full droop.
  const len: number[] = [0, 0, 0, 0];
  const staticLen = REST_LENGTH - STATIC_SAG;

  let steer = 0;
  let spin = 0;
  let braking = 0;
  let reverse = false;

  if (mode === 'articulate' || mode === 'both') {
    // Opposed diagonal: the RTI-ramp pose, which shows every link working.
    const s = Math.sin(t * 0.9);
    const c = Math.sin(t * 0.9 + Math.PI * 0.5);
    const amp = TRAVEL * 0.5;
    len[0] = staticLen + amp * s;
    len[1] = staticLen - amp * s;
    len[2] = staticLen - amp * c;
    len[3] = staticLen + amp * c;
  } else {
    for (let i = 0; i < 4; i++) len[i] = staticLen;
  }

  if (mode === 'steer' || mode === 'both') {
    steer = Math.sin(t * 0.8) * 0.6632;
  }

  if (mode === 'drive') {
    steer = Math.sin(t * 0.55) * 0.45;
    spin = 12 + Math.sin(t * 0.4) * 6;
    // Ripple the wheels as if crossing washboard.
    for (let i = 0; i < 4; i++) {
      len[i] = staticLen + Math.sin(t * 5 + i * 1.9) * TRAVEL * 0.16;
    }
    braking = Math.max(0, Math.sin(t * 0.5 - 1.2));
    reverse = Math.sin(t * 0.23) < -0.6;
  }

  // Chassis attitude follows the average wheel drop, so the body never floats
  // free of the axles during an articulation sweep.
  const avg = (len[0]! + len[1]! + len[2]! + len[3]!) / 4;
  const roll = ((len[0]! + len[2]!) - (len[1]! + len[3]!)) * 0.16;
  const pitch = ((len[2]! + len[3]!) - (len[0]! + len[1]!)) * 0.1;
  _qs.setFromAxisAngle(new THREE.Vector3(0, 0, 1), roll);
  _qp.setFromAxisAngle(AXIS_X, pitch);
  state.quaternion.copy(_qs).multiply(_qp);
  state.position.set(0, WHEEL_R + (staticLen - avg) * 0.35, 0);

  _down.set(0, -1, 0).applyQuaternion(state.quaternion);

  for (let i = 0; i < 4; i++) {
    const w = state.wheels[i]!;
    const steered = i < 2;
    w.steerAngle = steered ? steer : 0;
    w.spin = spin;
    w.compression = 1 - (len[i]! - (REST_LENGTH - TRAVEL)) / TRAVEL;
    spinAngle[i] = (spinAngle[i]! + spin * dt) % (Math.PI * 2);

    _v.copy(ATTACH[i]!).applyQuaternion(state.quaternion).add(state.position);
    w.position.copy(_v).addScaledVector(_down, len[i]!);

    _qs.setFromAxisAngle(AXIS_Y, w.steerAngle);
    _qp.setFromAxisAngle(AXIS_X, spinAngle[i]!);
    w.quaternion.copy(state.quaternion).multiply(_qs).multiply(_qp);
  }

  state.speed = Math.abs(spin) * WHEEL_R;
  state.forwardSpeed = reverse ? -state.speed : state.speed;
  state.gear = reverse ? -1 : 2;
  state.engineRpm = 780 + state.speed * 130;
  state.localAccel.set(0, 0, -braking * 0.6 * (reverse ? -1 : 1));
}

// ---------------------------------------------------------------------------
// controls
// ---------------------------------------------------------------------------

let modeIndex = 1;
let viewIndex = 3;
let spinning = true;
let wireframe = false;
let showGrid = true;
let headlights = true;
let az = VIEWS[viewIndex]!.az;
let animT = 0;

function applyWireframe(): void {
  jeep.object3d.traverse((o) => {
    const m = o as THREE.Mesh;
    if (!m.isMesh) return;
    const mats = Array.isArray(m.material) ? m.material : [m.material];
    for (const mat of mats) {
      const mm = mat as THREE.MeshPhongMaterial;
      if ('wireframe' in mm) mm.wireframe = wireframe;
    }
  });
}

addEventListener('keydown', (e) => {
  const k = e.key.toLowerCase();
  if (k >= '1' && k <= '8') {
    viewIndex = Number(k) - 1;
    az = VIEWS[viewIndex]!.az;
    return;
  }
  switch (k) {
    case ' ':
      spinning = !spinning;
      e.preventDefault();
      break;
    case 'a':
      modeIndex = modeIndex === 1 ? 0 : 1;
      break;
    case 's':
      modeIndex = modeIndex === 2 ? 0 : 2;
      break;
    case 'd':
      modeIndex = modeIndex === 4 ? 0 : 4;
      break;
    case 'm':
      modeIndex = (modeIndex + 1) % MODES.length;
      break;
    case 'w':
      wireframe = !wireframe;
      applyWireframe();
      break;
    case 'g':
      showGrid = !showGrid;
      grid.visible = showGrid;
      break;
    case 'l':
      headlights = !headlights;
      jeep.setHeadlights(headlights);
      break;
    case 'c':
      colorIndex = (colorIndex + 1) % COLORS.length;
      rebuild();
      jeep.setHeadlights(headlights);
      break;
    case 'r':
      animT = 0;
      break;
    default:
      break;
  }
});

addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

// Drag to orbit, wheel to dolly — for close inspection of a suspect detail.
let dragging = false;
let lastX = 0;
let lastY = 0;
let elBias = 0;
let distBias = 0;
renderer.domElement.addEventListener('pointerdown', (e) => {
  dragging = true;
  lastX = e.clientX;
  lastY = e.clientY;
});
addEventListener('pointerup', () => {
  dragging = false;
});
addEventListener('pointermove', (e) => {
  if (!dragging) return;
  az -= (e.clientX - lastX) * 0.006;
  elBias = Math.max(-0.9, Math.min(1.2, elBias + (e.clientY - lastY) * 0.004));
  lastX = e.clientX;
  lastY = e.clientY;
});
renderer.domElement.addEventListener(
  'wheel',
  (e) => {
    distBias = Math.max(-3.5, Math.min(12, distBias + e.deltaY * 0.004));
    e.preventDefault();
  },
  { passive: false },
);

// ---------------------------------------------------------------------------
// hud
// ---------------------------------------------------------------------------

const statsEl = document.getElementById('stats')!;
const keysEl = document.getElementById('keys')!;
const modeEl = document.getElementById('mode')!;

keysEl.innerHTML =
  '<b>JEEP MODEL</b>\n' +
  '<i>1-8</i>  camera preset\n' +
  '<i>space</i> pause turntable\n' +
  '<i>a</i>    articulation sweep\n' +
  '<i>s</i>    steering lock-to-lock\n' +
  '<i>m</i>    cycle mode\n' +
  '<i>d</i>    drive (spin + lights)\n' +
  '<i>w</i>    wireframe\n' +
  '<i>g</i>    grid\n' +
  '<i>l</i>    headlights\n' +
  '<i>c</i>    next paint colour\n' +
  '<i>drag</i> orbit  <i>wheel</i> zoom';

let hudAcc = 0;
let frames = 0;
let fps = 0;

function updateHud(dt: number): void {
  hudAcc += dt;
  frames++;
  if (hudAcc < 0.25) return;
  fps = frames / hudAcc;
  hudAcc = 0;
  frames = 0;

  const info = renderer.info.render;
  const modelTris = countTriangles(jeep.object3d);
  statsEl.innerHTML =
    `triangles  <b>${modelTris.toLocaleString()}</b>\n` +
    `scene tris ${info.triangles.toLocaleString()}\n` +
    `draw calls ${info.calls}\n` +
    `fps        ${fps.toFixed(0)}\n` +
    `view       ${VIEWS[viewIndex]!.name}\n` +
    `paint      ${COLORS[colorIndex]}`;
  modeEl.textContent = `${MODES[modeIndex]}${spinning ? ' · turntable' : ''}`;
}

// ---------------------------------------------------------------------------
// loop
// ---------------------------------------------------------------------------

const clock = new THREE.Clock();

function frame(): void {
  requestAnimationFrame(frame);
  const dt = Math.min(0.05, clock.getDelta());
  animT += dt;

  const mode = MODES[modeIndex]!;
  driveState(animT, mode, dt);
  jeep.update(state, dt);

  if (spinning) az += dt * 0.32;
  const v = VIEWS[viewIndex]!;
  const el = Math.max(-0.55, v.el + elBias);
  const dist = Math.max(3.2, v.dist + distBias);
  camera.position.set(
    Math.sin(az) * Math.cos(el) * dist,
    v.height + Math.sin(el) * dist,
    Math.cos(az) * Math.cos(el) * dist,
  );
  camera.lookAt(0, v.height * 0.82, 0);

  // Studio key light: keep the sun over the camera's shoulder so whichever face
  // the turntable presents is the one that is lit. A fixed sun means half the
  // presets are a critique of the shadow side rather than of the model.
  const sunAz = az + 0.62;
  sun.position.set(Math.sin(sunAz) * 7, 9.5, Math.cos(sunAz) * 7);
  sun.target.position.set(0, 0.85, 0);

  renderer.render(scene, camera);
  updateHud(dt);
}

scene.add(turntable);
frame();

// Expose for the console / screenshot tooling.
Object.assign(globalThis as Record<string, unknown>, {
  jeep,
  setView: (i: number) => {
    viewIndex = i;
    az = VIEWS[i]!.az;
  },
  setMode: (i: number) => {
    modeIndex = i;
  },
  setSpin: (on: boolean) => {
    spinning = on;
  },
  setAz: (a: number) => {
    az = a;
  },
  triangles: () => countTriangles(jeep.object3d),
});
