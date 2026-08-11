import * as THREE from 'three';
import { Engine } from '../core/Engine';
import { RetroPipeline } from './RetroPipeline';
import type { RetroOptions } from './RetroPipeline';
import { Sky, SKY_PRESETS } from '../world/Sky';
import { Lighting } from './Lighting';
import { getMap, getTexture } from './textures';
import { Noise2D } from '../world/noise';

/**
 * Visual test rig for the render/atmosphere layer.
 *
 * A representative outdoor scene — rolling textured ground, rocks and posts
 * casting shadows, a camera that keeps moving so shadow crawl has somewhere to
 * show itself — plus a toggle for every pass and every time-of-day preset.
 * The point is to be able to A/B any single pass in isolation.
 */

const engine = new Engine();
const pipeline = new RetroPipeline(engine);

const sky = new Sky(engine.scene, pipeline.atmosphere, {
  onChange: (s) => {
    // Fog density is part of the same atmosphere decision as the sky colour,
    // so it is pushed from here rather than tuned independently.
    pipeline.setOption('fogDensity', s.fogDensity);
    pipeline.setOption('fogHeightFalloff', s.fogHeightFalloff);
    pipeline.setOption('fogStart', s.fogStart);
  },
});
const lighting = new Lighting(engine.scene, sky, { shadowRadius: 52, mapSize: 2048 });

engine.camera.far = 3000;
engine.camera.updateProjectionMatrix();

// ------------------------------------------------------------------- scene

const noise = new Noise2D(20250808);

/** Rolling terrain: broad swell, a mid octave for shape, grain on top. */
function terrainHeight(x: number, z: number): number {
  let h = 0;
  h += noise.sample(x * 0.0032, z * 0.0032) * 26;
  h += noise.sample(x * 0.0081, z * 0.0081) * 9.5;
  h += noise.sample(x * 0.021, z * 0.021) * 2.6;
  h += noise.sample(x * 0.06, z * 0.06) * 0.6;
  // Flatten a driveable valley down the middle so there is a road-like surface
  // to judge texture filtering and shadow contact against.
  const valley = Math.exp(-((x * x) / (85 * 85)));
  return h * (1 - 0.72 * valley);
}

const GRASS = new THREE.Color(0.44, 0.78, 0.30);
const STONE = new THREE.Color(0.86, 0.84, 0.80);

function buildGround(): THREE.Mesh {
  const SIZE = 1500;
  const SEG = 260;
  const geo = new THREE.PlaneGeometry(SIZE, SIZE, SEG, SEG);
  geo.rotateX(-Math.PI / 2);
  const pos = geo.attributes.position as THREE.BufferAttribute;

  for (let i = 0; i < pos.count; i++) {
    pos.setY(i, terrainHeight(pos.getX(i), pos.getZ(i)));
  }
  geo.computeVertexNormals();

  // Vertex tint over a single dirt albedo: green where flat and low, pale rock
  // where steep. Cheap, and it is exactly what the era actually did.
  const nrm = geo.attributes.normal as THREE.BufferAttribute;
  const colors = new Float32Array(pos.count * 3);
  const c = new THREE.Color();
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i);
    const y = pos.getY(i);
    const z = pos.getZ(i);
    const slope = 1 - nrm.getY(i);
    const patch = noise.sample(x * 0.011 + 40, z * 0.011 - 17) * 0.5 + 0.5;
    const grass = THREE.MathUtils.clamp(patch * 1.35 - slope * 4.5 - y * 0.014, 0, 1);
    const rock = THREE.MathUtils.smoothstep(slope, 0.16, 0.45);
    c.setRGB(0.98, 0.94, 0.86, THREE.LinearSRGBColorSpace);
    c.lerp(GRASS, grass * 0.85);
    c.lerp(STONE, rock * 0.8);
    colors[i * 3] = c.r;
    colors[i * 3 + 1] = c.g;
    colors[i * 3 + 2] = c.b;
  }
  geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

  const set = getTexture('dirt', { repeat: [160, 160] });
  const mesh = new THREE.Mesh(
    geo,
    new THREE.MeshLambertMaterial({ map: set.map, vertexColors: true }),
  );
  mesh.receiveShadow = true;
  mesh.name = 'ground';
  return mesh;
}

engine.scene.add(buildGround());

// --- props ------------------------------------------------------------------

const props = new THREE.Group();
engine.scene.add(props);

const rockMat = new THREE.MeshLambertMaterial({ map: getMap('rock', { repeat: [2, 2] }) });
const crateMat = new THREE.MeshLambertMaterial({ map: getMap('planks') });
const postMat = new THREE.MeshLambertMaterial({ map: getMap('corrugatedMetal', { repeat: [1, 3] }) });
const scrubMat = new THREE.MeshLambertMaterial({
  map: getMap('scrub'),
  transparent: true,
  alphaTest: 0.45,
  side: THREE.DoubleSide,
});

const rockGeo = new THREE.DodecahedronGeometry(1, 0);
const crateGeo = new THREE.BoxGeometry(1, 1, 1);
const postGeo = new THREE.BoxGeometry(0.16, 3.2, 0.16);
const scrubGeo = new THREE.PlaneGeometry(3, 2.4);

let seed = 991;
const rnd = (): number => {
  seed = (seed * 1664525 + 1013904223) >>> 0;
  return seed / 4294967296;
};

function place(mesh: THREE.Mesh, x: number, z: number, yOffset: number): void {
  mesh.position.set(x, terrainHeight(x, z) + yOffset, z);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  props.add(mesh);
}

for (let i = 0; i < 220; i++) {
  const x = (rnd() - 0.5) * 620;
  const z = (rnd() - 0.5) * 900;
  const s = 0.5 + rnd() * 3.2;
  const m = new THREE.Mesh(rockGeo, rockMat);
  m.scale.set(s * (0.8 + rnd() * 0.5), s * (0.55 + rnd() * 0.5), s * (0.8 + rnd() * 0.5));
  m.rotation.set(rnd() * 3, rnd() * 6, rnd() * 3);
  place(m, x, z, s * 0.32);
}

for (let i = 0; i < 34; i++) {
  const x = (rnd() - 0.5) * 160;
  const z = (rnd() - 0.5) * 700;
  const s = 0.8 + rnd() * 1.4;
  const m = new THREE.Mesh(crateGeo, crateMat);
  m.scale.setScalar(s);
  m.rotation.y = rnd() * 6.283;
  place(m, x, z, s * 0.5);
}

// Marker posts along the valley: thin verticals are the harshest test for
// shadow crawl, because a one-texel wobble is instantly visible on them.
for (let i = 0; i < 60; i++) {
  const side = i % 2 === 0 ? -1 : 1;
  const z = -420 + i * 14;
  const x = side * (11 + Math.sin(z * 0.02) * 4);
  place(new THREE.Mesh(postGeo, postMat), x, z, 1.6);
}

for (let i = 0; i < 300; i++) {
  const x = (rnd() - 0.5) * 700;
  const z = (rnd() - 0.5) * 950;
  const m = new THREE.Mesh(scrubGeo, scrubMat);
  m.rotation.y = rnd() * 6.283;
  m.scale.setScalar(0.6 + rnd() * 0.9);
  m.position.set(x, terrainHeight(x, z) + 0.9, z);
  m.castShadow = true;
  props.add(m);
}

// Stand-in for the car: something to keep in frame that shadows read against.
const hero = new THREE.Mesh(
  new THREE.BoxGeometry(2.0, 1.3, 4.2),
  new THREE.MeshPhongMaterial({
    map: getMap('paintedMetal'),
    shininess: 26,
    specular: 0x333333,
  }),
);
hero.castShadow = true;
hero.receiveShadow = true;
engine.scene.add(hero);

// -------------------------------------------------------------- camera path

let camT = 0;
let camPaused = false;
const focus = new THREE.Vector3();
const camDir = new THREE.Vector3();
const _p0 = new THREE.Vector3();
const _p1 = new THREE.Vector3();
const _back = new THREE.Vector3();

function pathPoint(t: number, out: THREE.Vector3): THREE.Vector3 {
  const z = -380 + ((t * 30) % 780);
  const x = Math.sin(z * 0.0125) * 22 + Math.sin(z * 0.004) * 12;
  return out.set(x, terrainHeight(x, z) + 1.0, z);
}

function updateCamera(dt: number): void {
  if (!camPaused) camT += dt;
  pathPoint(camT, _p0);
  pathPoint(camT + 0.9, _p1);

  hero.position.copy(_p0);
  hero.position.y += 0.65;
  hero.lookAt(_p1.x, _p1.y + 0.65, _p1.z);

  _back.copy(_p0).sub(_p1).normalize().multiplyScalar(9.5);
  engine.camera.position.copy(_p0).add(_back);
  engine.camera.position.y += 3.4;
  engine.camera.lookAt(_p1.x, _p1.y + 1.2, _p1.z);
  focus.copy(_p0);
  engine.camera.getWorldDirection(camDir);
}

// ---------------------------------------------------------------- gpu timer

/**
 * Real GPU time for the whole chain when the driver exposes timer queries,
 * otherwise the CPU submit cost, which is the honest fallback. The rAF
 * interval is vsync-locked and therefore useless as a cost measure on its own.
 */
class GpuTimer {
  private readonly gl: WebGL2RenderingContext;
  private ext: { TIME_ELAPSED_EXT: number; GPU_DISJOINT_EXT: number } | null = null;
  private pending: WebGLQuery[] = [];
  private active: WebGLQuery | null = null;
  ms = 0;
  supported = false;

  constructor(renderer: THREE.WebGLRenderer) {
    this.gl = renderer.getContext() as WebGL2RenderingContext;
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
      const q = this.pending[0];
      if (!gl.getQueryParameter(q, gl.QUERY_RESULT_AVAILABLE)) break;
      const ns = gl.getQueryParameter(q, gl.QUERY_RESULT) as number;
      this.ms = this.ms === 0 ? ns / 1e6 : this.ms * 0.85 + (ns / 1e6) * 0.15;
      gl.deleteQuery(q);
      this.pending.shift();
    }
  }
}

const gpuTimer = new GpuTimer(engine.renderer);

// ---------------------------------------------------------------------- ui

const ui = document.getElementById('ui') as HTMLDivElement;
const statsEl = document.getElementById('stats') as HTMLDivElement;
const checkboxes = new Map<keyof RetroOptions, HTMLInputElement>();
const resHeightButtons: HTMLButtonElement[] = [];
const presetButtons: HTMLButtonElement[] = [];

function section(title: string): void {
  const h = document.createElement('h4');
  h.textContent = title;
  ui.appendChild(h);
}

function toggle(key: keyof RetroOptions, label: string, hotkey?: string): void {
  const row = document.createElement('label');
  row.className = 'row';
  const box = document.createElement('input');
  box.type = 'checkbox';
  box.checked = pipeline.options[key] as boolean;
  const txt = document.createElement('span');
  txt.textContent = label;
  row.append(box, txt);
  if (hotkey) {
    const k = document.createElement('kbd');
    k.textContent = hotkey;
    row.appendChild(k);
  }
  box.addEventListener('change', () => {
    pipeline.setOption(key, box.checked as RetroOptions[typeof key]);
  });
  ui.appendChild(row);
  checkboxes.set(key, box);
}

function slider(
  label: string,
  min: number,
  max: number,
  step: number,
  get: () => number,
  set: (v: number) => void,
): void {
  const digits = step < 0.01 ? 4 : step < 1 ? 2 : 0;
  const wrap = document.createElement('div');
  wrap.className = 'slider';
  const head = document.createElement('div');
  const name = document.createElement('span');
  name.textContent = label;
  const val = document.createElement('span');
  val.textContent = get().toFixed(digits);
  head.append(name, val);
  const inp = document.createElement('input');
  inp.type = 'range';
  inp.min = String(min);
  inp.max = String(max);
  inp.step = String(step);
  inp.value = String(get());
  inp.addEventListener('input', () => {
    const v = parseFloat(inp.value);
    set(v);
    val.textContent = v.toFixed(digits);
  });
  wrap.append(head, inp);
  ui.appendChild(wrap);
}

function buttonRow(): HTMLDivElement {
  const box = document.createElement('div');
  box.className = 'btns';
  ui.appendChild(box);
  return box;
}

function markPreset(i: number): void {
  presetButtons.forEach((b, j) => b.classList.toggle('on', j === i));
}

function markHeight(h: number): void {
  for (const b of resHeightButtons) b.classList.toggle('on', b.textContent === String(h));
}

function syncCheckboxes(): void {
  for (const [k, box] of checkboxes) box.checked = pipeline.options[k] as boolean;
}

section('Passes');
toggle('enabled', 'pipeline', 'p');
toggle('bloom', 'bloom');
toggle('godRays', 'god rays');
toggle('fog', 'fog');
toggle('edgeDarken', 'edge darken');
toggle('grade', 'LUT grade');
toggle('quantise', 'quantise');
toggle('dither', 'dither');
toggle('vignette', 'vignette');
toggle('aberration', 'aberration');
toggle('scanlines', 'scanlines');

section('Internal resolution');
{
  const box = buttonRow();
  for (const h of [270, 360, 480, 540, 720]) {
    const b = document.createElement('button');
    b.textContent = String(h);
    b.addEventListener('click', () => {
      pipeline.setInternalHeight(h);
      markHeight(h);
    });
    resHeightButtons.push(b);
    box.appendChild(b);
  }
  markHeight(pipeline.options.internalHeight);
}

section('Time of day');
{
  const box = buttonRow();
  SKY_PRESETS.forEach((p, i) => {
    const b = document.createElement('button');
    b.textContent = p.label;
    b.addEventListener('click', () => {
      sky.setPreset(i);
      markPreset(i);
    });
    presetButtons.push(b);
    box.appendChild(b);
  });
}
slider('time of day', 0, 1, 0.005, () => sky.timeOfDayValue, (v) => {
  sky.setTimeOfDay(v);
  markPreset(-1);
});

section('Tuning');
slider('exposure', 0.4, 2, 0.01, () => pipeline.options.exposure, (v) =>
  pipeline.setOption('exposure', v),
);
slider('bloom', 0, 1.2, 0.01, () => pipeline.options.bloomStrength, (v) =>
  pipeline.setOption('bloomStrength', v),
);
slider('god rays', 0, 1.2, 0.01, () => pipeline.options.godRayStrength, (v) =>
  pipeline.setOption('godRayStrength', v),
);
slider('dither amount', 0, 1, 0.01, () => pipeline.options.ditherAmount, (v) =>
  pipeline.setOption('ditherAmount', v),
);
slider('fog density', 0, 0.03, 0.0002, () => pipeline.options.fogDensity, (v) =>
  pipeline.setOption('fogDensity', v),
);
slider('vignette', 0, 0.8, 0.01, () => pipeline.options.vignetteStrength, (v) =>
  pipeline.setOption('vignetteStrength', v),
);
slider('aberration', 0, 4, 0.05, () => pipeline.options.aberrationStrength, (v) =>
  pipeline.setOption('aberrationStrength', v),
);

section('Colour depth');
{
  const box = buttonRow();
  const bitButtons: HTMLButtonElement[] = [];
  for (const bits of [15, 16, 18, 21] as const) {
    const b = document.createElement('button');
    b.textContent = `${bits}b`;
    b.classList.toggle('on', bits === pipeline.options.quantiseBits);
    b.addEventListener('click', () => {
      pipeline.setOption('quantiseBits', bits);
      for (const o of bitButtons) o.classList.toggle('on', o === b);
    });
    bitButtons.push(b);
    box.appendChild(b);
  }
  const modeBox = buttonRow();
  const modeButtons: HTMLButtonElement[] = [];
  for (const mode of ['bayer', 'noise'] as const) {
    const b = document.createElement('button');
    b.textContent = mode;
    b.classList.toggle('on', pipeline.options.ditherMode === mode);
    b.addEventListener('click', () => {
      pipeline.setOption('ditherMode', mode);
      for (const o of modeButtons) o.classList.toggle('on', o === b);
    });
    modeButtons.push(b);
    modeBox.appendChild(b);
  }
}

section('Scene');
{
  const box = buttonRow();
  const pause = document.createElement('button');
  pause.textContent = 'pause cam';
  pause.addEventListener('click', () => {
    camPaused = !camPaused;
    pause.classList.toggle('on', camPaused);
  });
  const shadowBtn = document.createElement('button');
  shadowBtn.textContent = 'shadows';
  shadowBtn.classList.add('on');
  let shadowsOn = true;
  shadowBtn.addEventListener('click', () => {
    shadowsOn = !shadowsOn;
    lighting.setShadowsEnabled(shadowsOn);
    shadowBtn.classList.toggle('on', shadowsOn);
  });
  box.append(pause, shadowBtn);
}

// ------------------------------------------------------------------ hotkeys

window.addEventListener('keydown', (e) => {
  switch (e.key.toLowerCase()) {
    case ' ':
      camPaused = !camPaused;
      e.preventDefault();
      break;
    case 't': {
      const label = sky.cycleTimeOfDay();
      markPreset(SKY_PRESETS.findIndex((p) => p.label === label));
      break;
    }
    case 'p':
      pipeline.setOption('enabled', !pipeline.options.enabled);
      syncCheckboxes();
      break;
    case 'h':
      ui.classList.toggle('hidden');
      break;
    case '1':
      pipeline.setInternalHeight(360);
      markHeight(360);
      break;
    case '2':
      pipeline.setInternalHeight(480);
      markHeight(480);
      break;
    case '3':
      pipeline.setInternalHeight(540);
      markHeight(540);
      break;
    default:
      break;
  }
});

// --------------------------------------------------------------------- loop

sky.setPreset(0);
markPreset(0);

let last = performance.now();
let frameStamp = 0;
let frameAcc = 0;
let frameCount = 0;
let frameMs = 16.7;
let cpuMs = 0;

function updateStats(): void {
  const size = pipeline.internalSize;
  const dpr = engine.renderer.getPixelRatio();
  const gpu = gpuTimer.supported ? `${gpuTimer.ms.toFixed(2)} ms` : 'n/a';
  statsEl.innerHTML =
    `<b>gpu</b>    ${gpu}\n` +
    `<b>cpu</b>    ${cpuMs.toFixed(2)} ms\n` +
    `<b>frame</b>  ${frameMs.toFixed(2)} ms (${(1000 / Math.max(frameMs, 0.01)).toFixed(0)} fps)\n` +
    `<b>out</b>    ${Math.round(engine.width * dpr)}x${Math.round(engine.height * dpr)}\n` +
    `<b>inner</b>  ${size.width}x${size.height}\n` +
    `<b>sky</b>    ${sky.state.label}\n` +
    `<b>sun</b>    ${sky.sunDirection.x.toFixed(2)} ${sky.sunDirection.y.toFixed(2)} ${sky.sunDirection.z.toFixed(2)}\n` +
    `<b>fog</b>    #${sky.fogColor.getHexString()}\n` +
    `<b>tris</b>   ${engine.renderer.info.render.triangles.toLocaleString()}`;
}

function frame(): void {
  const now = performance.now();
  const dt = Math.min(0.05, (now - last) / 1000);
  last = now;

  updateCamera(dt);
  sky.update(dt, engine.camera.position);
  lighting.update(dt, focus, camDir);

  const t0 = performance.now();
  gpuTimer.begin();
  pipeline.render(dt);
  gpuTimer.end();
  cpuMs = cpuMs === 0 ? performance.now() - t0 : cpuMs * 0.9 + (performance.now() - t0) * 0.1;

  if (frameStamp) frameAcc += now - frameStamp;
  frameStamp = now;
  frameCount++;
  if (frameCount >= 20) {
    frameMs = frameAcc / frameCount;
    frameAcc = 0;
    frameCount = 0;
    updateStats();
  }
  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);

// Exposed so screenshots and measurements can be driven from the console.
declare global {
  interface Window {
    preview: {
      pipeline: RetroPipeline;
      sky: Sky;
      lighting: Lighting;
      engine: Engine;
      setPaused(v: boolean): void;
      setCamT(v: number): void;
      hideUi(v: boolean): void;
      stats(): { gpuMs: number; cpuMs: number; frameMs: number };
    };
  }
}

window.preview = {
  pipeline,
  sky,
  lighting,
  engine,
  setPaused: (v: boolean) => {
    camPaused = v;
  },
  setCamT: (v: number) => {
    camT = v;
  },
  hideUi: (v: boolean) => {
    ui.style.display = v ? 'none' : '';
    statsEl.style.display = v ? 'none' : '';
    const hint = document.getElementById('hint');
    if (hint) hint.style.display = v ? 'none' : '';
  },
  stats: () => ({ gpuMs: gpuTimer.ms, cpuMs, frameMs }),
};
