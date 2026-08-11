/**
 * Scatter preview.
 *
 * A flyable camera over the real Terrain with the real Scatter on top, plus
 * the numbers that decide whether any of this is affordable: instance counts
 * per layer and LOD, draw calls, triangles, and where the frame time goes.
 *
 * The camera modes matter as much as the toggles. Driver's eye at 1.35 m is
 * the only view that tells you whether the forest has depth or is a flat wall;
 * overhead is the only one that tells you whether the placement is an
 * ecosystem or confetti.
 */

import * as THREE from 'three';
import { createPhysics } from '../physics/PhysicsWorld';
import { Terrain } from './Terrain';
import { Sky } from './Sky';
import { Lighting } from '../render/Lighting';
import { Scatter } from './Scatter';

const boot = document.getElementById('boot') as HTMLDivElement;
const ui = document.getElementById('ui') as HTMLDivElement;
const statsEl = document.getElementById('stats') as HTMLDivElement;

const LAYERS: { id: string; label: string; key: string }[] = [
  { id: 'conifer', label: 'Conifers', key: 't' },
  { id: 'broadleaf', label: 'Broadleaf', key: 'y' },
  { id: 'bush', label: 'Bushes', key: 'u' },
  { id: 'grass', label: 'Grass', key: 'g' },
  { id: 'boulder', label: 'Boulders', key: 'i' },
  { id: 'stone', label: 'Stones', key: 'o' },
  { id: 'scree', label: 'Scree', key: 'p' },
  { id: 'deadwood', label: 'Deadwood', key: 'k' },
  { id: 'prop', label: 'Trail furniture', key: 'j' },
];

async function main(): Promise<void> {
  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.NoToneMapping;
  document.body.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(62, window.innerWidth / window.innerHeight, 0.4, 4000);

  boot.textContent = 'GENERATING TERRAIN…';
  await new Promise((r) => setTimeout(r, 16));

  const physics = await createPhysics();
  const terrain = new Terrain(physics, { seed: 20260807 });
  scene.add(terrain.object3d);

  const sky = new Sky(scene, undefined, { timeOfDay: 0.62, weather: 'clear' });
  const lighting = new Lighting(scene, sky, { shadowRadius: 65, mapSize: 2048 });
  scene.fog = new THREE.FogExp2(sky.fogColor.getHex(), 0.0016);

  boot.textContent = 'SCATTERING…';
  await new Promise((r) => setTimeout(r, 16));

  const t0 = performance.now();
  const scatter = new Scatter(terrain, physics, { renderer });
  const scatterMs = performance.now() - t0;
  scene.add(scatter.object3d);

  boot.remove();

  /* ---------------------------------------------------------------- camera */

  const spawn = terrain.getSpawnPoint();
  const state = {
    mode: 'drive' as 'drive' | 'top' | 'orbit',
    yaw: spawn.heading,
    pitch: -0.03,
    pos: spawn.position.clone(),
    orbitAngle: 0,
    lodLock: -1,
    windPaused: false,
    frozen: false,
  };
  const keys = new Set<string>();

  function applyDriveHeight(): void {
    state.pos.y = terrain.heightAt(state.pos.x, state.pos.z) + 1.35;
  }
  applyDriveHeight();

  renderer.domElement.addEventListener('pointerdown', (e) => {
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
  });
  renderer.domElement.addEventListener('pointermove', (e) => {
    if (e.buttons & 1) {
      state.yaw -= e.movementX * 0.0032;
      state.pitch = Math.max(-1.5, Math.min(1.5, state.pitch - e.movementY * 0.0032));
    }
  });
  window.addEventListener('keydown', (e) => {
    keys.add(e.key.toLowerCase());
    if (e.repeat) return;
    const k = e.key.toLowerCase();
    if (k === 'h') ui.classList.toggle('hidden');
    else if (k === '1') { state.mode = 'drive'; state.pos.copy(spawn.position); state.yaw = spawn.heading; state.pitch = -0.03; applyDriveHeight(); }
    else if (k === '2') { state.mode = 'top'; }
    else if (k === '3') { state.mode = 'orbit'; }
    else if (k === 'l') { state.lodLock = state.lodLock >= 2 ? -1 : state.lodLock + 1; scatter.forceLod(state.lodLock); syncUi(); }
    else if (k === 'm') { state.windPaused = !state.windPaused; }
    else if (k === 'f') { scene.fog = scene.fog ? null : new THREE.FogExp2(sky.fogColor.getHex(), 0.0016); scene.traverse(() => {}); refreshFogFlag(); }
    else {
      const layer = LAYERS.find((l) => l.key === k);
      if (layer) {
        const cb = document.getElementById(`cb-${layer.id}`) as HTMLInputElement;
        cb.checked = !cb.checked;
        scatter.setLayerEnabled(layer.id, cb.checked);
      }
    }
  });
  window.addEventListener('keyup', (e) => keys.delete(e.key.toLowerCase()));
  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  function refreshFogFlag(): void {
    scene.traverse((o) => {
      const m = (o as THREE.Mesh).material;
      if (!m) return;
      for (const mm of Array.isArray(m) ? m : [m]) mm.needsUpdate = true;
    });
  }

  /* -------------------------------------------------------------------- ui */

  function section(label: string): void {
    const h = document.createElement('h4');
    h.textContent = label;
    ui.appendChild(h);
  }
  section('Layers');
  for (const l of LAYERS) {
    const row = document.createElement('label');
    row.className = 'row';
    row.innerHTML = `<input type="checkbox" id="cb-${l.id}" checked><span>${l.label}</span><kbd>${l.key}</kbd>`;
    ui.appendChild(row);
    (row.querySelector('input') as HTMLInputElement).addEventListener('change', (e) => {
      scatter.setLayerEnabled(l.id, (e.target as HTMLInputElement).checked);
    });
  }
  section('View');
  const viewBtns = document.createElement('div');
  viewBtns.className = 'btns';
  viewBtns.innerHTML =
    `<button data-mode="drive">driver</button><button data-mode="top">top</button><button data-mode="orbit">orbit</button>`;
  ui.appendChild(viewBtns);
  viewBtns.addEventListener('click', (e) => {
    const b = e.target as HTMLButtonElement;
    if (!b.dataset.mode) return;
    state.mode = b.dataset.mode as typeof state.mode;
    if (state.mode === 'drive') applyDriveHeight();
    syncUi();
  });
  section('LOD lock');
  const lodBtns = document.createElement('div');
  lodBtns.className = 'btns';
  lodBtns.innerHTML =
    `<button data-lod="-1">auto</button><button data-lod="0">near</button><button data-lod="1">mid</button><button data-lod="2">impostor</button>`;
  ui.appendChild(lodBtns);
  lodBtns.addEventListener('click', (e) => {
    const b = e.target as HTMLButtonElement;
    if (b.dataset.lod === undefined) return;
    state.lodLock = Number(b.dataset.lod);
    scatter.forceLod(state.lodLock);
    syncUi();
  });
  section('Shading');
  const shadeBtns = document.createElement('div');
  shadeBtns.className = 'btns';
  shadeBtns.innerHTML = `<button id="b-shadow" class="on">shadows</button><button id="b-fog" class="on">fog</button><button id="b-wind" class="on">wind</button>`;
  ui.appendChild(shadeBtns);
  shadeBtns.addEventListener('click', (e) => {
    const b = e.target as HTMLButtonElement;
    if (b.id === 'b-shadow') {
      renderer.shadowMap.enabled = !renderer.shadowMap.enabled;
      refreshFogFlag();
    } else if (b.id === 'b-fog') {
      scene.fog = scene.fog ? null : new THREE.FogExp2(sky.fogColor.getHex(), 0.0016);
      refreshFogFlag();
    } else if (b.id === 'b-wind') {
      state.windPaused = !state.windPaused;
    }
    syncUi();
  });
  (document.getElementById('uitoggle') as HTMLElement).addEventListener('click', () => ui.classList.toggle('hidden'));

  function syncUi(): void {
    for (const b of Array.from(viewBtns.querySelectorAll('button'))) {
      b.classList.toggle('on', b.dataset.mode === state.mode);
    }
    for (const b of Array.from(lodBtns.querySelectorAll('button'))) {
      b.classList.toggle('on', Number(b.dataset.lod) === state.lodLock);
    }
    (document.getElementById('b-shadow') as HTMLElement).classList.toggle('on', renderer.shadowMap.enabled);
    (document.getElementById('b-fog') as HTMLElement).classList.toggle('on', scene.fog !== null);
    (document.getElementById('b-wind') as HTMLElement).classList.toggle('on', !state.windPaused);
  }
  syncUi();

  /* ----------------------------------------------------------------- frame */

  let last = performance.now();
  let frameMs = 16;
  let cpuMs = 0;
  let windClock = 0;
  const fwd = new THREE.Vector3();
  const right = new THREE.Vector3();
  const focus = new THREE.Vector3();

  function frame(now: number): void {
    requestAnimationFrame(frame);
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    frameMs += (dt * 1000 - frameMs) * 0.08;
    if (!state.windPaused) windClock += dt;

    const speed = (keys.has('shift') ? 90 : 22) * dt;
    fwd.set(-Math.sin(state.yaw) * Math.cos(state.pitch), Math.sin(state.pitch), -Math.cos(state.yaw) * Math.cos(state.pitch));
    right.set(Math.cos(state.yaw), 0, -Math.sin(state.yaw));

    if (state.mode === 'drive' || state.mode === 'top') {
      if (keys.has('w')) state.pos.addScaledVector(fwd, speed);
      if (keys.has('s')) state.pos.addScaledVector(fwd, -speed);
      if (keys.has('a')) state.pos.addScaledVector(right, -speed);
      if (keys.has('d')) state.pos.addScaledVector(right, speed);
      if (keys.has('e')) state.pos.y += speed;
      if (keys.has('q')) state.pos.y -= speed;
    }

    const half = terrain.halfSize - 5;
    state.pos.x = Math.max(-half, Math.min(half, state.pos.x));
    state.pos.z = Math.max(-half, Math.min(half, state.pos.z));

    if (state.mode === 'drive') {
      const ground = terrain.heightAt(state.pos.x, state.pos.z) + 1.35;
      state.pos.y += (ground - state.pos.y) * Math.min(1, dt * 12);
      camera.position.copy(state.pos);
      camera.rotation.set(0, 0, 0);
      camera.rotateY(state.yaw);
      camera.rotateX(state.pitch);
    } else if (state.mode === 'top') {
      camera.position.set(state.pos.x, terrain.heightAt(state.pos.x, state.pos.z) + 320, state.pos.z + 30);
      camera.lookAt(state.pos.x, terrain.heightAt(state.pos.x, state.pos.z), state.pos.z);
    } else {
      state.orbitAngle += dt * 0.08;
      const r = 140;
      focus.set(state.pos.x, terrain.heightAt(state.pos.x, state.pos.z) + 10, state.pos.z);
      camera.position.set(focus.x + Math.cos(state.orbitAngle) * r, focus.y + 55, focus.z + Math.sin(state.orbitAngle) * r);
      camera.lookAt(focus);
    }
    camera.updateMatrixWorld();

    const c0 = performance.now();
    sky.update(dt, camera.position);
    lighting.update(dt, camera.position);
    terrain.update(camera.position);
    terrain.setFog(sky.fogColor, 0.0016);
    if (scene.fog) (scene.fog as THREE.FogExp2).color.copy(sky.fogColor);
    scatter.update(camera.position, camera);
    cpuMs += (performance.now() - c0 - cpuMs) * 0.1;

    renderer.info.reset();
    renderer.render(scene, camera);

    const s = scatter.getStats();
    const info = renderer.info.render;
    statsEl.innerHTML =
      `<b>frame</b>  ${frameMs.toFixed(2)} ms  (${(1000 / frameMs).toFixed(0)} fps)\n` +
      `<b>cpu</b>    ${cpuMs.toFixed(2)} ms   rebuild ${s.lastRebuildMs.toFixed(2)} ms\n` +
      `<b>calls</b>  ${info.calls}   tris ${(info.triangles / 1000).toFixed(0)}k\n` +
      `<b>drawn</b>  ${s.drawnInstances.toLocaleString()} / ${s.totalInstances.toLocaleString()}\n` +
      `<b>meshes</b> ${s.drawnMeshes}   grass ${s.grass}   cols ${s.colliders}\n` +
      `<b>build</b>  ${scatterMs.toFixed(0)} ms (impostors ${s.impostorMs.toFixed(0)} ms)\n` +
      `<b>lod</b>    ${state.lodLock < 0 ? 'auto' : state.lodLock}   cam ${state.pos.x.toFixed(0)},${state.pos.z.toFixed(0)}\n` +
      `\n` +
      s.perLayer
        .map((l) => `${l.id.padEnd(10)} ${String(l.drawn).padStart(6)} /${String(l.total).padStart(6)}  ${l.perLod.join('/')}`)
        .join('\n');
  }
  requestAnimationFrame(frame);

  // Handy for automated screenshots.
  (window as unknown as Record<string, unknown>).preview = { scene, camera, renderer, scatter, terrain, state, ui };
}

void main();
