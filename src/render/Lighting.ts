import * as THREE from 'three';
import type { Sky } from '../world/Sky';

/**
 * Outdoor lighting rig: one shadow-casting sun plus a hemisphere fill, both
 * driven entirely by the Sky's time-of-day state so nothing can drift out of
 * agreement with the sky you can actually see.
 *
 * Deliberately NOT physically based. Lambert and Phong are the era's shading
 * models and the retro pipeline's grade is tuned for them; a PBR rig fights
 * the target on every axis — rolled-off highlights, energy-conserving
 * desaturation, and a general reluctance to be as punchy as 1997 was.
 *
 * The whole difficulty here is the shadow map. Two things kill it:
 *
 *  1. A shadow frustum sized for the world instead of the view. 2048 texels
 *     spread over a 2 km box is a metre per texel — mush. We fit the ortho box
 *     to a radius around the point the camera is actually looking at.
 *  2. Shadow crawl. Move that frustum continuously and every texel boundary
 *     slides under the geometry, so shadow edges shimmer and creep as you
 *     drive. The fix is to quantise the frustum centre to whole shadow-map
 *     texels *in light space*, which makes the texel grid world-stationary:
 *     the map jumps by exact texels and the sampled pattern never changes.
 *
 * For (2) to work the light-space basis used for snapping must be byte-for-byte
 * the basis three itself uses when it builds the shadow matrix. three's
 * LightShadow does `shadowCamera.lookAt(target)` with `shadowCamera.up`, so we
 * snap against `lookAt(sunDir, origin, shadowCamera.up)` and write that same up
 * vector onto the shadow camera. Any mismatch and the crawl comes back.
 */

export interface LightingOptions {
  /** Shadow map resolution. 2048 over a 55 m radius is ~5 cm per texel. */
  mapSize?: number;
  /** Half-extent of the shadow ortho box, in metres. */
  shadowRadius?: number;
  /**
   * How far along the view direction to push the shadow box. Nothing behind
   * the camera needs shadows, so centring on the focus point alone wastes
   * roughly half the map.
   */
  lookAhead?: number;
  shadows?: boolean;
}

const _m = new THREE.Matrix4();
const _mInv = new THREE.Matrix4();
const _center = new THREE.Vector3();
const _lightDir = new THREE.Vector3();
const _ahead = new THREE.Vector3();
const ZERO = new THREE.Vector3(0, 0, 0);

export class Lighting {
  readonly sun: THREE.DirectionalLight;
  readonly hemi: THREE.HemisphereLight;
  /** Tiny flat term so nothing in deep shadow reads as pure black. */
  readonly ambient: THREE.AmbientLight;

  private readonly scene: THREE.Scene;
  private readonly sky: Sky;
  private readonly mapSize: number;
  private shadowRadius: number;
  private readonly lookAhead: number;
  private readonly viewDir = new THREE.Vector3(0, 0, -1);
  private disposed = false;

  constructor(scene: THREE.Scene, sky: Sky, opts: LightingOptions = {}) {
    this.scene = scene;
    this.sky = sky;
    this.mapSize = opts.mapSize ?? 2048;
    this.shadowRadius = opts.shadowRadius ?? 55;
    this.lookAhead = opts.lookAhead ?? 16;

    this.sun = new THREE.DirectionalLight(0xffffff, 2);
    this.sun.castShadow = opts.shadows !== false;
    this.sun.shadow.mapSize.set(this.mapSize, this.mapSize);
    // Fixed, axis-aligned up vector: the snapping grid must not rotate.
    this.sun.shadow.camera.up.set(0, 1, 0);

    // Slope-scaled acne control. `normalBias` pushes the receiver along its own
    // normal, which fixes acne on the shallow-lit slopes that dominate an
    // outdoor scene without the peter-panning a large constant bias causes.
    this.sun.shadow.bias = -0.0004;
    this.sun.shadow.normalBias = 0.04;

    this.applyShadowExtents();

    scene.add(this.sun);
    scene.add(this.sun.target);

    // Sky-blue from above, warm bounce from the ground. This one pair of
    // colours is worth more to an outdoor scene than any amount of extra light
    // count: it is what stops shadowed faces going flat grey.
    this.hemi = new THREE.HemisphereLight(0x6ba3e0, 0x6b563a, 0.55);
    scene.add(this.hemi);

    this.ambient = new THREE.AmbientLight(0xffffff, 0.06);
    scene.add(this.ambient);

    this.syncFromSky();
    this.fitShadow(ZERO);
  }

  /** Metres of half-extent covered by the shadow frustum. */
  setShadowRadius(r: number): void {
    this.shadowRadius = Math.max(4, r);
    this.applyShadowExtents();
  }

  setShadowsEnabled(on: boolean): void {
    this.sun.castShadow = on;
  }

  /**
   * @param focusPoint  what the shadows must be sharp around — the car.
   * @param viewDir     optional camera forward, used to bias the box ahead.
   */
  update(_dt: number, focusPoint: THREE.Vector3, viewDir?: THREE.Vector3): void {
    if (this.disposed) return;
    if (viewDir) {
      this.viewDir.copy(viewDir);
      this.viewDir.y = 0;
      if (this.viewDir.lengthSq() < 1e-6) this.viewDir.set(0, 0, -1);
      this.viewDir.normalize();
    }
    this.syncFromSky();
    this.fitShadow(focusPoint);
  }

  private applyShadowExtents(): void {
    const cam = this.sun.shadow.camera;
    cam.left = -this.shadowRadius;
    cam.right = this.shadowRadius;
    cam.top = this.shadowRadius;
    cam.bottom = -this.shadowRadius;
    cam.near = 1;
    // Must cover the light's standoff plus whatever sits above and below the
    // focus point, and no more: every extra metre costs depth precision.
    cam.far = this.shadowRadius * 6 + 220;
    cam.updateProjectionMatrix();
  }

  private syncFromSky(): void {
    const s = this.sky.state;
    this.sun.color.copy(s.sunColor);
    this.sun.intensity = s.sunIntensity;
    this.hemi.color.copy(s.ambientSkyColor);
    this.hemi.groundColor.copy(s.ambientGroundColor);
    this.hemi.intensity = s.ambientIntensity;
    // A hair of flat fill keyed to the ambient, so the darkest shadow still
    // carries the sky's colour rather than crushing to the grade's black.
    this.ambient.color.copy(s.ambientSkyColor);
    this.ambient.intensity = 0.05 + 0.05 * s.ambientIntensity;
  }

  /**
   * Fit the ortho shadow box around the focus point, then snap it to the
   * shadow map's own texel grid so the shadows stop crawling.
   */
  private fitShadow(focusPoint: THREE.Vector3): void {
    // A sun at the horizon gives a grazing frustum that covers nothing useful,
    // so clamp the elevation used for *placement* only. The shading direction
    // the scene sees is still the sky's real sun direction.
    _lightDir.copy(this.sky.sunDirection);
    if (_lightDir.lengthSq() < 1e-8) _lightDir.set(0, 1, 0);
    if (_lightDir.y < 0.18) {
      _lightDir.y = 0.18;
      _lightDir.normalize();
    }

    _ahead.copy(this.viewDir).multiplyScalar(this.lookAhead);
    _center.copy(focusPoint).add(_ahead);
    _center.y += 1.5;

    // The light-space basis three will use for the shadow matrix.
    _m.lookAt(_lightDir, ZERO, this.sun.shadow.camera.up);
    _mInv.copy(_m).invert();

    const texel = (this.shadowRadius * 2) / this.mapSize;
    _center.applyMatrix4(_mInv);
    _center.x = Math.round(_center.x / texel) * texel;
    _center.y = Math.round(_center.y / texel) * texel;
    // Depth is not quantised by the map's XY grid, but snapping it too keeps
    // the whole frustum on one lattice and costs nothing.
    _center.z = Math.round(_center.z / texel) * texel;
    _center.applyMatrix4(_m);

    const standoff = this.shadowRadius * 3 + 60;
    this.sun.position.copy(_center).addScaledVector(_lightDir, standoff);
    this.sun.target.position.copy(_center);
    this.sun.target.updateMatrixWorld();
    this.sun.updateMatrixWorld();
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.scene.remove(this.sun);
    this.scene.remove(this.sun.target);
    this.scene.remove(this.hemi);
    this.scene.remove(this.ambient);
    this.sun.shadow.dispose();
    this.sun.dispose();
    this.hemi.dispose();
    this.ambient.dispose();
  }
}
