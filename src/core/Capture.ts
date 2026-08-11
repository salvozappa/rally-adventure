import * as THREE from 'three';

export interface Vantage {
  name: string;
  /** Where to place the car, in world space. */
  carPosition: THREE.Vector3;
  /** Car heading, radians about +Y. */
  carHeading: number;
  /** Camera offset from the car, in the car's local frame. */
  cameraOffset: THREE.Vector3;
  /** Point the camera aims at, offset from the car in world space. */
  lookOffset: THREE.Vector3;
  fov: number;
  /** Optional time-of-day override, 0..1. */
  timeOfDay?: number;
}

/**
 * Scripted camera vantages for quality screenshots.
 *
 * The point is reproducibility: every iteration of the art must be judged from
 * exactly the same framing, or a comparison tells you nothing about whether the
 * work improved. Activated with `?capture=1` on the game URL.
 */
export class Capture {
  private index = 0;

  constructor(private vantages: Vantage[]) {}

  static get enabled(): boolean {
    return new URLSearchParams(location.search).has('capture');
  }

  /** Vantage requested by `?shot=<name|index>`, if any. */
  static get requested(): string | null {
    return new URLSearchParams(location.search).get('shot');
  }

  get current(): Vantage {
    return this.vantages[this.index];
  }

  select(nameOrIndex: string): Vantage {
    const asNum = Number(nameOrIndex);
    this.index = Number.isFinite(asNum)
      ? THREE.MathUtils.euclideanModulo(asNum, this.vantages.length)
      : Math.max(0, this.vantages.findIndex((v) => v.name === nameOrIndex));
    return this.current;
  }

  next(): Vantage {
    this.index = (this.index + 1) % this.vantages.length;
    return this.current;
  }

  /** Place the camera for the active vantage, given the car's actual transform. */
  apply(camera: THREE.PerspectiveCamera, carPos: THREE.Vector3, carQuat: THREE.Quaternion): void {
    const v = this.current;
    const offset = v.cameraOffset.clone().applyQuaternion(carQuat);
    camera.position.copy(carPos).add(offset);
    camera.lookAt(carPos.x + v.lookOffset.x, carPos.y + v.lookOffset.y, carPos.z + v.lookOffset.z);
    if (camera.fov !== v.fov) {
      camera.fov = v.fov;
      camera.updateProjectionMatrix();
    }
  }

  /**
   * Signals to the harness that the frame is settled and worth capturing. The
   * screenshot tool polls for this attribute rather than guessing at a delay.
   */
  static markReady(name: string): void {
    document.body.setAttribute('data-capture-ready', name);
  }
}

/**
 * The standard shot list. Offsets are in the chassis frame, where forward is
 * +Z (VehicleTuning's header), so a trailing camera sits at negative Z and aims
 * at positive Z.
 *
 * Chosen to mirror the framings that recur in the
 * reference screenshots: a low chase view with the horizon high in frame, a
 * hero three-quarter, a wide landscape establishing shot, and a close detail
 * pass on the car itself.
 */
export function defaultVantages(spawn: THREE.Vector3): Vantage[] {
  return [
    {
      name: 'chase',
      carPosition: spawn.clone(),
      carHeading: 0,
      cameraOffset: new THREE.Vector3(0, 2.4, -6.6),
      lookOffset: new THREE.Vector3(0, 1.0, 6),
      fov: 64,
    },
    {
      name: 'hero34',
      carPosition: spawn.clone(),
      carHeading: 0,
      cameraOffset: new THREE.Vector3(-4.4, 1.6, -5.2),
      lookOffset: new THREE.Vector3(0, 0.9, 0),
      fov: 48,
    },
    {
      name: 'landscape',
      carPosition: spawn.clone(),
      carHeading: 0,
      cameraOffset: new THREE.Vector3(14, 7.5, -18),
      lookOffset: new THREE.Vector3(0, 2, 30),
      fov: 58,
    },
    {
      name: 'lowdetail',
      carPosition: spawn.clone(),
      carHeading: 0,
      cameraOffset: new THREE.Vector3(-2.6, 0.55, -3.4),
      lookOffset: new THREE.Vector3(0, 0.75, 0),
      fov: 40,
    },
    {
      name: 'goldenhour',
      carPosition: spawn.clone(),
      carHeading: 0,
      cameraOffset: new THREE.Vector3(0, 2.1, -7.2),
      lookOffset: new THREE.Vector3(0, 1.1, 8),
      fov: 62,
      timeOfDay: 0.84,
    },
  ];
}
