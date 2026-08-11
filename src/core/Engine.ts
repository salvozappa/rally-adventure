import * as THREE from 'three';

/**
 * Owns the canvas, renderer, scene and camera. Deliberately thin — the retro
 * render pipeline plugs in on top of this rather than replacing it.
 */
export class Engine {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  readonly canvas: HTMLCanvasElement;

  /** Logical (CSS) size in pixels. */
  width = 1;
  height = 1;

  private listeners: Array<(w: number, h: number) => void> = [];

  constructor() {
    this.canvas = document.createElement('canvas');
    document.body.appendChild(this.canvas);

    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: false, // the retro pipeline resolves its own edges
      powerPreference: 'high-performance',
      stencil: false,
    });
    this.renderer.setClearColor(0x000000, 1);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.NoToneMapping;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;

    this.camera = new THREE.PerspectiveCamera(65, 1, 0.15, 4000);
    this.scene.add(this.camera);

    window.addEventListener('resize', this.resize);
    this.resize();
  }

  onResize(fn: (w: number, h: number) => void): void {
    this.listeners.push(fn);
    fn(this.width, this.height);
  }

  private resize = (): void => {
    this.width = window.innerWidth;
    this.height = window.innerHeight;
    this.camera.aspect = this.width / this.height;
    this.camera.updateProjectionMatrix();
    // Cap DPR: the internal render target is low-res by design, so paying for
    // retina here buys nothing but fill rate.
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(this.width, this.height, false);
    for (const fn of this.listeners) fn(this.width, this.height);
  };
}
