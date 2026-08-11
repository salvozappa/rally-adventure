import * as THREE from 'three';

/**
 * Hand-rolled full-screen pass. A single oversized triangle beats a quad:
 * no diagonal seam, one fewer vertex, and the GPU rasterises it in one go.
 */

const TRI_POSITIONS = new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]);
const TRI_UVS = new Float32Array([0, 0, 2, 0, 0, 2]);

let sharedGeometry: THREE.BufferGeometry | null = null;

function getGeometry(): THREE.BufferGeometry {
  if (!sharedGeometry) {
    sharedGeometry = new THREE.BufferGeometry();
    sharedGeometry.setAttribute('position', new THREE.BufferAttribute(TRI_POSITIONS, 3));
    sharedGeometry.setAttribute('uv', new THREE.BufferAttribute(TRI_UVS, 2));
  }
  return sharedGeometry;
}

export const FULLSCREEN_VERT = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

const passCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

export interface PassInit {
  fragmentShader: string;
  uniforms: Record<string, THREE.IUniform>;
  defines?: Record<string, string | number>;
  vertexShader?: string;
}

export class FullScreenPass {
  readonly material: THREE.ShaderMaterial;
  readonly uniforms: Record<string, THREE.IUniform>;
  private readonly scene = new THREE.Scene();
  private readonly mesh: THREE.Mesh;

  constructor(init: PassInit) {
    this.uniforms = init.uniforms;
    this.material = new THREE.ShaderMaterial({
      uniforms: init.uniforms,
      vertexShader: init.vertexShader ?? FULLSCREEN_VERT,
      fragmentShader: init.fragmentShader,
      defines: init.defines ?? {},
      depthTest: false,
      depthWrite: false,
      // Raw ShaderMaterials get no <colorspace_fragment> injection, so whatever
      // we write is exactly what lands in the target. That is essential: the
      // quantise step must not be silently re-encoded behind our back.
    });
    this.mesh = new THREE.Mesh(getGeometry(), this.material);
    this.mesh.frustumCulled = false;
    this.scene.add(this.mesh);
  }

  /** Recompile with a changed define set (used for toggling features cheaply). */
  setDefine(key: string, value: string | number | null): void {
    const defines = this.material.defines as Record<string, string | number>;
    const current = defines[key];
    if (value === null) {
      if (current === undefined) return;
      delete defines[key];
    } else {
      if (current === value) return;
      defines[key] = value;
    }
    this.material.needsUpdate = true;
  }

  render(renderer: THREE.WebGLRenderer, target: THREE.WebGLRenderTarget | null): void {
    const prevTarget = renderer.getRenderTarget();
    renderer.setRenderTarget(target);
    renderer.render(this.scene, passCamera);
    renderer.setRenderTarget(prevTarget);
  }

  dispose(): void {
    this.material.dispose();
    this.scene.clear();
  }
}

export interface RTOptions {
  half?: boolean;
  depth?: boolean;
  nearest?: boolean;
}

export function makeRenderTarget(w: number, h: number, o: RTOptions = {}): THREE.WebGLRenderTarget {
  const filter = o.nearest ? THREE.NearestFilter : THREE.LinearFilter;
  const rt = new THREE.WebGLRenderTarget(Math.max(1, w), Math.max(1, h), {
    minFilter: filter,
    magFilter: filter,
    format: THREE.RGBAFormat,
    type: o.half ? THREE.HalfFloatType : THREE.UnsignedByteType,
    depthBuffer: o.depth === true,
    stencilBuffer: false,
    generateMipmaps: false,
    // NoColorSpace keeps three from transforming values on the way in or out.
    colorSpace: THREE.NoColorSpace,
  });
  rt.texture.wrapS = THREE.ClampToEdgeWrapping;
  rt.texture.wrapT = THREE.ClampToEdgeWrapping;
  if (o.depth) {
    const dt = new THREE.DepthTexture(Math.max(1, w), Math.max(1, h));
    dt.type = THREE.UnsignedIntType;
    dt.format = THREE.DepthFormat;
    dt.minFilter = THREE.NearestFilter;
    dt.magFilter = THREE.NearestFilter;
    rt.depthTexture = dt;
  }
  return rt;
}
