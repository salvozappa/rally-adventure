import * as THREE from 'three';

/**
 * Ordered dither threshold matrix as an 8x8 R8 texture.
 *
 * Why a texture and not a const array in the shader: GLSL ES 1.00 (which is
 * what three emits unless you opt into GLSL3) forbids dynamic indexing of
 * local arrays. A repeating NEAREST texture indexed by gl_FragCoord is both
 * portable and free.
 */

function bayerMatrix(n: number): number[] {
  if (n === 1) return [0];
  const half = n >> 1;
  const inner = bayerMatrix(half);
  const out = new Array<number>(n * n);
  for (let y = 0; y < half; y++) {
    for (let x = 0; x < half; x++) {
      const v = inner[y * half + x] * 4;
      out[y * n + x] = v;
      out[y * n + (x + half)] = v + 2;
      out[(y + half) * n + x] = v + 3;
      out[(y + half) * n + (x + half)] = v + 1;
    }
  }
  return out;
}

export function createBayerTexture(size: 2 | 4 | 8 = 8): THREE.DataTexture {
  const m = bayerMatrix(size);
  const n = size * size;
  const data = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    // Map to the centre of each bucket: (v + 0.5) / n. Using the raw index
    // would bias the whole image down by half a quantisation step.
    data[i] = Math.round(((m[i] + 0.5) / n) * 255);
  }
  const tex = new THREE.DataTexture(data, size, size, THREE.RedFormat, THREE.UnsignedByteType);
  tex.magFilter = THREE.NearestFilter;
  tex.minFilter = THREE.NearestFilter;
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.generateMipmaps = false;
  tex.colorSpace = THREE.NoColorSpace;
  tex.needsUpdate = true;
  return tex;
}

/** Blue-ish noise fallback for the "noise" dither mode — 64x64, tileable. */
export function createNoiseTexture(size = 64): THREE.DataTexture {
  const data = new Uint8Array(size * size);
  // Void-and-cluster is overkill here; a hashed white noise with a mild
  // high-pass looks close enough at 480p and costs nothing to generate.
  const raw = new Float32Array(size * size);
  let seed = 1337;
  const rnd = (): number => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 4294967296;
  };
  for (let i = 0; i < raw.length; i++) raw[i] = rnd();
  const hp = new Float32Array(size * size);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      let sum = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          sum += raw[((y + dy + size) % size) * size + ((x + dx + size) % size)];
        }
      }
      hp[y * size + x] = raw[y * size + x] - sum / 9 + 0.5;
    }
  }
  for (let i = 0; i < data.length; i++) {
    data[i] = Math.max(0, Math.min(255, Math.round(hp[i] * 255)));
  }
  const tex = new THREE.DataTexture(data, size, size, THREE.RedFormat, THREE.UnsignedByteType);
  tex.magFilter = THREE.NearestFilter;
  tex.minFilter = THREE.NearestFilter;
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.generateMipmaps = false;
  tex.colorSpace = THREE.NoColorSpace;
  tex.needsUpdate = true;
  return tex;
}
