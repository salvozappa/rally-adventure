/**
 * Headless texture contact sheet.
 *
 * Generates every procedural texture, tiles each 2x2 so seams are obvious, and
 * writes a labelled PNG grid. Runs in Node with no browser and no GPU, which
 * makes it fast enough to sit in the iterate-and-look loop.
 *
 * Build+run:  npx esbuild tools/dump-textures.mts --bundle --platform=node \
 *               --format=esm --outfile=/tmp/dt.mjs && node /tmp/dt.mjs
 */
import { PATTERNS, DEFAULT_SIZE, DEFAULT_SEED, TEXTURE_NAMES } from '../src/render/texturePatterns';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';

const OUT = 'reference/progress';
fs.mkdirSync(OUT, { recursive: true });

/** Minimal PNG encoder — avoids pulling an image dependency into the project. */
function encodePng(rgba: Uint8ClampedArray, w: number, h: number): Buffer {
  const raw = Buffer.alloc((w * 4 + 1) * h);
  for (let y = 0; y < h; y++) {
    raw[y * (w * 4 + 1)] = 0; // filter: none
    Buffer.from(rgba.buffer, rgba.byteOffset + y * w * 4, w * 4).copy(raw, y * (w * 4 + 1) + 1);
  }
  const idat = zlib.deflateSync(raw, { level: 9 });

  const chunk = (type: string, data: Buffer) => {
    const len = Buffer.alloc(4);
    len.writeUInt32BE(data.length);
    const td = Buffer.concat([Buffer.from(type, 'ascii'), data]);
    const crc = Buffer.alloc(4);
    crc.writeUInt32BE(crc32(td) >>> 0);
    return Buffer.concat([len, td, crc]);
  };

  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0);
  ihdr.writeUInt32BE(h, 4);
  ihdr[8] = 8;   // bit depth
  ihdr[9] = 6;   // colour type RGBA
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr),
    chunk('IDAT', idat),
    chunk('IEND', Buffer.alloc(0)),
  ]);
}

let CRC_TABLE: number[] | null = null;
function crc32(buf: Buffer): number {
  if (!CRC_TABLE) {
    CRC_TABLE = [];
    for (let n = 0; n < 256; n++) {
      let c = n;
      for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
      CRC_TABLE[n] = c;
    }
  }
  let c = 0xffffffff;
  for (let i = 0; i < buf.length; i++) c = CRC_TABLE[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return c ^ 0xffffffff;
}

/** Tile a square RGBA image n x n times. */
function tile(src: Uint8ClampedArray, size: number, n: number): { data: Uint8ClampedArray; size: number } {
  const outSize = size * n;
  const out = new Uint8ClampedArray(outSize * outSize * 4);
  for (let y = 0; y < outSize; y++) {
    const sy = y % size;
    for (let x = 0; x < outSize; x++) {
      const sx = x % size;
      const si = (sy * size + sx) * 4;
      const di = (y * outSize + x) * 4;
      out[di] = src[si];
      out[di + 1] = src[si + 1];
      out[di + 2] = src[si + 2];
      out[di + 3] = src[si + 3];
    }
  }
  return { data: out, size: outSize };
}

const timings: Array<[string, number, number]> = [];
const files: string[] = [];

for (const name of TEXTURE_NAMES) {
  const size = DEFAULT_SIZE[name] ?? 128;
  const seed = DEFAULT_SEED[name] ?? 1;
  const t0 = performance.now();
  const r = PATTERNS[name]({ size, seed });
  const ms = performance.now() - t0;
  timings.push([name, ms, r.size]);

  // Sprites must not be tiled — they're meant to be seen once, on their own.
  const n = r.clamp ? 1 : 2;
  const t = tile(r.albedo, r.size, n);
  const file = path.join(OUT, `tex_${name}.png`);
  fs.writeFileSync(file, encodePng(t.data, t.size, t.size));
  files.push(file);
}

console.log('name'.padEnd(20), 'ms'.padStart(8), 'size'.padStart(6));
for (const [n, ms, sz] of timings) console.log(n.padEnd(20), ms.toFixed(1).padStart(8), String(sz).padStart(6));
console.log('TOTAL'.padEnd(20), timings.reduce((a, b) => a + b[1], 0).toFixed(1).padStart(8));

// Montage into one contact sheet if ImageMagick is around.
try {
  const font = ['/System/Library/Fonts/Supplemental/Arial.ttf'].find((f) => fs.existsSync(f));
  execFileSync('magick', [
    'montage', ...files,
    '-tile', '5x', '-geometry', '+6+6', '-background', '#14171c',
    '-fill', '#e8e2d4', ...(font ? ['-font', font] : []), '-pointsize', '15',
    '-label', '%t',
    path.join(OUT, 'textures_contact_sheet.png'),
  ]);
  console.log(`\ncontact sheet -> ${path.join(OUT, 'textures_contact_sheet.png')}`);
} catch (e) {
  console.log('\n(montage skipped)', (e as Error).message.slice(0, 120));
}
