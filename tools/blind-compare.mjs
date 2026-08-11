#!/usr/bin/env node
/**
 * Blind A/B comparison harness.
 *
 * Composites one of our screenshots beside a real-game reference image, with
 * the left/right assignment randomised and the panels labelled only "A" and
 * "B". The answer key is written to a separate file that the judging agent is
 * never shown, so its verdict is genuinely blind.
 *
 * Usage:
 *   node tools/blind-compare.mjs --ours <img> --refs <dir-or-glob> [--out <dir>] [--n 6]
 *
 * Writes:
 *   <out>/pair_<i>.png     the composite the judge sees
 *   <out>/ANSWER_KEY.json  which side was ours (do not show the judge)
 */
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';

const args = parseArgs(process.argv.slice(2));
const oursPath = args.ours;
const refsArg = args.refs;
const outDir = args.out ?? 'reference/blind';
const maxPairs = Number(args.n ?? 6);
/** Panels are normalised to this height so neither side wins on resolution. */
const PANEL_H = Number(args.height ?? 540);

/**
 * ImageMagick on macOS ships without a configured font list, so `-annotate`
 * fails unless we hand it an explicit font file.
 */
const FONT = [
  '/System/Library/Fonts/Supplemental/Arial.ttf',
  '/Library/Fonts/Arial Unicode.ttf',
  '/System/Library/Fonts/Helvetica.ttc',
].find((f) => fs.existsSync(f));

if (!oursPath || !refsArg) {
  console.error('usage: blind-compare.mjs --ours <img> --refs <dir> [--out <dir>] [--n 6]');
  process.exit(2);
}

const refs = collectRefs(refsArg);
if (refs.length === 0) {
  console.error(`no reference images found under ${refsArg}`);
  process.exit(1);
}

fs.mkdirSync(outDir, { recursive: true });

// Deterministic-but-unpredictable ordering: shuffle with a fresh random seed so
// a judging agent can't learn a positional habit across runs.
shuffle(refs);
const chosen = refs.slice(0, maxPairs);

const key = [];
chosen.forEach((ref, i) => {
  const oursLeft = crypto.randomInt(2) === 0;
  const left = oursLeft ? oursPath : ref;
  const right = oursLeft ? ref : oursPath;
  const out = path.join(outDir, `pair_${i}.png`);

  const leftTmp = path.join(outDir, `.l_${i}.png`);
  const rightTmp = path.join(outDir, `.r_${i}.png`);

  // Normalise both panels to identical height and label them A / B. Resolution
  // parity matters: a 1920px screenshot next to a 320px scan would give the
  // game away instantly and bias the verdict.
  labelPanel(left, leftTmp, 'A');
  labelPanel(right, rightTmp, 'B');

  // `+append` concatenates but leaves the canvas geometry of the FIRST image on
  // the result. A following `-flatten` then composites onto that stale canvas
  // and silently crops the right-hand panel clean off, yielding a single-panel
  // "comparison". `+repage` resets the geometry; no flatten is needed at all.
  magick([leftTmp, rightTmp, '+append', '+repage', out]);

  const [w, h] = execFileSync('magick', ['identify', '-format', '%w %h', out])
    .toString().trim().split(' ').map(Number);
  // A real pair is far wider than tall. If it isn't, the append collapsed and
  // every verdict downstream would be judged against half a comparison.
  if (w < h * 1.5) {
    throw new Error(`blind-compare: ${out} is ${w}x${h} — the two panels did not concatenate`);
  }
  fs.rmSync(leftTmp, { force: true });
  fs.rmSync(rightTmp, { force: true });

  key.push({ pair: `pair_${i}.png`, A: oursLeft ? 'OURS' : 'REFERENCE', B: oursLeft ? 'REFERENCE' : 'OURS', reference: ref });
  console.log(`wrote ${out}`);
});

fs.writeFileSync(path.join(outDir, 'ANSWER_KEY.json'), JSON.stringify(key, null, 2));
console.log(`\n${chosen.length} pairs in ${outDir}. Answer key: ${path.join(outDir, 'ANSWER_KEY.json')} (do not show the judge).`);

// ---------------------------------------------------------------- helpers

function labelPanel(src, dst, letter) {
  magick([
    src,
    '-resize', `x${PANEL_H}`,
    '-gravity', 'northwest',
    '-background', '#000000',
    '-splice', '0x34',
    '-fill', '#ffffff',
    ...(FONT ? ['-font', FONT] : []),
    '-pointsize', '24',
    '-annotate', '+12+6', letter,
    '-bordercolor', '#101014',
    '-border', '6',
    dst,
  ]);
}

function magick(argv) {
  execFileSync('magick', argv, { stdio: ['ignore', 'ignore', 'pipe'] });
}

function collectRefs(target) {
  const out = [];
  const walk = (p) => {
    const st = fs.statSync(p);
    if (st.isDirectory()) {
      for (const e of fs.readdirSync(p)) {
        if (e.startsWith('.') || e === 'blind' || e === 'progress') continue;
        walk(path.join(p, e));
      }
    } else if (/\.(png|jpe?g|webp)$/i.test(p) && st.size > 15000) {
      out.push(p);
    }
  };
  walk(target);
  return out;
}

function shuffle(a) {
  for (let i = a.length - 1; i > 0; i--) {
    const j = crypto.randomInt(i + 1);
    [a[i], a[j]] = [a[j], a[i]];
  }
}

function parseArgs(argv) {
  const o = {};
  for (let i = 0; i < argv.length; i++) {
    if (argv[i].startsWith('--')) o[argv[i].slice(2)] = argv[i + 1]?.startsWith('--') ? true : argv[++i];
  }
  return o;
}
