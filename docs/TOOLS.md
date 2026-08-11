# Development tools — Rally Adventure

Start the dev server first: `npm run dev` (serves on http://127.0.0.1:5183).

**Drive and judge things in a visible, focused browser tab.** Chrome throttles
hidden tabs hard — `requestAnimationFrame` stops entirely and `setTimeout` drops
to about 1 Hz — so a backgrounded tab shows a near-frozen frame. This cost real
debugging time twice; it looks exactly like a rendering bug.

## The game

| | |
|---|---|
| http://127.0.0.1:5183/ | the game |

WASD drive · SPACE handbrake · C camera · R recover · T time of day ·
P retro pipeline on/off · H hide controls · F3 frame counter

`window.game` is exposed for poking at runtime: `terrain`, `vehicle`, `jeep`,
`camera`, `engine`, `fx`, `sky`, `audio`, `hud`, `expedition`, `input`.

## Interactive preview pages

Each isolates one subsystem, with its own controls and live instrumentation.
Far faster to iterate in than the full game.

| Page | What it gives you |
|---|---|
| `/src/audio/audio.preview.html` | Live spectrum analyser + waveform. Sliders for rpm, load, gear, speed, slip, compression, surface, airborne. One-shots for landings and collisions. A scripted drive cycle. **RUN OFFLINE ANALYSIS** renders the cycle through an `OfflineAudioContext` and asserts peak, RMS, DC offset and that the firing frequency tracks rpm. |
| `/src/render/pipeline.preview.html` | Every post pass individually toggleable (bloom, god rays, fog, edge darken, LUT, quantise, dither, vignette, aberration, scanlines). Internal resolution 270–720. Four time-of-day presets plus a scrubber. Exposure and per-effect sliders. GPU/CPU/frame timing. |
| `/src/vehicle/jeep.preview.html` | Turntable. Camera presets 1–8, articulation sweep, steering lock-to-lock, drive mode (wheel spin + lights), wireframe, paint colour cycle. Live triangle count. |
| `/src/world/scatter.preview.html` | Flyable camera over the terrain. Per-layer and per-LOD toggles, instance counts, draw calls, frame time. |
| `/src/fx/fx.preview.html` | Scripted scenarios — cruise, launch, slide, jump, surface tour, water, rock sparks, manual WASD. Per-effect toggles, intensity and track-life sliders, particle count and timing. `window.fxPreview.seek(t)` replays deterministically for reproducible screenshots. |

## Headless verification

No browser, no GPU. Fast enough to sit in an edit-run loop.

```
npx esbuild src/physics/vehicle.test.ts --bundle --platform=node --format=esm \
  --outfile=/tmp/vt.mjs --log-level=error && node /tmp/vt.mjs      # 81/81

npx esbuild src/world/terrain.test.ts --bundle --platform=node --format=esm \
  --outfile=/tmp/tt.mjs --log-level=error && node /tmp/tt.mjs      # 53/53
```

`vehicle.test.ts` — rest, straight line, braking, cornering, jump landings,
a 60 s random-input soak, drivetrain sanity, determinism. Prints the real
numbers (0–100 time, stopping distance, lateral g) beside each check, which
matters more than the pass/fail.

`terrain.test.ts` — determinism, feature audit, drivability, surface mix. The
load-bearing one is **check 2**: 1000 downward Rapier raycasts compared against
`heightAt`, currently agreeing to 0.12 mm. If that regresses, every wheel
raycast in the game lies about where the ground is. Treat it as a correctness
gate, not a quality gate.

## Command-line tools

```
# Regenerate every procedural texture, tile each 2x2 to expose seams,
# and montage them into one contact sheet.
npx esbuild tools/dump-textures.mts --bundle --platform=node --format=esm \
  --outfile=/tmp/dt.mjs --log-level=error && node /tmp/dt.mjs
# -> reference/progress/textures_contact_sheet.png, plus timings per texture
```

```
# Blind A/B: composite our screenshot beside real reference shots with the
# left/right assignment randomised and the panels labelled only A and B.
node tools/blind-compare.mjs --ours reference/progress/game_hero_climb.jpg \
  --refs reference/screamer --out reference/blind --n 4
# -> reference/blind/pair_*.png  and  ANSWER_KEY.json (never show the judge)
```

Then hand the `pair_*.png` files to a fresh agent that has not seen the answer
key and ask which panel is better and why. The point is that it cannot know
which is ours. It asserts each composite is wider than it is tall — an earlier
version silently dropped the right-hand panel and produced single-image
"comparisons" that a judge dutifully refused to score.

## Reference material

- `reference/screamer/` — Screamer 4x4 (primary target) and Screamer Rally
- `reference/era/`, `reference/modern/`, `reference/jeep/`, `reference/terrain/`
- `reference/progress/` — our own captures, texture sheets, audio spectra
- `docs/ART_DIRECTION.md` — sampled palette and the quality-bar checklist
