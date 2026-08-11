# Rally Adventure — off-road driving game

Three.js + TypeScript + Rapier, ~32k lines. Retro look, modern craft.
Visual target: **Screamer 4x4 (2000)**, not Screamer Rally (1997) — see
`docs/ART_DIRECTION.md`. Tooling inventory: `docs/TOOLS.md` (read before
building anything new).

## Commands

```
npm run dev                                   # http://127.0.0.1:5183
npx tsc --noEmit -p tsconfig.json             # must stay at 0 errors

npx esbuild src/physics/vehicle.test.ts --bundle --platform=node --format=esm \
  --outfile=/tmp/vt.mjs --log-level=error && node /tmp/vt.mjs      # 81/81
npx esbuild src/world/terrain.test.ts --bundle --platform=node --format=esm \
  --outfile=/tmp/tt.mjs --log-level=error && node /tmp/tt.mjs      # 53/53
```

esbuild-then-node is how anything in `src/` runs headlessly. No test runner.

## Invariant 1 — chassis forward is local **+Z**

Not −Z. `Vehicle.readBody` uses `(0,0,1)`. A Three camera looks down its own −Z,
so a cockpit camera bolted to the car needs a 180° yaw.

Written wrong once, this produced bugs in five files: chase camera in front of
the car, mirrored `Capture` vantages, inverted `Expedition` slip angle, backwards
steering (masked, because a 180°-wrong camera also mirrors apparent steering),
and a spawn facing a 570 m boundary wall. **If anything is mirrored, check this
first.**

## Invariant 2 — albedo applied twice makes things black

Has bitten this codebase **four times**: black-spike trees, FX 5× too dark, dark
blob rocks, vehicle 5–20× too dark.

Lambert/Phong multiply `map` × `color` × `vertexColors`, each decoded sRGB→linear
first. Two reasonable mid-tones multiply to nothing — 0x2c4520 vertex colour ×
a needle texture at linear 0.079 = albedo 0.004. Lighting cannot recover it.

Decide per material which channel carries colour:
- **detail map** → author high-key (mean ~0.85 linear), colour in `color`/vertex
  colours. Full write-up in `src/world/scatterTextures.ts` header.
- **albedo map** → keep `color` near white.

`JeepModel.ts` has the robust pattern: `TARGET_ALBEDO` states the albedo each
material should reflect; `setTargetAlbedo` divides by the map's measured mean so
`color` means what it says. References: painted steel ~0.15, steel wheel ~0.12,
tyre ~0.02, black plastic ~0.035, terrain ~0.30. Under ~0.05 reads as an unlit
cut-out.

**Symptom:** an object that looks like a silhouette and "doesn't pick up the
light". Almost never a lighting bug.

## Recording

**V** records canvas + game audio to WebM; `tools/clip.sh` converts to MP4/GIF.
Click the page first — a suspended `AudioContext` yields a track that never
delivers a sample and stalls MediaRecorder entirely, producing an empty file
with no video either. `GameAudio.captureStream()` guards against it.

## Architecture

Subsystems know nothing about each other; everything crosses through
`src/types.ts` (`DriveInput`, `WheelState`, `VehicleState`, `TerrainSampler`,
`PhysicsContext`). `src/Game.ts` alone wires them and owns the frame loop.

- `physics/` — custom raycast vehicle on a Rapier body, **not** Rapier's built-in
  controller (tire model too crude). All tuning in `VehicleTuning.ts`, with units.
- `world/` — heightfield, terrain mesh + collider + splat material, scatter, sky
- `render/` — low-res RT → bloom → god rays → fog → LUT → quantise+dither →
  upscale → vignette. **Dither before upscale** or the effect dies. Lambert/Phong
  only; PBR fights the retro target.
- `core/`, `fx/`, `audio/`, `ui/`, `game/`

Sky, fog and lighting must share **one** `AtmosphereUniforms` instance. When they
didn't, the fog pass ran against module defaults (beige horizon, sun in the wrong
quarter) while the dome drew the real sky — the main cause of a uniformly beige
landscape.

## The one correctness gate

`terrain.test.ts` check 2: 1000 Rapier downward raycasts vs `heightAt`, agreeing
to 0.12 mm. Regress it and every wheel raycast lies about the ground. `heightAt`
must use `sampleGridTriangulated` (matches Rapier's anti-diagonal triangulation;
plain bilinear is off by decimetres). Everything else in that suite is a quality
gate.

## Measure while it's happening

Three wrong conclusions here came from observing dead state:

- *"Jeep is invisible"* — tab hidden, rAF stopped, every screenshot a stale frame.
  **Chrome throttles hidden tabs on both paths**: rAF to zero, `setTimeout` to
  ~1 Hz. Drive and judge in a visible, focused tab.
- *"No dust"* — measured on a stopped car. Moving, particles went 129 → 595.
- *"Winding is broken"* — the *test* was wrong: it sampled a LOD-2 chunk where the
  deliberately-vertical skirt is 20% of the buffer, so a perfect mesh scored 0.8
  against a 0.9 threshold.

Before concluding something is broken, confirm it is running. A one-line probe
(`document.hidden`, a stats readout sampled twice) beats reading code.
`window.game` exposes every subsystem for this.

## Judging visual quality

`tools/blind-compare.mjs` composites our screenshot beside references with sides
randomised; a fresh agent that hasn't seen `ANSWER_KEY.json` picks the better one.
Standing: **4–0 vs Screamer 4x4**.

Verify the tool's output before trusting a verdict — an earlier version used
`-flatten` after `+append`, which composites onto the first image's stale page
geometry and silently cropped the right panel, yielding single-image
"comparisons". Now asserted wider-than-tall.

Open critique: near-field sand smears at eye height; horizon mountains are a row
of near-identical cones.

## Subagents

Assign **disjoint file ownership** explicitly — three had to be killed mid-flight
for clobbering each other. Agents cold-start and re-derive context, so for a
well-diagnosed fix, doing it directly is cheaper. Give them the harness facts up
front (hidden-tab throttling, driving via patched input, the esbuild command) or
each burns a long detour rediscovering them.
