# Reference corpus

The subdirectories `screamer/`, `era/`, `modern/`, `jeep/` and `terrain/` hold
third-party screenshots and photographs gathered as visual reference during
development. **They are deliberately not committed** — they are copyrighted
material belonging to their respective publishers, and republishing them here
would be redistribution.

They are development aids only. Nothing in `src/` reads them; the game builds
and runs without them. They are consumed by `tools/blind-compare.mjs` and read
directly when judging art direction.

`progress/` *is* committed — those are our own captures, texture contact sheets
and audio spectra.

## What was gathered

| Directory | Contents | Sources |
|---|---|---|
| `screamer/` | Screamer 4x4 (2000) — the primary art target — plus Screamer Rally and Screamer 2 | MobyGames, myabandonware |
| `era/` | Sega Rally, Colin McRae Rally, NFS III, International Rally — 1995-99 rendering technique | myabandonware |
| `modern/` | SnowRunner, Dirt Rally 2.0, Forza Horizon 5, Expeditions — modern terrain/lighting bar | Steam CDN |
| `jeep/` | Willys MB, CJ7, Wrangler TJ/JK reference photographs | Wikimedia Commons |
| `terrain/` | Off-road trail, canyon and upland photography | Wikimedia Commons, Unsplash |

To rebuild the corpus, search for the titles above and drop the images into the
matching directory. Filenames follow `<title>_<source>_<n>.jpg`. Only the
relative proportions matter for `blind-compare`, not the exact files.

The sampled palette and every art-direction conclusion drawn from these images
is recorded in `docs/ART_DIRECTION.md`, so the guidance survives without them.
