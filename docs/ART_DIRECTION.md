# Art direction — Rally Adventure

**Primary reference: Screamer 4x4 (Milestone, 2000).**
Not Screamer Rally (1997). This matters — they are very different games and the
earlier one pulls the art in the wrong direction.

| | Screamer Rally (1997) | **Screamer 4x4 (2000)** |
|---|---|---|
| Surface | tarmac / gravel rally stages | **off-road, no roads** |
| Palette | punchy, saturated, high contrast | **muted olive, khaki, low saturation** |
| Landscape | alpine passes, villages, cliffs | **rolling open grassland** |
| Vegetation | dense roadside trees, walls, fences | **sparse individual trees and bushes** |
| Atmosphere | heavy white distance fog | **light haze, long visibility** |
| Vehicle | rally hatchbacks | **literal Jeep CJ/Wrangler, badged** |

Reference files: `reference/screamer/screamer4x4_myabandonware_*.jpg` (15 shots).
The `screamer_ccrally_*` and `screamer2*` files are the 1997 game — useful for
era rendering technique, **not** for palette or landscape.

## 1. Landscape

The defining shape is **rolling, rounded, open grassy hills**. Long smooth
convex ridgelines, broad shallow valleys, gentle gradients. It reads as
temperate upland pasture — Welsh or Californian coastal hills — not desert,
not alpine, not canyon.

- Ground is predominantly **grass**, in olive/khaki/straw tones, with brown
  dirt showing through on worn lines and steeper faces.
- Rock is a **minority surface**, appearing on the steepest faces only.
- Sand and bare earth are local features (a wash, a scar), never the theme.
- Horizons are soft and layered — successive hill shoulders receding, each a
  little paler.

Our terrain generator currently produces: dirt 37%, grass 32%, rock 14%,
gravel 11%, sand 6%, median slope 16.2°. That distribution is close to right.
The failure mode to avoid is the whole frame reading as one beige tone.

## 2. Palette

Sampled from the reference shots. Muted and desaturated — that restraint is the
single biggest difference from the 1997 game.

**Grass and ground cover**
- `#6b6b32` olive mid — the dominant landscape colour
- `#87874a` khaki light, sunlit slopes
- `#4a4f24` olive shadow
- `#9c9456` dry straw highlight
- `#5a6b33` greener, damper hollows

**Earth**
- `#7a6242` packed dirt track
- `#5c4a30` damp earth
- `#a08a63` dusty light
- `#3d3122` deep rut shadow

**Rock**
- `#7d7668` warm grey
- `#57514a` shadowed
- `#9a9384` sunlit face

**Sky and air**
- `#8fb4d6` zenith, soft — NOT a saturated blue
- `#c5d5e2` horizon pale
- `#dfe6ea` haze
- Overcast variant: `#b8bfc4` flat, low contrast

**Vehicle**
- `#c9a227` Renegade yellow · `#2d3f6b` blue · `#2f4034` dark green
- `#6b4a2a` caked mud · `#8a8f92` bare metal · `#1a1a1c` tyre

Overall saturation sits **low**. Where the 1997 game punches, this one sits back.

## 3. Vegetation

**Sparse, not forested.** Individual trees standing alone or in loose groups of
three or four, with wide open ground between them. Mostly broadleaf/bushy
silhouettes rather than dense conifer stands. Low scrub and bushes scattered
across the grass.

This is the correction most needed against our current build, which reads as a
dense conifer forest. Density should drop substantially, and the mix should
shift toward rounded broadleaf shapes with conifers as an accent on higher
ground.

## 4. Atmosphere

Light haze only. Distant hills stay **readable** — you can make out the shape
of a ridge several hundred metres out, it just goes paler and lower-contrast.
Avoid the heavy white fog wall of the 1997 game; it destroys the sense of open
landscape that defines this reference.

Sky is a soft vertical gradient, often slightly overcast. Some shots show rain.
Sun is diffuse rather than a hard disc.

## 5. Camera

Close and low. The vehicle occupies roughly a third of the frame width and sits
low in the composition, with the horizon around 35–45% from the top so the
landscape ahead dominates. Interior/cockpit views exist too, with a real
modelled dashboard and a paper map overlay.

## 6. Vehicle

The hero cars are literal **Jeep CJ / Wrangler** — seven-slot grille, round
headlights, flat fold-down windscreen, flared arches, roll cage, open top,
oversized knobbly tyres, bull bar. Exactly what we have built.

What ours is missing versus the reference:
- **Mud.** Every reference vehicle is visibly caked, heaviest on the lower
  panels, arches and behind the wheels. Ours is clean.
- **Livery.** Competition numbers, sponsor decals, model badging.
- **Black exhaust smoke** under load — prominent and characterful.

## 7. HUD

Busy and chunky, in the era style: time/distance/penalty readouts in the top
corners, analogue dials bottom-left and bottom-right, a checkpoint panel
bottom-centre, and a vehicle damage/status schematic. Ours is cleaner and more
minimal, which is a defensible modernisation — but the reference is a good
argument for adding a damage/status element later.

## 8. Rendering technique (from the 1997 game — still valid)

Era technique is worth borrowing even though the palette is not:
- Low internal resolution, upscaled — currently 480p internal.
- Ordered/Bayer dithering, restrained.
- Nearest-filter magnification with trilinear minification and anisotropy: this
  keeps texels crisp instead of the smeared bilinear of real 1997 hardware.
  Period-accurate blur reads as *low quality* on a modern display; crisp texels
  read as deliberate style. **Deviate from the reference here on purpose.**

## 9. Quality bar

A screenshot of our game is done when:
1. The dominant colour of an open landscape shot is olive-green, not beige.
2. Distant ridgelines are readable, layered, and progressively paler.
3. Trees read as trees, are sparse, and cast shadows.
4. The ground shows texture detail at driver's eye height without tiling or
   giant-texture banding.
5. The Jeep reads instantly as a Jeep and is visibly dirty.
6. The car sits low in frame with the landscape dominating.
7. Nothing in frame is pure black or pure white.
8. It survives a blind A/B against `screamer4x4_*` without being obviously the
   weaker image.
