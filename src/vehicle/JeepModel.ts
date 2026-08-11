/**
 * ============================================================================
 *  THE JEEP — procedural model + rig
 * ============================================================================
 *
 * A Wrangler-class open-top 4x4, built entirely in code from the primitives in
 * `jeepGeometry`. There are no asset files anywhere in this project.
 *
 * Coordinate frame (matches `VehicleTuning`):
 *     +X = right,  +Y = up,  +Z = forward
 * The model origin sits **on the ground** in the design pose, exactly where the
 * physics body origin sits, so `object3d.position = state.position` is correct
 * with no fudge offset. Wheel hubs are at y = 0.40 (the tyre rolling radius),
 * the body is 1.72 m wide over the tub and the wheelbase is 2.60 m — all three
 * lifted straight out of `VehicleTuning` so the model and the collider agree.
 *
 * Scene graph:
 *
 *   object3d                     chassis transform, straight from physics
 *    +- body                     bodywork + interior + everything frame-mounted
 *    |   +- axleFront/axleRear   solid axles: follow their wheel pair
 *    |   +- springs, shocks,     linkage recomputed every frame from the
 *    |      links, driveshafts   *actual* wheel positions
 *    +- wheel x4                 placed from WheelState in chassis-local space
 *
 * The wheels are deliberately NOT children of `body`: suspension travel is
 * already baked into `WheelState.position`, so they are positioned from that and
 * the axles are fitted *to them*. That inversion — bodywork drives nothing,
 * wheels drive everything underneath — is what makes the articulation read as
 * real when one corner drops into a rut.
 *
 * A note on `chamferBox` orientation. `extrudeShape` maps the extrusion *caps*
 * with clean world-scale UVs and the *side walls* with a cheap arc-length
 * approximation that stretches badly on a long panel. So every box here is
 * built with its thin axis as the extrusion axis (`chamferBoxOn('y', ...)` for
 * a horizontal panel, `'x'` for a door) which puts the big visible faces on the
 * caps. Get that backwards and the paint smears into vertical stripes.
 */
import * as THREE from 'three';

import type { VehicleState } from '../types';
import {
  Assembler,
  buildWheel,
  chamferBox,
  chamferBoxOn,
  coilSpring,
  countTriangles,
  extrudeShape,
  lathe,
  mergeAll,
  place,
  polyShape,
  tube,
  type TyreSpec,
} from './jeepGeometry';
import { getJeepTextures } from './jeepTextures';
import { getMap } from '../render/textures';

// ---------------------------------------------------------------------------
// dimensions — every number the body is built from
// ---------------------------------------------------------------------------

const D = {
  /** Hub centre height in the design pose = tyre rolling radius. */
  hubY: 0.4,
  /** Front / rear axle centreline. */
  axleZ: 1.3,
  /** Half track. */
  halfTrack: 0.8,

  /** Outer skin of the tub sides — half of the 1.72 m body width. */
  halfWidth: 0.86,
  /** Rocker line: bottom edge of the body sides. */
  rocker: 0.5,
  /** Radius of the wheel-arch cut. Only 9 cm of daylight over a 0.40 m tyre —
   *  a bigger gap is the single most common way a procedural 4x4 ends up
   *  looking like a monster truck instead of a Jeep. */
  archR: 0.465,
  /** Belt line: top of the tub sides and doors. */
  belt: 1.26,
  /** Flat top of the front fenders and the hood. */
  fenderTop: 1.14,
  /** Front-most body panel. */
  noseZ: 1.755,
  /** Rear-most body panel. */
  tailZ: -1.7,
  /** Cowl: where the fenders stop and the windscreen starts. */
  cowlZ: 0.6,

  /** Tub floor. */
  floorY: 0.6,
  /** Ladder frame rails. */
  railX: 0.42,
  railBottom: 0.46,
  railTop: 0.6,

  /** Windscreen. */
  screenBaseY: 1.245,
  screenBaseZ: 0.56,
  screenHeight: 0.6,
  screenRake: 0.26, // rad, ~15 degrees

  /** Roll cage. */
  cageY: 1.815,
  cageX: 0.74,
  hoopZ: -0.52,
  rearHoopZ: -1.46,
  /** Roof rack deck. */
  rackY: 1.875,

  /** Grille panel. */
  grilleHalfW: 0.65,
  grilleBottom: 0.6,
  grilleTop: 1.145,
  grilleZ: 1.775,

  /** Spare wheel on the tailgate. */
  spareZ: -1.87,
  spareY: 1.16,
  spareX: 0.07,
} as const;

/**
 * 33-inch mud terrain. The radius matches `TireTuning.radius` exactly — a visual
 * radius that disagrees with the physics one makes the car look like it is
 * hovering or sunk, and no amount of other detail recovers from that.
 */
const TYRE: TyreSpec = {
  radius: 0.4,
  width: 0.3,
  rimRadius: 0.19,
  treadDepth: 0.026,
  treadCount: 10,
  radialSegments: 12,
};

// ---------------------------------------------------------------------------
// small geometry helpers, local to the model
// ---------------------------------------------------------------------------

/** Plain box. ~12 triangles against a chamfered box's ~90 — use it for any
 *  detail small enough that the chamfer would never resolve on screen. */
function box(w: number, h: number, d: number): THREE.BufferGeometry {
  return new THREE.BoxGeometry(w, h, d);
}

/** Panel drawn as a (z, y) profile and extruded across X. */
function panelZY(pts: Array<[number, number]>, thickness: number, chamfer = 0.014): THREE.BufferGeometry {
  const g = extrudeShape(polyShape(pts), thickness, { chamfer });
  g.rotateY(-Math.PI / 2);
  return g;
}

/** Panel drawn as an (x, z) plan and extruded up Y. */
function panelXZ(pts: Array<[number, number]>, thickness: number, chamfer = 0.012): THREE.BufferGeometry {
  const g = extrudeShape(polyShape(pts), thickness, { chamfer });
  g.rotateX(Math.PI / 2);
  return g;
}

/**
 * Surface of revolution about +Z — lights, bezels, filler caps.
 *
 * Profiles are written outside-in (`[[rimRadius, 0], ... [0, peak]]`) because
 * `LatheGeometry` derives its winding from the direction the profile travels:
 * feed it a radius that grows and every lamp lens in the model renders
 * inside-out and reads as an empty black socket.
 */
function domeZ(profile: Array<[number, number]>, segments = 10): THREE.BufferGeometry {
  const g = lathe(profile, segments);
  g.rotateX(Math.PI / 2);
  return g;
}

/** Convex lamp lens: flat rim of `r`, domed forward by `depth`. */
function lens(r: number, depth: number, segments = 10): THREE.BufferGeometry {
  return domeZ(
    [
      [r, 0],
      [r * 0.92, depth * 0.5],
      [r * 0.62, depth * 0.85],
      [0, depth],
    ],
    segments,
  );
}

/** Cylinder running along +Y from y=0 to y=h. */
function rodY(r: number, h: number, seg = 6): THREE.BufferGeometry {
  const g = new THREE.CylinderGeometry(r, r, h, seg, 1, false);
  g.translate(0, h / 2, 0);
  return g;
}

/** Cylinder along X, centred. */
function rodX(r: number, len: number, seg = 8): THREE.BufferGeometry {
  const g = new THREE.CylinderGeometry(r, r, len, seg, 1, false);
  g.rotateZ(Math.PI / 2);
  return g;
}

/** Cylinder along Z, centred. */
function rodZ(r: number, len: number, seg = 8): THREE.BufferGeometry {
  const g = new THREE.CylinderGeometry(r, r, len, seg, 1, false);
  g.rotateX(Math.PI / 2);
  return g;
}

/** Samples a circular arc in the (z, y) plane. */
function arcZY(cz: number, cy: number, r: number, a0: number, a1: number, steps: number): Array<[number, number]> {
  const out: Array<[number, number]> = [];
  for (let i = 0; i <= steps; i++) {
    const a = a0 + ((a1 - a0) * i) / steps;
    out.push([cz + Math.cos(a) * r, cy + Math.sin(a) * r]);
  }
  return out;
}

/** Default tube settings: cheap, and round enough at 5 sides for a 32 mm bar. */
const TUBE = { radialSegments: 5, density: 3, cornerRadius: 0.16 } as const;

// ---------------------------------------------------------------------------
// materials
// ---------------------------------------------------------------------------

export type MatKey =
  | 'body'
  | 'trim'
  | 'steel'
  | 'grime'
  | 'chrome'
  | 'glass'
  | 'rubber'
  | 'rim'
  | 'vinyl'
  | 'gauge'
  | 'dark'
  | 'lampHead'
  | 'lampAmber'
  | 'lampBrake'
  | 'lampReverse';

type Mats = Record<MatKey, THREE.Material>;

/**
 * Mean linear luminance of a texture, or `null` if the pixels aren't reachable.
 *
 * Textures from the procedural library are `DataTexture`, so the bytes are
 * right there in `image.data`; canvas-backed ones have to be read back through
 * a 2D context.
 */
function meanLinearLuminance(tex: THREE.Texture | null): number | null {
  if (!tex) return null;
  const img = tex.image as { data?: Uint8ClampedArray; width?: number; height?: number } | undefined;
  let data = img?.data;
  if (!data && img?.width) {
    const c = document.createElement('canvas');
    c.width = img.width;
    c.height = img.height ?? img.width;
    const ctx = c.getContext('2d');
    if (!ctx) return null;
    ctx.drawImage(img as unknown as CanvasImageSource, 0, 0);
    data = ctx.getImageData(0, 0, c.width, c.height).data;
  }
  if (!data || data.length === 0) return null;

  const toLinear = (v: number) => (v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4));
  let sum = 0;
  let n = 0;
  for (let i = 0; i < data.length; i += 4) {
    sum +=
      0.2126 * toLinear(data[i]! / 255) +
      0.7152 * toLinear(data[i + 1]! / 255) +
      0.0722 * toLinear(data[i + 2]! / 255);
    n++;
  }
  return n ? sum / n : null;
}

/**
 * Make `material.color` mean the albedo it says it means.
 *
 * Three multiplies `map` by `color`, both in linear space. Where the map is a
 * *detail* map — sun fade, orange peel, stone chips, tread shadow — its mean is
 * around 0.45, so asking for a colour delivers less than half of it. Stacked on
 * tints that were already dark, the whole vehicle was landing at an effective
 * albedo of 0.006–0.05 against sand at ~0.3, which is why it read as an unlit
 * silhouette that ignored the sun.
 *
 * Dividing the tint by the map's mean makes the two independent again: the map
 * supplies variation around unity, `color` supplies the albedo.
 */
function setTargetAlbedo(
  mat: THREE.MeshLambertMaterial | THREE.MeshPhongMaterial,
  target: number,
): void {
  const mean = meanLinearLuminance(mat.map) ?? 1;
  const cur = 0.2126 * mat.color.r + 0.7152 * mat.color.g + 0.0722 * mat.color.b;
  const delivered = cur * mean;
  if (delivered < 1e-4) return;
  // Scale uniformly so the authored hue survives and only the level moves.
  mat.color.multiplyScalar(target / delivered);
}

/**
 * Target diffuse albedo per material, as linear luminance.
 *
 * These are what the surface should actually reflect, measured the way the
 * renderer sees it — not tint values to be multiplied by whatever the map
 * happens to average. Real references: weathered painted steel ~0.15, a steel
 * wheel ~0.12, tyre rubber ~0.02, black plastic trim ~0.035. The sand this
 * vehicle drives on sits near 0.30, so anything much under 0.05 reads as an
 * unlit cut-out however strong the sun is.
 *
 * Anything not listed keeps its authored colour.
 */
const TARGET_ALBEDO: Partial<Record<MatKey, number>> = {
  body: 0.135,
  steel: 0.155,
  rim: 0.125,
  rubber: 0.022,
  trim: 0.038,
  grime: 0.055,
  vinyl: 0.050,
};

function buildMaterials(color: THREE.ColorRepresentation): Mats {
  const jt = getJeepTextures();

  // Two different weathering maps, because they do different jobs:
  //
  //  - Body panels use `jeepTextures.paintPanel`: a luminance-only map of sun
  //    fade, orange peel and a scatter of stone chips, tinted by
  //    `material.color`. Panels stay the colour you asked for.
  //  - Bare steel — bull bar, cage, sliders, rack — uses the shared library's
  //    `paintedMetal`, whose baked-in rust blooms and deep chips are exactly
  //    right for gear that gets dragged over rocks, and far too much for a door.
  const steelMap = getMap('paintedMetal', { color: '5a6068', repeat: 2.4 });
  const tyreMap = getMap('tyre', { repeat: [3, 1] });
  const glassMap = getMap('glass');
  const chromeMap = getMap('chrome');
  const vinylMap = getMap('vinyl', { repeat: 3 });
  const headMap = getMap('headlight');

  const lamp = (hex: number, emissive: number): THREE.MeshPhongMaterial =>
    new THREE.MeshPhongMaterial({ color: hex, emissive, shininess: 70, specular: 0x333333 });

  const mats: Mats = {
    body: new THREE.MeshPhongMaterial({
      map: jt.paintPanel,
      color: new THREE.Color(color),
      shininess: 40,
      specular: 0x42423e,
    }),
    trim: new THREE.MeshLambertMaterial({ map: jt.plastic, color: 0xcfcfcf }),
    steel: new THREE.MeshPhongMaterial({ map: steelMap, color: 0x8f959c, shininess: 26, specular: 0x2e2e2e }),
    grime: new THREE.MeshLambertMaterial({ map: jt.grime, color: 0x807567 }),
    chrome: new THREE.MeshPhongMaterial({ map: chromeMap, color: 0xdde4ea, shininess: 96, specular: 0xcccccc }),
    glass: new THREE.MeshPhongMaterial({
      map: glassMap,
      color: 0xbcd2d8,
      transparent: true,
      opacity: 0.24,
      shininess: 120,
      specular: 0xffffff,
      depthWrite: false,
      side: THREE.DoubleSide,
    }),
    rubber: new THREE.MeshLambertMaterial({ map: tyreMap, color: 0x565452 }),
    rim: new THREE.MeshPhongMaterial({ map: jt.steel, color: 0x6a6f74, shininess: 40, specular: 0x444444 }),
    vinyl: new THREE.MeshLambertMaterial({ map: vinylMap, color: 0x9c8f78 }),
    gauge: new THREE.MeshLambertMaterial({ map: jt.gauges, emissive: 0x1a140a }),
    dark: new THREE.MeshLambertMaterial({ color: 0x191b1f }),
    lampHead: new THREE.MeshPhongMaterial({
      map: headMap,
      color: 0xf4f8fa,
      emissive: 0x000000,
      shininess: 100,
      specular: 0xffffff,
    }),
    lampAmber: lamp(0xd8811c, 0x140800),
    lampBrake: lamp(0x8e1712, 0x150202),
    lampReverse: lamp(0xd8d4c8, 0x0a0a0a),
  };

  for (const [key, target] of Object.entries(TARGET_ALBEDO)) {
    setTargetAlbedo(mats[key as MatKey] as THREE.MeshPhongMaterial, target);
  }
  return mats;
}

// ---------------------------------------------------------------------------
// body shell
// ---------------------------------------------------------------------------

/**
 * The side silhouette, as a (z, y) polygon with both wheel arches cut into the
 * bottom edge. This one outline carries most of the "that's a Jeep" signal:
 * flat fender tops, a step up at the cowl, a dead-level belt line, and both
 * ends chopped off right above the tyres for approach and departure clearance.
 */
function bodySideProfile(): Array<[number, number]> {
  const { rocker, archR, hubY, axleZ, noseZ, tailZ, belt, fenderTop, cowlZ } = D;
  const aRocker = Math.asin((rocker - hubY) / archR);
  const aNose = Math.acos((noseZ - axleZ) / archR);
  const aTail = Math.acos((tailZ + axleZ) / archR);

  return [
    [tailZ, belt],
    [cowlZ + 0.02, belt],
    // step down onto the flat fender tops
    [cowlZ + 0.11, fenderTop],
    [noseZ - 0.05, fenderTop],
    [noseZ, fenderTop - 0.06],
    // nose panel down to the front arch
    [noseZ, hubY + Math.sin(aNose) * archR],
    ...arcZY(axleZ, hubY, archR, aNose, Math.PI - aRocker, 6),
    [axleZ - Math.cos(aRocker) * archR, rocker],
    [-axleZ + Math.cos(aRocker) * archR, rocker],
    ...arcZY(-axleZ, hubY, archR, aRocker, Math.PI - aTail, 6),
    [tailZ, hubY + Math.sin(aTail) * archR],
  ];
}

function buildBodyShell(a: Assembler): void {
  const { halfWidth: HW, noseZ, tailZ, belt, fenderTop, floorY, rocker, cowlZ } = D;

  // --- tub sides -----------------------------------------------------------
  a.addPair('body', panelZY(bodySideProfile(), 0.05, 0.018), { pos: [HW - 0.025, 0, 0] });

  // Doors, proud of the skin by 8 mm so the shut line casts a real shadow. The
  // aperture is what makes the side read as a vehicle rather than a slab.
  const doorZ0 = 0.5;
  const doorZ1 = -0.5;
  a.addPair('body', chamferBoxOn('x', 0.05, belt - rocker - 0.08, doorZ0 - doorZ1 - 0.03, 0.016, 0.05), {
    pos: [HW + 0.004, (belt + rocker) / 2 + 0.01, (doorZ0 + doorZ1) / 2],
  });
  // Shut lines. A door that is merely *proud* disappears the moment the sun is
  // behind you; an explicit dark seam reads from any angle and costs 12 tris.
  for (const z of [doorZ0 - 0.012, doorZ1 + 0.012]) {
    a.addPair('dark', box(0.022, belt - rocker - 0.075, 0.02), { pos: [HW + 0.018, (belt + rocker) / 2 + 0.01, z] });
  }
  a.addPair('dark', box(0.022, 0.02, doorZ0 - doorZ1), { pos: [HW + 0.018, rocker + 0.038, (doorZ0 + doorZ1) / 2] });
  // Waist swage along the whole flank — a real Wrangler feature, and free shading.
  a.addPair('trim', box(0.014, 0.035, 2.9), { pos: [HW + 0.03, rocker + 0.17, -0.3] });
  a.addPair('chrome', chamferBoxOn('x', 0.035, 0.05, 0.17, 0.012), { pos: [HW + 0.04, belt - 0.16, -0.28] });
  for (const hz of [doorZ0 - 0.03, doorZ1 + 0.03]) {
    a.addPair('trim', box(0.05, 0.07, 0.05), { pos: [HW + 0.015, belt - 0.12, hz] });
    a.addPair('trim', box(0.05, 0.07, 0.05), { pos: [HW + 0.015, rocker + 0.14, hz] });
  }

  // Rolled lip along the top of the tub — real tubs have one, and it is what
  // stops the belt line reading as a raw cut through a slab.
  a.addPair('body', chamferBoxOn('y', 0.09, 0.03, 1.14, 0.01), { pos: [HW - 0.015, belt + 0.012, -1.12] });
  a.addPair('body', chamferBoxOn('y', 0.09, 0.03, 0.16, 0.01), { pos: [HW - 0.015, belt + 0.012, 0.57] });

  // --- floor, bulkhead, tailgate ------------------------------------------
  a.add('grime', panelXZ([[-HW, cowlZ - 0.02], [HW, cowlZ - 0.02], [HW, tailZ + 0.04], [-HW, tailZ + 0.04]], 0.05), {
    pos: [0, floorY, 0],
  });
  a.add('body', extrudeShape(polyShape([[-HW, floorY], [HW, floorY], [HW, belt], [-HW, belt]]), 0.05, { chamfer: 0.014 }), {
    pos: [0, 0, cowlZ - 0.05],
  });
  // tailgate: inner panel plus a proud outer skin with a hinge strip
  a.add('body', extrudeShape(polyShape([[-HW + 0.02, 0.7], [HW - 0.02, 0.7], [HW - 0.02, belt], [-HW + 0.02, belt]]), 0.05, { chamfer: 0.016 }), {
    pos: [0, 0, tailZ + 0.025],
  });
  a.add('body', chamferBox(1.6, 0.48, 0.035, 0.014, 0.035), { pos: [0, 0.97, tailZ - 0.015] });
  a.addPair('trim', box(0.055, 0.4, 0.035), { pos: [0.78, 0.97, tailZ - 0.026] });

  // --- fender tops ---------------------------------------------------------
  // Flat, square-edged and clearly separate from the hood. In the three-quarter
  // view this is the strongest single cue after the grille.
  a.addPair('body', panelXZ([[0.585, cowlZ + 0.07], [HW, cowlZ + 0.07], [HW, noseZ], [0.585, noseZ]], 0.045, 0.014), {
    pos: [0, fenderTop, 0],
  });
  a.addPair('body', chamferBoxOn('x', 0.05, 0.1, noseZ - cowlZ - 0.07, 0.014), {
    pos: [HW - 0.02, fenderTop - 0.065, (noseZ + cowlZ + 0.07) / 2],
  });

  // --- hood ----------------------------------------------------------------
  const hz0 = cowlZ + 0.05;
  const hz1 = noseZ - 0.01;
  a.add('body', chamferBoxOn('y', 1.14, 0.05, hz1 - hz0, 0.018, 0.05), { pos: [0, fenderTop + 0.012, (hz0 + hz1) / 2] });
  a.add('body', chamferBoxOn('y', 0.76, 0.045, hz1 - hz0 - 0.3, 0.018, 0.06), {
    pos: [0, fenderTop + 0.042, (hz0 + hz1) / 2 - 0.03],
  });
  for (const vx of [-0.25, 0.25]) a.add('dark', box(0.17, 0.018, 0.1), { pos: [vx, fenderTop + 0.062, hz1 - 0.52] });
  // Shut line between hood and fender — the hood is a separate, narrower panel
  // on a Jeep and it has to read that way from every angle.
  a.addPair('dark', box(0.022, 0.03, hz1 - hz0), { pos: [0.578, fenderTop + 0.022, (hz0 + hz1) / 2] });
  // hood catches — the little black clamps either side, right at the cowl
  a.addPair('trim', box(0.045, 0.075, 0.11), { pos: [0.53, fenderTop - 0.005, hz0 + 0.06] });

  // cowl panel + wipers
  a.add('body', chamferBoxOn('y', 1.68, 0.045, 0.15, 0.014), { pos: [0, D.screenBaseY - 0.035, cowlZ - 0.02] });
  for (const wx of [-0.36, 0.32]) {
    a.add('dark', box(0.018, 0.014, 0.44), { pos: [wx, D.screenBaseY - 0.005, cowlZ - 0.2], rot: [0, 0.12 * Math.sign(wx), 0] });
  }

  // --- engine bay filler ---------------------------------------------------
  // Stops daylight showing through the grille and under the hood from a low
  // camera, and gives the bay something that reads as machinery.
  a.add('dark', chamferBox(0.84, 0.44, 0.7, 0.04), { pos: [0, 0.86, 1.12] });
  a.add('dark', box(1.06, 0.42, 0.05), { pos: [0, 0.87, D.grilleZ - 0.11] });
}

function buildGrille(a: Assembler): void {
  const { grilleHalfW: GW, grilleBottom, grilleTop, grilleZ, noseZ, fenderTop, halfWidth: HW } = D;

  // Seven slots, cut as real holes and extruded so they have depth and
  // self-shadow. This is the identity feature: it is worth the triangles.
  const panel = polyShape([
    [-GW, grilleBottom],
    [GW, grilleBottom],
    [GW, grilleTop],
    [-GW, grilleTop],
  ]);

  const SLOTS = 7;
  const slotW = 0.078;
  const gap = 0.017;
  const spanHalf = (SLOTS * slotW + (SLOTS - 1) * gap) / 2;
  const slotY0 = grilleBottom + 0.085;
  const slotY1 = grilleTop - 0.065;
  const r = 0.028;
  for (let i = 0; i < SLOTS; i++) {
    const cx = -spanHalf + slotW / 2 + i * (slotW + gap);
    const hx = slotW / 2;
    // Rounded ends drawn as clipped corners — at any sane viewing distance
    // indistinguishable from a real radius, at a quarter of the vertices.
    panel.holes.push(
      new THREE.Path([
        new THREE.Vector2(cx - hx, slotY0 + r),
        new THREE.Vector2(cx - hx + r, slotY0),
        new THREE.Vector2(cx + hx - r, slotY0),
        new THREE.Vector2(cx + hx, slotY0 + r),
        new THREE.Vector2(cx + hx, slotY1 - r),
        new THREE.Vector2(cx + hx - r, slotY1),
        new THREE.Vector2(cx - hx + r, slotY1),
        new THREE.Vector2(cx - hx, slotY1 - r),
      ]),
    );
  }
  a.add('body', extrudeShape(panel, 0.08, { chamfer: 0.016 }), { pos: [0, 0, grilleZ] });
  // Radiator core, set well back so the slots read as openings, not as paint.
  a.add('dark', box(0.72, slotY1 - slotY0 + 0.05, 0.04), { pos: [0, (slotY0 + slotY1) / 2, grilleZ - 0.1] });

  // --- headlights ----------------------------------------------------------
  const hy = (slotY0 + slotY1) / 2;
  const hx = 0.49;
  // 16 sides, not 12: these two circles are what the eye lands on first and a
  // visible dodecagon is the one place faceting actually costs the illusion.
  a.addPair(
    'chrome',
    domeZ([[0.11, 0.0], [0.143, 0.016], [0.15, 0.046], [0.134, 0.05], [0.124, 0.026], [0.11, 0.014]], 16),
    { pos: [hx, hy, grilleZ + 0.034] },
  );
  a.addPair('lampHead', lens(0.122, 0.05, 16), { pos: [hx, hy, grilleZ + 0.03] });

  // --- fender front faces + turn signals -----------------------------------
  a.addPair(
    'body',
    extrudeShape(
      polyShape([
        [GW - 0.01, grilleBottom],
        [HW, grilleBottom + 0.02],
        [HW, fenderTop - 0.02],
        [GW - 0.01, fenderTop + 0.005],
      ]),
      0.06,
      { chamfer: 0.016 },
    ),
    { pos: [0, 0, noseZ - 0.02] },
  );
  a.addPair('lampAmber', lens(0.058, 0.03, 8), { pos: [0.755, fenderTop - 0.15, noseZ + 0.016] });
  // small round marker lamps under the headlights, CJ style
  a.addPair('dark', lens(0.046, 0.022, 8), { pos: [hx, grilleBottom + 0.048, grilleZ + 0.028] });
}

function buildFlares(a: Assembler): void {
  // A wheel-arch flare is a surface of revolution about the wheel axis, so a
  // partial lathe is both the cheapest and the most accurate way to build one.
  // Kept to a 10 cm lip: it has to shade the tyre without turning into a tube.
  const { archR, hubY, axleZ, halfWidth: HW, rocker } = D;
  const aRocker = Math.asin((rocker - hubY) / archR);
  const phiStart = -(Math.PI - aRocker + 0.06);
  const phiLength = Math.PI - 2 * aRocker + 0.12;

  // (radius from hub, distance out from the centreline)
  const profile: Array<[number, number]> = [
    [archR - 0.012, HW - 0.02],
    [archR + 0.02, HW + 0.012],
    [archR + 0.055, HW + 0.06],
    [archR + 0.062, HW + 0.105],
    [archR + 0.036, HW + 0.108],
    [archR + 0.004, HW + 0.06],
  ];

  // Inner liner. Without it the arch is a hole straight through the bodywork and
  // the wheel appears to float in a black void — the single biggest reason a
  // procedural 4x4 reads as unfinished from three-quarter front.
  //
  // This one is deliberately wound the opposite way to every other lathe here:
  // the liner sits at a larger radius than the tyre, so the face you see when
  // you look into the arch is its CONCAVE side, the one pointing back at the
  // axle. Reverse it "to match the lamps" and the liner vanishes and the void
  // comes back.
  const liner: Array<[number, number]> = [
    [archR - 0.005, HW - 0.03],
    [archR - 0.02, HW - 0.1],
    [archR - 0.02, HW - 0.24],
    [archR + 0.02, HW - 0.3],
  ];

  for (const z of [axleZ, -axleZ]) {
    const g = lathe(profile, 8, phiStart, phiLength);
    g.rotateZ(-Math.PI / 2);
    a.addPair('trim', g, { pos: [0, hubY, z] });

    const l = lathe(liner, 8, phiStart + 0.06, phiLength - 0.12);
    l.rotateZ(-Math.PI / 2);
    a.addPair('dark', l, { pos: [0, hubY, z] });
  }
}

function buildTopStructure(a: Assembler): void {
  const { screenBaseY, screenBaseZ, screenHeight: H, screenRake, cageX, cageY, hoopZ, rearHoopZ, halfWidth: HW, belt, rackY } = D;

  // --- windscreen frame ----------------------------------------------------
  // Built flat in its own plane, then tipped back as a unit, so every member
  // stays square to the glass exactly like the real folding frame.
  const local = new Assembler();
  local.addPair('body', chamferBoxOn('z', 0.075, H + 0.08, 0.06, 0.018), { pos: [0.8, H / 2, 0] });
  local.add('body', chamferBoxOn('z', 1.68, 0.095, 0.06, 0.018), { pos: [0, H + 0.035, 0] });
  local.add('body', chamferBoxOn('z', 1.68, 0.075, 0.06, 0.018), { pos: [0, -0.012, 0] });
  local.add('glass', new THREE.PlaneGeometry(1.53, H - 0.02), { pos: [0, H / 2 + 0.012, 0.002] });
  for (const [k, g] of local.build()) {
    a.add(k, g, { pos: [0, screenBaseY, screenBaseZ], rot: [-screenRake, 0, 0] });
  }

  // Top corners of the frame, where the cage picks up.
  const topY = screenBaseY + Math.cos(screenRake) * (H + 0.06);
  const topZ = screenBaseZ - Math.sin(screenRake) * (H + 0.06);

  const R = 0.033;

  // Main hoop, just behind the front seats.
  a.add(
    'steel',
    tube(
      [
        [cageX, belt - 0.06, hoopZ],
        [cageX, cageY, hoopZ],
        [-cageX, cageY, hoopZ],
        [-cageX, belt - 0.06, hoopZ],
      ],
      R,
      TUBE,
    ),
  );
  // Rear hoop, over the tailgate.
  a.add(
    'steel',
    tube(
      [
        [cageX, belt - 0.04, rearHoopZ],
        [cageX, cageY - 0.03, rearHoopZ],
        [-cageX, cageY - 0.03, rearHoopZ],
        [-cageX, belt - 0.04, rearHoopZ],
      ],
      R,
      TUBE,
    ),
  );
  // Top rails: screen header -> main hoop -> rear hoop, one continuous bar.
  a.addPair(
    'steel',
    tube(
      [
        [cageX - 0.02, topY - 0.03, topZ + 0.01],
        [cageX, cageY, hoopZ],
        [cageX, cageY - 0.03, rearHoopZ],
      ],
      R,
      { ...TUBE, cornerRadius: 0.3 },
    ),
  );
  // Rear down-braces from the main hoop into the tub corners.
  a.addPair('steel', tube([[cageX, cageY - 0.08, hoopZ], [cageX - 0.02, belt - 0.02, rearHoopZ + 0.1]], R * 0.82, TUBE));
  // Padding on the main hoop uprights.
  a.addPair('dark', rodY(R * 1.55, 0.36, 6), { pos: [cageX, belt - 0.06, hoopZ] });

  // --- roof rack -----------------------------------------------------------
  // Sits a clear 6 cm above the cage so it never reads as a hardtop.
  const rz0 = topZ - 0.08;
  const rz1 = rearHoopZ - 0.04;
  const rx = cageX - 0.01;
  a.addPair('steel', tube([[rx, rackY, rz0], [rx, rackY, rz1]], 0.026, { radialSegments: 5, density: 1.4 }));
  for (const z of [rz0, rz1]) {
    a.add('steel', tube([[rx, rackY, z], [-rx, rackY, z]], 0.026, { radialSegments: 5, density: 2 }));
  }
  const slats = 5;
  for (let i = 1; i < slats; i++) {
    a.add('steel', box(2 * rx - 0.02, 0.016, 0.05), { pos: [0, rackY - 0.016, rz0 + ((rz1 - rz0) * i) / slats] });
  }
  for (const z of [rz0 + 0.08, hoopZ, rz1 - 0.06]) {
    a.addPair('steel', rodY(0.018, rackY - cageY + 0.03, 5), { pos: [rx, cageY - 0.03, z] });
  }
  // Aux light pod facing forward off the rack's front rail.
  a.addPair('dark', rodZ(0.065, 0.11, 10), { pos: [0.44, rackY - 0.08, rz0 + 0.01] });
  a.addPair('dark', box(0.03, 0.09, 0.03), { pos: [0.44, rackY - 0.02, rz0 + 0.01] });
  a.addPair('lampHead', lens(0.06, 0.026, 10), { pos: [0.44, rackY - 0.08, rz0 + 0.064] });

  // --- mirrors on stalks ---------------------------------------------------
  a.addPair('steel', tube([[HW - 0.02, screenBaseY + 0.06, screenBaseZ - 0.01], [HW + 0.11, screenBaseY + 0.2, screenBaseZ + 0.01]], 0.018, {
    radialSegments: 5,
    density: 6,
  }));
  a.addPair('trim', chamferBoxOn('x', 0.035, 0.14, 0.11, 0.014), {
    pos: [HW + 0.125, screenBaseY + 0.235, screenBaseZ + 0.015],
  });
  a.addPair('chrome', new THREE.PlaneGeometry(0.09, 0.12), {
    pos: [HW + 0.107, screenBaseY + 0.235, screenBaseZ + 0.015],
    rot: [0, -Math.PI / 2, 0],
  });

  // --- snorkel -------------------------------------------------------------
  const sx = HW - 0.03;
  a.add(
    'steel',
    tube(
      [
        [sx, D.fenderTop - 0.06, 1.5],
        [sx + 0.045, D.fenderTop + 0.1, 1.28],
        [sx + 0.045, screenBaseY + 0.22, 0.74],
        [sx + 0.035, cageY - 0.04, 0.44],
      ],
      0.038,
      { radialSegments: 6, density: 4, cornerRadius: 0.22 },
    ),
  );
  a.add('steel', rodZ(0.05, 0.18, 8), { pos: [sx + 0.035, cageY + 0.02, 0.55] });
  a.add('dark', new THREE.CircleGeometry(0.042, 8), { pos: [sx + 0.035, cageY + 0.02, 0.643] });

  // --- antenna -------------------------------------------------------------
  a.add('dark', rodY(0.007, 0.66, 4), { pos: [-(HW - 0.05), D.fenderTop, 1.48], rot: [0, 0, 0.05] });
}

function buildFrameAndArmour(a: Assembler): void {
  const { railX, railBottom, railTop, halfWidth: HW, rocker } = D;

  // --- ladder frame --------------------------------------------------------
  const railH = railTop - railBottom;
  const railY = (railBottom + railTop) / 2;
  a.addPair('grime', chamferBoxOn('z', 0.1, railH, 3.76, 0.02), { pos: [railX, railY, -0.04] });
  for (const z of [1.7, 0.68, -0.5, -1.46, -1.76]) {
    a.add('grime', box(2 * railX - 0.1, railH * 0.72, 0.08), { pos: [0, railY, z] });
  }
  a.add('grime', chamferBox(0.8, 0.2, 0.62, 0.03), { pos: [0, railBottom + 0.02, -0.86] });
  a.add('steel', box(0.64, 0.035, 0.52), { pos: [0, railBottom - 0.05, 0.26] });

  // --- rock sliders --------------------------------------------------------
  a.addPair(
    'steel',
    tube(
      [
        [HW - 0.02, rocker - 0.11, 0.8],
        [HW + 0.05, rocker - 0.15, 0.62],
        [HW + 0.05, rocker - 0.15, -0.88],
        [HW - 0.02, rocker - 0.11, -1.04],
      ],
      0.046,
      { radialSegments: 6, density: 3, cornerRadius: 0.14 },
    ),
  );
  a.addPair('steel', box(0.11, 0.018, 1.42), { pos: [HW + 0.035, rocker - 0.11, -0.13] });
  for (const z of [0.5, -0.2, -0.8]) a.addPair('steel', box(0.28, 0.05, 0.06), { pos: [HW - 0.12, rocker - 0.13, z] });

  // --- mud flaps -----------------------------------------------------------
  for (const z of [D.axleZ - D.archR - 0.05, -D.axleZ - D.archR - 0.05]) {
    a.addPair('trim', box(0.014, 0.34, 0.28), { pos: [HW + 0.04, rocker - 0.14, z] });
    a.addPair('dark', box(0.02, 0.04, 0.28), { pos: [HW + 0.04, rocker + 0.02, z] });
  }

  // --- front bull bar + winch ---------------------------------------------
  const bz = 1.95;
  a.add(
    'steel',
    tube(
      [
        [HW + 0.02, 0.78, 1.76],
        [HW + 0.03, 0.72, bz - 0.05],
        [HW - 0.09, 0.68, bz],
        [-(HW - 0.09), 0.68, bz],
        [-(HW + 0.03), 0.72, bz - 0.05],
        [-(HW + 0.02), 0.78, 1.76],
      ],
      0.055,
      { radialSegments: 6, density: 3, cornerRadius: 0.14 },
    ),
  );
  // Pre-runner hoop. Its uprights sit outboard of the headlights and its top bar
  // clears the hood, so the seven slots — the whole point of the front end —
  // stay completely unobstructed.
  a.add(
    'steel',
    tube(
      [
        [0.8, 0.7, bz - 0.02],
        [0.78, 1.17, bz - 0.14],
        [-0.78, 1.17, bz - 0.14],
        [-0.8, 0.7, bz - 0.02],
      ],
      0.038,
      { radialSegments: 5, density: 3, cornerRadius: 0.22 },
    ),
  );
  a.add('dark', box(0.8, 0.055, 0.055), { pos: [0, 1.2, bz - 0.16] });
  a.addPair('lampHead', lens(0.056, 0.028, 8), { pos: [0.26, 1.2, bz - 0.13] });
  a.add('trim', chamferBox(1.66, 0.17, 0.12, 0.03), { pos: [0, 0.6, bz - 0.03] });
  // winch: drum, motor, fairlead, hook
  a.add('steel', rodX(0.075, 0.28, 8), { pos: [0, 0.65, 1.88] });
  a.add('dark', rodX(0.045, 0.5, 6), { pos: [0, 0.65, 1.88] });
  a.add('steel', box(0.2, 0.11, 0.04), { pos: [0, 0.62, 1.95] });
  a.add('chrome', box(0.045, 0.08, 0.03), { pos: [0.26, 0.6, 1.99] });
  a.addPair('lampAmber', box(0.05, 0.11, 0.07), { pos: [0.6, 0.58, bz + 0.02] });

  // --- rear bumper, hitch, exhaust ----------------------------------------
  const rz = -1.88;
  a.add('steel', chamferBoxOn('x', 1.78, 0.16, 0.13, 0.03), { pos: [0, 0.6, rz] });
  a.addPair('steel', box(0.09, 0.14, 0.16), { pos: [0.8, 0.6, rz + 0.1] });
  a.add('steel', box(0.14, 0.12, 0.16), { pos: [0, 0.54, rz - 0.05] });
  a.add('dark', box(0.07, 0.07, 0.1), { pos: [0, 0.54, rz - 0.11] });
  a.addPair('lampAmber', box(0.05, 0.1, 0.06), { pos: [0.62, 0.56, rz - 0.06] });
  a.add('steel', tube([[-0.2, 0.5, 0.3], [-0.3, 0.48, -0.6], [-0.44, 0.56, -1.04], [-0.66, 0.5, -1.16]], 0.032, {
    radialSegments: 5,
    density: 3,
    cornerRadius: 0.16,
  }));
  a.add('chrome', rodX(0.045, 0.12, 8), { pos: [-0.76, 0.5, -1.16] });
}

function buildRear(a: Assembler): void {
  const { tailZ, spareZ, spareY, spareX, halfWidth: HW } = D;

  for (const side of [1, -1]) {
    a.add('trim', chamferBoxOn('z', 0.2, 0.4, 0.05, 0.014), { pos: [side * 0.63, 1.0, tailZ - 0.036] });
    a.add('lampBrake', lens(0.078, 0.032, 10), { pos: [side * 0.63, 1.11, tailZ - 0.062], rot: [0, Math.PI, 0] });
    a.add('lampAmber', lens(0.05, 0.024, 8), { pos: [side * 0.63, 0.97, tailZ - 0.062], rot: [0, Math.PI, 0] });
    a.add('lampReverse', lens(0.05, 0.024, 8), { pos: [side * 0.63, 0.87, tailZ - 0.062], rot: [0, Math.PI, 0] });
  }
  a.add('chrome', box(0.42, 0.16, 0.015), { pos: [-0.22, 0.79, tailZ - 0.045] });

  // --- spare wheel carrier -------------------------------------------------
  a.add('steel', tube([[0.72, 0.76, tailZ - 0.06], [0.72, 1.3, tailZ - 0.06], [spareX + 0.22, 1.3, tailZ - 0.09]], 0.045, {
    radialSegments: 5,
    density: 3,
    cornerRadius: 0.18,
  }));
  a.add('steel', box(0.26, 0.26, 0.05), { pos: [spareX, spareY, spareZ + 0.12] });
  a.add('steel', rodZ(0.05, 0.17, 8), { pos: [spareX, spareY, spareZ + 0.05] });

  // Recessed filler pocket, so the cap is not just a disc stuck on a flat wall.
  a.add('dark', box(0.02, 0.16, 0.16), { pos: [-HW - 0.004, 0.95, -1.14] });
  // fuel filler on the left rear quarter
  a.add('trim', lens(0.055, 0.018, 8), { pos: [-HW - 0.014, 0.95, -1.14], rot: [0, -Math.PI / 2, 0] });
}

function buildInterior(a: Assembler): void {
  const { floorY, halfWidth: HW, belt, cowlZ, screenBaseY } = D;

  // --- dash ----------------------------------------------------------------
  a.add('trim', chamferBoxOn('z', 1.6, 0.26, 0.22, 0.03), { pos: [0, screenBaseY - 0.15, cowlZ - 0.15] });
  a.add('vinyl', chamferBoxOn('y', 1.6, 0.05, 0.28, 0.02), { pos: [0, screenBaseY - 0.025, cowlZ - 0.18] });
  a.add('trim', chamferBoxOn('z', 0.44, 0.23, 0.16, 0.03), { pos: [-0.4, screenBaseY - 0.14, cowlZ - 0.21] });
  a.add('gauge', new THREE.PlaneGeometry(0.36, 0.18), { pos: [-0.4, screenBaseY - 0.14, cowlZ - 0.29], rot: [0.16, Math.PI, 0] });
  a.add('dark', box(0.32, 0.2, 0.05), { pos: [0.06, screenBaseY - 0.17, cowlZ - 0.27] });
  for (const vx of [-0.72, 0.72]) a.add('dark', box(0.16, 0.08, 0.04), { pos: [vx, screenBaseY - 0.1, cowlZ - 0.26] });
  a.add('trim', tube([[0.66, screenBaseY - 0.05, cowlZ - 0.17], [0.44, screenBaseY - 0.05, cowlZ - 0.19]], 0.022, {
    radialSegments: 5,
    density: 6,
  }));

  // --- transmission tunnel + levers ---------------------------------------
  a.add('dark', chamferBox(0.36, 0.2, 1.0, 0.04), { pos: [0, floorY + 0.09, 0.06] });
  a.add('dark', rodY(0.016, 0.25, 5), { pos: [0.05, floorY + 0.18, 0.14], rot: [-0.2, 0, -0.1] });
  a.add('chrome', new THREE.SphereGeometry(0.035, 6, 4), { pos: [0.03, floorY + 0.42, 0.09] });
  a.add('dark', rodY(0.013, 0.2, 5), { pos: [0.15, floorY + 0.18, 0.0], rot: [-0.15, 0, -0.08] });
  a.add('chrome', new THREE.SphereGeometry(0.028, 5, 3), { pos: [0.13, floorY + 0.37, -0.03] });
  a.add('dark', rodY(0.014, 0.26, 5), { pos: [-0.15, floorY + 0.14, 0.0], rot: [-0.9, 0, 0] });

  // --- seats ---------------------------------------------------------------
  const seat = (x: number, z: number, backH: number): void => {
    a.add('vinyl', chamferBoxOn('y', 0.48, 0.12, 0.48, 0.04), { pos: [x, floorY + 0.2, z] });
    a.add('vinyl', chamferBoxOn('z', 0.48, backH, 0.14, 0.04), { pos: [x, floorY + 0.26 + backH / 2, z - 0.25], rot: [0.15, 0, 0] });
    a.add('vinyl', chamferBoxOn('z', 0.26, 0.14, 0.12, 0.035), { pos: [x, floorY + 0.33 + backH, z - 0.3], rot: [0.15, 0, 0] });
    a.add('dark', box(0.36, 0.06, 0.14), { pos: [x, floorY + 0.09, z] });
  };
  seat(-0.42, 0.02, 0.52);
  seat(0.42, 0.02, 0.52);
  a.add('vinyl', chamferBoxOn('y', 1.3, 0.12, 0.44, 0.04), { pos: [0, floorY + 0.18, -0.9] });
  a.add('vinyl', chamferBoxOn('z', 1.3, 0.46, 0.14, 0.04), { pos: [0, floorY + 0.46, -1.13], rot: [0.14, 0, 0] });
  a.add('dark', box(1.44, 0.045, 0.44), { pos: [0, floorY + 0.13, -1.48] });

  // --- inner skins so you never see through the tub ------------------------
  a.addPair('body', chamferBoxOn('x', 0.03, belt - floorY - 0.02, 2.26, 0.01), {
    pos: [HW - 0.06, (belt + floorY) / 2, -0.56],
  });

  // --- seat belts ----------------------------------------------------------
  for (const x of [-0.66, 0.66]) a.add('dark', box(0.05, 0.5, 0.012), { pos: [x, floorY + 0.62, -0.42], rot: [-0.1, 0, 0.05] });
}

// ---------------------------------------------------------------------------
// suspension rig
// ---------------------------------------------------------------------------

interface AimSeg {
  obj: THREE.Object3D;
  /** Length the geometry was authored at; the object scales Y by len/rest. */
  rest: number;
}

interface AxleRig {
  group: THREE.Group;
  knuckles: THREE.Object3D[];
  springs: AimSeg[];
  shockUpper: THREE.Object3D[];
  shockLower: THREE.Object3D[];
  lowerArms: AimSeg[];
  upperArms: AimSeg[];
  trackBar: AimSeg;
  /** Frame-side anchors, body-local. */
  springTop: THREE.Vector3[];
  shockTop: THREE.Vector3[];
  lowerArmFrame: THREE.Vector3[];
  upperArmFrame: THREE.Vector3[];
  trackBarFrame: THREE.Vector3;
  /** Axle-side pickups, axle-local. */
  springSeat: THREE.Vector3[];
  shockBase: THREE.Vector3[];
  lowerArmAxle: THREE.Vector3[];
  upperArmAxle: THREE.Vector3[];
  trackBarAxle: THREE.Vector3;
  steers: boolean;
  z: number;
}

const SPRING_REST = 0.45;
/** Steering column rake from vertical, rad. */
const COLUMN_TILT = 0.42;
const UP = new THREE.Vector3(0, 1, 0);
const AXIS_Z = new THREE.Vector3(0, 0, 1);

function mesh(geo: THREE.BufferGeometry, mat: THREE.Material, name: string): THREE.Mesh {
  const m = new THREE.Mesh(geo, mat);
  m.name = name;
  m.castShadow = true;
  m.receiveShadow = true;
  return m;
}

/** A link with a ball joint at each end, authored 1 m long along +Y. */
function linkGeometry(r: number, ball: number): THREE.BufferGeometry {
  return mergeAll([
    place(rodY(r, 1, 5), {}),
    place(new THREE.SphereGeometry(ball, 5, 3), {}),
    place(new THREE.SphereGeometry(ball, 5, 3), { pos: [0, 1, 0] }),
  ]);
}

function buildAxle(mats: Mats, front: boolean): AxleRig {
  const z = front ? D.axleZ : -D.axleZ;
  const group = new THREE.Group();
  group.name = front ? 'axleFront' : 'axleRear';

  // --- housing -------------------------------------------------------------
  const parts: THREE.BufferGeometry[] = [];
  parts.push(rodX(0.052, 1.3, 8));
  const dx = front ? 0.2 : 0.0;
  parts.push(place(new THREE.SphereGeometry(0.135, 8, 5), { pos: [dx, 0, 0], scale: [1, 1, 0.85] }));
  // Diff cover. Profile runs rim -> apex for the same winding reason `domeZ`
  // documents: written apex-first it lathes inside-out, and the pumpkin renders
  // as a hole in the axle from every angle you actually see it from.
  parts.push(place(lathe([[0.115, -0.05], [0.12, 0.0], [0.09, 0.05], [0.0, 0.06]], 8), { pos: [dx, 0, -0.13], rot: [Math.PI / 2, 0, 0] }));
  for (const s of [-1, 1]) {
    parts.push(place(box(0.16, 0.05, 0.16), { pos: [s * 0.5, 0.075, 0] }));
    parts.push(place(box(0.06, 0.1, 0.07), { pos: [s * 0.62, 0.04, front ? 0.07 : -0.07] }));
    parts.push(place(box(0.1, 0.09, 0.08), { pos: [s * 0.34, -0.06, front ? -0.07 : 0.07] }));
    parts.push(place(box(0.08, 0.07, 0.07), { pos: [s * 0.24, 0.11, front ? -0.06 : 0.06] }));
  }
  group.add(mesh(mergeAll(parts), mats.grime, 'axleHousing'));

  // --- knuckles (steer on the front axle) ---------------------------------
  const knuckles: THREE.Object3D[] = [];
  for (const s of [-1, 1]) {
    const k = new THREE.Group();
    k.position.set(s * 0.62, 0, 0);
    k.add(
      mesh(
        mergeAll([
          place(chamferBox(0.1, 0.2, 0.16, 0.03), { pos: [s * 0.06, 0, 0] }),
          place(rodX(0.05, 0.16, 8), { pos: [s * 0.15, 0, 0] }),
          place(box(0.06, 0.045, 0.2), { pos: [s * 0.03, 0.06, front ? -0.11 : 0.11] }),
        ]),
        mats.grime,
        'knuckle',
      ),
    );
    group.add(k);
    knuckles.push(k);
  }

  // --- coil springs --------------------------------------------------------
  const springGeo = coilSpring(SPRING_REST, 0.08, 0.017, 4.5, 5, 4);
  const springs: AimSeg[] = [];
  for (let i = 0; i < 2; i++) springs.push({ obj: mesh(springGeo.clone(), mats.steel, 'coil'), rest: SPRING_REST });

  // --- shocks --------------------------------------------------------------
  const shockUpper: THREE.Object3D[] = [];
  const shockLower: THREE.Object3D[] = [];
  const bodyGeo = mergeAll([
    place(rodY(0.034, 0.3, 6), { pos: [0, -0.3, 0] }),
    place(rodY(0.046, 0.05, 6), { pos: [0, -0.05, 0] }),
    place(new THREE.SphereGeometry(0.03, 5, 3), {}),
  ]);
  const rodGeo = mergeAll([
    place(rodY(0.014, 0.34, 5), {}),
    place(new THREE.SphereGeometry(0.03, 5, 3), {}),
    place(rodY(0.05, 0.05, 6), { pos: [0, 0.06, 0] }),
  ]);
  for (let i = 0; i < 2; i++) {
    shockUpper.push(mesh(bodyGeo.clone(), mats.dark, 'shockBody'));
    shockLower.push(mesh(rodGeo.clone(), mats.chrome, 'shockRod'));
  }

  // --- links ---------------------------------------------------------------
  const thick = linkGeometry(0.028, 0.04);
  const thin = linkGeometry(0.021, 0.032);
  const mk = (g: THREE.BufferGeometry, n: string): AimSeg => ({ obj: mesh(g.clone(), mats.grime, n), rest: 1 });
  const lowerArms = [mk(thick, 'lowerArm'), mk(thick, 'lowerArm')];
  const upperArms = [mk(thin, 'upperArm'), mk(thin, 'upperArm')];
  const trackBar = mk(thin, 'trackBar');

  const dir = front ? 1 : -1;
  return {
    group,
    knuckles,
    springs,
    shockUpper,
    shockLower,
    lowerArms,
    upperArms,
    trackBar,
    springTop: [
      new THREE.Vector3(-0.5, D.hubY + 0.075 + SPRING_REST, z),
      new THREE.Vector3(0.5, D.hubY + 0.075 + SPRING_REST, z),
    ],
    shockTop: [new THREE.Vector3(-0.6, 1.02, z - dir * 0.2), new THREE.Vector3(0.6, 1.02, z - dir * 0.2)],
    lowerArmFrame: [new THREE.Vector3(-0.38, 0.52, z - dir * 0.62), new THREE.Vector3(0.38, 0.52, z - dir * 0.62)],
    upperArmFrame: [new THREE.Vector3(-0.24, 0.72, z - dir * 0.56), new THREE.Vector3(0.24, 0.72, z - dir * 0.56)],
    trackBarFrame: new THREE.Vector3(-dir * 0.44, 0.62, z - dir * 0.04),
    springSeat: [new THREE.Vector3(-0.5, 0.075, 0), new THREE.Vector3(0.5, 0.075, 0)],
    shockBase: [new THREE.Vector3(-0.62, 0.02, dir * 0.07), new THREE.Vector3(0.62, 0.02, dir * 0.07)],
    lowerArmAxle: [new THREE.Vector3(-0.34, -0.07, -dir * 0.07), new THREE.Vector3(0.34, -0.07, -dir * 0.07)],
    upperArmAxle: [new THREE.Vector3(-0.24, 0.13, -dir * 0.06), new THREE.Vector3(0.24, 0.13, -dir * 0.06)],
    trackBarAxle: new THREE.Vector3(dir * 0.54, 0.04, -dir * 0.02),
    steers: front,
    z,
  };
}

// ---------------------------------------------------------------------------
// JeepModel
// ---------------------------------------------------------------------------

export interface JeepModelOptions {
  color?: THREE.ColorRepresentation;
  /** Headlights lit. Default true — it reads well under a dusty rally sky. */
  headlights?: boolean;
}

export class JeepModel {
  readonly object3d: THREE.Object3D;
  /** Rendered triangles in the whole model. */
  readonly triangles: number;

  private readonly body = new THREE.Group();
  private readonly mats: Mats;
  private readonly wheels: THREE.Group[] = [];
  private readonly wheelSpin: THREE.Group[] = [];
  private readonly axles: AxleRig[];
  private readonly driveshafts: AimSeg[] = [];
  private readonly tieRods: AimSeg[] = [];
  private readonly dragLink: AimSeg;
  private readonly steeringWheel = new THREE.Group();
  private readonly extraGeometries: THREE.BufferGeometry[] = [];

  private time = 0;
  private brakeGlow = 0;
  private headlightsOn: boolean;

  // scratch
  private readonly _q = new THREE.Quaternion();
  private readonly _qi = new THREE.Quaternion();
  private readonly _v = new THREE.Vector3();
  private readonly _a = new THREE.Vector3();
  private readonly _b = new THREE.Vector3();
  private readonly _c = new THREE.Vector3();
  private readonly _hub: THREE.Vector3[] = [
    new THREE.Vector3(),
    new THREE.Vector3(),
    new THREE.Vector3(),
    new THREE.Vector3(),
  ];

  constructor(opts: JeepModelOptions = {}) {
    const mats = (this.mats = buildMaterials(opts.color ?? '#7f3324'));
    this.headlightsOn = opts.headlights ?? true;

    this.object3d = new THREE.Group();
    this.object3d.name = 'jeep';
    this.object3d.add(this.body);
    this.body.name = 'jeepBody';

    // --- static bodywork ---------------------------------------------------
    const a = new Assembler();
    buildBodyShell(a);
    buildGrille(a);
    buildFlares(a);
    buildTopStructure(a);
    buildFrameAndArmour(a);
    buildRear(a);
    buildInterior(a);
    a.emit(this.body, mats as unknown as Record<string, THREE.Material>, '');

    this.buildSteeringWheel();

    // --- wheels ------------------------------------------------------------
    const wg = buildWheel(TYRE);
    const rubber = mergeAll([wg.carcass, wg.tread]);
    this.extraGeometries.push(rubber, wg.rim, wg.hub, wg.brake);
    for (let i = 0; i < 4; i++) {
      const hub = new THREE.Group();
      hub.name = `wheel${i}`;
      const spin = new THREE.Group();
      spin.name = `wheelSpin${i}`;
      spin.add(mesh(rubber, mats.rubber, 'tyre'));
      spin.add(mesh(wg.rim, mats.rim, 'rim'));
      spin.add(mesh(wg.hub, mats.chrome, 'hub'));
      spin.add(mesh(wg.brake, mats.grime, 'brake'));
      hub.add(spin);
      this.object3d.add(hub);
      this.wheels.push(hub);
      this.wheelSpin.push(spin);
    }

    // --- spare on the tailgate ---------------------------------------------
    const spare = new THREE.Group();
    spare.position.set(D.spareX, D.spareY, D.spareZ);
    spare.rotation.set(0, Math.PI / 2, 0.05);
    spare.add(mesh(rubber, mats.rubber, 'spareTyre'));
    spare.add(mesh(wg.rim, mats.rim, 'spareRim'));
    spare.add(mesh(wg.hub, mats.chrome, 'spareHub'));
    this.body.add(spare);

    // --- axles + linkage ---------------------------------------------------
    this.axles = [buildAxle(mats, true), buildAxle(mats, false)];
    const tieGeo = linkGeometry(0.022, 0.032);
    this.extraGeometries.push(tieGeo);
    for (const ax of this.axles) {
      this.body.add(ax.group);
      for (const s of ax.springs) this.body.add(s.obj);
      for (const o of ax.shockUpper) this.body.add(o);
      for (const o of ax.shockLower) this.body.add(o);
      for (const s of ax.lowerArms) this.body.add(s.obj);
      for (const s of ax.upperArms) this.body.add(s.obj);
      this.body.add(ax.trackBar.obj);
      // The tie rod lives in axle space: both knuckle arms swing together.
      const tr: AimSeg = { obj: mesh(tieGeo.clone(), mats.grime, 'tieRod'), rest: 1 };
      ax.group.add(tr.obj);
      this.tieRods.push(tr);
    }

    // Transfer case + driveshafts.
    const tcase = mesh(chamferBox(0.3, 0.26, 0.42, 0.05), mats.grime, 'transferCase');
    tcase.position.set(-0.06, 0.5, -0.18);
    this.body.add(tcase);
    const shaftGeo = linkGeometry(0.036, 0.05);
    this.extraGeometries.push(shaftGeo);
    for (let i = 0; i < 2; i++) {
      const seg: AimSeg = { obj: mesh(shaftGeo.clone(), mats.steel, 'driveshaft'), rest: 1 };
      this.body.add(seg.obj);
      this.driveshafts.push(seg);
    }

    this.dragLink = { obj: mesh(tieGeo.clone(), mats.grime, 'dragLink'), rest: 1 };
    this.body.add(this.dragLink.obj);

    // Prime the rig at the design pose so the first frame is never a mess.
    this.settle();
    this.triangles = countTriangles(this.object3d);
  }

  // -------------------------------------------------------------------------

  /**
   * Rim, spokes and hub, built in the XY plane so the whole thing spins about
   * its own local Z. The group is tilted back on X to the column angle, which
   * makes `rotation.z` a true rack rotation rather than a shear.
   */
  private buildSteeringWheel(): void {
    const parts: THREE.BufferGeometry[] = [new THREE.TorusGeometry(0.17, 0.019, 5, 13)];
    for (let i = 0; i < 3; i++) {
      const ang = -Math.PI / 2 + (i / 3) * Math.PI * 2;
      const sp = box(0.035, 0.16, 0.014);
      sp.translate(0, 0.085, 0);
      sp.rotateZ(ang + Math.PI / 2);
      parts.push(sp);
    }
    parts.push(place(new THREE.CylinderGeometry(0.055, 0.05, 0.05, 8), { rot: [Math.PI / 2, 0, 0] }));
    this.steeringWheel.add(mesh(mergeAll(parts), this.mats.dark, 'steeringWheel'));
    this.steeringWheel.position.set(-0.4, D.screenBaseY - 0.08, D.cowlZ - 0.36);
    this.steeringWheel.rotation.set(-COLUMN_TILT, 0, 0);
    this.body.add(this.steeringWheel);

    const col = mesh(rodY(0.028, 0.36, 6), this.mats.dark, 'column');
    col.position.copy(this.steeringWheel.position);
    col.rotation.x = Math.PI / 2 - COLUMN_TILT;
    this.body.add(col);
    for (const s of [-1, 1]) {
      const st = mesh(rodY(0.011, 0.13, 4), this.mats.dark, 'stalk');
      st.position.set(-0.4 + s * 0.05, D.screenBaseY - 0.11, D.cowlZ - 0.3);
      st.rotation.z = s * (Math.PI / 2 - 0.25);
      this.body.add(st);
    }
  }

  /** Places everything as if the car were sitting level at design ride height. */
  private settle(): void {
    for (let i = 0; i < 4; i++) {
      this._hub[i]!.set(i % 2 === 0 ? -D.halfTrack : D.halfTrack, D.hubY, i < 2 ? D.axleZ : -D.axleZ);
      this.wheels[i]!.position.copy(this._hub[i]!);
      this.wheels[i]!.quaternion.identity();
    }
    this.rigSuspension(0);
  }

  // -------------------------------------------------------------------------

  update(state: VehicleState, dt: number): void {
    this.time += dt;

    this.object3d.position.copy(state.position);
    this.object3d.quaternion.copy(state.quaternion);
    this._qi.copy(state.quaternion).invert();

    // --- wheels ------------------------------------------------------------
    // Physics publishes world-space hub poses with suspension travel already
    // applied, so convert those into the chassis frame rather than re-deriving.
    for (let i = 0; i < 4 && i < state.wheels.length; i++) {
      const w = state.wheels[i]!;
      const hub = this.wheels[i]!;
      this._hub[i]!.copy(w.position).sub(state.position).applyQuaternion(this._qi);
      hub.position.copy(this._hub[i]!);
      // Steer on the hub, spin on the child: the two can never fight.
      hub.rotation.set(0, w.steerAngle, 0);
      this.wheelSpin[i]!.rotation.x += w.spin * dt;
    }

    const steer = ((state.wheels[0]?.steerAngle ?? 0) + (state.wheels[1]?.steerAngle ?? 0)) * 0.5;

    // --- idle shake --------------------------------------------------------
    // Only at a standstill, only from the engine, and small: a body that jitters
    // while moving reads as a bug, not as a big lazy V6 lumping at idle.
    const idle = Math.max(0, 1 - state.speed / 1.8) * (state.engineRpm > 200 ? 1 : 0);
    if (idle > 0.001) {
      const f = (state.engineRpm / 60) * Math.PI;
      const s = Math.sin(this.time * f);
      const c = Math.cos(this.time * f * 0.73);
      this.body.position.set(0.0016 * idle * c, 0.0022 * idle * s, 0);
      this.body.rotation.set(0.0012 * idle * s, 0, 0.0022 * idle * c);
    } else {
      this.body.position.set(0, 0, 0);
      this.body.rotation.set(0, 0, 0);
    }

    this.rigSuspension(steer);

    // ~3.6 turns of rack for full lock: about right for a recirculating-ball
    // box on a solid front axle.
    this.steeringWheel.rotation.set(-COLUMN_TILT, 0, -steer * 3.6);

    // --- lights ------------------------------------------------------------
    // VehicleState carries no pedal input, so braking is read off the chassis
    // accelerometer: retardation opposing the direction of travel.
    const decel = state.forwardSpeed >= 0 ? -state.localAccel.z : state.localAccel.z;
    const braking = state.speed > 0.4 && decel > 0.12;
    this.brakeGlow += ((braking ? 1 : 0) - this.brakeGlow) * Math.min(1, dt * 18);
    (this.mats.lampBrake as THREE.MeshPhongMaterial).emissive.setRGB(
      0.06 + 0.86 * this.brakeGlow,
      0.01 + 0.05 * this.brakeGlow,
      0.008,
    );
    (this.mats.lampReverse as THREE.MeshPhongMaterial).emissive.setScalar(state.gear < 0 ? 0.84 : 0.04);
    (this.mats.lampHead as THREE.MeshPhongMaterial).emissive.setScalar(this.headlightsOn ? 0.7 : 0);
  }

  setHeadlights(on: boolean): void {
    this.headlightsOn = on;
  }

  // -------------------------------------------------------------------------

  /** Fits both axles and every link to the current wheel positions. */
  private rigSuspension(steer: number): void {
    for (let ai = 0; ai < 2; ai++) {
      const ax = this.axles[ai]!;
      const L = this._hub[ai * 2]!;
      const R = this._hub[ai * 2 + 1]!;

      // Axle position = midpoint of the hubs, roll = the line through them.
      // Nothing else is needed: relative to its wheel pair a solid axle has
      // exactly this one degree of freedom.
      ax.group.position.set((L.x + R.x) * 0.5, (L.y + R.y) * 0.5, (L.z + R.z) * 0.5);
      ax.group.rotation.set(0, 0, Math.atan2(R.y - L.y, R.x - L.x));

      if (ax.steers) for (const k of ax.knuckles) k.rotation.y = steer;

      for (let s = 0; s < 2; s++) {
        this.aim(ax.springs[s]!, ax.springTop[s]!, this.axleToBody(ax, ax.springSeat[s]!, this._a), true);
        this.aimPair(ax.shockUpper[s]!, ax.shockLower[s]!, ax.shockTop[s]!, this.axleToBody(ax, ax.shockBase[s]!, this._a));
        this.aim(ax.lowerArms[s]!, ax.lowerArmFrame[s]!, this.axleToBody(ax, ax.lowerArmAxle[s]!, this._a), true);
        this.aim(ax.upperArms[s]!, ax.upperArmFrame[s]!, this.axleToBody(ax, ax.upperArmAxle[s]!, this._a), true);
      }
      this.aim(ax.trackBar, ax.trackBarFrame, this.axleToBody(ax, ax.trackBarAxle, this._a), true);

      // Tie rod, in axle space, following both steering arms.
      const dir = ax.steers ? 1 : -1;
      const armY = 0.06;
      const armZ = -dir * 0.2;
      const th = ax.steers ? steer : 0;
      this._a.set(0, armY, armZ).applyAxisAngle(UP, th).add(this._c.set(-0.62, 0, 0));
      this._b.set(0, armY, armZ).applyAxisAngle(UP, th).add(this._c.set(0.62, 0, 0));
      this.aim(this.tieRods[ai]!, this._a, this._b, true);

      if (ax.steers) {
        // Drag link: frame-mounted steering box down to the left steering arm.
        this.axleToBody(ax, this._a, this._b);
        this.aim(this.dragLink, this._c.set(-0.42, 0.66, D.axleZ - 0.52), this._b, true);
      }

      // Driveshaft: transfer case to this axle's pinion flange.
      this.axleToBody(ax, this._c.set(ax.steers ? 0.2 : 0, 0.06, dir * 0.17), this._a);
      this.aim(
        this.driveshafts[ai]!,
        this._b.set(ax.steers ? 0.06 : -0.06, 0.5, ax.steers ? 0.02 : -0.37),
        this._a,
        true,
      );
    }
  }

  /** axle-local -> body-local. The axle only ever rolls about Z. */
  private axleToBody(ax: AxleRig, p: THREE.Vector3, out: THREE.Vector3): THREE.Vector3 {
    return out.copy(p).applyAxisAngle(AXIS_Z, ax.group.rotation.z).add(ax.group.position);
  }

  /** Points `seg`'s +Y axis from `from` to `to`, stretching if elastic. */
  private aim(seg: AimSeg, from: THREE.Vector3, to: THREE.Vector3, stretch: boolean): void {
    seg.obj.position.copy(from);
    this._v.subVectors(to, from);
    const len = this._v.length();
    if (len > 1e-5) {
      this._v.divideScalar(len);
      seg.obj.quaternion.copy(this._q.setFromUnitVectors(UP, this._v));
    }
    if (stretch) seg.obj.scale.set(1, len / seg.rest, 1);
  }

  /** A telescoping shock: body hangs from the top, rod stands on the axle. */
  private aimPair(upper: THREE.Object3D, lower: THREE.Object3D, top: THREE.Vector3, base: THREE.Vector3): void {
    upper.position.copy(top);
    lower.position.copy(base);
    this._v.subVectors(base, top);
    const len = this._v.length();
    if (len < 1e-5) return;
    this._v.divideScalar(len);
    upper.quaternion.copy(this._q.setFromUnitVectors(UP, this._v));
    lower.quaternion.copy(this._q.setFromUnitVectors(UP, this._v.negate()));
  }

  // -------------------------------------------------------------------------

  dispose(): void {
    this.object3d.traverse((o) => {
      const m = o as THREE.Mesh;
      if (m.isMesh && m.geometry) m.geometry.dispose();
    });
    for (const g of this.extraGeometries) g.dispose();
    // Textures are library-owned and shared; only the materials are ours.
    for (const m of Object.values(this.mats)) m.dispose();
    this.object3d.clear();
  }
}

export default JeepModel;
