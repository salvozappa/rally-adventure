/**
 * Terrain surface shader.
 *
 * `MeshLambertMaterial` + `onBeforeCompile`, deliberately: Lambert is the right
 * lighting model for the retro target (flat, saturated, readable at 320-ish
 * internal resolution) and piggy-backing on a built-in material means shadows,
 * lights and the rest of three's plumbing keep working for free. Going PBR
 * would cost fill rate and make everything look like plastic.
 *
 * What the shader does, in order:
 *
 *   1. Reads the baked RGBA splat control map: R=grass G=rock B=gravel A=sand,
 *      dirt is the remainder. Exactly the same decode `sampleSurface` uses on
 *      the CPU, so tyre grip and pixels never disagree.
 *   2. **Weight contrast.** The generator's weights are a real mixture — a
 *      grassy hillside is grass 0.5 / dirt 0.4 — and drawing that literally
 *      gives a 50:50 mud that reads as neither. A monotonic `pow` sharpens the
 *      split without ever changing which layer is dominant, so the CPU's
 *      `sampleSurface` and the picture cannot disagree.
 *   3. **Height-map blending** rather than a linear cross-fade. Each layer
 *      carries a greyscale height, and the key is `weight * (1 + height)` —
 *      *multiplicative*, so a layer the map says is absent can never punch
 *      through, while two layers that are close interlock texel by texel.
 *      Gravel pokes through dirt grain-by-grain instead of the two averaging.
 *   4. **Palette retargeting.** Each layer is scaled so its measured mean
 *      linear colour lands on the Screamer 4x4 reference value from
 *      docs/ART_DIRECTION.md, with its internal contrast untouched.
 *   5. **Triplanar cliff projection**, faded in by slope. A heightfield mapped
 *      top-down stretches into vertical smears wherever the ground goes past
 *      ~45 degrees; projecting `rockFace`/`cliffStrata` onto the XY and ZY
 *      planes instead is what stops the boundary range looking melted.
 *   6. **Detail texture** at ~0.38 m repeat, multiplied in and faded out with
 *      distance. Kills the blurry-giant-texture look under the bumper without
 *      adding shimmer on the horizon.
 *   7. **Macro variation** — a single very-low-frequency noise stretched across
 *      the entire map, plus a mid-frequency octave. Without this, any tiling
 *      rate you pick eventually reads as a grid.
 *   8. Normal mapping, blended with the same weights as albedo, applied as a
 *      world-space gradient offset so it works identically for the planar and
 *      the triplanar projections.
 *   9. Optional exponential-squared fog, mixed in **linear space before the
 *      output colour-space conversion**. Off in the game — `FogPass` owns
 *      aerial perspective there — and on for the preview pages.
 *
 * Texture budget: the seven surfaces live in two `DataArrayTexture`s (albedo,
 * and normal.rgb + height.a packed together) rather than fourteen samplers.
 * WebGL2 only guarantees sixteen fragment texture units, and the naive version
 * blows straight through that on real hardware.
 */

import * as THREE from 'three';
import { getTexture, type TextureName } from '../render/textures';
import { clamp, hash2i, lerp } from './noise';

/* -------------------------------------------------------------------------- */
/* Layer table                                                                */
/* -------------------------------------------------------------------------- */

/**
 * Array-texture layer order. 0..4 must match `SURFACE_LAYERS` in heightfield.ts
 * because the control map is decoded against it; 5..6 are the cliff materials.
 */
export const MATERIAL_LAYERS = [
  'dirt',
  'grass',
  'rock',
  'gravel',
  'sand',
  'rockFace',
  'cliffStrata',
] as const satisfies readonly TextureName[];

/** Layers driven by the splat control map. */
export const SPLAT_COUNT = 5;
const LAYER_ROCKFACE = 5;
const LAYER_STRATA = 6;
/** Which layer supplies the high-frequency detail multiply. Gravel has grain. */
const LAYER_DETAIL = 3;

/**
 * Metres per texture tile, per layer.
 *
 * These are sized for a camera 1.5 m off the deck, not for a map view. A 256px
 * tile at 2 m is 128 texels per metre, so a wind ripple drawn eight texels wide
 * in the source is a 6 cm ripple on the ground — which is what a wind ripple
 * is. The previous 7-11 m tiles made every one of them a metre across and
 * turned the whole foreground into corduroy.
 *
 * Deliberately mutually non-harmonic: with 2/1/4 you get visible beat
 * frequencies where two layers meet.
 */
const LAYER_METRES: readonly number[] = [2.3, 1.9, 3.7, 1.55, 1.15];
const CLIFF_METRES = 3.4;

/**
 * Per-layer UV rotation, radians.
 *
 * Several of these textures are directional — the sand is nothing but parallel
 * wind ripples — and a directional pattern mapped straight onto world XZ lines
 * itself up with the screen and with every other layer at once. Rotating each
 * layer into its own frame, by angles that are not multiples of a right angle
 * and not equal to each other, breaks that up for two multiply-adds.
 */
const LAYER_RADIANS: readonly number[] = [0.0, 0.62, 1.27, 2.09, 0.38];

/** Metres over which the domain warp completes a cycle, and its amplitude. */
const WARP_METRES = 17.0;
const WARP_AMPLITUDE = 7.0;
/**
 * Second, much higher frequency multiply. Roughly a sixth of the coarsest layer
 * tile, so it never beats against any of them, and fine enough that the ground
 * directly under the bumper still has grain in it.
 */
const DETAIL_METRES = 0.38;

/**
 * Target mean colour per layer, sampled from the Screamer 4x4 reference set and
 * recorded in docs/ART_DIRECTION.md.
 *
 * The procedural textures in `render/textures.ts` are shared with props, rocks
 * and vegetation, so they cannot be re-authored for the terrain's benefit. What
 * the terrain *can* do is scale each layer so its mean lands on the reference
 * colour while its internal contrast — the grain, the tufts, the pebbles — is
 * left completely intact. That is what turns a vivid green-and-brown grass tile
 * into the muted olive of the reference without flattening it.
 */
const LAYER_TARGET: readonly number[] = [
  0x7a6242, // dirt        — packed dirt track
  0x6b6b32, // grass       — olive mid, the dominant landscape colour
  0x7d7668, // rock        — warm grey
  0x857a63, // gravel      — broken ground, between dirt and rock
  0xab9a72, // sand        — dusty light, the wash
  0x7d7668, // rockFace    — warm grey
  0x8a8175, // cliffStrata — bedded rock, a shade paler
];

/** How far to pull each layer onto its target. 1 = all the way. */
const TINT_AMOUNT = 0.85;
/** Clamp on the per-channel correction, so a near-black channel cannot explode. */
const TINT_RANGE: readonly [number, number] = [0.35, 2.4];

/* -------------------------------------------------------------------------- */
/* Options                                                                    */
/* -------------------------------------------------------------------------- */

export interface TerrainMaterialOptions {
  /** RGBA8 splat map from `generateTerrain`. */
  control: Uint8Array;
  controlSize: number;
  /** Side length of the world in metres — used to map world XZ onto the map. */
  worldSize: number;
  /** Seed for the macro-variation noise, so a re-rolled world re-rolls tinting. */
  seed: number;
  /** Metres over which the macro tint completes one cycle. */
  macroMetres?: number;
}

/* -------------------------------------------------------------------------- */
/* Material                                                                   */
/* -------------------------------------------------------------------------- */

export class TerrainMaterial extends THREE.MeshLambertMaterial {
  /** Live uniform objects. Mutate `.value` to retune at runtime. */
  readonly uniforms: Record<string, THREE.IUniform>;

  private readonly ownedTextures: THREE.Texture[] = [];

  constructor(opts: TerrainMaterialOptions) {
    super({
      color: 0xffffff,
      vertexColors: true,
      // Fog is applied by hand below, in linear space, so it can be driven
      // independently of `scene.fog` and matched to the sky exactly.
      fog: false,
    });

    const sets = MATERIAL_LAYERS.map((name) => getTexture(name));
    const texSize = sets[0]!.size;

    const albedoLayers = sets.map((s) => imageData(s.map));
    const albedo = buildArrayTexture(albedoLayers, texSize, THREE.SRGBColorSpace);

    // Palette retargeting and the detail normaliser both need the layers'
    // average linear colour, so measure it once here rather than guessing.
    const means = albedoLayers.map(meanLinearColor);
    const tints = means.map((mean, i) => paletteTint(mean, LAYER_TARGET[i]!));
    const detailMid = luminance(means[LAYER_DETAIL]!);

    const normalHeight = buildArrayTexture(
      sets.map((s) => packNormalHeight(s.normalMap, s.heightMap, texSize)),
      texSize,
      THREE.NoColorSpace,
    );

    const control = new THREE.DataTexture(
      opts.control,
      opts.controlSize,
      opts.controlSize,
      THREE.RGBAFormat,
      THREE.UnsignedByteType,
    );
    control.colorSpace = THREE.NoColorSpace;
    control.wrapS = THREE.ClampToEdgeWrapping;
    control.wrapT = THREE.ClampToEdgeWrapping;
    // Linear, no mips: the map is already at roughly one texel per metre, and
    // mipping it would bleed distant layers into each other well before the
    // albedo mips do — you would see the splat pattern dissolve at range.
    control.magFilter = THREE.LinearFilter;
    control.minFilter = THREE.LinearFilter;
    control.generateMipmaps = false;
    control.needsUpdate = true;

    const macro = buildMacroTexture(opts.seed);

    this.ownedTextures.push(albedo, normalHeight, control, macro);

    const macroMetres = opts.macroMetres ?? opts.worldSize * 0.72;

    this.uniforms = {
      tAlbedo: { value: albedo },
      tNormalHeight: { value: normalHeight },
      tControl: { value: control },
      tMacro: { value: macro },

      uWorldSize: { value: opts.worldSize },
      uLayerScale: { value: LAYER_METRES.map((m) => 1 / m) },
      uLayerRot: {
        value: LAYER_RADIANS.map((a) => new THREE.Vector2(Math.cos(a), Math.sin(a))),
      },
      uCliffScale: { value: 1 / CLIFF_METRES },
      uLayerTint: { value: tints },

      uWarpScale: { value: 1 / WARP_METRES },
      uWarpStrength: { value: WARP_AMPLITUDE },

      uWeightContrast: { value: 2.4 },
      uHeightInfluence: { value: 1.05 },
      uBlendSharpness: { value: 0.26 },

      uSlopeRange: { value: new THREE.Vector2(0.34, 0.62) },
      uNormalStrength: { value: 0.85 },

      uDetailScale: { value: 1 / DETAIL_METRES },
      uDetailMid: { value: detailMid },
      uDetailStrength: { value: 0.5 },
      uDetailFade: { value: new THREE.Vector2(14, 65) },

      uMacroScale: { value: 1 / macroMetres },
      uMacroStrength: { value: 0.9 },

      // Below 1. The reference is muted upland pasture, not a saturated 1997
      // rally stage, and the layer tints above already carry the hue.
      uSaturation: { value: 0.88 },

      // The baked occlusion field runs 0.15..1.0 with a mean of 0.71, which at
      // full strength is a 60% multiply in the hollows — that is an ambient
      // occlusion term doing the job of a shadow map, and it is most of why
      // the ground was reading dark instead of the reference's bright khaki.
      // At 0.45 the floor lifts to 0.62 and the shape still reads.
      uAoStrength: { value: 0.45 },

      // Off by default: in the game the post-process `FogPass` owns aerial
      // perspective, and it does the job properly — it resolves to the literal
      // sky function per pixel direction, and it fogs the trees, rocks and car
      // with exactly the same curve as the ground. Running a second fog here
      // would double-count it and pull everything toward a flat wash.
      // `setFog` exists for the preview pages, which have no pipeline.
      uFogColor: { value: new THREE.Color(0xc5d5e2) },
      uFogDensity: { value: 0 },
    };
  }

  /**
   * Drive the in-material fog from outside — the sky owns the horizon colour,
   * and the two have to be identical or you get a hard line across the world.
   *
   * Only the standalone preview pages need this. Anything rendering through
   * `RetroPipeline` gets its aerial perspective from `FogPass`, which resolves
   * to the same sky function for every object rather than just the ground;
   * leave the density at its default 0 there or the two stack.
   */
  setFog(color: THREE.Color, density: number): void {
    (this.uniforms.uFogColor!.value as THREE.Color).copy(color);
    this.uniforms.uFogDensity!.value = density;
  }

  /** Strength of the baked vertex AO, 0..1. */
  setAmbientOcclusion(strength: number): void {
    this.uniforms.uAoStrength!.value = clamp(strength, 0, 1);
  }

  override onBeforeCompile(shader: THREE.WebGLProgramParametersWithUniforms): void {
    Object.assign(shader.uniforms, this.uniforms);
    shader.vertexShader = patchVertex(shader.vertexShader);
    shader.fragmentShader = patchFragment(shader.fragmentShader);
  }

  /** All instances compile the same program; keep them sharing one. */
  override customProgramCacheKey(): string {
    return 'rally-terrain-v2';
  }

  override dispose(): void {
    super.dispose();
    for (const t of this.ownedTextures) t.dispose();
    this.ownedTextures.length = 0;
  }
}

/* -------------------------------------------------------------------------- */
/* Shader injection                                                           */
/* -------------------------------------------------------------------------- */

const VARYINGS = /* glsl */ `
varying vec3 vTerrainWorldPos;
varying vec3 vTerrainWorldNrm;
varying float vTerrainViewDist;
`;

function patchVertex(src: string): string {
  let out = src.replace('#include <common>', `#include <common>\n${VARYINGS}`);

  // objectNormal exists from <beginnormal_vertex>; transformed from <begin_vertex>.
  out = out.replace(
    '#include <begin_vertex>',
    /* glsl */ `#include <begin_vertex>
	vec4 terrainWorld = modelMatrix * vec4( transformed, 1.0 );
	vTerrainWorldPos = terrainWorld.xyz;
	vTerrainWorldNrm = normalize( mat3( modelMatrix ) * objectNormal );`,
  );

  // mvPosition is declared inside <project_vertex> and stays in scope.
  out = out.replace(
    '#include <project_vertex>',
    /* glsl */ `#include <project_vertex>
	vTerrainViewDist = length( mvPosition.xyz );`,
  );

  return out;
}

const FRAG_UNIFORMS = /* glsl */ `
uniform sampler2DArray tAlbedo;
uniform sampler2DArray tNormalHeight;
uniform sampler2D tControl;
uniform sampler2D tMacro;

uniform float uWorldSize;
uniform float uLayerScale[ ${SPLAT_COUNT} ];
uniform vec2  uLayerRot[ ${SPLAT_COUNT} ];
uniform float uCliffScale;
uniform float uWarpScale;
uniform float uWarpStrength;
uniform vec3  uLayerTint[ ${MATERIAL_LAYERS.length} ];
uniform float uWeightContrast;
uniform float uHeightInfluence;
uniform float uBlendSharpness;
uniform vec2  uSlopeRange;
uniform float uNormalStrength;
uniform float uDetailScale;
uniform float uDetailMid;
uniform float uDetailStrength;
uniform vec2  uDetailFade;
uniform float uMacroScale;
uniform float uMacroStrength;
uniform float uSaturation;
uniform float uAoStrength;
uniform vec3  uFogColor;
uniform float uFogDensity;

const vec3 TERRAIN_LUMA = vec3( 0.2126, 0.7152, 0.0722 );
`;

/**
 * The whole surface evaluation. Replaces `<map_fragment>`, which is empty for
 * this material (no `map` is ever assigned), and leaves `terrainNormalW` in
 * scope for the `<normal_fragment_maps>` replacement further down `main`.
 */
const FRAG_SURFACE = /* glsl */ `
	vec3 wpos = vTerrainWorldPos;
	vec3 wnrm = normalize( vTerrainWorldNrm );

	// -- splat weights ---------------------------------------------------
	vec2 ctlUv = ( wpos.xz + uWorldSize * 0.5 ) / uWorldSize;
	vec4 ctl = texture2D( tControl, ctlUv );
	float lw[ ${SPLAT_COUNT} ];
	lw[ 0 ] = max( 0.0, 1.0 - ( ctl.r + ctl.g + ctl.b + ctl.a ) );
	lw[ 1 ] = ctl.r;
	lw[ 2 ] = ctl.g;
	lw[ 3 ] = ctl.b;
	lw[ 4 ] = ctl.a;

	// Contrast the control weights before anything else looks at them. The
	// generator hands back a genuine mixture — a grassy hillside is typically
	// grass 0.5 / dirt 0.4 — and rendering that literally gives a 50:50 mud
	// that reads as neither. pow() is monotonic, so the *dominant* layer never
	// changes and sampleSurface() on the CPU still agrees with the picture
	// pixel for pixel; what changes is that the hillside now reads as grass
	// with dirt showing through it, which is what the numbers actually mean.
	float wsum0 = 0.0;
	for ( int i = 0; i < ${SPLAT_COUNT}; i ++ ) {
		lw[ i ] = pow( lw[ i ], uWeightContrast );
		wsum0 += lw[ i ];
	}
	float invW0 = 1.0 / max( wsum0, 1.0e-5 );
	for ( int i = 0; i < ${SPLAT_COUNT}; i ++ ) lw[ i ] *= invW0;

	// -- domain warp -------------------------------------------------------
	// Every layer is sampled through one slow world-space wobble, about a metre
	// of displacement over a 17 m cycle, so a directional pattern never runs
	// dead straight for a hundred metres. A tile small enough to look right up
	// close still gives itself away as ruler-straight lines without this; it is
	// what turns the sand's wind ripples from corduroy into drifted sand.
	vec2 wp = wpos.xz + ( texture2D( tMacro, wpos.xz * uWarpScale ).rg - 0.5 ) * uWarpStrength;

	// -- height-map blending ---------------------------------------------
	// key = weight * (1 + height*influence). Multiplicative, not additive: a
	// layer the control map says is not here has weight 0 and therefore key 0
	// no matter how tall its texel is, so sand and rock can never punch
	// through a grass slope. Where two layers *are* close the per-texel height
	// decides between them, which is the crisp interlocking transition height
	// blending exists for. The additive form this replaces let any layer with
	// a bright height texel beat a layer the map had at 0.5, and that is
	// precisely how a green landscape came out uniformly brown.
	vec3 layerAlb[ ${SPLAT_COUNT} ];
	vec2 layerNrm[ ${SPLAT_COUNT} ];
	float key[ ${SPLAT_COUNT} ];
	float peak = 0.0;

	for ( int i = 0; i < ${SPLAT_COUNT}; i ++ ) {
		vec2 r = uLayerRot[ i ];
		vec2 uv = vec2( wp.x * r.x - wp.y * r.y, wp.x * r.y + wp.y * r.x ) * uLayerScale[ i ];
		layerAlb[ i ] = texture( tAlbedo, vec3( uv, float( i ) ) ).rgb * uLayerTint[ i ];
		vec4 nh = texture( tNormalHeight, vec3( uv, float( i ) ) );
		layerNrm[ i ] = nh.xy * 2.0 - 1.0;
		key[ i ] = lw[ i ] * ( 1.0 + nh.w * uHeightInfluence );
		peak = max( peak, key[ i ] );
	}
	// Relative, not absolute: the keys scale with the weights, so a fixed
	// offset would be a hard cut on a strong layer and a full cross-fade on a
	// weak one.
	float thresh = peak * ( 1.0 - uBlendSharpness );

	vec3 albedo = vec3( 0.0 );
	vec2 flatN = vec2( 0.0 );
	float wsum = 0.0;
	for ( int i = 0; i < ${SPLAT_COUNT}; i ++ ) {
		float b = max( key[ i ] - thresh, 0.0 );
		wsum += b;
		albedo += layerAlb[ i ] * b;
		flatN += layerNrm[ i ] * b;
	}
	float invW = 1.0 / max( wsum, 1.0e-6 );
	albedo *= invW;
	flatN *= invW;

	// -- macro variation ---------------------------------------------------
	// One cycle over the whole map, plus a mid-frequency octave. This is the
	// single cheapest thing that stops tiling from reading.
	vec3 macroLo = texture2D( tMacro, ctlUv * ( uWorldSize * uMacroScale ) ).rgb;
	vec3 macroHi = texture2D( tMacro, wpos.xz * uMacroScale * 4.3 + vec2( 0.37, 0.61 ) ).rgb;
	// R swings overall tone, G swings dry-straw against damp-green. B is read
	// *only* by the cliff selector below and must never reach the tint: it
	// carries the full 0..1 range, and multiplying albedo by it was crushing
	// blue by up to 70% across whole hillsides — which is a yellow filter, and
	// a large part of why the world came out beige.
	float mTone = mix( macroLo.r, macroHi.r, 0.38 ) - 0.5;
	float mWarm = mix( macroLo.g, macroHi.g, 0.38 ) - 0.5;
	vec3 macro = vec3( 1.0 )
		+ mTone * uMacroStrength * vec3( 0.85, 0.82, 0.72 )
		+ mWarm * uMacroStrength * vec3( 0.50, 0.26, -0.34 );

	// -- triplanar cliffs --------------------------------------------------
	float steep = smoothstep( uSlopeRange.x, uSlopeRange.y, 1.0 - abs( wnrm.y ) );
	float kx = abs( wnrm.x );
	float kz = abs( wnrm.z );
	float kn = 1.0 / max( kx + kz, 1.0e-4 );
	kx *= kn;
	kz *= kn;

	// Broad bands decide whether a cliff face is blocky rock or bedded strata.
	float strata = smoothstep( 0.36, 0.64, macroLo.b );

	vec2 uvX = wpos.zy * uCliffScale;
	vec2 uvZ = wpos.xy * uCliffScale;

	vec3 aX = mix( texture( tAlbedo, vec3( uvX, ${LAYER_ROCKFACE}.0 ) ).rgb * uLayerTint[ ${LAYER_ROCKFACE} ],
	               texture( tAlbedo, vec3( uvX, ${LAYER_STRATA}.0 ) ).rgb * uLayerTint[ ${LAYER_STRATA} ], strata );
	vec3 aZ = mix( texture( tAlbedo, vec3( uvZ, ${LAYER_ROCKFACE}.0 ) ).rgb * uLayerTint[ ${LAYER_ROCKFACE} ],
	               texture( tAlbedo, vec3( uvZ, ${LAYER_STRATA}.0 ) ).rgb * uLayerTint[ ${LAYER_STRATA} ], strata );
	vec2 nX = mix( texture( tNormalHeight, vec3( uvX, ${LAYER_ROCKFACE}.0 ) ).xy,
	               texture( tNormalHeight, vec3( uvX, ${LAYER_STRATA}.0 ) ).xy, strata ) * 2.0 - 1.0;
	vec2 nZ = mix( texture( tNormalHeight, vec3( uvZ, ${LAYER_ROCKFACE}.0 ) ).xy,
	               texture( tNormalHeight, vec3( uvZ, ${LAYER_STRATA}.0 ) ).xy, strata ) * 2.0 - 1.0;

	vec3 cliffAlb = aX * kx + aZ * kz;

	// Normal maps are combined as world-space gradient offsets, which lets the
	// planar and triplanar frames be mixed without ever building a TBN matrix.
	float sx = wnrm.x < 0.0 ? -1.0 : 1.0;
	float sz = wnrm.z < 0.0 ? -1.0 : 1.0;
	vec3 cliffOff = vec3( 0.0, nX.y, nX.x * sx ) * kx
	              + vec3( nZ.x * sz, nZ.y, 0.0 ) * kz;

	albedo = mix( albedo, cliffAlb, steep );
	vec3 nOffset = mix( vec3( flatN.x, 0.0, flatN.y ), cliffOff, steep );

	// -- detail multiply ---------------------------------------------------
	// A second tap at ~0.38 m, six times finer than the finest surface layer.
	// Normalised against the layer's own measured mean so it modulates contrast
	// without shifting exposure — the old "lum * 1.85" darkened the ground by a
	// third on average, which is a brightness bug dressed up as a detail knob.
	// Sampled unconditionally: putting the fetch behind the fade test would
	// break its derivatives and mip-select garbage along the fade boundary.
	float detailFade = 1.0 - smoothstep( uDetailFade.x, uDetailFade.y, vTerrainViewDist );
	vec3 detail = texture( tAlbedo, vec3( wp.yx * uDetailScale, ${LAYER_DETAIL}.0 ) ).rgb;
	float detailLum = dot( detail, TERRAIN_LUMA ) / max( uDetailMid, 1.0e-3 );
	albedo *= 1.0 + ( detailLum - 1.0 ) * uDetailStrength * detailFade;

	albedo *= macro;

	// Muted, per the reference: the hue comes from the per-layer tints, and
	// pushing vividness on top of that is what makes a landscape look like a
	// toy. Below 1.0 this pulls chroma back without touching brightness.
	float albedoLum = dot( albedo, TERRAIN_LUMA );
	albedo = max( mix( vec3( albedoLum ), albedo, uSaturation ), vec3( 0.0 ) );

	vec3 terrainNormalW = normalize( wnrm + nOffset * uNormalStrength );

	diffuseColor.rgb *= albedo;
`;

function patchFragment(src: string): string {
  let out = src.replace('#include <common>', `#include <common>\n${VARYINGS}\n${FRAG_UNIFORMS}`);

  out = out.replace('#include <map_fragment>', FRAG_SURFACE);

  // Baked AO lives in the vertex colour; apply it with a strength control
  // rather than letting three multiply it in at full force.
  out = out.replace(
    '#include <color_fragment>',
    /* glsl */ `	diffuseColor.rgb *= mix( vec3( 1.0 ), vColor.rgb, uAoStrength );`,
  );

  out = out.replace(
    '#include <normal_fragment_maps>',
    /* glsl */ `	normal = normalize( ( viewMatrix * vec4( terrainNormalW, 0.0 ) ).xyz );`,
  );

  // Fog *before* <colorspace_fragment>, unlike three's own fog, so it composites
  // in linear space and lands on exactly the sky colour at full density.
  out = out.replace(
    'vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;',
    /* glsl */ `vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;
	float terrainFog = 1.0 - exp( - uFogDensity * uFogDensity * vTerrainViewDist * vTerrainViewDist );
	outgoingLight = mix( outgoingLight, uFogColor, clamp( terrainFog, 0.0, 1.0 ) );`,
  );

  return out;
}

/* -------------------------------------------------------------------------- */
/* Texture assembly                                                           */
/* -------------------------------------------------------------------------- */

function imageData(tex: THREE.Texture): Uint8Array {
  const data = (tex.image as { data?: ArrayBufferView } | undefined)?.data;
  if (!data || !ArrayBuffer.isView(data)) {
    throw new Error('TerrainMaterial: expected a DataTexture with raw pixel data');
  }
  // `getTexture` hands back Uint8ClampedArray; reinterpret rather than copy.
  return data instanceof Uint8Array
    ? data
    : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
}

/* -------------------------------------------------------------------------- */
/* Palette retargeting                                                        */
/* -------------------------------------------------------------------------- */

/** sRGB byte -> linear, matching what the sampler does to an SRGB8 texture. */
const SRGB_TO_LINEAR = (() => {
  const t = new Float32Array(256);
  for (let i = 0; i < 256; i++) {
    const c = i / 255;
    t[i] = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  }
  return t;
})();

/**
 * Average linear colour of one RGBA8 layer.
 *
 * Linear rather than sRGB because that is the space the shader multiplies in:
 * averaging the encoded bytes would put the target roughly 30% off, and the
 * correction would over-shoot every dark layer.
 */
function meanLinearColor(rgba: Uint8Array): THREE.Vector3 {
  let r = 0;
  let g = 0;
  let b = 0;
  const n = rgba.length / 4;
  for (let i = 0; i < rgba.length; i += 4) {
    r += SRGB_TO_LINEAR[rgba[i]!]!;
    g += SRGB_TO_LINEAR[rgba[i + 1]!]!;
    b += SRGB_TO_LINEAR[rgba[i + 2]!]!;
  }
  return new THREE.Vector3(r / n, g / n, b / n);
}

function luminance(c: THREE.Vector3): number {
  return c.x * 0.2126 + c.y * 0.7152 + c.z * 0.0722;
}

/**
 * Per-channel scale that moves `mean` onto `targetHex`.
 *
 * A ratio rather than an offset, so the layer's own contrast survives: every
 * pebble and tuft keeps its relative brightness, only the average shifts. The
 * clamp stops a layer with almost no blue in it from having its blue channel
 * multiplied by twenty.
 */
function paletteTint(mean: THREE.Vector3, targetHex: number): THREE.Vector3 {
  // `THREE.Color` decodes hex as sRGB into the linear working space, which is
  // the same space `mean` is measured in.
  const target = new THREE.Color(targetHex);
  const ratio = (t: number, m: number): number =>
    clamp(lerp(1, t / Math.max(m, 1e-4), TINT_AMOUNT), TINT_RANGE[0], TINT_RANGE[1]);
  return new THREE.Vector3(
    ratio(target.r, mean.x),
    ratio(target.g, mean.y),
    ratio(target.b, mean.z),
  );
}

/**
 * Normal in RGB, height in A. Halves the sampler count and the fetch count for
 * the splat loop, which is the difference between "runs everywhere" and
 * "compiles but exceeds MAX_TEXTURE_IMAGE_UNITS on a laptop".
 */
function packNormalHeight(
  normalMap: THREE.Texture | undefined,
  heightMap: THREE.Texture | undefined,
  size: number,
): Uint8Array {
  const out = new Uint8Array(size * size * 4);
  const n = normalMap ? imageData(normalMap) : null;
  const h = heightMap ? imageData(heightMap) : null;
  for (let i = 0; i < size * size; i++) {
    const o = i * 4;
    if (n) {
      out[o] = n[o]!;
      out[o + 1] = n[o + 1]!;
      out[o + 2] = n[o + 2]!;
    } else {
      // Flat normal.
      out[o] = 128;
      out[o + 1] = 128;
      out[o + 2] = 255;
    }
    out[o + 3] = h ? h[o]! : 128;
  }
  return out;
}

function buildArrayTexture(
  layers: Uint8Array[],
  size: number,
  colorSpace: THREE.ColorSpace,
): THREE.DataArrayTexture {
  const stride = size * size * 4;
  const data = new Uint8Array(stride * layers.length);
  for (let i = 0; i < layers.length; i++) {
    const src = layers[i]!;
    if (src.length !== stride) {
      throw new Error(
        `TerrainMaterial: layer ${i} is ${src.length} bytes, expected ${stride} — all terrain textures must share one resolution`,
      );
    }
    data.set(src, i * stride);
  }
  const tex = new THREE.DataArrayTexture(data, size, size, layers.length);
  tex.colorSpace = colorSpace;
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  // Same filtering contract as the rest of the texture library: crisp texels up
  // close, trilinear + aniso at range so nothing crawls at speed.
  tex.magFilter = THREE.NearestFilter;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.generateMipmaps = true;
  tex.anisotropy = 8;
  tex.needsUpdate = true;
  return tex;
}

/* -------------------------------------------------------------------------- */
/* Macro variation noise                                                      */
/* -------------------------------------------------------------------------- */

const MACRO_SIZE = 128;

/**
 * Tileable low-frequency value noise, one octave set per channel.
 *
 * Tileable specifically because the shader samples it twice at different rates:
 * once stretched across the entire world (the macro tint) and once repeating
 * every couple of hundred metres (the mid-frequency break-up). A non-periodic
 * field would show a hard seam on the second lookup.
 */
function buildMacroTexture(seed: number): THREE.DataTexture {
  const data = new Uint8Array(MACRO_SIZE * MACRO_SIZE * 4);
  // R/G tint the albedo, B selects between the two cliff materials, so B gets
  // the full range while R/G stay gentle.
  const spread = [0.30, 0.30, 1.0];
  const seeds = [seed ^ 0x1f3a5c, seed ^ 0x77c21b, seed ^ 0x2bd490];

  for (let c = 0; c < 3; c++) {
    for (let y = 0; y < MACRO_SIZE; y++) {
      for (let x = 0; x < MACRO_SIZE; x++) {
        const v = periodicFbm(x / MACRO_SIZE, y / MACRO_SIZE, seeds[c]!, 4);
        const t = clamp(0.5 + (v - 0.5) * spread[c]!, 0, 1);
        data[(y * MACRO_SIZE + x) * 4 + c] = Math.round(t * 255);
      }
    }
  }
  for (let i = 0; i < MACRO_SIZE * MACRO_SIZE; i++) data[i * 4 + 3] = 255;

  const tex = new THREE.DataTexture(
    data,
    MACRO_SIZE,
    MACRO_SIZE,
    THREE.RGBAFormat,
    THREE.UnsignedByteType,
  );
  tex.colorSpace = THREE.NoColorSpace;
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.magFilter = THREE.LinearFilter;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.generateMipmaps = true;
  tex.needsUpdate = true;
  return tex;
}

/** Value noise on a lattice that wraps every `period` cells. */
function periodicValue(u: number, v: number, period: number, seed: number): number {
  const x = u * period;
  const y = v * period;
  const xi = Math.floor(x);
  const yi = Math.floor(y);
  const tx = x - xi;
  const ty = y - yi;
  const sx = tx * tx * (3 - 2 * tx);
  const sy = ty * ty * (3 - 2 * ty);
  const at = (a: number, b: number): number =>
    hash2i(((a % period) + period) % period, ((b % period) + period) % period, seed);
  return lerp(
    lerp(at(xi, yi), at(xi + 1, yi), sx),
    lerp(at(xi, yi + 1), at(xi + 1, yi + 1), sx),
    sy,
  );
}

function periodicFbm(u: number, v: number, seed: number, octaves: number): number {
  let amp = 1;
  let period = 2;
  let sum = 0;
  let norm = 0;
  for (let o = 0; o < octaves; o++) {
    sum += amp * periodicValue(u, v, period, seed + o * 977);
    norm += amp;
    amp *= 0.5;
    period *= 2;
  }
  return sum / norm;
}
