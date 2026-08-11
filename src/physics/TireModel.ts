/**
 * ============================================================================
 *  TIRE MODEL — combined-slip Pacejka with load sensitivity.
 * ============================================================================
 *
 * Everything here is pure maths on plain numbers: no THREE, no Rapier, no
 * state beyond a small memo cache. `Vehicle` owns the slip state and calls in.
 *
 * Sign conventions (all in the wheel's ground-plane frame):
 *   +x = wheel rolling direction (forward)
 *   +y = wheel lateral direction (right)
 *   slipRatio  kappa  > 0 = the tyre is spinning faster than the road (wheelspin)
 *                     < 0 = the tyre is turning slower than the road (braking)
 *                    = -1 = fully locked
 *   tanSlipAngle       > 0 = the contact patch is sliding to the LEFT, so the
 *                            tyre pushes RIGHT. Fy comes out with this sign.
 *
 * The force magnitude comes from the simplified Magic Formula
 *
 *     F = D * sin(C * atan(B*x - E*(B*x - atan(B*x))))
 *
 * with D = mu(Fz) * Fz. The `sin(...)` factor peaks at exactly 1, so D is the
 * true peak force and the friction coefficient in the tuning means what it
 * says. Past the peak the curve settles at sin(C*pi/2) — that is the "tail"
 * that makes a loose surface forgiving and rock snappy.
 *
 * Combined slip uses the normalised-slip-vector method (Beckman / RCVD):
 * each slip is divided by the slip at which *its own* curve peaks, the two are
 * combined into one vector, one scalar force magnitude is taken from the curve
 * at the combined slip, and it is handed back out along the slip direction.
 * This automatically produces a friction ellipse, and it is why a locked wheel
 * (|kappa| = 1, far past its peak) has essentially no cornering grip left.
 */

import type { SurfaceTuning, TireTuning } from './VehicleTuning';

/** Result of one tyre evaluation, in the wheel's ground-plane frame. */
export interface TireForces {
  /** Longitudinal force, N. Positive accelerates the car forwards. */
  fx: number;
  /** Lateral force, N. Positive pushes the chassis to its right. */
  fy: number;
  /** Peak force the contact patch could produce at this load, N (= mu*Fz). */
  capacity: number;
  /** Effective friction coefficient after load sensitivity. */
  mu: number;
  /**
   * Combined normalised slip. 1.0 = exactly at the friction peak, > 1 = sliding.
   * Handy for skid audio and particle intensity.
   */
  saturation: number;
}

/**
 * The slip value at which a Magic Formula curve reaches its peak. Depends only
 * on the shape factors, so it is solved once per surface and memoised.
 */
export interface SlipPeaks {
  /** Slip ratio at the longitudinal peak. */
  kappa: number;
  /** tan(slip angle) at the lateral peak. */
  alpha: number;
}

/** The raw normalised Magic Formula. Returns a value in roughly [-1, 1]. */
export function magicFormula(x: number, B: number, C: number, E: number): number {
  const bx = B * x;
  const inner = bx - E * (bx - Math.atan(bx));
  return Math.sin(C * Math.atan(inner));
}

/**
 * Slope of `magicFormula` at x = 0, i.e. the normalised slip stiffness.
 * d/dx sin(C*atan(B*x - ...)) at 0 = B*C.
 */
export function magicFormulaStiffness(B: number, C: number): number {
  return B * C;
}

/**
 * Solve for the slip at which the Magic Formula peaks.
 *
 * The peak is where the argument of the outer sine reaches pi/2, i.e.
 *     atan(inner) = pi/(2C)   ->   inner = tan(pi/(2C)).
 * `inner(x)` is strictly increasing for E < 1, so a bisection is exact and
 * cheap. If C <= 1 the sine never reaches its crest and the curve is
 * monotonic; we fall back to the slip at 99% of the asymptote so the
 * normalisation used by the combined-slip solver still has a sane scale.
 */
function solvePeak(B: number, C: number, E: number): number {
  if (!(B > 0) || !Number.isFinite(B) || !Number.isFinite(C) || !Number.isFinite(E)) return 0.2;
  const target = C > 1.0001 ? Math.tan(Math.PI / (2 * C)) : Math.tan(Math.PI / 2.0002);
  const inner = (x: number): number => {
    const bx = B * x;
    return bx - E * (bx - Math.atan(bx));
  };
  let lo = 0;
  let hi = 1 / B;
  let guard = 0;
  while (inner(hi) < target && hi < 1e4 && guard++ < 200) hi *= 2;
  if (!(hi < 1e4)) return 4; // degenerate shape; clamp to something usable
  for (let i = 0; i < 60; i++) {
    const mid = 0.5 * (lo + hi);
    if (inner(mid) < target) lo = mid;
    else hi = mid;
  }
  return 0.5 * (lo + hi);
}

const peakCache = new WeakMap<SurfaceTuning, SlipPeaks>();

/** Peak slips for a surface, memoised on the tuning object itself. */
export function slipPeaks(surface: SurfaceTuning): SlipPeaks {
  let p = peakCache.get(surface);
  if (p === undefined) {
    p = {
      kappa: Math.max(1e-3, solvePeak(surface.Bx, surface.Cx, surface.Ex)),
      alpha: Math.max(1e-3, solvePeak(surface.By, surface.Cy, surface.Ey)),
    };
    peakCache.set(surface, p);
  }
  return p;
}

/**
 * Effective friction coefficient at a given vertical load.
 *
 * mu = friction * (1 - k * (Fz/Fz_nom - 1))
 *
 * Grip per newton falls as load rises, so an axle that gains load in a weight
 * transfer gives back less than it took. That asymmetry is the whole reason a
 * car understeers on entry and oversteers on power — it is not scripted
 * anywhere, it falls out of this one line.
 */
export function loadSensitiveMu(surface: SurfaceTuning, tire: TireTuning, load: number): number {
  const ratio = load / Math.max(1, tire.nominalLoad);
  const scale = clamp(
    1 - tire.loadSensitivity * (ratio - 1),
    tire.loadSensitivityMin,
    tire.loadSensitivityMax,
  );
  return surface.friction * scale * tire.gripScale;
}

/**
 * Combined-slip tyre forces.
 *
 * @param slipRatio     longitudinal slip ratio (already relaxed by the caller)
 * @param tanSlipAngle  tan of the lateral slip angle (already relaxed)
 * @param load          vertical load on the contact patch, N (>= 0)
 */
export function tireForces(
  slipRatio: number,
  tanSlipAngle: number,
  load: number,
  surface: SurfaceTuning,
  tire: TireTuning,
  out: TireForces,
): TireForces {
  out.fx = 0;
  out.fy = 0;
  out.saturation = 0;
  out.mu = 0;
  out.capacity = 0;
  if (!(load > 0) || !Number.isFinite(load)) return out;

  const mu = loadSensitiveMu(surface, tire, load);
  const peaks = slipPeaks(surface);

  // Normalised slips: 1.0 means "exactly at this axis's own friction peak".
  const nx = slipRatio / peaks.kappa;
  const ny = tanSlipAngle / peaks.alpha;
  const n = Math.hypot(nx, ny);

  out.mu = mu;
  out.capacity = mu * load;
  out.saturation = n;

  if (n < 1e-9) return out;

  // Direction of the combined slip vector; the force comes back along -it.
  const dirX = nx / n;
  const dirY = ny / n;

  // Evaluate each axis's own curve at the *combined* slip, expressed back in
  // that axis's own units. With pure slip on one axis this reduces exactly to
  // that axis's pure-slip curve, which is what makes the model tunable.
  const fxMag = magicFormula(n * peaks.kappa, surface.Bx, surface.Cx, surface.Ex);
  const fyMag = magicFormula(n * peaks.alpha, surface.By, surface.Cy, surface.Ey);

  const d = mu * load;
  out.fx = d * tire.longGripScale * dirX * fxMag;
  out.fy = d * tire.latGripScale * dirY * fyMag;
  return out;
}

/**
 * Local slope dFx/dkappa of the combined model, for the caller's semi-implicit
 * wheel integrator. Clamped non-negative: past the peak the true slope is
 * negative, and feeding that into an implicit solve would make it unstable.
 * Zero there is correct enough because the curve is nearly flat anyway.
 */
export function longitudinalSlope(
  slipRatio: number,
  tanSlipAngle: number,
  load: number,
  surface: SurfaceTuning,
  tire: TireTuning,
  scratch: TireForces,
): number {
  if (!(load > 0)) return 0;
  const peaks = slipPeaks(surface);
  const h = 0.02 * peaks.kappa;
  const a = tireForces(slipRatio - h, tanSlipAngle, load, surface, tire, scratch).fx;
  const b = tireForces(slipRatio + h, tanSlipAngle, load, surface, tire, scratch).fx;
  const slope = (b - a) / (2 * h);
  if (!Number.isFinite(slope)) return 0;
  // Never exceed the true origin stiffness — a bad finite difference must not
  // make the implicit solve over-stiff either.
  const maxSlope = (mu0(surface, tire, load) * load * magicFormulaStiffness(surface.Bx, surface.Cx)) / 1;
  return clamp(slope, 0, maxSlope);
}

function mu0(surface: SurfaceTuning, tire: TireTuning, load: number): number {
  return loadSensitiveMu(surface, tire, load) * tire.longGripScale;
}

/** Allocate a reusable output record. */
export function makeTireForces(): TireForces {
  return { fx: 0, fy: 0, capacity: 0, mu: 0, saturation: 0 };
}

export function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}
