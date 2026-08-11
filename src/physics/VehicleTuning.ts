/**
 * ============================================================================
 *  VEHICLE TUNING — every magic number in the car lives here.
 * ============================================================================
 *
 * Coordinate convention for the whole vehicle subsystem:
 *   +Y = chassis UP
 *   +Z = chassis FORWARD
 *   -X = chassis RIGHT
 * (right-handed: X cross Y = Z, and forward matches THREE.Object3D.lookAt,
 * which points a non-camera object's +Z at its target. Note the sign on X: the
 * right hand of something facing +Z with +Y up is forward x up = -X. Positive X
 * is therefore the chassis LEFT, which is why the wheels named FL/FR below sit
 * at negative/positive X respectively and not the other way round. Everything
 * in the car is mirror-symmetric about that axis, so the labels cost nothing —
 * but the STEERING sign depends on it, see Vehicle.updateSteering.)
 *
 * The rigid body's ORIGIN is the geometric reference point of the chassis —
 * the place a renderer should put the body mesh. The centre of mass is offset
 * from it by `chassis.centreOfMass`, which is what makes the car pitch and roll
 * around a sensible point instead of around the middle of its bounding box.
 *
 * Units are SI throughout unless a field name says otherwise:
 *   distance m, mass kg, force N, torque N·m, angle rad, time s,
 *   spring rate N/m, damping N·s/m, rotational inertia kg·m², speed rad/s.
 *
 * The reference vehicle is a Wrangler-class 4x4: ~1900 kg, 2.60 m wheelbase,
 * 1.60 m track, 33-inch tyres, 5-speed automatic, part-time 4WD with a
 * low-range transfer case and limited-slip differentials on all three diffs.
 */

import type { SurfaceKind } from '../types';

/** Plain xyz literal — avoids dragging THREE into the data file. */
export interface Vec3Lit {
  x: number;
  y: number;
  z: number;
}

// ---------------------------------------------------------------------------
// Chassis
// ---------------------------------------------------------------------------

export interface ChassisTuning {
  /** Total vehicle mass including wheels and a driver, kg. */
  mass: number;
  /** Half-extents of the chassis collider box, m. (x = half width, y = half height, z = half length) */
  halfExtents: Vec3Lit;
  /**
   * Centre of the collider box relative to the body origin, m. The body origin
   * sits at ground level in the *design* pose, so the box centre is raised.
   */
  colliderOffset: Vec3Lit;
  /**
   * Centre of mass in body-local space, m. Low (keeps the roll moment sane on
   * a tall 4x4) and a whisker AHEAD of the wheelbase centre — see the note on
   * COM_Z, where that sign turns out to be what decides whether the car can
   * stand on three wheels with the fourth in a hole.
   */
  centreOfMass: Vec3Lit;
  /**
   * Principal moments of inertia about the centre of mass, kg·m².
   *   x = PITCH (nose up/down), y = YAW, z = ROLL.
   * Set explicitly rather than derived from the box: a real vehicle carries
   * most of its mass low and central, so its roll inertia is far below what a
   * uniform-density box of the same size would give.
   */
  inertia: Vec3Lit;
  /** Rapier linear damping. Aero drag is modelled explicitly, so keep this at 0. */
  linearDamping: number;
  /** Rapier angular damping — a whisker of it stops slow parasitic tumbling. */
  angularDamping: number;
  /** Friction of the chassis collider itself (only matters when you land on the roof). */
  colliderFriction: number;
  /** Restitution of the chassis collider — bodywork does not bounce. */
  colliderRestitution: number;
}

// ---------------------------------------------------------------------------
// Wheels & suspension
// ---------------------------------------------------------------------------

export type AxleId = 'front' | 'rear';

export interface WheelTuning {
  /** Human-readable id, e.g. 'FL'. */
  name: string;
  /**
   * Suspension attachment (top of the strut) in body-local space, m.
   * The wheel hub hangs below this along the chassis -Y axis.
   */
  attach: Vec3Lit;
  axle: AxleId;
  /** Does this wheel steer with the front rack? */
  steered: boolean;
  /** Is this wheel driven by the transfer case? */
  driven: boolean;
  /** Fraction of the handbrake torque this wheel receives (rear-only on a Jeep). */
  handbrakeShare: number;
  /**
   * Sign of the steering angle. Both front wheels turn the same way; this only
   * exists so a rear-steer variant is a data change rather than a code change.
   */
  steerSign: number;
}

export interface AxleSuspensionTuning {
  /**
   * Coil rate at the wheel, N/m. Derived, not typed in: cornerLoad / designSag,
   * which puts the static sag at 27% of travel and the sprung natural frequency
   * at 1.29 Hz on both axles — soft enough for real articulation, firm enough
   * that it does not wallow.
   */
  springRate: number;
  /** Damper coefficient on compression, N·s/m. Low: the wheel must be free to move up over a rock. */
  bumpDamping: number;
  /**
   * Damper coefficient on extension, N·s/m. Deliberately ~1.6x the bump rate,
   * and linear where the bump side is digressive (see `reboundDigressiveRatio`).
   * That asymmetry is what stops the car pogoing after a landing.
   */
  reboundDamping: number;
  /**
   * Anti-roll bar rate, N/m of *compression difference* across the axle.
   * Kept soft on purpose: an off-road truck wants articulation, not flat
   * cornering. Front is stiffer than rear, which nudges the balance toward
   * mild understeer at the limit.
   */
  antiRollRate: number;
}

export interface SuspensionTuning {
  /** Distance from the attach point to the hub with the spring fully extended, m. */
  restLength: number;
  /** Total usable travel, m. Fully compressed length = restLength - travel. */
  travel: number;
  /**
   * Fraction of travel at which the bump stop starts to bite (0.45 = last 55%).
   * Long and early rather than short and late: see the note at the value.
   */
  bumpStopZone: number;
  /**
   * Bump stop rate, N/m². Force = stiffness * penetration², so it ramps in
   * progressively instead of hitting like a wall. At full bottom-out on the
   * default numbers this is worth roughly 7x the static corner load, on top of
   * the coil's own 3.7x.
   */
  bumpStopStiffness: number;
  /**
   * Hysteresis of the bump stop, 0..1. A microcellular urethane stop is not a
   * spring: it gives back appreciably less than it took, and that lost energy
   * is exactly what stops a bottomed-out landing catapulting the car back into
   * the air. 0 = a perfectly elastic stop, 1 = it returns nothing.
   *
   * Applied as a multiplier on the stop force that runs from 1 while the stop
   * is being compressed down to (1 - hysteresis) while it is releasing, so the
   * loading and unloading curves enclose a real area. It costs nothing until
   * the car is actually in the stop, which is what makes it the right tool
   * here: ride quality over chatter never sees it.
   */
  bumpStopHysteresis: number;
  /**
   * Velocity over which the hysteresis blends between the loading and the
   * unloading curve, m/s. Small, but non-zero: a hard switch at v = 0 makes the
   * force discontinuous and the solver chatters at the bottom of the stroke.
   */
  bumpStopHysteresisVelocity: number;
  /**
   * Damper force above which the curve goes digressive (m/s). Beyond this the
   * damper gets much softer per m/s, which is how a real off-road shock lets a
   * wheel snap up over a square edge without spiking the chassis.
   */
  damperKneeVelocity: number;
  /**
   * Compression-side damping coefficient multiplier beyond the knee (0..1).
   * This is the number that decides ride quality over chatter, because a
   * washboard is a pure high-speed *bump* input.
   */
  damperDigressiveRatio: number;
  /**
   * Extension-side damping multiplier beyond the knee (0..1). Kept much closer
   * to linear than the bump side: digressive rebound is what lets a bottomed
   * suspension fire the body back into the air, and unlike the bump side it
   * buys nothing over chatter — the wheel needs to move UP quickly over a rock,
   * not down.
   */
  reboundDigressiveRatio: number;
  /** Absolute clamp on damper force, N. Pure numerical safety at 120 Hz. */
  maxDamperForce: number;
  /** Absolute clamp on total suspension force per wheel, N. Safety net. */
  maxSuspensionForce: number;
  front: AxleSuspensionTuning;
  rear: AxleSuspensionTuning;
}

// ---------------------------------------------------------------------------
// Tyres
// ---------------------------------------------------------------------------

/**
 * Per-surface tyre behaviour. The Pacejka shape factors do the interesting
 * work here:
 *
 *   B — stiffness. Large B = the peak arrives at a small slip (hard, grippy
 *       surface). Small B = you can slide a long way before the peak (sand).
 *   C — shape. This sets the *tail*: the curve settles at D*sin(C*pi/2) at
 *       large slip. C near 1 means almost no drop after the peak (snow, sand:
 *       progressive, forgiving), C near 1.6 means a distinct cliff (rock).
 *   E — curvature near the peak. Below 1; smaller = rounder peak.
 */
export interface SurfaceTuning {
  /** Peak friction coefficient at the nominal wheel load. */
  friction: number;
  /** Longitudinal Pacejka stiffness / shape / curvature. */
  Bx: number;
  Cx: number;
  Ex: number;
  /** Lateral Pacejka stiffness / shape / curvature. */
  By: number;
  Cy: number;
  Ey: number;
  /** Rolling resistance coefficient (fraction of vertical load). Sand and mud drag hard. */
  rollingResistance: number;
}

export type SurfaceTable = Record<SurfaceKind, SurfaceTuning>;

export interface TireTuning {
  /** Rolling radius, m. 33-inch off-road tyre. */
  radius: number;
  /** Section width, m — used by the renderer and for particle FX placement. */
  width: number;
  /** Rotational inertia of wheel + tyre + hub, kg·m². */
  inertia: number;
  /**
   * Load at which the surface friction coefficient is exactly `friction`, N.
   * Defaults to a quarter of the static weight.
   */
  nominalLoad: number;
  /**
   * Load sensitivity. Effective mu = friction * (1 - k*(Fz/Fz_nom - 1)).
   * This is *the* mechanism behind believable weight transfer: the heavily
   * loaded outside tyre gives back less grip per newton than it took, so the
   * axle that transfers the most load is the axle that gives up first.
   */
  loadSensitivity: number;
  /** Clamp on the load-sensitivity multiplier so a wild load spike cannot invert grip. */
  loadSensitivityMin: number;
  loadSensitivityMax: number;
  /**
   * Longitudinal relaxation length, m. The contact patch is a spring: it takes
   * this much rolling distance to build up to the steady-state slip. Also the
   * thing that makes the slip model behave at a standstill instead of dividing
   * by zero.
   */
  relaxLengthLong: number;
  /** Lateral relaxation length, m. Longer than longitudinal, as on a real tyre. */
  relaxLengthLat: number;
  /**
   * Floor on the speed used in the slip relaxation decay term, m/s. Below this
   * the contact patch behaves like a pure spring with a slow leak, which is
   * what holds a parked car still on a slope.
   */
  relaxMinSpeed: number;
  /**
   * Extra viscous damping on contact-patch slip velocity at low speed,
   * N·s/m at the nominal load. The relaxation spring above is almost
   * undamped on its own; without this the car buzzes at ~2.5 Hz when parked.
   * It acts on *slip* velocity, so it costs nothing while rolling freely.
   */
  lowSpeedDamping: number;
  /** Speed above which the low-speed damper has fully faded out, m/s. */
  lowSpeedDampingFade: number;
  /** Hard clamp on stored slip ratio and tan(slip angle). Numerical safety. */
  maxSlipRatio: number;
  maxTanSlipAngle: number;
  /** Global grip trim. 1.0 = as tuned; raise for an arcade-grippier car. */
  gripScale: number;
  /** Extra multiplier on longitudinal grip only (traction vs cornering balance). */
  longGripScale: number;
  /** Extra multiplier on lateral grip only. */
  latGripScale: number;
  /** Maximum wheel spin speed, rad/s. Stops a free-spinning airborne wheel running away. */
  maxSpin: number;
  /** Viscous drag on a free (airborne) wheel, N·m per rad/s — bearings and windage. */
  freeSpinDamping: number;
}

// ---------------------------------------------------------------------------
// Drivetrain
// ---------------------------------------------------------------------------

export interface EngineTuning {
  /** Torque curve as [rpm, N·m] pairs, ascending in rpm. Linearly interpolated. */
  torqueCurve: ReadonlyArray<readonly [number, number]>;
  /** Rotating inertia of crank + flywheel + clutch cover, kg·m². */
  inertia: number;
  /** Idle target, rpm. */
  idleRpm: number;
  /** Below this the engine is considered stalled and gets caught by the governor, rpm. */
  stallRpm: number;
  /** Governor authority: throttle added per rpm below idle (1/rpm), clamped to `idleMaxThrottle`. */
  idleGovernorGain: number;
  idleMaxThrottle: number;
  /** Soft rev limit, rpm — fuel starts being cut here. */
  redlineRpm: number;
  /** Fuel is fully cut this far above the redline, rpm. */
  limiterBandRpm: number;
  /** How long a hard cut lasts once the limiter trips, s. Gives the classic stutter. */
  limiterCutTime: number;
  /** Absolute ceiling, rpm — the state is clamped here no matter what. */
  maxRpm: number;
  /** Closed-throttle friction: constant term, N·m. */
  frictionConstant: number;
  /** Closed-throttle friction: viscous term, N·m per rad/s. Sets engine braking. */
  frictionViscous: number;
  /**
   * How much of the friction torque survives at full throttle (0..1). Pumping
   * losses fall away as the throttle opens, so engine braking only really
   * exists on lift-off.
   */
  frictionThrottleRelief: number;
}

export interface ClutchTuning {
  /** Maximum torque the clutch can transmit when fully clamped, N·m. */
  capacity: number;
  /** Torque per rad/s of slip while slipping, N·m·s/rad. High = it locks up crisply. */
  stiffness: number;
  /** Engine speed the auto-clutch aims to hold during a standing launch at full throttle, rpm. */
  launchRpm: number;
  /** Width of the launch engagement ramp, rpm. */
  launchBandRpm: number;
  /**
   * Driveline speed (expressed at the engine, rpm) above which the clutch is
   * always fully locked regardless of throttle.
   */
  lockRpm: number;
}

export interface DifferentialTuning {
  /**
   * Static preload, N·m. This is the bit that matters off-road: it is the
   * torque the diff will bias to the slower wheel with *no* input torque at
   * all, which is how a Jeep crawls out of a hole with one wheel in the air.
   */
  preload: number;
  /** Torque sensitivity: extra bias per N·m of axle torque (the ramp angle). */
  torqueSensitivity: number;
  /** Absolute ceiling on bias torque, N·m. */
  maxBias: number;
  /** Speed difference at which the bias is ~76% of maximum, rad/s (tanh scale). */
  slipReference: number;
}

export interface TransmissionTuning {
  /**
   * Forward gear ratios, index 0 = 1st. Reverse is `reverseRatio`.
   * Total ratio = gear * transferCase * finalDrive.
   */
  gearRatios: readonly number[];
  /** Reverse gear ratio, positive magnitude — the sign is applied in code. */
  reverseRatio: number;
  finalDrive: number;
  /** Transfer case ratios. Low range is the off-roader's crawl gear. */
  transferHigh: number;
  transferLow: number;
  /** Mechanical efficiency of the whole driveline, 0..1. */
  efficiency: number;
  /** Time a gearchange takes, s. Torque is cut and the clutch released for this long. */
  shiftTime: number;
  /** Minimum time between shifts, s. Stops hunting. */
  shiftCooldown: number;
  /** Auto upshift rpm at closed throttle .. at full throttle. */
  upshiftRpmLow: number;
  upshiftRpmHigh: number;
  /** Auto downshift rpm at closed throttle .. at full throttle (kickdown). */
  downshiftRpmLow: number;
  downshiftRpmHigh: number;
  /** A downshift is refused if it would put the engine above this fraction of the redline. */
  downshiftRpmCeilingFraction: number;
  /** Start in automatic mode. shiftUp/shiftDown always work as a manual override. */
  automatic: boolean;
  /**
   * Arcade reverse: hold the brake at a standstill for `autoReverseDelay` and
   * the gearbox selects reverse, with the brake pedal acting as the reverse
   * throttle. Set false for a strictly manual reverse via shiftDown.
   */
  autoReverse: boolean;
  autoReverseDelay: number;
  /** Speed below which the gearbox is allowed to change direction, m/s. */
  directionChangeSpeed: number;
}

export interface DrivetrainTuning {
  engine: EngineTuning;
  clutch: ClutchTuning;
  transmission: TransmissionTuning;
  /** Fraction of drive torque sent to the front axle by the centre diff. */
  frontTorqueSplit: number;
  centreDiff: DifferentialTuning;
  frontDiff: DifferentialTuning;
  rearDiff: DifferentialTuning;
}

// ---------------------------------------------------------------------------
// Brakes, steering, aero, assists
// ---------------------------------------------------------------------------

export interface BrakeTuning {
  /** Total brake torque at the wheels at full pedal, N·m (split by bias). */
  maxTorque: number;
  /** Fraction of brake torque on the front axle. */
  frontBias: number;
  /** Handbrake torque, N·m, applied to the rear wheels only. */
  handbrakeTorque: number;
}

export interface SteeringTuning {
  /** Steering angle at a standstill, rad. */
  maxAngleLowSpeed: number;
  /** Steering angle floor at high speed, rad. */
  maxAngleHighSpeed: number;
  /**
   * Reference speed for the speed-sensitive falloff, m/s. The limit follows
   * 1/(1+(v/vRef)^2), so at vRef the extra angle is already halved.
   */
  speedFalloffRef: number;
  /** Steering rack rate at a standstill, rad/s — sets how quickly the wheels turn. */
  rateLowSpeed: number;
  /** Steering rack rate at high speed, rad/s. Slower = the car feels heavier. */
  rateHighSpeed: number;
  /** Speed at which `rateHighSpeed` is reached, m/s. */
  rateFalloffRef: number;
  /** Rack return-to-centre rate multiplier when the driver lets go. */
  returnRateScale: number;
  /**
   * Ackermann blend, 0 = both front wheels parallel, 1 = full Ackermann
   * geometry (inside wheel steers more). Real 4x4s run partial Ackermann.
   */
  ackermann: number;
  /** Wheelbase used by the Ackermann solve, m. Derived from the wheel layout if left 0. */
  wheelbase: number;
  /** Front track used by the Ackermann solve, m. Derived from the wheel layout if left 0. */
  frontTrack: number;
  /**
   * Counter-steer assist gain: steering angle added per rad of rear-axle
   * sideslip. Set to 0 for a completely unassisted car.
   */
  counterSteerGain: number;
  /** Maximum angle the counter-steer assist may add on its own, rad. */
  counterSteerMaxAngle: number;
  /** Sideslip below this is ignored by the assist, rad. */
  counterSteerDeadzone: number;
  /** The assist fades in above this speed and is full by `counterSteerFullSpeed`, m/s. */
  counterSteerMinSpeed: number;
  counterSteerFullSpeed: number;
}

export interface AeroTuning {
  /** Air density, kg/m³. */
  airDensity: number;
  /** Drag coefficient. A Wrangler is a brick: ~0.58. */
  dragCoefficient: number;
  /** Frontal area, m². */
  frontalArea: number;
  /**
   * Downforce coefficient, N per (m/s)². Essentially nothing on a box like
   * this, but non-zero keeps it planted at silly speeds.
   */
  downforceCoefficient: number;
  /** Fraction of downforce applied at the front axle. */
  downforceFrontBias: number;
}

export interface AssistTuning {
  /** Airborne self-righting gain, 1/s² (torque = gain * inertia * sin(tilt)). */
  airUprightGain: number;
  /** Airborne pitch/roll rate damping, 1/s. */
  airPitchDamping: number;
  /** Airborne yaw rate damping, 1/s. Lower, so the car can still be steered in flight. */
  airYawDamping: number;
  /** Airtime before the attitude assist starts fading in, s. */
  airAssistDelay: number;
  /** Fade-in duration for the attitude assist, s. */
  airAssistFade: number;
  /** Clamp on any single assist torque, N·m. */
  maxAssistTorque: number;
  /** Height the car is lifted by when `recover` is pressed, m. */
  recoverLift: number;
  /** Minimum time between recover presses, s. */
  recoverCooldown: number;
  /**
   * Below this dot(chassisUp, worldUp) the car counts as flipped. Recover also
   * works when upright (it acts as an unstick), but this drives the UI hint.
   */
  flippedThreshold: number;
}

// ---------------------------------------------------------------------------
// Top-level
// ---------------------------------------------------------------------------

export interface VehicleTuning {
  chassis: ChassisTuning;
  /** Wheel order is fixed: [front-left, front-right, rear-left, rear-right]. */
  wheels: readonly WheelTuning[];
  suspension: SuspensionTuning;
  tire: TireTuning;
  surfaces: SurfaceTable;
  drivetrain: DrivetrainTuning;
  brakes: BrakeTuning;
  steering: SteeringTuning;
  aero: AeroTuning;
  assists: AssistTuning;
}

// ===========================================================================
//  THE JEEP
// ===========================================================================

const WHEELBASE = 2.6; // m, front-to-rear axle
const TRACK = 1.6; // m, left-to-right wheel centres
const WHEEL_RADIUS = 0.4; // m, 33-inch tyre
const MASS = 1900; // kg
const GRAVITY = 9.81;
const WEIGHT = MASS * GRAVITY; // N, 18639
const STATIC_CORNER_LOAD = WEIGHT / 4; // N, ~4660

/**
 * Centre of mass, 20 mm AHEAD of the wheelbase centre.
 *
 * The sign of that 20 mm is not a taste call, it is a stability constraint, and
 * it is worth spelling out because it looks like a rounding error.
 *
 * Drop one rear wheel into a hole deep enough that it finds nothing and the car
 * is a three-legged table standing on FL, FR and RR. A three-point support only
 * holds if the centre of mass projects INSIDE that triangle, and the triangle's
 * hypotenuse is the FL-RR diagonal, which for a symmetric track runs exactly
 * through (x = 0, z = 0). So with the mass even a centimetre behind the
 * wheelbase centre the projection falls outside, the front-right wheel is
 * required to pull DOWN to balance the moment, and since a suspension can only
 * push, the car simply tips onto the diagonal instead. Measured: it lay over at
 * 17.4 degrees with the front-right wheel in the air and the body resting on
 * the lip of the hole. No amount of spring, damper or bar rate changes that —
 * a three-point solution is statically determinate, so only the geometry and
 * the centre of mass appear in it.
 *
 * Solving the same three equations forwards, the front-right corner load with
 * the mass at z is W*z/(2*WHEELBASE/2) = 7169*z newtons, so +20 mm buys about
 * 143 N of static contact: light, but the right side of zero, and it settles at
 * 6.7 degrees with all three wheels down instead of 17.4 with two.
 *
 * Moving it forward is also the more honest number for the vehicle this is:
 * a cast-iron V6, a solid front axle, a winch and a bull bar all live ahead of
 * the front wheels, and real Wrangler-class 4x4s weigh in slightly nose-heavy.
 * At 50.8/49.2 it is barely off neutral, and the cornering behaviour it was
 * originally biased rearward to get is unchanged in measurement: 0.676 g at the
 * limit against 0.673, and a 23.9 m steady-state radius against 24.1.
 *
 * COM_Y comes down from 0.62 to 0.55 for the same scenario. Height plays no
 * part in the static test above, but it is the whole of the dynamic one: the
 * mass sits above the diagonal it is tipping about, so as the car leans, its
 * projection walks further out at 0.55 * sin(roll) and the tip runs away from
 * the small static margin. Lowering it shrinks that lever enough that the car
 * converges on the three-wheel stance instead of oscillating past it, and it
 * leaves the static rollover threshold at a still-tall-4x4 1.45 g.
 */
const COM_Z = 0.02;
const COM_Y = 0.55;
/** Share of the static weight carried by the front axle: 0.5077 -> 51/49 F/R. */
const FRONT_WEIGHT_FRACTION = (COM_Z + WHEELBASE / 2) / WHEELBASE;
/** Static load on one front / one rear corner, N. 4731.4 / 4588.1. */
const FRONT_CORNER_LOAD = (WEIGHT * FRONT_WEIGHT_FRACTION) / 2;
const REAR_CORNER_LOAD = (WEIGHT * (1 - FRONT_WEIGHT_FRACTION)) / 2;

/**
 * Strut geometry.
 *
 * `restLength` is the whole stroke: the strut can never be shorter than zero
 * (the hub reaching the attach point) so the usable travel and the rest length
 * are the same number. Of that stroke the car uses `STATIC_SAG` sitting still,
 * which leaves 150 mm of droop for articulation and 400 mm of compression for
 * landings.
 *
 * The compression figure is sized straight off the energy sum, not off a feel.
 * A 5.3 m/s touchdown carries 26.4 kJ, and gravity adds another 5.6 kJ while
 * the car settles through the stroke: 32 kJ to be absorbed across four corners.
 * Force x distance is all there is, so with a peak corner load capped at 10x
 * static (46.6 kN — the point past which a landing stops reading as a squash
 * and starts reading as a hit) the four struts need roughly
 *   32 kJ / (4 x 46.6 kN) = 172 mm  of stroke at the *peak* force,
 * which a real, non-rectangular force curve turns into something near 280 mm.
 * 300 mm left nothing in reserve and put every jump hard into the stops; 400 mm
 * lands the moderate jump at ~91% of travel with the stop doing real work but
 * never running out. Beyond ~400 mm the chassis floor becomes the binding
 * constraint instead (see `colliderOffset`), so this is the end of that road.
 *
 * `ATTACH_Y` is then chosen so that at the design sag the body origin sits
 * exactly at ground level:
 *   hubY = -restLength + sag  ->  attachY = WHEEL_RADIUS + restLength - sag
 */
const SUSPENSION_REST_LENGTH = 0.55; // m
const SUSPENSION_TRAVEL = 0.55; // m — the full stroke: 150 mm droop + 400 mm bump
const STATIC_SAG = 0.15; // m — 27% of travel
const ATTACH_Y = WHEEL_RADIUS + SUSPENSION_REST_LENGTH - STATIC_SAG; // 0.80

/**
 * Spring rates are *derived* from the design sag rather than typed in, so the
 * two axles cannot drift apart. k = cornerLoad / sag gives an identical ride
 * frequency at both ends, f_n = sqrt(g/sag)/2pi = 1.29 Hz, which is exactly the
 * soft-but-not-wallowy figure a long-travel 4x4 wants.
 *   front 31543 N/m, rear 30587 N/m.
 */
const FRONT_SPRING_RATE = FRONT_CORNER_LOAD / STATIC_SAG;
const REAR_SPRING_RATE = REAR_CORNER_LOAD / STATIC_SAG;
/** Critical damping at the corner, 2*sqrt(k*m). 7479 / 8203 N·s/m. */
const FRONT_CRITICAL_DAMPING = 2 * Math.sqrt((FRONT_SPRING_RATE * FRONT_CORNER_LOAD) / GRAVITY);
const REAR_CRITICAL_DAMPING = 2 * Math.sqrt((REAR_SPRING_RATE * REAR_CORNER_LOAD) / GRAVITY);
/**
 * Damping ratios. Bump stays light so the wheel can climb a rock without
 * spiking the chassis; rebound is well above it, which is the single setting
 * that decides whether the car pogos after a landing or plants.
 * Below ~0.40 of critical rebound it pogos; above ~0.70 it lands dead and skitters.
 *
 * Bump sits at 0.38 rather than the 0.32 it ran before. A damper is the only
 * element that can take energy out at the START of the stroke, where the coil
 * is still soft and the bump stop is not awake yet, and it does so without
 * adding to the peak load — damper force peaks on touchdown (high shaft speed,
 * low displacement) while spring force peaks at the bottom (high displacement,
 * zero shaft speed), so the two never stack. Sweeping it against the washboard
 * probe, 0.32 -> 0.38 costs 1% of body-acceleration RMS over chatter and buys
 * 2 points of landing travel and 0.4x of peak corner load; 0.45 buys a little
 * more of both and costs 6% of ride, which is the wrong side of the trade.
 */
const BUMP_DAMPING_RATIO = 0.38;
const REBOUND_DAMPING_RATIO = 0.6;

/**
 * Design ride height, m: the ground clearance under the chassis collider floor
 * with the car at rest on flat ground. The body ORIGIN sits at ground level in
 * the design pose (that is what ATTACH_Y above is chosen for), so this is just
 * the collider's own floor height — and a measured clearance that disagrees
 * with it means the springs, the mass or the strut geometry have drifted apart.
 * Exported so the test harness and any level-design tooling agree on it.
 */
export const DESIGN_RIDE_HEIGHT = 0.95 - 0.5; // colliderOffset.y - halfExtents.y = 0.45

/** Surface friction and shape factors. */
export const SURFACES: SurfaceTable = {
  // Hard-packed dirt: the reference surface. Decent grip, forgiving tail.
  dirt: { friction: 0.85, Bx: 11.0, Cx: 1.42, Ex: 0.9, By: 8.5, Cy: 1.34, Ey: 0.88, rollingResistance: 0.03 },
  // Grass: slightly slipperier than dirt, similar feel.
  grass: { friction: 0.75, Bx: 9.5, Cx: 1.36, Ex: 0.88, By: 7.4, Cy: 1.3, Ey: 0.86, rollingResistance: 0.04 },
  // Rock: the grippiest surface and the only one with a real cliff past the peak.
  rock: { friction: 1.0, Bx: 13.5, Cx: 1.55, Ex: 0.94, By: 10.5, Cy: 1.46, Ey: 0.92, rollingResistance: 0.02 },
  // Gravel: low peak, loose, but slides predictably — the classic rally surface.
  gravel: { friction: 0.7, Bx: 8.5, Cx: 1.3, Ex: 0.85, By: 6.6, Cy: 1.24, Ey: 0.82, rollingResistance: 0.035 },
  // Sand: very low B (you slide a long way before the peak) and C near 1 (no cliff).
  sand: { friction: 0.6, Bx: 6.0, Cx: 1.18, Ex: 0.8, By: 4.6, Cy: 1.14, Ey: 0.78, rollingResistance: 0.1 },
  // Mud: greasy, and it drags hard on the wheels.
  mud: { friction: 0.5, Bx: 6.8, Cx: 1.2, Ex: 0.8, By: 5.2, Cy: 1.16, Ey: 0.78, rollingResistance: 0.08 },
  // Snow: the lowest peak and the gentlest falloff of all — long, lazy slides.
  snow: { friction: 0.35, Bx: 5.2, Cx: 1.12, Ex: 0.76, By: 4.0, Cy: 1.1, Ey: 0.74, rollingResistance: 0.05 },
};

export const JEEP_TUNING: VehicleTuning = {
  chassis: {
    mass: MASS,
    // The collider is deliberately smaller than the bodywork: 1.64 m wide,
    // 3.70 m long, 1.00 m tall. A Wrangler's approach and departure angles come
    // from short overhangs and high bumpers, and a box drawn round the visual
    // body would drag its corners on every landing that the suspension is
    // perfectly capable of absorbing. This box is the *hard* part of the car.
    halfExtents: { x: 0.82, y: 0.5, z: 1.85 },
    // Body origin sits on the ground in the design pose, so the collider box
    // centre is 0.95 m up: floor at 0.45 m clearance, roofline at 1.45 m. The
    // floor clearance has to exceed what the suspension can compress (300 mm)
    // or a big landing puts the belly on the ground before the bump stops are
    // even awake.
    colliderOffset: { x: 0, y: 0.95, z: 0 },
    // 0.55 m up (static rollover threshold = 0.8/0.55 = 1.45 g — still a tall
    // 4x4) and 20 mm ahead of the wheelbase centre, giving 51/49 F/R.
    centreOfMass: { x: 0, y: COM_Y, z: COM_Z },
    // pitch / yaw / roll. Roll is well under the uniform-box value (1010) since
    // the drivetrain and chassis rails carry the mass low.
    inertia: { x: 2900, y: 2800, z: 850 },
    linearDamping: 0,
    angularDamping: 0.12,
    colliderFriction: 0.45,
    colliderRestitution: 0.05,
  },

  wheels: [
    {
      name: 'FL',
      attach: { x: -TRACK / 2, y: ATTACH_Y, z: WHEELBASE / 2 },
      axle: 'front',
      steered: true,
      driven: true,
      handbrakeShare: 0,
      steerSign: 1,
    },
    {
      name: 'FR',
      attach: { x: TRACK / 2, y: ATTACH_Y, z: WHEELBASE / 2 },
      axle: 'front',
      steered: true,
      driven: true,
      handbrakeShare: 0,
      steerSign: 1,
    },
    {
      name: 'RL',
      attach: { x: -TRACK / 2, y: ATTACH_Y, z: -WHEELBASE / 2 },
      axle: 'rear',
      steered: false,
      driven: true,
      handbrakeShare: 1,
      steerSign: 0,
    },
    {
      name: 'RR',
      attach: { x: TRACK / 2, y: ATTACH_Y, z: -WHEELBASE / 2 },
      axle: 'rear',
      steered: false,
      driven: true,
      handbrakeShare: 1,
      steerSign: 0,
    },
  ],

  suspension: {
    restLength: SUSPENSION_REST_LENGTH,
    travel: SUSPENSION_TRAVEL,
    // The stop bites over the last 55% of the stroke: after 247 mm of travel,
    // which is 97 mm of coil compression past the static ride height, and it
    // then has 302 mm of range to work in.
    //
    // That long, early, soft stop is the whole trick, and it is the opposite of
    // what the short one before it did. A quadratic stop delivers most of its
    // force in the last few centimetres, so a short stiff one (126 mm at
    // 2.5e6) is a wall: it spiked the peak corner load to 16x static and still
    // could not absorb the landing, because energy is force x DISTANCE and it
    // had no distance. Spreading the same job over 302 mm at a seventh of the
    // rate turns the wall into a long squash — same energy, 9.4x peak instead
    // of 16x — and because it starts 97 mm past static, well outside anything
    // chatter or cornering reaches, it is still a last resort and not a
    // routine part of the ride.
    bumpStopZone: 0.45,
    // 3.8e5 * 0.302^2 = 35 kN at the very end of the stroke, on top of 17 kN of
    // coil. Swept against the jump test: softer runs out of travel (3.5e5 puts
    // the moderate jump at 91.4%), stiffer wins nothing back and pushes the
    // peak load through the 10x ceiling again (4.8e5 -> 10.2x).
    bumpStopStiffness: 3.8e5,
    // Urethane, not steel. Half the energy that goes into the stop never comes
    // back out, which is what stops a bottomed-out landing firing the car
    // straight back into the air — see the field docs. This and the linear
    // rebound curve below are between them the whole of the no-pogo fix, and
    // neither of them touches the compression side that ride quality lives on.
    bumpStopHysteresis: 0.5,
    bumpStopHysteresisVelocity: 0.25,
    // Digressive past 0.6 m/s on the BUMP side only, so a square-edged rock can
    // push the wheel up without the damper spiking the whole chassis. Rebound
    // stays linear: a digressive rebound curve is exactly what let the car fire
    // itself off the ground after bottoming out, and it buys nothing over
    // chatter, where the input that matters is the wheel being driven up.
    damperKneeVelocity: 0.6,
    damperDigressiveRatio: 0.7,
    reboundDigressiveRatio: 1.0,
    maxDamperForce: 26000,
    maxSuspensionForce: 120000,
    front: {
      // 4731.4 N static corner load / 0.15 m sag = 31543 N/m, f_n = 1.29 Hz.
      springRate: FRONT_SPRING_RATE,
      // c_crit = 7801. Bump 2964 N·s/m, rebound 4681 N·s/m.
      bumpDamping: FRONT_CRITICAL_DAMPING * BUMP_DAMPING_RATIO,
      reboundDamping: FRONT_CRITICAL_DAMPING * REBOUND_DAMPING_RATIO,
      antiRollRate: 9000,
    },
    rear: {
      // 4588.1 N static corner load / 0.15 m sag = 30587 N/m, f_n = 1.29 Hz.
      springRate: REAR_SPRING_RATE,
      // c_crit = 7565. Bump 2875 N·s/m, rebound 4539 N·s/m.
      bumpDamping: REAR_CRITICAL_DAMPING * BUMP_DAMPING_RATIO,
      reboundDamping: REAR_CRITICAL_DAMPING * REBOUND_DAMPING_RATIO,
      antiRollRate: 6500,
    },
  },

  tire: {
    radius: WHEEL_RADIUS,
    width: 0.285,
    // 33x12.5 tyre on a steel wheel, ~32 kg: I ~= 0.6*m*r^2.
    inertia: 2.0,
    nominalLoad: STATIC_CORNER_LOAD,
    loadSensitivity: 0.1,
    loadSensitivityMin: 0.55,
    loadSensitivityMax: 1.3,
    relaxLengthLong: 0.25,
    relaxLengthLat: 0.35,
    relaxMinSpeed: 0.5,
    lowSpeedDamping: 9000,
    lowSpeedDampingFade: 2.5,
    maxSlipRatio: 4,
    maxTanSlipAngle: 4,
    gripScale: 1.0,
    longGripScale: 1.0,
    latGripScale: 1.0,
    maxSpin: 420,
    freeSpinDamping: 0.35,
  },

  surfaces: SURFACES,

  drivetrain: {
    engine: {
      // 3.6 L naturally-aspirated V6: fat low-mid torque, 285 hp up top.
      torqueCurve: [
        [0, 150],
        [800, 205],
        [1200, 262],
        [1600, 302],
        [2000, 326],
        [2400, 341],
        [2800, 351],
        [3200, 357],
        [3600, 360],
        [4000, 358],
        [4400, 350],
        [4800, 338],
        [5200, 322],
        [5600, 304],
        [6000, 284],
        [6400, 258],
        [7000, 200],
      ],
      inertia: 0.35,
      idleRpm: 760,
      stallRpm: 380,
      idleGovernorGain: 0.0016,
      idleMaxThrottle: 0.32,
      redlineRpm: 6200,
      limiterBandRpm: 180,
      limiterCutTime: 0.06,
      maxRpm: 7200,
      frictionConstant: 14,
      frictionViscous: 0.09,
      frictionThrottleRelief: 0.25,
    },
    clutch: {
      capacity: 780,
      stiffness: 150,
      launchRpm: 2600,
      launchBandRpm: 700,
      lockRpm: 1150,
    },
    transmission: {
      // 5-speed automatic.
      gearRatios: [3.59, 2.19, 1.41, 1.0, 0.84],
      reverseRatio: 3.16,
      finalDrive: 4.1,
      transferHigh: 1.0,
      transferLow: 2.72,
      efficiency: 0.9,
      shiftTime: 0.28,
      shiftCooldown: 0.55,
      upshiftRpmLow: 2350,
      upshiftRpmHigh: 5750,
      downshiftRpmLow: 1150,
      // Must stay below (upshift rpm x next ratio / current ratio) for EVERY
      // ratio step, or the box hunts: the tightest step here is 1st->2nd, which
      // drops 5750 rpm to 5750*2.19/3.59 = 3507. `Drivetrain` also refuses any
      // downshift that would immediately trip the upshift, so a future ratio
      // change cannot reintroduce the hunt, but the calibration should be right
      // on its own.
      downshiftRpmHigh: 3200,
      downshiftRpmCeilingFraction: 0.94,
      automatic: true,
      autoReverse: true,
      autoReverseDelay: 0.35,
      directionChangeSpeed: 1.2,
    },
    // 40/60 front/rear: enough rear bias to rotate on throttle, enough front
    // to claw out of a rut.
    frontTorqueSplit: 0.4,
    // `slipReference` is the sharpness of the lock, and it is the one number in
    // the driveline you can feel without knowing why. The bias gain at zero
    // slip is maxBias/slipReference: at 2 rad/s that was 450 N·m per rad/s on
    // the front axle, high enough that float-level asymmetry at a standing
    // start grew into a measurable pull (0.8 deg of heading over a 2 km run).
    // At 5 rad/s the diff still locks completely for a wheel in the air — that
    // wheel is 30+ rad/s adrift — but engages progressively under power.
    centreDiff: { preload: 90, torqueSensitivity: 0.35, maxBias: 700, slipReference: 6.0 },
    frontDiff: { preload: 70, torqueSensitivity: 0.4, maxBias: 900, slipReference: 5.0 },
    rearDiff: { preload: 110, torqueSensitivity: 0.5, maxBias: 1200, slipReference: 5.0 },
  },

  brakes: {
    // 6500 N·m total gives ~0.85 g of retardation before the tyres let go, so
    // the tyre model — not an arbitrary cap — decides the stopping distance.
    maxTorque: 6500,
    frontBias: 0.62,
    handbrakeTorque: 2600,
  },

  steering: {
    maxAngleLowSpeed: 0.6632, // 38 deg
    maxAngleHighSpeed: 0.1396, // 8 deg
    speedFalloffRef: 14,
    rateLowSpeed: 3.6,
    rateHighSpeed: 1.5,
    rateFalloffRef: 22,
    returnRateScale: 1.6,
    ackermann: 0.75,
    wheelbase: WHEELBASE,
    frontTrack: TRACK,
    counterSteerGain: 0.25,
    counterSteerMaxAngle: 0.12,
    counterSteerDeadzone: 0.035,
    counterSteerMinSpeed: 5,
    counterSteerFullSpeed: 12,
  },

  aero: {
    airDensity: 1.225,
    // Cd·A = 1.89. A lifted 4x4 on 33s with a roof rack and mirrors is a worse
    // brick than the showroom figure; this is what holds the top end at a
    // believable 160 km/h instead of letting 285 hp drag it to 178.
    dragCoefficient: 0.7,
    frontalArea: 2.7,
    downforceCoefficient: 0.35,
    downforceFrontBias: 0.45,
  },

  assists: {
    // Sized off the jump test, not off a feeling. Leaving a 10 degree kicker
    // the front wheels drop off the lip while the rears are still on the ramp,
    // and for the ~0.27 s that the nose is unsupported gravity pitches the car
    // nose-down at 0.87 rad/s — twice what a back-of-envelope estimate gives,
    // and it is all acquired BEFORE the car counts as airborne, so the assist
    // never gets to prevent it. It only ever gets to undo it.
    //
    // Treated as what it is — a second-order attitude controller — the two
    // gains are not free parameters. Damping alone cannot work: a pure rate
    // damper with time constant tau just converts the launch rate into a
    // standing pitch error of omega0 * tau, which at the old 3.5/s was
    // 0.87 * 0.29 = 14 degrees no matter how long the flight lasted. That was
    // exactly the measured failure. What is needed is a spring as well, sized
    // so the response to an initial rate peaks inside the target: for a
    // critically damped system that peak is omega0 / (omega_n * e), so landing
    // a 0.87 rad/s launch inside 4 degrees wants omega_n ~ 4.6 rad/s. Hence
    // damping 9/s (= 2 * omega_n) and an upright gain of 18 — the upright term
    // is scaled by the mean of pitch and roll inertia rather than by pitch
    // inertia alone, so 18 * 1875 / 2900 = 11.6 = omega_n^2 is the number that
    // actually acts. Measured result: 14.3 degrees of landing pitch becomes
    // 3.6, and the front axle stops having to swallow the whole jump on its
    // own. Yaw stays deliberately free so the driver can still point the car.
    //
    // The delay exists only so a wheel skipping over a crest cannot trigger the
    // assist; 40 ms is long enough for that and short enough that a 0.6 s jump
    // still gets almost all of its flight under control.
    airUprightGain: 18,
    airPitchDamping: 9,
    airYawDamping: 0.25,
    airAssistDelay: 0.04,
    airAssistFade: 0.08,
    // 16 kN·m is 5.5 rad/s^2 on the pitch inertia: enough that the gains above
    // are never clipped at the rates a kicker actually produces, low enough to
    // stay a correction rather than a jet thruster.
    maxAssistTorque: 16000,
    recoverLift: 0.8,
    recoverCooldown: 1.2,
    flippedThreshold: 0.35,
  },
};
