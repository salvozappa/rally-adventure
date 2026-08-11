import type { DriveInput } from '../types';

/**
 * Keyboard + gamepad driver input with analogue ramping on the digital keys, so
 * a keyboard still gives smooth steering instead of instant lock-to-lock.
 */
export class Input {
  readonly state: DriveInput = {
    steer: 0,
    throttle: 0,
    brake: 0,
    handbrake: 0,
    shiftUp: false,
    shiftDown: false,
    recover: false,
  };

  /** Raw key set. */
  private keys = new Set<string>();
  private prevKeys = new Set<string>();
  private padIndex: number | null = null;
  /** Held steering position for keyboard ramping. */
  private steerHeld = 0;

  /** Speed-sensitive steering is applied by the vehicle, not here. */
  constructor(private target: EventTarget = window) {
    this.target.addEventListener('keydown', this.onKeyDown as EventListener);
    this.target.addEventListener('keyup', this.onKeyUp as EventListener);
    window.addEventListener('gamepadconnected', this.onPadConnect as EventListener);
    window.addEventListener('gamepaddisconnected', this.onPadDisconnect as EventListener);
    // Drop held keys when focus leaves, so alt-tabbing away mid-corner does not
    // leave the throttle pinned. Deliberately NOT on visibilitychange: a hidden
    // tab still runs the loop (see Game.schedule), and clearing there would make
    // the car undrivable under automated capture.
    window.addEventListener('blur', () => this.keys.clear());
  }

  private onKeyDown = (e: KeyboardEvent) => {
    if (e.repeat) return;
    this.keys.add(e.code);
    // Stop the page scrolling out from under the game.
    if (SWALLOW.has(e.code)) e.preventDefault();
  };

  private onKeyUp = (e: KeyboardEvent) => {
    this.keys.delete(e.code);
    if (SWALLOW.has(e.code)) e.preventDefault();
  };

  private onPadConnect = (e: GamepadEvent) => {
    this.padIndex = e.gamepad.index;
  };

  private onPadDisconnect = (e: GamepadEvent) => {
    if (this.padIndex === e.gamepad.index) this.padIndex = null;
  };

  private down(...codes: string[]): boolean {
    return codes.some((c) => this.keys.has(c));
  }

  private pressed(...codes: string[]): boolean {
    return codes.some((c) => this.keys.has(c) && !this.prevKeys.has(c));
  }

  /** Call once per frame before reading `state`. */
  update(dt: number): void {
    const s = this.state;
    const pad = this.padIndex !== null ? navigator.getGamepads()[this.padIndex] : null;

    let steerTarget = 0;
    if (this.down('KeyA', 'ArrowLeft')) steerTarget -= 1;
    if (this.down('KeyD', 'ArrowRight')) steerTarget += 1;

    // Ramp toward the held direction; snap back to centre roughly twice as fast
    // as we ramp out, which is what makes keyboard rally driving feel alive.
    const RAMP_IN = 3.6; // units/sec
    const RAMP_OUT = 7.0;
    if (steerTarget !== 0) {
      // Crossing the centre counts as returning, so it stays responsive.
      const rate = Math.sign(steerTarget) === Math.sign(this.steerHeld) || this.steerHeld === 0 ? RAMP_IN : RAMP_OUT;
      this.steerHeld += Math.sign(steerTarget - this.steerHeld) * rate * dt;
      this.steerHeld = clamp(this.steerHeld, -1, 1);
    } else {
      const mag = Math.max(0, Math.abs(this.steerHeld) - RAMP_OUT * dt);
      this.steerHeld = mag * Math.sign(this.steerHeld);
    }
    s.steer = this.steerHeld;

    s.throttle = this.down('KeyW', 'ArrowUp') ? 1 : 0;
    s.brake = this.down('KeyS', 'ArrowDown') ? 1 : 0;
    s.handbrake = this.down('Space') ? 1 : 0;
    s.shiftUp = this.pressed('KeyE', 'ShiftLeft');
    s.shiftDown = this.pressed('KeyQ', 'ControlLeft');
    s.recover = this.pressed('KeyR');

    if (pad) {
      const ax = applyDeadzone(pad.axes[0] ?? 0, 0.12);
      if (Math.abs(ax) > Math.abs(s.steer)) s.steer = ax;
      // Triggers: standard mapping puts them on buttons 6/7 with analogue value.
      const rt = pad.buttons[7]?.value ?? 0;
      const lt = pad.buttons[6]?.value ?? 0;
      s.throttle = Math.max(s.throttle, rt);
      s.brake = Math.max(s.brake, lt);
      s.handbrake = Math.max(s.handbrake, pad.buttons[0]?.pressed ? 1 : 0);
      s.shiftUp ||= pad.buttons[5]?.pressed ?? false;
      s.shiftDown ||= pad.buttons[4]?.pressed ?? false;
      s.recover ||= pad.buttons[3]?.pressed ?? false;
    }

    this.prevKeys = new Set(this.keys);
  }

  /** True on the frame a key goes down — for UI toggles outside the drive loop. */
  wasPressed(code: string): boolean {
    return this.keys.has(code) && !this.prevKeys.has(code);
  }

  isDown(code: string): boolean {
    return this.keys.has(code);
  }
}

const SWALLOW = new Set([
  'Space',
  'ArrowUp',
  'ArrowDown',
  'ArrowLeft',
  'ArrowRight',
  'Tab',
]);

function clamp(v: number, lo: number, hi: number) {
  return v < lo ? lo : v > hi ? hi : v;
}

function applyDeadzone(v: number, dz: number) {
  const a = Math.abs(v);
  if (a < dz) return 0;
  return Math.sign(v) * ((a - dz) / (1 - dz));
}
