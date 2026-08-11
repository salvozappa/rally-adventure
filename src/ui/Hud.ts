import type { VehicleState } from '../types';

/**
 * Retro HUD drawn into a low-resolution 2D canvas and scaled up with
 * nearest-neighbour, so it shares the chunky pixel grid of the 3D pipeline
 * instead of floating over it as crisp modern vector text.
 */
export class Hud {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  /** Logical HUD resolution. Everything is authored against this. */
  private readonly W = 480;
  private readonly H = 270;

  /** Smoothed display values so the needles don't twitch. */
  private dispSpeed = 0;
  private dispRpm = 0;
  private message = '';
  private messageTime = 0;
  /** F3 — frame counter and other developer readouts. Off by default. */
  showTelemetry = false;
  /** H — the control reference. On by default; hide it for clean screenshots. */
  showControls = true;

  constructor(parent: HTMLElement = document.getElementById('ui') ?? document.body) {
    this.canvas = document.createElement('canvas');
    this.canvas.width = this.W;
    this.canvas.height = this.H;
    Object.assign(this.canvas.style, {
      position: 'absolute',
      inset: '0',
      width: '100%',
      height: '100%',
      imageRendering: 'pixelated',
      pointerEvents: 'none',
    } as CSSStyleDeclaration);
    parent.appendChild(this.canvas);
    const ctx = this.canvas.getContext('2d');
    if (!ctx) throw new Error('HUD: 2D context unavailable');
    this.ctx = ctx;
    this.ctx.imageSmoothingEnabled = false;
  }

  /** Show a transient centred message, e.g. "AIRTIME 2.1s". */
  notify(text: string, seconds = 2): void {
    this.message = text;
    this.messageTime = seconds;
  }

  update(dt: number, s: VehicleState, fps: number): void {
    const c = this.ctx;
    const { W, H } = this;
    c.clearRect(0, 0, W, H);

    // Needle smoothing. RPM chases faster than the speedo, which is what real
    // instruments do and what makes gearshifts feel percussive.
    const kmh = s.speed * 3.6;
    this.dispSpeed += (kmh - this.dispSpeed) * (1 - Math.exp(-9 * dt));
    this.dispRpm += (s.engineRpm - this.dispRpm) * (1 - Math.exp(-16 * dt));

    // Bottom-right instrument cluster: a tacho arc with a separate digital
    // readout beneath it. Keeping the numerals out of the dial face is the
    // whole point — overlaid on the ticks they read as part of the scale.
    const dialX = W - 62;
    const dialY = H - 64;
    this.drawTacho(dialX, dialY, 42, s);
    this.drawReadout(dialX, dialY + 50, s);
    this.drawThrottleBars(18, H - 26, s);
    this.drawTiltIndicator(30, H - 74, s);
    if (this.showControls) this.drawControls(8, 8);

    if (this.messageTime > 0) {
      this.messageTime -= dt;
      const a = Math.min(1, this.messageTime * 2.5);
      c.save();
      c.globalAlpha = a;
      this.text(this.message, W / 2, 40, 'center', '#ffe9b0', 1, '#241505');
      c.restore();
    }

    // Developer telemetry only. A permanent frame counter is the single loudest
    // "unfinished build" signal a screenshot can carry, so it stays off unless
    // asked for (F3).
    if (this.showTelemetry) {
      this.text(`${Math.round(fps)} FPS`, 6, 12, 'left', '#8fb08f', 1, '#0d160d');
    }
  }

  // ---------------------------------------------------------------- widgets

  /**
   * Persistent control reference, top-left.
   *
   * Keys sit in their own left column so the eye can scan down them without
   * reading the descriptions — which is the only way a reference like this
   * earns its screen space once you already know half the bindings.
   */
  private drawControls(x: number, y: number): void {
    const c = this.ctx;
    const rows: Array<[string, string]> = [
      ['WASD', 'DRIVE'],
      ['SPACE', 'HANDBRAKE'],
      ['C', 'CAMERA'],
      ['R', 'RECOVER'],
      ['T', 'TIME OF DAY'],
      ['P', 'RETRO FX'],
      ['H', 'HIDE THIS'],
    ];

    const pad = 5;
    const lead = 9;
    const keyCol = 30;
    const w = 96;
    const h = pad * 2 + rows.length * lead;

    c.fillStyle = PANEL_FILL;
    c.fillRect(x, y, w, h);
    c.strokeStyle = PANEL_EDGE;
    c.lineWidth = 1;
    c.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);

    rows.forEach(([key, label], i) => {
      const ty = y + pad + lead * i + 7;
      this.text(key, x + pad, ty, 'left', '#ffd27a', 1, '#241505');
      this.text(label, x + pad + keyCol, ty, 'left', '#9fb2bd', 1, '#10161a');
    });
  }

  private drawTacho(cx: number, cy: number, r: number, s: VehicleState): void {
    const c = this.ctx;
    const START = Math.PI * 0.78;
    const SWEEP = Math.PI * 1.44;
    const maxRpm = 7000;
    const redline = 5800;

    // Dial face: dark translucent disc with a lighter rim.
    c.beginPath();
    c.arc(cx, cy, r, 0, Math.PI * 2);
    c.fillStyle = PANEL_FILL;
    c.fill();
    c.lineWidth = 2;
    c.strokeStyle = PANEL_EDGE;
    c.stroke();

    // Ticks.
    for (let i = 0; i <= 7; i++) {
      const t = i / 7;
      const a = START + SWEEP * t;
      const inner = t * maxRpm >= redline ? r - 11 : r - 8;
      const col = t * maxRpm >= redline ? '#d8452e' : '#9fb2bd';
      c.beginPath();
      c.moveTo(cx + Math.cos(a) * inner, cy + Math.sin(a) * inner);
      c.lineTo(cx + Math.cos(a) * (r - 3), cy + Math.sin(a) * (r - 3));
      c.lineWidth = t * maxRpm >= redline ? 3 : 2;
      c.strokeStyle = col;
      c.stroke();
    }

    // Redline arc.
    c.beginPath();
    c.arc(cx, cy, r - 5, START + SWEEP * (redline / maxRpm), START + SWEEP);
    c.lineWidth = 3;
    c.strokeStyle = '#d8452e';
    c.stroke();

    // Needle.
    const t = Math.min(1, this.dispRpm / maxRpm);
    const a = START + SWEEP * t;
    c.beginPath();
    c.moveTo(cx - Math.cos(a) * 6, cy - Math.sin(a) * 6);
    c.lineTo(cx + Math.cos(a) * (r - 9), cy + Math.sin(a) * (r - 9));
    c.lineWidth = 3;
    c.strokeStyle = this.dispRpm >= redline ? '#ff6a4a' : '#ffd27a';
    c.stroke();
    c.beginPath();
    c.arc(cx, cy, 3.5, 0, Math.PI * 2);
    c.fillStyle = '#e8ddc4';
    c.fill();
  }

  /**
   * Digital speed + gear panel. Laid out as one plate so the two values read as
   * a single instrument, the way a rally dash groups them.
   */
  private drawReadout(cx: number, cy: number, s: VehicleState): void {
    const c = this.ctx;
    const w = 86;
    const h = 26;
    const x = cx - w / 2;
    const y = cy - h / 2;

    c.fillStyle = PANEL_FILL;
    c.fillRect(x, y, w, h);
    c.strokeStyle = PANEL_EDGE;
    c.lineWidth = 1;
    c.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
    // Divider between the speed field and the gear field.
    c.beginPath();
    c.moveTo(x + w - 24, y + 3);
    c.lineTo(x + w - 24, y + h - 3);
    c.strokeStyle = '#2b353d';
    c.stroke();

    const v = Math.round(Math.max(0, this.dispSpeed));
    this.text(String(v), x + w - 30, y + 19, 'right', '#f4e7c8', 3, '#1a1208');
    this.text('KM/H', x + w - 30, y + 25, 'right', '#7d8b95', 1, '#10161a');

    const label = s.gear < 0 ? 'R' : s.gear === 0 ? 'N' : String(s.gear);
    this.text(label, x + w - 12, y + 19, 'center', '#ffd27a', 3, '#241505');
  }

  private drawThrottleBars(x: number, y: number, s: VehicleState): void {
    // Wheel-slip readout: four small bars, one per wheel, filling with the
    // magnitude of combined slip. It's the most useful thing a driver can see.
    const c = this.ctx;
    const w = 7;
    const h = 20;
    for (let i = 0; i < Math.min(4, s.wheels.length); i++) {
      const wh = s.wheels[i];
      const slip = Math.min(1, Math.hypot(wh.slipRatio, Math.sin(wh.slipAngle) * 1.6));
      const bx = x + i * (w + 3);
      c.fillStyle = PANEL_FILL;
      c.fillRect(bx, y - h, w, h);
      const fh = Math.round(slip * h);
      c.fillStyle = !wh.grounded ? '#4a5560' : slip > 0.75 ? '#e05a3a' : slip > 0.4 ? '#e0b23a' : '#6fbf5a';
      c.fillRect(bx, y - fh, w, fh);
      c.strokeStyle = PANEL_EDGE;
      c.lineWidth = 1;
      c.strokeRect(bx + 0.5, y - h + 0.5, w - 1, h - 1);
    }
    this.text('GRIP', x, y + 9, 'left', '#8b9aa4', 1, '#10161a');
  }

  private drawTiltIndicator(cx: number, cy: number, s: VehicleState): void {
    // Roll/pitch bubble — genuinely useful in an off-roader, and it's a nod to
    // the inclinometers real 4x4s carry.
    const c = this.ctx;
    const r = 17;
    c.beginPath();
    c.arc(cx, cy, r, 0, Math.PI * 2);
    c.fillStyle = PANEL_FILL;
    c.fill();
    c.strokeStyle = PANEL_EDGE;
    c.lineWidth = 2;
    c.stroke();

    // Extract roll and pitch from the chassis basis vectors.
    const q = s.quaternion;
    const rx = 1 - 2 * (q.y * q.y + q.z * q.z);
    const ry = 2 * (q.x * q.y + q.z * q.w);
    const fy = 2 * (q.y * q.z - q.x * q.w);
    const roll = Math.asin(clamp(ry, -1, 1));
    const pitch = Math.asin(clamp(fy, -1, 1));
    void rx;

    const bx = cx + clamp(roll / 0.9, -1, 1) * (r - 5);
    const by = cy + clamp(pitch / 0.9, -1, 1) * (r - 5);
    const danger = Math.abs(roll) > 0.62;
    c.beginPath();
    c.arc(bx, by, 3.5, 0, Math.PI * 2);
    c.fillStyle = danger ? '#e04a3a' : '#7fd06a';
    c.fill();

    c.beginPath();
    c.moveTo(cx - r + 4, cy);
    c.lineTo(cx + r - 4, cy);
    c.moveTo(cx, cy - r + 4);
    c.lineTo(cx, cy + r - 4);
    c.strokeStyle = 'rgba(150, 170, 180, 0.35)';
    c.lineWidth = 1;
    c.stroke();
  }

  /**
   * Chunky text with a hard drop shadow. Uses a monospace stack; the pixel grid
   * of the upscaled canvas does most of the retro work regardless of the face.
   */
  private text(
    str: string,
    x: number,
    y: number,
    align: CanvasTextAlign,
    color: string,
    scale: number,
    shadow: string,
  ): void {
    const c = this.ctx;
    c.font = `bold ${7 * scale}px "Courier New", monospace`;
    c.textAlign = align;
    c.textBaseline = 'alphabetic';
    c.fillStyle = shadow;
    c.fillText(str, x + 1, y + 1);
    c.fillStyle = color;
    c.fillText(str, x, y);
  }

  dispose(): void {
    this.canvas.remove();
  }
}

/**
 * One fill and one edge colour for every HUD panel. Mixed opacities let the
 * scene show through some widgets and not others, which reads as several
 * unrelated UI styles sharing a screen.
 */
const PANEL_FILL = 'rgba(11, 15, 19, 0.86)';
const PANEL_EDGE = '#3a4650';

function clamp(v: number, lo: number, hi: number) {
  return v < lo ? lo : v > hi ? hi : v;
}
