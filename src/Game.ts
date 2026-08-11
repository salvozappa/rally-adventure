import * as THREE from 'three';
import { Engine } from './core/Engine';
import { Input } from './core/Input';
import { ChaseCamera } from './core/ChaseCamera';
import { Hud } from './ui/Hud';
import { createPhysics, FixedStepper, PHYSICS_DT } from './physics/PhysicsWorld';
import { Vehicle } from './physics/Vehicle';
import { JEEP_TUNING } from './physics/VehicleTuning';
import { Terrain } from './world/Terrain';
import { Sky } from './world/Sky';
import { Lighting } from './render/Lighting';
import { RetroPipeline } from './render/RetroPipeline';
import { JeepModel } from './vehicle/JeepModel';
import { Scatter } from './world/Scatter';
import { DriveFx } from './fx/DriveFx';
import { GameAudio } from './audio/GameAudio';
import { Expedition } from './game/Expedition';
import { Recorder } from './core/Recorder';
import type { PhysicsContext } from './types';

/**
 * Wires every subsystem together and owns the frame loop. Subsystems know
 * nothing about each other — everything they need arrives through the
 * interfaces in types.ts.
 */
export class Game {
  private engine!: Engine;
  private input!: Input;
  private physics!: PhysicsContext;
  private stepper = new FixedStepper(PHYSICS_DT);
  private terrain!: Terrain;
  private sky!: Sky;
  private lighting!: Lighting;
  private pipeline!: RetroPipeline;
  private vehicle!: Vehicle;
  private jeep!: JeepModel;
  private camera!: ChaseCamera;
  private hud!: Hud;
  private scatter!: Scatter;
  private fx!: DriveFx;
  private audio!: GameAudio;
  private expedition!: Expedition;
  private recorder!: Recorder;

  private running = false;
  private lastTime = 0;
  private fps = 60;
  private airborneSince = -1;
  private lastGrounded = true;
  private prevVelY = 0;
  /** HUD overlay state to restore when a recording stops. */
  private hudBeforeRecord: { controls: boolean; telemetry: boolean } | null = null;

  private surfaceAt = (x: number, z: number) => this.terrain.surfaceAt(x, z);

  async init(): Promise<void> {
    this.engine = new Engine();
    this.input = new Input();
    this.physics = await createPhysics();

    // Before anything builds a texture: the pipeline is the only thing that
    // knows the driver's real anisotropy limit, and it hands that to the
    // texture library in its constructor.
    this.pipeline = new RetroPipeline(this.engine);

    this.terrain = new Terrain(this.physics, { seed: 20260807 });
    this.engine.scene.add(this.terrain.object3d);

    // The sky writes into the pipeline's own atmosphere uniforms rather than a
    // private set. Without this the fog pass and the god rays run against the
    // module defaults — a warm beige horizon and a sun in the wrong quarter of
    // the sky — while the dome overhead draws the real thing, and every metre
    // of distance drags the landscape toward a colour the sky never had.
    this.sky = new Sky(this.engine.scene, this.pipeline.atmosphere, {
      onChange: (s) => this.applySkyState(s),
    });
    this.applySkyState(this.sky.state);

    this.lighting = new Lighting(this.engine.scene, this.sky);

    this.scatter = new Scatter(this.terrain, this.physics);
    this.engine.scene.add(this.scatter.object3d);

    const spawn = this.terrain.getSpawnPoint();
    this.vehicle = new Vehicle(this.physics, JEEP_TUNING, spawn);

    this.jeep = new JeepModel();
    this.engine.scene.add(this.jeep.object3d);

    this.camera = new ChaseCamera(this.engine.camera, this.terrain);
    this.hud = new Hud();
    this.fx = new DriveFx(this.engine.scene, { terrain: this.terrain });
    this.audio = new GameAudio();

    // Free-roam objective layer: viewpoints to discover, plus driving stats.
    this.expedition = new Expedition(this.terrain, { seed: 0xC0FFEE });
    this.engine.scene.add(this.expedition.object3d);
    this.expedition.onDiscover = (w, stats) => {
      this.hud.notify(`${w.name} FOUND  ${stats.found}/${stats.total}`, 2.6);
    };
    this.expedition.onStunt = (kind, value) => {
      const v = kind === 'DRIFT' ? `${value.toFixed(0)}m` : `${value.toFixed(1)}s`;
      this.hud.notify(`${kind} ${v}`, 1.6);
    };

    // Records the canvas plus the game's own audio mix. Hides the HUD overlays
    // while rolling — a control legend and a frame counter are the two things
    // you least want in footage meant to show the game off.
    this.recorder = new Recorder(this.engine.canvas, () => this.audio.captureStream());
    this.recorder.onChange((rec) => {
      this.hud.recording = rec;
      if (rec) {
        this.hudBeforeRecord = {
          controls: this.hud.showControls,
          telemetry: this.hud.showTelemetry,
        };
        this.hud.showControls = false;
        this.hud.showTelemetry = false;
      } else if (this.hudBeforeRecord) {
        this.hud.showControls = this.hudBeforeRecord.controls;
        this.hud.showTelemetry = this.hudBeforeRecord.telemetry;
        this.hudBeforeRecord = null;
        const f = this.recorder.lastFilename;
        if (f) {
          this.hud.notify(
            this.recorder.capturedAudio ? `SAVED ${f}` : `SAVED ${f} (NO AUDIO)`,
            3.5,
          );
        }
      }
    });

    this.engine.onResize((w, h) => this.pipeline.resize(w, h));
    this.bindHotkeys();

    // No boot banner: the control panel is on screen permanently now, so a
    // transient copy of the same text just covers the view for five seconds.
  }

  /**
   * Push the sky's derived numbers into the passes that have to agree with it.
   * Fires on construction and on every time-of-day or weather change.
   *
   * Fog density in particular: the pipeline ships with a placeholder that is
   * three times the sky's, which puts a hill 200 m away at 78% haze instead of
   * 32% and flattens the whole landscape into one tone.
   */
  private applySkyState(s: import('./world/Sky').SkyState): void {
    this.pipeline.setOption('fogDensity', s.fogDensity);
    this.pipeline.setOption('fogHeightFalloff', s.fogHeightFalloff);
    this.pipeline.setOption('fogStart', s.fogStart);
  }

  private bindHotkeys(): void {
    window.addEventListener('keydown', (e) => {
      if (e.repeat) return;
      switch (e.code) {
        case 'KeyC':
          this.hud.notify(`CAMERA: ${this.camera.cycleMode().toUpperCase()}`, 1.2);
          break;
        case 'KeyT':
          this.sky.cycleTimeOfDay?.();
          break;
        case 'KeyP':
          this.pipeline.setOption('enabled', !this.pipeline.options.enabled);
          this.hud.notify(`RETRO FX: ${this.pipeline.options.enabled ? 'ON' : 'OFF'}`, 1.2);
          break;
        case 'F3':
          this.hud.showTelemetry = !this.hud.showTelemetry;
          break;
        case 'KeyH':
          this.hud.showControls = !this.hud.showControls;
          break;
        case 'KeyV':
          this.recorder.toggle();
          break;
      }
    });
    // Browsers require a gesture before audio may start.
    const unlock = () => {
      void this.audio.resume();
      window.removeEventListener('pointerdown', unlock);
      window.removeEventListener('keydown', unlock);
    };
    window.addEventListener('pointerdown', unlock);
    window.addEventListener('keydown', unlock);
  }

  start(): void {
    this.running = true;
    this.lastTime = performance.now();
    this.schedule();
  }

  /**
   * Drive the loop with rAF, falling back to a timer whenever the page is
   * hidden.
   *
   * Chrome stops firing rAF entirely for a background tab, which freezes the
   * simulation on whatever frame it happened to be showing. For a player that
   * is the right behaviour, but it makes the game untestable: any automated
   * screenshot comes back as a stale frame from just after boot — before the
   * chase camera has even converged on the car — which looks exactly like a
   * rendering bug and is maddening to chase. Keeping a slow timer alive costs
   * nothing and means what a capture sees is what the game is doing.
   */
  private schedule(): void {
    if (!this.running) return;
    if (typeof document !== 'undefined' && document.hidden) {
      setTimeout(() => this.frame(performance.now()), 16);
    } else {
      requestAnimationFrame(this.frame);
    }
  }

  private frame = (now: number): void => {
    if (!this.running) return;
    this.schedule();

    const rawDt = (now - this.lastTime) / 1000;
    this.lastTime = now;
    // Clamp so an alt-tab doesn't teleport the world on return.
    const dt = Math.min(rawDt, 0.1);
    this.fps += (1 / Math.max(rawDt, 1e-4) - this.fps) * 0.08;

    this.input.update(dt);

    const alpha = this.stepper.advance(dt, (fdt) => {
      this.vehicle.fixedUpdate(fdt, this.input.state, this.surfaceAt);
      this.physics.world.step();
    });

    const s = this.vehicle.state;

    // Interpolate the visual transform between physics steps so a 120Hz sim
    // renders smoothly at any refresh rate.
    this.vehicle.getInterpolated(alpha, s.position, s.quaternion);

    this.detectEvents(dt, s);

    this.jeep.update(s, dt);
    this.camera.update(dt, s);
    this.terrain.update(this.engine.camera.position);
    this.scatter.update(this.engine.camera.position);
    this.sky.update(dt, this.engine.camera.position);
    this.lighting.update(dt, s.position);
    this.fx.update(dt, s);
    this.audio.update(dt, s);
    this.expedition.update(dt, s);
    this.hud.update(dt, s, this.fps);

    this.pipeline.render(dt);
  };

  /** Jump/landing detection, used for camera shake, audio and HUD flavour. */
  private detectEvents(dt: number, s: import('./types').VehicleState): void {
    const grounded = !s.airborne;
    if (!grounded && this.lastGrounded) {
      this.airborneSince = performance.now();
    } else if (grounded && !this.lastGrounded) {
      const air = this.airborneSince > 0 ? (performance.now() - this.airborneSince) / 1000 : 0;
      // Landing severity comes from how much downward speed we just killed.
      const impact = Math.max(0, -this.prevVelY);
      if (air > 0.35) {
        this.camera.addShake(Math.min(1.6, impact * 0.09));
        this.audio.playLanding(impact);
        if (air > 0.8) this.hud.notify(`AIRTIME ${air.toFixed(1)}s`, 1.6);
      }
      this.airborneSince = -1;
    }
    this.lastGrounded = grounded;
    this.prevVelY = s.velocity.y;

    if (this.input.state.recover) {
      this.vehicle.recover?.();
      this.camera.reset();
    }
  }

  dispose(): void {
    this.running = false;
    this.hud.dispose();
    this.expedition.dispose();
    this.fx.dispose();
    this.audio.dispose();
    this.jeep.dispose();
    this.vehicle.dispose();
    this.scatter.dispose();
    this.terrain.dispose();
    this.lighting.dispose();
    this.sky.dispose();
    this.pipeline.dispose();
  }
}

// Re-exported for the preview pages.
export { THREE };
