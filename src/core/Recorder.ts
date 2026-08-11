/**
 * Records the game canvas — and the game's own audio mix — straight to a file.
 *
 * Recording the canvas rather than the screen means no window chrome, no
 * compositor scaling, no cursor, and a clean fixed frame size. `captureStream`
 * pulls frames from the same buffer the renderer already produced, so the
 * capture costs almost nothing and never tears.
 *
 * The audio is tapped off the game's master bus rather than the speakers, so
 * the recording is identical whether or not anything is actually audible.
 */

export interface RecorderOptions {
  /** Frames per second requested from the canvas. */
  fps?: number;
  /** Video bitrate, bits/sec. 12 Mb/s is generous for 1080p and keeps the
   *  dithering intact — low bitrates smear it into mud, which is exactly the
   *  detail the retro look depends on. */
  videoBitsPerSecond?: number;
  /** Base name for the downloaded file. */
  name?: string;
}

type Listener = (recording: boolean, seconds: number) => void;

export class Recorder {
  private mediaRecorder: MediaRecorder | null = null;
  private chunks: Blob[] = [];
  private startedAt = 0;
  private audioStream: MediaStream | null = null;
  private listeners: Listener[] = [];
  private lastFile: string | null = null;
  private hasAudio = false;

  /** False when the take is video-only because audio was not unlocked. */
  get capturedAudio(): boolean {
    return this.hasAudio;
  }

  constructor(
    private canvas: HTMLCanvasElement,
    private getAudioStream: () => MediaStream | null,
    private opts: RecorderOptions = {},
  ) {}

  static get supported(): boolean {
    return (
      typeof MediaRecorder !== 'undefined' &&
      typeof HTMLCanvasElement.prototype.captureStream === 'function'
    );
  }

  get recording(): boolean {
    return this.mediaRecorder?.state === 'recording';
  }

  /** Seconds elapsed in the current take, 0 when idle. */
  get elapsed(): number {
    return this.recording ? (performance.now() - this.startedAt) / 1000 : 0;
  }

  /** Path of the most recent download, for reporting in the UI. */
  get lastFilename(): string | null {
    return this.lastFile;
  }

  onChange(fn: Listener): void {
    this.listeners.push(fn);
  }

  toggle(): void {
    if (this.recording) this.stop();
    else this.start();
  }

  start(): void {
    if (this.recording || !Recorder.supported) return;

    const fps = this.opts.fps ?? 60;
    const stream = this.canvas.captureStream(fps);

    // Fold the game mix in as a second track. Without this the file is silent,
    // and the engine note is half of why the footage reads as a driving game.
    //
    // `getAudioStream` returns null when the AudioContext has not been unlocked
    // by a user gesture. Attaching a track from a suspended context is worse
    // than having no audio at all: it never delivers a sample, MediaRecorder
    // blocks on it, and the entire file — video included — comes out empty.
    this.audioStream = this.getAudioStream();
    const audioTracks = this.audioStream?.getAudioTracks() ?? [];
    for (const track of audioTracks) stream.addTrack(track);
    this.hasAudio = audioTracks.length > 0;

    const mimeType = pickMimeType();
    if (!mimeType) return;

    this.chunks = [];
    this.mediaRecorder = new MediaRecorder(stream, {
      mimeType,
      videoBitsPerSecond: this.opts.videoBitsPerSecond ?? 12_000_000,
    });
    this.mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) this.chunks.push(e.data);
    };
    this.mediaRecorder.onstop = () => this.save(mimeType);
    // A timeslice keeps chunks flowing, so a crashed tab still leaves a
    // playable partial file rather than nothing at all.
    this.mediaRecorder.start(1000);
    this.startedAt = performance.now();
    this.emit();
  }

  stop(): void {
    if (!this.recording) return;
    this.mediaRecorder?.stop();
    this.emit();
  }

  private save(mimeType: string): void {
    const blob = new Blob(this.chunks, { type: mimeType });
    this.chunks = [];

    const stamp = new Date()
      .toISOString()
      .replace(/[:.]/g, '-')
      .replace('T', '_')
      .slice(0, 19);
    const ext = mimeType.includes('mp4') ? 'mp4' : 'webm';
    const name = `${this.opts.name ?? 'rally-adventure'}_${stamp}.${ext}`;

    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = name;
    a.click();
    // Revoking immediately can cancel the download in some Chrome builds.
    setTimeout(() => URL.revokeObjectURL(url), 60_000);

    this.lastFile = name;
    this.mediaRecorder = null;
    this.emit();
  }

  private emit(): void {
    for (const fn of this.listeners) fn(this.recording, this.elapsed);
  }

  dispose(): void {
    if (this.recording) this.mediaRecorder?.stop();
    this.mediaRecorder = null;
    this.listeners = [];
  }
}

/**
 * Best container/codec this browser will actually accept.
 *
 * VP9 first: it holds the dithered gradients and the fine terrain detail far
 * better than VP8 at the same bitrate, and ffmpeg transcodes it cleanly for a
 * blog. H.264 in MP4 is the fallback where a browser offers it.
 */
function pickMimeType(): string | null {
  const candidates = [
    'video/webm;codecs=vp9,opus',
    'video/webm;codecs=vp9',
    'video/mp4;codecs=avc1.42E01E,mp4a.40.2',
    'video/webm;codecs=vp8,opus',
    'video/webm',
  ];
  for (const c of candidates) {
    if (MediaRecorder.isTypeSupported(c)) return c;
  }
  return null;
}
