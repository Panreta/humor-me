// Pure math for the oscilloscope line, kept apart from the canvas so it can be checked under plain Node.

export const HUE = 145; // phosphor green; the voice moves shades only, never the hue
const CENTROID_LOW = 300; // Hz; a low, dull voice
const CENTROID_HIGH = 3000; // Hz; a bright, hissy one
const SILENT_RMS = 0.004;
const PEAK_FLOOR = 0.03; // keeps room noise from being blown up to full height
const PEAK_RELEASE = 0.995; // per frame, about a second and a half to forget a loud moment

export const clamp01 = (x: number) => Math.min(1, Math.max(0, x));

// Exponential smoothing; k near 0 is calm, 1 is no smoothing.
export const smooth = (prev: number, next: number, k: number) => prev + (next - prev) * k;

export function rms(samples: ArrayLike<number>): number {
  let sum = 0;
  for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
  return samples.length ? Math.sqrt(sum / samples.length) : 0;
}

export const isSilent = (level: number) => level < SILENT_RMS;

// Spectral centroid (Hz) of analyser dB bins: where the voice's energy sits, a cheap stand-in for pitch and brightness.
export function centroidHz(db: ArrayLike<number>, sampleRate: number): number {
  let weighted = 0;
  let total = 0;
  const binHz = sampleRate / 2 / db.length;
  for (let i = 1; i < db.length; i++) {
    const magnitude = Number.isFinite(db[i]) ? 10 ** (db[i] / 20) : 0;
    weighted += magnitude * i * binHz;
    total += magnitude;
  }
  return total > 0 ? weighted / total : 0;
}

// 0 (dull) to 1 (bright) on a log scale, because pitch and brightness are heard in ratios.
export const brightness = (hz: number) =>
  clamp01(Math.log(Math.max(hz, 1) / CENTROID_LOW) / Math.log(CENTROID_HIGH / CENTROID_LOW));

// Slow peak envelope (a simple AGC): rises at once, falls slowly, so quiet and loud speech both fill the height.
export const trackPeak = (prev: number, frame: number) => Math.max(frame, prev * PEAK_RELEASE, PEAK_FLOOR);

// Speech sits around 0.02 to 0.3 RMS, so loudness is stretched before it drives glow and amplitude.
export const loudness = (level: number) => clamp01(level * 5);
