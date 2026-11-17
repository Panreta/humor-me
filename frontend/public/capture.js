// The recorder's capture logic, free of browser globals so scripts/test-recorder.mjs runs it under Node. Frames come
// from the microphone track itself (MediaStreamTrackProcessor), stamped by the capture clock. An AudioWorklet is driven
// by Chrome's render thread, which under CPU load simply stops being called: it produced 65% of real time here.
import { Resampler, OUT_RATE } from "/resampler.js";

const CHUNK = 4096; // output samples per message, 256 ms
const TOLERANCE_US = 1000;

export class Capture {
  /** Turns timestamped mono frames into 16 kHz Int16 whose sample index is (timestamp - first timestamp) * 16 kHz. */
  constructor() {
    this.resampler = null;
    this.rate = 0;
    this.firstUs = 0;
    this.fed = 0; // input samples handed to the resampler, silence included
    this.real = 0; // input samples that really arrived
    this.gapSamples = 0;
    this.endUs = 0;
    this.pending = [];
  }

  /** One frame in (timestamp in microseconds on the capture clock); returns the Int16Array chunks now complete. */
  push(timestampUs, rate, samples) {
    if (!this.resampler) {
      this.resampler = new Resampler(rate);
      this.rate = rate;
      this.firstUs = timestampUs;
    }
    const expectedUs = this.firstUs + (this.fed / rate) * 1e6;
    let input = samples;
    if (timestampUs - expectedUs > TOLERANCE_US) {
      // The frames that should be here never came: the file must still put what follows at the right time.
      const missing = Math.round(((timestampUs - expectedUs) * rate) / 1e6);
      this.gapSamples += missing;
      this.take(new Float32Array(missing));
    } else if (expectedUs - timestampUs > TOLERANCE_US) {
      input = samples.subarray(Math.min(samples.length, Math.round(((expectedUs - timestampUs) * rate) / 1e6)));
    }
    this.real += input.length;
    this.endUs = timestampUs + (samples.length / rate) * 1e6;
    this.take(input);
    const out = [];
    while (this.pending.length >= CHUNK) out.push(Int16Array.from(this.pending.splice(0, CHUNK)));
    return out;
  }

  take(input) {
    this.fed += input.length;
    for (const s of this.resampler.push(input)) this.pending.push(Math.max(-1, Math.min(1, s)) * 32767);
  }

  /** Audio actually received against the capture clock's elapsed time; gaps and wall clock show a stall elsewhere. */
  health(wallSeconds) {
    return {
      produced_s: this.real / this.rate,
      elapsed_s: (this.endUs - this.firstUs) / 1e6,
      gap_ms: (this.gapSamples / this.rate) * 1000,
      wall_s: wallSeconds,
      rate: this.rate,
    };
  }
}

function mono(frame) {
  const n = frame.numberOfFrames;
  const out = new Float32Array(n);
  const plane = new Float32Array(n);
  for (let c = 0; c < frame.numberOfChannels; c++) {
    frame.copyTo(plane, { planeIndex: c, format: "f32-planar" });
    for (let i = 0; i < n; i++) out[i] += plane[i] / frame.numberOfChannels;
  }
  return out;
}

/**
 * Reads AudioData frames until the stream ends. `onStart({wall_ms, source})` fires before the first chunk with the
 * wall-clock time (epoch ms) of the first sample: from the capture timestamp when that clock is `performance.now()`'s
 * (within 2 s of the frame's arrival), else the arrival time minus the frame's length.
 */
export async function pump(readable, onChunk, onStart, onHealth, everyMs = 5000) {
  const capture = new Capture();
  const reader = readable.getReader();
  let arrivedAt = 0;
  let firstSeconds = 0;
  let lastReport = 0;
  for (;;) {
    const { value: frame, done } = await reader.read();
    if (done) return;
    const now = performance.now();
    const samples = mono(frame);
    const timestamp = frame.timestamp;
    const rate = frame.sampleRate;
    frame.close();
    if (!arrivedAt) {
      arrivedAt = lastReport = now;
      firstSeconds = samples.length / rate;
      const arrivalWall = performance.timeOrigin + now;
      const captured = performance.timeOrigin + timestamp / 1000;
      const sameClock = Math.abs(captured - arrivalWall) < 2000;
      onStart({ wall_ms: sameClock ? captured : arrivalWall - firstSeconds * 1000, source: sameClock ? "capture" : "arrival" });
    }
    for (const chunk of capture.push(timestamp, rate, samples)) onChunk(chunk.buffer);
    if (now - lastReport >= everyMs) {
      lastReport = now;
      onHealth(capture.health((now - arrivedAt) / 1000 + firstSeconds));
    }
  }
}
