// Plain JS in public/ because addModule needs a URL. Posts 4096-sample mono 16-bit PCM chunks at 16 kHz (256 ms each).
// Runs at the device's own rate: a context forced to 16 kHz over a 48 kHz microphone made Chrome drop a third of the
// samples. A quantum with no input still advances time, so it is written as silence and counted.
import { Resampler, OUT_RATE } from "/resampler.js";

const QUANTUM = 128;

class Recorder extends AudioWorkletProcessor {
  constructor() {
    super();
    this.resampler = new Resampler(sampleRate);
    this.chunk = new Int16Array(4096);
    this.filled = 0;
    this.zeroQuanta = 0;
    this.silence = new Float32Array(QUANTUM);
  }

  process(inputs) {
    let channel = inputs[0][0];
    if (!channel) {
      channel = this.silence;
      this.zeroQuanta++;
    }
    for (const sample of this.resampler.push(channel)) {
      this.chunk[this.filled++] = Math.max(-1, Math.min(1, sample)) * 32767;
      if (this.filled === this.chunk.length) {
        const pcm = this.chunk.slice().buffer;
        this.port.postMessage({ pcm, zeroQuanta: this.zeroQuanta, rate: sampleRate, outRate: OUT_RATE }, [pcm]);
        this.filled = 0;
      }
    }
    return true;
  }
}

registerProcessor("recorder", Recorder);
