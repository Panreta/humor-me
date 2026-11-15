// Plain JS in public/ because addModule needs a URL. Posts 4096-sample mono 16-bit PCM chunks, 256 ms each at 16 kHz.
class Recorder extends AudioWorkletProcessor {
  constructor() {
    super();
    this.chunk = new Int16Array(4096);
    this.filled = 0;
  }

  process(inputs) {
    const channel = inputs[0][0];
    if (!channel) return true;
    for (let i = 0; i < channel.length; i++) {
      this.chunk[this.filled++] = Math.max(-1, Math.min(1, channel[i])) * 32767;
      if (this.filled === this.chunk.length) {
        const out = this.chunk.slice();
        this.port.postMessage(out.buffer, [out.buffer]);
        this.filled = 0;
      }
    }
    return true;
  }
}

registerProcessor("recorder", Recorder);
