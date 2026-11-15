// Streaming downsampler to 16 kHz for the recorder worklet. Plain ES module in public/ (the worklet imports it by URL)
// with no browser globals, so scripts/test-recorder.mjs runs it under Node.
export const OUT_RATE = 16000;
const CUTOFF_HZ = 7200; // below the 8 kHz Nyquist of the output, so nothing above it folds back as an alias
const ZEROS = 8; // sinc zero-crossings per side, measured at the output rate

export class Resampler {
  constructor(inRate) {
    this.ratio = inRate / OUT_RATE;
    this.half = Math.ceil(ZEROS * Math.max(1, this.ratio));
    const scale = (2 * CUTOFF_HZ) / inRate;
    // Windowed-sinc kernel sampled on a fine grid: low-pass and fractional-delay interpolation in one step.
    this.grid = 64;
    this.kernel = new Float32Array(2 * this.half * this.grid + 1);
    for (let i = 0; i < this.kernel.length; i++) {
      const x = i / this.grid - this.half;
      const arg = Math.PI * scale * x;
      const sinc = arg === 0 ? 1 : Math.sin(arg) / arg;
      this.kernel[i] = scale * sinc * (0.5 + 0.5 * Math.cos((Math.PI * x) / this.half));
    }
    this.buf = new Float32Array(this.half); // zeros before the first sample, so output 0 sits at input 0
    this.base = -this.half; // input index of buf[0]
    this.pos = 0; // input index of the next output; fractional, carried across calls so no sample is lost or doubled
  }

  /** Feed input samples; returns the 16 kHz samples that are now complete. */
  push(input) {
    const joined = new Float32Array(this.buf.length + input.length);
    joined.set(this.buf);
    joined.set(input, this.buf.length);
    this.buf = joined;
    const out = [];
    while (Math.floor(this.pos) + this.half < this.base + this.buf.length) {
      const first = Math.floor(this.pos) - this.half + 1;
      let acc = 0;
      for (let k = first; k < first + 2 * this.half; k++) {
        const x = k - this.pos + this.half; // 0..2*half along the kernel
        acc += this.buf[k - this.base] * this.kernel[Math.round(x * this.grid)];
      }
      out.push(acc);
      this.pos += this.ratio;
    }
    const keepFrom = Math.floor(this.pos) - this.half + 1;
    if (keepFrom > this.base) {
      this.buf = this.buf.slice(keepFrom - this.base);
      this.base = keepFrom;
    }
    return out;
  }
}
