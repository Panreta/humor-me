// `npm run test:recorder`: the recorder's resampler under Node, no dependencies. It must make 16,000 samples per
// second of input from any device rate, in 128-sample quanta, and keep a tone's frequency.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const source = readFileSync(new URL("../public/resampler.js", import.meta.url), "utf8");
const { Resampler, OUT_RATE } = await import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);

function run(inRate, hz, seconds) {
  const r = new Resampler(inRate);
  const total = Math.round(inRate * seconds);
  const out = [];
  for (let i = 0; i < total; i += 128) {
    const q = new Float32Array(128);
    for (let k = 0; k < 128; k++) q[k] = 0.5 * Math.sin((2 * Math.PI * hz * (i + k)) / inRate);
    for (const s of r.push(q)) out.push(s);
  }
  return { out, seconds: (Math.floor(total / 128) * 128) / inRate };
}

const peakOf = (xs) => xs.slice(200).reduce((m, v) => Math.max(m, Math.abs(v)), 0);

function frequency(samples) {
  // zero crossings of the steady part, skipping the filter's start-up
  const x = samples.slice(200);
  let crossings = 0;
  for (let i = 1; i < x.length; i++) if (x[i - 1] < 0 && x[i] >= 0) crossings++;
  return crossings / (x.length / OUT_RATE);
}

for (const rate of [48000, 44100, 16000, 32000]) {
  const { out, seconds } = run(rate, 1000, 10);
  // The filter holds back a few samples at the start (a constant), so the rate is the growth between two lengths.
  const shorter = run(rate, 1000, 5);
  const growth = (out.length - shorter.out.length) / (seconds - shorter.seconds);
  assert.ok(Math.abs(growth * 1 - OUT_RATE) <= 1, `${rate} Hz: ${growth} samples per second of input`);
  const hz = frequency(out);
  assert.ok(Math.abs(hz - 1000) < 5, `${rate} Hz: tone came out at ${hz} Hz`);
  const peak = peakOf(out);
  assert.ok(peak > 0.45 && peak < 0.55, `${rate} Hz: tone amplitude ${peak}`);
  console.log(`${rate} Hz in -> ${out.length} samples for ${seconds.toFixed(3)} s, tone ${hz.toFixed(1)} Hz, peak ${peak.toFixed(2)}`);
}

// A tone above the output's Nyquist must be removed, not folded back below it.
const { out } = run(48000, 12000, 1);
const alias = peakOf(out);
assert.ok(alias < 0.02, `12 kHz tone leaked at ${alias}`);
console.log(`12 kHz tone at 48 kHz in -> peak ${alias.toFixed(4)} (removed)`);

// The same input fed in different quantum sizes gives the same samples: phase carries across calls.
const a = new Resampler(44100), b = new Resampler(44100);
const input = Float32Array.from({ length: 4410 }, (_, i) => Math.sin(i * 0.05));
const whole = a.push(input);
const pieces = [];
for (let i = 0; i < input.length; i += 128) pieces.push(...b.push(input.subarray(i, i + 128)));
assert.equal(pieces.length, whole.length);
assert.ok(whole.every((v, i) => Math.abs(v - pieces[i]) < 1e-5), "chunking changed the output");
console.log("chunked and whole input agree");
