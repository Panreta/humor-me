// `npm run test:recorder`: the recorder's resampler and capture timeline under Node, no dependencies. The resampler must
// make 16,000 samples per second of input from any device rate and keep a tone's frequency; the capture must put every
// sample where its timestamp says, whatever the frames did (gap, overlap).
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const asModule = (text) => `data:text/javascript;base64,${Buffer.from(text).toString("base64")}`;
const source = readFileSync(new URL("../public/resampler.js", import.meta.url), "utf8");
const { Resampler, OUT_RATE } = await import(asModule(source));
// capture.js imports the resampler by its URL in public/, which Node cannot resolve.
const captureSource = readFileSync(new URL("../public/capture.js", import.meta.url), "utf8");
const { Capture, pump } = await import(asModule(captureSource.replace('"/resampler.js"', JSON.stringify(asModule(source)))));

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

// Capture timeline: a 48 kHz microphone in 10 ms frames, with a 200 ms hole at 1.0 s and a frame that overlaps its
// predecessor by 5 ms at 2.0 s. The 1 kHz tone is a function of absolute time, so any misplaced sample shows.
const RATE = 48000;
const T0 = 5_000_000; // the capture clock starts anywhere
const tone = (t) => 0.5 * Math.sin(2 * Math.PI * 1000 * t);
const frameAt = (startS, n) => ({
  timestamp: T0 + startS * 1e6,
  sampleRate: RATE,
  numberOfFrames: n,
  numberOfChannels: 1,
  copyTo(dst) {
    for (let i = 0; i < n; i++) dst[i] = tone(startS + i / RATE);
  },
  close() {},
});
function frames({ gap, overlap }) {
  const list = [];
  for (let k = 0; k < 300; k++) {
    const t = k * 0.01;
    if (gap && t >= 1.0 - 1e-9 && t < 1.2 - 1e-9) continue;
    if (overlap && Math.abs(t - 2.0) < 1e-9) list.push(frameAt(1.995, 720));
    else list.push(frameAt(t, 480));
  }
  return list;
}
async function capture(list) {
  const readable = new ReadableStream({ start(c) { list.forEach((f) => c.enqueue(f)); c.close(); } });
  const out = [];
  let start = null;
  await pump(readable, (buf) => out.push(...new Int16Array(buf)), (s) => (start = s), () => {});
  return { out: out.map((v) => v / 32767), start };
}

const clean = await capture(frames({ gap: false, overlap: false }));
const holed = await capture(frames({ gap: true, overlap: true }));
// 3.0 s of capture clock in both; chunking to 4096 leaves up to one chunk unsent, so compare in whole chunks
const chunks = Math.min(Math.floor(clean.out.length / 4096), Math.floor(holed.out.length / 4096));
assert.ok(chunks >= 11, `only ${chunks} chunks`);
assert.ok(Math.abs(holed.out.length - clean.out.length) <= 4096, "gap and overlap changed the length by more than a chunk");
let worst = 0;
let silent = true;
for (let i = 200; i < chunks * 4096 - 200; i++) {
  const inGap = i >= 16000 && i < 19200;
  const nearEdge = Math.abs(i - 16000) < 20 || Math.abs(i - 19200) < 20; // the low-pass rings for a few samples there
  if (inGap) {
    if (!nearEdge && Math.abs(holed.out[i]) > 0.01) silent = false;
  } else if (!nearEdge) {
    worst = Math.max(worst, Math.abs(holed.out[i] - tone(i / OUT_RATE)));
  }
}
assert.ok(silent, "the 200 ms hole was not silence");
assert.ok(worst < 0.02, `a sample is off its timestamp: worst error ${worst}`);
console.log(`gap + overlap: ${chunks} chunks, tone matches absolute time (worst ${worst.toFixed(4)}), hole silent`);

// Exact count: the sample index of the last whole chunk equals the timestamp span times 16 kHz.
const c = new Capture();
let produced = 0;
for (const f of frames({ gap: true, overlap: true })) {
  const buf = new Float32Array(f.numberOfFrames);
  f.copyTo(buf);
  produced += c.push(f.timestamp, RATE, buf).reduce((n, ch) => n + ch.length, 0);
}
const spanSeconds = (c.endUs - c.firstUs) / 1e6;
const total = produced + c.pending.length;
assert.ok(Math.abs(total - (spanSeconds * OUT_RATE - 8)) <= 2, `${total} samples for ${spanSeconds} s`);
assert.ok(Math.abs(c.health(3).gap_ms - 200) < 1, `gap_ms ${c.health(3).gap_ms}`);
console.log(`${total} samples for a ${spanSeconds.toFixed(3)} s span (filter holds back 8), gap_ms ${c.health(3).gap_ms.toFixed(1)}`);

// The start time: a capture clock that is performance.now()'s is trusted; any other is replaced by arrival.
const near = frameAt(0, 480);
near.timestamp = (performance.now() - 10) * 1000; // 10 ms ago on performance.now()'s clock
const same = await capture([near]);
assert.equal(same.start.source, "capture");
const other = await capture([frameAt(0, 480)]); // T0 is nowhere near performance.now() in epoch terms
assert.equal(other.start.source, "arrival");
assert.ok(Math.abs(other.start.wall_ms - (performance.timeOrigin + performance.now() - 10)) < 200);
console.log("start time: capture clock trusted only when it matches performance.now()");
