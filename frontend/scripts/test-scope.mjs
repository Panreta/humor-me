// `npm run test:scope`: the oscilloscope line's pure math under Node, no dependencies.
import assert from "node:assert/strict";
import { HUE, brightness, centroidHz, isSilent, loudness, rms, smooth, trackPeak } from "../src/app/scope.ts";

assert.equal(rms([]), 0);
assert.ok(Math.abs(rms(Array.from({ length: 1000 }, (_, i) => Math.sin(i / 10))) - Math.SQRT1_2) < 0.01);
assert.ok(isSilent(0) && !isSilent(0.05));

assert.equal(smooth(0, 1, 0.25), 0.25);
assert.equal(smooth(0.4, 0.4, 0.5), 0.4);

assert.equal(loudness(0), 0);
assert.equal(loudness(1), 1);
assert.ok(loudness(0.05) < loudness(0.1));

// A single bin's frequency is the centroid; the rest sit at the -Infinity floor the analyser reports for silence.
const bins = (at) => Array.from({ length: 1024 }, (_, i) => (i === at ? -20 : -Infinity));
const hz = centroidHz(bins(100), 48000);
assert.ok(Math.abs(hz - 100 * (24000 / 1024)) < 1e-6);
assert.equal(centroidHz(bins(-1), 48000), 0);

assert.equal(HUE, 145);
assert.equal(brightness(0), 0);
assert.equal(brightness(300), 0);
assert.equal(brightness(1e6), 1);
const tones = [400, 800, 1600, 2800].map(brightness);
assert.ok(tones.every((v, i) => i === 0 || v > tones[i - 1]), "brightness rises with the centroid");

// The peak jumps up at once, decays slowly, and never drops below the floor, so silence is not amplified.
assert.equal(trackPeak(0.1, 0.5), 0.5);
assert.ok(trackPeak(0.5, 0.1) < 0.5 && trackPeak(0.5, 0.1) > 0.49);
assert.equal(trackPeak(0, 0.001), 0.03);
let p = 0.5;
for (let i = 0; i < 600; i++) p = trackPeak(p, 0.05);
assert.ok(p < 0.06, "forgets a loud moment within seconds");

console.log("scope: ok");
