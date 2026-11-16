// `npm run test:scope`: the oscilloscope line's pure math under Node, no dependencies.
import assert from "node:assert/strict";
import { IDLE_HUE, centroidHz, hueFromCentroid, isSilent, loudness, rms, smooth } from "../src/app/scope.ts";

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

assert.equal(hueFromCentroid(0), IDLE_HUE);
assert.equal(hueFromCentroid(300), IDLE_HUE);
assert.equal(hueFromCentroid(1e6), 270);
const hues = [200, 500, 1000, 2000].map(hueFromCentroid);
assert.ok(hues.every((h, i) => i === 0 || h > hues[i - 1]), "hue rises with brightness");
assert.ok(hues[0] >= 145 && hues[3] <= 270);

console.log("scope: ok");
