"use client";

import { useEffect, useRef } from "react";
import { IDLE_HUE, centroidHz, hueFromCentroid, isSilent, loudness, rms, smooth } from "./scope";

const POINTS = 256;

// The agent's only body: one glowing line traced from the audio it is speaking. Its colour and strength come from the
// voice alone; the comedy mode must never show in the operator UI.
export default function Waveform({ analyser }: { analyser: AnalyserNode | null }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    const calm = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    const traceK = calm ? 0.08 : 0.35;
    const paramK = calm ? 0.02 : 0.1;
    const samples = new Float32Array(analyser ? analyser.fftSize : 0);
    const bins = new Float32Array(analyser ? analyser.frequencyBinCount : 0);
    const trace = new Float32Array(POINTS);
    let hue = IDLE_HUE;
    let strength = 0;
    let frame = 0;

    const draw = () => {
      const scale = window.devicePixelRatio || 1;
      const width = canvas.clientWidth * scale;
      const height = canvas.clientHeight * scale;
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }

      let level = 0;
      let targetHue = IDLE_HUE;
      if (analyser) {
        analyser.getFloatTimeDomainData(samples);
        level = rms(samples);
        if (!isSilent(level)) {
          analyser.getFloatFrequencyData(bins);
          targetHue = hueFromCentroid(centroidHz(bins, analyser.context.sampleRate));
        }
        // Start each sweep at a rising zero crossing so a steady tone holds still instead of scrolling.
        let start = 0;
        if (!calm) while (start < samples.length / 2 && !(samples[start] < 0 && samples[start + 1] >= 0)) start++;
        const span = samples.length - start;
        for (let i = 0; i < POINTS; i++) trace[i] = smooth(trace[i], samples[start + Math.floor((i / POINTS) * span)], traceK);
      } else {
        for (let i = 0; i < POINTS; i++) trace[i] = smooth(trace[i], 0, traceK);
      }
      hue = smooth(hue, targetHue, paramK);
      strength = smooth(strength, isSilent(level) ? 0 : loudness(level), paramK);

      const mid = height / 2;
      const reach = height * 0.45;
      const gain = 3 * (0.6 + 0.4 * strength);
      ctx.clearRect(0, 0, width, height);
      ctx.lineJoin = "round";
      ctx.beginPath();
      for (let i = 0; i < POINTS; i++) {
        const y = mid - Math.max(-1, Math.min(1, trace[i] * gain)) * reach;
        if (i === 0) ctx.moveTo(0, y);
        else ctx.lineTo((i / (POINTS - 1)) * width, y);
      }

      const lightness = dark ? 62 : 38;
      const alpha = 0.35 + 0.65 * strength;
      ctx.shadowColor = `hsla(${hue} 100% ${lightness}% / ${0.4 + 0.6 * strength})`;
      ctx.shadowBlur = (10 + 30 * strength) * scale;
      ctx.strokeStyle = `hsla(${hue} 100% ${lightness}% / ${0.25 + 0.35 * strength})`;
      ctx.lineWidth = 6 * scale;
      ctx.stroke();
      ctx.shadowBlur = 0;
      ctx.strokeStyle = `hsla(${hue} 100% ${lightness + (dark ? 14 : 0)}% / ${alpha})`;
      ctx.lineWidth = 1.5 * scale;
      ctx.stroke();
      frame = requestAnimationFrame(draw);
    };
    draw();
    return () => cancelAnimationFrame(frame);
  }, [analyser]);

  return <canvas ref={canvasRef} data-testid="waveform" style={{ width: "100%", height: "100%", display: "block" }} />;
}
