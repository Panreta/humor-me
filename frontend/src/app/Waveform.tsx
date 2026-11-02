"use client";

import { useEffect, useRef } from "react";

const BARS = 64;

// The agent's only body: frequency bars read live from the audio it's speaking.
export default function Waveform({ analyser }: { analyser: AnalyserNode | null }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    const bins = new Uint8Array(analyser ? analyser.frequencyBinCount : BARS);
    let frame = 0;

    const draw = () => {
      const scale = window.devicePixelRatio || 1;
      const width = canvas.clientWidth * scale;
      const height = canvas.clientHeight * scale;
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
      ctx.clearRect(0, 0, width, height);
      if (analyser) analyser.getByteFrequencyData(bins);
      ctx.fillStyle = getComputedStyle(canvas).color;
      // Speech energy sits in the low bins, so the bars cover only the lower part of the spectrum.
      const usable = Math.floor(bins.length * 0.35);
      const gap = 2 * scale;
      const barWidth = Math.max(1, width / BARS - gap);
      for (let i = 0; i < BARS; i++) {
        const level = analyser ? bins[Math.floor((i / BARS) * usable)] / 255 : 0;
        const barHeight = Math.max(2 * scale, level * height * 0.9);
        const x = i * (barWidth + gap) + gap / 2;
        ctx.beginPath();
        ctx.roundRect(x, (height - barHeight) / 2, barWidth, barHeight, barWidth / 2);
        ctx.fill();
      }
      frame = requestAnimationFrame(draw);
    };
    draw();
    return () => cancelAnimationFrame(frame);
  }, [analyser]);

  return <canvas ref={canvasRef} data-testid="waveform" style={{ width: "100%", height: "100%", display: "block" }} />;
}
