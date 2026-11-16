"use client";

import { useEffect, useRef } from "react";

// About the pitch of the old 64 bars in a 900px box, so bars keep their look at any width.
const PITCH = 14;

// The agent's only body: frequency bars read live from the audio it's speaking.
export default function Waveform({ analyser }: { analyser: AnalyserNode | null }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    const bins = new Uint8Array(analyser ? analyser.frequencyBinCount : 0);
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
      const bars = Math.max(1, Math.round(width / (PITCH * scale)));
      const pitch = width / bars;
      const gap = Math.min(2 * scale, pitch / 4);
      const barWidth = pitch - gap;
      // A solid baseline under the bars, so silence reads as one line rather than a row of dots.
      ctx.globalAlpha = 0.6;
      ctx.fillRect(0, height / 2 - scale / 2, width, scale);
      ctx.globalAlpha = 1;
      for (let i = 0; i < bars; i++) {
        // Mirrored from the centre: the strongest low bins sit in the middle, higher ones spread out to both edges.
        const x01 = (i + 0.5) / bars;
        const level = analyser ? bins[Math.floor(Math.abs(2 * x01 - 1) * usable)] / 255 : 0;
        // Soft taper toward the edges, with a floor so edge bars still move when the voice is loud.
        const taper = 0.2 + 0.8 * Math.sin(Math.PI * x01) ** 0.6;
        const barHeight = level * taper * height * 0.9;
        if (barHeight <= scale) continue;
        const x = i * pitch + gap / 2;
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
