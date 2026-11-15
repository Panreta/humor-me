export const RECORD_RATE = 16000;
const HEALTH_EVERY_MS = 5000;

// The browser's voice processing is for calls: it would suppress the laughter this recording exists for.
const RAW = { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false };

export type Health = { produced_s: number; elapsed_s: number; zero_quanta: number; rate: number };

type WorkletMessage = { pcm: ArrayBuffer; zeroQuanta: number; rate: number };

/**
 * Opens the microphone and calls `onChunk` with 16-bit mono PCM at 16 kHz. The context runs at the device's own rate
 * and the worklet downsamples: a context forced to 16 kHz over a 48 kHz microphone lost a third of the samples.
 * Every few seconds `onHealth` reports audio produced against wall clock, which is how a loss shows up in the logs.
 * Resolves to a stop function.
 */
export async function startRecorder(
  onChunk: (pcm: ArrayBuffer) => void,
  onHealth: (health: Health) => void,
): Promise<() => void> {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: RAW });
  let context: AudioContext | null = null;
  try {
    context = new AudioContext();
    await context.audioWorklet.addModule("/recorder-worklet.js");
    // Made in a websocket handler, not a click, so it may start suspended; a suspended context records silence.
    await context.resume();
    if (context.state !== "running") throw new Error("the browser would not start audio capture");
    const node = new AudioWorkletNode(context, "recorder");
    let samples = 0;
    let first = 0;
    let lastReport = 0;
    node.port.onmessage = (event: MessageEvent<WorkletMessage>) => {
      const { pcm, zeroQuanta, rate } = event.data;
      const now = performance.now();
      if (!first) first = lastReport = now;
      else samples += pcm.byteLength / 2;
      onChunk(pcm);
      // The first chunk's samples were captured before it arrived, so they are left out of both sides of the ratio.
      if (now - lastReport >= HEALTH_EVERY_MS) {
        lastReport = now;
        onHealth({ produced_s: samples / RECORD_RATE, elapsed_s: (now - first) / 1000, zero_quanta: zeroQuanta, rate });
      }
    };
    // Silent sink: some browsers only run a node that reaches the destination.
    const sink = context.createGain();
    sink.gain.value = 0;
    context.createMediaStreamSource(stream).connect(node).connect(sink).connect(context.destination);
  } catch (error) {
    stream.getTracks().forEach((t) => t.stop());
    void context?.close();
    throw error;
  }
  const open = context;
  return () => {
    stream.getTracks().forEach((t) => t.stop());
    void open.close();
  };
}
