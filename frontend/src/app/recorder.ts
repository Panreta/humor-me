export const RECORD_RATE = 16000;

// The browser's voice processing is for calls: it would suppress the laughter this recording exists to capture.
const RAW = { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false };

/** Opens the microphone and calls `onChunk` with 16-bit mono PCM at 16 kHz. Resolves to a stop function. */
export async function startRecorder(onChunk: (pcm: ArrayBuffer) => void): Promise<() => void> {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: RAW });
  let context: AudioContext | null = null;
  try {
    context = new AudioContext({ sampleRate: RECORD_RATE });
    await context.audioWorklet.addModule("/recorder-worklet.js");
    // Made in a websocket handler, not a click, so it may start suspended; a suspended context records silence.
    await context.resume();
    if (context.state !== "running") throw new Error("the browser would not start audio capture");
    const node = new AudioWorkletNode(context, "recorder");
    node.port.onmessage = (event: MessageEvent<ArrayBuffer>) => onChunk(event.data);
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
