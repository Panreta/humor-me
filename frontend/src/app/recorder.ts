export const RECORD_RATE = 16000;

// The browser's voice processing is for calls: it would suppress the laughter this recording exists for.
const RAW = { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false };

// Chrome only (the operator UI is); not in TypeScript's DOM types yet.
declare const MediaStreamTrackProcessor: new (init: {
  track: MediaStreamTrack;
  maxBufferSize?: number;
}) => { readable: ReadableStream<unknown> };

export type Start = { wall_ms: number; source: "capture" | "arrival" };
export type Health = { produced_s: number; elapsed_s: number; gap_ms: number; wall_s: number; rate: number };

type Pump = (
  readable: ReadableStream<unknown>,
  onChunk: (pcm: ArrayBuffer) => void,
  onStart: (start: Start) => void,
  onHealth: (health: Health) => void,
) => Promise<void>;

export type Sinks = {
  onChunk: (pcm: ArrayBuffer) => void;
  onStart: (start: Start) => void;
  onHealth: (health: Health) => void;
  onFail: (reason: string) => void;
};

/**
 * Opens the microphone and reads its frames straight off the track, stamped by the capture clock, so a busy render
 * thread cannot cost samples (an AudioWorklet lost a third of them under Kokoro's CPU load). `capture.js` fills any gap
 * with silence and drops overlap, so the 16 kHz Int16 chunks keep an exact timeline. Prefers a worker (the page cannot
 * delay it) and reads on the main thread if the track will not transfer. Resolves to a stop function.
 */
export async function startRecorder(sinks: Sinks): Promise<() => void> {
  if (typeof MediaStreamTrackProcessor === "undefined") throw new Error("this browser cannot read microphone frames");
  const stream = await navigator.mediaDevices.getUserMedia({ audio: RAW });
  const [track] = stream.getAudioTracks();
  let worker: Worker | null = null;
  let clone: MediaStreamTrack | null = null;
  let stopped = false;

  const onMain = async () => {
    // A URL import the bundler must leave alone: the file lives in public/.
    const url = "/capture.js";
    const { pump } = (await import(/* webpackIgnore: true */ url)) as { pump: Pump };
    const processor = new MediaStreamTrackProcessor({ track, maxBufferSize: 1000 });
    await pump(processor.readable, sinks.onChunk, sinks.onStart, sinks.onHealth);
  };
  const fallBack = (why: string) => {
    worker?.terminate();
    worker = null;
    clone?.stop(); // a clone keeps the microphone open until it is stopped too
    if (stopped) return;
    console.warn(`recording on the main thread: ${why}`);
    onMain().catch((e) => !stopped && sinks.onFail(String(e)));
  };

  try {
    const sent = (clone = track.clone()); // the worker gets the clone; the original stays here for the fallback
    worker = new Worker("/capture-worker.js", { type: "module" });
    worker.onmessage = ({ data }) => {
      if (data.pcm) sinks.onChunk(data.pcm);
      else if (data.start) sinks.onStart(data.start);
      else if (data.health) sinks.onHealth(data.health);
      else if (data.error) fallBack(data.error);
    };
    worker.onerror = () => fallBack("worker failed to load");
    worker.postMessage({ track: sent }, [sent as unknown as Transferable]);
  } catch (error) {
    fallBack(String(error));
  }

  return () => {
    stopped = true;
    worker?.terminate();
    clone?.stop();
    stream.getTracks().forEach((t) => t.stop());
  };
}
