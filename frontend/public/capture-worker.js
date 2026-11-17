// Reads the transferred microphone track in a worker, so a busy page cannot delay or drop frames.
import { pump } from "/capture.js";

self.onmessage = async ({ data }) => {
  try {
    if (typeof MediaStreamTrackProcessor === "undefined") throw new Error("unsupported");
    const processor = new MediaStreamTrackProcessor({ track: data.track, maxBufferSize: 1000 });
    await pump(
      processor.readable,
      (pcm) => self.postMessage({ pcm }, [pcm]),
      (start) => self.postMessage({ start }),
      (health) => self.postMessage({ health }),
    );
    self.postMessage({ ended: true });
  } catch (error) {
    self.postMessage({ error: String(error) });
  }
};
