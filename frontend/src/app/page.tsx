"use client";

import { useShow, type ShowState } from "./useShow";
import Waveform from "./Waveform";
import styles from "./page.module.css";

const STATUS: Record<ShowState, string> = {
  idle: "Ready when you are.",
  connecting: "Warming up the voice. The first show can take a little while.",
  speaking: "",
  done: "That's the set.",
  error: "Can't reach the backend on localhost:8000. Is it running?",
};

export default function Home() {
  const { state, line, analyser, start } = useShow();
  const busy = state === "connecting" || state === "speaking";

  return (
    <main className={styles.stage}>
      <div className={styles.wave} aria-hidden="true">
        <Waveform analyser={analyser} />
      </div>
      <p className={styles.line} aria-live="polite">
        {state === "speaking" ? line : STATUS[state]}
      </p>
      <button className={styles.start} onClick={start} disabled={busy}>
        {state === "done" ? "Run it again" : "Start the show"}
      </button>
    </main>
  );
}
