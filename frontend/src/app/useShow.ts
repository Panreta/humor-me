"use client";

import { useCallback, useEffect, useRef, useState } from "react";

const WS_URL = "ws://localhost:8000/ws";

export type ShowState = "idle" | "connecting" | "speaking" | "done" | "error";

export function useShow() {
  const [state, setState] = useState<ShowState>("idle");
  const [line, setLine] = useState("");
  const [analyser, setAnalyser] = useState<AnalyserNode | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const contextRef = useRef<AudioContext | null>(null);

  useEffect(
    () => () => {
      socketRef.current?.close();
      contextRef.current?.close();
    },
    [],
  );

  const start = useCallback(() => {
    // Browsers only allow audio that begins from a click, so the context is created here.
    const context = contextRef.current ?? new AudioContext();
    contextRef.current = context;
    void context.resume();
    const node = context.createAnalyser();
    node.fftSize = 2048;
    node.smoothingTimeConstant = 0.75;
    node.connect(context.destination);
    setAnalyser(node);
    setLine("");
    setState("connecting");

    socketRef.current?.close();
    const socket = new WebSocket(WS_URL);
    socket.binaryType = "arraybuffer";
    socketRef.current = socket;
    let finished = false;

    socket.onopen = () =>
      socket.send(
        JSON.stringify({
          type: "start",
          context: { venue: "bar", energy: "normal", age: "mixed" },
          room: "sim",
          operator: "demo",
        }),
      );
    socket.onerror = () => setState("error");
    socket.onclose = () => setState((s) => (finished || s === "error" ? s : "idle"));
    socket.onmessage = async (event) => {
      if (typeof event.data === "string") {
        const msg = JSON.parse(event.data);
        if (msg.type === "line") {
          setLine(msg.text);
          setState("speaking");
        } else if (msg.type === "show_end") {
          finished = true;
          setState("done");
          socket.close();
        }
        return;
      }
      const audio = await context.decodeAudioData(event.data);
      const source = context.createBufferSource();
      source.buffer = audio;
      source.connect(node);
      source.onended = () => {
        if (socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ type: "played" }));
      };
      source.start();
    };
  }, []);

  return { state, line, analyser, start };
}
