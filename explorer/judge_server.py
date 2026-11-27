"""Serve the A/B judge over HTTP from a GPU job, so a show on the laptop can ask it which thread is funnier.

Usage (cluster, repo root; normally through explorer/judge_server.sbatch):
    python -m explorer.judge_server --adapter $COURSE_DIR/runs/dpo_judge_v1/final [--port 8765]

POST /score {"prompts": ["Which joke is funnier?\\nJoke A: ...", ...]} -> {"p_a": [...], "seconds": t}
GET /health -> {"ok": true, "adapter": ...}
The job writes `host:port` to runs/judge_server.addr; the laptop reaches it with
    ssh -L 8765:<host>:8765 <user>@login.explorer.northeastern.edu
Scoring is explorer/judge_model.py's, the one Exp 17 measured. One request at a time (a lock), since one GPU.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def make_handler(score, adapter: str | None):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._send(200, {"ok": True, "adapter": adapter}) if self.path == "/health" else self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/score":
                return self._send(404, {"error": "not found"})
            try:
                prompts = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))["prompts"]
                if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
                    raise ValueError("prompts must be a list of strings")
            except Exception as e:
                return self._send(400, {"error": f"{type(e).__name__}: {e}"})
            t = time.perf_counter()
            with lock:
                p_a = score(prompts)
            self._send(200, {"p_a": [None if p != p else p for p in p_a], "seconds": time.perf_counter() - t})

        def log_message(self, fmt, *args):  # one line per request, no prompt text
            print(f"{self.address_string()} {fmt % args}", flush=True)

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--dtype", default="auto", help="auto | fp16 | bf16 (fp16 on a V100)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--addr-file", default="runs/judge_server.addr")
    args = ap.parse_args()

    from explorer.judge_model import load, make_scorer

    tok, model = load(args.base, args.adapter, args.dtype)
    score = make_scorer(tok, model, batch_size=16)
    score(["Which joke is funnier?\nJoke A: warm\nJoke B: up\nAnswer with A or B."])  # first call compiles kernels
    server = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(score, args.adapter))
    addr = f"{socket.gethostname()}:{args.port}"
    pathlib.Path(args.addr_file).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.addr_file).write_text(addr + "\n")
    print(f"serving on {addr}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
