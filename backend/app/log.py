"""Append-only session log, so a show can be replayed (invariant 6). Text and timings only, never audio of people."""
import json
import pathlib
import time
import uuid

SESSIONS = pathlib.Path(__file__).resolve().parents[2] / "data" / "sessions"


class SessionLog:
    def __init__(self, directory: pathlib.Path = SESSIONS):
        directory.mkdir(parents=True, exist_ok=True)
        self.session_id = uuid.uuid4().hex[:12]
        self.path = directory / f"{self.session_id}.jsonl"

    def write(self, event: str, **fields) -> None:
        record = {"session": self.session_id, "at": time.time(), "event": event, **fields}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
