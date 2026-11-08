"""Append-only session log, so a show can be replayed (invariant 6). This module is the only code that knows the
file format (schema v1, docs/plans/live-loop.md). Text and timings always; raw audio of people is never written
here, it lives in its own file next to the log and only with the room's consent."""
import json
import pathlib
import time
import uuid

SCHEMA = 1
SESSIONS = pathlib.Path(__file__).resolve().parents[2] / "data" / "sessions"

# Required fields per event. A field may be present with value None where the plan allows null.
EVENTS = {
    "show_start": ("room", "context", "policy", "generator_id", "selector_id", "git_sha", "operator", "audio"),
    "bit_decision": ("bit", "obs", "scores", "greedy", "action", "probs"),
    "candidates": ("bit", "joke", "prompt_id", "prompt", "generator_id", "items"),
    "joke_played": ("bit", "joke", "chosen", "pick_prob", "text", "tts_seconds", "audio_seconds",
                    "audio_start_at", "audio_end_at"),
    "reaction": ("bit", "joke", "source", "level", "key", "latency_s"),
    "gap": ("bit", "joke", "planned_hold_s", "actual_gap_s"),
    "bit_end": ("bit", "n_labels", "reward_version", "reward"),
    "rating": ("value",),
    "show_end": ("status",),
}
ENVELOPE = ("schema", "session", "at", "event")
CHOICES = {
    ("show_start", "room"): ("live", "sim"),
    ("reaction", "source"): ("button", "detector", "sim"),
    ("show_end", "status"): ("finished", "aborted"),
}


def validate(record: dict) -> None:
    """Raise ValueError unless the row is a schema v1 row with every required field."""
    missing = [k for k in ENVELOPE if k not in record]
    if missing:
        raise ValueError(f"missing envelope fields {missing}")
    if record["schema"] != SCHEMA:
        raise ValueError(f"unsupported schema {record['schema']!r}")
    event = record["event"]
    if event not in EVENTS:
        raise ValueError(f"unknown event {event!r}")
    missing = [k for k in EVENTS[event] if k not in record]
    if missing:
        raise ValueError(f"{event}: missing fields {missing}")
    for (name, field), allowed in CHOICES.items():
        if name == event and record[field] not in allowed:
            raise ValueError(f"{event}.{field} must be one of {allowed}, got {record[field]!r}")
    if event == "reaction" and record["level"] not in (None, 0, 1, 2, 3):
        raise ValueError(f"reaction.level must be 0-3 or null, got {record['level']!r}")
    if event == "rating" and record["value"] not in (1, 2, 3, 4, 5):
        raise ValueError(f"rating.value must be 1-5, got {record['value']!r}")


class SessionLog:
    def __init__(self, directory: pathlib.Path = SESSIONS):
        directory.mkdir(parents=True, exist_ok=True)
        self.session_id = uuid.uuid4().hex[:12]
        self.path = directory / f"{self.session_id}.jsonl"

    def _append(self, record: dict) -> None:
        # One write call per row keeps a crash from interleaving rows; it can still tear the last one.
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def write(self, event: str, **fields) -> None:
        record = {"schema": SCHEMA, "session": self.session_id, "at": time.time(), "event": event, **fields}
        validate(record)
        self._append(record)

    def write_demo(self, event: str, **fields) -> None:
        """The fixed-lines demo's events, outside schema v1 (marked `schema: 0`, so read_session rejects them).
        T4 replaces the demo loop with real events; delete this then."""
        self._append({"schema": 0, "session": self.session_id, "at": time.time(), "event": event, **fields})


def read_session(path: pathlib.Path) -> list[dict]:
    """Rows of one show. A torn final line (crash mid-write) is dropped; a torn line anywhere else is corruption."""
    lines = pathlib.Path(path).read_text(encoding="utf-8").splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    rows = []
    for number, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            if number == len(lines):
                break
            raise ValueError(f"{path}: line {number} is not valid JSON (corrupt log)") from e
        validate(record)
        rows.append(record)
    return rows
