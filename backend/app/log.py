"""Append-only session log, so a show can be replayed (invariant 6). This module is the only code that knows the
file format (schema v2, docs/plans/live-loop.md; v1 logs stay readable). Text and timings always; raw audio of people is never written
here, it lives in its own file next to the log and only with the room's consent."""
import json
import pathlib
import time
import uuid

SCHEMA = 2  # v2 adds the `bridge`, `cache`, `pause`, `resume`, `prep`, `prediction` and `material` events; a v1 log has none and replays unchanged
READABLE = (1, 2)
SESSIONS = pathlib.Path(__file__).resolve().parents[2] / "data" / "sessions"

# Required fields per event. A field may be present with value None where the plan allows null.
EVENTS = {
    "show_start": ("room", "context", "policy", "generator_id", "selector_id", "git_sha", "operator", "audio"),
    "bit_decision": ("bit", "obs", "scores", "greedy", "action", "probs"),
    "candidates": ("bit", "joke", "prompt_id", "prompt", "generator_id", "items"),
    "joke_played": ("bit", "joke", "chosen", "pick_prob", "text", "tts_seconds", "audio_seconds",
                    "audio_start_at", "audio_end_at"),
    # Where a bit's wait went (D-113): stage durations on the show clock, written when the bit is ready to play.
    "prep": ("bit", "started_at", "generate_s", "safety_s", "render_wait_s", "ready_at", "bank_fallbacks"),
    # The mode guessed for a bit while the one before it played its last jokes, and whether the director then picked it
    # (D-115). Written right after the bit's `bit_decision`; absent when no guess was made.
    "prediction": ("bit", "predicted", "decided", "hit", "started_at"),
    # What a played bit was written from (D-117): its topic and the two character-sheet facts; only a bit that played.
    "material": ("bit", "topic", "from", "facts", "sheet_id", "list_id", "seed", "recycled"),
    # Tonight's approved context (D-119): written once right after show_start, then one `tonight_use` row for each played
    # bit that took a headline as its topic.
    "tonight": ("facts", "market", "gists", "report"),
    "tonight_use": ("bit", "gist"),
    "bridge": ("bit", "text", "tts_seconds", "audio_start_at", "audio_end_at"),
    "cache": ("bits",),  # the cached bits the show could draw on, written right after show_start (D-91)
    "reaction": ("bit", "joke", "source", "level", "key", "latency_s"),
    "gap": ("bit", "joke", "planned_hold_s", "actual_gap_s"),
    "bit_end": ("bit", "n_labels", "reward_version", "reward"),
    "pause": ("paused_at",),  # the show clock (the same timebase as `audio_start_at`), written when the browser pauses
    "resume": ("resumed_at",),
    "rating": ("value",),
    "show_end": ("status",),
}
ENVELOPE = ("schema", "session", "at", "event")
CHOICES = {
    ("show_start", "room"): ("live", "sim"),
    ("reaction", "source"): ("button", "detector", "sim"),
    ("show_end", "status"): ("finished", "aborted"),
}
# Optional fields: checked only when present, so a log written before they existed stays valid.
OPTIONAL_CHOICES = {("bridge", "kind"): ("filler", "cached")}
# Optional numbers of a `prep` row (null when the voice cannot split the render into queue and synthesis)
PREP_OPTIONAL = ("bank_s", "other_s", "render_queue_s", "render_s")
PREP_DURATIONS = ("generate_s", "safety_s", "render_wait_s", *PREP_OPTIONAL)
SPEC_STAGES = ("generating", "safety", "rendering", "ready")  # how far a guessed bit's prep had got when the bit ended
CACHED_SOURCE = ("session", "bit", "joke")  # where a cached line came from; a bridge row of kind "cached" needs it


NESTED = {
    ("show_start", "context"): ("venue", "energy", "age"),
    ("show_start", "policy"): ("name", "epsilon", "alpha", "prior"),
    ("show_start", "audio"): ("consent", "path"),
    ("bit_decision", "obs"): ("t", "n_bits", "context", "last_mode"),
}
ITEM_KEYS = ("text", "verdict", "reason")
# Frozen reaction keys (plan, "Reaction labels") and the level each one means.
KEY_LEVELS = {"W": 3, "K": 2, "Space": 1, "D": 0}


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)  # True == 1 must not pass as a level


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _need_keys(where: str, obj, keys) -> None:
    if not isinstance(obj, dict):
        raise ValueError(f"{where} must be an object")
    missing = [k for k in keys if k not in obj]
    if missing:
        raise ValueError(f"{where}: missing fields {missing}")


def validate(record: dict) -> None:
    """Raise ValueError unless the row is a schema v1 or v2 row with every required field."""
    if not isinstance(record, dict):
        raise ValueError("row must be a JSON object")
    missing = [k for k in ENVELOPE if k not in record]
    if missing:
        raise ValueError(f"missing envelope fields {missing}")
    if record["schema"] not in READABLE:
        raise ValueError(f"unsupported schema {record['schema']!r}")
    event = record["event"]
    if not isinstance(event, str) or event not in EVENTS:
        raise ValueError(f"unknown event {event!r}")
    if event in ("prep", "bridge", "cache", "pause", "resume", "prediction", "material", "tonight", "tonight_use") and record["schema"] < 2:
        raise ValueError(f"a schema 1 log has no {event} event")
    _need_keys(event, record, EVENTS[event])
    for (name, field), allowed in CHOICES.items():
        if name == event and record[field] not in allowed:
            raise ValueError(f"{event}.{field} must be one of {allowed}, got {record[field]!r}")
    for (name, field), allowed in OPTIONAL_CHOICES.items():
        if name == event and field in record and record[field] not in allowed:
            raise ValueError(f"{event}.{field} must be one of {allowed}, got {record[field]!r}")
    if event == "bridge" and record.get("kind") == "cached":
        _need_keys("bridge.source", record.get("source"), CACHED_SOURCE)
    for (name, field), keys in NESTED.items():
        if name == event:
            _need_keys(f"{event}.{field}", record[field], keys)
    if event == "show_start" and record["audio"]["path"] is not None             and not isinstance(record["audio"].get("start_at"), (int, float)):
        raise ValueError("show_start.audio.start_at must be a number when a recording exists")
    if event == "cache":
        if not isinstance(record["bits"], list):
            raise ValueError("cache.bits must be a list")
        for i, b in enumerate(record["bits"]):
            _need_keys(f"cache.bits[{i}]", b, ("session", "bit", "mode", "reward", "source_age", "lines"))
            for j, line in enumerate(b["lines"]):
                _need_keys(f"cache.bits[{i}].lines[{j}]", line, ("joke", "text", "verdict", "reason"))
        for key in ("heard_dropped", "bridge_seed"):  # optional (D-112): absent in older logs
            if record.get(key) is not None and not (_is_int(record[key]) and record[key] >= 0):
                raise ValueError(f"cache.{key} must be a non-negative integer")
        if record.get("skip_heard") is not None and not isinstance(record["skip_heard"], bool):
            raise ValueError("cache.skip_heard must be a boolean")
    if event == "prep":
        for key in PREP_DURATIONS:
            if key in PREP_OPTIONAL and record.get(key) is None:
                continue
            if not _is_number(record[key]) or record[key] < 0:
                raise ValueError(f"prep.{key} must be a non-negative number, got {record[key]!r}")
        for key in ("started_at", "ready_at"):
            if not _is_number(record[key]):
                raise ValueError(f"prep.{key} must be a number, got {record[key]!r}")
        if record["ready_at"] < record["started_at"]:
            raise ValueError("prep.ready_at is before prep.started_at")
        if record.get("first_render_failed") is not None and not isinstance(record["first_render_failed"], bool):
            raise ValueError("prep.first_render_failed must be a boolean")
        if record.get("speculative") is not None and not isinstance(record["speculative"], bool):
            raise ValueError("prep.speculative must be a boolean")  # D-115: prepared on a guess that was then adopted
        if not (_is_int(record["bank_fallbacks"]) and record["bank_fallbacks"] >= 0):
            raise ValueError("prep.bank_fallbacks must be a non-negative integer")
    if event == "material":
        if not (isinstance(record["topic"], str) and isinstance(record["facts"], list)
                and all(isinstance(f, str) for f in record["facts"])):
            raise ValueError("material.topic must be a string and material.facts a list of strings")
        if record["from"] not in ("character", "everyday"):
            raise ValueError("material.from must be 'character' or 'everyday'")
        if not isinstance(record["recycled"], bool) or not _is_int(record["seed"]):
            raise ValueError("material.recycled must be a boolean and material.seed an integer")
        if record.get("variant", "full") not in ("full", "topic"):  # optional (D-118): absent in a D-117 log, which is "full"
            raise ValueError("material.variant must be 'full' or 'topic'")
    if event == "tonight":
        if not (isinstance(record["facts"], list) and isinstance(record["gists"], list)
                and all(isinstance(t, str) for t in record["facts"] + record["gists"])
                and isinstance(record["report"], dict)):
            raise ValueError("tonight.facts and tonight.gists must be lists of strings and tonight.report an object")
    if event == "tonight_use" and not (_is_int(record["bit"]) and isinstance(record["gist"], str)):
        raise ValueError("tonight_use.bit must be an integer and tonight_use.gist a string")
    if event == "prediction":
        if not isinstance(record["hit"], bool):
            raise ValueError("prediction.hit must be a boolean")
        if not (isinstance(record["predicted"], str) and isinstance(record["decided"], str)):
            raise ValueError("prediction.predicted and .decided must be mode names")
        if record["hit"] != (record["predicted"] == record["decided"]):
            raise ValueError("prediction.hit must say whether predicted equals decided")
        if not _is_number(record["started_at"]):
            raise ValueError("prediction.started_at must be a number")
        for key in ("spec_prefetched", "learned", "adopted"):  # optional: what the guess cost, and whether it was used
            if record.get(key) is not None and not isinstance(record[key], bool):
                raise ValueError(f"prediction.{key} must be a boolean")
        if record.get("after_joke") is not None and not (_is_int(record["after_joke"]) and record["after_joke"] >= 1):
            raise ValueError("prediction.after_joke must be a positive integer")
        if record.get("spec_stage") is not None and record["spec_stage"] not in SPEC_STAGES:
            raise ValueError(f"prediction.spec_stage must be one of {SPEC_STAGES}")
    if event == "joke_played" and record.get("bridge_overhang_s") is not None:  # optional (D-113): the bit's first line
        if not _is_number(record["bridge_overhang_s"]) or record["bridge_overhang_s"] < 0:
            raise ValueError("joke_played.bridge_overhang_s must be a non-negative number")
    if event == "candidates":
        if not isinstance(record["items"], list):
            raise ValueError("candidates.items must be a list")
        for i, item in enumerate(record["items"]):
            _need_keys(f"candidates.items[{i}]", item, ITEM_KEYS)
        lost = record.get("threads_lost", [])  # optional (D-110): absent in older logs
        if not isinstance(lost, list) or not all(isinstance(x, str) for x in lost):
            raise ValueError("candidates.threads_lost must be a list of strings")
    if event == "reaction":
        level, key = record["level"], record["key"]
        if level is not None and not (_is_int(level) and 0 <= level <= 3):
            raise ValueError(f"reaction.level must be 0-3 or null, got {level!r}")
        if key is not None and key not in KEY_LEVELS:
            raise ValueError(f"reaction.key must be one of {tuple(KEY_LEVELS)} or null, got {key!r}")
        if key is not None and level != KEY_LEVELS[key]:
            raise ValueError(f"reaction.key {key!r} means level {KEY_LEVELS[key]}, got {level!r}")
        if record["source"] == "button" and key is None and level is not None:
            raise ValueError("a button reaction with no key is a timeout, so level must be null")
    if event in ("pause", "resume"):
        at = record["paused_at" if event == "pause" else "resumed_at"]
        if isinstance(at, bool) or not isinstance(at, (int, float)):
            raise ValueError(f"{event} time must be a number, got {at!r}")
    if event == "rating" and not (_is_int(record["value"]) and 1 <= record["value"] <= 5):
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


def read_session(path: pathlib.Path) -> list[dict]:
    """Rows of one show. A torn final line (crash mid-write, possibly inside a multi-byte character) is dropped;
    a torn line anywhere else is corruption."""
    # Split on the newline byte only: str.splitlines() also splits on U+2028 and friends, which
    # ensure_ascii=False leaves literal inside a row.
    lines = pathlib.Path(path).read_bytes().split(b"\n")
    while lines and not lines[-1].strip():
        lines.pop()
    rows = []
    for number, line in enumerate(lines, start=1):
        try:
            record = json.loads(line.decode("utf-8"))
        except ValueError as e:  # UnicodeDecodeError and JSONDecodeError are both ValueErrors
            if number == len(lines):
                break
            raise ValueError(f"{path}: line {number} is not valid JSON (corrupt log)") from e
        validate(record)
        rows.append(record)
    return rows
