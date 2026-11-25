"""JSONL files that survive a killed job: steps 1-2 append in chunks and resume where they stopped.

Same checkpoint rule as label_modes.py: once a chunk is flushed and fsynced it is never redone, and a torn last
line (a job killed mid-write) is dropped on the next start, so appending stays valid JSONL.
"""
import json
import os
import pathlib


def read_jsonl(path) -> list[dict]:
    """All complete rows; a torn last line is dropped and the file rewritten without it."""
    path = pathlib.Path(path)
    if not path.exists():
        return []
    good, torn = [], False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            good.append(json.loads(line))
        except json.JSONDecodeError:
            torn = True
    if torn:
        tmp = path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in good), encoding="utf-8")
        os.replace(tmp, path)
    return good


def append_jsonl(path, rows: list[dict]) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
        f.flush()
        os.fsync(f.fileno())


def write_jsonl(path, rows: list[dict]) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


def chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]
