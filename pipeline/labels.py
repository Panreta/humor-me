"""Hand labels for bit boundaries (reference for S6 Pk/WindowDiff).

Usage:
    python -m pipeline.labels template <set_id>   # write data/labels/work/<set_id>.txt (never overwritten)
    python -m pipeline.labels import   <set_id>   # read your ">>>" marks -> data/labels/<set_id>.yaml

The work file holds transcript text and stays local (gitignored). The YAML holds only line
indices and your notes, so it is committed. Units are `line_idx` from lines.jsonl, the same
units S4 predicts in, so labels and predictions compare directly.
"""
import json
import re
import sys

import yaml

from pipeline.common import PROCESSED, ROOT

LABELS = ROOT / "data" / "labels"
WORK = LABELS / "work"
MARK = ">>>"
LINE = re.compile(r"^\[(\d{4,5})\] ")

HEADER = """\
# Bit-boundary labeling: {set_id}
#
# A bit = one premise, explored with its tags and act-outs. A callback to an earlier bit
# starts a NEW bit. Crowd-work exchanges are their own bit.
#
# HOW: put a line with only {mark} directly ABOVE the first line of each new bit.
#      Optional note after it, e.g.  {mark} callback: the dentist
#      The first line of the set is always a bit start; no need to mark it.
#      ⟨L2⟩ etc. show the laugh that followed a line (context only, don't edit them).
#      Don't edit or reorder the [NNNN] lines. Save, then run:
#          python -m pipeline.labels import {set_id}
#
"""


def load_lines(set_id: str) -> list[dict]:
    for dataset in ("laugh_labeled", "structure_only"):
        path = PROCESSED / dataset / "lines.jsonl"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                rows = [json.loads(x) for x in f if f'"set_id": "{set_id}"' in x]
            if rows:
                return [r for r in rows if r["set_id"] == set_id]
    sys.exit(f"{set_id}: not found in data/processed/*/lines.jsonl; run pipeline.beats first")


def template(set_id: str) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    out = WORK / f"{set_id}.txt"
    if out.exists():
        sys.exit(f"{out} already exists; refusing to overwrite your labels")
    lines = load_lines(set_id)
    body = []
    for ln in lines:
        laugh = f"  ⟨L{ln['label']}⟩" if ln["label"] else ""
        body.append(f"[{ln['line_idx']:04d}] {ln['text']}{laugh}")
    out.write_text(HEADER.format(set_id=set_id, mark=MARK) + "\n".join(body) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(lines)} lines). Mark bit starts with '{MARK}' and run import.")


def import_labels(set_id: str) -> None:
    src = WORK / f"{set_id}.txt"
    if not src.exists():
        sys.exit(f"{src} missing; run `template` first")
    expected = [ln["line_idx"] for ln in load_lines(set_id)]
    seen, starts, pending = [], [], None
    for raw in src.read_text(encoding="utf-8").splitlines():
        if raw.startswith("#") or not raw.strip():
            continue
        if raw.strip().startswith(MARK):
            pending = raw.strip()[len(MARK):].strip()  # "" for a bare marker; None means no marker
            continue
        m = LINE.match(raw)
        if not m:
            sys.exit(f"unrecognized line (edited by accident?): {raw[:40]!r}")
        idx = int(m.group(1))
        seen.append(idx)
        if pending is not None or not starts and not seen[:-1]:
            starts.append({"line": idx, **({"note": pending} if pending else {})})
        pending = None
    if seen != expected:
        sys.exit("line numbers were edited, removed or reordered; restore them and re-import")
    if starts and starts[0]["line"] != expected[0]:
        starts.insert(0, {"line": expected[0]})
    LABELS.mkdir(parents=True, exist_ok=True)
    out = LABELS / f"{set_id}.yaml"
    out.write_text(yaml.safe_dump({"set_id": set_id, "unit": "line_idx", "n_lines": len(expected),
                                   "bit_starts": starts}, sort_keys=False, allow_unicode=True),
                   encoding="utf-8")
    print(f"{set_id}: {len(starts)} bits -> {out}")


def main() -> None:
    if len(sys.argv) != 3 or sys.argv[1] not in ("template", "import"):
        sys.exit(__doc__)
    {"template": template, "import": import_labels}[sys.argv[1]](sys.argv[2])


if __name__ == "__main__":
    main()
