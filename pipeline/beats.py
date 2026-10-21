"""S3: clean text + reaction events -> beats (one training example each).

Usage: python -m pipeline.beats
Reads  data/interim/<set>.clean.txt, .events.jsonl and data/processed/datasets.json.
Writes data/processed/<dataset>/beats.jsonl and data/interim/<set>.labeled.txt (debug view).

A beat is the text between two audience reactions; its label is the reaction that follows.
Long reaction-free stretches are split into sentence groups: in a densely tagged set, text
that got no laugh is a real level-0 example. In structure_only sets every label is null,
because a missing marker there does not mean "no laugh".
"""
import json
import re
import sys

from pipeline.common import INTERIM, PROCESSED

MAX_BEAT_WORDS = 80
SENTENCE_END = re.compile(r"(?<=[.!?…])[\"”’)]*\s+|\n+")
# A laugh after a one-word line ("Yeah.") belongs to the punchline before it.
MIN_LINE_WORDS = 3
# Words of preceding text given as context; becomes bit-scoped once S4 exists.
CONTEXT_WORDS = 150


def load_events(set_id: str) -> list[dict]:
    with open(INTERIM / f"{set_id}.events.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def sentence_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    spans, cur = [], start
    for m in SENTENCE_END.finditer(text, start, end):
        if m.end() > cur:
            spans.append((cur, m.end()))
            cur = m.end()
    if cur < end:
        spans.append((cur, end))
    return spans


def chunk(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Split [start, end) at sentence ends into pieces of at most ~MAX_BEAT_WORDS words."""
    pieces, cur_start, cur_words = [], start, 0
    for s, e in sentence_spans(text, start, end):
        w = len(text[s:e].split())
        if cur_words and cur_words + w > MAX_BEAT_WORDS:
            pieces.append((cur_start, s))
            cur_start, cur_words = s, 0
        cur_words += w
    pieces.append((cur_start, end))
    return [(s, e) for s, e in pieces if text[s:e].strip()]


def build_beats(set_id: str, clean: str, events: list[dict], labeled: bool) -> list[dict]:
    reactions = [e for e in events if e["category"] == "reaction" and e.get("source") == "audience"]
    others = [e for e in events if e["category"] in ("act_out", "act_out_end", "song")]
    beats, prev = [], 0
    # Sentinel end-of-set "reaction" with no label closes the final stretch.
    for r in reactions + [None]:
        end = r["offset"] if r else len(clean)
        pieces = chunk(clean, prev, end)
        for i, (s, e) in enumerate(pieces):
            closes = r is not None and i == len(pieces) - 1
            level = (r["level"] if closes else 0) if labeled else None
            inside = [o for o in others if s <= o["offset"] < e or (o["offset"] == e and not closes)]
            beats.append({
                "set_id": set_id,
                "span": [s, e],
                "text": " ".join(clean[s:e].split()),
                "words": len(clean[s:e].split()),
                "label": level,
                "flags": (r["flags"] if closes else []),
                "reaction": (r["raw"] if closes else None),
                "interrupt": bool(closes and r["interrupt"]),
                "split": "reaction" if closes else ("chunk" if r is not None or i < len(pieces) - 1 else "end"),
                "act_outs": sum(o["category"] == "act_out" for o in inside),
                "song": any(o["category"] == "song" for o in inside),
                "rel_pos": round(s / max(len(clean), 1), 4),
            })
        prev = end
    for i, b in enumerate(beats):
        b["beat_idx"] = i
    return beats


def build_lines(clean: str, beats: list[dict]) -> list[dict]:
    """Per-sentence laugh labels: the line a reaction follows gets its level, setup lines get 0.

    Fixes the beat-level shortcut where beat length alone predicts the label (beats end at
    laughs, so almost all are positive and only long chunks are negative).
    """
    lines = []
    for b in beats:
        spans = sentence_spans(clean, *b["span"])
        spans = [(s, e) for s, e in spans if clean[s:e].strip()]
        if not spans:
            continue
        # Fold a too-short final line into the one before it so the label lands on the punchline.
        if len(spans) > 1 and len(clean[spans[-1][0]:spans[-1][1]].split()) < MIN_LINE_WORDS:
            spans[-2:] = [(spans[-2][0], spans[-1][1])]
        for i, (s, e) in enumerate(spans):
            last = i == len(spans) - 1
            if b["label"] is None:
                label = None
            else:
                label = b["label"] if last else 0
            ctx_words = clean[:s].split()[-CONTEXT_WORDS:]
            lines.append({
                "set_id": b["set_id"],
                "beat_idx": b["beat_idx"],
                "span": [s, e],
                "text": " ".join(clean[s:e].split()),
                "words": len(clean[s:e].split()),
                "label": label,
                "flags": b["flags"] if last else [],
                "interrupt": bool(last and b["interrupt"]),
                "context": " ".join(ctx_words),
                "rel_pos": round(s / max(len(clean), 1), 4),
            })
    for i, line in enumerate(lines):
        line["line_idx"] = i
    return lines


def labeled_view(clean: str, events: list[dict], beats: list[dict]) -> str:
    """Human-readable check file: normalized levels inline. Never a training input."""
    inserts: list[tuple[int, str]] = []
    for e in events:
        if e["category"] == "reaction" and e.get("source") == "audience":
            flags = "".join(f" {f}" for f in e["flags"])
            inserts.append((e["offset"], f" ⟨L{e['level']}{flags}⟩ "))
        elif e["category"] == "song":
            inserts.append((e["offset"], " ⟨sings⟩ "))
        elif e["category"] == "act_out":
            inserts.append((e["offset"], f" ⟨act: {e['raw']}⟩ "))
        elif e["category"] == "act_out_end":
            inserts.append((e["offset"], " ⟨/act⟩ "))
    inserts += [(b["span"][1], " ⟨·⟩\n") for b in beats if b["split"] == "chunk"]
    out, pos = [], 0
    for off, tag in sorted(inserts, key=lambda t: t[0]):
        out.append(clean[pos:off].rstrip(" "))  # offsets sit after the space; avoid "word  ⟨L2⟩"
        out.append(tag)
        pos = off
    out.append(clean[pos:])
    return "".join(out)


def main() -> None:
    manifest_path = PROCESSED / "datasets.json"
    if not manifest_path.exists():
        sys.exit("data/processed/datasets.json missing; run `python -m pipeline.markers` first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for dataset in ("laugh_labeled", "structure_only"):
        out_dir = PROCESSED / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        all_beats, all_lines = [], []
        for set_id in manifest[dataset]:
            clean = (INTERIM / f"{set_id}.clean.txt").read_text(encoding="utf-8")
            events = load_events(set_id)
            beats = build_beats(set_id, clean, events, labeled=dataset == "laugh_labeled")
            # Spans must tile the clean text exactly, or labels drift onto the wrong text.
            covered = "".join(clean[s:e] for s, e in (b["span"] for b in beats))
            if covered.split() != clean.split():
                sys.exit(f"{set_id}: beat spans do not tile the clean text")
            (INTERIM / f"{set_id}.labeled.txt").write_text(labeled_view(clean, events, beats), encoding="utf-8")
            lines = build_lines(clean, beats)
            all_beats += beats
            all_lines += lines
            if dataset == "laugh_labeled":
                rate = sum((ln["label"] or 0) >= 2 for ln in lines) / max(len(lines), 1)
                print(f"{set_id:50} beats={len(beats):4} lines={len(lines):5} line_laugh_rate={rate:.2f}")
            else:
                print(f"{set_id:50} beats={len(beats):4} lines={len(lines):5} (unlabeled)")
        for name, rows in (("beats", all_beats), ("lines", all_lines)):
            with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\n{dataset}: {len(all_beats)} beats, {len(all_lines)} lines -> {out_dir}")
        if dataset == "laugh_labeled":
            for name, rows in (("beats", all_beats), ("lines", all_lines)):
                pos = [r["words"] for r in rows if r["label"] >= 2]
                neg = [r["words"] for r in rows if r["label"] < 2]
                print(f"  {name:5} laugh rate {len(pos) / len(rows):.2f} | median words "
                      f"laugh={median(pos)} no-laugh={median(neg)}")
        print()


def median(xs: list[int]) -> int:
    return sorted(xs)[len(xs) // 2] if xs else 0


if __name__ == "__main__":
    main()
