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
                "text": clean[s:e].strip(),
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
        out.append(clean[pos:off])
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
        all_beats = []
        for set_id in manifest[dataset]:
            clean = (INTERIM / f"{set_id}.clean.txt").read_text(encoding="utf-8")
            events = load_events(set_id)
            beats = build_beats(set_id, clean, events, labeled=dataset == "laugh_labeled")
            # Spans must tile the clean text exactly, or labels drift onto the wrong text.
            covered = "".join(clean[s:e] for s, e in (b["span"] for b in beats))
            if covered.split() != clean.split():
                sys.exit(f"{set_id}: beat spans do not tile the clean text")
            (INTERIM / f"{set_id}.labeled.txt").write_text(labeled_view(clean, events, beats), encoding="utf-8")
            all_beats += beats
            if dataset == "laugh_labeled":
                hist = [sum(b["label"] == lv for b in beats) for lv in range(5)]
                print(f"{set_id:50} beats={len(beats):4} labels0-4={hist}")
            else:
                print(f"{set_id:50} beats={len(beats):4} (unlabeled)")
        with open(out_dir / "beats.jsonl", "w", encoding="utf-8") as f:
            for b in all_beats:
                f.write(json.dumps(b, ensure_ascii=False) + "\n")
        words = [b["words"] for b in all_beats]
        summary = f"{dataset}: {len(all_beats)} beats, median {sorted(words)[len(words) // 2]} words"
        if dataset == "laugh_labeled":
            pos = sum((b["label"] or 0) >= 2 for b in all_beats)
            summary += f", laugh (level>=2) rate {pos / len(all_beats):.2f}"
        print(f"\n{summary} -> {out_dir / 'beats.jsonl'}\n")


if __name__ == "__main__":
    main()
