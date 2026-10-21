"""Eyeball checks for S1+S2 output. Prints transcript context locally; don't paste it into issues/commits.

Usage:
    python scripts/inspect_events.py laughs  <set_id> [--n 10] [--chars 100]
    python scripts/inspect_events.py actouts <set_id>
    python scripts/inspect_events.py sets
"""
import argparse
import collections
import json
import pathlib

INTERIM = pathlib.Path(__file__).resolve().parent.parent / "data" / "interim"


def load(set_id: str) -> tuple[str, list[dict]]:
    clean = (INTERIM / f"{set_id}.clean.txt").read_text(encoding="utf-8")
    with open(INTERIM / f"{set_id}.events.jsonl", encoding="utf-8") as f:
        return clean, [json.loads(line) for line in f]


def laughs(set_id: str, n: int, chars: int) -> None:
    """Text right before each audience laugh: the line before should read like a punchline."""
    clean, events = load(set_id)
    shown = 0
    for e in events:
        if e.get("source") != "audience" or not 1 <= e.get("level", 0) <= 4:
            continue
        before = clean[max(0, e["offset"] - chars):e["offset"]].replace("\n", " ")
        flag = " [interrupt]" if e["interrupt"] else ""
        print(f"L{e['level']}{flag:12} …{before}  ⟵ [{e['raw']}]\n")
        shown += 1
        if shown >= n:
            break


def actouts(set_id: str) -> None:
    """Distinct act_out markers: should be voices, impressions, sound effects."""
    _, events = load(set_id)
    counts = collections.Counter(e["raw"] for e in events if e["category"] == "act_out")
    for raw, c in counts.most_common():
        print(f"{c:4}  {raw}")
    print(f"\n{sum(counts.values())} act_out markers, {len(counts)} distinct")


def sets() -> None:
    for meta_path in sorted(INTERIM.glob("*.meta.json")):
        m = json.loads(meta_path.read_text(encoding="utf-8"))
        print(f"{m['set_id']:50} labeled={str(m.get('laugh_labeled')):5} words/laugh={m.get('words_per_laugh')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["laughs", "actouts", "sets"])
    ap.add_argument("set_id", nargs="?")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--chars", type=int, default=100)
    args = ap.parse_args()
    if args.cmd == "sets":
        sets()
    elif not args.set_id:
        ap.error("set_id required (see: inspect_events.py sets)")
    elif args.cmd == "laughs":
        laughs(args.set_id, args.n, args.chars)
    else:
        actouts(args.set_id)


if __name__ == "__main__":
    main()
