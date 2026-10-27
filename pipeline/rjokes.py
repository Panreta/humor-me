"""R1: rJokes -> deduped, filtered, era-normalized jokes with coarse topic flags.

Usage:
    python -m pipeline.rjokes            # reads data/external/rJokesData/data, writes data/processed/rjokes/

Split files carry the paper's 0-11 score (~ln upvotes) but no date; preprocessed.csv has raw
upvotes and a date. Rows are joined on whitespace-normalized text to get the year.

Writes <split>.jsonl and stats.json. Text stays local: Reddit terms forbid redistribution.
"""
import bisect
import collections
import csv
import gzip
import hashlib
import json
import re
import sys
from datetime import datetime, timezone

from pipeline.common import PROCESSED, ROOT

SRC = ROOT / "data" / "external" / "rJokesData" / "data"
OUT = PROCESSED / "rjokes"
SPLITS = ("train", "dev", "test")  # order = priority when a repost spans splits
MIN_WORDS, MAX_WORDS = 4, 200  # under 4 is a title fragment; over 200 is a story, not a joke

# Coarse topic flags, not a safety verdict: they mark jokes for the LLM labeler and for the
# teens filter. Word stems, so expect false positives ("black coffee"); never used to drop rows.
FLAGS = {
    "protected": r"\b(black|white|asian|mexican|jew|jewish|muslim|islam|christian|catholic|"
                 r"gay|lesbian|trans|race|racist|immigrant|arab|indian|chinese|african|blonde|"
                 r"disabled|retard\w*|autis\w*|deaf|blind|midget)\b",
    "body": r"\b(fat|obese|ugly|skinny|bald|short|boobs?|breasts?|dick|penis|vagina|weight)\b",
    "sexual": r"\b(sex|sexy|hooker|prostitute|porn\w*|virgin|orgasm|blowjob|condom|horny)\b",
    "alcohol_drugs": r"\b(beer|drunk|wine|vodka|whiskey|bar|weed|cocaine|coke|stoned|high)\b",
    "death_violence": r"\b(dead|death|die[sd]?|kill\w*|murder\w*|rape\w*|suicide|abortion)\b",
}
FLAG_RES = {k: re.compile(v, re.I) for k, v in FLAGS.items()}


def norm(s: str | None) -> str:
    return " ".join((s or "").split())


def flags(text: str) -> list[str]:
    return [k for k, rx in FLAG_RES.items() if rx.search(text)]


def joke_id(text: str) -> str:
    return hashlib.sha1(text.lower().encode()).hexdigest()[:12]


def dedupe(rows: list[dict]) -> list[dict]:
    """Collapse reposts of the same text. The first split in SPLITS order owns the joke so a
    repost can't leak train text into test; score is the max over posts (the version that
    found its audience), and n_posts is kept because repost score spread measures noise."""
    groups: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        groups[r["text"].lower()].append(r)
    out = []
    for g in groups.values():
        g.sort(key=lambda r: (SPLITS.index(r["split"]), -r["score"]))
        best = dict(g[0])
        best["score"] = max(r["score"] for r in g)
        best["n_posts"] = len(g)
        best["score_spread"] = best["score"] - min(r["score"] for r in g)
        years = [r["year"] for r in g if r["year"]]
        best["year"] = min(years) if years else None
        out.append(best)
    return out


def year_percentiles(rows: list[dict]) -> None:
    """Score percentile within the post's year (mid-rank for ties). r/Jokes grew ~1000x over
    2008-2019, so a raw score of 5 means different things in different years."""
    by_year: dict = collections.defaultdict(list)
    for r in rows:
        by_year[r["year"]].append(r["score"])
    for v in by_year.values():
        v.sort()
    for r in rows:
        v = by_year[r["year"]]
        lo, hi = bisect.bisect_left(v, r["score"]), bisect.bisect_right(v, r["score"])
        r["pct"] = round((lo + hi) / 2 / len(v), 4)


def load_dates() -> dict[str, int]:
    csv.field_size_limit(10**8)
    years: dict[str, int] = {}
    with gzip.open(SRC / "preprocessed.csv.gz", "rt", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["date"]:
                ts = datetime.fromtimestamp(float(r["date"]), timezone.utc)
                years.setdefault(norm(r["joke"]), ts.year)
    return years


def load_splits(years: dict[str, int]) -> list[dict]:
    rows = []
    for split in SPLITS:
        with gzip.open(SRC / f"{split}.tsv.gz", "rt", encoding="utf-8") as f:
            for line in f:
                score, text = line.rstrip("\n").split("\t", 1)
                text = norm(text)
                rows.append({"split": split, "text": text, "score": int(score), "year": years.get(text)})
    return rows


def build(rows: list[dict]) -> tuple[list[dict], dict]:
    stats = {"rows_in": len(rows)}
    rows = [r for r in rows if r["text"]]
    rows = dedupe(rows)
    stats["after_dedupe"] = len(rows)
    rows = [r for r in rows if MIN_WORDS <= len(r["text"].split()) <= MAX_WORDS]
    stats["after_length"] = len(rows)
    stats["no_year"] = sum(r["year"] is None for r in rows)
    year_percentiles(rows)
    for r in rows:
        r["id"] = joke_id(r["text"])
        r["n_words"] = len(r["text"].split())
        r["flags"] = flags(r["text"])
    stats["per_split"] = dict(collections.Counter(r["split"] for r in rows))
    stats["flag_rate"] = {k: round(sum(k in r["flags"] for r in rows) / len(rows), 4) for k in FLAGS}
    stats["unflagged_rate"] = round(sum(not r["flags"] for r in rows) / len(rows), 4)
    stats["score_hist"] = dict(sorted(collections.Counter(r["score"] for r in rows).items()))
    reposted = [r for r in rows if r["n_posts"] > 1]
    stats["reposted"] = len(reposted)
    if reposted:
        spreads = sorted(r["score_spread"] for r in reposted)
        stats["repost_spread_median"] = spreads[len(spreads) // 2]
    return rows, stats


def main() -> None:
    if not SRC.exists():
        sys.exit(f"missing {SRC}; git clone https://github.com/orionw/rJokesData data/external/rJokesData")
    rows, stats = build(load_splits(load_dates()))
    OUT.mkdir(parents=True, exist_ok=True)
    fields = ("id", "split", "text", "score", "pct", "year", "n_words", "n_posts", "score_spread", "flags")
    for split in SPLITS:
        with open(OUT / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                if r["split"] == split:
                    f.write(json.dumps({k: r[k] for k in fields}, ensure_ascii=False) + "\n")
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
