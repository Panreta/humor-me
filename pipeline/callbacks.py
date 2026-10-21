"""Callback candidates without an LLM: rare terms that vanish and then return much later in a set.

A callback reuses a specific image or name from earlier (a front door, a town, a person), after
other material in between. So a candidate is a term that is rare across the whole corpus (not
everyday words every comic uses), appears in a set, disappears for a long stretch, and returns.
The LLM confirms or rejects candidates; this pass only has to have good recall.
"""
import json
import math
import re
from collections import defaultdict

from pipeline.common import PROCESSED

MIN_GAP_LINES = 30      # ~2 typical bits: absent this long, it's a return, not the same bit
MAX_SET_OCCURRENCES = 8  # very frequent in the set = the set's general subject, not a callback hook
MAX_HINTS = 25
STOP = set("""
about above after again against also always another anything around because been before being
below between both called came cause come comes coming could didn doesn doing done down during
each else even ever every from gets getting give goes going gonna gotta great guys have having here
just know like little look looking made make many maybe more most much must myself never next
nothing okay only other over people pretty probably really right said same says should since some
something still such sure take tell than thank thanks that thats their them then there these they
thing things think this those though thought through time today told took true very want wanted
wants were what whatever when where which while whole will with without would yeah years your yours
youre doesnt dont cant wont isnt wasnt arent theyre were thats yall gonna kind back well good
""".split())


def terms(text: str) -> set[str]:
    words = [w for w in re.findall(r"[a-z][a-z']+", text.lower().replace("’", "'"))
             if len(w) >= 4 and w.replace("'", "") not in STOP]
    words = [w.replace("'", "") for w in words]
    return set(words) | {f"{a} {b}" for a, b in zip(words, words[1:])}


def corpus_df() -> tuple[dict[str, int], int]:
    """Number of sets each term appears in, across both datasets."""
    per_set = defaultdict(set)
    for dataset in ("laugh_labeled", "structure_only"):
        path = PROCESSED / dataset / "lines.jsonl"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for row in map(json.loads, f):
                    per_set[row["set_id"]] |= terms(row["text"])
    df = defaultdict(int)
    for ts in per_set.values():
        for t in ts:
            df[t] += 1
    return df, len(per_set)


def candidates(lines: list[dict], df: dict[str, int], n_sets: int) -> list[dict]:
    """[{"line", "earlier_line", "term", "score"}], at most MAX_HINTS, best first."""
    order = [ln["line_idx"] for ln in lines]
    occ = defaultdict(list)  # term -> positions (index into order)
    for i, ln in enumerate(lines):
        for t in terms(ln["text"]):
            occ[t].append(i)
    best: dict[int, dict] = {}
    for t, pos in occ.items():
        if not 2 <= len(pos) <= MAX_SET_OCCURRENCES:
            continue
        idf = math.log((n_sets + 1) / (df.get(t, 1) + 0.5))
        if idf < 1.0:  # appears in most sets: an everyday word
            continue
        score = idf * (1.5 if " " in t else 1.0)
        for prev, cur in zip(pos, pos[1:]):
            if cur - prev >= MIN_GAP_LINES:
                # Point at the last mention before the gap: where the image was last in play.
                hint = {"line": order[cur], "earlier_line": order[prev], "term": t, "score": round(score, 2)}
                if order[cur] not in best or best[order[cur]]["score"] < score:
                    best[order[cur]] = hint
    return sorted(best.values(), key=lambda h: -h["score"])[:MAX_HINTS]


def render_hints(hints: list[dict]) -> str:
    if not hints:
        return ""
    rows = "\n".join(f"- [{h['line']:04d}] reuses \"{h['term']}\" first seen at [{h['earlier_line']:04d}]"
                     for h in sorted(hints, key=lambda h: h["line"]))
    return ("\n\nCALLBACK CANDIDATES (automatic, often wrong; confirm only if the later bit really "
            "returns to the earlier image or premise after other material):\n" + rows)
