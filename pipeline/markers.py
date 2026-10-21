"""S1+S2: extract bracketed markers from bodies and classify them via the lexicon.

Usage:
    python -m pipeline.markers            # classify; exits non-zero on unmapped reaction strings
    python -m pipeline.markers --draft    # print proposed lexicon entries for unmapped strings

Writes data/interim/<set_id>.clean.txt (markers removed) and <set_id>.events.jsonl.
Only reaction-like strings must be in the lexicon; a laugh level is never guessed at run time.
Other stage directions are bucketed by keyword (act_out / music / direction).
"""
import argparse
import json
import re
import sys

import yaml

from pipeline.common import INTERIM, LEXICON

BRACKET = re.compile(r"[\[\(]([^\[\]\(\)\n]{2,60})[\]\)]")
NOTE = "♪"
SONG_SPAN = re.compile(r"♪[^♪\n]*♪")
SONG_LABEL = "[sings]"
MIXED_MIN_WORDS = 4  # a song paragraph with this many words outside ♪ spans also has speech
REACTION = re.compile(
    r"laugh|chuckl|giggl|snicker|titter|applau|clap|cheer|whoop|whistl|ovation|roar|"
    r"groan|gasp|aww|boo\b|booing|scream|murmur|sigh",
    re.I,
)
ACT_OUT = re.compile(
    r"imitat|impression|voice|accent|mimic|mouth|sing|grunt|mumbl|whisper|shout|yell|"
    r"stammer|stutter|scoff|growl|mock|squeal|howl|panting|gibberish",
    re.I,
)
ACT_OUT_END = re.compile(r"^(normal|regular|own|natural) voice$|^as (self|himself|herself)\b")
MUSIC = re.compile(r"music|song|playing|band|♪", re.I)

KIND_LEVEL = {"chuckle": 1, "laugh": 2, "applause": 4, "laugh_applause": 4}
EDGE = 0.02  # entrance/exit window as a fraction of the set
MERGE_GAP_WORDS = 3  # adjacent audience reactions closer than this are one reaction
DENSE_WORDS_PER_LAUGH = 150  # dense sets sit at 26-80; sparse ones at 480+


def normalize(raw: str) -> str:
    return re.sub(r"\s+", " ", raw.lower()).strip(" .,;:!")


def load_lexicon() -> dict:
    if not LEXICON.exists():
        return {}
    return (yaml.safe_load(LEXICON.read_text(encoding="utf-8")) or {}).get("reactions", {})


def classify(raw: str, lexicon: dict) -> dict | None:
    """Returns event fields, or None if this is an unmapped reaction string."""
    if REACTION.search(raw):
        entry = lexicon.get(raw)
        if entry is None:
            return None
        return {"category": "reaction", **entry}
    if raw == "sings":
        return {"category": "song"}  # performer sings mid-set (from strip_songs or a transcriber)
    if ACT_OUT_END.match(raw):
        return {"category": "act_out_end"}  # back in own voice: closes a character span
    if ACT_OUT.search(raw):
        return {"category": "act_out"}
    if MUSIC.search(raw):
        return {"category": "music"}
    return {"category": "direction"}


def level_of(ev: dict) -> int:
    if ev.get("source") != "audience" or ev.get("kind") not in KIND_LEVEL:
        return 0
    # Applause/cheers alone are often for a hometown or a sentiment, not a joke; they only
    # count as an applause break when merged with an adjacent laugh (see merge_adjacent).
    if ev["kind"] == "applause":
        return 0
    level = KIND_LEVEL[ev["kind"]]
    if ev["kind"] == "laugh" and ev.get("intense"):
        level = 3
    return level


def merge_adjacent(events: list[dict], clean: str) -> list[dict]:
    """Back-to-back audience reactions are one sustained reaction; two laughs in a row = big laugh."""
    merged: list[dict] = []
    for ev in events:
        prev = merged[-1] if merged else None
        if (
            prev
            and prev.get("source") == ev.get("source") == "audience"
            and len(clean[prev["offset"]:ev["offset"]].split()) < MERGE_GAP_WORDS
        ):
            both_laugh = prev.get("kind") in ("laugh", "laugh_applause") and ev.get("kind") in ("laugh", "laugh_applause")
            laugh_level = max(prev["level"], ev["level"])
            flags = set(prev["flags"]) | set(ev["flags"])
            if laugh_level >= 2 and "applause" in flags:
                laugh_level = 4  # laugh rolling into applause = applause break
            elif both_laugh and laugh_level >= 2:
                laugh_level = max(laugh_level, 3)
            prev["level"] = laugh_level
            prev["flags"] = sorted(flags)
            prev["raw"] += " | " + ev["raw"]
            continue
        merged.append(ev)
    return merged


def _strip_notes(para: str) -> str:
    """Remove sung text from a paragraph that also has speech; keep the speech."""
    para = SONG_SPAN.sub(" ", para)
    if NOTE in para:  # unpaired note: the song runs to/from a paragraph edge
        i = para.index(NOTE)
        para = para[i + 1:] if i < len(para) - i else para[:i]
    return para.replace(NOTE, " ")


def strip_songs(body: str) -> str:
    """Drop walk-on/outro songs entirely; label mid-set singing as [sings]. Lyrics are never kept.

    Songs are found at paragraph level because lyric lines often lack a closing note.
    """
    paras = body.split("\n")
    is_song = [NOTE in p for p in paras]
    spoken = [i for i, p in enumerate(paras) if not is_song[i] and BRACKET.sub("", p).strip()]
    first, last = (spoken[0], spoken[-1]) if spoken else (len(paras), -1)
    out: list[str] = []
    for i, p in enumerate(paras):
        if not is_song[i]:
            out.append(p)
            continue
        edge = i < first or i > last
        speech = _strip_notes(p)
        if len(BRACKET.sub("", speech).split()) >= MIXED_MIN_WORDS:
            out.append(speech if edge else f"{SONG_LABEL} {speech}")
        elif not edge and (not out or out[-1] != SONG_LABEL):
            out.append(SONG_LABEL)  # one label per run of song-only paragraphs
    return "\n".join(out)


def _append_clean(out: list[str], segment: str) -> None:
    # Collapse whitespace per segment and at the seam left by a removed marker, so the running
    # length is the final clean-text offset. (Collapsing once at the end shifted every later
    # offset by one char per removed marker.)
    segment = re.sub(r"[ \t]+", " ", segment)
    if out and out[-1] and out[-1][-1] in " \n" and segment.startswith(" "):
        segment = segment[1:]
    out.append(segment)


def process(set_id: str, body: str, lexicon: dict, unknown: dict) -> tuple[str, list[dict]]:
    parts, events, pos = [], [], 0
    for m in BRACKET.finditer(body):
        _append_clean(parts, body[pos:m.start()])
        length = sum(map(len, parts))
        pos = m.end()
        raw = normalize(m.group(1))
        fields = classify(raw, lexicon)
        if fields is None:
            unknown.setdefault(raw, set()).add(set_id)
            continue
        rel = m.start() / max(len(body), 1)
        # A marker that interrupts a sentence means the laugh landed before the line finished.
        before = body[max(0, m.start() - 2):m.start()].strip()
        after = body[m.end():m.end() + 2].lstrip()
        interrupt = bool(before) and before[-1] not in ".?!…\"”" and after[:1].islower()
        ev = {"offset": length, "rel_pos": round(rel, 4), "raw": raw, "interrupt": interrupt, **fields}
        ev["level"] = level_of(ev)
        ev["flags"] = [f for f in ("groan", "gasp") if ev.get("kind") == f]
        # Entrance/exit applause says nothing about a joke, so it gets no flag either.
        if ev.get("source") == "audience" and ev.get("kind") in ("applause", "laugh_applause") and EDGE < rel < 1 - EDGE:
            ev["flags"].append("applause")
        events.append(ev)
    _append_clean(parts, body[pos:])
    clean = "".join(parts)
    return clean, merge_adjacent(events, clean)


def draft_entry(raw: str) -> dict:
    """Heuristic proposal only, for a human to review before it enters the lexicon."""
    audience = re.search(r"audience|crowd|people|members?|someone|man|woman|laughter|applause|ovation", raw)
    named = re.match(r"^[a-z]+ (laughs|chuckles|giggles)", raw) and not audience
    if "groan" in raw:
        kind = "groan"
    elif "gasp" in raw:
        kind = "gasp"
    elif re.search(r"laugh", raw) and re.search(r"applau|cheer|clap|whoop|whistl", raw):
        kind = "laugh_applause"
    elif re.search(r"laugh|roar", raw):
        kind = "laugh"
    elif re.search(r"chuckl|giggl|snicker|titter", raw):
        kind = "chuckle"
    elif re.search(r"applau|cheer|clap|whoop|whistl|ovation", raw):
        kind = "applause"
    else:
        kind = "other"
    if named:
        source = "person"
    elif audience or kind in ("applause", "laugh_applause") or raw == "laughter":
        source = "audience"
    else:
        source = "performer"  # subtitle convention: unattributed sounds are the speaker's
    entry = {"kind": kind, "source": source}
    if re.search(r"roar|wild|loud|continues|uproar|big|huge|hysterical|erupt", raw):
        entry["intense"] = True
    return entry


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", action="store_true")
    args = ap.parse_args()

    lexicon = load_lexicon()
    unknown: dict[str, set] = {}
    results = {}
    for body_path in sorted(INTERIM.glob("*.body.txt")):
        set_id = body_path.name.removesuffix(".body.txt")
        body = strip_songs(body_path.read_text(encoding="utf-8"))
        results[set_id] = process(set_id, body, lexicon, unknown)

    if unknown:
        if args.draft:
            proposal = {raw: draft_entry(raw) for raw in sorted(unknown)}
            print(yaml.safe_dump({"reactions": proposal}, sort_keys=False, allow_unicode=True))
            return
        print(f"{len(unknown)} unmapped reaction strings; add them to {LEXICON.name}:", file=sys.stderr)
        for raw, sets in sorted(unknown.items(), key=lambda kv: -len(kv[1])):
            print(f"  {raw!r}  ({len(sets)} sets)", file=sys.stderr)
        sys.exit(1)

    for set_id, (clean, events) in results.items():
        (INTERIM / f"{set_id}.clean.txt").write_text(clean, encoding="utf-8")
        with open(INTERIM / f"{set_id}.events.jsonl", "w", encoding="utf-8") as f:
            for ev in events:
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        reactions = [e for e in events if e["category"] == "reaction" and e["source"] == "audience"]
        hist = [sum(e["level"] == lv for e in reactions) for lv in range(5)]
        laughs = sum(1 <= e["level"] <= 3 for e in reactions)
        words_per_laugh = len(clean.split()) / max(laughs, 1)
        meta_path = INTERIM / f"{set_id}.meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        # Sparse transcribers leave most laughs untagged, so absent markers would read as
        # "no laugh" and poison the reward labels; those sets are structure-only.
        meta |= {"audience_laughs": laughs, "words_per_laugh": round(words_per_laugh),
                 "laugh_labeled": words_per_laugh <= DENSE_WORDS_PER_LAUGH}
        meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
        print(f"{set_id:50} audience={len(reactions):4} levels0-4={hist} "
              f"act_out={sum(e['category'] == 'act_out' for e in events):3} "
              f"labeled={meta['laugh_labeled']}")


if __name__ == "__main__":
    main()
