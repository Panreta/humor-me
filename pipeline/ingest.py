"""S0: PDF -> transcript body text with site chrome removed.

Usage:
    python -m pipeline.ingest             # trim every set per data/body_overrides.yaml
    python -m pipeline.ingest --suggest   # append proposed entries for sets not in the file

Each set's body is the region between the site breadcrumb and the comment section, trimmed by
explicit per-set counts. Heuristics only *propose* those counts; a human reviews them, so a
rerun never silently changes where a set starts or ends.

Writes data/interim/<set_id>.body.txt and <set_id>.meta.json.
"""
import argparse
import json
import re
import sys

import pymupdf
import yaml

from pipeline.common import INTERIM, OVERRIDES, RAW, parse_filename

# Browser print headers/footers repeated on every page.
PAGE_CHROME = [
    re.compile(r"^https?://\S+$"),
    re.compile(r"^\d+/\d+$"),
    re.compile(r"\|\s*Scraps from the loft\s*$", re.I),
    re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}"),
]
PRINT_MARGIN = 0.035
SECTION_LABEL = "Stand-Up Transcripts"
END_ANCHORS = ("Share this article", "Leave a Comment")

# --- suggestion heuristics (only used by --suggest) ---
INFOBOX = re.compile(r"^SPECIAL\b.*\bYEAR \d{4}$")
BLURB = re.compile(
    r"\b(released|recorded|premiere[sd]?|filmed|taped|streaming|netflix|hbo|prime video|"
    r"youtube|comedy special|stand-up special|\d+-minute)\b",
    re.I,
)
LEADING_WINDOW = 4
HEADINGS = {"transcript", "full transcript"}
HEADING_WINDOW = 40
DISCLAIMER = "Note for Students"
SEPARATOR = "* * *"
SEPARATOR_WINDOW = 15
# Reaction markers only: the post title above a synopsis has "(2023)"-style parentheses too.
REACTION_MARKER = re.compile(r"[\[\(][^\]\)\n]{0,30}(laugh|chuckl|applau|cheer|clap)[^\]\)\n]{0,30}[\]\)]", re.I)
MORE_FOOTER = re.compile(r"^MORE\b.*Stand-Up Transcripts")  # both "MORE <name> S-U T" and "MORE S-U T <name>"


def page_paragraphs(page: pymupdf.Page) -> list[str]:
    # The printed page emits one block per visual line, so paragraphs are rebuilt from
    # vertical gaps: a gap wider than ~half a line height starts a new paragraph.
    paras, current, prev_bottom = [], [], None
    height = page.rect.height
    for x0, y0, x1, y1, text, _, kind in page.get_text("blocks", sort=True):
        # Print header (timestamp + title) and footer (URL + page no.) live in the outer ~3%.
        if kind != 0 or y1 < PRINT_MARGIN * height or y0 > (1 - PRINT_MARGIN) * height:
            continue
        line = re.sub(r"\s+", " ", text).strip()
        if not line or any(p.search(line) for p in PAGE_CHROME):
            continue
        if prev_bottom is not None and y0 - prev_bottom > 0.5 * (y1 - y0) and current:
            paras.append(" ".join(current))
            current = []
        current.append(line)
        prev_bottom = y1
    if current:
        paras.append(" ".join(current))
    return paras


def post_region(paras: list[str]) -> list[str]:
    """Paragraphs between the breadcrumb and the comment section; the trim counts index into this."""
    start = next((i + 1 for i, p in enumerate(paras) if SECTION_LABEL in p), None)
    if start is None:
        raise ValueError("section label not found; page layout changed?")
    end = next((i for i in range(start, len(paras)) if paras[i] in END_ANCHORS), None)
    if end is None:
        raise ValueError("end anchor not found; page layout changed?")
    return paras[start:end]


def suggest_trim(region: list[str], meta: dict) -> tuple[int, int, str]:
    """Propose (skip_leading, skip_trailing, rule) for a human to review."""
    trailing = 0
    while trailing < len(region) and MORE_FOOTER.match(region[-1 - trailing]):
        trailing += 1

    heading = next((i for i, p in enumerate(region[:HEADING_WINDOW]) if p.strip().lower() in HEADINGS), None)
    if heading is not None:
        # Newer posts put a "Transcript" heading (+ study disclaimer) right before the set.
        leading = heading + 1
        if leading < len(region) and region[leading].startswith(DISCLAIMER):
            leading += 1
            # The disclaimer can wrap across a page break; its tail starts lowercase.
            while leading < len(region) and region[leading][:1].islower():
                leading += 1
        return leading, trailing, "heading"

    # A "* * *" divider with no reactions above it separates the synopsis from the set; one
    # with reactions above it is a section break inside the performance.
    sep = next((i for i, p in enumerate(region[:SEPARATOR_WINDOW]) if p.strip() == SEPARATOR), None)
    if sep is not None and not REACTION_MARKER.search("\n".join(region[:sep])):
        return sep + 1, trailing, "separator"

    # Older posts: infobox, title and blurb name the special or comedian, which the opening
    # lines of a set almost never do verbatim.
    names = (meta["title"].lower(), meta["comedian"].lower())
    leading = 0
    while leading < min(LEADING_WINDOW, len(region)) and (
        INFOBOX.match(region[leading])
        or BLURB.search(region[leading])
        or any(n in region[leading].lower() for n in names)
    ):
        leading += 1
    rule = "heuristic, hit cap: verify" if leading == LEADING_WINDOW else "heuristic"
    return leading, trailing, rule


def load_overrides() -> dict:
    if not OVERRIDES.exists():
        return {}
    return yaml.safe_load(OVERRIDES.read_text(encoding="utf-8")) or {}


def append_suggestions(entries: list[tuple[str, int, int, str]]) -> None:
    lines = [
        f"{set_id}:  # {rule}\n  skip_leading: {lead}\n  skip_trailing: {trail}\n"
        for set_id, lead, trail, rule in entries
    ]
    with open(OVERRIDES, "a", encoding="utf-8") as f:
        f.write("\n" + "\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suggest", action="store_true")
    args = ap.parse_args()

    INTERIM.mkdir(parents=True, exist_ok=True)
    overrides = load_overrides()
    sets = []
    for pdf in sorted(RAW.glob("*.pdf")):
        meta = parse_filename(pdf.name)
        doc = pymupdf.open(pdf)
        region = post_region([p for page in doc for p in page_paragraphs(page)])
        meta |= {"source_pdf": pdf.name, "pages": doc.page_count}
        sets.append((meta, region))

    missing = [(m, r) for m, r in sets if m["set_id"] not in overrides]
    if missing:
        if args.suggest:
            entries = [(m["set_id"], *suggest_trim(r, m)) for m, r in missing]
            append_suggestions(entries)
            for set_id, lead, trail, rule in entries:
                print(f"proposed {set_id:50} leading={lead:2} trailing={trail}  ({rule})")
            print(f"\nReview {OVERRIDES.name}, then rerun without --suggest.")
            return
        print(f"{len(missing)} sets have no entry in {OVERRIDES.name}; run with --suggest:", file=sys.stderr)
        for m, _ in missing:
            print(f"  {m['set_id']}", file=sys.stderr)
        sys.exit(1)

    for meta, region in sets:
        trim = overrides[meta["set_id"]]
        lead, trail = trim.get("skip_leading", 0), trim.get("skip_trailing", 0)
        body = region[lead: len(region) - trail]
        text = "\n".join(body)
        (INTERIM / f"{meta['set_id']}.body.txt").write_text(text, encoding="utf-8")
        meta |= {"skip_leading": lead, "skip_trailing": trail, "paragraphs": len(body),
                 "words": len(text.split())}
        (INTERIM / f"{meta['set_id']}.meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        print(f"{meta['set_id']:55} paras={len(body):4} words={meta['words']:6}")
    print(f"\n{len(sets)} sets -> {INTERIM}")


if __name__ == "__main__":
    main()
