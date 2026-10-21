"""S0: PDF -> transcript body text with site chrome removed.

Usage: python -m pipeline.ingest
Writes data/interim/<set_id>.body.txt and <set_id>.meta.json.
"""
import json
import re

import pymupdf

from pipeline.common import INTERIM, RAW, parse_filename

# Browser print headers/footers repeated on every page.
PAGE_CHROME = [
    re.compile(r"^https?://\S+$"),
    re.compile(r"^\d+/\d+$"),
    re.compile(r"\|\s*Scraps from the loft\s*$", re.I),
    re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}"),
]
PRINT_MARGIN = 0.035
SECTION_LABEL = "Stand-Up Transcripts"
INFOBOX = re.compile(r"^SPECIAL\b.*\bYEAR \d{4}$")
BLURB = re.compile(
    r"\b(released|recorded|premiere[sd]?|filmed|taped|streaming|netflix|hbo|prime video|"
    r"youtube|comedy special|stand-up special|\d+-minute)\b",
    re.I,
)
LEADING_WINDOW = 4
END_ANCHORS = ("Share this article", "Leave a Comment")


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


def extract_body(paras: list[str], meta: dict) -> list[str]:
    # Site synopsis sits above the "ARCHIVE Stand-Up Transcripts" breadcrumb; body follows it.
    start = next((i + 1 for i, p in enumerate(paras) if SECTION_LABEL in p), None)
    if start is None:
        raise ValueError("section label not found; page layout changed?")
    end = next((i for i in range(start, len(paras)) if paras[i] in END_ANCHORS), None)
    if end is None:
        raise ValueError("end anchor not found; page layout changed?")
    body = paras[start:end]
    # Post infobox, title heading and site blurb precede the performance; they name the
    # special or comedian, which the opening lines of a set almost never do verbatim.
    names = (meta["title"].lower(), meta["comedian"].lower())
    dropped = 0
    while body and dropped < LEADING_WINDOW and (
        INFOBOX.match(body[0]) or BLURB.search(body[0]) or any(n in body[0].lower() for n in names)
    ):
        body.pop(0)
        dropped += 1
    meta["leading_dropped"] = dropped
    return body


def main() -> None:
    INTERIM.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(RAW.glob("*.pdf"))
    for pdf in pdfs:
        meta = parse_filename(pdf.name)
        doc = pymupdf.open(pdf)
        paras = [p for page in doc for p in page_paragraphs(page)]
        body = extract_body(paras, meta)
        text = "\n".join(body)
        (INTERIM / f"{meta['set_id']}.body.txt").write_text(text, encoding="utf-8")
        meta |= {"source_pdf": pdf.name, "pages": doc.page_count, "paragraphs": len(body),
                 "words": len(text.split())}
        (INTERIM / f"{meta['set_id']}.meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        print(f"{meta['set_id']:55} paras={len(body):4} words={meta['words']:6}")
    print(f"\n{len(pdfs)} sets -> {INTERIM}")


if __name__ == "__main__":
    main()
