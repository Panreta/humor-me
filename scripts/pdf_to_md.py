"""Convert reference PDFs to Markdown for agent consumption.

Usage:
    python scripts/pdf_to_md.py references/paper.pdf [references/md/out.md]
    python scripts/pdf_to_md.py references/          # every PDF in the folder
"""
import pathlib
import re
import sys

import pymupdf
import pymupdf4llm

DEFAULT_OUT = pathlib.Path("references/md")
LINENO = re.compile(r"\*\*\d{3}\*\*")


def strip_review_linenos(md: str) -> str:
    # ACL/ARR review copies number every line; a number mid-word marks a dropped hyphenation break
    md = re.sub(r"(?<=[A-Za-z0-9])\*\*\d{3}\*\* (?=[a-z])", "", md)
    return re.sub(r"\*\*\d{3}\*\* ?", "", md)


def convert(pdf: pathlib.Path, out: pathlib.Path) -> None:
    doc = pymupdf.open(pdf)
    md = pymupdf4llm.to_markdown(doc, show_progress=False)
    if len(LINENO.findall(md)) > 50:
        md = strip_review_linenos(md)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = f"<!-- source: {pdf.as_posix()} | pages: {doc.page_count} -->\n\n"
    out.write_text(header + md, encoding="utf-8")
    print(f"{pdf} -> {out} ({doc.page_count} pages)")


def slug(stem: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")


def main(argv: list[str]) -> None:
    if not argv:
        sys.exit(__doc__)
    src = pathlib.Path(argv[0])
    if src.is_dir():
        for pdf in sorted(src.glob("*.pdf")):
            convert(pdf, DEFAULT_OUT / f"{slug(pdf.stem)}.md")
    else:
        out = pathlib.Path(argv[1]) if len(argv) > 1 else DEFAULT_OUT / f"{slug(src.stem)}.md"
        convert(src, out)


if __name__ == "__main__":
    main(sys.argv[1:])
