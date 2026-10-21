import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
LEXICON = ROOT / "data" / "markers.yaml"

# Filenames come from the browser's "save as PDF": "Comedian_ Title (Year) – Transcript _ Scraps from the loft.pdf"
FILENAME = re.compile(r"^(?P<comedian>[^_]+)_\s*(?P<title>.+?)\s*\((?P<year>\d{4})\)")


def _slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def parse_filename(name: str) -> dict:
    m = FILENAME.match(name)
    if not m:
        raise ValueError(f"unrecognized transcript filename: {name}")
    comedian, title, year = m["comedian"].strip(), m["title"].strip(), int(m["year"])
    return {
        "set_id": f"{_slugify(comedian)}_{_slugify(title)}_{year}",
        "comedian": comedian,
        "title": title,
        "year": year,
    }
