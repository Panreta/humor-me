# PDF extraction (S0)

Code: [`pipeline/ingest.py`](../ingest.py) · Config: [`data/body_overrides.yaml`](../../data/body_overrides.yaml)

## The problem

The transcripts are web pages saved with the browser's "Print → Save as PDF". A PDF has no
paragraphs or reading structure, only positioned text, and each printed page adds its own
header (timestamp, page title) and footer (URL, "2/44"). The page also carries the site's
navigation, a synopsis, a "MORE … Transcripts" link and a comment form. We need the
performance text and nothing else.

## How we do it

1. **Text blocks with coordinates.** `pymupdf` returns text blocks with bounding boxes
   (`page.get_text("blocks", sort=True)`). In these PDFs each visual line is its own block.
2. **Drop print chrome by position.** Header and footer sit in the outer ~3.5% of every page
   (`PRINT_MARGIN`). Filtering by position is more robust than regexes, because the URL wraps
   over two lines and the title varies per set.
3. **Rebuild paragraphs from vertical gaps.** If the gap between two lines is more than half
   a line height, a new paragraph starts. Inside a paragraph, lines are joined with a space.
4. **Cut the post region.** Everything after the "Stand-Up Transcripts" breadcrumb and before
   "Share this article" / "Leave a Comment".
5. **Trim per set.** `skip_leading` / `skip_trailing` from `body_overrides.yaml` remove the
   synopsis at the top and the "MORE" link at the bottom.

## Why explicit trims instead of pure heuristics

The site's layout changed over the years. Newer posts have a "Transcript" heading and a
study disclaimer; older ones use a `* * *` divider or nothing at all. Any heuristic is wrong
for some set, and a silent change would shift every downstream offset. So heuristics only
**propose** counts (`--suggest`), and the reviewed counts are the source of truth.

`--suggest` tries, in order:

| Rule | Fires when | Example sets |
|---|---|---|
| heading | a "Transcript" paragraph in the first 40 | ISMO, Arcuri, Hamilton, Morril |
| separator | a `* * *` in the first 15 **with no reaction markers above it** | Gulman, Morgan, Bargatze, Notaro |
| heuristic | leading paragraphs mention the title, comedian or release terms (max 4) | the rest |

The reaction test matters. A `* * *` that has `[audience laughs]` above it is a section
break *inside* the performance (Birbiglia), not the end of the synopsis. The test looks only
for laugh, applause and cheer words, because the post title's "(2023)" also has parentheses.

## Checking the result

- `meta.json` records `skip_leading`, `skip_trailing`, paragraph and word counts.
- Open `data/interim/<set>.body.txt`: the first line should be the performance (announcer or
  first joke), the last line the closer.
- After any code change, compare `md5sum data/interim/*.body.txt` against the previous run.

## Gotchas

- Paragraphs are layout-based, not semantic. Bits are found later (S4), not here.
- A page break always ends a paragraph, so some sentences are split across two paragraphs.
  Later stages work on sentences, so this doesn't matter.
