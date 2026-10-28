# scripts/

Standalone tools that aren't part of the pipeline or the server.

| Script | Env | What it does |
|---|---|---|
| `pdf_to_md.py` | root `.venv` | Converts reference paper PDFs to markdown in `references/md/` so agents can read them |
| `latency_probe.py` | `backend/.venv` | Measures LLM time-to-first-token and total time against `backend/.env` |
| `inspect_events.py` | root `.venv` | Eyeball checks on pipeline output: text before each laugh, act-out markers, set list. Prints transcript text locally; don't commit or share its output. |

## pdf_to_md.py

```sh
.venv/Scripts/python scripts/pdf_to_md.py references/                       # every PDF
.venv/Scripts/python scripts/pdf_to_md.py references/paper.pdf references/md/name.md
```

Strips ACL review line numbers when it detects them. Pinned to `pymupdf4llm` 0.x because 2.x
ships an empty wheel on Python 3.14.

## latency_probe.py

```sh
backend/.venv/Scripts/python scripts/latency_probe.py --runs 5
```

Prints per-run TTFT and the median **warm** TTFT (the first run includes connection setup or
model load). Compare providers by changing `backend/.env`. Target: well under 1 s, leaving
room for TTS in the ~1 s budget.

## `sync_wiki.py`: publish docs/wiki to the GitHub wiki

```sh
git clone https://github.com/supriya-kotturu/humor-me.wiki.git ../humor-me.wiki   # once
python scripts/sync_wiki.py ../humor-me.wiki
cd ../humor-me.wiki && git add -A && git commit -m "Sync from docs/wiki @ <sha>" && git push
```

Run after any merged change to `docs/wiki/`. README.md becomes Home, pages are named from
their H1, wiki-to-wiki links become wiki links, links to other repo files become github.com
URLs, and a sidebar lists pages in the index's reading order. Pages that exist only on the
GitHub wiki are left alone and listed.
