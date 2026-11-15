# backend/

FastAPI server for the live show loop: director, bank generator, safety filter, Kokoro voice on the CPU and a
reward source (operator keys or the simulated room), as described in [docs/design.md](../docs/design.md#architecture).

## Setup

```sh
cd backend
uv python install 3.12  # or any Python 3.12
uv venv --seed --python 3.12 .venv
.venv/Scripts/python -m pip install -r requirements.txt
cp .env.example .env
```

`.env` points at any OpenAI-compatible chat endpoint (see `.env.example`):

| Variable | Example |
|---|---|
| `LLM_BASE_URL` | `https://api.groq.com/openai/v1`, `http://localhost:11434/v1` (Ollama) |
| `LLM_API_KEY` | provider key (empty for Ollama) |
| `LLM_MODEL` | e.g. `llama-3.1-8b-instant` |

## Run

```sh
cd backend
.venv/Scripts/python -m uvicorn app.main:app --reload --port 8000
```

- `GET /health` → `{"status": "ok"}`
- `WS /ws`, JSON text frames plus binary WAV frames from the server. A show runs beside the receive loop.

| Client sends | Fields | Meaning |
|---|---|---|
| `start` | `context` (`venue`, `energy`, `age`), `room` (`sim`\|`live`), `operator`, `consent` (bool, default false) | Start a show. One at a time. `consent: true` means the operator confirmed the room agreed to be recorded |
| binary frame | 16-bit mono 16 kHz PCM, under 64 KiB | Room audio, only after `record`. Refused with an `error` when no recording was agreed |
| `audio_failed` | none | The browser could not open the microphone; the show runs unrecorded |
| `played` | none | The browser finished playing the last audio |
| `reaction` | `bit`, `joke`, `key` (`W`, `K`, `Space`, `D`) | A key press. Counts only while that joke's window is open |

| Server sends | Fields | Meaning |
|---|---|---|
| `line` | `text` | The next line; the WAV follows as a binary frame. Send `played` when it ends |
| `reaction_window` | `state` (`open`\|`closed`), `bit`, `joke` | Opens when audio ends, closes at the end of the hold (or on timeout) |
| `sim_reaction` | `bit`, `joke`, `key` | `sim` room only: the key the simulated room "pressed" |
| `record` | none | Sent after a `start` with `consent`: begin sending audio. The show waits for the first frame (15 s) |
| `show_end` | `status` (`finished`\|`aborted`) | The show is over |
| `error` | `message` | A bad client message or a failed start or show. A bad message never ends the show |

The mode the director picked is never sent. Presses outside their window are not logged as reactions (replay counts every
row), only noted by the Python logger. Within a hold the last press wins and each change is a `reaction` row.
By default the show uses the bank and a bank-only filter (`BankOnlySafety`), which needs no key. `GENERATOR=llm` switches to the
real stack (`app/generate.py`): one model call per bit for 5 jokes x 3 candidates, the safety filter's rules plus one LLM verdict call
per bit, the next line rendered while the current one plays, and in-character bridge lines while a bit loads. A failed call is retried
twice inside a deadline; a failed generation or a bit whose candidates are all rejected falls back to the hand-checked bank (rules only, no API call), logged in
`candidates`. A `live` room refuses to start without `app/blocklist.local.txt`.

```sh
.venv/Scripts/python -m uvicorn app.main:app --port 8000 --env-file .env   # with GENERATOR=llm in .env
```

## Pick a provider first

Comedy timing needs the next line within ~1 s of a laugh ending, so measure time-to-first-token:

```sh
backend/.venv/Scripts/python scripts/latency_probe.py --runs 5   # from repo root
```

## Planned modules

| Module | Role | Week |
|---|---|---|
| `sim/` | simulated audience env + bandit/MDP policies | 2 |
| `show/` | director (RL planner), generator, safety filter, memorization guard | 3–5 |
| `audio/` | VAD, laughter detection, laugh-aware turn-taking | 5 |
| `tts/` | streaming TTS with timing marks | 3–4 |

Python note: audio/ML wheels (faster-whisper, torch) may lag on 3.14. If installs fail in
week 5, recreate this venv with 3.12 or 3.13.

`app/bank/` holds `bank-v1.yaml`, 120 hand-written lines (6 modes x 4 bits x 5 jokes), and `BankGenerator`: model-free material for tests, calibration and outages.
