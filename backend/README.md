# backend/

FastAPI server for the live show loop: director, bank generator, safety filter, Kokoro voice on the CPU and a
reward source (operator keys or the simulated room), as described in [docs/design.md](../docs/design.md#architecture).

## Setup

```sh
cd backend
uv python install 3.12  # or any Python 3.12
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
uv pip install --python .venv/Scripts/python.exe https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
cp sample.env .env
```

The venv must be Python 3.12: Kokoro 0.9.4 needs Python below 3.13, and on 3.14 the install fails (spaCy's build dependencies; see `requirements.txt` line 1 and the dev-log, 2026-11-01) ("no matching distribution", or `ModuleNotFoundError: numpy` in an old venv). A `uv` venv has no pip, so Kokoro cannot download spaCy's English model by itself and the first render stops with `E050 can't find model 'en_core_web_sm'`: the last install line above is that model. (To use pip instead, make the venv with `uv venv --seed --python 3.12 .venv`.)

`.env` points at any OpenAI-compatible chat endpoint (see `sample.env`):

| Variable | Example |
|---|---|
| `GEMINI_BASE_URL` | `https://api.groq.com/openai/v1` (for Ollama use `JOKE_WRITER_PROVIDER=ollama` instead: only that route warms the model and checks the GPU) |
| `GEMINI_API_KEY` | provider key (empty for Ollama) |
| `JOKE_WRITER_MODEL` | e.g. `llama-3.1-8b-instant` |
| `JOKE_WRITER_PROVIDER` | `ollama` (the shipped writer, `bonsai-2:4b` on this laptop, no key, `OLLAMA_URL` to move it; D-107, D-109), `openai`/`gemini` (the three above) or `anthropic` (official SDK, key in `ANTHROPIC_API_KEY`, `JOKE_WRITER_MODEL=claude-haiku-5-5`) |

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
| binary frame | 16-bit mono 16 kHz PCM, under 64 KiB | Room audio, only after `record`. Refused with an `error` when no recording was ever agreed; dropped silently once it ended |
| `audio_start` | `wall_ms` (epoch ms) | Sent before the first audio frame: when the first sample was captured. Sets `start_at`; the arrival-time estimate is the fallback |
| `audio_health` | `produced_s`, `elapsed_s`, `gap_ms`, `wall_s`, `rate` (numbers) | Every ~5 s of recording: audio the microphone delivered against capture time, silence filled for gaps, wall time, device rate. Warns below 99% of either; diagnostics only, not in the session log |
| `audio_failed` | `reason` (optional text) | The browser could not open the microphone; the show runs unrecorded and the reason comes back in `record_stop` |
| `played` | none | The browser finished playing the last audio |
| `pause` / `resume` | none | The operator paused or resumed (the browser has already suspended or resumed playback). Written as a `pause` / `resume` row on the show clock. Refused with an `error`, the show going on, when no show is running (also during the microphone prompt), on `pause` while paused, and on `resume` while not paused. While paused the show holds: no line, bridge or cached line starts (the server waits for `played` with no timeout, and awaits the resume before the next one), and the reaction window and the hold stop their timers and resume with the time left; keys pressed meanwhile still count. A pause left open when the show ends runs to the end |
| `reaction` | `bit`, `joke`, `key` (`W`, `K`, `Space`, `D`) | A key press. Counts only while that joke's window is open |
| `rating` | `value` (integer 1 to 5) | After `show_end`, finished or aborted, on the same connection: appends one `rating` row to that show's log. One per show; refused (an `error`, the show's log untouched) when there is no finished show, the show is still running, or it was already rated |

| Server sends | Fields | Meaning |
|---|---|---|
| `line` | `text` | The next line; the WAV follows as a binary frame. Send `played` when it ends |
| `reaction_window` | `state` (`open`\|`closed`), `bit`, `joke` | Opens when audio ends, closes at the end of the hold (or on timeout) |
| `sim_reaction` | `bit`, `joke`, `key` | `sim` room only: the key the simulated room "pressed" |
| `record` | none | Sent after a `start` with `consent`: begin sending audio. The show waits for the first frame (15 s) |
| `record_stop` | `reason` (text or null) | Stop the microphone, sent once when the recording ends for any reason: the show ended, no audio arrived in time, the browser failed, or the disk refused a write (the show goes on) |
| `show_end` | `status` (`finished`\|`aborted`), `ratable` (bool) | The show is over. `ratable` is false when it died before writing its log (nothing to rate; the `error` before it says why). The connection stays open for a `rating` |
| `rating_saved` | `value` | The rating row was written |
| `pause_state` | `paused` (bool) | The pause or resume row was written; the browser keeps (or corrects itself to) this state. A refused one comes back as an `error` instead |
| `error` | `message`, optional `about` | A bad client message or a failed start or show. `about: "pause"` marks the refusal of a `pause` or `resume`. A bad message never ends the show |

**What a pause does to the data (D-92):** every pause and resume is a log row on the show clock, and the recording keeps running, so `paused_at - show_start.audio.start_at` is the offset in the WAV. Only a joke whose playback was interrupted loses its label: it keeps its `reaction` row but counts as unlabeled in the bit's reward, in the live show and in replay (the rest of the bit still counts if 3 of 5 labels remain), and its audio should not be used to train the laugh detector (`app.pauses.paused_jokes`). A pause in the reaction window or between lines or bits costs nothing, apart from a pause pressed within milliseconds of a line ending, which can drop that line. Replay assumes the server clock does not step backwards. A show that was paused or cut short keeps its finished bits for the director, but its rating is left out of the reported rating (`python -m sim.acceptance` lists it as not counted). Pause only when you must: each pause costs the label of the joke it lands on.

The mode the director picked is never sent. Presses outside their window are not logged as reactions (replay counts every
row), only noted by the Python logger. Within a hold the last press wins and each change is a `reaction` row.
By default the show uses the bank and a bank-only filter (`BankOnlySafety`), which needs no key. `JOKE_SOURCE=llm` switches to the
real stack (`app/generate.py`, gen-v4, D-110): each bit is 3 model calls at once on the same prompt, one thread of 5 jokes each (so up to 3 candidates per joke; a call that fails drops only its thread; a badly shaped reply is asked for once more first (D-111), and on Ollama the reply schema counts the 5 jokes, and the 3 share one 12 s budget), the safety filter's rules plus one LLM verdict call
per bit (Claude Haiku 5.5 with thinking off by default, D-100; needs `ANTHROPIC_API_KEY`; `SAFETY_CHECK_PROVIDER=gemini` goes back to Flash-Lite), the next line rendered while the current one plays, and filler while a bit loads: cached lines from the best bits of earlier live shows (`app/cache.py`, rules re-checked for this room's age, a teen room only from teen shows, never rated, logged in a `cache` row) at the show open and between bits, then in-character bridge lines (the 10 in `app/generate.py`, `BRIDGES`). By default a cached line is skipped when any show logged on this laptop already played it (D-112), so with one listener only the in-character lines play; `BRIDGE_SKIP_HEARD=0` in `.env` brings the cached lines back for a new audience. The filler lines play in a fresh order each show (a seeded shuffle; still no repeat while an unused one remains). While a bit's last jokes play the show guesses the next bit's mode and writes it ahead (D-115, `PREDICT_NEXT_BIT`, on by default, `0` turns it off): a right guess has the next bit ready at the boundary, a wrong one is discarded; the director's choice and the rewards are the same either way. The log gets a `prediction` row per guessed bit and `speculative` on a kept bit's `prep` row. The `cache` row logs `skip_heard` (the setting), `heard_dropped` (pool lines the rule removed: the pool without it minus the pool with it) and `bridge_seed` (the shuffle seed, so the show replays). A failed call is retried
twice inside a deadline; a failed generation or a bit whose candidates are all rejected falls back to the hand-checked bank (rules only, no API call), logged in
`candidates`. A `live` room refuses to start without `app/blocklist.local.txt`.

**The writer is `bonsai-2:4b` on Ollama (D-107).** Start Ollama before the backend: quit the tray app, then in PowerShell
`$env:OLLAMA_CONTEXT_LENGTH=8192; $env:OLLAMA_KEEP_ALIVE="-1"; $env:OLLAMA_NUM_PARALLEL=3; ollama serve` (once: `ollama pull bonsai-2:4b`). `NUM_PARALLEL` must be at least 3: the generator sends 3 calls at once, and with fewer slots Ollama queues them and the bit takes about the sum of the calls, not the longest (each slot keeps its own KV cache, so VRAM use rises; the GPU check below still has to pass). Each show
loads the writer before it starts and checks that Ollama put it all on the GPU (D-109). If it can't load, or another model
holds the GPU, or it runs at a context other than 8192, the show stops before it begins and says why (e.g. "also loaded:
gemma4:12b (`ollama stop <name>`)"). `KEEP_ALIVE=-1` keeps every model loaded, so after Exp 24 or judge runs `ollama stop`
the others. `OLLAMA_URL` must end in `/v1`.

```sh
.venv/Scripts/python -m uvicorn app.main:app --port 8000 --env-file .env   # with JOKE_SOURCE=llm in .env
```

## Acceptance run (T11, the Stage 0 exit)

The user runs the shows; `sim/acceptance.py` scores them. Needed: 3 simulated shows in a row, then 1 live show.

**Before the live show (setup rules; they cost a run in Exp 21 when skipped):**

- No Bluetooth headset: use the laptop microphone and speakers (a Bluetooth headset drops the mic to a phone-call codec and shifts the audio clock).
- Windows Sound settings, Communications tab: **Do nothing** (otherwise Windows ducks the robot's voice when it hears a "call").
- Microphone properties, Advanced: **Audio enhancements off** (with them on, echo cancellation removed the laptop's own output from the recording).
- The room is told the show is recorded and agrees, then tick the consent box. Without it the show runs and nothing is recorded.
- `backend/.env` has `ANTHROPIC_API_KEY` (the safety check) and `backend/app/blocklist.local.txt` exists (a live room refuses to start without it).
- Ollama is serving bonsai and nothing else is loaded (`ollama ps`); see the writer note above.

**Start the backend from the main checkout** (`D:/vs-code-workspace/humor-me`, not a worktree: the key and the blocklist are there), in a Python 3.12 venv as in Setup (on the operator laptop the working one is `D:/vs-code-workspace/humor-me-demo/backend/.venv`; the main checkout's `backend/.venv` is now a 3.12 venv too, rebuilt 2026-11-25; an older one was 3.14 and had no numpy or kokoro):

```sh
cd backend
.venv/Scripts/python -m uvicorn app.main:app --port 8000 --env-file .env   # with JOKE_SOURCE=llm in .env
cd ../frontend && npm run dev   # then open http://localhost:3000
```

Set `JOKE_SOURCE=llm` in `backend/.env` (a one-command prefix differs between Bash and PowerShell, and `--env-file` reads it either way). Simulated shows can run on `JOKE_SOURCE=bank` (no quota) or `llm`; the live show uses `llm` (decisions-detail.md 2026-11-15).

**Run:** (1) three Simulated shows back to back, untouched; (2) one Live show, you press W / K / Space / D for every joke, consent ticked; (3) after each show answer the rating prompt (keys 1 to 5, or Skip; a show that failed before it began shows its error and no prompt). Then score them. The session ids are the file names in `data/sessions/` without `.jsonl`:

```sh
# from the repo root, root venv. Plain: prints the report and saves nothing, so try ids freely.
python -m sim.acceptance <sim id 1> <sim id 2> <sim id 3> <live id>
# When the right shows are in the list, keep it as the Exp 22 evidence (replaces the last record):
python -m sim.acceptance <sim id 1> <sim id 2> <sim id 3> <live id> --record
```

Pass only the shows meant to count: an unreadable log among the ids fails check (a).

It prints a table per show (valid rows, status, bank share, bridges, label counts, pauses with paused seconds and jokes dropped, rating, audio coverage), a reported rating that leaves out paused and aborted shows (they stay listed as not counted), and the four verdicts: (a) 3 simulated in a row, (b) a valid finished live show, (c) label spread (one key over 60% of presses is flagged, reported only), (d) per-bit reward noise against the sim's 0.15, with "RERUN Exp 18" when it is more than half away. With one live show the noise rests on 8 bits and the interval is wide; the output says when the call is a guess. `--record` saves numbers and the session ids only, in `reports/metrics/acceptance.json` (no text, no audio, no paths); then run `python scripts/run_notebook.py` to refresh the Exp 22 section, which lists the recorded ids. Logs and recordings in `data/sessions/` are gitignored: never commit them.

## Comparing generator models (Exp 24)

Compares Claude Haiku 5.5 and two local Ollama models (`bonsai-2:4b`, `qwen3:4b-instruct`; D-96 explains why the Gemini writers were dropped, D-101 why qwen replaced mimo), plus `gemma4:12b` as a reference, on 60 bits (30 are voted on; all 60 count toward failures and speed). The bar is fixed in [the plan](../docs/plans/live-loop.md) (T13). Run from the repo root, in the root venv, in the main checkout (the keys are there).

0. **Local models (once):** install Ollama for Windows from ollama.com, then `ollama pull bonsai-2:4b` and `ollama pull qwen3:4b-instruct`, and check the tags with `ollama list` (if one differs, set `QWEN_OLLAMA_TAG` or `BONSAI_OLLAMA_TAG` in `backend/.env`; `OLLAMA_URL` changes the server, default `http://127.0.0.1:11434/v1`). Start the server with a context that fits the prompt (about 2k tokens): in PowerShell `$env:OLLAMA_CONTEXT_LENGTH=8192; ollama serve`. (Exp 24 uses the frozen one-call generator, so it needs no parallel slots; Exp 26 and shows do: add `$env:OLLAMA_NUM_PARALLEL=3`.) Quit the Ollama tray app first if it is already serving, or it keeps its own context. If Ollama is not running, the run stops on the first local call, says so and caches nothing.
1. **Keys in `backend/.env`** (names as in `backend/sample.env`): `ANTHROPIC_API_KEY` (a paid key; Claude is not free): it pays for the Haiku candidate and the safety filter, which is Claude Haiku 5.5 with thinking off for every candidate, as in a show (D-100; about $0.02 for the run's 240 checks). No Gemini key is needed. The models and the filter are named by the harness, not by `JOKE_WRITER_MODEL` or `SAFETY_CHECK_*`. If `app/blocklist.local.txt` exists the filter's rules use it, as in a live show.
2. **Generate (the filter is paced for the free tier and a local bit may take seconds to tens of seconds, so allow an hour or more; resumable):** `.venv/Scripts/python -m sim.compare_generators run --env-file backend/.env`. Each local model runs all its 60 prompts in one block after one untimed warm-up (Ollama keeps one local model loaded at a time; a cold load took 35 to 126 s), so the first call of each local model is slow and does not count; its time is saved as `warmup_s`. If a warm-up times out it is tried once more, and if that fails too the run stops with a clear error and saves nothing for that model. Every time you rerun `run`, a local model whose GPU reading was unknown is checked again first, even when all its prompts are done, and one line says what was found (for example a wrong context: restart Ollama with `OLLAMA_CONTEXT_LENGTH=8192` and rerun). The hosted model runs prompt by prompt. Stop and rerun any time; finished calls are not repeated. **Idea for a live show, not built:** warm the writer when the server starts, so the first bit does not pay the cold load. A bad key, a wrong model name or any unexpected error on a model's first call stops the run and saves nothing. `--redo-failed` adds a new attempt where the last one failed for a key, setting or provider-availability reason (every attempt is kept and counts toward the failure rate); `--models mimo-v2.6-9b,claude-haiku-5-5` picks models, `--n 72` adds prompts, `--ballot` sets how many of the first prompts are voted on (30).
3. **Ballot:** `python -m sim.compare_generators export` (the first 30 prompts) (after `run` has finished) writes `ballot.md` and `votes.csv` under `data/model_comparison/gen-<digest>/` and the key apart in `data/model_comparison/key/`. A model with no safe thread for a prompt is not shown and is ranked last for it.
4. **Vote (about 50 minutes: four options per prompt), then judge:** `python -m sim.compare_generators vote` shows each prompt and asks for the order, best first (`BDAC`). It saves after every answer, `s` skips, `q` quits; run it again to continue. Don't open the `key` folder. Or fill `votes.csv` by hand from `ballot.md`.
5. **Judge (runs on Explorer, Qwen2.5-7B-Instruct, a few hours at most, resumable; the default judge, no API key):**
   - `python -m sim.compare_generators judge --export-bundle data/model_comparison/judge` after you have voted on every ballot prompt and `run` has finished. It writes `calls.jsonl` (our own jokes, your rankings, no model names). **Smoke test first:** add `--smoke 3` (9 calls), run it on the cluster with `LIMIT=3 sbatch ...` or the full smoke job, and check `replies.jsonl` (see `explorer/README.md`, "Exp 24 judge").
   - Upload `calls.jsonl` to the cluster (OOD Files app) and run `sbatch explorer/judge_generators.sbatch`; resubmit the same command if it stops, and again if it exits with "calls were not rankings".
   - Download `replies.jsonl` and run `python -m sim.compare_generators judge --import-replies <file>`. It scores the judge leave-one-out and prints the verdict; only if it passes top pick, pairwise order and per-model bias do its rankings of prompts 31 to 60 join yours. A run with unanswered calls is marked incomplete and joins nothing.
   - **First model to try: Ollama's `qwen3.8:27b`.** Same bundle, but export with `--judge-provider ollama --export-bundle data/model_comparison/judge-ollama` (smoke: add `--smoke 3`), upload `calls.jsonl` to `data/model_comparison/judge-ollama/` on the cluster, run `sbatch explorer/judge_ollama.sbatch` (smoke: `LIMIT=3 sbatch ...`), download `replies.jsonl` and run `python -m sim.compare_generators judge --judge-provider ollama --import-replies <file>`. If you can reach an Ollama server directly (started with `OLLAMA_CONTEXT_LENGTH=32768`), `judge --judge-provider ollama` with no bundle talks to it at `OLLAMA_URL` (default `http://127.0.0.1:11434/v1`). Either way the judge refuses prompts that do not fit the context, and a reply that shows Ollama read far fewer tokens than sent is treated as truncation, never as an answer. Qwen2.5-7B (`judge_generators.sbatch`) is the fallback.
   - `--judge-model claude-haiku-5-5` (needs `ANTHROPIC_API_KEY`, about $0.30) or `gemini-flash-lite` runs a judge through an API instead, with no bundle.
6. **Score:** `python -m sim.compare_generators score` prints the table and the pick, and saves numbers only to `reports/metrics/model_comparison.json`. Then `python scripts/run_notebook.py` to refresh the Exp 24 section and commit the metrics file.

**Which local models can win (D-97).** Kokoro speaks on the CPU, so a local model counts only if Ollama puts it 100% on the GPU at context 8192, with its own defaults. `run` reads the placement from Ollama (`/api/ps`) after each local model's first good call and saves it in the run folder; `score` shows it in a GPU column. If Ollama's report is missing or looks wrong the model counts as unknown and does not pass, so rerun `run` (it measures again; stop other models first with `ollama stop <model>`). Measured on the RTX 4060 laptop: `bonsai-2:4b` 100% and `qwen3:4b-instruct` 100% (`mimo-v2.6:9b` read 87% and was dropped, D-101). `gemma4:12b` was a reference until D-104 dropped it (too slow to finish a bit in 12 s).

**The first Exp 24 run was aborted by a Google outage and archived, not scored (D-96).** The new candidates start a fresh `data/model_comparison/gen-<digest>/` folder on their own; rename the old one (`gen-f036d862`) to `gen-f036d862.aborted-gemini-outage-2026-11-25` so it is not mistaken for data. **Caveat:** the comparison runs without the Kokoro voice. Kokoro speaks on the CPU, so it does not take GPU memory; a model fully on the GPU (the D-97 gate) should be close to its measured speed in a show, but Kokoro's CPU use alongside it is not measured.

Everything under `data/model_comparison/` (joke text, ballot, key, votes) is gitignored: never commit it. Shows write with bonsai on Ollama (D-107, D-109). To run one on Claude instead, set `JOKE_WRITER_PROVIDER=anthropic` (the model defaults to `claude-haiku-5-5`).

## Per-thread timing and connected threads (Exp 26, Exp 27)

Both took their run on 2026-11-25. Rerun after a merge that touches the generator, from the repo root with the root venv. Ollama up with `OLLAMA_NUM_PARALLEL=3` and `OLLAMA_CONTEXT_LENGTH=8192`, nothing else loaded, and `JOKE_WRITER_PROVIDER=ollama` (the default for these two). About 5 minutes together.

```bash
python -m sim.thread_timing                # Exp 26 (done, passed): does a bit fit 12 s as 3 parallel calls? -> reports/metrics/thread_timing.json
python -m sim.thread_coherence run --env-file backend/.env   # Exp 27 (done, inconclusive_judge, D-111): both prompts write 20 prompts, Haiku judges each thread (needs ANTHROPIC_API_KEY)
python -m sim.thread_coherence review      # you answer y/n "is this one bit?" for 10 threads, 5 the judge approved and 5 it rejected, arm and verdict hidden; q saves and quits. Required, whatever the bar says
python -m sim.thread_coherence score       # the numbers again with your answers in, no model call -> reports/metrics/thread_coherence.json
```

Then `python scripts/run_notebook.py` and commit the two metrics files. `run` can be stopped and restarted: finished bits and judge verdicts are kept in `data/thread_coherence/run-<digest>/` (gitignored: thread text lives only there, never commit it). The judge costs about 120 Haiku calls, under a cent. Both commands take `--env-file backend/.env` to load the keys. Exp 27's bar and rule are in `sim/thread_coherence.py` and D-111.

## Judge v2 on a fresh sample (Exp 30)

Exp 27's judge agreed with you on 4 of 10 threads (too low to trust), so a second judge (v2) was written from those 10 answers. This checks both on threads neither was written from (D-114). From the repo root with the root venv, Ollama up with `OLLAMA_NUM_PARALLEL=3` and `OLLAMA_CONTEXT_LENGTH=8192`, nothing else loaded, `ANTHROPIC_API_KEY` for the judges (`--env-file` loads it):

```bash
python -m sim.judge_calibration run --env-file backend/.env   # writes 10 fresh prompts x 2 arms (about 3 min), both judges judge the 20 drawn threads
python -m sim.judge_calibration review                        # you label 20 threads: y = one bit, n = separate jokes, blind (no verdict, no arm); q saves and quits, rerun resumes
python -m sim.judge_calibration score                         # agreement of v1 and v2 with you, paired CI, verdict -> reports/metrics/judge_calibration.json
python -m sim.thread_coherence rescore --judge v2             # only if `score` says rescore_allowed: re-judge Exp 27's stored threads with v2 (about 102 Haiku calls)
```

Rule, fixed before the run: keep v2 if it agrees with you on more of the 20 than v1 (a tie keeps v1); the kept judge may re-score Exp 27 only if it disagrees with you on under 40% (`rescore` refuses otherwise). Then `python scripts/run_notebook.py` and commit `judge_calibration.json` (and `thread_coherence_rescore.json` if it ran). Threads and your labels live under `data/thread_coherence/calib-<digest>/` (gitignored: never commit).

## Exp 27 extension to 100 prompts (D-111 addendum): done 2026-11-26, passed, gen-v5 live

Exp 30 kept judge v2, and v2 over Exp 27's 20 prompts gave +0.133 with a CI of 0.0 to +0.25 (failed at exactly 0). So the user extends Exp 27 to 100 prompts, with no new labels. Run from the checkout that holds `data/thread_coherence/` (the 20 prompts' bits must be in it; the command stops if not), Ollama up with `OLLAMA_NUM_PARALLEL=3` and `OLLAMA_CONTEXT_LENGTH=8192`, nothing else loaded, `ANTHROPIC_API_KEY` set for the second command.

```bash
python -m sim.thread_coherence run --n 100 --env-file backend/.env                 # writes only the missing bits (c20 to c99, 80 x 2 arms, about 15 to 25 min), no judge call -> reports/metrics/thread_coherence_n100.json
python -m sim.thread_coherence rescore --judge v2 --n 100 --env-file backend/.env   # judges all 100 prompts with v2 (c00 to c19 cached, about 470 new Haiku calls, a few cents) -> reports/metrics/thread_coherence_rescore_n100.json
```

`run` can be stopped and restarted, and rewrites a bit that lost every thread to a timeout by itself, once (no manual step; it lists them as `bits_rewritten`). Bar: v2's paired B - A over the **80 fresh prompts** (c20 to c99), CI lower bound above 0; the pooled 100 is shown beside it and does not decide. There is no review step: v1's spot-check is superseded by v2. Then `python scripts/run_notebook.py` and commit the two metrics files.

## Cached bits run (Exp 23)

After a live show, run three simulated shows on `GENERATOR=llm`, then score them from the repo root. Plain, it prints and saves nothing; `--record` keeps numbers and ids only (`reports/metrics/cached_bits.json`):

```sh
python -m sim.evaluate_cached <sim id 1> <sim id 2> <sim id 3> [--record]
```

Bar: show-open silence at most 2 s on each of the 3. A show whose bit 0 had no cached bit does not count. Between-bit silence is printed beside it, with no bar.

## Heard bridges (Exp 28)

Counts, from the local logs and with no model call, how many of each show's bridge lines the listener had heard in an earlier show, and how many of the 10 filler lines. Plain, it prints and saves nothing; `--record` keeps counts and session ids only (`reports/metrics/heard_bridges.json`), then run `python scripts/run_notebook.py`:

```sh
python -m sim.evaluate_heard [--dir data/sessions] [--record]
```

## Where a bit's wait goes (Exp 29)

After the live shows, one command prints where each bit's wait went, per show and pooled (seconds only, never a line of text): writing, safety check, voice queue, voice synthesis, other, and the bridge overhang. Name the session ids of the shows to count (file names in `data/sessions/` without `.jsonl`). With none it reads every log and skips shows written by the hand-written bank. Plain, it prints and saves nothing; `--record` keeps seconds and session ids (`reports/metrics/stage_timing.json`), then run `python scripts/run_notebook.py`:

```sh
python -m sim.stage_timing <session id> [<session id> ...] [--dir data/sessions] [--record]
```

Bit 0 is shown apart; bits with a pause in their wait or a failed first render are dropped. The last line names the biggest stage and the fix it points to, or says "inconclusive" (the rule needs 2 shows and 16 bits and an interval above 0; see `sim/stage_timing.py`). Shows from before D-113 have no `prep` rows and are listed without stages.

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
week 5, recreate this venv with 3.12 (the backend venv is already 3.12).

`app/bank/` holds `bank-v1.yaml`, 120 hand-written lines (6 modes x 4 bits x 5 jokes), and `BankGenerator`: model-free material for tests, calibration and outages.
