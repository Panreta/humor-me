# Status (read this first in a new session)

Last updated: 2026-10-21. Update this file whenever the state below changes.

## Where we are

| Area | State |
|---|---|
| Milestone | M1 = Dec 15, 2026 (RL course final project, school demo). Plan: [roadmap.md](roadmap.md) |
| Branch | `m1/scaffold` (not pushed) |
| Pipeline S0–S3 | Built, tested, eyeball-checked on 20 sets. `lines.jsonl` is the laugh-predictor input |
| Pipeline S4+S5 (`pipeline.bits`) | Built (Gemini, one call/set). Prompt revised after first eval; **new prompt not yet run** (quota) |
| Pipeline S6 (`pipeline.validate`) | Built. Scores bits + threads + callbacks vs hand labels and an even-spacing baseline |
| Tests | `python -m pytest tests -q`, 24 passing, synthetic data only |
| Backend / frontend | Scaffold only (FastAPI echo websocket, default Next.js page) |
| RL simulator | Not started (roadmap week 2) |

## Models and providers

| Use | Provider / model | Notes |
|---|---|---|
| Live show loop | Gemini `gemini-flash-lite-latest` (`LLM_MODEL`) | TTFT ~0.6 s measured |
| Offline pipeline (S4/S5) | Gemini `gemini-flash-latest` (`PIPELINE_MODEL`) | Thinking model; 20–530 s per set; free tier |
| Config | `backend/.env` (gitignored), template `backend/.env.example` | Base URL is Gemini's OpenAI-compatible endpoint |
| Retired name | `gemini-2.5-flash` returns 404 | Use `-latest` aliases; check `GET {base}/models` |
| Considered | Claude API (no training on API data; Haiku 5.5 for live, Opus/Sonnet 5.5 for pipeline), local 8B via Ollama on the RTX 4060 | See decisions.md 2026-10-21 |

**Gemini free-tier quota exhausted on 2026-10-21** (HTTP 429 "exceeded your current quota").
An hourly in-session check reruns Bargatze when it resets; it doesn't survive a session restart.
To check by hand: probe one request, then run the Bargatze commands below.

## Evaluation protocol (don't break this)

- **Tuning set:** Nate Bargatze, *Hello World* (hand-labeled: 52 bits, 37 threads, 3 callbacks).
- **Held-out test:** Mike Birbiglia, *The Old Man and the Pool*. The user is labeling it now.
  **Do not run `pipeline.bits` on Birbiglia until the labels are imported**, and score it once
  with the final prompt.
- Prompt changes must be general (stand-up rules), never fitted to Bargatze.

First S4 result on Bargatze (old prompt): bits F1 0.75 (P 0.89, R 0.65, Pk 0.172), threads F1
0.61 (Pk 0.277), callbacks 0/3; baseline F1 ~0.21. Details: [dev-log.md](dev-log.md).

## Next steps

1. When quota is back: `python -u -m pipeline.bits nate-bargatze_hello-world_2023` then
   `python -m pipeline.validate`; compare with the numbers above.
2. User finishes Birbiglia labels → `python -m pipeline.labels import mike-birbiglia_the-old-man-and-the-pool_2023`
   → run bits on Birbiglia → validate once (final test).
3. Run `pipeline.bits --all` for the remaining sets (needs quota; ~20 calls).
4. Roadmap week 2 work is behind: simulated audience env + bandit/MDP baselines (the RL grade).

## Open questions

- Paid Gemini tier or Claude API, to get past free-tier limits?
- Local 8B (Ollama) as an offline fallback for the live demo?
