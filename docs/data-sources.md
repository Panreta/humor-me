# Data Sources

No public dataset covers (venue, joke, live audience reaction). **Our own logged sessions are
the primary dataset**; everything below is a cold-start aid.

## Finalized

| Source | Role | Stage | Notes |
|---|---|---|---|
| **Own session logs** | Primary: bandit reward, RM training, DPO pairs | 1 → ∞ | Schema defined in Stage 1. The real asset. |
| **rJokes** ([orionw/rJokesData](https://github.com/orionw/rJokesData)) | Cold-start humor prior; sanity-check/calibrate the LLM judge | 3 | ~570k Reddit jokes with upvote-derived scores. One-liners, Reddit taste — not stand-up. Verify license before redistributing derivatives. |
| **HumorGen** (paper, [references/md](../references/md/humorgen-cognitive-synergy-persona-distillation.md)) | Persona mode definitions; candidate-gen + Elo-judge recipe | 3 | Use the *method* now. Their dataset is on an anonymous review repo — wait for camera-ready/license before depending on it. |
| **LoL** (paper, [references/md](../references/md/structured-thought-leaps-humor-iclr2025.md)) | Thought-leap prompting; pairwise judge design | 3 | Method only. |
| **Christiano et al. 2017** (paper) | Preference → reward model recipe for training mode | 5–7 | Method only. |

## Deferred / restricted

| Source | Decision | Why |
|---|---|---|
| scrapsfromtheloft transcripts | **Deferred.** Optional local-only style analysis (set structure, callbacks, tags) | Fan transcripts of copyrighted specials. Never redistribute, never train a released model on them. Check ToS/robots.txt before any scraping. |
| HumorGen dataset | Deferred until public with license | Anonymous repo may disappear. |

If transcripts are used at all, the low-risk use is **aggregate statistics only**, computed
locally and never committed as text: laugh markers per minute, words between laughs, bit
length. These can seed the simulated audience's timing priors. Not needed for M1.

## Tooling (to evaluate, not data)

| Need | Candidates | Stage |
|---|---|---|
| VAD | Silero VAD, WebRTC VAD | 2 |
| Laughter detection | jrgillick/laughter-detection (Gillick et al.), or fine-tuned audio classifier | 2 |
| ASR | faster-whisper / streaming ASR | 3 |
| TTS | Low-latency streaming TTS with SSML-ish pause control | 1 |
| Scene snapshot | Any VLM, async, text-out only | 3 |

## To investigate

- Public stand-up audio datasets with laughter annotations (would let us pretrain/validate
  the laugh detector and study laugh timing). Verify existence and license before use.
