# Introduction to humor-me

humor-me is a voice-only stand-up comedy robot. It tells jokes, reads the room through laughter, and adapts its
material during and across shows. The deadline is a live demo on **Dec 8**.

For more, [README.md](README.md) is the overview and [docs/status.md](docs/status.md) is where things stand now.

## The three parts

| Part | Where | What it is |
|---|---|---|
| Simulation | `sim/` | A fake audience for testing the director quickly, without real people |
| Live show | `backend/` + `frontend/` | The real show loop, described below |
| Model | `judge/`, `explorer/` | Training a better joke writer. The DPO model fits here |

## How one show runs

```
Setup → Director picks a mode → Joke writer → Safety check → Voice → Laugh score → Learning
                 ↑                                                                     │
                 └─────────────────────────────────────────────────────────────────────┘
```

### 1. Setup

The operator enters the room: **venue**, **energy** and **age**, and chooses **Simulated** or **Live**.

- **Simulated:** the fake audience reacts by itself and the reaction buttons light up on their own.
- **Live:** a real audience. The operator presses a key for every joke (see step 6).

### 2. Director

Picks a joke style (**mode**) for the next bit. The six modes are neurotic, cynic, observer, wordsmith, optimist and
absurdist (`explorer/label_modes.py`). It learns which modes work for which rooms with a method called **LinUCB**.

Code: `backend/app/director.py`, which uses pieces from `sim/policies.py`:

- `LinUCB`: the learner itself.
- `EpsilonSoft`: wraps LinUCB and picks a random mode 20% of the time (ε = 0.2), so it keeps exploring.
- `prior_means`: the director's starting guesses. The live show uses `"uniform"`, so no mode is favoured.
- `features`: the room info the director looks at, called inside LinUCB.

`load_director` rebuilds the director at the start of each show by replaying every logged live show.

TODO: read Exp 18, the director experiment: `docs/decisions.md` "2026-11-08 — Exp 18" and
`notebooks/decisions-director.ipynb`.

### 3. Joke writer

Writes jokes in the chosen mode. This is **Gemini** now (`GeminiGenerator`, `backend/app/generate.py:85`). With no
API key, it falls back to the hand-written bank (`BankGenerator`, `backend/app/bank/`).

TODO: how to plug in the DPO model. Any class with a `generate(mode, bit, joke, context) -> Batch` method can be
the joke writer (the `Generator` interface in `backend/app/show.py`). Write one that calls the DPO model, then
choose it in `make_stack` in `backend/app/main.py`, next to `GENERATOR=llm`.

**Bridge lines** (`BRIDGES` in `backend/app/generate.py`): short hand-written filler lines in the robot's voice.
They play only when the next bit isn't ready yet, so the room never sits in silence. Each one is logged as a
`bridge` row. These are not jokes; skip them when building training data.

### 4. Safety check

Blocks jokes about bodies, appearance or protected groups (invariant 1 in `CLAUDE.md`).

- `backend/app/safety.py`: the filter itself. Rules first, then one LLM yes/no call per bit. A line is rejected if
  anything fails, including a timeout. The hard rules are at line 8.
- `backend/app/generate.py:120`: `LiveSafety`, which plugs the filter into the live show. Line 36 also tells the
  joke writer the same rule up front.
- `backend/app/live.py:100`: `BankOnlySafety`, the default when no key is set. It only passes hand-written bank lines.
- `backend/app/blocklist.local.txt`: the local keyword list. Not in git; live shows need it.

### 5. Voice

**Kokoro** reads the joke aloud, and the wave on screen moves. Code: `backend/app/tts.py`.

- The voice is `am_michael`: **a** = American English, **m** = male, **michael** = the voice's name.
- Kokoro has 54 voices; 20 are American (11 female `af_…`, 9 male `am_…`) and fit the current `lang_code="a"`.

TODO: try other voices. Keep **one voice per show** at most, and log which one: the robot has one identity
(invariant 3), and an unlogged voice change would mix up which laughs came from the joke.

### 6. Laugh score

Each joke gets a score from 0 to 3. Right now the operator presses a key in **Live** mode; a laugh detector
comes later.

| Key | Score | Meaning |
|---|---|---|
| `W` | 3 | Most of the room, sustained |
| `K` | 2 | A clear laugh from a good part of the room, or applause |
| `Space` | 1 | A few chuckles, or a groan |
| `D` | 0 | Nothing, it fell flat |

Keys only count while the status reads **"React now"**. In Simulated mode the buttons can't be clicked. The
claude.ai artifact version of the page is a static demo where the buttons never work.

### 7. Learning

The scores train the director, and the logged shows (`data/sessions/*.jsonl`) become training data.

## Code concepts

### Interfaces in `show.py` (Python `Protocol`)

The top of `backend/app/show.py` defines interfaces: `TTS`, `Generator`, `SafetyFilter`, `Selector`, `Player`,
`Clock`, `RewardSource`. They are like Java interfaces, with one difference: no `implements` is needed.

In Java, a class has to say which interface it implements:

```java
class KokoroTTS implements TTS { ... }   // must name TTS, or it can't be used as one
```

In Python with `Protocol`, the class never mentions `TTS`:

```python
class TTS(Protocol):
    def synthesize(self, text: str) -> bytes: ...

class KokoroTTS:                          # no "TTS" anywhere
    def synthesize(self, text: str) -> bytes:
        ...real code...
```

`KokoroTTS` still counts as a `TTS` because it has a `synthesize(text) -> bytes` method. This is called
**structural typing** ("duck typing"). In practice, `run_show` takes any object with the right methods, real or
simulated, without changing `show.py`.

| Interface | Live show | Simulated / offline |
|---|---|---|
| `TTS` | `KokoroTTS` (`tts.py`) | same |
| `Generator` | `GeminiGenerator` (`generate.py`) | `BankGenerator` (`bank/`) |
| `SafetyFilter` | `LiveSafety` (`generate.py`) | `BankOnlySafety` (`live.py`) |
| `Player` | `WebSocketPlayer` (`live.py`) | `RealTimePlayer` (`sim/evaluate_live.py`) |
| `RewardSource` | `ButtonSource` (`live.py`), the operator's keys | `SimRoom` (`rooms.py`) |

### Tasks (T1–T12) vs experiments (Exp N)

- **T1–T12** are the build tasks for the live show, all done. They are listed in `docs/plans/live-loop.md`
  (table at line 107, one section per task after it).
- **Exp N** is a measurement that backs a decision, numbered across the whole project. Each has a section in a
  decisions notebook and an entry in `docs/decisions.md`. Some tasks include one: T1 → Exp 18, T8 → Exp 19,
  T9 → Exp 20, T10 → Exp 21, T11 → Exp 22. The DPO judge check is Exp 17.

## Running it locally

No deployment is needed. Two terminals:

```sh
# Terminal 1: backend
cd backend
.venv/bin/python -m uvicorn app.main:app --port 8000

# Terminal 2: frontend (npm install only the first time)
cd frontend
npm install
npm run dev
```

Then open http://localhost:3000. First-time backend setup (`uv venv`, `pip install -r requirements.txt`) is in
[backend/README.md](backend/README.md). Delete test shows from `data/sessions/` so they don't become training data.
