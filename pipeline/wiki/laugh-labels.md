# Reaction markers & laugh labels (S1–S2)

Code: [`pipeline/markers.py`](../markers.py) · Config: [`data/markers.yaml`](../../data/markers.yaml)

## Why laugh labels at all

The agent needs a reward that says "this landed". A live audience gives laughter; for
training before any live show, the transcripts' audience markers (`[audience laughs]`,
`(laughter)`) are the only real signal of what got a laugh. They become labels for a **laugh
predictor** (reward model) and priors for the simulated audience.

## The problem: every transcriber writes differently

Across 20 transcripts there are ~4,500 bracketed strings and ~425 distinct ones:

- The same reaction has many spellings: "laughter", "audience laughing", "crowd laughs".
- Most transcribers never mark a chuckle-level audience reaction or an intensity.
- **Unattributed "chuckles", "laughs" and "sighs" are the *performer*.** Subtitle convention:
  speaker sounds have no subject, audience sounds are prefixed. "jeff laughs" is Jeff
  Arcuri himself, not an audience member.
- Some transcribers tag laughs every ~30 words, others every ~500+, so missing markers in a
  sparse transcript don't mean "no laugh".

## How we label

1. **Extract** every `[...]` / `(...)` (2–60 chars), remember its offset, remove it from the
   text → `clean.txt` + `events.jsonl`.
2. **Bucket.** Reaction-like strings (laugh, chuckle, applause, cheer, groan, gasp…) must be
   in the lexicon. Everything else is `act_out` (voices, imitations), `music`, or `direction`.
3. **Lexicon lookup.** `markers.yaml` maps each exact reaction string to `kind`, `source` and
   optional `intense`. An unknown reaction string **stops the run**. `--draft` proposes entries
   for review; the level is never guessed at run time.
4. **Levels** (audience only):

   | Level | Meaning | From |
   |---|---|---|
   | 0 | none | performer sounds, entrance/exit applause (first/last 2%) |
   | 1 | chuckle | audience chuckles, scattered |
   | 2 | laugh | laughter, audience laughs |
   | 3 | big laugh | laugh + "roars / wildly / continues", or back-to-back laugh markers |
   | 4 | applause break | laugh + applause/cheer mid-set, or applause mid-set |

   Groans and gasps are **flags**, not levels: a groan at a pun can be the payoff.
5. **Merge.** Audience reactions fewer than 3 words apart are one sustained reaction.
6. **Interrupt flag.** A marker that cuts a sentence mid-line means the laugh came before the
   line finished, which is a strong-laugh signal.
7. **Density gate.** `laugh_labeled = words_per_audience_laugh ≤ 150`. Dense sets sit at
   26–80, sparse ones at 480+. Sparse and unmarked sets are structure-only.

## How the labels are used

- **Binary (level ≥ 2)** is the primary target; it holds up across transcribers.
- **Ordinal** is only compared *within* a set ("this beat got more than that one").
- The laugh predictor is evaluated on **held-out comedians**, not held-out beats, so it can't
  succeed by learning one transcriber's habits.

## Known limits

- Levels 1 and 3 are rare because few transcribers write them, so ordinal signal is thin.
- Markers reflect the *edited special*, not the raw room: editors cut dead air and sweeten laughs.
- 8 training sets is small; plot a learning curve before collecting more.
