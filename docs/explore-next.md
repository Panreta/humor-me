# Explore next (backlog)

Ideas raised but not started. One at a time: pick from here, open a feature branch, and move the
row to "Done" with its PR. Add a row whenever something is deferred so it isn't lost.

| # | Idea | Why it might matter | Blocked by | Raised |
|---|---|---|---|---|
| 1 | **Human mode preference from rJokes upvotes**: Bradley–Terry over modes on year/length-matched pairs; rank-correlate with HumorGen's judge ranking | Tests whether an LLM judge ranks personas the way people do; would replace the sim's `base` (PR 3) | Explorer labeling run (`explorer/`) | 2026-10-27 |
| 2 | SemEval-2021 Task 7 "HaHackathon" humor ratings | Ratings from annotators of different ages → could inform the `age` context | Verify license + contents; only if #1 falls short | 2026-10-27 |
| 3 | New Yorker caption contest crowd ratings (Hessel et al. 2023) | Large-scale human funniness ratings, not popularity-driven like upvotes | Verify license + contents; only if #1 falls short | 2026-10-27 |
| 5 | Callbacks + stateful director (show-so-far features) | Turns the bandit into a sequential decision; the shift experiment showed a bandit can't react within a show | – | 2026-10-21 |
| 6 | Tag-or-move-on within a bit | The director's second decision; matches how comedians ride a laugh | Callbacks (#5) | 2026-10-27 |
| 7 | Prior strength sweep (weak vs strong HumorGen prior) | Robustness: how much a confident wrong prior costs | PR 2 results | 2026-10-27 |
| 8 | Judge-as-audience (idea B): LLM judge scores generated jokes in the loop | More realistic reward than a parametric room | Sim + bandits done (stretch) | 2026-10-26 |
| 9 | Learned joke scorer on rJokes (encoder regressor) | Cheap reranker for K candidates | Explorer; noise ceiling caps it | 2026-10-26 |
| 10 | Transcript callback pass + Birbiglia held-out eval | Completes the paused pipeline's report appendix | Gemini quota; Birbiglia labels | 2026-10-21 |

## Done

| Idea | PR | Outcome |
|---|---|---|
| Per-joke vs per-bit decisions | #1 | Per bit (decisions.md 2026-10-27) |
| Priors: HumorGen vs uniform vs wrong | #4 | No significant difference at weight 1; LinUCB + uniform chosen |
| Mid-set preference shift | (this PR) | Learners drop 95% → 80–83% of oracle; recover no faster than random; reaction feature doesn't help → needs show state |
