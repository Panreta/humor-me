# Stand-up structure

**Where it fits:** the bridge between "how comedians work" and "how the agent decides".
**Used in:** the long-form [set planner and bit writer](../design.md#long-form-set-generation),
the director's actions (tag again / next bit / callback), the transcript pipeline's labels
([beats, bits, sets](../../pipeline/wiki/beats-bits-sets.md)), and the sim's callback model.

## How a comedian builds material

```text
premise   "airport security is theater"          the idea, and the attitude toward it
 └ setup → punch                                  set an expectation, then break it so it still makes sense
    └ tags                                        more punches on the same setup, no new setup
       └ act-out                                  play the character or the moment
bit       one premise worked through              smallest unit that stands on its own
thread    consecutive bits on one topic
set       opener … body … closer                  ordered bits; closer often a callback
callback  return to an earlier bit's image later  pays off something planted earlier
```

Comedians write on paper but finish on stage: the same bit is rewritten over dozens of shows,
keeping the wording that gets the laugh. That loop (try → hear the room → revise) is what
the RL framing borrows.

## Why it matters for an agent

| Comedian's decision | Timescale | Agent's equivalent |
|---|---|---|
| Which premise and attitude for the next bit | Per bit | Director picks a **mode** ([contextual bandit](contextual-bandits.md)) |
| Tag again or move on | Per beat | Director: continue or cut on the laugh signal |
| When to spend a callback | Across bits | Delayed reward: a sequential decision, beyond a bandit |
| Which bits survive to the next show | Across shows | Offline learning from logged sessions |

The callback is the decision that makes this more than a bandit: planting a hook earns nothing
now and pays off later, and only if the hook landed in the first place.

## In this project

- **Hand-labeled definitions** (decisions.md 2026-10-21): a bit is the smallest standalone
  stretch; same-topic bits form a thread; a callback must reach back ≥ 2 bits.
- **Pipeline result:** bits F1 0.80 on Bargatze vs 0.21 baseline; callbacks 0/3 found
  (paused; [dev-log](../dev-log.md)).
- **Long-form generation:** we copy *structure* (setup–punch–tags, rule of three, callbacks,
  closer), never a comedian's wording or voice.

## Further reading

- [Seinfeld on writing the Pop-Tart joke (Salon, with the NYT video)](https://salon.com/2012/12/20/jerry_seinfeld_explains_how_he_writes_comedy):
  a five-minute look at how one bit is rewritten word by word.
- [Judy Carter: writing stand-up for beginners](https://www.judycarter.com/blog/writing-stand-up-comedy-for-beginners):
  premise, attitude, act-outs, mixes; the vocabulary we use for bits.
- [Backstage: how to use callbacks](https://www.backstage.com/magazine/article/callback-comedy-explained-76712/)
  and [Callback (comedy), Wikipedia](https://en.wikipedia.org/wiki/Callback_(comedy)): why
  callbacks spread as sets moved from one-liners to longer stories.
