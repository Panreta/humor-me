# humor-me

Research project: a live, voice-only stand-up comedy agent that reads the room via laughter,
adapts its material, and learns from audience feedback across sessions.

Read before working:
- [docs/design.md](docs/design.md) — architecture, reward design, research hypothesis
- [docs/data-sources.md](docs/data-sources.md) — what data we use and for what role
- [docs/pipeline.md](docs/pipeline.md) — transcript → beats/bits/sets pipeline, labels, schemas
- [docs/roadmap.md](docs/roadmap.md) — stages, exit criteria, estimates
- [docs/decisions.md](docs/decisions.md) — decision log; append, don't rewrite history
- [references/md/](references/md/) — markdown versions of the reference papers

## Invariants (do not violate without a new entry in decisions.md)

1. **Never joke about bodies, appearance, or protected attributes** of people present.
   Punch at self, the situation, and what people *say* — never at who they *are*.
2. **No face/video in the realtime loop.** Audio only. One optional async scene snapshot at
   session start, converted to text context, then discarded. Never persist images or faces.
3. **One core identity.** A self-aware, slightly anxious robot. HumorGen personas are *modes*
   the identity draws on, not separate characters.
4. **Can take a roast.** Roasts aimed at the bot get concede-and-escalate, never hostility.
5. **Per-joke laugh signal is the training reward; episode rating is the reported outcome.**
   Don't optimize raw laugh counts (gameable by joke spam, confounded by room size).
6. **Log everything** needed to replay an episode (context, candidates, chosen joke, mode,
   timing, laugh features, ratings). Logged sessions are the primary dataset.

## Paper roles (common mix-up)

- **HumorGen** = six cognitive personas → candidate generation, LLM-judge Elo, distillation.
- **LoL (Innovative Thinking, Infinite Humor)** = thought-leap reasoning + frozen pairwise
  judge for online DPO. It does *not* classify personas.
