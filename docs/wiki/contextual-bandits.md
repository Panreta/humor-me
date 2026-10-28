# Contextual bandits

**Where it fits:** the RL core of M1, and what the course grade rests on.
**Used in:** the director's mode choice (`sim/` policies, PR 2), the live loop once laughs are
detected, and the regret curves in the report.

## The problem

Each round the agent sees a **context** (venue, energy, age, how the set is going), picks one
of K **arms** (the 6 modes), and gets a **reward** only for the arm it picked (laugh level / 3).
It must **explore** (try modes to learn them) and **exploit** (use what works) at the same time.

It's a bandit, not full RL, because the choice doesn't change future states. That stops being
true once fatigue (repeating a mode tires the room) and callbacks enter; see "Where the bandit
view breaks" below.

**Regret** = Σ (best expected reward − expected reward of the chosen arm). Good algorithms grow
regret like √T or log T; a policy that never learns grows it linearly.

## The algorithms we compare

| Policy | Idea | Strength | Weakness |
|---|---|---|---|
| Fixed / random | Baselines | Fixed = "best single persona" (HumorGen says Neurotic) | Can't adapt |
| ε-greedy | Exploit the best mean; explore at random with probability ε | Trivial to explain | Explores blindly, forever |
| UCB / **LinUCB** | Pick the arm with the highest *optimistic* estimate; LinUCB models reward as linear in context features | Shares learning across contexts; deterministic, easy to inspect | Linear assumption; needs features designed |
| **Thompson sampling** | Sample a belief per arm, play the best sample | Strong in practice; priors come in naturally (rJokes, HumorGen) | Needs a reward model (Gaussian here, since reward is level / 3) |

## Where the bandit view breaks (and why that's the interesting part)

1. **Fatigue.** If repeating a mode tires the room, the best arm depends on recent history.
   The sim's first metrics showed random beating every fixed mode for exactly this reason
   ([simulated audience](simulated-audience.md)). Fix: put recent modes in the context, or
   decide per bit instead of per joke.
2. **Too little data per show.** Separating the top two modes takes ~550–7,000 jokes; a set
   has 40. Within one show, priors matter most; across shows, the bandit keeps learning
   per venue.
3. **Callbacks.** Reward arrives bits later → an MDP. The planned experiment: greedy bandit vs a
   sequential policy that invests in hooks.
4. **Coherence.** Switching mode every joke is incoherent for a performer; a switching penalty
   (or deciding per bit) encodes that.

## In this project

- Arms: 6 HumorGen modes. Context: venue / energy / age (+ recent history, pending decision).
- Reward: 0–3 laugh level scaled to [0, 1]; logs keep the level.
- Evaluated in [simulation](simulated-audience.md) first, then live with an operator laugh
  button as a fallback reward.

## Further reading

- [Li et al. 2010, LinUCB (arXiv 1003.0146)](https://arxiv.org/abs/1003.0146): the original
  contextual bandit for news recommendation; Algorithm 1 is what we implement.
- [Russo et al., A Tutorial on Thompson Sampling (arXiv 1707.02038)](https://arxiv.org/abs/1707.02038):
  chapters 1–4 are enough; good intuition for priors.
- [Lattimore & Szepesvári, Bandit Algorithms (free online edition)](https://tor-lattimore.com/downloads/book/book.pdf):
  the reference; ch. 1 (intro), ch. 19 (LinUCB), ch. 36 (Thompson sampling).
- [Sutton & Barto, RL: An Introduction, ch. 2](http://incompleteideas.net/book/the-book-2nd.html):
  ε-greedy, UCB, and non-stationary rewards, the course-friendly version.
- [Simons Institute: Online Learning and Bandits, part 2 (video)](https://simons.berkeley.edu/talks/online-learning-bandits-part-2):
  UCB, EXP3, contextual bandits, lower bounds.
