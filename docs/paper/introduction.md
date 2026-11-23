<!-- Mirror of introduction.tex; edit both together. Citations as in references.bib. -->
# 1 Introduction

Large language models can write a passable joke on request, but stand-up comedy is not a list of jokes. A
comedian performs to a specific room, notices what lands and what falls flat, and changes course mid-set. Over
many shows they also learn which material suits which audience, so that the cynical bit that works in a late-night
bar gets dropped at a company party. This feedback loop matches reinforcement learning closely: a comedian takes
an action (a bit), the audience returns a reward (laughter), and later choices change in response. Recent work on
humor generation improves what a model writes, either by combining several comedic personas [HumorGen] or by
training on preferences from a frozen judge [Wang et al., 2025]. These methods score jokes offline, one at a time.
None of them perform for an audience, listen to the response, or carry what they learned into the next show.

We present **humor-me**, a voice-only stand-up agent that reads the room through laughter and adapts its material
both within a show and across shows. The agent has one fixed identity, a self-aware and slightly anxious robot. It
performs a set of eight bits with five jokes each. For each bit, it picks one of six comedic *modes* from HumorGen
[HumorGen]: neurotic, cynic, observer, wordsmith, optimist and absurdist. The modes are styles that this single identity uses,
not separate characters. A show runs as a closed loop of six stages:

1. **Setup.** The operator describes the room by venue, energy and audience age. Those features are the context
   for every later decision.
2. **Director.** A contextual bandit chooses the mode for the next bit. We use LinUCB [Li et al., 2010] with a
   uniform prior over modes, wrapped in an ε-soft policy (ε = 0.2) so it keeps exploring. The director is rebuilt
   at the start of each show by replaying every logged show, so lessons from earlier rooms carry over to similar
   ones.
3. **Joke writer.** A language model writes candidate jokes in the chosen mode. It generates one batch per bit
   ahead of playback, so the room never waits on the model. When the next bit is not ready, a short bridge line
   fills the gap. A hand-written joke bank serves as the fallback.
4. **Safety check.** Every line passes a filter that fails closed. Hard rules run first, then one yes/no call to a
   language model per bit. Any failure rejects the line, including a timeout. The filter enforces a fixed
   invariant: the agent jokes about itself, the situation and what people *say*, never about bodies, appearance or
   protected attributes.
5. **Voice.** A local text-to-speech model, Kokoro [Kokoro], speaks each line. The agent works from audio only and never
   sees the audience.
6. **Laugh score and learning.** Each joke gets a score from 0 (nothing) to 3 (most of the room, sustained). The
   per-joke score is the reward that updates the director. The audience's rating of the whole show is the outcome
   we report, because raw laugh counts can be gamed by telling more jokes and are confounded by room size. Every
   show is logged in enough detail to replay it, and these logs become the training data for the agent.

Right now a human operator enters the per-joke score with one key press per joke. An automatic laugh detector
will take over this role later, with the keys kept as a fallback. The modular design lets either one stand in,
along with a simulated audience, without changing the show loop.

Our central hypothesis is that **choosing the mode from live laughter yields more laughter and higher show
ratings than performing a single fixed persona or switching modes at random.** We test it in two settings. The
first is a simulated audience that has a hidden taste for each mode in each kind of room, which lets us compare
policies over hundreds of shows using the same random seeds for every policy. The second is the live loop above,
run in front of real people.

In simulation, the LinUCB director reaches 97% of the laughter collected by an oracle that knows the room's true
preferences. It also earns 18% more laughter than always performing in the neurotic mode, which an LLM judge rated
the funniest. Most of the gain comes from transferring lessons between similar rooms, not from the starting guess.
The result holds when the simulated tastes are replaced with ones taken from human upvotes on Reddit jokes (96% of
the oracle). Two caveats limit these numbers. First, the simulated tastes are invented, so the simulation can
compare policies but cannot show that they are funny. Second, the director recognizes rooms but not a room that
*changes*: after a mid-set shift in taste, its share of the oracle falls from 95% to 80%. Richer state
representations and in-show planning did not fix this in our experiments. We report them as negative results.

The live system passed its acceptance test. It ran three clean simulated shows and one clean live show of 40
jokes with full audio capture. The safety filter rejected 29 of 30 fresh, indirect, unsafe lines. The per-bit
reward noise measured in the live show (0.121) was below what the simulation assumes (0.15), so the noise level
behind the simulated results is not optimistic.

This paper makes four contributions:

- **A closed-loop stand-up agent** that performs to a room, listens for laughter and learns from it across shows,
  combining persona-based generation with a contextual bandit for selection.
- **A simulated audience and evaluation protocol** that compares director policies on paired seeds against an
  oracle, with every pass criterion set before the experiment runs.
- **Empirical evidence about adaptation.** Cross-room learning recovers almost all of the oracle's laughter.
  Adapting within a show to a room whose taste changes remains an open problem.
- **A deployable live system and dataset.** The system covers safety filtering, text-to-speech, consent-gated
  audio capture and replayable show logs. The logs supply preference data for later training of the joke writer
  with direct preference optimization (DPO) [Rafailov et al., 2023].

Section 2 reviews related work on computational humor, preference learning [Christiano et al., 2017] and
contextual bandits. Section 3 describes the system. Section 4 describes the simulated audience. Section 5 reports
the experiments, and Section 6 discusses limitations and the planned move from operator scores to automatic laugh
detection and from a general-purpose LLM to a DPO-trained joke writer.

## References

- [HumorGen] HumorGen: Cognitive Synergy for Humor Generation in LLMs via Persona-Based Distillation.
- [Wang et al., 2025] Innovative Thinking, Infinite Humor: Humor Research of LLMs through Structured Thought Leaps. ICLR 2025.
- [Li et al., 2010] A Contextual-Bandit Approach to Personalized News Article Recommendation. WWW 2010.
- [Christiano et al., 2017] Deep Reinforcement Learning from Human Preferences. NeurIPS 2017.
- [Rafailov et al., 2023] Direct Preference Optimization: Your Language Model is Secretly a Reward Model. NeurIPS 2023.
- [Kokoro] hexgrad. Kokoro-82M. https://huggingface.co/hexgrad/Kokoro-82M, 2025.
