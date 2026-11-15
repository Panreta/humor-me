"""The LLM generator: one call per bit returns K threads of the bit, as K candidate lines for each of its jokes.
Candidate i of every joke belongs to thread i, and the show plays one thread through (app.show._pick_thread).

Failure is the generator's to report and the show's to absorb: `generate` raises when the call or the reply is bad
(the same exception for every joke of that bit, with no second call), and the show falls back to the hand-checked
bank (app.show._prepare). Retries inside the call live in app.llm.
"""
import json

from app import llm, safety
from app.bank import BankGenerator
from app.show import Batch

K = 3  # threads per bit: enough for the filter to reject some and the selector to still choose
JOKES = 5
TEMPERATURE = 0.9
PROMPT_ID = "gen-v2"

# One identity, six attitudes (HumorGen personas as modes, CLAUDE.md invariant 3).
MODES = {
    "neurotic": "anxious and over-analytical: spirals about small things, rehearses worst cases",
    "cynic": "dry and deadpan: unimpressed, resigned, finds the system faintly absurd",
    "observer": "an observational comic: notices the small odd detail of everyday situations and says it plainly",
    "wordsmith": "built on wordplay: puns, double meanings, a phrase taken too literally",
    "optimist": "relentlessly cheerful about things going badly, which is the joke",
    "absurdist": "starts from a normal situation and commits to an escalating, illogical idea",
}
SYSTEM = """You write material for a live stand-up show that one character performs out loud: a self-aware, slightly anxious robot. It is always the same robot; the mode is only the attitude it takes in this bit.

Mode: {mode}: {mode_note}.

Write one bit in {k} alternative threads. A thread has its own premise angle and {jokes} jokes played in order (opener, tag, tag, tag, closer): the opener sets up the situation, each tag builds on it, and the closer calls back to a word or image from that same thread's opener. The {k} threads must take clearly different angles. Return them slot by slot: jokes[j][i] is slot j of thread i, so candidate i of every slot belongs to thread i and fits thread i's opener, never another thread's. The show plays one thread through.

Rules:
- Each line is one or two short sentences made for speaking aloud, under 30 words, with no stage directions or emoji.
- Aim every joke at the robot itself, technology, the situation, or things people say. Never at who anyone is: no jokes about bodies or appearance, no protected groups, no real or named people, nobody in the room singled out.
{teens}
Reply with JSON only: {{"premises": ["thread 0 premise", ...], "jokes": [[{k} lines], ... {jokes} lists]}}"""
TEENS_RULE = "- The audience is teenagers: no alcohol, drugs, sexual content, innuendo or swearing."

# In character, hand-written, one breath each: they only have to fill a few seconds while a bit loads.
BRIDGES = (
    "Give me a second. I'm checking my notes, and they just say don't panic, which is not a note.",
    "Hang on, I'm reorganizing my thoughts. Most of them weren't where I left them.",
    "One moment please. I'm running a confidence check, and it's buffering, like everything else I own.",
    "Okay, next bit. Let me make sure I haven't ruined it in advance. I usually have.",
    "Bear with me. I'm rehearsing this part one more time, mostly to be nervous about it properly.",
    "Quick pause while I load the next bit. Please enjoy this brief moment of me worrying.",
    "Just a second. My second half is in another folder, and I'm afraid to open it.",
    "While I think of something, let's all agree this is going great. I'll go first.",
    "Sorry, I'm recalculating. My plan said everything would go well, so something is wrong with the plan.",
    "Almost there. I'm double-checking the next joke for problems, and so far the main problem is me.",
)

# Seconds: (per-attempt timeout, deadline over all attempts), and the retries after the first attempt.
GENERATOR_TIMING = (10.0, 12.0)
FILTER_TIMING = (6.0, 8.0)
RETRIES = 2


class GenerationError(Exception):
    pass


def parse(reply: str) -> tuple[str, list[list[str]]]:
    try:
        data = json.loads(llm.strip_fence(reply))
        rows = data["jokes"]
        if not isinstance(rows, list) or not all(isinstance(r, list) for r in rows):
            raise TypeError("jokes must be a list of lists")  # a string row would be read as characters
        premises = data.get("premises") or data.get("premise") or ""
        premise = " | ".join(map(str, premises)) if isinstance(premises, list) else str(premises)
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        raise GenerationError(f"malformed reply: {type(e).__name__}") from e
    # Position is the thread, so a short or blank slot cannot be squeezed up: the whole reply is rejected.
    lines = [[s.strip() if isinstance(s, str) else "" for s in row[:K]] for row in rows[:JOKES]]
    if len(rows) < JOKES or any(len(row) < K or not all(row) for row in lines):
        raise GenerationError(f"expected {JOKES} jokes of {K} candidates, got {[len(r) for r in rows]}")
    return premise, lines


class GeminiGenerator:
    """One instance per show: it remembers premises already told so a bit does not repeat one."""

    def __init__(self, client: llm.Client):
        self.client = client
        self.generator_id = f"{PROMPT_ID}:{client.model}"
        self._bits: dict[int, tuple[str, list[list[str]]] | Exception] = {}
        self._premises: list[str] = []

    def generate(self, mode: str, bit: int, joke: int, context: dict) -> Batch:
        if bit not in self._bits:
            self._bits[bit] = self._fetch(mode, context)
        got = self._bits[bit]
        if isinstance(got, Exception):
            raise got
        prompt, lines = got
        return Batch(f"{PROMPT_ID}:{mode}:{bit}:{joke}", prompt, lines[joke])

    def _fetch(self, mode: str, context: dict) -> tuple[str, list[list[str]]] | Exception:
        teens = TEENS_RULE if context["age"] not in safety.STANDARD_AGES else ""
        system = SYSTEM.format(mode=mode, mode_note=MODES[mode], k=K, jokes=JOKES, teens=teens)
        user = json.dumps({"venue": context["venue"], "energy": context["energy"], "premises_already_told": self._premises})
        try:
            premise, lines = parse(self.client.chat(system, user, json_reply=True, temperature=TEMPERATURE))
        except (llm.LLMError, GenerationError) as e:
            return e
        self._premises.append(premise)
        return f"{system}\n\n{user}", lines


class LiveSafety:
    """The show's filter: rules, then one LLM verdict call per bit (`prefetch`), then cached `check`s."""

    def __init__(self, client: llm.Client):
        self.client = client

    def prefetch(self, texts: list[str], age: str) -> None:
        safety.prefetch(texts, age, self.client)

    def check(self, text: str, age: str) -> safety.Verdict:
        return safety.check(text, age)

    def check_rules(self, text: str, age: str) -> safety.Verdict:
        """The rules alone, for the bank fallback: hand-checked lines need no API call."""
        return safety.check_rules(text, age)


def llm_stack(room: str):
    """(generator, safety, fallback, bridge lines) for a show on the real model. Raises SafetyNotReady for a live
    room without the slur list. The deadlines keep a failing provider inside the bridge lines' cover."""
    safety.require_ready(room)
    return (GeminiGenerator(llm.from_env(GENERATOR_TIMING[0], retries=RETRIES, deadline_s=GENERATOR_TIMING[1])),
            LiveSafety(llm.from_env(FILTER_TIMING[0], retries=RETRIES, deadline_s=FILTER_TIMING[1])),
            BankGenerator(), list(BRIDGES))
