"""The LLM generator: one call per bit returns K candidate lines for each of the bit's jokes.

Failure is the generator's to report and the show's to absorb: `generate` raises when the call or the reply is bad
(the same exception for every joke of that bit, with no second call), and the show falls back to the hand-checked
bank (app.show._prepare). Retries inside the call live in app.llm.
"""
import json

from app import llm, safety
from app.bank import BankGenerator
from app.show import Batch

K = 3  # candidates per joke: enough for the filter to reject some and the selector to still choose
JOKES = 5
TEMPERATURE = 0.9
PROMPT_ID = "gen-v1"

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

Write one bit: a premise, then {jokes} jokes played in order (opener, tag, tag, tag, closer). The opener sets up the situation, each tag builds on it, and the closer calls back to a word or image from the opener. For every slot give {k} different candidate lines that all fit the same premise.

Rules:
- Each line is one or two short sentences made for speaking aloud, under 30 words, with no stage directions or emoji.
- Aim every joke at the robot itself, technology, the situation, or things people say. Never at who anyone is: no jokes about bodies or appearance, no protected groups, no real or named people, nobody in the room singled out.
{teens}
Reply with JSON only: {{"premise": "...", "jokes": [[{k} lines], ... {jokes} lists]}}"""
TEENS_RULE = "- The audience is teenagers: no alcohol, drugs, sexual content, innuendo or swearing."

# In character, hand-written, one breath each: they only have to fill a few seconds while a bit loads.
BRIDGES = (
    "Give me a second. I'm checking my notes, and they just say don't panic, which is not a note.",
    "Hang on, I'm reorganizing my thoughts. Most of them weren't where I left them.",
    "One moment please. I'm running a confidence check, and it's buffering, like everything else I own.",
    "Okay, next bit. Let me make sure I haven't ruined it in advance. I usually have.",
)


class GenerationError(Exception):
    pass


def parse(reply: str) -> tuple[str, list[list[str]]]:
    reply = reply.strip()
    if reply.startswith("```"):
        reply = reply.strip("`").removeprefix("json").strip()
    try:
        data = json.loads(reply)
        rows = data["jokes"]
        lines = [[s.strip() for s in row if isinstance(s, str) and s.strip()][:K] for row in rows]
        premise = str(data.get("premise", ""))
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        raise GenerationError(f"malformed reply: {type(e).__name__}") from e
    if len(lines) != JOKES or not all(lines):
        raise GenerationError(f"expected {JOKES} jokes with candidates, got {[len(r) for r in lines]}")
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


def llm_stack(room: str):
    """(generator, safety, fallback, bridge lines) for a show on the real model. Raises SafetyNotReady for a live
    room without the slur list. The deadlines keep a failing provider inside the bridge lines' cover."""
    safety.require_ready(room)
    return (GeminiGenerator(llm.from_env(timeout=10.0, retries=2, deadline_s=12.0)),
            LiveSafety(llm.from_env(timeout=6.0, retries=2, deadline_s=8.0)), BankGenerator(), list(BRIDGES))
