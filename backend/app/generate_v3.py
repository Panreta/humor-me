"""FROZEN gen-v3: the one-call generator (3 threads in one reply) that Exp 24's stored run was made with.
Nothing here may change: `sim.compare_generators` hashes these names (`gen_digest`, `2a512264`) and its `score`
asserts the hash still matches the stored calls. The show itself runs gen-v4 (app.generate, D-110); this module is
only for the Exp 24 harness (`sim.compare_generators`, `sim.screen_local`). Shared pieces (modes, teens rule, the
filter's timing, the error class) come from app.generate and are hashed too, so a change there is caught as well.
"""
import json

from app import llm, safety
from app.generate import FILTER_RETRIES, FILTER_TIMING, MODES, TEENS_RULE, GenerationError  # noqa: F401 - the harness reads them here
from app.show import Batch

K = 3
JOKES = 5
TEMPERATURE = 0.9
PROMPT_ID = "gen-v3"
GENERATOR_TIMING = (10.0, 12.0)
RETRIES = 2

# The one reply shape `parse` accepts, for providers that can enforce it (Claude structured output, Ollama json_schema).
REPLY_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["premises", "threads"],
                "properties": {"premises": {"type": "array", "items": {"type": "string"}},
                               "threads": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}}}}

SYSTEM = """You write material for a live stand-up show that one character performs out loud: a self-aware, slightly anxious robot. It is always the same robot; the mode is only the attitude it takes in this bit.

Mode: {mode}: {mode_note}.

Write one bit in {k} alternative threads. A thread has its own premise angle and {jokes} jokes played in order (opener, tag, tag, tag, closer): the opener sets up the situation, each tag builds on it, and the closer calls back to a word or image from that same thread's opener. The {k} threads must take clearly different angles. Return each thread as its own list of {jokes} lines, in playing order. The show plays one thread through, or swaps single lines between threads.

Rules:
- Each line is one or two short sentences made for speaking aloud, under 30 words, with no stage directions or emoji.
- Aim every joke at the robot itself, technology, the situation, or things people say. Never at who anyone is: no jokes about bodies or appearance, no protected groups, no real or named people, nobody in the room singled out.
{teens}
Reply with JSON only, exactly this shape: {{"premises": ["premise of thread 0", ...], "threads": [[opener, tag, tag, tag, closer], ...]}} with {k} threads of {jokes} lines each."""


def parse(reply: str) -> tuple[list[str], list[list[str]]]:
    """Accepts one shape only: {"premises": [...], "threads": [[JOKES lines] x K]}, thread-major. Returns the premises
    (one per thread when given) and the lines joke-major (lines[j][i] is joke j of thread i), which is what the show's
    batches hold. Any other shape or count raises, and so does a line with no letters (emoji or symbols only, which
    the voice cannot say): a thread cannot be repaired by guessing."""
    try:
        data = json.loads(llm.strip_fence(reply))
        threads = data["threads"]
        premises = data.get("premises") or []
        premises = [str(x) for x in premises] if isinstance(premises, list) else [str(premises)]
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        raise GenerationError(f"malformed reply: {type(e).__name__}") from e
    if (not isinstance(threads, list) or len(threads) != K
            or not all(isinstance(t, list) and len(t) == JOKES for t in threads)
            or not all(isinstance(line, str) and any(c.isalpha() for c in line) for t in threads for line in t)):
        shape = [len(t) if isinstance(t, list) else type(t).__name__ for t in threads] if isinstance(threads, list) else "?"
        raise GenerationError(f"expected {K} threads of {JOKES} lines with letters in them, got {shape}")
    return premises, [[threads[i][j].strip() for i in range(K)] for j in range(JOKES)]


class GeminiGenerator:
    """One instance per show. It tells the model which premises the room has heard, so a bit does not repeat one:
    only the thread the show actually played counts (`played_thread`), not the alternatives that were generated."""

    def __init__(self, client: llm.Client):
        self.client = client
        self.generator_id = f"{PROMPT_ID}:{client.model}"
        self._bits: dict[int, tuple[str, list[list[str]], list[str]] | Exception] = {}
        self._told: list[str] = []

    def generate(self, mode: str, bit: int, joke: int, context: dict) -> Batch:
        if bit not in self._bits:
            self._bits[bit] = self._fetch(mode, context)
        got = self._bits[bit]
        if isinstance(got, Exception):
            raise got
        prompt, lines, _ = got
        return Batch(f"{PROMPT_ID}:{mode}:{bit}:{joke}", prompt, lines[joke])

    def played_thread(self, bit: int, thread: int) -> None:
        got = self._bits.get(bit)
        if not isinstance(got, Exception) and got and thread < len(got[2]):
            self._told.append(got[2][thread])

    def _fetch(self, mode: str, context: dict) -> tuple[str, list[list[str]], list[str]] | Exception:
        teens = TEENS_RULE if context["age"] not in safety.STANDARD_AGES else ""
        system = SYSTEM.format(mode=mode, mode_note=MODES[mode], k=K, jokes=JOKES, teens=teens)
        user = json.dumps({"venue": context["venue"], "energy": context["energy"], "premises_already_told": self._told})
        try:
            premises, lines = parse(self.client.chat(system, user, json_reply=REPLY_SCHEMA, temperature=TEMPERATURE))
        except (llm.LLMError, GenerationError) as e:
            return e
        return f"{system}\n\n{user}", lines, premises
