"""Material for a bit (D-117): topic and character-sheet facts, drawn per show and spent only when a bit is played.
Synthetic topics and facts only; no model."""
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))

from app import generate, material  # noqa: E402
from app.live import load_sessions  # noqa: E402
from app.log import SessionLog, read_session, validate  # noqa: E402
from tests.test_generate import CTX, Filter, thread_reply  # noqa: E402
from tests.test_headless_show import director, same  # noqa: E402
from tests.test_predict_next import core, play, rows_of  # noqa: E402

TOPICS = [(f"topic {i}", i in (2, 5), "character" if i >= 6 else "everyday") for i in range(8)]  # 2 and 5 are adult; 6 and 7 are character
FACTS = [f"fact {i}" for i in range(6)]


def mat(seed=1, history=None, topics=None):
    return material.ShowMaterial(seed, history, facts=FACTS, topics=TOPICS if topics is None else topics)


def row(topic, at, event="material"):
    return {"event": event, "topic": topic, "at": at}


# --- the draw ---

def test_a_draw_is_deterministic_for_a_seed_and_twice_the_same_without_a_play():
    a, b = mat(7), mat(7)
    assert a.draw(0, "mixed") == b.draw(0, "mixed") == a.draw(0, "mixed")
    assert a.draw(0, "mixed")["sheet_id"] == "character-v1" and a.draw(0, "mixed")["list_id"] == "topics-v1"
    assert len({mat(s).draw(0, "mixed")["topic"] for s in range(20)}) > 1  # the seed matters


def test_a_topic_played_this_show_is_not_drawn_again_and_facts_rotate():
    m = mat()
    seen_topics, seen_facts = [], []
    for bit in range(3):
        d = m.draw(bit, "mixed")
        assert len(d["facts"]) == 2 and not d["recycled"]
        seen_topics.append(d["topic"])
        seen_facts += d["facts"]
        m.played(bit, d)
    assert len(set(seen_topics)) == 3 and len(set(seen_facts)) == 6  # 6 facts, 2 a bit: none repeats in 3 bits


def test_a_topic_used_in_an_earlier_show_is_skipped():
    first = mat(3).draw(0, "mixed")["topic"]
    later = mat(3, history={material.normalise(first): 10.0}).draw(0, "mixed")
    assert later["topic"] != first and not later["recycled"]


def test_a_teen_room_skips_adult_topics_a_standard_room_may_draw_them():
    for seed in range(30):
        m = mat(seed)
        for bit in range(6):  # all 6 teen-safe topics, so no draw may be adult
            d = m.draw(bit, "teens")
            assert d["topic"] not in ("topic 2", "topic 5")
            m.played(bit, d)
    assert any(mat(s).draw(0, "mixed")["topic"] in ("topic 2", "topic 5") for s in range(30))


def test_when_every_topic_is_used_the_least_recently_used_is_recycled():
    history = {material.normalise(t): float(10 + i) for i, (t, *_) in enumerate(TOPICS)}
    d = mat(history=history).draw(0, "mixed")
    assert d["topic"] == "topic 0" and d["recycled"]
    # this show's own plays count as the latest
    m = mat(history=history)
    m.played(0, m.draw(0, "mixed"))
    assert m.draw(2, "mixed")["topic"] == "topic 1"


def test_a_teen_room_recycles_only_among_teen_safe_topics():
    history = {material.normalise(t): float(i) for i, (t, *_) in enumerate(TOPICS)}  # topic 2 is the oldest adult one
    d = mat(history=history).draw(0, "teens")
    assert d["topic"] == "topic 0" and d["recycled"]
    history["topic 0"] = history["topic 1"] = 99.0
    assert mat(history=history).draw(0, "teens")["topic"] == "topic 3"


def test_every_third_bit_takes_a_character_topic_the_rest_everyday():
    assert material.CHARACTER_EVERY == 3
    m = mat(topics=[(f"c{i}", False, "character") for i in range(4)] + [(f"e{i}", False, "everyday") for i in range(8)])
    got = []
    for bit in range(8):
        d = m.draw(bit, "mixed")
        got.append((d["from"], d["topic"][0]))
        m.played(bit, d)
    assert [g[0] for g in got] == ["everyday", "character", "everyday", "everyday", "character", "everyday", "everyday",
                                   "character"]
    assert all(src[0] == t for src, t in [(g[0], g[1]) for g in got])


def test_an_empty_pool_falls_back_to_the_other_before_recycling():
    m = mat(topics=[("c0", False, "character"), ("e0", False, "everyday"), ("e1", False, "everyday")])
    drawn = []
    for bit in range(3):  # bit 1 takes c0; bit 4 wants character but none is left
        d = m.draw(bit, "mixed")
        drawn.append((d["topic"], d["recycled"]))
        m.played(bit, d)
    assert drawn[1] == ("c0", False) and {drawn[0][0], drawn[2][0]} == {"e0", "e1"} and not drawn[0][1] and not drawn[2][1]
    d = m.draw(4, "mixed")  # every topic used: recycle within the bit's own pool (character), the oldest play
    assert d["recycled"] and d["from"] == "character" and d["topic"] == "c0"


def test_playing_a_bit_twice_spends_it_once():
    m = mat()
    d = m.draw(0, "mixed")
    m.played(0, d)
    m.played(0, d)
    assert m.draw(1, "mixed")["facts"] != d["facts"] and len(m._topic_plays) == 1


def test_used_topics_reads_every_show_and_keeps_the_latest_time():
    sessions = [[row("Topic A", 5.0), row("topic b", 9.0), row("x", 1.0, event="joke_played")], [row("topic a", 7.0)]]
    assert material.used_topics(sessions) == {"topic a": 7.0, "topic b": 9.0}


def test_the_shipped_files_load_with_the_ids_and_enough_to_rotate():
    assert len(material.load_sheet()) >= 2 * 8
    topics = material.load_topics()
    assert len({t for t, *_ in topics}) == len(topics)
    assert any(adult for _, adult, _ in topics) and {src for *_, src in topics} == {"character", "everyday"}
    # 5 shows of 8 bits never recycle: bits 1, 4, 7 are character topics, the other 5 are everyday ones
    n_char = sum(src == "character" for *_, src in topics)
    assert n_char >= 5 * 3 and len(topics) - n_char >= 5 * 5


# --- the generator ---

class Writer:
    """A client that records what it was asked and answers every call with one thread."""
    model, deadline_s, local = "w", None, False

    def __init__(self):
        self.calls = []

    def chat(self, system, user, json_reply=None, temperature=None, validate=None, deadline_s=None):
        self.calls.append((system, json.loads(user)))
        return thread_reply(tag="ok")


def test_prompts_are_untouched_and_the_off_generator_sends_the_old_request():
    w = Writer()
    g = generate.ThreadGenerator(w)
    g.generate("cynic", 0, 0, CTX)
    system, user = w.calls[0]
    assert generate.MATERIAL_PARAGRAPH not in system and set(user) == {"venue", "energy", "premises_already_told"}
    assert g.generator_id == f"{generate.PROMPT_ID}:w"
    assert all(generate.MATERIAL_PARAGRAPH not in p for p in generate.PROMPTS.values())


@pytest.mark.parametrize("prompt_id", ["gen-v4", "gen-v5"])
def test_with_material_on_the_user_json_and_one_paragraph_are_added(prompt_id):
    w = Writer()
    g = generate.ThreadGenerator(w, prompt_id=prompt_id, material=mat())
    g.generate("cynic", 0, 0, CTX)
    system, user = w.calls[0]
    assert system == generate.PROMPTS[prompt_id].format(mode="cynic", mode_note=generate.MODES["cynic"], jokes=5,
                                                        teens="") + "\n\n" + generate.MATERIAL_PARAGRAPH
    assert user["topic"] == mat().draw(0, "mixed")["topic"] and len(user["robot_facts"]) == 2
    assert g.generator_id == f"{prompt_id}+mat-v1:w"


def test_material_is_spent_when_the_thread_is_played_not_when_it_is_written():
    g = generate.ThreadGenerator(Writer(), material=mat())
    g.generate("cynic", 0, 0, CTX)
    assert g.material_row(0) is None and g.material._topic_plays == []
    g.played_thread(0, 0)
    got = g.material_row(0)
    assert got["topic"] == g.material._topic_plays[0] and got["seed"] == 1 and got["recycled"] is False
    assert set(got) == {"topic", "from", "facts", "sheet_id", "list_id", "recycled", "seed"}


def test_a_discarded_guess_consumes_nothing_and_the_real_bit_draws_the_same():
    g = generate.ThreadGenerator(Writer(), material=mat())
    g.played_thread(0, 0) if g.generate("cynic", 0, 0, CTX) else None
    g.generate_speculative("cynic", 1, 0, CTX)
    first = g._spec_drawn[(1, "cynic")]
    g.discard(1, "cynic")
    assert g._spec_drawn == {} and g.material_row(1) is None and len(g.material._topic_plays) == 1
    g.generate("observer", 1, 0, CTX)
    assert g._drawn[1] == first  # nothing was spent, so the next draw for the bit is the same


def test_an_adopted_bit_keeps_the_material_it_was_written_with():
    w = Writer()
    g = generate.ThreadGenerator(w, material=mat())
    g.generate_speculative("cynic", 1, 0, CTX)
    written_with = w.calls[0][1]["topic"], w.calls[0][1]["robot_facts"]
    g.material.history = {material.normalise(written_with[0]): 1.0}  # even if the history moved meanwhile
    g.adopt(1, "cynic")
    g.played_thread(1, 0)
    got = g.material_row(1)
    assert (got["topic"], got["facts"]) == written_with
    assert g.material._topic_plays == [written_with[0]]


def test_a_bank_bit_has_no_material_row():
    g = generate.ThreadGenerator(Writer(), material=mat())
    assert g.material_row(3) is None


def test_material_setting_is_off_unless_asked():
    assert not material.enabled({}) and not material.enabled({"MATERIAL": "0"}) and not material.enabled({"MATERIAL": ""})
    assert all(material.enabled({"MATERIAL": v}) for v in ("1", "true", "YES", " on "))


def test_llm_stack_builds_material_only_when_the_setting_is_on(monkeypatch, tmp_path):
    from app import safety
    monkeypatch.setattr(safety, "BLOCKLIST_LOCAL", tmp_path / "missing.txt")
    for name in ("JOKE_WRITER_PROVIDER", "JOKE_WRITER_MODEL", "SAFETY_CHECK_PROVIDER", "SAFETY_CHECK_MODEL", "GEMINI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_BASE_URL", "http://x")
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.delenv("MATERIAL", raising=False)
    assert generate.llm_stack("sim")[0].material is None
    monkeypatch.setenv("MATERIAL", "1")
    gen = generate.llm_stack("sim")[0]
    assert isinstance(gen.material, material.ShowMaterial) and "+mat-v1" in gen.generator_id


# --- the show and the log ---

def show(tmp_path, predict, seed):
    gen = generate.ThreadGenerator(Writer(), material=mat(seed=5))
    gen.speculative_safe = True
    rows, _, _, d = play(tmp_path, n_bits=4, gen=gen, seed=seed, d=director(seed=seed), room_seed=seed, predict=predict)
    return rows, d


def test_choices_and_reward_are_identical_with_the_guess_on_or_off_and_material_on(tmp_path):
    for seed in range(3):
        on, d_on = show(tmp_path, True, seed)
        off, d_off = show(tmp_path, False, seed)
        assert rows_of(on, "prediction") and not rows_of(off, "prediction")
        assert core(on) == core(off) and same(d_on, d_off)  # includes the material rows: same topics, same facts
        mats = rows_of(on, "material")
        assert [m["bit"] for m in mats] == [0, 1, 2, 3] and len({m["topic"] for m in mats}) == 4


def test_a_log_with_material_rows_loads_and_older_readers_of_other_events_ignore_them(tmp_path):
    rows, _ = show(tmp_path, True, 0)
    sessions = load_sessions(tmp_path)
    assert sessions and any(r["event"] == "material" for r in sessions[0])
    assert set(material.used_topics(sessions)) == {material.normalise(r["topic"]) for r in rows if r["event"] == "material"}
    from app.cache import heard_texts, popular_bits
    from app.director import load_director
    assert heard_texts(sessions) and popular_bits(sessions) is not None
    load_director(sessions, seed=0)  # replays the career from reaction rows; a material row changes nothing


def mrow(**fields):
    base = dict(schema=2, session="s", at=1.0, event="material", bit=0, topic="t", **{"from": "everyday"}, facts=["a", "b"],
                sheet_id="character-v1", list_id="topics-v1", seed=3, recycled=False)
    return {**base, **fields}


def test_the_log_accepts_a_material_row_and_refuses_a_malformed_one():
    validate(mrow())
    for bad in (dict(topic=3), {"from": "news"}, dict(facts="ab"), dict(facts=[1]), dict(recycled="no"), dict(seed="x"), dict(seed=True)):
        with pytest.raises(ValueError):
            validate(mrow(**bad))
    with pytest.raises(ValueError):
        validate(mrow(schema=1))
    bare = mrow()
    del bare["seed"]
    with pytest.raises(ValueError):
        validate(bare)
