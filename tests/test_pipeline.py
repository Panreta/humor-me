"""Pipeline tests on synthetic transcripts only; never reads data/ (corpus is local and copyrighted)."""
import pytest

from pipeline.beats import CONTEXT_WORDS, MAX_BEAT_WORDS, build_beats, build_lines, labeled_view
from pipeline.markers import merge_adjacent, process, strip_songs

LEX = {
    "laughter": {"kind": "laugh", "source": "audience"},
    "audience laughs": {"kind": "laugh", "source": "audience"},
    "audience roars": {"kind": "laugh", "source": "audience", "intense": True},
    "applause": {"kind": "applause", "source": "audience"},
    "chuckles": {"kind": "chuckle", "source": "performer"},
}
FILLER = "Plain setup words here. " * 40  # keeps test markers out of the 2% entrance/exit window


def run(body: str):
    unknown: dict = {}
    clean, events = process("t", body, LEX, unknown)
    return clean, events, unknown


# --- S1/S2: offsets (regression: offsets drifted 1 char per removed marker) ---

def test_offsets_exact_after_many_markers():
    body = FILLER + "".join(f"Joke number {i} lands. [laughter] " for i in range(60)) + FILLER
    clean, events, _ = run(body)
    assert len(events) == 60
    for i, e in enumerate(events):
        assert clean[:e["offset"]].rstrip().endswith(f"Joke number {i} lands.")


def test_no_double_spaces_left_at_marker_seams():
    clean, _, _ = run(FILLER + "One. [laughter] Two. (audience laughs) Three." + FILLER)
    assert "  " not in clean
    assert "[" not in clean and "(" not in clean


def test_unknown_reaction_is_reported_not_guessed():
    _, events, unknown = run(FILLER + "Line. [crowd goes bananas laughing] Next." + FILLER)
    assert "crowd goes bananas laughing" in unknown
    assert not any(e.get("category") == "reaction" for e in events)


# --- S2: levels ---

def test_levels_laugh_intense_and_performer():
    # Reactions need ≥ 3 words between them, or they merge into one sustained reaction.
    _, events, _ = run(FILLER + "First joke lands here. [laughter] Second joke lands harder. "
                       "[audience roars] Third line gets a smirk. [chuckles] Moving on." + FILLER)
    by_raw = {e["raw"]: e for e in events}
    assert by_raw["laughter"]["level"] == 2
    assert by_raw["audience roars"]["level"] == 3
    assert by_raw["chuckles"]["level"] == 0  # performer sound, not the audience


def test_applause_alone_is_not_a_laugh():
    _, events, _ = run(FILLER + "I'm from Ohio! [applause] Anyway." + FILLER)
    (e,) = [e for e in events if e["raw"] == "applause"]
    assert e["level"] == 0 and "applause" in e["flags"]


def test_laugh_rolling_into_applause_is_level_4():
    _, events, _ = run(FILLER + "Big punch. [laughter] [applause] Next." + FILLER)
    audience = [e for e in events if e.get("source") == "audience"]
    assert len(audience) == 1 and audience[0]["level"] == 4


def test_entrance_applause_gets_no_flag():
    _, events, _ = run("[applause] Hello everyone. " + FILLER * 3)
    assert events[0]["level"] == 0 and events[0]["flags"] == []


# --- S1: songs ---

def test_songs_edge_dropped_mid_labeled_mixed_keeps_speech():
    body = "\n".join([
        "♪ walk on lyric one ♪",
        "♪ walk on lyric two ♪",
        "Hello, good to be here.",
        "♪ a sung line ♪",
        "♪ another sung line ♪",
        "Back to talking now about my week here.",
        "So I said ♪ la la ♪ to my neighbour over the fence today.",
        "Thank you, goodnight.",
        "♪ outro lyric ♪",
    ])
    out = strip_songs(body)
    assert "walk on" not in out and "outro" not in out
    assert out.count("[sings]") == 2  # one per mid-set run + one for the mixed paragraph
    assert "a sung line" not in out and "la la" not in out
    assert "to my neighbour over the fence today" in out
    assert "♪" not in out


# --- S3: beats ---

def labeled_set(n_jokes: int = 5):
    body = FILLER + "".join(
        f"Setup {i} goes here. Punchline number {i} lands hard! [laughter] " for i in range(n_jokes)
    ) + FILLER
    clean, events, _ = run(body)
    return clean, events


def test_beats_tile_clean_text_and_label_reactions():
    clean, events = labeled_set()
    beats = build_beats("t", clean, events, labeled=True)
    assert "".join(clean[s:e] for s, e in (b["span"] for b in beats)).split() == clean.split()
    reaction_beats = [b for b in beats if b["split"] == "reaction"]
    assert len(reaction_beats) == 5 and all(b["label"] == 2 for b in reaction_beats)
    assert all(b["label"] == 0 for b in beats if b["split"] != "reaction")


def test_long_quiet_stretch_is_chunked_at_sentence_ends():
    clean, events = labeled_set(1)
    beats = build_beats("t", clean, events, labeled=True)
    assert all(b["words"] <= MAX_BEAT_WORDS + 10 for b in beats)
    for b in beats:
        assert b["text"].endswith((".", "!", "?", "…")), b["text"][-20:]


def test_structure_only_labels_are_null():
    clean, events = labeled_set()
    beats = build_beats("t", clean, events, labeled=False)
    assert all(b["label"] is None for b in beats)
    assert all(line["label"] is None for line in build_lines(clean, beats))


# --- S3: lines ---

def test_line_labels_punchline_only():
    clean, events = labeled_set()
    lines = build_lines(clean, build_beats("t", clean, events, labeled=True))
    for i in range(5):
        (punch,) = [ln for ln in lines if ln["text"] == f"Punchline number {i} lands hard!"]
        (setup,) = [ln for ln in lines if ln["text"] == f"Setup {i} goes here."]
        assert punch["label"] == 2 and setup["label"] == 0


def test_short_final_line_folds_into_punchline():
    body = FILLER + "This is the actual punchline here. Yeah. [laughter] " + FILLER
    clean, events, _ = run(body)
    lines = build_lines(clean, build_beats("t", clean, events, labeled=True))
    (labeled,) = [ln for ln in lines if ln["label"] == 2]
    assert labeled["text"] == "This is the actual punchline here. Yeah."


def test_context_precedes_line_and_never_contains_it():
    clean, events = labeled_set()
    lines = build_lines(clean, build_beats("t", clean, events, labeled=True))
    for ln in lines:
        ctx = ln["context"].split()
        assert len(ctx) <= CONTEXT_WORDS
        # Context is exactly the words before the line's start, so the line (and its label) can't leak in.
        before = clean[:ln["span"][0]].split()
        assert ctx == before[len(before) - len(ctx):]


def test_labeled_view_tags_follow_punchlines():
    clean, events = labeled_set(2)
    beats = build_beats("t", clean, events, labeled=True)
    view = labeled_view(clean, events, beats)
    assert "lands hard! ⟨L2⟩" in view and view.count("⟨L2⟩") == 2


def test_merge_requires_adjacent_audience_reactions():
    clean = "w " * 50
    a = {"offset": 10, "source": "audience", "kind": "laugh", "level": 2, "flags": [], "raw": "a"}
    b = {"offset": 40, "source": "audience", "kind": "laugh", "level": 2, "flags": [], "raw": "b"}
    assert len(merge_adjacent([dict(a), dict(b)], clean)) == 2  # 15 words apart: separate laughs
    b_close = dict(b, offset=12)
    merged = merge_adjacent([dict(a), b_close], clean)
    assert len(merged) == 1 and merged[0]["level"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-q"])


# --- hand labels ---

def test_label_import_reads_marks_and_rejects_edits(tmp_path, monkeypatch):
    import yaml
    import pipeline.labels as L

    fake = [{"line_idx": i, "text": f"Line {i} text.", "label": 2 if i % 3 == 0 else 0} for i in range(6)]
    monkeypatch.setattr(L, "WORK", tmp_path / "work")
    monkeypatch.setattr(L, "LABELS", tmp_path)
    monkeypatch.setattr(L, "load_lines", lambda set_id: fake)

    L.template("s")
    with pytest.raises(SystemExit):
        L.template("s")  # never overwrites the user's work

    work = tmp_path / "work" / "s.txt"
    text = work.read_text(encoding="utf-8")
    text = text.replace("[0002]", ">>> second bit\n[0002]").replace("[0004]", ">>>\n[0004]")
    work.write_text(text, encoding="utf-8")
    L.import_labels("s")
    starts = yaml.safe_load((tmp_path / "s.yaml").read_text(encoding="utf-8"))["bit_starts"]
    assert starts == [{"line": 0}, {"line": 2, "note": "second bit"}, {"line": 4}]

    work.write_text(text.replace("[0003] Line 3 text.  ⟨L2⟩\n", ""), encoding="utf-8")
    with pytest.raises(SystemExit):
        L.import_labels("s")  # a deleted line is caught
