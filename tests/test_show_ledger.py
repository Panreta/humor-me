"""Exp 32 (sim/show_ledger.py): setups, live vs sim never pooled, aborted and unreadable logs, rating arithmetic, the
committed bit files, and re-recording. Synthetic logs only."""
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))

from sim import cache, show_ledger as sl  # noqa: E402

REAL_PR_OF = sl.pr_of  # the git test must not use the stub below

JOKE = "ZEBRA-JOKE"       # a played line
BRIDGE = "ZEBRA-BRIDGE"   # a bridge line
SECRET = "ZEBRA-SECRET"   # text in fields that are not played lines


@pytest.fixture(autouse=True)
def no_git(monkeypatch):
    monkeypatch.setattr(sl, "pr_of", lambda sha, ref="main": {"sha": sha[:7], "pr": None, "branch": None})
    monkeypatch.setattr(sl, "main_head", lambda ref="main": "head")


def show(session, at, room="sim", gen="gen-v4:m", rating=None, status="finished", guess=False, skip_heard=None,
         cache_row=True, bits=2, bank_bit=None, lost=(), level=2, reason_text=SECRET, bridge_kinds=("filler",)):
    rows = [{"event": "show_start", "at": at, "room": room, "generator_id": gen, "git_sha": "abcdef1234",
             "context": {"venue": reason_text, "energy": "low", "age": "mixed"}}]
    if cache_row:
        rows.append({"event": "cache", "bits": [], **({} if skip_heard is None else {"skip_heard": skip_heard})})
    t = at
    for b in range(bits):
        rows.append({"event": "bit_decision", "bit": b, "action": "cynic"})
        if b and b - 1 < len(bridge_kinds):  # played in the gap: after the last line of the bit before, before this bit's first
            kind = bridge_kinds[b - 1]
            rows.append({"event": "bridge", "bit": b, "text": f"{BRIDGE} {b}", "audio_start_at": t - 2,
                         **({} if kind is None else {"kind": kind}), **({"source": {"session": "earlier", "bit": 0, "joke": 0}} if kind == "cached" else {})})
        for j in range(5):
            banked = b == bank_bit
            rows.append({"event": "candidates", "bit": b, "joke": j, "generator_id": "bank-v1" if banked else gen,
                         "items": [{"text": SECRET, "verdict": "pass", "reason": None}, {"text": SECRET, "verdict": "pass", "reason": None}],
                         "threads_lost": list(lost) if b == 0 else [], "prompt": SECRET})
            rows.append({"event": "joke_played", "bit": b, "joke": j, "chosen": 0, "text": f"{JOKE} {b}.{j}",
                         "audio_start_at": t + 1})
            rows.append({"event": "reaction", "bit": b, "joke": j, "source": "button" if room == "live" else "sim", "level": level, "key": None})
            t += 5
        rows.append({"event": "bit_end", "at": t, "bit": b, "reward": level / 3})
        if guess and b:
            rows.append({"event": "prediction", "bit": b, "hit": b % 2 == 1, "learned": True, "adopted": b % 2 == 1})
    if status:
        rows.append({"event": "show_end", "at": t, "status": status})
    if rating is not None:
        rows.append({"event": "rating", "value": rating})
    return rows


@pytest.fixture
def logs(tmp_path, monkeypatch):
    monkeypatch.setattr(sl, "read_session", lambda path: json.loads(path.read_text(encoding="utf-8")))
    return tmp_path


def put(directory, session, rows):
    (directory / f"{session}.jsonl").write_text(json.dumps(rows), encoding="utf-8")


def test_setups_group_by_writer_guess_and_bridge_in_start_order(logs):
    put(logs, "a", show("a", 100, gen="gen-v3:m", cache_row=False, rating=2))
    put(logs, "b", show("b", 200, gen="gen-v4:m", skip_heard=True, rating=3))
    put(logs, "c", show("c", 300, gen="gen-v4:m", skip_heard=True, guess=True, rating=4))
    put(logs, "d", show("d", 400, gen="gen-v4:m", skip_heard=True, guess=True, rating=2))
    res = sl.run(logs)
    assert [s["setup"] for s in res["setups"]] == ["gen-v3:m / guess off / bridges no-cache", "gen-v4:m / guess off / bridges skip-heard",
                                                   "gen-v4:m / guess on / bridges skip-heard"]
    assert [s["shows"] for s in res["setups"]] == [1, 1, 2]
    assert res["setups"][2]["change"] == "D-115 next bit prepared early" and res["setups"][1]["change"] == "D-110 one call per thread + D-112 skips heard lines"


def test_bridge_rule_reads_the_cache_row():
    assert sl.bridge_rule([{"event": "show_start"}]) == "no-cache"
    assert sl.bridge_rule([{"event": "cache"}]) == "repeat-ok"
    assert sl.bridge_rule([{"event": "cache", "skip_heard": False}]) == "repeat-ok"
    assert sl.bridge_rule([{"event": "cache", "skip_heard": True}]) == "skip-heard"


def test_live_and_simulated_audience_numbers_are_never_pooled(logs):
    put(logs, "live", show("live", 100, room="live", level=3, rating=3))
    put(logs, "sim", show("sim", 200, room="sim", level=0, rating=3))
    res = sl.run(logs)
    (setup,) = res["setups"]
    assert setup["audience"]["live"]["reaction_mean"] == 3 and setup["audience"]["live"]["shows"] == 1
    assert setup["audience"]["simulated audience"]["reaction_mean"] == 0
    assert "reaction_mean" not in setup and "reward_mean" not in setup  # no pooled number exists to misread
    kinds = {s["session"]: s["audience"]["kind"] for s in res["shows"]}
    assert kinds == {"live": "live", "sim": "simulated audience"}


def test_aborted_show_is_a_row_but_counts_nowhere_else(logs):
    put(logs, "ok", show("ok", 100, rating=4))
    put(logs, "cut", show("cut", 200, status="aborted", bits=1, rating=1))
    res = sl.run(logs)
    cut = next(s for s in res["shows"] if s["session"] == "cut")
    assert cut["status"] == "aborted" and cut["bits_played"] == 1
    (setup,) = res["setups"]
    assert (setup["finished"], setup["aborted"], setup["rating"]["n"], setup["rating"]["mean"]) == (1, 1, 1, 4)


def test_unreadable_logs_are_listed_with_the_error_class(logs, monkeypatch):
    put(logs, "ok", show("ok", 100))
    (logs / "bad.jsonl").write_text("{", encoding="utf-8")
    res = sl.run(logs)
    assert [u["session"] for u in res["unreadable"]] == ["bad"] and res["unreadable"][0]["reason"] == "JSONDecodeError"
    assert [s["session"] for s in res["shows"]] == ["ok"] and res["n_files"] == 2


def test_rating_stats_and_exact_permutation_test(logs):
    assert sl.permutation_p([1, 1], [5, 5]) == pytest.approx(2 / 6)  # 6 splits, the 2 extreme ones tie the observed gap
    assert sl.permutation_p([2, 3], [2, 3]) == 1.0
    put(logs, "a1", show("a1", 100, gen="gen-v3:m", rating=2))
    put(logs, "a2", show("a2", 110, gen="gen-v3:m", rating=3))
    put(logs, "b1", show("b1", 200, gen="gen-v5:m", rating=4))
    put(logs, "b2", show("b2", 210, gen="gen-v5:m", rating=5))
    res = sl.run(logs)
    a, b = res["setups"]
    assert (a["rating"]["mean"], a["rating"]["min"], a["rating"]["max"]) == (2.5, 2, 3)
    (t,) = res["adjacent"]
    assert t["n"] == [2, 2] and t["diff_of_means"] == 2.0 and t["p"] == pytest.approx(2 / 6) and t["within_noise"] is None  # 2 v 2 cannot reach 0.05, so no claim


def test_a_setup_with_one_rating_is_not_tested(logs):
    put(logs, "a", show("a", 100, gen="gen-v3:m", rating=2))
    put(logs, "b", show("b", 200, gen="gen-v5:m", rating=4))
    (t,) = sl.run(logs)["adjacent"]
    assert t["p"] is None and t["within_noise"] is None


def test_delivery_signals_bank_lost_threads_and_guesses(logs):
    put(logs, "a", show("a", 100, bank_bit=1, guess=True,
                        lost=["LLMError: transport: ReadTimeout", "GenerationError: expected a premise and 5 lines", "re-ask failed: X"]))
    (s,) = sl.run(logs)["shows"]
    d = s["delivery"]
    assert d["bank_share"] == 0.5 and d["threads_lost"] == 3 and d["bits_with_lost_threads"] == 1
    assert d["lost_by_class"] == {"timeout": 1, "bad reply shape": 1, "re-ask failed": 1}
    assert (d["guesses"], d["guess_hits"], d["bridge_lines"]) == (1, 1, 1)


def test_the_record_holds_ids_and_hashes_only(logs):
    put(logs, "a", show("a", 100, rating=3))
    text = json.dumps(sl.run(logs))
    for needle in (JOKE, BRIDGE, SECRET):
        assert needle not in text
    (s,) = json.loads(text)["shows"]
    assert s["bits"][0]["hash"] == sl.bit_hash([f"{JOKE} 0.{j}" for j in range(5)]) and len(s["bits"][0]["hash"]) == 16


def test_export_reads_the_show_top_to_bottom_in_play_order_with_labels(logs, tmp_path):
    put(logs, "live1", show("live1", 100, room="live", bank_bit=0, level=3, rating=4, bridge_kinds=[None], reason_text=SECRET))
    put(logs, "sim1", show("sim1", 200, room="sim", level=1, bridge_kinds=["cached"]))
    out = tmp_path / "bits"
    files = sl.export(logs, out)
    assert files[-1] == out / "index.md" and len(files) == 3
    live = next(f for f in files if f.name == "live1.md").read_text(encoding="utf-8")
    sim = next(f for f in files if f.name == "sim1.md").read_text(encoding="utf-8")
    # play order: bit 1's lines, then the gap and its bridge line, then bit 2
    assert live.index("## Bit 1") < live.index(f"{JOKE} 0.0") < live.index(f"{JOKE} 0.4") < live.index("Gap before this bit") \
        < live.index(f"{BRIDGE} 1") < live.index("## Bit 2") < live.index(f"{JOKE} 1.0")
    assert "1. " + f"{JOKE} 0.0" in live and "*(from the bank)*" in live and live.count("from the bank") == 5
    assert f"{JOKE} 1.0 — big laugh" in live
    assert "*(filler while the next bit is written)* " + f"{BRIDGE} 1" in live  # a row with no kind is filler
    assert "*(replayed line from show earlier)* " + f"{BRIDGE} 1" in sim
    assert "(simulated audience: small laugh)" in sim and "simulated audience" not in live
    assert "Your rating: 4" in live and "Your rating: none" in sim and "Room: live" in live and "Room: simulated" in sim
    assert "## Bit 1 — Cynic" in live
    index = (out / "index.md").read_text(encoding="utf-8")
    assert "| Room | How it ended | Your rating | Show |" in index and "Date (local, UTC" in index
    assert index.index("sim1") < index.index("live1")  # newest first
    assert "Setup: m writer, one call per thread, 3 in parallel (gen-v4); next bit prepared after; may replay heard lines" in live
    assert index.index("## Code names") < index.index("| bank-v1 | hand-written bank only | D-66 | #40 |") < index.index("## m writer")
    assert "## m writer, one call per thread, 3 in parallel (gen-v4); next bit prepared after" in index  # plain words, then the code name
    for text in (live, sim, index):
        assert SECRET not in text  # context fields, candidates, prompts and rejected items are never read


def test_opening_and_trailing_bridge_lines_are_placed(logs, tmp_path):
    rows = show("a", 100)
    rows.append({"event": "bridge", "bit": 0, "text": "ZEBRA-OPEN", "audio_start_at": 99.5, "kind": "filler"})
    rows.append({"event": "bridge", "bit": 1, "text": "ZEBRA-TAIL", "audio_start_at": 10_000.0, "kind": "filler"})
    put(logs, "a", rows)
    text = next(f for f in sl.export(logs, tmp_path / "b") if f.name == "a.md").read_text(encoding="utf-8")
    assert text.index("Before the first bit") < text.index("ZEBRA-OPEN") < text.index("## Bit 1")
    assert text.index(f"{JOKE} 1.4") < text.index("After the last bit") < text.index("ZEBRA-TAIL")


def test_gap_notes_match_the_ledger_numbers(logs, tmp_path):
    put(logs, "a", show("a", 100, bits=3, bridge_kinds=["filler", "filler"]))
    res = sl.run(logs)
    (s,) = res["shows"]
    text = next(f for f in sl.export(logs, tmp_path / "b") if f.name == "a.md").read_text(encoding="utf-8")
    notes = [ln for ln in text.splitlines() if ln.startswith("*Gap before")]
    assert notes == ["*Gap before this bit: waited 1 s, 1 filler line.*"] * 2
    assert s["delivery"]["total_wait_s"] == 2 and s["delivery"]["bridge_lines"] == 2 and s["delivery"]["gaps"] == 2


def test_gap_note_wording():
    assert sl.gap_note(0.0, []) == "no wait"
    assert sl.gap_note(14.2, ["filler", "filler"]) == "waited 14 s, 2 filler lines"
    assert sl.gap_note(5.0, []) == "waited 5 s, no bridge lines"
    assert sl.gap_note(5.0, ["cached", "cached"]) == "waited 5 s, 2 bridge lines, all replayed"
    assert sl.gap_note(5.0, ["cached"]) == "waited 5 s, 1 bridge line, replayed"
    assert sl.gap_note(5.0, ["cached", "filler", "filler"]) == "waited 5 s, 3 bridge lines (2 filler, 1 replayed)"
    assert sl.gap_note(None, ["filler"]).startswith("wait not measured")
    assert sl.bridge_note({"text": "x"}) == "filler while the next bit is written"
    assert sl.bridge_note({"kind": "cached", "source": {"session": "abc"}}) == "replayed line from show abc"


def test_export_holds_only_played_text_fields(logs, tmp_path):
    rows = show("a", 100)
    for r in rows:
        if r["event"] == "show_start":
            r["operator"] = SECRET  # any field that is not a played line must stay out
            r["audio"] = {"consent": True, "path": SECRET}
        if r["event"] == "reaction":
            r["key"] = SECRET
    put(logs, "a", rows)
    files = sl.export(logs, tmp_path / "b")
    assert all(SECRET not in f.read_text(encoding="utf-8") for f in files)


def test_export_skips_unreadable_logs(logs, tmp_path):
    put(logs, "ok", show("ok", 100))
    (logs / "bad.jsonl").write_text("{", encoding="utf-8")
    assert [f.name for f in sl.export(logs, tmp_path / "b")] == ["ok.md", "index.md"]


def test_re_record_picks_up_a_new_show(logs, tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "METRICS", tmp_path / "metrics")
    put(logs, "a", show("a", 100, rating=3))
    first = sl.record(logs)
    assert [s["session"] for s in first["shows"]] == ["a"]
    assert sl.record(logs) == first  # nothing changed: loaded, not recomputed
    put(logs, "b", show("b", 200, rating=1))
    second = sl.record(logs)
    assert [s["session"] for s in second["shows"]] == ["a", "b"]
    assert sl.stored() == second


def test_pr_label_falls_back_to_the_sha():
    assert sl.pr_label({"pr": {"pr": None, "branch": None}, "git_sha": "abcdef1"}) == "abcdef1"
    assert sl.pr_label({"pr": {"pr": 80, "branch": "feat/x"}, "git_sha": "abcdef1"}) == "#80 feat/x"


def test_setup_words_give_plain_words_then_the_code_name():
    base = {"prompt": "gen-v5", "model": "bonsai-2:4b", "prediction": True, "bridge": "skip-heard"}
    assert sl.setup_words(base) == ("local (bonsai) writer, connected-bit prompt, each line builds on the last (gen-v5); "
                                    "next bit prepared early; skips heard lines")
    assert sl.setup_words({**base, "prompt": "bank-v1", "model": "", "prediction": False, "bridge": "no-cache"}) ==         "hand-written bank only (bank-v1); next bit prepared after; no cached lines"
    assert sl.setup_words({**base, "prompt": "gen-v3", "model": "gemini-flash-lite-latest"}).startswith("Gemini writer, 3 threads in one reply (gen-v3)")


def test_chart_labels_are_code_names_only():
    rec = {"prompt": "gen-v4", "model": "bonsai-2:4b", "prediction": True, "bridge": "skip-heard"}
    assert sl.code_label(rec) == "gen-v4 + skip-heard + guess"
    assert sl.code_label({**rec, "prompt": "gen-v3", "model": "gemini-flash-lite-latest", "prediction": False, "bridge": "no-cache"}) == "gen-v3 Gemini"
    assert sl.code_label({**rec, "prompt": "bank-v1", "model": "", "prediction": False, "bridge": "no-cache"}) == "bank-v1"


def test_gap_note_counts_replayed_lines_apart_from_filler(logs, tmp_path):
    put(logs, "a", show("a", 100, bits=2, bridge_kinds=["cached"]))
    text = next(f for f in sl.export(logs, tmp_path / "b") if f.name == "a.md").read_text(encoding="utf-8")
    assert "*Gap before this bit: waited 1 s, 1 bridge line, replayed.*" in text and "filler" not in text.split("Gap before")[1].split("\n")[0]


def test_a_rating_test_that_cannot_reach_005_is_not_called_noise(logs):
    assert sl.smallest_p(2, 2) == pytest.approx(2 / 6) and sl.smallest_p(3, 4) == pytest.approx(2 / 35) and sl.smallest_p(5, 5) < 0.05
    put(logs, "a1", show("a1", 100, gen="gen-v3:m", rating=1))
    put(logs, "a2", show("a2", 110, gen="gen-v3:m", rating=1))
    put(logs, "b1", show("b1", 200, gen="gen-v5:m", rating=5))
    put(logs, "b2", show("b2", 210, gen="gen-v5:m", rating=5))
    (t,) = sl.run(logs)["adjacent"]
    assert t["p"] == pytest.approx(2 / 6) and t["p_min"] == pytest.approx(2 / 6) and t["within_noise"] is None  # cannot reject: no claim either way
    assert "too few ratings to tell" in sl.table(sl.run(logs)) and "could be a reshuffle" not in sl.table(sl.run(logs))


def test_a_test_that_can_reject_says_which_way(logs):
    for i, (gen, r) in enumerate([("gen-v3:m", 1)] * 5 + [("gen-v5:m", 5)] * 5):
        put(logs, f"s{i}", show(f"s{i}", 100 + i, gen=gen, rating=r))
    (t,) = sl.run(logs)["adjacent"]
    assert t["p_min"] < sl.ALPHA and t["p"] == t["p_min"] and t["within_noise"] is False


def test_pr_of_credits_a_merge_commit_to_itself_and_a_branch_commit_to_its_merge(tmp_path):
    import subprocess

    def git(*a):
        return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", *a], cwd=tmp_path,
                              capture_output=True, text=True, check=True).stdout.strip()
    git("init", "-q", "-b", "main")
    (tmp_path / "f").write_text("0")
    git("add", "f"); git("commit", "-q", "-m", "root")
    shas = {}
    for n in (1, 2):
        git("checkout", "-q", "-b", f"feat/x{n}")
        (tmp_path / "f").write_text(str(n))
        git("commit", "-qam", f"work {n}")
        shas[f"branch{n}"] = git("rev-parse", "HEAD")
        git("checkout", "-q", "main")
        git("merge", "-q", "--no-ff", f"feat/x{n}", "-m", f"Merge pull request #{n}0 from someone/feat/x{n}")
        shas[f"merge{n}"] = git("rev-parse", "HEAD")
    assert REAL_PR_OF(shas["merge1"], cwd=tmp_path)["pr"] == 10  # not the next merge, #20
    assert REAL_PR_OF(shas["merge2"], cwd=tmp_path) == {"sha": shas["merge2"][:7], "pr": 20, "branch": "feat/x2"}
    assert REAL_PR_OF(shas["branch2"], cwd=tmp_path)["pr"] == 20
    assert REAL_PR_OF(shas["branch1"], cwd=tmp_path)["pr"] == 10
    assert REAL_PR_OF("deadbeef", cwd=tmp_path)["pr"] is None


def test_gen_v4_before_the_reply_fixes_is_its_own_setup(logs, monkeypatch):
    prs = {"aaaaaaa": 78, "bbbbbbb": 79}
    monkeypatch.setattr(sl, "pr_of", lambda sha, ref="main": {"sha": sha[:7], "pr": prs.get(sha[:7]), "branch": "b"})
    for sid, sha in (("old", "aaaaaaa"), ("new", "bbbbbbb")):
        rows = show(sid, 100 if sid == "old" else 200, rating=2)
        rows[0]["git_sha"] = sha + "1234"
        put(logs, sid, rows)
    res = sl.run(logs)
    assert [s["code_label"] for s in res["setups"]] == ["gen-v4 before D-111 fixes", "gen-v4"]
    assert res["setups"][1]["change"].startswith("D-111 reply-format fixes")
    assert "(gen-v4 before D-111 fixes)" in res["setups"][0]["label"] and "before the reply-format fixes" in res["setups"][0]["label"]


def test_a_show_cut_short_on_guess_code_is_filed_with_the_guess(logs, monkeypatch):
    monkeypatch.setattr(sl, "pr_of", lambda sha, ref="main": {"sha": sha[:7], "pr": 83, "branch": "early-prep"})
    put(logs, "cut", show("cut", 100, status="aborted", bits=1))
    put(logs, "full", show("full", 200, guess=True, rating=3))
    res = sl.run(logs)
    cut = next(s for s in res["shows"] if s["session"] == "cut")
    assert cut["prediction"] is True and cut["prediction_inferred"] and len(res["setups"]) == 1


def test_export_removes_stale_files(logs, tmp_path):
    out = tmp_path / "bits"
    (out / "old-setup").mkdir(parents=True)
    (out / "old-setup" / "gone.md").write_text("stale", encoding="utf-8")
    put(logs, "ok", show("ok", 100))
    files = sl.export(logs, out)
    assert not (out / "old-setup").exists() and {f.name for f in files} == {"ok.md", "index.md"}


def test_dates_are_local_with_their_offset(logs):
    put(logs, "a", show("a", 1_700_000_000))
    (s,) = sl.run(logs)["shows"]
    assert len(s["started"]) == 16 and s["utc_offset"][0] in "+-" and s["utc_offset"][3] == ":"
