"""rJokes prep on synthetic rows only; never reads data/external."""
from pipeline.rjokes import build, dedupe, flags, year_percentiles


def row(text, score, split="train", year=2015):
    return {"split": split, "text": text, "score": score, "year": year}


def test_repost_across_splits_lands_in_train_only():
    rows = dedupe([row("Same joke here ok", 2, "test"), row("same joke HERE ok", 7, "train")])
    assert len(rows) == 1
    assert rows[0]["split"] == "train"
    assert rows[0]["score"] == 7 and rows[0]["n_posts"] == 2 and rows[0]["score_spread"] == 5


def test_repost_keeps_earliest_year():
    rows = dedupe([row("a b c d", 1, year=2018), row("a b c d", 1, year=2012)])
    assert rows[0]["year"] == 2012


def test_percentile_is_within_year_not_global():
    rows = [row("x", 1, year=2010), row("y", 9, year=2019), row("z", 10, year=2019)]
    year_percentiles(rows)
    # the only 2010 joke sits mid-rank in its own year even though its raw score is lowest
    assert rows[0]["pct"] == 0.5
    assert rows[1]["pct"] < rows[2]["pct"]


def test_ties_share_a_percentile():
    rows = [row("a", 3), row("b", 3), row("c", 5)]
    year_percentiles(rows)
    assert rows[0]["pct"] == rows[1]["pct"]


def test_length_filter_and_flags():
    out, stats = build([row("too short", 3), row("why did the robot drink a beer at the bar", 4)])
    assert stats["after_length"] == 1
    assert out[0]["flags"] == ["alcohol_drugs"]


def test_flags_match_words_not_substrings():
    assert flags("the barn door") == []
    assert "alcohol_drugs" in flags("walks into a bar")
