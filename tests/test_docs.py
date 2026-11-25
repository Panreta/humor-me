"""Checks on the markdown docs themselves."""
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent
LIGHT = '%%{init: {"theme": "default"}}%%'


def tracked_markdown():
    out = subprocess.run(["git", "ls-files", "*.md"], cwd=ROOT, capture_output=True, text=True, check=True)
    return [ROOT / p for p in out.stdout.split()]


def test_mermaid_diagrams_force_light_theme():
    # GitHub renders Mermaid dark in dark mode, which washes out the pastel component colours.
    missing = []
    for md in tracked_markdown():
        lines = md.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if line.strip() == "```mermaid" and (i + 1 >= len(lines) or lines[i + 1].strip() != LIGHT):
                missing.append(f"{md.relative_to(ROOT)}:{i + 1}")
    assert not missing, f"Mermaid blocks without {LIGHT}: {missing}"


def test_decision_logs_share_ids():
    # decisions.md is the plain log for people, decisions-detail.md the full one for agents; IDs are the join.
    import re
    detail = (ROOT / "docs/agent/decisions-detail.md").read_text(encoding="utf-8")
    plain = (ROOT / "docs/decisions.md").read_text(encoding="utf-8")
    detail_ids = re.findall(r'^## <a id="d-(\d+)"></a>D-(\d+) · ', detail, re.M)
    assert all(a == b for a, b in detail_ids), "anchor and heading IDs differ"
    ids = [int(a) for a, _ in detail_ids]
    assert ids == list(range(1, len(ids) + 1)), "detail IDs must run D-01, D-02, ... in file order"
    assert len(re.findall(r"^## ", detail, re.M)) == len(ids), "a detail heading has no D-xx ID"
    # an entry is a line that starts with its ID (table row, list item or heading), so the format stays the reader's choice
    plain_ids = [int(i) for i in re.findall(r"^(?:\||[-*]|#+)\s*\[?D-(\d+)\b", plain, re.M)]
    assert sorted(plain_ids) == ids, (
        f"only in detail: {sorted(set(ids) - set(plain_ids))}, only in plain: {sorted(set(plain_ids) - set(ids))}, "
        f"duplicates in plain: {sorted({i for i in plain_ids if plain_ids.count(i) > 1})}")
