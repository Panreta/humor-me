"""Which rJokes features move with upvotes, and did Exp 10 adjust for the right ones?

Usage (laptop, no GPU):
    python explorer/rjokes_correlations.py [--boots 200]

Design and rule were fixed before running (decisions.md 2026-11-01). Spearman rho, because upvotes are
heavily skewed. Each feature is measured raw and within the same year and length group: a feature that
only looks linked to upvotes through length or year fades in the second number.
"""
from __future__ import annotations  # cluster venv is Python 3.9: keeps `X | None` hints unevaluated

import argparse
import json
import pathlib

import numpy as np

from explorer.mode_ranking import JOKES_PATH, MODES, MODES_PATH, length_bins, midranks

FLAGS = ("protected", "body", "sexual", "alcohol_drugs", "death_violence")
SAFETY = ("targets_group", "about_bodies", "sexual", "alcohol_drugs", "slur")
TEXT_FEATURES = ["n_words", "year", "n_posts", "score_spread"] + [f"flag_{f}" for f in FLAGS]
LABEL_FEATURES = [f"mode_{m}" for m in MODES] + [f"safety_{s}" for s in SAFETY]
THRESHOLD = 0.05
FIG = pathlib.Path("reports/figures/rjokes_correlations.png")


def load(modes: str = MODES_PATH, jokes: str = JOKES_PATH) -> tuple[np.ndarray, np.ndarray]:
    """(all jokes with a year, labeled subset), columns = pct + features. Same population as Exp 10."""
    labels = {}
    for line in pathlib.Path(modes).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("mode") in MODES:
            labels[r["id"]] = ([float(r["mode"] == m) for m in MODES]
                               + [float(r["safety"][s]) for s in SAFETY])
    every, labeled = [], []
    with open(jokes, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["year"] is None:
                continue
            row = [r["pct"], r["n_words"], r["year"], r["n_posts"], r["score_spread"]]
            row += [float(k in r["flags"]) for k in FLAGS]
            every.append(row)
            if r["id"] in labels:
                labeled.append(row + labels[r["id"]])
    return np.array(every, dtype=float), np.array(labeled, dtype=float)


def rank_columns(X: np.ndarray) -> np.ndarray:
    one_group = np.zeros(len(X), dtype=int)
    return np.column_stack([midranks(one_group, X[:, j]) for j in range(X.shape[1])])


def within_cells(R: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Subtract each cell's mean, leaving only differences between jokes in the same cell."""
    n = np.bincount(cell)
    out = R.copy()
    for j in range(R.shape[1]):
        out[:, j] -= (np.bincount(cell, weights=R[:, j]) / n)[cell]
    return out


def corr_with_first(R: np.ndarray) -> np.ndarray:
    c = R - R.mean(axis=0)
    norm = np.sqrt((c ** 2).sum(axis=0))
    norm[norm < 1e-9] = np.nan  # no variation left, e.g. year within a year: rho doesn't exist
    return (c.T @ c[:, 0]) / (norm * norm[0])


def corr_matrix(R: np.ndarray) -> np.ndarray:
    c = R - R.mean(axis=0)
    norm = np.sqrt((c ** 2).sum(axis=0))
    norm[norm < 1e-9] = np.nan
    return (c.T @ c) / np.outer(norm, norm)


def cells(X: np.ndarray, names: list[str], k: int = 10) -> np.ndarray:
    year, words = X[:, names.index("year")], X[:, names.index("n_words")]
    return np.unique(year * 100 + length_bins(words, k), return_inverse=True)[1].ravel()


def nullable(a: np.ndarray) -> list:
    """NaN isn't valid JSON, so a rho that doesn't exist is stored as null."""
    return [[None if np.isnan(x) else float(x) for x in row] if isinstance(row, np.ndarray)
            else (None if np.isnan(row) else float(row)) for row in a]


def versus_pct(X: np.ndarray, names: list[str], boots: int, seed: int) -> dict:
    cell = cells(X, names)
    def both(idx):
        R = rank_columns(X[idx])
        return corr_with_first(R)[1:], corr_with_first(within_cells(R, cell[idx]))[1:]
    raw, adj = both(np.arange(len(X)))
    rng = np.random.default_rng(seed)
    bs = [both(rng.integers(0, len(X), len(X))) for _ in range(boots)]
    def ci(vals):
        return nullable(np.array(np.percentile(np.array(vals), [2.5, 97.5], axis=0)))
    return {"n": len(X), "names": names, "raw": nullable(raw), "raw_ci": ci([b[0] for b in bs]),
            "adj": nullable(adj), "adj_ci": ci([b[1] for b in bs])}


def apply_rule(names: list[str], adj: list[list], threshold: float = THRESHOLD) -> dict:
    """A confound needs both links: to upvotes and to some mode, within the same year and length group."""
    modes = [j for j, n in enumerate(names) if n.startswith("mode_")]
    pct = names.index("pct")
    hits = []
    for j, n in enumerate(names):
        if n.startswith(("flag_", "safety_")):
            to_mode = max(abs(adj[j][m]) for m in modes)
            if abs(adj[j][pct]) >= threshold and to_mode >= threshold:
                hits.append(n)
    return {"threshold": threshold, "rerun_for": [h for h in hits if h.startswith("flag_")],
            "look_closer": [h for h in hits if h.startswith("safety_")]}


def run(modes: str = MODES_PATH, jokes: str = JOKES_PATH, boots: int = 200, seed: int = 0) -> dict:
    every, labeled = load(modes, jokes)
    names = ["pct"] + TEXT_FEATURES + LABEL_FEATURES
    ranks = rank_columns(labeled)
    raw = corr_matrix(ranks)
    adj = corr_matrix(within_cells(ranks, cells(labeled, names)))
    adj_list = nullable(adj)
    return {"all_jokes": versus_pct(every, ["pct"] + TEXT_FEATURES, boots, seed),
            "labeled": versus_pct(labeled, names, boots, seed),
            "names": names, "matrix_raw": nullable(raw), "matrix_adj": adj_list,
            "rule": apply_rule(names, adj_list)}


def table(block: dict) -> str:
    def cell(v, ci, j):
        if v[j] is None or ci[0][j] is None:
            return "n/a"
        return f"{v[j]:+.3f} [{ci[0][j]:+.3f}, {ci[1][j]:+.3f}]"
    out = ["| Feature | ρ with `pct`, raw [95% CI] | Same year and length group [95% CI] |", "|---|---|---|"]
    for j, n in enumerate(block["names"][1:]):
        out.append(f"| {n} | {cell(block['raw'], block['raw_ci'], j)} | {cell(block['adj'], block['adj_ci'], j)} |")
    return "\n".join(out)


def plot(res: dict):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("rho", ["#2a78d6", "#f0efec", "#c8553d"])  # gray at rho = 0
    names = res["names"]
    fig, axes = plt.subplots(1, 2, figsize=(17, 8.5), facecolor="#fcfcfb", layout="constrained")
    last = None
    for ax, key, title in ((axes[0], "matrix_raw", "Raw"), (axes[1], "matrix_adj", "Same year and length group")):
        m = np.array([[np.nan if x is None else x for x in row] for row in res[key]])
        np.fill_diagonal(m, np.nan)
        last = ax.imshow(m, cmap=cmap, vmin=-0.5, vmax=0.5)
        ax.set_xticks(range(len(names)), names, rotation=90, fontsize=8, color="#52514e")
        ax.set_yticks(range(len(names)), names, fontsize=8, color="#52514e")
        ax.tick_params(length=0)
        ax.set_frame_on(False)
        ax.set_title(title, fontsize=12, color="#0b0b0b")
        for i, row in enumerate(m):
            for j, v in enumerate(row):
                if not np.isnan(v) and abs(v) >= THRESHOLD:  # only cells the rule could care about
                    ax.text(j, i, f"{v:.2f}".replace("0.", "."), ha="center", va="center", fontsize=6, color="#0b0b0b")
    fig.colorbar(last, ax=axes, shrink=0.6, label="Spearman ρ (numbers shown where |ρ| ≥ 0.05)")
    return fig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boots", type=int, default=200)
    a = ap.parse_args()
    res = run(boots=a.boots)
    print(table(res["all_jokes"]), "\n\n", table(res["labeled"]), "\n\n", res["rule"])
    import matplotlib
    matplotlib.use("Agg")  # CLI only; the notebook renders inline
    plot(res).savefig(FIG, dpi=150, facecolor="#fcfcfb")


if __name__ == "__main__":
    main()
