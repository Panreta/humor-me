"""Cache experiment results by name + parameters, so the decision notebook opens instantly and
only recomputes when a parameter changes (or rerun=True).

Files: reports/metrics/<name>.json = {"params": {...}, "results": {...}}. Committed, so the
numbers behind every decision are in git next to the code that produced them.
"""
import json
import pathlib
from typing import Callable

METRICS = pathlib.Path(__file__).resolve().parent.parent / "reports" / "metrics"


def save(name: str, params: dict, results: dict) -> pathlib.Path:
    path = METRICS / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    # One top-level key per line keeps diffs readable; nested curves stay on their line.
    body = ",\n".join(f"  {json.dumps(k)}: {json.dumps(v)}" for k, v in results.items())
    path.write_text(f'{{"params": {json.dumps(params, sort_keys=True)}, "results": {{\n{body}\n}}}}\n')
    return path


def load(name: str, params: dict) -> dict | None:
    path = METRICS / f"{name}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return data["results"] if data.get("params") == json.loads(json.dumps(params)) else None


def run_or_load(name: str, fn: Callable[..., dict], rerun: bool = False, **params) -> dict:
    """Return cached results for exactly these params, else run fn(**params), save, return."""
    if not rerun:
        cached = load(name, params)
        if cached is not None:
            return cached
    results = fn(**params)
    save(name, params, results)
    return results
