"""Restart + run all for a notebook, in place, with clean metadata. Run before committing it.

Usage:
    .venv/Scripts/python scripts/run_notebook.py [notebooks/decisions.ipynb] [--timeout 1200]

Runs with the interpreter that launches this script (the "python3" kernel resolves to it), so use
the project venv. Outputs are kept (GitHub renders them); timing metadata isn't recorded, so
re-running without changes produces no diff beyond the outputs themselves.
"""
import argparse
import pathlib

import nbformat
from nbclient import NotebookClient


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="notebooks/decisions.ipynb")
    ap.add_argument("--timeout", type=int, default=1200)
    a = ap.parse_args()
    path = pathlib.Path(a.path)
    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=a.timeout, kernel_name="python3", record_timing=False,
                   resources={"metadata": {"path": str(path.parent)}}).execute()
    for cell in nb.cells:
        cell.metadata.pop("execution", None)
    nb.metadata.pop("widgets", None)
    nb.metadata["kernelspec"] = {"name": "humor-me", "display_name": "humor-me (.venv)", "language": "python"}
    nbformat.write(nb, path)
    failed = [c for c in nb.cells if c.cell_type == "code" and any("❌" in o.get("data", {}).get("text/markdown", "")
                                                                for o in c.get("outputs", []))]
    print(f"executed {path}; {len(failed)} cell(s) with a failing claim check")


if __name__ == "__main__":
    main()
