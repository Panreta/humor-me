"""Every runnable module must at least import: a syntax error in an evaluation script once sat
on main unnoticed because no test touched it (dev-log 2026-10-28)."""
import importlib
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODULES = [f"{d}.{p.stem}" for d in ("sim", "pipeline", "scripts", "explorer")
           for p in (ROOT / d).glob("*.py") if p.stem != "__init__"]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    try:
        importlib.import_module(name)
    except ModuleNotFoundError as e:  # optional third-party deps (pymupdf, vllm, httpx...) aren't a code bug
        if e.name and e.name.split(".")[0] in ("sim", "pipeline", "scripts", "explorer"):
            raise
        pytest.skip(f"optional dependency missing: {e.name}")
