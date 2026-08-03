"""Shared eval scoring — used by both the pytest harness and `make bench`."""
from __future__ import annotations

import json
from pathlib import Path

CASES_PATH = Path(__file__).resolve().parent.parent / "tests" / "eval_cases.json"


def load_cases() -> list[dict]:
    return json.loads(CASES_PATH.read_text())["cases"]


def subset_match(expected, actual, path: str = "") -> list[str]:
    """Every key in `expected` must match `actual` exactly. Lists of plain
    strings compare as sets (ticker lists); other lists element-wise."""
    problems: list[str] = []
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{path}: expected object, got {type(actual).__name__}"]
        for key, val in expected.items():
            if key not in actual:
                problems.append(f"{path}.{key}: missing")
            else:
                problems += subset_match(val, actual[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list):
            return [f"{path}: expected list"]
        if expected and all(isinstance(x, str) for x in expected):
            if sorted(expected) != sorted(actual):
                problems.append(f"{path}: {sorted(actual)} != {sorted(expected)}")
        elif len(expected) != len(actual):
            problems.append(f"{path}: length {len(actual)} != {len(expected)}")
        else:
            for i, (e, a) in enumerate(zip(expected, actual)):
                problems += subset_match(e, a, f"{path}[{i}]")
    else:
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            if abs(float(expected) - float(actual)) > 1e-9:
                problems.append(f"{path}: {actual} != {expected}")
        elif expected != actual:
            problems.append(f"{path}: {actual!r} != {expected!r}")
    return problems
