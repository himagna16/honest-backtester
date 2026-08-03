"""Live LLM translation eval harness (M3 acceptance, Gap #7).

Run with:  make eval   (requires ANTHROPIC_API_KEY in .env)

Acceptance bars:
- spec cases: ≥ 90% subset-exact-match on the expanded spec
- clarify + unsupported cases: 100% correct status

Skipped in the normal test run — live API calls are opt-in via
HONEST_RUN_LLM_EVALS=1 so `make test` stays offline and deterministic.
"""
import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.llm

CASES_PATH = Path(__file__).parent / "eval_cases.json"


def _subset_match(expected, actual, path="") -> list[str]:
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
                problems += _subset_match(val, actual[key], f"{path}.{key}")
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
                problems += _subset_match(e, a, f"{path}[{i}]")
    else:
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            if abs(float(expected) - float(actual)) > 1e-9:
                problems.append(f"{path}: {actual} != {expected}")
        elif expected != actual:
            problems.append(f"{path}: {actual!r} != {expected!r}")
    return problems


@pytest.mark.skipif(
    os.environ.get("HONEST_RUN_LLM_EVALS") != "1",
    reason="live LLM evals are opt-in: run `make eval`",
)
def test_translation_eval_suite():
    from config import settings
    if not settings.anthropic_api_key:
        pytest.skip("ANTHROPIC_API_KEY not configured")

    from llm.translate import translate

    print(f"\nEval model: {settings.llm_model} "
          f"(override with HONEST_LLM_MODEL to A/B a cheaper tier)")
    cases = json.loads(CASES_PATH.read_text())["cases"]
    spec_results: dict[str, list[str]] = {}
    gate_results: dict[str, str | None] = {}

    for case in cases:
        expect = case["expect"]
        try:
            result = translate(case["input"], use_cache=False)
        except Exception as exc:  # noqa: BLE001 — a crash is a failed case
            result = {"status": f"exception: {exc}"}

        if expect["status"] == "spec":
            if result.get("status") != "spec":
                spec_results[case["id"]] = [f"status={result.get('status')}"]
            else:
                spec_results[case["id"]] = _subset_match(
                    expect["spec_subset"], result["spec"], "spec"
                )
        else:
            ok = result.get("status") == expect["status"]
            gate_results[case["id"]] = None if ok else f"status={result.get('status')}"

    spec_pass = sum(1 for v in spec_results.values() if not v)
    gate_fail = {k: v for k, v in gate_results.items() if v}

    lines = [f"\nSpec cases: {spec_pass}/{len(spec_results)} exact-match"]
    for cid, problems in sorted(spec_results.items()):
        mark = "PASS" if not problems else "FAIL " + "; ".join(problems[:3])
        lines.append(f"  {cid}: {mark}")
    lines.append(
        f"Clarify/refuse cases: {len(gate_results) - len(gate_fail)}/{len(gate_results)}"
    )
    for cid, why in sorted(gate_fail.items()):
        lines.append(f"  {cid}: FAIL {why}")
    report = "\n".join(lines)
    print(report)

    assert spec_pass / max(len(spec_results), 1) >= 0.90, report
    assert not gate_fail, report
