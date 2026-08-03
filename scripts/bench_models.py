"""Benchmark translator models on accuracy AND measured cost.

    make bench                       # default candidate set
    make bench MODELS=gpt-5-nano,claude-sonnet-5
    make bench CASES=8               # quick smoke run

For each model it replays the same eval cases (backend/tests/eval_cases.json)
through the real provider API, then reports:

  spec accuracy   — subset-exact-match on the fields the input determines
  gate accuracy   — clarify/unsupported cases classified correctly (must be 100%)
  $ / 1k runs     — from MEASURED tokens, not estimates
  p50 latency     — wall clock per call

Accuracy is the gate, not cost: a translator that silently reads "6-month" as
26 days is worthless at any price. Cost only breaks ties among models that
clear the accuracy bar.

Models without an API key configured are skipped and reported as skipped —
never silently dropped.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backend"))

from config import settings  # noqa: E402
from llm.evalkit import load_cases, subset_match  # noqa: E402
from llm.pricing import (  # noqa: E402
    BY_KEY, DEFAULT_BENCH, PRICES_FETCHED, PROVIDER_ENV,
)
from llm.providers import ProviderError, call_model  # noqa: E402
from llm.translate import system_prompt  # noqa: E402
from dsl.validator import validate_spec  # noqa: E402


def run_model(model_key: str, cases: list[dict], verbose: bool) -> dict:
    spec = BY_KEY[model_key]
    api_key = settings.key_for(spec.provider)
    if not api_key:
        return {"model": model_key, "skipped": f"no {PROVIDER_ENV[spec.provider]}"}

    system = system_prompt()
    spec_total = spec_ok = gate_total = gate_ok = 0
    errors: list[str] = []
    latencies: list[float] = []
    cost = 0.0
    tokens_in = tokens_out = tokens_cached = 0
    fallback_calls = 0
    failures: list[str] = []

    for case in cases:
        want = case["expect"]["status"]
        try:
            reply, usage = call_model(
                model_key, api_key, system,
                [{"role": "user", "content": case["input"]}],
            )
            latencies.append(usage.latency_s)
            tokens_in += usage.input_tokens
            tokens_out += usage.output_tokens
            tokens_cached += usage.cached_tokens
            cost += spec.cost(usage.input_tokens, usage.output_tokens,
                              usage.cached_tokens)
            if usage.path != "native":
                fallback_calls += 1
        except ProviderError as exc:
            errors.append(f"{case['id']}: {exc.code}")
            if want == "spec":
                spec_total += 1
            else:
                gate_total += 1
            failures.append(f"{case['id']} ({exc.code})")
            continue

        if want == "spec":
            spec_total += 1
            if reply.status != "spec" or reply.spec is None:
                failures.append(f"{case['id']} (returned {reply.status})")
                continue
            # score against the EXPANDED spec, exactly as the app would run it
            validated = validate_spec(reply.spec.model_dump(mode="json"))
            actual = validated.expanded or reply.spec.model_dump(mode="json")
            problems = subset_match(case["expect"]["spec_subset"], actual, "spec")
            if problems:
                failures.append(f"{case['id']}: {problems[0]}")
            else:
                spec_ok += 1
        else:
            gate_total += 1
            if reply.status == want:
                gate_ok += 1
            else:
                failures.append(f"{case['id']} (wanted {want}, got {reply.status})")

        if verbose:
            print(f"    {case['id']:<28} {reply.status}", flush=True)

    n = len(cases)
    return {
        "model": model_key,
        "provider": spec.provider,
        "spec_ok": spec_ok, "spec_total": spec_total,
        "gate_ok": gate_ok, "gate_total": gate_total,
        "spec_pct": spec_ok / spec_total if spec_total else None,
        "gate_pct": gate_ok / gate_total if gate_total else None,
        "cost_total": cost,
        "cost_per_1k": cost / n * 1000 if n else 0.0,
        "p50_latency": statistics.median(latencies) if latencies else None,
        "tokens_in": tokens_in, "tokens_out": tokens_out,
        "tokens_cached": tokens_cached,
        "fallback_calls": fallback_calls,
        "errors": errors,
        "failures": failures,
        "note": spec.note,
    }


def fmt(rows: list[dict]) -> str:
    head = (f"{'model':<24}{'prov':<11}{'spec':>10}{'gates':>8}"
            f"{'$/1k runs':>12}{'p50 s':>8}  notes")
    lines = [head, "-" * len(head)]
    for r in rows:
        if r.get("skipped"):
            lines.append(f"{r['model']:<24}{'—':<11}{'skipped':>10}"
                         f"{'':>8}{'':>12}{'':>8}  {r['skipped']}")
            continue
        spec_s = (f"{r['spec_pct']:.0%} ({r['spec_ok']}/{r['spec_total']})"
                  if r["spec_pct"] is not None else "—")
        gate_s = (f"{r['gate_pct']:.0%}" if r["gate_pct"] is not None else "—")
        notes = []
        if r["fallback_calls"]:
            notes.append(f"{r['fallback_calls']} json-fallback")
        if r["errors"]:
            notes.append(f"{len(r['errors'])} API errors")
        if r["note"]:
            notes.append(r["note"])
        lines.append(
            f"{r['model']:<24}{r['provider']:<11}{spec_s:>10}{gate_s:>8}"
            f"{r['cost_per_1k']:>11.2f}$"
            f"{(r['p50_latency'] or 0):>8.1f}  {'; '.join(notes)}"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(DEFAULT_BENCH))
    ap.add_argument("--cases", type=int, default=0, help="limit case count")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default=str(REPO / "data" / "bench"))
    args = ap.parse_args()

    keys = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [k for k in keys if k not in BY_KEY]
    if unknown:
        sys.exit(f"unknown model(s): {', '.join(unknown)}\n"
                 f"available: {', '.join(BY_KEY)}")

    cases = load_cases()
    if args.cases:
        # keep the mix: spec cases first, then the gate cases
        spec_cases = [c for c in cases if c["expect"]["status"] == "spec"]
        gate_cases = [c for c in cases if c["expect"]["status"] != "spec"]
        half = max(args.cases // 2, 1)
        cases = spec_cases[:half] + gate_cases[: args.cases - half]

    print(f"Translator bench · {len(cases)} cases · prices fetched {PRICES_FETCHED}")
    print("Accuracy is the gate; cost only breaks ties.\n")

    rows = []
    for key in keys:
        print(f"  running {key}…", flush=True)
        t0 = time.perf_counter()
        row = run_model(key, cases, args.verbose)
        row["wall_s"] = time.perf_counter() - t0
        rows.append(row)

    table = fmt(rows)
    print("\n" + table)

    graded = [r for r in rows if not r.get("skipped")]
    passing = [r for r in graded
               if (r["spec_pct"] or 0) >= 0.90 and (r["gate_pct"] or 0) == 1.0]
    print()
    if passing:
        best = min(passing, key=lambda r: r["cost_per_1k"])
        print(f"Cheapest model clearing the bars (>=90% spec, 100% gates): "
              f"{best['model']} at ${best['cost_per_1k']:.2f}/1k translations")
    elif graded:
        print("No model cleared >=90% spec accuracy AND 100% gate accuracy. "
              "Do not pick on price alone — raise effort, improve the prompt, "
              "or stay on the incumbent.")
    else:
        print("Nothing ran — set at least one provider API key in .env "
              "(ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY).")

    for r in graded:
        if r["failures"]:
            print(f"\n  {r['model']} misses:")
            for f in r["failures"][:8]:
                print(f"    - {f}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat()
    (out_dir / f"bench-{stamp}.json").write_text(json.dumps(rows, indent=1))
    (out_dir / f"bench-{stamp}.txt").write_text(table + "\n")
    print(f"\nWrote {out_dir}/bench-{stamp}.json")


if __name__ == "__main__":
    main()
