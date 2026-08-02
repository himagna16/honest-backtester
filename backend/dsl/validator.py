"""Semantic validation beyond the schema (Section 2) — deterministic, no LLM.

This is the security boundary: nothing reaches the engine except a spec that
passed HERE. Errors are plain English; warnings surface anything the audit
will later flag so there are no surprises after the run.

Also expands defaults (Gap #8): the returned `expanded` dict is the spec with
every default filled in — that exact document is what gets stored on runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import pydantic

from dsl.schema import SPEC_VERSION, StrategySpec
from dsl.symbols import check_tickers
from dsl.universes import preset_tickers, read_watchlist_file

REQUIRED_SWEEP = {0.0, 5.0, 20.0}
DEFAULT_WINDOW_YEARS = 10
V0_UNSUPPORTED_FIELDS = {
    "stop_loss_pct": (
        "stop-losses were cut from V0: honest stop fills need intraday data "
        "and gap-through rules that daily closes can't provide. Supported "
        "exits: hold_days and rebalancing."
    ),
    "take_profit_pct": (
        "take-profit targets were cut from V0 for the same reason as "
        "stop-losses (no honest intraday fill model on daily bars)."
    ),
}


@dataclass
class ValidationResult:
    valid: bool
    spec: StrategySpec | None
    expanded: dict | None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    resolved_universe: list[str] = field(default_factory=list)


def _friendly_pydantic_errors(exc: pydantic.ValidationError) -> list[str]:
    out = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"])
        field_name = str(err["loc"][-1]) if err["loc"] else ""
        if field_name in V0_UNSUPPORTED_FIELDS:
            out.append(f"{loc}: {V0_UNSUPPORTED_FIELDS[field_name]}")
        elif err["type"] == "extra_forbidden":
            out.append(
                f"'{loc}' is not part of the strategy DSL — specs are a closed "
                f"vocabulary; unknown fields are rejected, never ignored"
            )
        else:
            out.append(f"{loc}: {err['msg']}")
    return out


def validate_spec(raw: dict) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(raw, dict):
        return ValidationResult(False, None, None, ["spec must be a JSON object"])

    raw = dict(raw)
    raw.setdefault("spec_version", SPEC_VERSION)
    if raw.get("spec_version") != SPEC_VERSION:
        return ValidationResult(
            False, None, None,
            [f"spec_version {raw.get('spec_version')} is not supported "
             f"(current: {SPEC_VERSION}); re-translate the strategy"],
        )

    try:
        spec = StrategySpec.model_validate(raw)
    except pydantic.ValidationError as exc:
        return ValidationResult(False, None, None, _friendly_pydantic_errors(exc))

    sel = spec.selection
    # exactly one of rank_by / static_allocation
    if (sel.rank_by is None) == (sel.static_allocation is None):
        errors.append(
            "selection must use either rank_by (with direction and n) or "
            "static_allocation — exactly one, never both"
        )
    if sel.rank_by is not None:
        if sel.direction is None or sel.n is None:
            errors.append("ranked selection needs both direction (top/bottom) and n")
        errors.extend(_check_signal(sel.rank_by, "selection.rank_by"))
    if sel.static_allocation is not None:
        total = sum(p.weight for p in sel.static_allocation)
        if abs(total - 1.0) > 0.001:
            errors.append(
                f"static allocation weights must sum to 1.0 (±0.001); got {total:.4f}"
            )
        if spec.entry_exit.hold_days is not None:
            errors.append("hold_days does not apply to static allocations")
        if len({p.ticker for p in sel.static_allocation}) != len(sel.static_allocation):
            errors.append("static allocation lists a ticker twice")

    for i, f in enumerate(spec.filters):
        errors.extend(_check_signal(f.signal, f"filters[{i}].signal"))
        if f.op == "between":
            if not (isinstance(f.value, list) and len(f.value) == 2):
                errors.append(f"filters[{i}]: 'between' needs a [low, high] pair")
            elif f.value[0] > f.value[1]:
                errors.append(f"filters[{i}]: 'between' range is reversed")
        elif isinstance(f.value, list):
            errors.append(f"filters[{i}]: op '{f.op}' takes a single number")

    # Costs: the sweep is a product rule, not a preference.
    sweep = {float(b) for b in spec.costs.sweep_bps}
    if not REQUIRED_SWEEP.issubset(sweep):
        errors.append(
            "costs.sweep_bps must include 0, 5 and 20 — the cost sweep is how "
            "this tool stays honest and cannot be reduced"
        )
    if any(b < 0 for b in sweep):
        errors.append("negative trading costs do not exist")

    # Window defaults + sanity
    end = spec.window.end or date.today().isoformat()
    start = spec.window.start or (
        date.fromisoformat(end) - timedelta(days=365 * DEFAULT_WINDOW_YEARS)
    ).isoformat()
    try:
        if date.fromisoformat(start) >= date.fromisoformat(end):
            errors.append(f"window.start ({start}) must be before window.end ({end})")
        if (date.fromisoformat(end) - date.fromisoformat(start)).days < 365:
            warnings.append(
                "test window is under a year — too short to mean much; the "
                "regime-coverage audit will flag this"
            )
    except ValueError:
        errors.append("window dates must be YYYY-MM-DD")
    if not spec.window.start or not spec.window.end:
        warnings.append(f"window defaulted to {start} → {end}")
    spec.window.start, spec.window.end = start, end

    # Universe resolution + symbol master (Gap #4)
    resolved: list[str] = []
    if spec.universe.preset == "user_watchlist":
        tickers: list[str] = []
        if sel.static_allocation is not None:
            # a static allocation defines its own universe
            tickers = [p.ticker for p in sel.static_allocation]
        elif spec.universe.resolved_tickers:
            tickers = spec.universe.resolved_tickers
        elif spec.universe.watchlist_file:
            try:
                tickers = read_watchlist_file(spec.universe.watchlist_file)
            except FileNotFoundError as exc:
                errors.append(str(exc))
        else:
            errors.append(
                "user_watchlist universe needs watchlist_file (upload one) or "
                "resolved_tickers"
            )
        if tickers:
            valid, problems = check_tickers(tickers)
            for p in problems:
                sug = f" — did you mean {p['suggestion']}?" if p.get("suggestion") else ""
                errors.append(f"unknown ticker '{p['ticker']}'{sug}")
            resolved = valid
            if valid and len(valid) < 3 and sel.rank_by is not None:
                warnings.append(
                    f"only {len(valid)} tickers in the watchlist — ranking "
                    f"needs a real cross-section to mean anything"
                )
            spec.universe.resolved_tickers = resolved
        if resolved and sel.rank_by is not None:
            warnings.append(
                "hand-picked universe: the selection-bias audit will compare "
                "this against a neutral universe and flag hindsight"
            )
    else:
        try:
            resolved = preset_tickers(spec.universe.preset)
        except (FileNotFoundError, ValueError) as exc:
            errors.append(str(exc))
        if spec.universe.watchlist_file:
            errors.append("watchlist_file only applies to the user_watchlist preset")
        # static allocation tickers must exist even on preset universes
        if sel.static_allocation is not None:
            valid, problems = check_tickers([p.ticker for p in sel.static_allocation])
            for p in problems:
                sug = f" — did you mean {p['suggestion']}?" if p.get("suggestion") else ""
                errors.append(f"unknown ticker '{p['ticker']}'{sug}")

    if spec.rebalance == "daily" and spec.entry_exit.hold_days is None and sel.rank_by is not None:
        warnings.append(
            "daily rebalancing with no holding period churns the whole "
            "portfolio every day — expect the cost sweep to hurt"
        )

    if errors:
        return ValidationResult(False, None, None, errors, warnings)

    expanded = spec.model_dump(mode="json")
    return ValidationResult(True, spec, expanded, [], warnings, resolved)


def _check_signal(sig, where: str) -> list[str]:
    errs = []
    if sig.type == "fundamental":
        errs.append(
            f"{where}: fundamental data (P/E, P/B, dividend yield) requires "
            f"licensed sources — coming in V1. V0 signals: return, sma_cross, "
            f"drawdown_from_high, volatility, rsi, daily_return."
        )
    if sig.type == "sma_cross":
        if sig.fast is None or sig.slow is None:
            errs.append(f"{where}: sma_cross needs fast and slow periods")
        elif sig.fast >= sig.slow:
            errs.append(f"{where}: sma_cross fast ({sig.fast}) must be shorter "
                        f"than slow ({sig.slow})")
        if sig.lookback_days is not None:
            errs.append(f"{where}: sma_cross uses fast/slow, not lookback_days")
    elif sig.type in ("return", "daily_return", "volatility", "drawdown_from_high"):
        if sig.lookback_days is None:
            errs.append(f"{where}: signal '{sig.type}' needs lookback_days")
    # A field the engine would silently ignore is a spec-fidelity break — the
    # user would confirm a readback of a strategy that never runs. Reject.
    if sig.type not in ("return", "daily_return") and sig.skip_days:
        errs.append(f"{where}: skip_days only applies to return signals — "
                    f"it does nothing on '{sig.type}' and is rejected rather "
                    f"than silently ignored")
    if sig.type != "sma_cross" and (sig.fast is not None or sig.slow is not None):
        errs.append(f"{where}: fast/slow only apply to sma_cross")
    if sig.type != "fundamental" and sig.field is not None:
        errs.append(f"{where}: 'field' only applies to fundamental signals")
    return errs
