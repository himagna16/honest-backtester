"""M2 acceptance: the three worked examples validate; hallucinated tickers are
rejected with useful errors; invalid specs can never reach the engine."""
import pytest

from dsl.readback import readback
from dsl.validator import validate_spec

# ---- the three worked examples from Section 2 (verbatim, minus $schema) ----

EXAMPLE_MOMENTUM = {
    "name": "6mo momentum top-5",
    "universe": {"preset": "sp500"},
    "selection": {"rank_by": {"type": "return", "lookback_days": 126, "skip_days": 0},
                  "direction": "top", "n": 5},
    "weighting": "equal",
    "rebalance": "monthly",
    "costs": {"sweep_bps": [0, 5, 20]},
    "benchmark": "SPY",
    "window": {"start": "2016-01-01", "end": "2026-01-01"},
}

EXAMPLE_DIP = {
    "name": "Dip buyer, 5-day hold",
    "universe": {"preset": "sp500"},
    "filters": [{"signal": {"type": "daily_return", "lookback_days": 1},
                 "op": "lte", "value": -0.05}],
    "selection": {"rank_by": {"type": "daily_return", "lookback_days": 1},
                  "direction": "bottom", "n": 20},
    "weighting": "equal",
    "rebalance": "daily",
    "entry_exit": {"hold_days": 5},
    "costs": {"sweep_bps": [0, 5, 20]},
    "benchmark": "SPY",
}

EXAMPLE_6040 = {
    "name": "60/40 annual rebalance",
    "universe": {"preset": "user_watchlist", "watchlist_file": "sixty_forty.txt"},
    "selection": {"static_allocation": [
        {"ticker": "SPY", "weight": 0.60},
        {"ticker": "AGG", "weight": 0.40}]},
    "weighting": "equal",
    "rebalance": "annual",
    "costs": {"sweep_bps": [0, 5, 20]},
    "benchmark": "sixty_forty",
}


@pytest.mark.parametrize("spec", [EXAMPLE_MOMENTUM, EXAMPLE_DIP, EXAMPLE_6040],
                         ids=["momentum", "dip", "sixty_forty"])
def test_worked_examples_validate(spec):
    res = validate_spec(spec)
    assert res.valid, res.errors
    assert res.expanded is not None
    assert res.expanded["spec_version"] == 1
    # defaults are expanded: window always concrete, costs always the sweep
    assert res.expanded["window"]["start"] and res.expanded["window"]["end"]
    assert set(res.expanded["costs"]["sweep_bps"]) >= {0, 5, 20}
    # readback is deterministic template output
    assert readback(res.spec) == readback(validate_spec(spec).spec)


def test_universe_sizes():
    res = validate_spec(EXAMPLE_MOMENTUM)
    assert len(res.resolved_universe) > 400  # full S&P list resolved


VALID_SPECS = [
    EXAMPLE_MOMENTUM, EXAMPLE_DIP, EXAMPLE_6040,
    # 12-1 momentum
    {**EXAMPLE_MOMENTUM, "name": "12-1",
     "selection": {"rank_by": {"type": "return", "lookback_days": 252, "skip_days": 21},
                   "direction": "top", "n": 10}},
    # low-vol tilt
    {**EXAMPLE_MOMENTUM, "name": "low vol",
     "selection": {"rank_by": {"type": "volatility", "lookback_days": 126},
                   "direction": "bottom", "n": 20}},
    # sector rotation on ETFs
    {**EXAMPLE_MOMENTUM, "name": "sectors", "universe": {"preset": "sector_etfs"},
     "selection": {"rank_by": {"type": "return", "lookback_days": 63},
                   "direction": "top", "n": 3}},
    # SMA crossover filter
    {**EXAMPLE_MOMENTUM, "name": "trend filter",
     "filters": [{"signal": {"type": "sma_cross", "fast": 50, "slow": 200},
                  "op": "gt", "value": 0}]},
    # RSI mean reversion
    {**EXAMPLE_MOMENTUM, "name": "rsi dip",
     "selection": {"rank_by": {"type": "rsi", "lookback_days": 14},
                   "direction": "bottom", "n": 10}},
    # inverse vol weighting on nasdaq
    {**EXAMPLE_MOMENTUM, "name": "nasdaq iv", "universe": {"preset": "nasdaq100"},
     "weighting": "inverse_vol"},
    # drawdown filter with between
    {**EXAMPLE_MOMENTUM, "name": "dd band",
     "filters": [{"signal": {"type": "drawdown_from_high", "lookback_days": 252},
                  "op": "between", "value": [-0.3, -0.1]}]},
]

INVALID_SPECS = [
    # hallucinated ticker
    ({**EXAMPLE_6040, "selection": {"static_allocation": [
        {"ticker": "FAKECO", "weight": 0.5}, {"ticker": "SPY", "weight": 0.5}]}},
     "FAKECO"),
    # renamed ticker gets a suggestion
    ({**EXAMPLE_6040, "selection": {"static_allocation": [
        {"ticker": "FB", "weight": 0.5}, {"ticker": "SPY", "weight": 0.5}]}},
     "META"),
    # both rank_by and static
    ({**EXAMPLE_MOMENTUM, "selection": {
        "rank_by": {"type": "return", "lookback_days": 126},
        "direction": "top", "n": 5,
        "static_allocation": [{"ticker": "SPY", "weight": 1.0}]}},
     "exactly one"),
    # neither
    ({**EXAMPLE_MOMENTUM, "selection": {}}, "exactly one"),
    # weights don't sum to 1
    ({**EXAMPLE_6040, "selection": {"static_allocation": [
        {"ticker": "SPY", "weight": 0.6}, {"ticker": "AGG", "weight": 0.6}]}},
     "sum to 1.0"),
    # stop-loss cut from V0
    ({**EXAMPLE_DIP, "entry_exit": {"hold_days": 5, "stop_loss_pct": 0.1}},
     "stop-loss"),
    # fundamental gated to V1
    ({**EXAMPLE_MOMENTUM, "selection": {
        "rank_by": {"type": "fundamental", "field": "pe"},
        "direction": "bottom", "n": 10}},
     "V1"),
    # fast >= slow
    ({**EXAMPLE_MOMENTUM,
      "filters": [{"signal": {"type": "sma_cross", "fast": 200, "slow": 50},
                   "op": "gt", "value": 0}]},
     "shorter"),
    # window reversed
    ({**EXAMPLE_MOMENTUM, "window": {"start": "2026-01-01", "end": "2016-01-01"}},
     "before"),
    # cost sweep cannot be reduced
    ({**EXAMPLE_MOMENTUM, "costs": {"sweep_bps": [0]}}, "cannot be reduced"),
    # unknown field smuggled in
    ({**EXAMPLE_MOMENTUM, "leverage": 3}, "closed vocabulary"),
    # unknown preset
    ({**EXAMPLE_MOMENTUM, "universe": {"preset": "crypto"}}, "universe.preset"),
]


@pytest.mark.parametrize("spec", VALID_SPECS,
                         ids=[s["name"] for s in VALID_SPECS])
def test_valid_specs_pass(spec):
    res = validate_spec(spec)
    assert res.valid, res.errors


@pytest.mark.parametrize("spec,needle", INVALID_SPECS,
                         ids=[n for _, n in INVALID_SPECS])
def test_invalid_specs_fail_with_useful_errors(spec, needle):
    res = validate_spec(spec)
    assert not res.valid
    joined = " ".join(res.errors)
    assert needle.lower() in joined.lower(), f"expected {needle!r} in: {joined}"


def test_fakeco_error_is_useful():
    res = validate_spec({**EXAMPLE_6040, "selection": {"static_allocation": [
        {"ticker": "FAKECO", "weight": 1.0}]}})
    assert not res.valid
    assert any("FAKECO" in e for e in res.errors)


def test_readback_mentions_conversions():
    res = validate_spec(EXAMPLE_MOMENTUM)
    text = readback(res.spec)
    assert "126 trading days" in text and "6 months" in text
    assert "0, 5 and 20 bps" in text
    assert "S&P 500" in text


def test_readback_dip():
    res = validate_spec(EXAMPLE_DIP)
    text = readback(res.spec)
    assert "at or below -5%" in text
    assert "5 trading days" in text


def test_expanded_spec_revalidates_identically():
    """Storing the expanded spec and re-validating it must be a fixed point."""
    first = validate_spec(EXAMPLE_DIP)
    second = validate_spec(first.expanded)
    assert second.valid, second.errors
    assert second.expanded == first.expanded
