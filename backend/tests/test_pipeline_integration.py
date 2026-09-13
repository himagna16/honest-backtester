"""The audit pipeline's newer wiring: risk-free path, trial ledger, purge
disclosure, and the inference/attribution payloads.

These cover the SEAMS, not the modules — each module has its own suite. What
can break here is the plumbing between them: a rate averaged in one place and
pathed in another, a ledger that lets a strategy raise its own bar, a payload
whose refusal is silently dropped.

Every expected value below is derived independently of the function it checks
(hand arithmetic, a closed form, or a property that must hold for any correct
implementation). A test that recomputes the implementation cannot fail.
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

from audit import extra
from audit.pipeline import _rf_daily, _seed, build_report
from dsl.validator import validate_spec
from engine.benchmark import BENCHMARK_TICKERS
from engine.data import PricePanel, get_panel


# --------------------------------------------------------------------------
# the risk-free path
# --------------------------------------------------------------------------

def _panel_with_irx(irx_values, index) -> PricePanel:
    prices = pd.DataFrame({"^IRX": irx_values}, index=index)
    return PricePanel(prices=prices, as_of="2020-01-01", sources={"^IRX": "fixture"})


def test_rf_is_a_path_not_a_window_average():
    """A rate that MOVES must reach the engine as a path.

    Hand-derived: ^IRX quotes an annualized percentage, so a quote of 5.04
    is a daily rate of 5.04/100/252 = 0.0002. A quote of 2.52 is 0.0001.
    An averaging implementation would return the same 0.00015 for both days.
    """
    idx = pd.bdate_range("2020-01-01", periods=2)
    panel = _panel_with_irx([5.04, 2.52], idx)
    rf, mean_daily, source, note = _rf_daily(panel, idx)

    assert source == "^IRX"
    assert note is None
    assert isinstance(rf, pd.Series), "the engine must receive a path, not a scalar"
    assert rf.iloc[0] == pytest.approx(0.0002, abs=1e-12)
    assert rf.iloc[1] == pytest.approx(0.0001, abs=1e-12)
    # the two days genuinely differ — an averaged path would not
    assert rf.iloc[0] != rf.iloc[1]
    assert mean_daily == pytest.approx(0.00015, abs=1e-12)


def test_a_gap_at_the_window_start_is_disclosed_and_never_backfilled():
    """Carrying a later rate onto an earlier day would let the future set an
    earlier day's hurdle. Those days take 0.0 and the shortfall is DISCLOSED."""
    idx = pd.bdate_range("2020-01-01", periods=4)
    panel = _panel_with_irx([float("nan"), float("nan"), 5.04, 5.04], idx)
    rf, _, source, note = _rf_daily(panel, idx)

    assert source == "^IRX"
    assert rf.iloc[0] == 0.0 and rf.iloc[1] == 0.0, "no backfill from the future"
    assert rf.iloc[2] == pytest.approx(0.0002, abs=1e-12)
    assert note is not None and "2 session" in note, "the gap must surface"


def test_an_interior_hole_carries_forward_rather_than_refusing():
    """^IRX keeps its own holiday calendar, so holes INSIDE the window are
    normal. They carry forward — and coverage must stay complete, because
    metrics._excess refuses a partial rate path."""
    idx = pd.bdate_range("2020-01-01", periods=4)
    panel = _panel_with_irx([5.04, float("nan"), float("nan"), 2.52], idx)
    rf, _, _, note = _rf_daily(panel, idx)

    assert note is None, "an interior hole is not a shortfall"
    assert rf.notna().all() and len(rf) == len(idx)
    assert rf.iloc[1] == pytest.approx(0.0002, abs=1e-12)   # carried forward
    assert rf.iloc[2] == pytest.approx(0.0002, abs=1e-12)


def test_no_irx_column_means_zero_and_says_so():
    idx = pd.bdate_range("2020-01-01", periods=3)
    panel = PricePanel(prices=pd.DataFrame({"SPY": [1.0, 2.0, 3.0]}, index=idx),
                       as_of="2020-01-01", sources={"SPY": "fixture"})
    rf, mean_daily, source, note = _rf_daily(panel, idx)
    assert (rf, mean_daily, source, note) == (0.0, 0.0, "none", None)


# --------------------------------------------------------------------------
# the trial ledger
# --------------------------------------------------------------------------

def test_the_ledger_rejects_replays_and_degenerate_sharpes():
    """A replay of a stored run must not inflate its own multiple-testing
    haircut, and a non-finite Sharpe is not a trial."""
    key = "test_universe_dedup"
    assert extra.trial_sharpes(key) == []

    extra.record_trial_sharpe("run_a", key, 0.8)
    extra.record_trial_sharpe("run_b", key, 1.4)
    extra.record_trial_sharpe("run_a", key, 99.0)          # same run replayed
    extra.record_trial_sharpe("run_c", key, float("nan"))  # degenerate
    extra.record_trial_sharpe("run_d", key, float("inf"))

    led = extra.trial_sharpes(key)
    assert led == [0.8, 1.4], "replay and non-finite entries must not enter"
    assert 99.0 not in led, "a replay must not overwrite the original Sharpe"


def test_one_recorded_trial_is_not_a_spread_of_zero(sp500_panel):
    """REGRESSION. A single-entry ledger has no variance. Passing it through
    raised ValueError out of trial_sharpe_variance_periodic and took down the
    whole run; treating it as variance 0 would be worse still, claiming the
    search was free and handing the strategy an undeserved pass.

    The honest answer is that the DSR is not computable. Mutant: pass the
    1-element list straight through and this test dies (with the ValueError
    that took the run down).

    `attempts` must exceed 1 to exercise this at all: at N=1 the DSR reduces to
    the PSR and needs no trial variance, so a single-attempt run is computable
    either way and would hide the bug.
    """
    spec, uni, panel = _small_run()

    rep = build_report(spec, panel, uni, attempts=7, trial_sharpes_annual=[0.9])
    dsr = rep["significance"]["sharpe"]["dsr"]
    assert dsr["available"] is False
    assert dsr["dsr"] is None

    finding = _finding(rep, "sharpe_significance")
    assert finding["severity"] == "note", "a refusal is not a pass"
    assert "not computable" in finding["name"].lower()


def test_two_recorded_trials_make_the_deflated_sharpe_computable(sp500_panel):
    """The complement of the test above: at two entries there IS a variance,
    so the DSR must actually be produced rather than refused."""
    spec, uni, panel = _small_run()
    rep = build_report(spec, panel, uni, attempts=2,
                       trial_sharpes_annual=[0.4, 1.6])
    dsr = rep["significance"]["sharpe"]["dsr"]
    assert dsr["available"] is True
    assert dsr["dsr"] is not None and 0.0 <= dsr["dsr"] <= 1.0
    # The luck threshold must be a real annualized Sharpe, and a wider spread
    # of trials must set a HIGHER bar than a narrow one (Bailey & LdP 2014:
    # SR0 scales with sqrt(V[{SR_n}])).
    narrow = build_report(spec, panel, uni, attempts=2,
                          trial_sharpes_annual=[0.95, 1.05])
    wide_sr0 = dsr["sr0_annual"]
    narrow_sr0 = narrow["significance"]["sharpe"]["dsr"]["sr0_annual"]
    assert wide_sr0 > narrow_sr0, (
        "a wider spread of trials must raise the luck threshold, not lower it"
    )


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------

def test_the_bootstrap_seed_is_a_function_of_the_spec_alone():
    """Resampling must not make a report irreproducible (Gap #2)."""
    a = validate_spec(_SPEC).spec
    b = validate_spec(_SPEC).spec
    assert _seed(a) == _seed(b)

    other = dict(_SPEC)
    other["rebalance"] = "monthly"
    assert _seed(validate_spec(other).spec) != _seed(a), (
        "a different spec must not silently reuse another spec's interval"
    )


# --------------------------------------------------------------------------
# the payloads
# --------------------------------------------------------------------------

def test_the_report_discloses_how_far_the_holdout_boundary_leaks(sp500_panel):
    """The 70/30 cut is a plain date split. The overlap is a data problem, and
    data problems surface."""
    spec, uni, panel = _small_run()
    rep = build_report(spec, panel, uni, attempts=1)

    purge = rep["holdout_purge"]
    assert purge is not None and purge["total"] > 0
    # `reach` is the MAX of the component reaches (they are alternative paths
    # to the same bar, not additive), and the total is reach + the rest.
    assert purge["reach"] == max(purge["signal_reach"], purge["staleness_reach"],
                                 purge["extra_reach"])
    assert purge["total"] == (purge["reach"] + purge["carry"]
                              + purge["execution_lag"] + purge["cost_charge"]), (
        f"total {purge['total']} must equal its disclosed terms {purge}"
    )

    finding = _finding(rep, "holdout_purge")
    assert str(purge["total"]) in finding["message"]


def test_a_purge_that_cannot_be_measured_surfaces_instead_of_vanishing(
    sp500_panel, monkeypatch
):
    """REGRESSION. The purge disclosure used to be gated on `if purge:` behind a
    bare `except: purge = None`, so a run that could NOT size its own overlap
    printed nothing at all — the most dangerous case reported as the quietest.

    SplitError carries a machine-readable reason precisely so a caller can turn
    the refusal into a Finding "instead of silently falling back to a leaky
    split" (validation/splits.py). Mutant: swallow it back to `purge = None`
    with no Finding, and this test dies.
    """
    import audit.pipeline as pipeline_mod
    from validation import SplitError

    def _refuse(*_a, **_k):
        raise SplitError("signal dates are not calendar bars",
                         reason="signal_dates_not_on_calendar",
                         n_missing=70, n_signal_dates=70)

    monkeypatch.setattr(pipeline_mod, "purge_components", _refuse)

    spec, uni, panel = _small_run()
    rep = build_report(spec, panel, uni, attempts=1)

    finding = _finding(rep, "holdout_purge")
    assert finding["severity"] == "warning", "an unmeasurable overlap is not a note"
    assert finding["evidence"]["reason"] == "signal_dates_not_on_calendar", (
        "the machine-readable reason must reach the report"
    )
    # "unknown" is the honest word here; "zero" would be a lie.
    assert "unknown" in finding["message"].lower()
    assert rep["holdout_purge"] is None


def test_inference_and_attribution_payloads_carry_their_own_refusals(sp500_panel):
    """`available: False` must travel WITH a reason — a refusal that is merely
    absent reads as a pass."""
    spec, uni, panel = _small_run()
    rep = build_report(spec, panel, uni, attempts=1)

    for payload in (rep["significance"]["sharpe"],
                    rep["significance"]["excess"],
                    rep["attribution"]):
        assert "available" in payload
        if payload["available"] is False:
            assert payload.get("unavailable_reason"), (
                "a refusal must say why"
            )

    # the whole report still serializes — an inf or a NaN here would 500 the API
    import json
    json.dumps(rep)


def test_a_benchmark_verdict_always_travels_with_its_coverage_caveat(sp500_panel):
    """The three-state verdict rests on an interval that covers less than
    nominal. The reader is entitled to see that next to the word 'beats'."""
    spec, uni, panel = _small_run()
    rep = build_report(spec, panel, uni, attempts=1)
    excess = rep["significance"]["excess"]
    if not excess["available"]:
        pytest.skip("this fixture window refuses a margin verdict")

    assert excess["ci_coverage_caveat"], "a verdict without its caveat is a lie"
    finding = _finding(rep, "excess_significance")
    assert excess["ci_coverage_caveat"] in finding["message"]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

_SPEC = {
    "name": "pipeline integration fixture",
    "universe": {"preset": "dow30"},
    "selection": {"rank_by": {"type": "volatility", "lookback_days": 20},
                  "direction": "bottom", "n": 5},
    "weighting": "equal",
    "rebalance": "quarterly",
    "benchmark": "SPY",
    "window": {"start": "2018-01-02", "end": "2021-12-31"},
}

_CACHE: dict = {}


def _small_run():
    """A real but cheap spec/panel pair, built once per session."""
    if "run" not in _CACHE:
        res = validate_spec(_SPEC)
        assert res.valid, res.errors
        spec, uni = res.spec, res.resolved_universe
        needed = uni + BENCHMARK_TICKERS[spec.benchmark] + ["SPY", "^IRX"]
        panel = get_panel(needed, start=spec.window.start, end=spec.window.end)
        uni_in = [t for t in uni if t in panel.prices.columns]
        _CACHE["run"] = (spec, uni_in, panel)
    return _CACHE["run"]


def _finding(report: dict, check: str) -> dict:
    matches = [f for f in report["audit"] if f["check"] == check]
    assert matches, f"no finding with check={check!r} in the report"
    return matches[0]
