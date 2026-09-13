"""Formula-level tests for the professional metric suite.

Every expected value here is computed with EXACT arithmetic written out in
the test — closed forms, hand-rolled sums, fixtures whose answer is known
analytically — never by calling the function under test (or a sibling that
shares its code path). A test that computes the expectation with the same
code it is testing pins the bug, not the formula.

Edge cases are first-class: all-positive returns (no downside at all), zero
volatility, a single observation, and a drawdown that never recovers.
"""
import json
import math

import numpy as np
import pandas as pd
import pytest

from dsl import conventions
from engine import metrics

YEAR = conventions.TRADING_DAYS_PER_YEAR
SQRT_YEAR = math.sqrt(YEAR)


def series(values: list[float], start: str = "2021-01-04") -> pd.Series:
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)),
                     dtype=float)


def daily(values: list[float], start: str = "2021-01-04") -> pd.Series:
    return series(values, start)


# --------------------------------------------------------------------------
# annualization comes from conventions.py, not from a literal in the module
# --------------------------------------------------------------------------

def test_no_hardcoded_trading_day_count_in_metrics_module():
    source = open(metrics.__file__).read()
    assert "252" not in source, (
        "metrics.py must take the trading-day count from dsl/conventions.py"
    )
    assert metrics.TRADING_DAYS_PER_YEAR is conventions.TRADING_DAYS_PER_YEAR


def test_cagr_uses_the_conventions_constant():
    r = daily([0.01, -0.02, 0.03])
    growth = 1.01 * 0.98 * 1.03
    assert metrics.cagr(r) == pytest.approx(growth ** (YEAR / 3) - 1, abs=1e-12)


# --------------------------------------------------------------------------
# backward compatibility of summary()
# --------------------------------------------------------------------------

def test_summary_keeps_the_original_keys_and_their_meaning():
    r = daily([0.01, -0.02, 0.03, 0.004, -0.001])
    b = daily([0.005, -0.01, 0.02, 0.001, 0.002])
    out = metrics.summary(r, 0.0001, b)

    growth = 1.01 * 0.98 * 1.03 * 1.004 * 0.999
    assert out["total_return"] == pytest.approx(growth - 1, abs=1e-12)
    assert out["cagr"] == pytest.approx(growth ** (YEAR / 5) - 1, abs=1e-12)
    assert out["days"] == 5

    # vol: sample std (ddof=1) of the five returns, annualized by sqrt(year)
    vals = [0.01, -0.02, 0.03, 0.004, -0.001]
    mean = sum(vals) / 5
    var = sum((v - mean) ** 2 for v in vals) / 4
    assert out["vol"] == pytest.approx(math.sqrt(var) * SQRT_YEAR, abs=1e-12)

    # sharpe on excess returns with the same ddof
    ex = [v - 0.0001 for v in vals]
    ex_mean = sum(ex) / 5
    ex_var = sum((v - ex_mean) ** 2 for v in ex) / 4
    assert out["sharpe"] == pytest.approx(
        ex_mean / math.sqrt(ex_var) * SQRT_YEAR, abs=1e-12)

    # max drawdown: trough 1.01*0.98 against the peak 1.01 it fell from
    assert out["max_drawdown"] == pytest.approx(0.98 - 1.0, abs=1e-12)
    for key in ("total_return", "cagr", "vol", "sharpe", "max_drawdown",
                "days", "hit_rate"):
        assert key in out


def test_summary_without_benchmark_omits_only_the_relative_family():
    out = metrics.summary(daily([0.01, -0.01, 0.02]))
    assert "hit_rate" not in out and "beta" not in out
    assert "sortino" in out and "calmar" in out and "var_95" in out


def test_nan_refusal_is_not_negotiable():
    with pytest.raises(ValueError, match="NaN"):
        metrics.summary(daily([0.01, float("nan"), 0.02]))


# --------------------------------------------------------------------------
# standard error of the Sharpe: IID vs autocorrelation-consistent
# --------------------------------------------------------------------------

def _ar1_paths(rng, phi, n_paths, n_obs, mu, sd, burn=200):
    """n_paths independent AR(1) daily return series with stationary standard
    deviation exactly `sd` and mean exactly `mu`."""
    eps_sd = sd * math.sqrt(1.0 - phi * phi)
    x = np.zeros(n_paths)
    out = np.zeros((n_paths, n_obs))
    for t in range(burn + n_obs):
        x = phi * x + rng.normal(0.0, eps_sd, n_paths)
        if t >= burn:
            out[:, t - burn] = x
    return mu + out


def test_sharpe_stderr_hac_tracks_the_real_sampling_dispersion_and_iid_does_not():
    """The IID Lo (2002) form is what this engine's own audit file flags as a
    major open defect: overlapping hold_days cohorts make daily returns
    autocorrelated by construction, and positive autocorrelation makes the IID
    standard error TOO SMALL — a confidence interval that is too narrow,
    driving a GREEN verdict, is a false honesty claim.

    The independent value here is the sampling distribution itself: 250
    separate AR(1) paths are drawn and the SAMPLE standard deviation of the
    realized annualized Sharpe ACROSS those paths is measured directly. It is
    cross-checked against the closed form sqrt(252*(1+phi)/((1-phi)*T)), the
    standard first-order correction for the variance of the mean of an AR(1).
    Neither number comes from the functions under test.
    """
    phi, n_obs, n_paths = 0.4, 750, 250
    rng = np.random.default_rng(20240917)
    paths = _ar1_paths(rng, phi, n_paths, n_obs, mu=0.0005, sd=0.011)
    idx = pd.bdate_range("2015-01-01", periods=n_obs)

    sharpes, hac, iid = [], [], []
    for row in paths:
        s = pd.Series(row, index=idx)
        sharpes.append(metrics.sharpe(s))
        hac.append(metrics.sharpe_stderr_hac(s))
        iid.append(metrics.sharpe_stderr_iid(s))

    realized = float(np.std(sharpes, ddof=1))
    closed_form = math.sqrt(YEAR * (1 + phi) / ((1 - phi) * n_obs))
    # the simulation agrees with the closed form, so `realized` is trustworthy
    assert realized == pytest.approx(closed_form, rel=0.15)

    mean_hac, mean_iid = float(np.mean(hac)), float(np.mean(iid))
    assert mean_hac == pytest.approx(realized, rel=0.20)     # ~0.93x here
    assert mean_iid < 0.80 * realized                        # ~0.66x here
    assert mean_hac > 1.25 * mean_iid       # kills a "hac = iid" mutant

    # The HAC form does not CLOSE the gap — a kernel estimator is downward
    # biased in finite samples — and the residual shortfall is what the report
    # has to disclose. The disclosed figure is pinned against THIS simulation,
    # so the sentence cannot drift away from the measurement it claims.
    assert mean_hac / realized == pytest.approx(metrics.HAC_DISPERSION_RATIO,
                                                abs=0.04)
    assert mean_iid / realized < metrics.HAC_DISPERSION_RATIO - 0.2


def test_the_hac_standard_error_carries_its_understatement_with_the_value():
    """The residual ~7% understatement at phi = 0.4 is argued correctly in the
    docstring — which no report reader will ever open. A disclosure only counts
    if it travels with the VALUE, so it is a field beside the number."""
    rng = np.random.default_rng(8)
    r = pd.Series(rng.normal(0.0005, 0.011, 600),
                  index=pd.bdate_range("2015-01-01", periods=600))
    note = metrics.summary(r)["sharpe_stderr_hac_disclosure"]
    assert note == metrics.SHARPE_STDERR_HAC_DISCLOSURE
    assert metrics.HAC_DISPERSION_RATIO == 0.93
    assert "0.93x" in note          # the simulated ratio, in the payload
    assert "UNDERSTATES" in note
    assert "0.4" in note and "floor" in note
    # deterministic template text, and strict-JSON encodable
    assert note == metrics.summary(r)["sharpe_stderr_hac_disclosure"]
    json.dumps({"sharpe_stderr_hac_disclosure": note}, allow_nan=False)


def test_the_two_sharpe_standard_errors_agree_on_genuinely_iid_returns():
    """The HAC form must REDUCE to Lo eq. (9) when there is nothing to
    correct — otherwise it is not a refinement, it is a different number.

    Compared IN EXPECTATION over 150 IID draws, because a HAC estimate on one
    sample carries real noise in its lag terms (±10% at T = 750 is normal, and
    pretending otherwise would be the circular move of tuning the tolerance to
    one lucky seed). Also checked exactly at lag = 0, where the Bartlett sum is
    empty and the estimator is algebraically Lo's own expression."""
    rng = np.random.default_rng(4242)
    n_obs = 750
    idx = pd.bdate_range("2015-01-01", periods=n_obs)
    hac, iid = [], []
    for _ in range(150):
        r = pd.Series(rng.normal(0.0005, 0.011, n_obs), index=idx)
        hac.append(metrics.sharpe_stderr_hac(r))
        iid.append(metrics.sharpe_stderr_iid(r))
    assert float(np.mean(hac)) == pytest.approx(float(np.mean(iid)), rel=0.05)

    # at lag = 0 the delta-method variance IS Lo's (1 + SR^2/2)/T, up to the
    # ddof=1 vs ddof=0 difference of O(1/T)
    r = pd.Series(rng.normal(0.0005, 0.011, 1500),
                  index=pd.bdate_range("2015-01-01", periods=1500))
    sr_period = metrics.sharpe(r) / SQRT_YEAR
    lo = math.sqrt((1.0 + 0.5 * sr_period ** 2) / 1500) * SQRT_YEAR
    assert metrics.sharpe_stderr_hac(r, lag=0) == pytest.approx(lo, rel=0.01)


def test_sharpe_stderr_iid_matches_lo_2002_equation_9_by_hand():
    vals = [0.01, -0.02, 0.03, 0.004, -0.001, 0.012, -0.008, 0.02]
    r = daily(vals)
    rf = 0.0001
    ex = [v - rf for v in vals]
    n = len(ex)
    mean = sum(ex) / n
    var = sum((v - mean) ** 2 for v in ex) / (n - 1)
    sr_period = mean / math.sqrt(var)
    expected = math.sqrt((1.0 + 0.5 * sr_period ** 2) / n) * SQRT_YEAR
    assert metrics.sharpe_stderr_iid(r, rf) == pytest.approx(expected, abs=1e-12)


def test_summary_names_the_iid_standard_error_so_nobody_publishes_off_it():
    rng = np.random.default_rng(8)
    r = pd.Series(rng.normal(0.0005, 0.011, 600),
                  index=pd.bdate_range("2015-01-01", periods=600))
    out = metrics.summary(r)
    assert "sharpe_stderr" not in out, (
        "the unqualified name invites a consumer to build a CI off the IID form"
    )
    assert np.isfinite(out["sharpe_stderr_iid"])
    assert np.isfinite(out["sharpe_stderr_hac"])
    # LITERAL, not metrics.newey_west_lag(600): comparing the summary against
    # the function summary() itself called moves both sides of the equality
    # together, so the assertion cannot fail. floor(4*(600/100)**(2/9)) = 5.
    assert out["sharpe_stderr_hac_lag"] == 5


def test_newey_west_bandwidth_is_the_standard_rule():
    """floor(4 * (T/100) ** (2/9)), Newey & West (1994).

    Every expectation is a LITERAL. Restating the formula in the test moves the
    expectation with the implementation, and T = 750 / 1250 alone do not
    discriminate the exponent: 2/9, 1/4 and 0.25 all give 6 and 7 there. The
    sample sizes below are chosen so the exponent is load-bearing —
        T =  250: 2/9 -> 4   but 1/4 -> 5
        T =  300: 2/9 -> 5   but 1/5 -> 4
        T = 2000: 2/9 -> 7   but 1/4 -> 8
        T = 5000: 2/9 -> 9   but 1/4 -> 10 and 1/5 -> 8
    — while T = 100 pins the coefficient alone (any exponent gives 4·1 = 4),
    and the 600 and 5000 rows are on the wrong side of a rounding boundary for
    round() (5.96 and 9.54), so floor is pinned too.
    """
    assert metrics.newey_west_lag(100) == 4
    assert metrics.newey_west_lag(250) == 4
    assert metrics.newey_west_lag(300) == 5
    assert metrics.newey_west_lag(600) == 5
    assert metrics.newey_west_lag(750) == 6
    assert metrics.newey_west_lag(1250) == 7
    assert metrics.newey_west_lag(2000) == 7
    assert metrics.newey_west_lag(5000) == 9
    # a series too short to have a lag structure gets no lags at all, and two
    # observations get one — the n < 2 cutoff, pinned in both directions
    assert metrics.newey_west_lag(1) == 0
    assert metrics.newey_west_lag(2) == 1


# --------------------------------------------------------------------------
# NaN discipline on a risk-free / MAR PATH, not just on returns
# --------------------------------------------------------------------------

def test_a_risk_free_path_with_holes_is_refused_by_every_consumer_of_it():
    """^IRX keeps its own holiday calendar, so the daily rf path the pipeline
    is meant to graduate to WILL have holes that the return index does not.
    `_check` runs on `returns` before the subtraction, so those NaNs used to
    reach pandas' skipna and vanish: 10 missing days out of 300 inflated
    Sortino by 13% and left alpha_beta returning beta=nan while still
    reporting n_obs=300. Data problems surface as findings, never absorbed."""
    rng = np.random.default_rng(1234)
    idx = pd.bdate_range("2021-01-04", periods=300)
    r = pd.Series(rng.normal(0.0006, 0.01, 300), index=idx)
    b = pd.Series(rng.normal(0.0003, 0.008, 300), index=idx)
    rf_full = pd.Series(6e-5, index=idx)
    rf_holes = rf_full.copy()
    rf_holes.iloc[[3, 17, 44, 60, 91, 120, 155, 200, 240, 280]] = np.nan

    calls = {
        "sharpe": lambda rf: metrics.sharpe(r, rf),
        "sharpe_stderr_iid": lambda rf: metrics.sharpe_stderr_iid(r, rf),
        "sharpe_stderr_hac": lambda rf: metrics.sharpe_stderr_hac(r, rf),
        "downside_deviation": lambda rf: metrics.downside_deviation(r, rf),
        "sortino": lambda rf: metrics.sortino(r, rf),
        "alpha_beta": lambda rf: metrics.alpha_beta(r, b, rf),
        "rolling_sharpe": lambda rf: metrics.rolling_sharpe(r, 60, rf),
        "rolling_beta": lambda rf: metrics.rolling_beta(r, b, 60, rf),
        "martin_ratio": lambda rf: metrics.martin_ratio(r, rf),
        "summary": lambda rf: metrics.summary(r, rf, b),
    }
    for name, call in calls.items():
        call(rf_full)                       # the complete path is fine
        # the refusal must NAME the damage — 10 missing sessions out of 300 —
        # because a bare "bad rf" is not a finding anybody can act on. (The
        # previous `assert name` here was true for every non-empty string and
        # so could not fail.)
        with pytest.raises(
                ValueError,
                match=r"(does not cover 10 of the 300|"
                      r"risk-free path contains 10 NaN)") as exc:
            call(rf_holes)
        assert "10" in str(exc.value), f"{name} hid the missing-session count"


def test_a_complete_rf_path_gives_the_same_answer_as_the_equivalent_scalar():
    """The path route must not be a different formula — a constant path and
    the scalar it equals have to agree exactly."""
    vals = [0.01, -0.02, 0.03, -0.01, 0.004, 0.012]
    r = daily(vals)
    rf = 0.0002
    path = pd.Series(rf, index=r.index)
    assert metrics.sharpe(r, path) == pytest.approx(metrics.sharpe(r, rf),
                                                    abs=1e-15)
    assert metrics.sortino(r, path) == pytest.approx(metrics.sortino(r, rf),
                                                     abs=1e-12)
    # ...and hand arithmetic, so neither side is the function under test
    dev = [min(v - rf, 0.0) for v in vals]
    dd = math.sqrt(sum(d * d for d in dev) / 6) * SQRT_YEAR
    assert metrics.downside_deviation(r, path) == pytest.approx(dd, abs=1e-15)


def test_a_varying_rf_path_is_subtracted_day_by_day_not_averaged():
    """A flat path and a wobbling path with the SAME mean must give different
    downside deviations — otherwise the path is being collapsed to a scalar
    somewhere and the upgrade to a real ^IRX series buys nothing."""
    vals = [0.01, -0.02, 0.03, -0.01]
    r = daily(vals)
    wobble = pd.Series([0.0, 0.0004, 0.0, 0.0004], index=r.index)
    assert float(wobble.mean()) == pytest.approx(0.0002, abs=1e-15)
    dev = [min(v - w, 0.0) for v, w in zip(vals, [0.0, 0.0004, 0.0, 0.0004])]
    expected = math.sqrt(sum(d * d for d in dev) / 4) * SQRT_YEAR
    assert metrics.downside_deviation(r, wobble) == pytest.approx(expected,
                                                                  abs=1e-15)
    assert metrics.downside_deviation(r, wobble) != pytest.approx(
        metrics.downside_deviation(r, 0.0002), abs=1e-9)


# --------------------------------------------------------------------------
# Sortino / downside deviation
# --------------------------------------------------------------------------

def test_downside_deviation_divides_by_all_observations():
    vals = [0.01, -0.02, 0.03, -0.01]
    r = daily(vals)
    # squared clipped deviations: 0, 0.0004, 0, 0.0001 — averaged over N = 4,
    # NOT over the 2 negative days.
    expected = math.sqrt((0.0004 + 0.0001) / 4) * SQRT_YEAR
    assert metrics.downside_deviation(r) == pytest.approx(expected, abs=1e-15)
    # the wrong convention (divide by the count of negatives) is sqrt(2) bigger
    wrong = math.sqrt((0.0004 + 0.0001) / 2) * SQRT_YEAR
    assert metrics.downside_deviation(r) != pytest.approx(wrong, abs=1e-9)


def test_sortino_mar_zero_hand_computed():
    vals = [0.01, -0.02, 0.03, -0.01]
    r = daily(vals)
    mean = sum(vals) / 4                       # 0.0025
    dd = math.sqrt((0.0004 + 0.0001) / 4) * SQRT_YEAR
    assert metrics.sortino(r) == pytest.approx(mean * YEAR / dd, abs=1e-12)


def test_sortino_with_mar_equal_to_the_risk_free_rate():
    vals = [0.01, -0.02, 0.03, -0.01]
    rf = 0.0002
    r = daily(vals)
    dev = [min(v - rf, 0.0) for v in vals]
    dd = math.sqrt(sum(d * d for d in dev) / 4) * SQRT_YEAR
    mean = sum(v - rf for v in vals) / 4
    assert metrics.sortino(r, rf) == pytest.approx(mean * YEAR / dd, abs=1e-12)
    # the two MAR conventions are reported side by side, and they differ
    out = metrics.summary(r, rf)
    assert out["sortino"] != pytest.approx(out["sortino_rf"], abs=1e-9)


def test_sortino_is_nan_never_inf_when_there_is_no_downside_at_all():
    r = daily([0.01, 0.02, 0.005, 0.003])
    assert metrics.downside_deviation(r) == 0.0
    # empyrical and quantstats return +inf here. This module does not: a
    # window that never fell below the MAR has FAILED TO OBSERVE downside,
    # which is not the same as having none, and json.dumps writes inf as the
    # bare token `Infinity` that JSON.parse rejects.
    assert math.isnan(metrics.sortino(r))
    flat = daily([0.0, 0.0, 0.0])
    assert math.isnan(metrics.sortino(flat, 0.0))


def test_no_metric_in_the_module_can_emit_an_infinity():
    """+inf reaching json.dumps writes the bare token `Infinity`, which
    JSON.parse rejects with a SyntaxError — the whole payload is lost, not one
    cell. This walks the two inputs that used to produce one (all-positive
    returns -> sortino; a benchmark regressed on itself -> perfect fit, so the
    OLS t-statistics were 0/0) and proves the serialized text is clean."""
    all_positive = daily([0.01, 0.02, 0.005, 0.003])
    perfect = daily([0.01, -0.02, 0.03, 0.004, -0.011, 0.007])

    fit = metrics.alpha_beta(perfect, perfect, rf_daily=0.0)
    assert fit["beta"] == pytest.approx(1.0, abs=1e-12)
    assert fit["beta_stderr"] == 0.0            # the measured fact, reported
    assert math.isnan(fit["beta_tstat"]) and math.isnan(fit["alpha_tstat"])

    for r, b in ((all_positive, None), (perfect, perfect)):
        out = metrics.summary(r, 0.0, b)
        bad = {k: v for k, v in out.items()
               if isinstance(v, float) and math.isinf(v)}
        assert bad == {}, f"infinite values leaked: {bad}"
        assert "Infinity" not in json.dumps(out)


# --------------------------------------------------------------------------
# Calmar / Ulcer / Martin
# --------------------------------------------------------------------------

def test_calmar_is_cagr_over_abs_max_drawdown():
    vals = [0.10, -0.20, 0.05]
    r = daily(vals)
    growth = 1.10 * 0.80 * 1.05
    cagr = growth ** (YEAR / 3) - 1
    mdd = 1.10 * 0.80 / 1.10 - 1.0             # trough 0.88 against peak 1.10
    assert metrics.max_drawdown(r) == pytest.approx(mdd, abs=1e-12)
    assert metrics.calmar(r) == pytest.approx(cagr / abs(mdd), abs=1e-9)


def test_calmar_is_nan_when_nothing_ever_drew_down():
    assert math.isnan(metrics.calmar(daily([0.01, 0.02, 0.03])))


def test_ulcer_index_and_martin_ratio_hand_computed():
    vals = [0.10, -0.20, 0.05]
    r = daily(vals)
    eq = [1.10, 0.88, 0.924]
    peaks = [1.10, 1.10, 1.10]
    dd = [e / p - 1.0 for e, p in zip(eq, peaks)]
    ui = math.sqrt(sum(d * d for d in dd) / (3 - 1))
    assert metrics.ulcer_index(r) == pytest.approx(ui, abs=1e-12)

    rf = 0.0003
    growth = 1.10 * 0.80 * 1.05
    cagr = growth ** (YEAR / 3) - 1
    rf_annual = (1 + rf) ** YEAR - 1
    assert metrics.martin_ratio(r, rf) == pytest.approx(
        (cagr - rf_annual) / ui, abs=1e-9)


# --------------------------------------------------------------------------
# drawdown analytics
# --------------------------------------------------------------------------

def test_drawdown_episode_dates_durations_and_recovery():
    # up, up, down, down, up, up(back above the peak), flat
    r = daily([0.10, 0.10, -0.20, -0.10, 0.20, 0.30, 0.0])
    idx = r.index
    eps = metrics.drawdown_episodes(r)
    assert len(eps) == 1
    e = eps[0]
    # peak is session 1 (equity 1.21); trough is session 3 (1.21*0.8*0.9)
    assert e["peak_date"] == str(idx[1].date())
    assert e["trough_date"] == str(idx[3].date())
    depth = (1.21 * 0.80 * 0.90) / 1.21 - 1.0
    assert e["depth"] == pytest.approx(depth, abs=1e-12)
    # equity: 0.8712, 1.04544 (still under 1.21), 1.359072 -> recovered at 5
    assert e["recovery_date"] == str(idx[5].date())
    assert e["recovered"] is True
    assert e["peak_to_trough_days"] == 2      # sessions 1 -> 3
    assert e["recovery_days"] == 2            # sessions 3 -> 5
    assert e["duration_days"] == 4            # sessions 1 -> 5
    assert e["underwater_days"] == 3          # sessions 2, 3, 4


def test_drawdown_from_inception_reports_no_peak_date():
    # straight down from the funding line: the peak IS the funding point, and
    # inventing a date for it would be a lie.
    r = daily([-0.05, -0.05, 0.02])
    e = metrics.drawdown_episodes(r)[0]
    assert e["peak_date"] is None
    assert e["peak_to_trough_days"] == 2
    assert e["recovered"] is False


def test_a_drawdown_that_never_recovers():
    r = daily([0.10, -0.30, 0.01, 0.01])
    e = metrics.drawdown_episodes(r)[0]
    assert e["recovery_date"] is None
    assert e["recovery_days"] is None
    assert e["recovered"] is False
    # days-so-far runs from the peak (session 0) to the last session (3)
    assert e["duration_days"] == 3
    out = metrics.summary(r)
    assert out["currently_underwater"] is True
    assert out["max_drawdown_recovered"] is False
    assert out["max_drawdown_recovery_date"] is None
    assert out["current_drawdown"] == pytest.approx(
        1.10 * 0.70 * 1.01 * 1.01 / 1.10 - 1.0, abs=1e-12)


def test_top_five_table_is_ordered_deepest_first_and_capped():
    # six separate dips of increasing depth, each fully recovered
    vals: list[float] = []
    for pct in (0.02, 0.04, 0.06, 0.08, 0.10, 0.12):
        vals += [-pct, pct / (1 - pct) + 1e-6, 0.0]
    r = daily(vals)
    table = metrics.drawdown_table(r, top=5)
    assert len(table) == 5
    depths = [e["depth"] for e in table]
    assert depths == sorted(depths)                 # most negative first
    assert depths[0] == pytest.approx(-0.12, abs=1e-9)
    assert len(metrics.drawdown_episodes(r)) == 6   # the table only shows 5


def test_time_under_water_is_the_share_of_sessions_below_the_prior_peak():
    r = daily([0.10, -0.20, 0.30, 0.0])
    # equity 1.10, 0.88, 1.144, 1.144 -> underwater on sessions 1 only
    assert metrics.time_under_water(r) == pytest.approx(1 / 4, abs=1e-12)
    out = metrics.summary(r)
    assert out["longest_drawdown_days"] == 2        # peak 0 -> recovery 2


# --------------------------------------------------------------------------
# distribution: skew, excess kurtosis, tail ratio, VaR / CVaR
# --------------------------------------------------------------------------

def test_skew_and_excess_kurtosis_against_the_textbook_formulas():
    vals = [0.01, -0.02, 0.03, -0.01, 0.005, 0.012, -0.004, 0.02]
    r = daily(vals)
    n = len(vals)
    mean = sum(vals) / n
    m2 = sum((v - mean) ** 2 for v in vals) / n
    m3 = sum((v - mean) ** 3 for v in vals) / n
    m4 = sum((v - mean) ** 4 for v in vals) / n
    # bias-corrected Fisher-Pearson (what pandas .skew() returns)
    g1 = m3 / m2 ** 1.5
    G1 = math.sqrt(n * (n - 1)) / (n - 2) * g1
    assert metrics.skew(r) == pytest.approx(G1, abs=1e-12)
    # sample EXCESS kurtosis (pandas .kurtosis(); normal = 0)
    g2 = m4 / m2 ** 2 - 3.0
    G2 = ((n + 1) * g2 + 6) * (n - 1) / ((n - 2) * (n - 3))
    assert metrics.excess_kurtosis(r) == pytest.approx(G2, abs=1e-12)


def test_skew_and_kurtosis_are_the_bias_corrected_estimator_not_pyfolios():
    """pyfolio's SIMPLE_STAT_FUNCS call scipy at its default bias=True (g1,
    g2); this module reports the bias-CORRECTED G1/G2. Keeping the better
    estimator is the right call, but a citation to a source that returns a
    different number is worse than no citation, so the divergence is pinned
    here and stated in the docstrings. Both sides are computed from raw
    moments in the test — neither is the function under test."""
    rng = np.random.default_rng(404)
    vals = rng.standard_t(4, 40)
    r = pd.Series(vals, index=pd.bdate_range("2015-01-01", periods=40))
    n = 40
    mean = float(np.mean(vals))
    m2 = float(np.mean((vals - mean) ** 2))
    m3 = float(np.mean((vals - mean) ** 3))
    m4 = float(np.mean((vals - mean) ** 4))
    g1 = m3 / m2 ** 1.5                       # scipy/pyfolio, bias=True
    g2 = m4 / m2 ** 2 - 3.0
    G1 = math.sqrt(n * (n - 1)) / (n - 2) * g1
    G2 = ((n + 1) * g2 + 6) * (n - 1) / ((n - 2) * (n - 3))

    assert metrics.skew(r) == pytest.approx(G1, abs=1e-12)
    assert metrics.excess_kurtosis(r) == pytest.approx(G2, abs=1e-12)
    # and they are NOT the pyfolio figures — >10% apart on the kurtosis at
    # the window lengths this product reports (OOS splits, regime segments)
    assert abs(metrics.excess_kurtosis(r) / g2 - 1.0) > 0.10
    assert metrics.skew(r) != pytest.approx(g1, rel=1e-3)


def test_tail_ratio_and_var_cvar_on_a_hand_ordered_sample():
    # 20 sorted returns: the percentile positions are exact rationals
    vals = [-0.10, -0.08, -0.06, -0.05, -0.04, -0.03, -0.02, -0.01, -0.005,
            0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
    r = daily(vals)
    n = len(vals)

    # numpy's linear interpolation: pos = q*(n-1)
    pos5 = 0.05 * (n - 1)                       # 0.95 -> between vals[0], [1]
    var95 = vals[0] + (pos5 - 0) * (vals[1] - vals[0])
    assert metrics.value_at_risk(r, 0.95) == pytest.approx(var95, abs=1e-15)

    pos1 = 0.01 * (n - 1)                       # 0.19
    var99 = vals[0] + pos1 * (vals[1] - vals[0])
    assert metrics.value_at_risk(r, 0.99) == pytest.approx(var99, abs=1e-15)

    tail = [v for v in vals if v <= var95]      # only -0.10
    assert metrics.conditional_value_at_risk(r, 0.95) == pytest.approx(
        sum(tail) / len(tail), abs=1e-15)

    pos95 = 0.95 * (n - 1)                      # 18.05 -> vals[18], vals[19]
    p95 = vals[18] + (pos95 - 18) * (vals[19] - vals[18])
    assert metrics.tail_ratio(r) == pytest.approx(abs(p95) / abs(var95), abs=1e-12)

    # CVaR is never above VaR, and both are negative here
    out = metrics.summary(r)
    assert out["cvar_95"] <= out["var_95"] < 0


# --------------------------------------------------------------------------
# multi-session tail risk: the ~1.7x wrong-numbers finding
# --------------------------------------------------------------------------

def _simulated_horizon_tail(rng, mu, sd, horizon, level, paths=60_000):
    """The TRUE horizon quantile of an IID N(mu, sd) process, by brute-force
    simulation of compounded `horizon`-day paths. Nothing in here touches
    metrics.py — this is the independent value the module is pinned against."""
    sim = rng.normal(mu, sd, size=(paths, horizon))
    total = np.prod(1.0 + sim, axis=1) - 1.0
    var = float(np.percentile(total, (1.0 - level) * 100.0))
    return var, float(total[total <= var].mean())


def test_horizon_var_matches_a_simulated_truth_that_sqrt_scaling_misses():
    """`var_95_annualized` used to be the daily 95% VaR times sqrt(252). A
    quantile is (drift + z*dispersion) and only the dispersion term scales by
    sqrt of time, so the drift got sqrt(H) where it needed H. On a process
    that is EXACTLY IID normal — the one case the old docstring claimed was
    exact — that made the reported one-year VaR 1.7x too deep.

    This pins the replacement against a 60k-path simulation of the same
    process, and asserts the tolerance is tight enough that the sqrt-scaled
    figure would blow straight through it."""
    rng = np.random.default_rng(20240917)
    mu, sd = 0.0005, 0.011
    true_21, true_cvar_21 = _simulated_horizon_tail(rng, mu, sd, 21, 0.95)
    true_252, _ = _simulated_horizon_tail(rng, mu, sd, YEAR, 0.95)

    r = pd.Series(rng.normal(mu, sd, 12_000),
                  index=pd.bdate_range("1980-01-02", periods=12_000))

    # one-month horizon: the default, and accurate to well under a point
    assert metrics.horizon_value_at_risk(r, 0.95, 21) == pytest.approx(
        true_21, abs=0.004)
    assert metrics.horizon_conditional_value_at_risk(r, 0.95, 21) == \
        pytest.approx(true_cvar_21, abs=0.006)

    # one-year horizon, on a sample long enough to earn it (47 independent
    # years). Noisier, hence the wider band — but still nowhere near the bug.
    assert metrics.horizon_value_at_risk(r, 0.95, YEAR) == pytest.approx(
        true_252, abs=0.035)

    # the mutation this test exists to kill: restore sqrt-of-time scaling of
    # the daily quantile and both assertions above fail by a wide margin.
    sqrt_scaled_21 = metrics.value_at_risk(r, 0.95) * math.sqrt(21)
    sqrt_scaled_252 = metrics.value_at_risk(r, 0.95) * SQRT_YEAR
    assert abs(sqrt_scaled_21 - true_21) > 0.008         # vs a 0.004 tolerance
    assert abs(sqrt_scaled_252 - true_252) > 0.10        # vs 0.035
    assert sqrt_scaled_252 / true_252 > 1.5              # the reported ~1.68x


def test_horizon_var_refuses_a_quantile_the_sample_cannot_support():
    """A 95% quantile needs at least two independent observations past the
    cutoff. Five years of data hold five independent years, and the one-year
    empirical VaR computed off them reads about -10% against a truth near
    -16% — so it is refused rather than printed."""
    rng = np.random.default_rng(5150)
    r = pd.Series(rng.normal(0.0005, 0.011, 1260),
                  index=pd.bdate_range("2015-01-01", periods=1260))
    assert metrics.independent_windows(r, YEAR) == 5
    assert math.isnan(metrics.horizon_value_at_risk(r, 0.95, YEAR))
    assert math.isnan(metrics.horizon_conditional_value_at_risk(r, 0.95, YEAR))
    # ...while the one-month horizon on the same sample has 60 independent
    # windows and is reported.
    assert metrics.independent_windows(r, 21) == 60
    assert np.isfinite(metrics.horizon_value_at_risk(r, 0.95, 21))
    # and a 99% quantile needs 200 independent windows, which 60 is not
    assert math.isnan(metrics.horizon_value_at_risk(r, 0.99, 21))


def test_the_wrong_annualized_tail_keys_are_gone_not_silently_corrected():
    rng = np.random.default_rng(31)
    r = pd.Series(rng.normal(0.0005, 0.011, 1000),
                  index=pd.bdate_range("2015-01-01", periods=1000))
    out = metrics.summary(r)
    for key in ("var_95_annualized", "var_99_annualized",
                "cvar_95_annualized", "cvar_99_annualized"):
        assert key not in out, f"{key} must not come back under its old name"
    assert out["tail_horizon_days"] == conventions.TRADING_DAYS_PER_MONTH
    assert out["tail_horizon_windows"] == 1000 // conventions.TRADING_DAYS_PER_MONTH
    assert out["var_95_horizon"] < 0 and out["cvar_95_horizon"] <= out["var_95_horizon"]


def test_a_refused_horizon_tail_is_none_in_the_summary_not_a_nan():
    """A refusal is not an undefined statistic, and None is the only non-value
    a strict JSON encoder accepts. `tail_horizon_windows` carries the reason."""
    short = daily([0.01, -0.02, 0.03] * 20)                # 60 sessions
    out = metrics.summary(short)
    assert out["tail_horizon_windows"] == 60 // conventions.TRADING_DAYS_PER_MONTH
    assert out["var_95_horizon"] is None
    assert out["cvar_95_horizon"] is None
    json.dumps({k: out[k] for k in ("var_95_horizon", "cvar_95_horizon",
                                    "tail_horizon_days",
                                    "tail_horizon_windows")}, allow_nan=False)
    # the standalone functions keep the float contract and return NaN
    assert math.isnan(metrics.horizon_value_at_risk(short))


def test_a_refused_horizon_tail_says_why_it_is_missing():
    """The refusal is the COMMON case, not a corner one: 40 independent windows
    at the one-month horizon is 840 sessions, ~3.3 years, and most backtests run
    here are shorter. A bare None reads as "omitted" and lets a report drop the
    row in silence, so the reason travels in the payload — counted from the
    sample, never narrated."""
    month = conventions.TRADING_DAYS_PER_MONTH
    short = daily([0.01, -0.02, 0.03] * 20)                 # 60 sessions
    out = metrics.summary(short)
    assert out["var_95_horizon"] is None and out["cvar_95_horizon"] is None
    reason = out["tail_horizon_unavailable_reason"]
    assert f"95% quantile at a {month}-session horizon" in reason
    assert f"needs 40 independent windows ({40 * month} sessions)" in reason
    assert f"this backtest has {60 // month} (60 sessions)" in reason
    json.dumps({"tail_horizon_unavailable_reason": reason}, allow_nan=False)

    # a sample shorter than one window gets its OWN reason, not that sentence
    tiny = daily([0.01] * 5)
    assert metrics.summary(tiny)["tail_horizon_unavailable_reason"] == (
        f"a {month}-session horizon needs at least {month} sessions; this "
        f"backtest has 5")

    # the field is None EXACTLY when both figures are there — a reason beside a
    # reported number, or a None beside a missing one, would each be a lie.
    for n in (5, 60, 40 * month - 1, 40 * month, 900):
        r = pd.Series(0.001, index=pd.bdate_range("2015-01-01", periods=n))
        s = metrics.summary(r)
        assert (s["tail_horizon_unavailable_reason"] is None) is \
            (s["var_95_horizon"] is not None)
    earned = metrics.summary(
        pd.Series(0.001, index=pd.bdate_range("2015-01-01", periods=40 * month)))
    assert earned["tail_horizon_unavailable_reason"] is None
    assert earned["tail_horizon_windows"] == 40


def test_horizon_windows_compound_geometrically():
    r = daily([0.10, -0.20, 0.05, 0.02, -0.01])
    w = metrics.horizon_returns(r, 3)
    assert len(w) == 3
    assert float(w.iloc[0]) == pytest.approx(1.10 * 0.80 * 1.05 - 1, abs=1e-12)
    assert float(w.iloc[1]) == pytest.approx(0.80 * 1.05 * 1.02 - 1, abs=1e-12)
    assert float(w.iloc[2]) == pytest.approx(1.05 * 1.02 * 0.99 - 1, abs=1e-12)
    assert w.index[0] == r.index[2]         # stamped on the closing session
    assert len(metrics.horizon_returns(r, 9)) == 0


def test_best_and_worst_periods_carry_their_dates():
    r = daily([0.01, -0.05, 0.03], start="2021-01-04")
    out = metrics.period_extremes(r)
    assert out["best_day"] == pytest.approx(0.03)
    assert out["best_day_date"] == "2021-01-06"
    assert out["worst_day"] == pytest.approx(-0.05)
    assert out["worst_day_date"] == "2021-01-05"
    # one calendar month in the sample, so best month == worst month == it
    assert out["months"] == 1
    assert out["best_month"] == pytest.approx(1.01 * 0.95 * 1.03 - 1, abs=1e-12)
    assert out["best_month_date"] == "2021-01"
    # three sessions is not a complete month
    assert out["complete_months"] == 0


# --------------------------------------------------------------------------
# alpha / beta by OLS
# --------------------------------------------------------------------------

def _ols_fixture():
    """y = alpha + beta*x + e with residuals orthogonal to x and summing to
    zero, so the OLS estimates equal the true parameters EXACTLY and every
    downstream statistic is exact hand arithmetic."""
    x = [-0.02, -0.01, 0.0, 0.01, 0.02]
    e = [0.001, -0.001, 0.0, -0.001, 0.001]      # sum(e) = 0, sum(e*x) = 0
    alpha, beta = 0.0004, 1.25
    y = [alpha + beta * xi + ei for xi, ei in zip(x, e)]
    return x, y, e, alpha, beta


def test_alpha_beta_ols_recovers_the_generating_parameters_with_inference():
    x, y, e, alpha, beta = _ols_fixture()
    b = daily(x)
    r = daily(y)
    out = metrics.alpha_beta(r, b, rf_daily=0.0)

    n = 5
    assert out["n_obs"] == n
    assert out["beta"] == pytest.approx(beta, abs=1e-12)
    assert out["alpha_daily"] == pytest.approx(alpha, abs=1e-12)
    assert out["alpha_annual"] == pytest.approx((1 + alpha) ** YEAR - 1, abs=1e-9)

    sxx = sum(xi ** 2 for xi in x)               # x is centred already
    ssr = sum(ei ** 2 for ei in e)
    s2 = ssr / (n - 2)
    se_beta = math.sqrt(s2 / sxx)
    se_alpha = math.sqrt(s2 * (1 / n + 0.0 ** 2 / sxx))
    assert out["beta_stderr"] == pytest.approx(se_beta, abs=1e-12)
    assert out["alpha_stderr"] == pytest.approx(se_alpha, abs=1e-12)
    assert out["beta_tstat"] == pytest.approx(beta / se_beta, abs=1e-9)
    assert out["alpha_tstat"] == pytest.approx(alpha / se_alpha, abs=1e-9)

    y_bar = sum(y) / n
    sst = sum((yi - y_bar) ** 2 for yi in y)
    assert out["r_squared"] == pytest.approx(1 - ssr / sst, abs=1e-12)


def test_beta_of_the_benchmark_against_itself_is_exactly_one():
    b = daily([0.01, -0.02, 0.03, 0.004, -0.011, 0.007])
    out = metrics.alpha_beta(b, b, rf_daily=0.0001)
    assert out["beta"] == pytest.approx(1.0, abs=1e-12)
    assert out["alpha_daily"] == pytest.approx(0.0, abs=1e-15)
    assert out["r_squared"] == pytest.approx(1.0, abs=1e-12)
    assert out["correlation"] == pytest.approx(1.0, abs=1e-12)


def test_beta_uses_excess_returns_so_the_risk_free_rate_matters():
    x, y, _, _, _ = _ols_fixture()
    # a benchmark with a non-zero mean: subtracting rf shifts x and y, which
    # changes the intercept (alpha) even though the slope is unchanged.
    b = daily([xi + 0.001 for xi in x])
    r = daily([yi + 0.001 for yi in y])
    zero_rf = metrics.alpha_beta(r, b, 0.0)
    with_rf = metrics.alpha_beta(r, b, 0.001)
    assert with_rf["beta"] == pytest.approx(zero_rf["beta"], abs=1e-12)
    assert with_rf["alpha_daily"] != pytest.approx(zero_rf["alpha_daily"], abs=1e-9)


def test_alpha_beta_needs_three_overlapping_observations():
    out = metrics.alpha_beta(daily([0.01, 0.02]), daily([0.01, 0.02]))
    assert math.isnan(out["beta"]) and out["n_obs"] == 2


def test_flat_benchmark_identifies_no_beta():
    out = metrics.alpha_beta(daily([0.01, -0.02, 0.03]), daily([0.0, 0.0, 0.0]))
    assert math.isnan(out["beta"])


# --------------------------------------------------------------------------
# tracking error, information ratio, capture
# --------------------------------------------------------------------------

def test_tracking_error_and_information_ratio_hand_computed():
    r_vals = [0.01, -0.02, 0.03, 0.004]
    b_vals = [0.005, -0.01, 0.02, 0.001]
    r, b = daily(r_vals), daily(b_vals)
    active = [a - c for a, c in zip(r_vals, b_vals)]
    mean = sum(active) / 4
    sd = math.sqrt(sum((a - mean) ** 2 for a in active) / 3)
    te = sd * SQRT_YEAR
    assert metrics.tracking_error(r, b) == pytest.approx(te, abs=1e-12)
    assert metrics.information_ratio(r, b) == pytest.approx(
        mean * YEAR / te, abs=1e-12)
    # the algebraically identical form, as a cross-check of the periodicity
    assert metrics.information_ratio(r, b) == pytest.approx(
        mean / sd * SQRT_YEAR, abs=1e-12)


def test_information_ratio_is_nan_when_the_strategy_tracks_perfectly():
    r = daily([0.01, -0.02, 0.03])
    assert math.isnan(metrics.information_ratio(r, r))


def test_capture_ratios_are_monthly_and_geometric():
    # four one-session months: two benchmark-up, two benchmark-down
    idx = pd.to_datetime(["2021-01-04", "2021-02-01", "2021-03-01", "2021-04-01"])
    r = pd.Series([0.04, -0.03, 0.06, -0.01], index=idx)
    b = pd.Series([0.05, -0.05, 0.08, -0.02], index=idx)
    out = metrics.capture_ratios(r, b)
    assert out["capture_frequency"] == "monthly"
    assert "frequency" not in out       # not a property of the whole block
    assert out["up_periods"] == 2 and out["down_periods"] == 2

    def ann(vals):
        return (np.prod([1 + v for v in vals])) ** (12 / len(vals)) - 1

    up = ann([0.04, 0.06]) / ann([0.05, 0.08])
    down = ann([-0.03, -0.01]) / ann([-0.05, -0.02])
    assert out["up_capture"] == pytest.approx(up, abs=1e-12)
    assert out["down_capture"] == pytest.approx(down, abs=1e-12)
    assert out["up_down_capture"] == pytest.approx(up / down, abs=1e-12)


def test_the_capture_zero_denominator_tolerance_is_pinned_in_both_directions():
    """`_ABS_TOL` decides when a capture denominator counts as zero. Nothing in
    the suite touched it: raising it to 1e-3 or dropping it to 1e-30 both left
    all 66 tests green, so the constant was free. Both sides are pinned here.
    """
    assert metrics._ABS_TOL == 1e-12
    idx = pd.to_datetime(["2021-01-04", "2021-02-01"])
    r = pd.Series([0.01, 0.01], index=idx)

    # (a) a tiny but REAL benchmark: the annualized up-month denominator is
    #     6.0e-4 — above the tolerance, so it must be divided by, not skipped.
    b = pd.Series([5e-5, 5e-5], index=idx)
    denom = (1.00005 ** 2) ** 6 - 1.0
    assert metrics._ABS_TOL < denom < 1e-3
    out = metrics.capture_ratios(r, b)
    assert out["up_capture"] == pytest.approx(((1.01 ** 2) ** 6 - 1.0) / denom,
                                              rel=1e-9)

    # (b) a benchmark that is float dust: the denominator is 1.2e-13, the ratio
    #     would read ~1e11, and that number measures nothing. Refused.
    dust = pd.Series([1e-14, 1e-14], index=idx)
    dust_denom = ((1.0 + 1e-14) ** 2) ** 6 - 1.0
    assert 0.0 < dust_denom < metrics._ABS_TOL
    assert math.isnan(metrics.capture_ratios(r, dust)["up_capture"])


def test_capture_is_nan_without_enough_down_months():
    idx = pd.to_datetime(["2021-01-04", "2021-02-01", "2021-03-01"])
    r = pd.Series([0.04, 0.03, 0.06], index=idx)
    b = pd.Series([0.05, 0.05, 0.08], index=idx)
    out = metrics.capture_ratios(r, b)
    assert math.isnan(out["down_capture"])      # no bear market is a finding
    assert out["down_periods"] == 0


def test_hit_rate_reports_the_month_count_beside_the_rate():
    idx = pd.to_datetime(["2021-01-04", "2021-01-05", "2021-02-01", "2021-03-01"])
    r = pd.Series([0.02, 0.01, -0.01, 0.03], index=idx)
    b = pd.Series([0.01, 0.005, 0.00, 0.04], index=idx)
    out = metrics.summary(r, 0.0, b)
    # Jan: 1.02*1.01-1 > 1.01*1.005-1 (win); Feb: loss; Mar: loss
    assert out["hit_rate"] == pytest.approx(1 / 3, abs=1e-12)
    assert out["hit_rate_months"] == 3
    assert out["complete_months"] == 0          # every bucket is a stub here


# --------------------------------------------------------------------------
# calendar tables
# --------------------------------------------------------------------------

def test_annual_returns_table_strategy_beside_benchmark():
    idx = pd.to_datetime(["2021-06-01", "2021-07-01", "2022-01-03", "2022-02-01"])
    r = pd.Series([0.10, 0.05, -0.04, 0.02], index=idx)
    b = pd.Series([0.08, 0.02, -0.01, 0.03], index=idx)
    rows = metrics.annual_returns_table(r, b)
    assert [row["year"] for row in rows] == [2021, 2022]
    assert rows[0]["strategy"] == pytest.approx(1.10 * 1.05 - 1, abs=1e-12)
    assert rows[0]["benchmark"] == pytest.approx(1.08 * 1.02 - 1, abs=1e-12)
    assert rows[0]["excess"] == pytest.approx(
        (1.10 * 1.05 - 1) - (1.08 * 1.02 - 1), abs=1e-12)
    assert rows[0]["partial"] is True and rows[0]["days"] == 2
    assert rows[1]["strategy"] == pytest.approx(0.96 * 1.02 - 1, abs=1e-12)


def test_a_month_missing_a_quarter_of_its_sessions_is_not_complete():
    """`complete_months` is the honest denominator behind a monthly hit rate.
    The threshold was TRADING_DAYS_PER_MONTH - 6 = 15, so a month with 15 of
    ~21 sessions — a 29% stub — counted as complete. It is now - 3 = 18: short
    enough to admit a genuine holiday-shortened NYSE month, long enough to
    reject a half month."""
    assert metrics.MIN_COMPLETE_MONTH_DAYS == 18
    assert metrics.MIN_COMPLETE_YEAR_DAYS == YEAR - 15

    stub = pd.Series(0.001, index=pd.bdate_range("2021-03-01", periods=15))
    assert metrics.complete_months(stub) == 0
    holiday_month = pd.Series(0.001,
                              index=pd.bdate_range("2021-03-01", periods=18))
    assert metrics.complete_months(holiday_month) == 1
    # the monthly matrix flags the same stub as partial
    assert metrics.monthly_returns_matrix(stub)[0]["partial"] is True


def test_a_full_calendar_year_fills_exactly_twelve_month_slots():
    """MONTHS_PER_YEAR is both the WIDTH of a heatmap row and the completeness
    test for a year. The capture test pins it from below (an 11-month year
    annualizes to a different number); nothing pinned it from above, so 13 left
    the suite green while handing the frontend a 13-column year with a None in
    the last cell and every year flagged partial."""
    assert metrics.MONTHS_PER_YEAR == 12
    r = pd.Series(0.0005, index=pd.bdate_range("2021-01-04", "2021-12-31"))
    rows = metrics.monthly_returns_matrix(r)
    assert len(rows) == 1
    assert len(rows[0]["months"]) == 12
    assert all(v is not None for v in rows[0]["months"])
    assert rows[0]["partial"] is False


def test_monthly_matrix_is_geometric_and_flags_partial_years():
    idx = pd.to_datetime(["2021-01-04", "2021-01-05", "2021-03-01"])
    r = pd.Series([0.02, 0.03, -0.01], index=idx)
    rows = metrics.monthly_returns_matrix(r)
    assert len(rows) == 1
    row = rows[0]
    assert row["year"] == 2021
    assert row["months"][0] == pytest.approx(1.02 * 1.03 - 1, abs=1e-12)
    assert row["months"][1] is None             # February never traded
    assert row["months"][2] == pytest.approx(-0.01, abs=1e-12)
    assert row["eoy"] == pytest.approx(1.02 * 1.03 * 0.99 - 1, abs=1e-12)
    assert row["partial"] is True


# --------------------------------------------------------------------------
# rolling series
# --------------------------------------------------------------------------

def test_rolling_sharpe_first_point_equals_the_window_sharpe():
    rng = np.random.default_rng(7)
    r = pd.Series(rng.normal(0.0004, 0.01, 300),
                  index=pd.bdate_range("2020-01-01", periods=300))
    roll = metrics.rolling_sharpe(r, window=YEAR, rf_daily=0.0)
    assert len(roll) == 300 - YEAR + 1
    window = r.iloc[:YEAR].to_numpy()
    mean = window.mean()
    sd = math.sqrt(((window - mean) ** 2).sum() / (YEAR - 1))
    assert float(roll.iloc[0]) == pytest.approx(mean / sd * SQRT_YEAR, abs=1e-10)
    assert roll.index[0] == r.index[YEAR - 1]


def test_rolling_beta_first_point_equals_the_window_ols_slope():
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2020-01-01", periods=300)
    b = pd.Series(rng.normal(0.0003, 0.009, 300), index=idx)
    r = pd.Series(0.8 * b.to_numpy() + rng.normal(0, 0.004, 300), index=idx)
    roll = metrics.rolling_beta(r, b, window=YEAR, rf_daily=0.0)
    assert len(roll) == 300 - YEAR + 1
    x = b.iloc[:YEAR].to_numpy()
    y = r.iloc[:YEAR].to_numpy()
    x_bar, y_bar = x.mean(), y.mean()
    slope = ((x - x_bar) * (y - y_bar)).sum() / ((x - x_bar) ** 2).sum()
    assert float(roll.iloc[0]) == pytest.approx(slope, abs=1e-10)


def test_rolling_excess_1y_is_wired_into_the_tearsheet():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2019-01-01", periods=400)
    r = pd.Series(rng.normal(0.0006, 0.01, 400), index=idx)
    b = pd.Series(rng.normal(0.0002, 0.008, 400), index=idx)
    sheet = metrics.tearsheet(r, 0.0, b)
    excess = sheet["series"]["rolling_excess_1y"]
    assert len(excess) == 400 - YEAR + 1
    expected = (float((1 + r.iloc[:YEAR]).prod() - 1)
                - float((1 + b.iloc[:YEAR]).prod() - 1))
    assert float(excess.iloc[0]) == pytest.approx(expected, abs=1e-12)


def test_rolling_series_are_empty_rather_than_faked_on_short_windows():
    r = daily([0.01] * 30)
    assert len(metrics.rolling_sharpe(r, window=YEAR)) == 0
    assert len(metrics.rolling_beta(r, r, window=YEAR)) == 0


# --------------------------------------------------------------------------
# exposure
# --------------------------------------------------------------------------

def test_exposure_stats_from_weights_and_from_an_explicit_cash_series():
    idx = pd.bdate_range("2021-01-04", periods=3)
    w = pd.DataFrame({"A": [0.5, 0.0, 0.25], "B": [0.5, 0.4, 0.25]}, index=idx)
    out = metrics.exposure_stats(weights=w)
    assert out["avg_positions"] == pytest.approx((2 + 1 + 2) / 3, abs=1e-12)
    assert out["max_positions"] == 2
    assert out["avg_exposure"] == pytest.approx((1.0 + 0.4 + 0.5) / 3, abs=1e-12)
    # the AVERAGE CASH WEIGHT — mean(1 - gross) — under its own name
    assert out["avg_cash_weight"] == pytest.approx((0.0 + 0.6 + 0.5) / 3,
                                                  abs=1e-12)
    # ...and TIME in cash is a share of PERIODS: this book holds something on
    # all three days, so it is never in cash.
    assert out["time_in_cash"] == pytest.approx(0.0, abs=1e-12)
    assert out["pct_periods_invested"] == pytest.approx(1.0, abs=1e-12)
    assert out["positions_periods"] == 3 and out["exposure_periods"] == 3

    cash = pd.Series([0.0, 1.0, 0.5], index=idx)
    out2 = metrics.exposure_stats(weights=w, cash=cash)
    assert out2["avg_cash_weight"] == pytest.approx(0.5, abs=1e-12)
    assert out2["time_in_cash"] == pytest.approx(1 / 3, abs=1e-12)   # day 2 only
    assert out2["avg_exposure"] == pytest.approx(0.5, abs=1e-12)
    assert out2["pct_periods_invested"] == pytest.approx(2 / 3, abs=1e-12)
    assert out2["avg_positions"] == pytest.approx(5 / 3, abs=1e-12)


def test_time_in_cash_is_a_share_of_time_not_an_average_weight():
    """The reviewer's counter-example: a book that deploys exactly 50% every
    single day and is NEVER in cash. mean(1 - sum(w)) reads 0.5 here, so the
    old `time_in_cash` would have had a report state that the strategy sat in
    cash half the time when it never did once."""
    idx = pd.bdate_range("2021-01-04", periods=4)
    w = pd.DataFrame({"A": [0.25] * 4, "B": [0.25] * 4}, index=idx)
    out = metrics.exposure_stats(weights=w)
    assert out["avg_exposure"] == pytest.approx(0.5, abs=1e-12)
    assert out["avg_cash_weight"] == pytest.approx(0.5, abs=1e-12)
    assert out["time_in_cash"] == pytest.approx(0.0, abs=1e-12)
    assert out["pct_periods_invested"] == pytest.approx(1.0, abs=1e-12)
    assert out["time_in_cash"] == pytest.approx(
        1.0 - out["pct_periods_invested"], abs=1e-12)

    # ...and the converse: a book that is genuinely flat on half the days but
    # fully deployed on the others has the SAME average cash weight and a
    # completely different time in cash.
    w2 = pd.DataFrame({"A": [1.0, 0.0, 1.0, 0.0]}, index=idx)
    out2 = metrics.exposure_stats(weights=w2)
    assert out2["avg_cash_weight"] == pytest.approx(0.5, abs=1e-12)
    assert out2["time_in_cash"] == pytest.approx(0.5, abs=1e-12)


def test_exposure_uses_gross_weights_so_a_hedged_book_is_not_read_as_cash():
    """Signed weights sum to zero for a market-neutral book, which a signed
    sum reads as 100% cash, 0 positions of exposure. Harmless for today's
    long-only engine; a silent zero the day shorts appear."""
    idx = pd.bdate_range("2021-01-04", periods=2)
    w = pd.DataFrame({"LONG": [0.5, 0.5], "SHORT": [-0.5, -0.5]}, index=idx)
    out = metrics.exposure_stats(weights=w)
    assert out["avg_exposure"] == pytest.approx(1.0, abs=1e-12)
    assert out["time_in_cash"] == pytest.approx(0.0, abs=1e-12)
    assert out["avg_positions"] == pytest.approx(2.0, abs=1e-12)


def test_exposure_reports_a_denominator_per_family_not_one_for_both():
    """`weights` is per-REBALANCE and `cash` is per-SESSION, so they are
    legitimately different samples — but the dict used to report a single
    `periods` taken from the cash series while avg_positions came from the
    weight frame, i.e. a 10-row statistic sitting next to periods = 3."""
    w = pd.DataFrame({"A": [1.0] * 10},
                     index=pd.bdate_range("2021-01-04", periods=10))
    cash = pd.Series([0.0, 0.0, 0.0],
                     index=pd.bdate_range("2021-01-04", periods=3))
    out = metrics.exposure_stats(weights=w, cash=cash)
    assert out["positions_periods"] == 10       # avg_positions' denominator
    assert out["exposure_periods"] == 3         # avg_exposure's denominator
    assert out["periods"] == 3                  # back-compat alias
    assert out["avg_positions"] == pytest.approx(1.0, abs=1e-12)


def test_summary_picks_up_exposure_only_when_it_is_given():
    r = daily([0.01, -0.01, 0.02])
    assert "avg_positions" not in metrics.summary(r)
    w = pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=r.index)
    assert metrics.summary(r, 0.0, None, w)["avg_positions"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# degenerate inputs
# --------------------------------------------------------------------------

def test_zero_volatility_series_reports_nan_not_a_giant_sharpe():
    # an all-cash strategy: zeros plus floating-point dust, judged against a
    # non-zero risk-free rate (the confirmed audit reproduction).
    vals = [0.0] * 400
    vals[17] = 1e-12                            # floating-point dust, not a trade
    r = daily(vals)
    # the standard deviation here is ~5e-14: not exactly zero, so the old
    # `sd == 0` guard let it through and printed a Sharpe of -1e17.
    assert float((r - 5e-5).std(ddof=1)) > 0.0
    assert math.isnan(metrics.sharpe(r, 5e-5))
    out = metrics.summary(r, 5e-5)
    assert math.isnan(out["sharpe"])
    assert out["vol"] == pytest.approx(0.0, abs=1e-11)
    assert out["total_return"] == pytest.approx(1e-12, abs=1e-15)
    assert math.isnan(out["calmar"])            # no drawdown ever happened
    assert out["time_under_water"] == pytest.approx(0.0, abs=1e-12)
    assert out["max_drawdown"] == pytest.approx(0.0, abs=1e-15)


def test_rounding_dust_above_float_epsilon_still_reports_nan():
    """The second reproduction of the same defect class. The scale-relative
    tolerance that replaced `sd == 0` was keyed to `excess.abs().mean()`,
    which for an all-cash book is dominated by the RISK-FREE RATE (~5e-5): the
    relative term came to 5e-13, BELOW the 1e-12 absolute floor, so max() always
    picked the floor and the "scale-relative" guard was a fixed constant
    relative to the wrong quantity. 400 sessions of exact zeros with N(0, 1e-9)
    dust every 7th day cleared it and printed a Sharpe of -2.0e6.

    The guard is now an absolute floor in the units returns are expressed in,
    so it does not care what the risk-free rate happens to be."""
    rng = np.random.default_rng(20240917)
    vals = np.zeros(400)
    dust_days = range(0, 400, 7)
    vals[list(dust_days)] = rng.normal(0.0, 1e-9, len(list(dust_days)))
    r = pd.Series(vals, index=pd.bdate_range("2021-01-04", periods=400))

    sd = float((r - 5e-5).std(ddof=1))
    assert 1e-12 < sd < 1e-8, f"the counter-example needs dust above 1e-12, got {sd}"
    assert math.isnan(metrics.sharpe(r, 5e-5))
    assert math.isnan(metrics.summary(r, 5e-5)["sharpe"])
    assert metrics.annual_vol(r) < 1e-7


def test_dust_is_refused_even_when_its_ratio_looks_perfectly_reasonable():
    """The case a plausibility check on the RATIO cannot catch. This series is
    pure float noise — an annualized volatility of 1.6e-8 — but its Sharpe
    works out to about 1.6, which no "is this ratio absurd?" test would ever
    flag. Only a floor on the standard deviation itself refuses it, so this is
    what pins _MIN_DAILY_SD rather than the ratio guard standing behind it."""
    rng = np.random.default_rng(60613)
    vals = rng.normal(1e-10, 1e-9, 500)
    r = pd.Series(vals, index=pd.bdate_range("2015-01-01", periods=500))
    naive = float(np.mean(vals)) / float(np.std(vals, ddof=1)) * SQRT_YEAR
    assert 0.5 < abs(naive) < 20.0, f"ratio must look innocuous, got {naive}"
    assert metrics.annual_vol(r) < 1e-7
    assert math.isnan(metrics.sharpe(r))
    assert math.isnan(metrics.information_ratio(r, r * 0.0))


def test_an_absurd_ratio_on_a_no_risk_series_is_refused_above_the_floor():
    """And the converse case, which the floor alone cannot catch: a standard
    deviation an order of magnitude ABOVE the floor, carrying a Sharpe of 30
    on an annualized volatility of 0.0016%. Both guards are load-bearing."""
    rng = np.random.default_rng(1848)
    vals = rng.normal(2e-6, 1e-6, 400)
    r = pd.Series(vals, index=pd.bdate_range("2015-01-01", periods=400))
    sd = float(np.std(vals, ddof=1))
    assert sd > 10 * 1e-7, "this case must clear the standard-deviation floor"
    assert abs(float(np.mean(vals)) / sd * SQRT_YEAR) > 20.0
    assert metrics.annual_vol(r) < 5e-3
    assert math.isnan(metrics.sharpe(r))


def test_the_degeneracy_guard_does_not_fire_on_a_quiet_but_real_book():
    """A guard that swallows real low-volatility strategies is as bad as one
    that lets dust through. A 0.6%/yr book — quieter than any money-market
    fund — must still get a number, checked against hand arithmetic."""
    rng = np.random.default_rng(77)
    daily_sd = 0.006 / SQRT_YEAR                       # 0.6% annualized
    vals = rng.normal(0.00002, daily_sd, 500)
    r = pd.Series(vals, index=pd.bdate_range("2015-01-01", periods=500))
    mean = float(np.mean(vals))
    sd = math.sqrt(float(np.sum((vals - mean) ** 2)) / 499)
    assert metrics.sharpe(r) == pytest.approx(mean / sd * SQRT_YEAR, abs=1e-10)
    assert np.isfinite(metrics.sharpe(r))
    assert metrics.annual_vol(r) == pytest.approx(0.006, rel=0.10)


def test_the_degeneracy_guard_thresholds_sit_where_they_are_documented():
    """Both guards are TUNED magnitudes the module argues for at length, and
    every behavioural test above pins them from ONE side only: the dust cases
    still refuse if the floor is RAISED to 1e-5, and the quiet 0.6%/yr book
    still gets its number if _ABSURD_RATIO is dropped to 2 or
    _NEGLIGIBLE_ANNUAL_VOL is raised to 5e-2. Each boundary is pinned here in
    both directions, on the two predicates the whole module routes through."""
    assert metrics._MIN_DAILY_SD == 1e-7
    assert metrics._degenerate(0.99e-7) is True
    assert metrics._degenerate(1e-7) is True            # at the floor is dust
    assert metrics._degenerate(1.01e-7) is False        # just above it is not
    assert metrics._degenerate(float("nan")) is True

    assert metrics._ABSURD_RATIO == 20.0
    assert metrics._NEGLIGIBLE_ANNUAL_VOL == 5e-3
    # a daily sd of 1e-4 is 0.16%/yr — inside the "essentially no risk" band,
    # where the ratio guard is what decides
    quiet = 1e-4
    assert quiet * SQRT_YEAR < metrics._NEGLIGIBLE_ANNUAL_VOL
    assert metrics._implausible(19.9, quiet) is False
    assert metrics._implausible(20.1, quiet) is True
    assert metrics._implausible(float("nan"), quiet) is True
    # ...and 6e-4 daily is 0.95%/yr, OUTSIDE that band: a book with real
    # volatility keeps its number however large the ratio, because there the
    # ratio is a measurement rather than an artefact of dividing by dust.
    loud = 6e-4
    assert loud * SQRT_YEAR > metrics._NEGLIGIBLE_ANNUAL_VOL
    assert metrics._implausible(1000.0, loud) is False


def test_single_observation_degrades_to_nan_without_raising():
    r = daily([0.01])
    out = metrics.summary(r, 0.0)
    assert out["days"] == 1
    assert math.isnan(out["vol"]) and math.isnan(out["sharpe"])
    assert math.isnan(out["skew"]) and math.isnan(out["excess_kurtosis"])
    assert math.isnan(out["ulcer_index"])
    assert out["total_return"] == pytest.approx(0.01, abs=1e-15)
    assert out["best_day"] == pytest.approx(0.01, abs=1e-15)
    assert metrics.drawdown_episodes(r) == []


def test_empty_series_is_survivable():
    empty = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    out = metrics.summary(empty)
    assert out["days"] == 0
    assert math.isnan(out["cagr"])
    assert metrics.monthly_returns_matrix(empty) == []
    assert metrics.annual_returns_table(empty) == []


# --------------------------------------------------------------------------
# tearsheet shape
# --------------------------------------------------------------------------

def test_tearsheet_returns_scalars_tables_and_chartable_series():
    rng = np.random.default_rng(19)
    idx = pd.bdate_range("2018-01-01", periods=700)
    r = pd.Series(rng.normal(0.0005, 0.011, 700), index=idx)
    b = pd.Series(rng.normal(0.0003, 0.009, 700), index=idx)
    w = pd.DataFrame({"A": [0.5] * 700, "B": [0.5] * 700}, index=idx)
    sheet = metrics.tearsheet(r, 0.00005, b, weights=w)

    assert set(sheet) == {"summary", "drawdowns", "annual_returns",
                          "monthly_returns", "exposure", "series",
                          "rolling_window"}
    assert sheet["summary"]["days"] == 700
    assert len(sheet["drawdowns"]) <= 5
    assert all(isinstance(v, pd.Series) for v in sheet["series"].values())
    assert set(sheet["series"]) == {"equity_curve", "drawdown", "rolling_sharpe",
                                    "rolling_excess_1y", "rolling_beta"}
    assert sheet["exposure"]["avg_positions"] == pytest.approx(2.0)
    assert sheet["rolling_window"] == YEAR
    # tables are JSON-safe: no pandas objects left in them
    for row in sheet["annual_returns"] + sheet["monthly_returns"] + sheet["drawdowns"]:
        for value in row.values():
            assert value is None or isinstance(value, (int, float, str, bool, list))


def test_tearsheet_without_a_benchmark_omits_the_relative_series():
    r = daily([0.01, -0.01, 0.02] * 100)
    sheet = metrics.tearsheet(r)
    assert "rolling_beta" not in sheet["series"]
    assert "rolling_excess_1y" not in sheet["series"]
    assert sheet["annual_returns"][0]["benchmark"] is None
