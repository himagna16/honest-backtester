"""Tests for the inference layer (backend/audit/significance.py).

The load-bearing tests here are the PUBLISHED WORKED EXAMPLES: Bailey &
López de Prado's own DSR numbers (N=100 → 0.9004, N=46 → 0.9505) and the
MinBTL calibration from the pseudo-mathematics paper (N=45 → 5 years). If a
refactor breaks a unit conversion, those fail loudly instead of silently
returning a plausible-looking wrong probability.

The rest are Monte-Carlo sanity checks with fixed seeds (PSR of pure noise
≈ 0.5, bootstrap coverage ≈ nominal) and the invariants the house rules
demand: units are unmistakable, randomness is seeded, NaN is refused.
"""
import json
import math
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from audit import significance as sg
from dsl.conventions import TRADING_DAYS_PER_YEAR
from engine import metrics

DAILY_VOL = 0.01
#: Imported, never restated: dsl/conventions.py is the single source of truth
#: for the annualization constant. A test that spells "252.0" in its own
#: assertion cannot detect a desync — it just moves with whatever the module
#: happens to use.
PERIODS = float(TRADING_DAYS_PER_YEAR)


def make_returns(n: int, *, seed: int, mu: float = 0.0,
                 vol: float = DAILY_VOL) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-02", periods=n)
    return pd.Series(rng.standard_normal(n) * vol + mu, index=idx)


# ------------------------------------------------------ normal distribution

@pytest.mark.parametrize("p,expected", [
    (0.5, 0.0),
    (0.975, 1.9599639845400545),
    (0.99, 2.3263478740408408),
    (0.025, -1.9599639845400545),
    (1e-10, -6.361340902404056),
    (1 - 1e-10, 6.361340902404056),
])
def test_norm_ppf_matches_published_quantiles(p, expected):
    # Relative tolerance: 10 significant figures is all a double-precision p
    # carries in the far tails (1 - 1e-10 is not exactly representable), and
    # every use here is at |z| < 4 where the absolute error is ~1e-16.
    assert sg.norm_ppf(p) == pytest.approx(expected, rel=1e-8, abs=1e-12)


def test_norm_cdf_ppf_roundtrip():
    for p in (0.001, 0.05, 0.3, 0.5, 0.87, 0.999):
        assert sg.norm_cdf(sg.norm_ppf(p)) == pytest.approx(p, abs=1e-12)


def test_norm_ppf_rejects_out_of_range():
    with pytest.raises(ValueError):
        sg.norm_ppf(0.0)
    with pytest.raises(ValueError):
        sg.norm_ppf(1.0)


# ------------------------------------------------------------------ moments

def test_skew_and_kurtosis_match_pandas_unbiased_estimators():
    """pandas uses the same adjusted Fisher-Pearson estimators as
    scipy(bias=False); this is an independent implementation cross-check."""
    rng = np.random.default_rng(7)
    x = rng.standard_t(5, size=400)
    s = pd.Series(x)
    assert sg.sample_skew(x) == pytest.approx(float(s.skew()), rel=1e-12)
    # RAW kurtosis, not excess — the PSR/DSR formulas want 3.0 for a normal.
    assert sg.sample_kurtosis_raw(x) == pytest.approx(float(s.kurt()) + 3.0, rel=1e-12)


def test_kurtosis_of_normal_sample_is_about_three():
    x = make_returns(5000, seed=1).to_numpy()
    assert sg.sample_kurtosis_raw(x) == pytest.approx(3.0, abs=0.2)
    assert abs(sg.sample_skew(x)) < 0.15


def test_moments_refuse_flat_series():
    flat = pd.Series([0.0] * 100, index=pd.bdate_range("2020-01-01", periods=100))
    with pytest.raises(ValueError, match="zero dispersion"):
        sg.ReturnMoments.from_returns(flat)


def test_degeneracy_policy_is_the_same_one_engine_metrics_uses():
    """An all-cash series carries floating-point dust, and an ``sd == 0`` test
    let it through as a Sharpe of 1.6e8 while ``metrics.sharpe`` returned NaN
    on the identical input.

    metrics.sharpe is the independent oracle here: whenever it refuses, this
    module must refuse too. The dust series below is the reviewer's own
    counter-example (300 days of 1e-9 returns with 1e-16 of jitter).
    """
    idx = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(404)
    dust = pd.Series(1e-9 + rng.standard_normal(300) * 1e-16, index=idx)
    cases = [
        dust,
        pd.Series([0.0] * 300, index=idx),
        pd.Series([0.001] * 300, index=idx),
    ]
    for s in cases:
        assert math.isnan(metrics.sharpe(s)), "oracle changed; revisit this test"
        assert sg.ReturnMoments.try_from_returns(s) is None
        with pytest.raises(ValueError, match="zero dispersion"):
            sg.ReturnMoments.from_returns(s)
    # ...and a real series is still accepted, so the guard is not a blanket no.
    real = make_returns(300, seed=405, mu=0.0004)
    assert sg.ReturnMoments.try_from_returns(real) is not None


def test_assess_sharpe_returns_unavailable_on_a_degenerate_series():
    """A data condition must surface as a finding, not as a 500 from an inline
    pipeline step."""
    flat = pd.Series([0.0] * 300, index=pd.bdate_range("2020-01-01", periods=300))
    out = sg.assess_sharpe(flat, seed=1)
    assert out.available is False
    assert out.moments is None and out.psr is None and out.bootstrap is None
    assert "undefined" in out.interpretation
    d = out.to_dict()
    assert d["available"] is False and d["n_obs"] == 300
    assert json.dumps(d, allow_nan=False)


def test_assess_excess_returns_unavailable_when_the_strategy_tracks_exactly():
    a = make_returns(300, seed=406, mu=0.0004)
    out = sg.assess_excess(a, a.copy(), seed=1, n_resamples=100)
    assert out.available is False
    assert out.verdict_source == "none"
    assert out.verdict == sg.TIES_WITHIN_NOISE      # never a winner
    assert "no dispersion" in out.unavailable_reason
    assert json.dumps(out.to_dict(), allow_nan=False)


def test_assess_excess_returns_unavailable_when_the_series_barely_overlap():
    a = make_returns(300, seed=407, mu=0.0004)
    b = make_returns(300, seed=408, mu=0.0001).iloc[298:]
    out = sg.assess_excess(a, b, seed=1, n_resamples=100)
    assert out.available is False and out.n_obs == 2
    assert json.dumps(out.to_dict(), allow_nan=False)


# ------------------------------------------------------------------- units

def test_sharpe_periodic_annual_conversion_matches_engine_metrics():
    """The whole module is in per-period units; the report is in annualized
    units. If these ever disagree every downstream probability is wrong.

    This is the named guard against a conventions desync, so it must not
    restate the constant: it asserts that ``significance.PERIODS_PER_YEAR`` IS
    ``dsl.conventions.TRADING_DAYS_PER_YEAR``, and scales by the imported
    symbol. The previous version spelled ``math.sqrt(252.0)`` in its own
    assertion, so hardcoding 252 in significance.py — the exact desync the
    module's comment says this guards — passed it.
    """
    r = make_returns(1000, seed=5, mu=0.0004)
    m = sg.ReturnMoments.from_returns(r)
    assert sg.PERIODS_PER_YEAR == PERIODS
    assert m.sharpe_annual == pytest.approx(metrics.sharpe(r), rel=1e-12)
    assert sg.periodic_to_annual(m.sharpe_periodic) == pytest.approx(m.sharpe_annual)
    assert sg.annual_to_periodic(m.sharpe_annual) == pytest.approx(m.sharpe_periodic)
    assert m.sharpe_annual == pytest.approx(m.sharpe_periodic * math.sqrt(PERIODS))


def test_rf_daily_flows_through_like_metrics_sharpe():
    r = make_returns(600, seed=6, mu=0.0005)
    m = sg.ReturnMoments.from_returns(r, rf_daily=0.0001)
    assert m.sharpe_annual == pytest.approx(metrics.sharpe(r, 0.0001), rel=1e-12)


def test_a_risk_free_path_longer_than_the_window_does_not_inflate_n_obs():
    """A pandas-Series rf must be ALIGNED, not index-unioned.

    ``metrics.sharpe`` documents ``rf_daily: float | pd.Series`` and the
    pipeline is meant to pass a daily path. Subtracting it without aligning
    unions the two indexes: a 400-day return series minus a 500-day rf path
    reported n_obs=500 and NaN moments, and PSR then printed "is nan%" with
    valid=True. The pins here are independent of this module: 400 is the length
    of the series the test constructed, and the Sharpe comes from
    engine.metrics.
    """
    r = make_returns(400, seed=301, mu=0.0006)
    rf = pd.Series(0.0001, index=pd.bdate_range("2015-01-02", periods=500))
    m = sg.ReturnMoments.from_returns(r, rf)
    assert m.n_obs == 400
    assert math.isfinite(m.sharpe_annual)
    assert m.sharpe_annual == pytest.approx(metrics.sharpe(r, rf), rel=1e-12)
    # Same series, same rate as a scalar: the path is constant, so the two
    # must agree exactly.
    assert m.sharpe_annual == pytest.approx(
        sg.ReturnMoments.from_returns(r, 0.0001).sharpe_annual, rel=1e-12)
    # And the whole bundled entry point agrees rather than reporting "nan%".
    out = sg.assess_sharpe(r, seed=1, rf_daily=rf, n_resamples=100)
    assert out.available is True
    assert out.n_obs == 400 and out.psr.psr is not None
    assert "nan" not in out.interpretation.lower()


def test_a_risk_free_path_with_holes_is_refused_not_averaged_around():
    r = make_returns(200, seed=302, mu=0.0004)
    rf = pd.Series(0.0001, index=r.index[:-5])       # 5 sessions missing
    with pytest.raises(ValueError):
        sg.ReturnMoments.from_returns(r, rf)


def test_bootstrap_sharpe_ci_aligns_a_series_rf_the_same_way():
    r = make_returns(300, seed=303, mu=0.0007)
    rf = pd.Series(0.0001, index=pd.bdate_range("2015-01-02", periods=450))
    ci = sg.bootstrap_sharpe_ci(r, seed=4, rf_daily=rf, n_resamples=150)
    assert ci.n_obs == 300
    assert ci.point == pytest.approx(metrics.sharpe(r, rf), rel=1e-12)


def test_nan_returns_are_refused_not_skipped():
    r = make_returns(200, seed=8)
    r.iloc[10] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        sg.ReturnMoments.from_returns(r)


# --------------------------------------------------------- Sharpe std error

def test_lo_standard_error_matches_closed_form():
    """Lo (2002): SE = sqrt((1 + SR^2/2)/T) per period, under IID normality."""
    m = sg.ReturnMoments(sharpe_periodic=0.1, n_obs=1000, skew=0.0, kurtosis_raw=3.0)
    se = sg.sharpe_standard_error(m, non_normal=False)
    assert se.se_periodic == pytest.approx(math.sqrt((1 + 0.1 ** 2 / 2) / 1000))
    assert se.se_annual == pytest.approx(se.se_periodic * math.sqrt(PERIODS))
    # For a normal the non-normal (Mertens) form collapses onto Lo's.
    se_nn = sg.sharpe_standard_error(m, non_normal=True)
    assert se_nn.se_periodic == pytest.approx(se.se_periodic, rel=1e-12)


def test_negative_skew_and_fat_tails_widen_the_sharpe_standard_error():
    normal = sg.ReturnMoments(sharpe_periodic=0.1, n_obs=1000, skew=0.0,
                              kurtosis_raw=3.0)
    ugly = sg.ReturnMoments(sharpe_periodic=0.1, n_obs=1000, skew=-1.5,
                            kurtosis_raw=9.0)
    assert (sg.sharpe_standard_error(ugly).se_annual
            > sg.sharpe_standard_error(normal).se_annual)


def test_standard_error_shrinks_with_sample_length():
    short = sg.ReturnMoments(sharpe_periodic=0.08, n_obs=250, skew=0.0, kurtosis_raw=3.0)
    long_ = sg.ReturnMoments(sharpe_periodic=0.08, n_obs=2500, skew=0.0, kurtosis_raw=3.0)
    assert (sg.sharpe_standard_error(long_).se_annual
            == pytest.approx(sg.sharpe_standard_error(short).se_annual / math.sqrt(10),
                             rel=1e-9))


def test_sharpe_difference_test_calls_a_noisy_gap_a_tie():
    """The old overfit detector fired on a 2x ratio of two point estimates.
    3.5y in-sample vs 1.5y holdout cannot resolve a Sharpe gap of ~1."""
    is_m = sg.ReturnMoments(sharpe_periodic=sg.annual_to_periodic(1.4),
                            n_obs=880, skew=0.0, kurtosis_raw=3.0)
    oos_m = sg.ReturnMoments(sharpe_periodic=sg.annual_to_periodic(0.6),
                             n_obs=380, skew=0.0, kurtosis_raw=3.0)
    out = sg.sharpe_difference_test(is_m, oos_m)
    assert out["difference_annual"] == pytest.approx(0.8, abs=1e-9)
    assert out["significant_at_95"] is False
    assert "not distinguishable" in out["interpretation"]


# ------------------------------------------------------------------- PSR

def test_psr_worked_example_from_the_deflated_sharpe_paper():
    """Bailey & López de Prado (2014), the paper's own worked example:
    SR=2.5/sqrt(250) daily, T=1250, g3=-3, g4=10, threshold SR_0=0.1132
    → PSR = DSR ≈ 0.9004."""
    m = sg.ReturnMoments(sharpe_periodic=2.5 / math.sqrt(250), n_obs=1250,
                         skew=-3.0, kurtosis_raw=10.0)
    sr0 = sg.expected_max_sharpe(0.5 / 250.0, 100)
    assert sr0 == pytest.approx(0.1132, abs=5e-5)
    assert sg.probabilistic_sharpe_ratio(m, sr0).psr == pytest.approx(0.9004, abs=5e-4)


def test_psr_of_pure_noise_is_about_one_half():
    """Monte Carlo, fixed seed. Under the null there is no edge, so the
    probability that the true Sharpe exceeds zero should average ~0.5."""
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2015-01-02", periods=500)
    vals = []
    for _ in range(400):
        r = pd.Series(rng.standard_normal(500) * DAILY_VOL, index=idx)
        vals.append(sg.probabilistic_sharpe_ratio(sg.ReturnMoments.from_returns(r)).psr)
    assert float(np.mean(vals)) == pytest.approx(0.5, abs=0.05)


def test_psr_rises_with_sample_length_at_the_same_sharpe():
    lo = sg.ReturnMoments(sharpe_periodic=0.05, n_obs=250, skew=0.0, kurtosis_raw=3.0)
    hi = sg.ReturnMoments(sharpe_periodic=0.05, n_obs=2500, skew=0.0, kurtosis_raw=3.0)
    assert (sg.probabilistic_sharpe_ratio(hi).psr
            > sg.probabilistic_sharpe_ratio(lo).psr)


def test_psr_penalises_negative_skew():
    sym = sg.ReturnMoments(sharpe_periodic=0.08, n_obs=1000, skew=0.0, kurtosis_raw=3.0)
    tail_seller = sg.ReturnMoments(sharpe_periodic=0.08, n_obs=1000, skew=-1.5,
                                   kurtosis_raw=9.0)
    assert (sg.probabilistic_sharpe_ratio(tail_seller).psr
            < sg.probabilistic_sharpe_ratio(sym).psr)


def test_psr_refuses_short_samples_instead_of_guessing():
    m = sg.ReturnMoments(sharpe_periodic=0.3, n_obs=20, skew=0.0, kurtosis_raw=3.0)
    res = sg.probabilistic_sharpe_ratio(m)
    assert res.psr is None
    assert res.sufficient_sample is False
    assert "20" in res.interpretation


def test_psr_flags_a_degenerate_variance_term_rather_than_returning_nan():
    # Extreme positive skew with a large Sharpe drives 1 - g3*SR + ... negative.
    m = sg.ReturnMoments(sharpe_periodic=0.9, n_obs=500, skew=3.0, kurtosis_raw=1.0)
    res = sg.probabilistic_sharpe_ratio(m)
    assert res.psr is None and res.valid is False


# -------------------------------------------------------------------- DSR

def test_expected_max_sharpe_matches_the_papers_threshold():
    assert sg.expected_max_sharpe(0.5 / 250.0, 100) == pytest.approx(0.11317, abs=1e-5)
    assert sg.expected_max_sharpe(0.0, 50) == 0.0


def test_expected_max_sharpe_requires_at_least_two_trials():
    with pytest.raises(ValueError):
        sg.expected_max_sharpe(0.002, 1)


def test_dsr_worked_example_passes_at_46_trials_and_fails_at_100():
    """The paper's headline: the same Sharpe 2.5 strategy clears 95% after 46
    trials and fails after 100. This is the product's whole thesis."""
    m = sg.ReturnMoments(sharpe_periodic=2.5 / math.sqrt(250), n_obs=1250,
                         skew=-3.0, kurtosis_raw=10.0)
    v = 0.5 / 250.0
    at100 = sg.deflated_sharpe_ratio(m, n_trials=100,
                                     variance_trial_sharpes_periodic=v)
    at46 = sg.deflated_sharpe_ratio(m, n_trials=46,
                                    variance_trial_sharpes_periodic=v)
    assert at100.dsr == pytest.approx(0.9004, abs=5e-4)
    assert at46.dsr == pytest.approx(0.9505, abs=5e-4)
    assert at100.dsr < 0.95 <= at46.dsr
    assert "does not clear it" in at100.interpretation
    assert "clears the 95% bar" in at46.interpretation


def test_dsr_falls_monotonically_as_trials_accumulate():
    m = sg.ReturnMoments(sharpe_periodic=0.09, n_obs=1260, skew=-0.2,
                         kurtosis_raw=4.0)
    v = 0.4 / 252.0
    prev = 1.1
    for n in (2, 5, 20, 100, 500):
        d = sg.deflated_sharpe_ratio(m, n_trials=n, variance_trial_sharpes_periodic=v)
        assert d.dsr < prev
        prev = d.dsr


def test_dsr_with_one_trial_degenerates_to_psr_and_says_so():
    m = sg.ReturnMoments(sharpe_periodic=0.09, n_obs=1260, skew=0.0, kurtosis_raw=3.0)
    d = sg.deflated_sharpe_ratio(m, n_trials=1)
    assert d.dsr == pytest.approx(sg.probabilistic_sharpe_ratio(m).psr)
    assert "no search to deflate" in d.interpretation


def test_dsr_refuses_to_invent_a_trial_variance():
    """A bare attempt counter is not enough: DSR needs the spread of the
    trials' Sharpes. Fabricating one would be exactly the failure this
    product exists to catch."""
    m = sg.ReturnMoments(sharpe_periodic=0.09, n_obs=1260, skew=0.0, kurtosis_raw=3.0)
    d = sg.deflated_sharpe_ratio(m, n_trials=40)
    assert d.dsr is None and d.available is False
    assert "not recorded" in d.interpretation


def test_trial_sharpe_variance_converts_annual_ledger_to_periodic():
    annual = [0.5, 1.5, -0.5, 2.0, 0.25]
    v = sg.trial_sharpe_variance_periodic(annual)
    assert v == pytest.approx(float(np.var(annual, ddof=1)) / PERIODS)
    with pytest.raises(ValueError):
        sg.trial_sharpe_variance_periodic([1.0])


def test_dsr_accepts_a_trial_ledger_directly():
    m = sg.ReturnMoments(sharpe_periodic=0.09, n_obs=1260, skew=0.0, kurtosis_raw=3.0)
    ledger = [0.3, 1.1, -0.4, 0.8, 1.6, 0.2]
    d = sg.deflated_sharpe_ratio(m, n_trials=len(ledger), trial_sharpes_annual=ledger)
    assert d.dsr is not None and d.sr0_annual > 0


# ----------------------------------------------------------- MinBTL / MinTRL

def test_min_backtest_length_matches_the_papers_calibration():
    """Bailey, Borwein, López de Prado & Zhu (2014): 45 independent trials is
    the most 5 years of data supports at an in-sample Sharpe of 1; with 7
    trials, 2 years already manufactures Sharpe 1 out of noise."""
    assert sg.min_backtest_length(45, 1.0).min_years_required == pytest.approx(5.0, abs=0.01)
    assert sg.min_backtest_length(7, 1.0).min_years_required == pytest.approx(1.92, abs=0.02)


def test_min_backtest_length_is_below_the_papers_upper_bound():
    for n in (5, 20, 100, 1000):
        years = sg.min_backtest_length(n, 1.0).min_years_required
        assert years < 2.0 * math.log(n)          # Theorem 3.1 bound


def test_min_backtest_length_flags_an_over_searched_window():
    res = sg.min_backtest_length(60, target_sharpe_annual=1.0, observed_years=4.0)
    assert res.sufficient is False
    assert res.expected_lucky_sharpe_annual > 1.0
    assert "needs about" in res.interpretation


def test_expected_lucky_sharpe_scales_with_one_over_sqrt_years():
    a = sg.expected_lucky_sharpe_annual(50, 1.0)
    b = sg.expected_lucky_sharpe_annual(50, 4.0)
    assert b == pytest.approx(a / 2.0, rel=1e-9)


def test_min_track_record_length_inverts_psr_exactly():
    """At T = MinTRL the PSR should sit exactly on the 1-alpha bar."""
    m = sg.ReturnMoments(sharpe_periodic=0.05, n_obs=1000, skew=-0.5, kurtosis_raw=6.0)
    trl = sg.min_track_record_length(m, 0.0, alpha=0.05)
    at_t = sg.ReturnMoments(sharpe_periodic=0.05, n_obs=int(round(trl.n_obs_required)),
                            skew=-0.5, kurtosis_raw=6.0)
    assert sg.probabilistic_sharpe_ratio(at_t).psr == pytest.approx(0.95, abs=1e-3)


def test_min_track_record_length_is_unattainable_below_the_threshold():
    m = sg.ReturnMoments(sharpe_periodic=-0.02, n_obs=1000, skew=0.0, kurtosis_raw=3.0)
    trl = sg.min_track_record_length(m)
    assert trl.attainable is False
    assert trl.years_required is None
    assert trl.years_required_display == "never at this Sharpe"


def test_min_track_record_length_refuses_a_non_positive_variance_term():
    """MinTRL is PSR inverted, so it must refuse where PSR refuses.

    PSR returns ``valid=False`` on a non-positive Mertens variance term. The
    inversion had no such guard: the negative numerator went straight into
    ``1 + numer·(z/ΔSR)²`` and came back as a NEGATIVE required length reported
    with ``attainable=True`` and a display string of "-3.6 years" — the exact
    inconsistency the round-2 fix removed from the standard error and the
    bootstrap.
    """
    m = sg.ReturnMoments(sharpe_periodic=0.9, n_obs=500, skew=3.0, kurtosis_raw=1.0)
    assert sg._psr_variance_numerator(m) < 0.0
    assert sg.probabilistic_sharpe_ratio(m).valid is False       # the oracle
    trl = sg.min_track_record_length(m)
    assert trl.attainable is False
    assert trl.n_obs_required is None and trl.years_required is None
    assert "non-positive" in trl.interpretation
    assert "year" not in trl.years_required_display
    assert json.dumps(trl.to_dict(), allow_nan=False)
    # Not a blanket refusal: an ordinary distribution still gets a length.
    ok = sg.ReturnMoments(sharpe_periodic=0.05, n_obs=1000, skew=-0.5,
                          kurtosis_raw=6.0)
    assert sg.min_track_record_length(ok).attainable is True
    assert sg.min_track_record_length(ok).years_required > 0.0


def test_min_track_record_length_caps_the_displayed_years():
    m = sg.ReturnMoments(sharpe_periodic=0.005, n_obs=1000, skew=0.0, kurtosis_raw=3.0)
    trl = sg.min_track_record_length(m)
    assert trl.years_required > sg.MAX_DISPLAY_YEARS
    assert trl.years_required_display == ">50 years"


def test_negative_skew_lengthens_the_required_track_record():
    sym = sg.ReturnMoments(sharpe_periodic=0.06, n_obs=1000, skew=0.0, kurtosis_raw=3.0)
    ugly = sg.ReturnMoments(sharpe_periodic=0.06, n_obs=1000, skew=-2.0, kurtosis_raw=12.0)
    assert (sg.min_track_record_length(ugly).years_required
            > sg.min_track_record_length(sym).years_required)


# --------------------------------------------------------------- Newey-West

def test_newey_west_at_lag_zero_is_the_unbiased_standard_error():
    """At lag 0 the HAC standard error must be the ORDINARY unbiased one,
    s/√T with ddof=1.

    The previous version of this test asserted against ``std(ddof=0)`` and so
    enshrined the bias it was meant to catch: dividing the autocovariances by T
    instead of T−1 understates the standard error by √((T−1)/T) — 0.06% at
    T=800 but 1.7% at T=30, always in the anti-conservative direction. The
    second assertion is the mutation guard: the two denominators differ, and
    the ddof=0 value is strictly the smaller of the two.
    """
    r = make_returns(800, seed=21, mu=0.0003)
    hac = sg.newey_west_tstat(r, lag=0)
    unbiased = float(r.std(ddof=1)) / math.sqrt(len(r))
    biased = float(r.std(ddof=0)) / math.sqrt(len(r))
    assert hac.se_daily == pytest.approx(unbiased, rel=1e-12)
    assert hac.se_daily > biased
    # A 30-day window makes the same bug worth 1.7%, not 0.06%.
    short = sg.newey_west_tstat(make_returns(30, seed=22, mu=0.0003), lag=0)
    assert short.se_daily == pytest.approx(
        float(make_returns(30, seed=22, mu=0.0003).std(ddof=1)) / math.sqrt(30),
        rel=1e-12)


def test_newey_west_widens_the_standard_error_under_positive_autocorrelation():
    """Cohort mode holds each book for hold_days, so the daily series is
    autocorrelated by construction and an IID standard error is flattering."""
    rng = np.random.default_rng(31)
    n = 2000
    e = rng.standard_normal(n)
    ar = np.zeros(n)
    for t in range(1, n):
        ar[t] = 0.5 * ar[t - 1] + e[t]
    s = pd.Series(ar * DAILY_VOL, index=pd.bdate_range("2010-01-04", periods=n))
    iid = sg.newey_west_tstat(s, lag=0)
    hac = sg.newey_west_tstat(s)
    assert hac.se_daily > iid.se_daily * 1.3
    assert hac.autocorr_lag1 == pytest.approx(0.5, abs=0.06)
    assert abs(hac.t_stat) < abs(iid.t_stat)


def test_newey_west_lag_is_floored_by_the_holding_period():
    assert sg.newey_west_lag(1000, holding_period_days=1) == 6
    assert sg.newey_west_lag(1000, holding_period_days=21) == 21
    assert sg.newey_west_lag(10, holding_period_days=200) == 8   # clamped to T-2


def test_newey_west_reports_a_significant_edge_when_one_exists():
    r = make_returns(2500, seed=41, mu=0.0012)      # ~1.9 annualized Sharpe
    hac = sg.newey_west_tstat(r)
    assert hac.t_stat > 2.0
    assert hac.ci_low_annualized > 0
    # The sentence quotes THE critical value and reads its claim off THE
    # interval, so the number in the prose is the number in the arithmetic.
    assert f"{sg.critical_z(0.95):.2f}" in hac.interpretation
    assert "reaches it" in hac.interpretation
    assert "lies entirely above zero" in hac.interpretation


def test_newey_west_needs_a_minimum_sample():
    with pytest.raises(ValueError):
        sg.newey_west_tstat(pd.Series([0.01, -0.01],
                                      index=pd.bdate_range("2020-01-01", periods=2)))


# ------------------------------------------------- dust over dust: a CONSTANT
#
# The failure these tests exist for: an `se > 0` test cannot see its own
# denominator collapse. On a series that is a nonzero constant, both the mean
# and the HAC standard error are rounding artefacts, `se > 0` is True, and the
# t-statistic is a constant divided by a rounding error — finite, 17 digits
# long, and fed straight into a verdict and into reader-facing prose.

#: The committed fixtures, reached the way conftest reaches them.
FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


def _committed_flat_rf() -> pd.Series:
    """The longest stretch of the COMMITTED fixture over which ^IRX does not
    move, converted to a daily rate the way ``pipeline._rf_daily`` converts it
    (an annualized discount rate in percent → ``irx / 100 / 252``).

    Nothing here is synthetic: these are the quotes in
    ``fixtures/extras_prices.csv``, and the flatness is asserted rather than
    constructed.
    """
    irx = pd.read_csv(FIXTURES / "extras_prices.csv",
                      index_col=0, parse_dates=True)["^IRX"]
    run = (irx / 100.0 / 252.0).loc["2021-10-19":"2021-10-27"]
    assert run.nunique() == 1 and len(run) >= 3, "fixture changed; refind a flat run"
    return run


def _fixture_sessions() -> pd.DatetimeIndex:
    """793 real sessions, 2022-12-01..2026-01-30, from the committed war-list
    price fixture — the window the reachable case was found on."""
    idx = pd.read_csv(FIXTURES / "war_list_prices.csv",
                      index_col=0, parse_dates=True).loc["2022-12-01":"2026-01-30"].index
    assert len(idx) == 793, "fixture changed; the window is no longer 793 sessions"
    return idx


def _fabricated_hac_tstat(x: np.ndarray) -> tuple[float, float]:
    """→ (the standard error the old code accepted, the t it handed back).

    Recomputed from the module's own HAC machinery rather than quoted, so this
    tracks the implementation instead of pinning a magic number.
    """
    se = math.sqrt(sg._hac_long_run_variance(x, sg.hac_lag(x, 1)) / x.size)
    return se, (float(x.mean()) / se if se > 0 else float("nan"))


@pytest.mark.parametrize("case", ["committed_flat_rf_run", "793_session_window"])
def test_a_constant_excess_series_refuses_the_hac_tstat(case):
    """REACHABLE input: a strategy whose filter never qualifies returns 0.0
    every day; subtract a cash rate that does not move and the excess series is
    a nonzero CONSTANT.

    Two cases, because the fabrication has two faces and a guard that catches
    only one is not a guard:

      * ``committed_flat_rf_run`` — 7 sessions of the committed fixture where
        ^IRX genuinely does not move. The deviations from the mean are exactly
        zero, so ``se == 0``, the old ``se > 0`` test failed and ``t`` was NaN —
        which sails through ``abs(t) >= 2`` into "not distinguishable from
        noise" and printed the literal word "nan" as the t-statistic. A false
        TIE.
      * ``793_session_window`` — the same construction over the committed
        fixture's own 793-session calendar. Here the mean of 793 identical
        floats is off by an ulp, so the deviations are ~5e-20, ``se > 0`` is
        TRUE, and the t-statistic came out at ~-1e16 with ``p = 0.0`` and "That
        clears the conventional |t| > 2 bar." A false LOSS.

    Either way the sample supports no verdict, and a false "loses" is exactly
    as wrong as a false "beats".
    """
    if case == "committed_flat_rf_run":
        rf = _committed_flat_rf()
    else:
        rf = pd.Series(0.0002, index=_fixture_sessions())
    strategy = pd.Series(0.0, index=rf.index)
    # Built by metrics._excess — the one call the whole codebase uses for it.
    excess = metrics._excess(strategy, rf)
    x = excess.to_numpy(dtype=float)
    assert len(np.unique(x)) == 1 and x[0] != 0.0

    # The old guard's own arithmetic, and what it handed back.
    dust_se, fabricated = _fabricated_hac_tstat(x)
    if case == "committed_flat_rf_run":
        assert dust_se == 0.0 and math.isnan(fabricated)
    else:
        assert dust_se > 0.0, "the `se > 0` guard passes on this input"
        assert abs(fabricated) > 1e15, "…and hands back a 16-digit t-statistic"

    hac = sg.newey_west_tstat(excess)
    assert hac.available is False
    assert "dispersion" in hac.unavailable_reason
    # Every statistic that would rest on the collapsed denominator is absent.
    for value in (hac.t_stat, hac.p_value_two_sided, hac.se_daily,
                  hac.se_annualized, hac.ci_low_annualized,
                  hac.ci_high_annualized, hac.autocorr_lag1):
        assert math.isnan(value)
    # The mean IS an arithmetic fact about the sample and is still reported.
    assert hac.mean_daily == pytest.approx(float(x.mean()), rel=1e-15)

    # No verdict reaches the reader, in EITHER direction, and the fabricated
    # number is absent as this module would have formatted it — not as a literal.
    for text in (hac.interpretation, hac.unavailable_reason):
        assert f"{fabricated:.2f}" not in text
        assert "clears the conventional" not in text
        assert "not distinguishable from noise" not in text
        assert "|t|" not in text
        assert not re.search(r"t-statistic of\s*[-+]?[\d.]", text)
        assert "nan" not in text.lower()
    assert json.dumps(hac.to_dict(), allow_nan=False)
    assert hac.to_dict()["t_stat"] is None


def test_a_constant_excess_series_refuses_the_bootstrap_interval():
    """The SIBLING call site, which has the same hole for the same reason.

    ``se_point <= 0.0`` is the bootstrap's version of ``se > 0``: on a nonzero
    constant the long-run variance is dust rather than zero, every resample is
    the same constant, and the studentized interval came back as the zero-width
    [-5.04%, -5.04%] with ``excludes_zero=True`` — "The interval excludes zero"
    on a series that cannot support an interval at all.
    """
    rf = pd.Series(0.0002, index=_fixture_sessions())
    excess = metrics._excess(pd.Series(0.0, index=rf.index), rf)
    x = excess.to_numpy(dtype=float)
    # The old guard passes: the long-run variance is positive, just dust.
    lag = sg.hac_lag(x, 1)
    assert sg._hac_long_run_variance(x, lag) > 0.0
    with pytest.raises(ValueError, match="dispersion"):
        sg.bootstrap_excess_return_ci(excess, seed=1, n_resamples=200)


def test_assess_excess_refuses_a_constant_offset_and_says_what_it_is():
    """The bundled entry point must refuse BEFORE either rendering site runs,
    and its reason must be true of the data.

    The absolute-floor guard already refused this input, but it said "the
    strategy tracks the benchmark exactly" about a series that differs from it
    by 5.04%/yr on every single session — a refusal for the wrong reason reads
    as a different bug and invites someone to "fix" it.
    """
    rf = pd.Series(0.0002, index=_fixture_sessions())
    strategy = pd.Series(0.0, index=rf.index)
    out = sg.assess_excess(strategy, rf, seed=1, n_resamples=200, label="cash")

    assert out.available is False
    assert out.verdict_source == "none"
    assert out.verdict == sg.TIES_WITHIN_NOISE          # never a winner
    assert out.hac is None and out.bootstrap is None
    assert out.ci_coverage_caveat is None
    assert "no dispersion" in out.unavailable_reason
    assert "tracks cash exactly" not in out.unavailable_reason
    # The constant itself is named, recomputed rather than pinned.
    assert f"{-0.0002 * PERIODS:+.2%}" in out.unavailable_reason
    assert "793" in out.unavailable_reason
    # "We cannot tell" is not "they are indistinguishable".
    assert "it is a refusal" in out.interpretation
    for text in (out.interpretation, out.unavailable_reason):
        assert "beats" not in text and "loses" not in text
        assert "clears the conventional" not in text
    assert json.dumps(out.to_dict(), allow_nan=False)


def test_a_non_finite_difference_series_refuses_instead_of_calling_it_a_tie():
    """An inf must take the REFUSAL path, not the reassuring one.

    ``metrics._check`` refuses NaN, not ±inf, so an inf return (a zero price in
    the panel, divided into) reaches this module. Its dispersion share is NaN —
    and ``share <= FLOOR`` is False for a NaN, so the first version of this
    guard waved it through exactly the way the bug it was written for did:
    measured, the module printed "+inf% annualized, with a
    serial-correlation-robust t-statistic of nan … That is short of the
    conventional |t| > 2 bar, so the gap is not distinguishable from noise."
    A NaN fails every ``if`` and lands in the branch that sounds safe.
    """
    idx = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(6100)
    x = rng.standard_normal(300) * 0.01
    x[17] = np.inf
    series = pd.Series(x, index=idx)
    assert math.isnan(sg.dispersion_share(x))
    assert not (sg.dispersion_share(x) <= sg.DISPERSION_SHARE_FLOOR)

    hac = sg.newey_west_tstat(series)
    assert hac.available is False
    assert "not finite" in hac.unavailable_reason
    assert math.isnan(hac.t_stat) and math.isnan(hac.se_daily)
    for text in (hac.interpretation, hac.unavailable_reason):
        assert "nan" not in text.lower()
        assert "inf" not in text.replace("finite", "").lower()
        assert "not distinguishable from noise" not in text
        assert "clears the conventional" not in text
    assert json.dumps(hac.to_dict(), allow_nan=False)

    with pytest.raises(ValueError, match="dispersion"):
        sg.bootstrap_excess_return_ci(series, seed=1, n_resamples=100)

    out = sg.assess_excess(series, pd.Series(0.0, index=idx), seed=1,
                           n_resamples=100)
    assert out.available is False and out.verdict_source == "none"
    assert "not finite" in out.unavailable_reason
    # and NOT the wrong reason: this series does not track its benchmark.
    assert "tracks" not in out.unavailable_reason
    assert json.dumps(out.to_dict(), allow_nan=False)


def test_dispersion_share_floor_sits_between_the_dust_and_the_flattest_real_series():
    """The threshold's justification, RECOMPUTED. Two quantities bracket it.

    An absolute floor on the standard error is the mis-calibration that has bitten
    this codebase before (the engine's inverse-vol floor sat 250x above its threat
    and bound on a real asset class), so the bar is a scale-free share — and the
    gap it sits in is measured here rather than asserted in a comment.
    """
    floor = sg.DISPERSION_SHARE_FLOOR

    # --- artefact side: an exact constant's share is arithmetic noise only.
    worst_dust = 0.0
    n_series = 0
    for t in (30, 100, 253, 793, 2264, 5000):
        idx = pd.bdate_range("2015-01-02", periods=t)
        for c in (1e-6, 1e-4, 2e-4, 1e-3, 0.01, 0.1, 1.0, 7.0, 1e3, 1e6,
                  -2e-4, -0.05):
            worst_dust = max(worst_dust, sg.dispersion_share(np.full(t, c)))
            n_series += 1
        # and built the way the pipeline builds one: excess of two flat series
        for a, b in ((0.0, 2e-4), (1e-3, 1e-3 + 2e-4), (0.05, 0.05 + 1e-5)):
            built = metrics._excess(pd.Series(a, index=idx),
                                    pd.Series(b, index=idx)).to_numpy()
            worst_dust = max(worst_dust, sg.dispersion_share(built))
            n_series += 1
    assert n_series >= 84
    assert worst_dust < 1e-30
    assert floor / worst_dust > 1e14        # ~15 orders of margin above the dust
    # An all-zeros series has no total to take a share of. It must report 0.0,
    # NOT NaN: every `<= floor` comparison is False for a NaN, so a NaN here
    # turns the refusal back into a pass.
    assert sg.dispersion_share(np.zeros(300)) == 0.0

    # --- real side: the flattest genuinely-varying series the fixtures contain
    # is the ^IRX-derived risk-free PATH; an all-cash strategy against it is the
    # very input this guard was found on, and it must stay testable.
    irx = pd.read_csv(FIXTURES / "extras_prices.csv",
                      index_col=0, parse_dates=True)["^IRX"]
    rf_path = (irx / 100.0 / 252.0).loc["2022-12-01":"2026-01-30"]
    flattest_real = sg.dispersion_share(rf_path.to_numpy())
    assert flattest_real > 0.01
    assert flattest_real / floor > 1e12     # ~13 orders of margin below it
    prices = pd.read_csv(FIXTURES / "sp500_prices.csv", index_col=0,
                         parse_dates=True).loc["2022-12-01":"2026-01-30"]
    shares = [sg.dispersion_share(prices[c].pct_change().dropna().to_numpy())
              for c in prices.columns[:120]]
    assert min(shares) > 0.9

    # --- a genuine dispersion five orders below its own mean sits 1e4 above the
    # SHARE floor — and is nonetheless refused, by the ABSOLUTE half, because
    # its daily sd of 1e-9 is a hundredth of ABSOLUTE_SD_FLOOR. That is the
    # deliberate cost of covering the band, and which half refuses it is pinned
    # so nobody can read this as the share floor over-correcting.
    rng = np.random.default_rng(6101)
    tight = 2e-4 + rng.standard_normal(793) * 1e-9
    assert sg.dispersion_share(tight) / floor > 1e4
    assert sg.degeneracy_refusal(tight).kind == "absolute"
    assert sg.newey_west_tstat(
        pd.Series(tight, index=_fixture_sessions())).available is False
    # The bar is a bar, not a taste: sd just under ABSOLUTE_SD_FLOOR is refused
    # and sd just over it is testable, on otherwise identical series.
    z = rng.standard_normal(793)
    z = (z - z.mean()) / z.std(ddof=1)          # so sd is EXACTLY the target
    for sd, testable in ((1e-8, False), (9.9e-8, False), (1.01e-7, True),
                         (1e-6, True)):
        y = 2e-4 + z * sd
        assert float(np.std(y, ddof=1)) == pytest.approx(sd, rel=1e-9)
        assert (sg.degeneracy_refusal(y) is None) is testable, sd

    # --- and the share is a monotone function of the t-statistic it guards, so
    # no fit with a reportable t can be refused: at lag 0 the identity is
    # share = (T-1)/(t² + T-1).
    t_obs = 793
    for target in (2.0, 20.0, 1e3, 1e8):
        share_at = (t_obs - 1.0) / (target ** 2 + t_obs - 1.0)
        assert share_at > floor * 100.0, target
    assert math.sqrt((t_obs - 1.0) * (1.0 - floor) / floor) > 1e9


def test_the_dispersion_floor_does_not_refuse_a_genuinely_tiny_edge():
    """The over-correction check. This bar must bind on CONSTANTS, never on
    smallness: a 1bp/day edge against 1bp/day of noise is 1/100th the scale of a
    normal return series and is a perfectly testable fit, which is exactly what
    an absolute floor on the standard error gets wrong.
    """
    rng = np.random.default_rng(6102)
    idx = _fixture_sessions()
    bench = pd.Series(rng.standard_normal(len(idx)) * 0.01, index=idx)
    strategy = bench + 0.0001 + rng.standard_normal(len(idx)) * 0.0001

    out = sg.assess_excess(strategy, bench, seed=3, n_resamples=200)
    assert out.available is True
    assert out.verdict == sg.BEATS
    assert out.hac.available is True and math.isfinite(out.hac.t_stat)
    assert out.hac.t_stat > 2.0

    # Scale-free in the literal sense: multiply the whole series by 1e6 or 1e-6
    # and the share — hence the verdict — does not move.
    x = (strategy - bench).to_numpy()
    base = sg.dispersion_share(x)
    for k in (1e-6, 1e-3, 1e3, 1e6):
        assert sg.dispersion_share(x * k) == pytest.approx(base, rel=1e-9)
    # A constant of the SAME tiny size is refused, which is the distinction the
    # bar is drawing — size is not the criterion.
    assert sg.dispersion_share(np.full(len(idx), 0.0001)) <= sg.DISPERSION_SHARE_FLOOR


def test_the_sharpe_family_is_already_covered_by_the_degeneracy_policy():
    """WHICH SIBLING SITES NEEDED A SECOND GUARD, measured rather than assumed.

    Everything built on :class:`ReturnMoments` — the Sharpe standard error,
    ``sharpe_difference_test``, PSR, DSR, MinTRL and ``bootstrap_sharpe_ci`` —
    is gated at construction by ``_degenerate_sharpe``, and across the whole
    range of PLAUSIBLE daily returns that gate is strictly more aggressive than
    the dispersion floor: every series the floor would refuse, it refuses first.
    The two only part company at a mean daily return of ~1e5 (ten million
    percent per day), which is why those sites are NOT double-guarded.

    The second half is the structural reason ``sharpe_difference_test``'s
    ``se > 0`` cannot collapse: its denominator is
    ``sqrt(252·numer/T)`` with ``numer`` guarded positive and floored by the
    Lo fallback, so it is bounded below by ~0.16 at T=10,000.
    """
    rng = np.random.default_rng(6103)
    for mean in (2e-4, 1e-3, 0.05, 0.5, 1.0):
        for rel in (1e-9, 1e-12, 1e-16, 0.0):
            x = mean + rng.standard_normal(400) * mean * rel
            if sg.dispersion_share(x) <= sg.DISPERSION_SHARE_FLOOR:
                assert sg._degenerate_sharpe(x), (mean, rel)
                assert sg.ReturnMoments.try_from_returns(
                    pd.Series(x, index=pd.bdate_range("2015-01-02", periods=400))
                ) is None, (mean, rel)

    # The Mertens/PSR variance term is what a collapsing Sharpe standard error
    # would need, and it does not get near zero on reachable data: a series with
    # a single enormous spike is the extreme case (it drives sample skew to
    # √T and kurtosis to T at once) and still leaves the term at ~0.25.
    worst = 1.0
    for spikes in (1, 2, 3, 5):
        x = rng.standard_normal(793) * 1e-6
        x[:spikes] += 1.0
        m = sg.ReturnMoments.from_returns(
            pd.Series(x, index=pd.bdate_range("2015-01-02", periods=793)))
        worst = min(worst, sg._psr_variance_numerator(m))
        se = sg.sharpe_standard_error(m)
        assert se.se_annual > 0.1
    assert worst > 0.2
    # sharpe_difference_test's denominator inherits that floor, so its z-stat
    # cannot be a ratio of dust.
    a = sg.ReturnMoments.from_returns(make_returns(2500, seed=6104, mu=0.0006))
    b = sg.ReturnMoments.from_returns(make_returns(2500, seed=6105, mu=0.0002))
    d = sg.sharpe_difference_test(a, b)
    assert d["se_difference_annual"] > 0.1
    assert abs(d["z_stat"]) < 10.0


# ============================================================================
# ONE DEGENERACY PREDICATE, EVERY SITE, EVERY LEVEL
#
# Six rounds each fixed one degeneracy instance and left a sibling one LEVEL
# down or one SITE over. These tests are about the POLICY, not the instances:
# that there is one predicate, that both halves are in it, that every site that
# renders a claim calls it, and that the quantities with algebraic bounds are
# checked against them rather than against a calibration.
# ============================================================================

def _rare_signal_case(n_events: int = 2, move: float = 0.03, t: int = 793):
    """A RARE-SIGNAL strategy: in cash except ``n_events`` days, against a flat
    cash benchmark, over the committed 793-session window.

    Nothing about this series is near-constant by either whole-series floor
    (sd ~1.5e-3, dispersion share ~0.993), which is the point: the collapse it
    provokes lives one LEVEL down, inside the resampling loop.
    """
    idx = _fixture_sessions()[:t]
    rf = pd.Series(0.0002, index=idx)
    r = np.zeros(t)
    r[np.linspace(40, t - 40, n_events).astype(int)] = move
    return pd.Series(r, index=idx), rf


def _three_level_case(n_events: int = 3, t: int = 793):
    """A near-constant base, a 1e-9 second level, and ``n_events`` real events,
    so a resample can be near-constant WITHOUT being exactly constant.

    An exactly constant resample has ``sd == 0`` and is already excluded by
    ``isfinite``, which is why the Sharpe-side hole needed a series shaped like
    this to show: here the near-constant resample makes the Mertens plug-in
    studentizer HUGE rather than tiny, so the pivot goes to zero instead of to
    1e16. It does not have to be a strategy anyone would run — the predicate has
    to be right. Every observation is POSITIVE, which is what makes a negative
    annualized Sharpe in the interval an artefact rather than a wide answer.
    """
    x = np.full(t, 2e-4)
    x[::3] = 2e-4 + 1e-9
    x[np.linspace(60, t - 60, n_events).astype(int)] = 0.01
    return pd.Series(x, index=pd.bdate_range("2015-01-02", periods=t)), x


def test_one_resample_collapse_cannot_fabricate_an_interval_the_series_denies():
    """DEFECT 1 — the per-resample studentizer, the ``se > 0`` shape one level
    down, inside the bootstrap loop where no whole-series guard can reach.

    Reachable: a strategy in cash except 2 event days, against flat cash. Its
    whole-series sd is 1.5e-3 (four orders ABOVE the absolute floor) and its
    dispersion share is 0.993 (thirteen orders above the share floor), so
    NOTHING refuses it. But the resamples that miss the event days are exactly
    constant, carry ``se_b = 5.09e-21`` — positive dust, so ``se_b > 0`` passes
    — and their pivots are ~1.5e16. Measured before the first fix, at seed 7:

        "[-4.97%, +19879096992716836.00%]"

    printed verbatim into the reader-facing interpretation AND into the
    pipeline's ``excess_significance`` finding.

    RE-PINNED, because the first fix DROPPED those resamples and reported an
    interval from the survivors, and this test asserted that interval. Dropping
    is BIASED, not merely lossy — asserted directly below — so what the module
    owes this input is a REFUSAL of the resampled interval plus the unbiased HAC
    verdict, LABELLED as coming from there. The old assertion (the reported CI
    equals the dust-free CI, verdict ``loses``) is true and harmless at 2 events
    and one step wrong at 1, which is the sibling case the next test covers.

    Everything below is recomputed from the module, not pinned to those digits.
    """
    strat, rf = _rare_signal_case()
    excess = (strat - rf).rename("excess")
    x = excess.to_numpy(dtype=float)

    # 1. No whole-series guard can see this. Both halves pass, with room.
    assert sg.degeneracy_refusal(x) is None
    assert float(np.std(x, ddof=1)) / sg.ABSOLUTE_SD_FLOOR > 1e4
    assert sg.dispersion_share(x) > 0.99

    # 2. The dust IS there, and the old mask would have accepted it.
    lag = sg.hac_lag(x, 1)
    idx = sg.stationary_bootstrap_indices(
        len(x), 1000, sg.bootstrap_block_days(len(x), 1), seed=7)
    boot = x[idx]
    means = boot.mean(axis=1)
    se_b = np.sqrt(sg._hac_long_run_variance_matrix(boot, lag) / len(x))
    se_point = math.sqrt(sg._hac_long_run_variance(x, lag) / len(x))
    old_mask = np.isfinite(means) & np.isfinite(se_b) & (se_b > 0)
    dust = old_mask & (se_b < se_point * 1e-6)
    assert dust.sum() > 100, "the reachable case must actually produce dust"
    assert se_b[dust].max() / se_point < 1e-10
    old_pivots = (means[old_mask] - x.mean()) / se_b[old_mask]
    assert np.abs(old_pivots).max() > 1e15      # the 1.5e16 pivot, recomputed

    # 3. The mask excludes exactly those, and its count is the WHOLE exclusion
    #    (n_usable + n_dropped == n_resamples), not just the finite part of it.
    ok, dropped = sg._usable_pivots(boot, means, se_b, "mean")
    assert not ok[dust].any()
    assert int(ok.sum()) + dropped == boot.shape[0]
    assert np.abs((means[ok] - x.mean()) / se_b[ok]).max() < 100.0

    # 4. DROPPING THEM IS BIASED, NOT LOSSY — the reason this is a refusal and
    #    not a narrower interval. Every excluded resample sits on ONE side of
    #    the point estimate, because it is precisely a resample in which the
    #    rare event did not occur.
    assert bool((means[~ok] < x.mean()).all())
    assert dropped / boot.shape[0] > sg.max_discarded_pivot_share(0.95)

    # 5. So the resampled interval is REFUSED, by the named predicate, naming
    #    the share and the bar rather than a symptom.
    with pytest.raises(sg.ResampleRefusal) as exc:
        sg.bootstrap_excess_return_ci(excess, seed=7, n_resamples=1000,
                                      label="cash")
    assert f"{dropped:,} of 1,000" in str(exc.value)
    assert "biased" in str(exc.value).lower()

    # 6. And the verdict still gets made, from the interval that cannot lose a
    #    rare day because it never resamples — and SAYS SO.
    out = sg.assess_excess(strat, rf, seed=7, n_resamples=1000, label="cash")
    assert out.available is True
    assert out.bootstrap is None
    assert out.verdict_source == "hac"
    assert out.ci_coverage_caveat == sg.CI_COVERAGE_CAVEAT_HAC
    assert out.verdict == sg.classify_excess(
        out.hac.mean_annualized, out.hac.ci_low_annualized,
        out.hac.ci_high_annualized)
    # this particular series really does lose to cash on the HAC evidence, so
    # the refusal is not hiding behind a tie
    assert out.hac.t_stat < -2.0 and out.verdict == sg.LOSES
    assert "No resampled interval stands behind this verdict" in out.interpretation
    for text in (out.interpretation, out.hac.interpretation):
        assert "e+" not in text and "e-" not in text
        assert not re.search(r"[-+]?\d{7,}", text.replace(",", ""))


@pytest.mark.parametrize("seed", [1, 2, 3, 7, 11, 42])
def test_the_resample_collapse_is_not_one_unlucky_seed(seed):
    """Reproduced at six seeds, because a single-seed fix is a single-seed fix.
    Each seed produces its own dust count and each must refuse."""
    strat, rf = _rare_signal_case()
    excess = (strat - rf).rename("excess")
    with pytest.raises(sg.ResampleRefusal):
        sg.bootstrap_excess_return_ci(excess, seed=seed, n_resamples=1000,
                                      label="cash")
    out = sg.assess_excess(strat, rf, seed=seed, n_resamples=1000, label="cash")
    assert out.verdict_source == "hac", seed
    assert out.bootstrap is None, seed
    # The HAC statistic does not depend on the bootstrap seed at all, which is
    # the point of falling back to it: the verdict stops moving with the seed.
    assert out.hac.t_stat == pytest.approx(
        sg.newey_west_tstat(excess, label="cash").t_stat, rel=1e-12), seed
    assert out.verdict == sg.LOSES, seed


@pytest.mark.parametrize("t,move", [(253, 0.05), (504, 0.05), (504, 0.10),
                                    (793, 0.10), (793, 0.20)])
def test_a_single_rare_day_cannot_fabricate_a_directional_verdict(t, move):
    """THE SIBLING THE LAST ROUND CREATED, at ``n_events = 1``.

    With the dust resamples DROPPED rather than refused, the pivot distribution
    is truncated on one side only: 33.9%-38.3% of resamples have zero copies of
    the single event day, are exactly constant and were discarded, and of the
    survivors most carry the sample's own multiset, so the empirical 2.5% pivot
    quantile landed inside an atom at zero (measured -5.5e-16) and
    ``_studentized_interval`` returned ``ci_high == point`` to 128 ulps.
    ``excludes_zero`` then degenerated to ``sign(point)``. The report said, in
    one paragraph, "The strategy loses to cash by more than sampling noise
    explains … t-statistic of -0.01 … That is short of the conventional |t| > 2
    bar, so the gap is not distinguishable from noise."

    These five cells returned ``ties_within_noise`` under the pre-fix
    ``se_b > 0`` mask and ``loses`` after it — a REGRESSION, in the direction of
    a fabricated claim. They must not report a directional verdict that the
    t-statistic denies.
    """
    strat, rf = _rare_signal_case(n_events=1, move=move, t=t)
    excess = (strat - rf).rename("excess")
    assert sg.degeneracy_refusal(excess.to_numpy(dtype=float)) is None

    with pytest.raises(sg.ResampleRefusal):
        sg.bootstrap_excess_return_ci(excess, seed=7, n_resamples=1000,
                                      label="cash")
    out = sg.assess_excess(strat, rf, seed=7, n_resamples=1000, label="cash")
    assert out.available is True and out.bootstrap is None
    assert out.verdict_source == "hac"
    # The headline and the t-statistic now agree, which is the whole complaint:
    # a directional verdict requires the HAC interval to exclude zero, and that
    # is the same comparison the sentence beside it reports.
    #
    # RE-PINNED AT THE RIGHT CRITICAL VALUE. This line used to read
    # ``abs(out.hac.t_stat) < 2.0`` and that assertion is FALSE for every |t| in
    # [1.9599639845, 2.0) — it passed only because this parametrization happens
    # to avoid the window. The window is reachable (T=738/739/740 at
    # n_events=1, move=0.05) and is swept directly by
    # ``test_the_one_critical_value_holds_across_the_window_that_used_to_disagree``
    # below. The bar is now read from the module rather than spelled here, so a
    # test cannot re-introduce a second critical value either.
    z = sg.critical_z(0.95)
    assert (out.verdict == sg.TIES_WITHIN_NOISE) == (abs(out.hac.t_stat) < z)
    if out.verdict == sg.TIES_WITHIN_NOISE:
        assert "not distinguishable" in out.interpretation
        assert "does not reach it" in out.hac.interpretation
    else:
        assert "reaches it" in out.hac.interpretation
        assert "not distinguishable from noise" not in out.hac.interpretation
    assert sg.CI_COVERAGE_CAVEAT_BOOTSTRAP not in (out.ci_coverage_caveat or "")


@pytest.mark.parametrize("t", [735, 736, 737, 738, 739, 740, 741, 742])
def test_the_one_critical_value_holds_across_the_window_that_used_to_disagree(t):
    """THE TWO CRITICAL VALUES, THREE LINES APART — swept across the boundary.

    ``classify_excess`` read an interval built at ``norm_ppf(0.5 + conf/2)`` =
    1.9599639845 while the sentence three lines below compared ``|t|`` against a
    folklore 2.0. They disagree for every ``|t|`` in [1.95996, 2.0), and round 8
    made that window reachable on a whole family by routing the rare-signal case
    through HAC. Deterministic — HAC does not resample, so every seed agrees.

    Measured before the fix, at T=738: ``t = -1.9600236384``, interval
    [-6.665e-02, -1.014e-06], ``verdict = loses``, and the paragraph the pipeline
    renders verbatim at severity WARNING under the title "Trails cash by more
    than noise" said "… t-statistic of -1.96 … That is short of the conventional
    |t| > 2 bar, so the gap is not distinguishable from noise."

    This sweep CROSSES the window rather than avoiding it, and the assertion is
    that there is one bar: the verdict, the interval's relationship to zero and
    the sentence are the same statement at every T.
    """
    strat, rf = _rare_signal_case(n_events=1, move=0.05, t=t)
    out = sg.assess_excess(strat, rf, seed=7, n_resamples=1000, label="cash")
    assert out.verdict_source == "hac"          # the bootstrap is refused here
    z = sg.critical_z(0.95)
    hac = out.hac
    directional = out.verdict != sg.TIES_WITHIN_NOISE

    # 1. ONE BAR: the verdict is the interval's relationship to zero, and that is
    #    |t| against z. Not against 2.0 — which is the bug, asserted as such.
    assert directional == (abs(hac.t_stat) >= z)
    # 2. THE SENTENCE SAYS THE SAME THING, including its direction.
    text = hac.interpretation
    assert f"{z:.2f}" in text
    if directional:
        assert "reaches it" in text
        assert "lies entirely below zero" in text      # this family loses
        assert "not distinguishable from noise" not in text
        assert "loses to cash" in out.interpretation
    else:
        assert "does not reach it" in text
        assert "not distinguishable" in text
    # 3. NO SECOND BAR ANYWHERE IN THE PARAGRAPH.
    for rendered in (text, out.interpretation):
        assert "|t| > 2" not in rendered
        assert "conventional" not in rendered


def test_the_swept_window_really_contains_the_disagreement():
    """The sweep above is only a test if it crosses the window. Asserted, not
    assumed: at least one T in it has ``|t|`` inside [1.9599639845, 2.0), where
    the old pair of critical values gave opposite answers, and the module now
    calls it directional in agreement with its own interval."""
    z = sg.critical_z(0.95)
    inside = {}
    for t in (735, 736, 737, 738, 739, 740, 741, 742):
        strat, rf = _rare_signal_case(n_events=1, move=0.05, t=t)
        out = sg.assess_excess(strat, rf, seed=7, n_resamples=1000, label="cash")
        if z <= abs(out.hac.t_stat) < 2.0:
            inside[t] = out
    assert inside, "the sweep no longer crosses the critical-value window"
    for t, out in inside.items():
        # the OLD sentence's bar would have said "not distinguishable" here...
        assert abs(out.hac.t_stat) < 2.0, t
        # ...while the interval it was printed beside excluded zero.
        assert out.verdict == sg.LOSES, t
        assert out.hac.ci_high_annualized < 0.0, t
        assert "reaches it" in out.hac.interpretation, t


@pytest.mark.parametrize("seed", [1, 2, 3, 5, 7, 11, 42, 99])
def test_the_fabricated_verdict_is_not_one_unlucky_seed_and_is_sign_symmetric(seed):
    """Eight seeds, and BOTH signs. The pre-fix collapse pinned ``ci_high`` to
    the point estimate at every seed; flipping the sign of the excess series
    pinned ``ci_low`` instead and reported ``beats``. A fix that only held for
    one seed, or only for losers, would not be a fix.
    """
    strat, rf = _rare_signal_case(n_events=1, move=0.10, t=504)
    excess = (strat - rf).rename("excess")
    for series in (excess, (-excess).rename("excess")):
        with pytest.raises(sg.ResampleRefusal):
            sg.bootstrap_excess_return_ci(series, seed=seed, n_resamples=1000,
                                          label="cash")
    # ...and the mirrored series gets the mirrored HAC verdict, not "beats"
    out = sg.assess_excess(rf, strat, seed=seed, n_resamples=1000, label="cash")
    assert out.verdict_source == "hac"
    assert out.verdict == sg.TIES_WITHIN_NOISE, seed
    assert abs(out.hac.t_stat) < 2.0, seed


def test_the_row_predicate_is_the_same_predicate_as_the_scalar_one():
    """PRINCIPLE 1 AND 3 TOGETHER: one predicate, and the vectorized form used
    inside the resampling loop must not be allowed to drift from it.

    Row by row, over rows built to sit on every branch: constant, all-zero,
    one-tick, sub-floor dispersion, non-finite, and ordinary.
    """
    rows = [
        np.full(60, -2e-4),                         # nonzero constant
        np.zeros(60),                               # all zero
        np.full(60, 5.0),                           # large constant
        np.concatenate([np.full(59, -2e-4), [-2e-4 + 3.97e-8]]),   # one tick
        2e-4 + np.linspace(-1e-9, 1e-9, 60),        # dispersion under the floor
        2e-4 + np.linspace(-1e-6, 1e-6, 60),        # dispersion over it
        np.concatenate([np.full(59, 0.01), [np.inf]]),
        np.linspace(-0.02, 0.03, 60),               # ordinary
        # A row only the SHARE half can refuse: sd 1e-6 is TEN TIMES the
        # absolute floor, but about a mean of 100 the dispersion share is 1e-16,
        # under eps. Unreachable as a return series; the predicate still has to
        # be right, and without this row the share half of the ROW form is dead
        # weight no test can show to be load-bearing.
        100.0 + np.linspace(-1.732e-6, 1.732e-6, 60),
    ]
    m = np.vstack(rows)
    mask = sg.degenerate_rows(m)
    for i, row in enumerate(rows):
        assert bool(mask[i]) is (sg.degeneracy_refusal(row) is not None), i
    # and it really does exercise both answers, so agreement is not vacuous
    assert mask.any() and not mask.all()
    # both halves are live in the row form: the one-tick row is refused by the
    # absolute half alone, the nonzero-constant row by the share half alone.
    assert sg.degeneracy_refusal(rows[3]).kind == "absolute"
    assert sg.dispersion_share(rows[3]) > sg.DISPERSION_SHARE_FLOOR
    assert sg.degeneracy_refusal(rows[0]).kind == "constant"
    # the share-half-only row, with its two numbers recomputed
    share_only = rows[-1]
    assert float(np.std(share_only, ddof=1)) > 10.0 * sg.ABSOLUTE_SD_FLOOR
    assert sg.dispersion_share(share_only) <= sg.DISPERSION_SHARE_FLOOR
    assert sg.degeneracy_refusal(share_only).kind == "constant"
    assert bool(sg.degenerate_rows(np.vstack([share_only, share_only]))[0])


def test_the_sharpe_bootstrap_gets_the_same_row_level_policy():
    """THE SIBLING SITE, one level down, in the OTHER bootstrap.

    ``bootstrap_sharpe_ci`` carried the identical ``se_b > 0`` mask. The
    collapse points the other way there — a near-constant resample makes the
    Mertens plug-in studentizer HUGE rather than tiny, so the pivot goes to
    zero instead of to 1e16 — and the damage is the same kind: the pivot
    distribution acquires a tail that is arithmetic, not sampling.

    Three-level series (a near-constant base, a 1e-9 second level, three real
    events) so a resample can be near-constant WITHOUT being exactly constant;
    an exactly constant resample has ``sd == 0`` and is already dropped by
    ``isfinite``, which is why the hole needed a series shaped like this to
    show. It does not have to be a strategy anyone would run — the predicate
    has to be right.
    """
    series, x = _three_level_case(n_events=3)
    t = x.size
    # the whole series is perfectly healthy: no whole-series guard fires
    assert sg.degeneracy_refusal(x) is None
    assert sg.dispersion_share(x) > 0.8

    idx = sg.stationary_bootstrap_indices(
        t, 1000, sg.bootstrap_block_days(t, 1), seed=7)
    boot = x[idx]
    sr_b, se_b, _ = sg._sharpe_and_se_matrix(boot)
    old_mask = np.isfinite(sr_b) & np.isfinite(se_b) & (se_b > 0)
    bad = sg.degenerate_rows(boot)
    assert int((old_mask & bad).sum()) > 10, "the hole must actually be open"
    assert sr_b[old_mask & bad].min() > 1e5          # dust-driven Sharpes

    moments = sg.ReturnMoments.from_returns(series)
    numer = sg._psr_variance_numerator(moments)
    se_point = math.sqrt((numer if numer > 0 else
                          1.0 + moments.sharpe_periodic ** 2 / 2.0) / (t - 1))
    ok, dropped = sg._usable_pivots(boot, sr_b, se_b, "sharpe")
    assert not ok[bad].any()
    assert int(ok.sum()) + dropped == boot.shape[0]

    def interval(mask):
        piv = (sr_b[mask] - moments.sharpe_periodic) / se_b[mask]
        lo, hi, _straddles = sg._studentized_interval(
            moments.sharpe_periodic, se_point, piv, 0.95)
        return sg.periodic_to_annual(lo), sg.periodic_to_annual(hi)

    old_lo, _ = interval(old_mask)
    # Every observation in this series is POSITIVE, so an annualized Sharpe of
    # -91 is not a wide answer, it is an artefact.
    assert x.min() > 0.0
    assert old_lo < -50.0
    assert interval(ok)[0] > 0.0
    # ...and the excluded rows are BIASED, not a thinning: a resample that
    # missed the events has a near-constant mean, hence a Sharpe far above the
    # sample's, so every one of them sits on ONE side of the point estimate.
    assert bool((sr_b[~ok & old_mask] > moments.sharpe_periodic).all())
    # above the bar, so the interval from the survivors is not reported at all
    assert dropped / boot.shape[0] > sg.max_discarded_pivot_share(0.95)
    with pytest.raises(sg.ResampleRefusal):
        sg.bootstrap_sharpe_ci(series, seed=7, n_resamples=1000)
    # and assess_sharpe surfaces that instead of raising out of the pipeline
    out = sg.assess_sharpe(series, seed=7, n_resamples=1000)
    assert out.available is True and out.bootstrap is None
    assert out.standard_error is not None            # the parametric CI stands
    assert "No resampled interval is reported" in out.interpretation
    assert "biased" in out.interpretation


def test_the_sharpe_bootstrap_surfaces_a_sub_threshold_discard(
):
    """The other side of the same gate: a discard UNDER the bar is reported and
    qualified rather than refused, so the refusal keeps its meaning and the
    count keeps being surfaced. Six events instead of three.
    """
    series, x = _three_level_case(n_events=6)
    ci = sg.bootstrap_sharpe_ci(series, seed=7, n_resamples=1000)
    n = ci.extra["n_degenerate_resamples"]
    assert 0 < n / ci.n_resamples <= sg.max_discarded_pivot_share(0.95)
    assert ci.n_usable + n == ci.n_resamples
    assert f"{n:,}" in ci.interpretation
    assert "discarded" in ci.interpretation
    assert "lower bound on the uncertainty" in ci.interpretation
    assert x.min() > 0.0 and ci.ci_low > 0.0


def test_the_dropped_count_is_the_whole_exclusion_on_the_sharpe_path():
    """"Surfaced, not absorbed" DID NOT HOLD on the Sharpe path. The count only
    added up rows that were finite-and-degenerate, so on the rare-signal case it
    reported 0 while ``n_usable`` reported 618 of 1,000 — 382 resamples dropped,
    invisibly, and the caveat sentence never printed. A count that disagrees
    with the mask it describes is an absorbed data problem wearing the label of a
    surfaced one.
    """
    # The divergence needs a resample that is not merely degenerate but has NO
    # finite studentizer at all, which is where the old count went blind: a
    # rare-signal series against a ZERO benchmark has exact zeros, so a resample
    # that misses every event is exactly all-zero — sd is exactly 0, its HAC
    # long-run variance is exactly 0 and its Mertens numerator is not finite, so
    # ``isfinite(se_b) & (se_b > 0)`` is False and ``finite & degenerate`` counts
    # it as nothing. The mask still excludes it.
    strat, _ = _rare_signal_case(n_events=6, move=0.03)
    zero_bench = pd.Series(0.0, index=strat.index)
    excess = (strat - zero_bench).rename("excess")
    assert float(excess.min()) == 0.0        # the exact zeros are really there
    for ci in (sg.bootstrap_sharpe_ci(strat, seed=7, rf_daily=zero_bench,
                                      n_resamples=1000),
               sg.bootstrap_excess_return_ci(excess, seed=7, n_resamples=1000,
                                             label="cash")):
        n = ci.extra["n_degenerate_resamples"]
        assert n == ci.n_resamples - ci.n_usable, ci.statistic
        assert n > 0, ci.statistic
        assert f"{n:,}" in ci.interpretation
        assert "discarded" in ci.interpretation
    # ...and on a clean series the identity still holds at zero.
    clean = make_returns(793, seed=6201, mu=0.0004)
    for out in (sg.bootstrap_sharpe_ci(clean, seed=7, n_resamples=400),
                sg.bootstrap_excess_return_ci(clean, seed=7, n_resamples=400)):
        assert out.extra["n_degenerate_resamples"] == 0
        assert out.n_usable == out.n_resamples
        assert "discarded" not in out.interpretation


def test_the_discarded_resample_count_is_surfaced_not_absorbed():
    """House rule: a data problem surfaces. An interval computed from the
    resamples that survived is CONDITIONAL on surviving, which is a narrower
    question than the one the reader asked, and the report has to say so —
    below the share at which it stops being reportable at all."""
    strat, rf = _rare_signal_case(n_events=5, move=0.03)
    ci = sg.bootstrap_excess_return_ci((strat - rf).rename("excess"), seed=7,
                                       n_resamples=1000, label="cash")
    n = ci.extra["n_degenerate_resamples"]
    assert 0 < n / ci.n_resamples <= sg.max_discarded_pivot_share(0.95)
    assert f"{n:,}" in ci.interpretation
    assert "discarded" in ci.interpretation
    assert "lower bound on the uncertainty" in ci.interpretation
    assert ci.interpretation.count("%") >= 4
    # the note names the bar it is under, so a reader can tell a qualified
    # interval from a refused one
    assert f"{sg.max_discarded_pivot_share(0.95):.1%}" in ci.interpretation
    # and a clean series says nothing, so the caveat keeps its meaning
    clean = sg.bootstrap_excess_return_ci(
        make_returns(793, seed=6201, mu=0.0004), seed=7, n_resamples=400)
    assert clean.extra["n_degenerate_resamples"] == 0
    assert "discarded" not in clean.interpretation


# ------------------------- dropping is biased, so it is refused, not dropped

def test_the_survivorship_bar_is_derived_and_both_halves_are_live():
    """PRINCIPLE 1 at the resample level. What was here before was a PURE
    ABSOLUTE floor (``< 20 usable``) with no relative companion, which at 1,000
    resamples permitted 98% of the pivot distribution to be discarded — one
    sidedly — and still rendered a verdict.

    Both halves are derived from the SAME tail probability α/2, recomputed here,
    and both are shown to be the one that fires on some input.
    """
    for conf in (0.90, 0.95, 0.99):
        alpha = 1.0 - conf
        assert sg.max_discarded_pivot_share(conf) == pytest.approx(alpha / 2.0,
                                                                  rel=1e-15)
        # THE OFF-BY-ONE, FIXED. ``np.quantile``'s default linear method reads
        # the α/2 quantile at fractional INDEX (n−1)·α/2, not at n·α/2, so the
        # condition for the endpoint to reach PAST the smallest pivot drawn is
        # (n−1)·α/2 ≥ 1 — one resample more than the old derivation. Recomputed
        # against numpy itself below rather than argued.
        floor = sg.min_usable_pivots(conf)
        q = alpha / 2.0
        assert (floor - 1) * q >= 1.0 - 1e-12       # the condition holds...
        assert (floor - 2) * q < 1.0                # ...and is minimal
    assert sg.min_usable_pivots(0.95) == 41        # 40 under the old derivation
    assert sg.min_usable_pivots(0.90) == 21
    assert sg.min_usable_pivots(0.99) == 201
    # AND THE CONDITION IS THE ONE NUMPY ACTUALLY APPLIES: at the floor the 2.5%
    # quantile of a set of distinct pivots is at or above the SECOND smallest,
    # and one resample below the floor it is strictly between the smallest two.
    floor = sg.min_usable_pivots(0.95)
    for n, reaches in ((floor, True), (floor - 1, False)):
        piv = np.arange(float(n))
        assert (float(np.quantile(piv, 0.025)) >= 1.0) is reaches, n
    for bad in (0.0, 1.0, -0.5, 1.5):
        with pytest.raises(ValueError):
            sg.max_discarded_pivot_share(bad)

    # the RELATIVE half fires on a big pivot set that is materially thinned...
    rel = sg.survivorship_refusal(n_usable=960, n_dropped=40,
                                  n_resamples=1000, confidence=0.95)
    assert rel is not None and "biased" in rel and "4.0%" in rel
    # ...and the ABSOLUTE half on a small one that is barely thinned at all.
    absolute = sg.survivorship_refusal(n_usable=40, n_dropped=0,
                                       n_resamples=40, confidence=0.95)
    assert absolute is not None and "extrapolated" in absolute
    # neither is consulted alone, so they cannot disagree: the union stands
    assert sg.survivorship_refusal(975, 25, 1000, 0.95) is None
    assert sg.survivorship_refusal(41, 0, 41, 0.95) is None
    # ...and 40, the old floor, is now refused: that is the off-by-one, live
    assert sg.survivorship_refusal(40, 0, 40, 0.95) is not None


def test_discarding_a_degenerate_resample_is_one_sided_at_every_share():
    """WHY THE FIX IS A REFUSAL AND NOT A NARROWER INTERVAL. The resamples the
    row predicate excludes are exactly the ones in which the rare part of the
    series did not occur, so their means lie on ONE side of the point estimate.
    That is a BIASED subsample, not a smaller sample, and no amount of
    re-quantiling makes a truncated pivot distribution into a sampling
    distribution.

    Measured at four event counts, i.e. four discard shares spanning two orders:
    one-sidedness does not decay as the share shrinks — only the damage does.
    """
    shares = []
    for n_events in (1, 2, 3, 5):
        strat, rf = _rare_signal_case(n_events=n_events, move=0.05)
        x = (strat - rf).to_numpy(dtype=float)
        lag = sg.hac_lag(x, 1)
        idx = sg.stationary_bootstrap_indices(
            len(x), 1000, sg.bootstrap_block_days(len(x), 1), seed=7)
        boot = x[idx]
        means = boot.mean(axis=1)
        se_b = np.sqrt(sg._hac_long_run_variance_matrix(boot, lag) / len(x))
        ok, dropped = sg._usable_pivots(boot, means, se_b, "mean")
        assert dropped > 0, n_events
        # 100% of them below the point estimate, at every share
        assert bool((means[~ok] < x.mean()).all()), n_events
        shares.append(dropped / boot.shape[0])
    assert max(shares) > 0.3 and min(shares) < 0.01     # two orders apart
    # and the bar is CONSERVATIVE relative to where the fabrication becomes
    # visible: the endpoint does not collapse onto the point estimate until the
    # discarded share is an order of magnitude above the bar.
    assert max(shares) / sg.max_discarded_pivot_share(0.95) > 10.0


def test_nothing_in_the_fixtures_is_refused_by_the_survivorship_bar():
    """THE OVER-CORRECTION CHECK for the new bar. It has to be free on real
    data, and it is: not one real series discards a single resample.
    """
    window = slice("2022-12-01", "2026-01-30")
    extras = pd.read_csv(FIXTURES / "extras_prices.csv", index_col=0,
                         parse_dates=True).loc[window]
    rf_path = (extras["^IRX"] / 100.0 / 252.0)
    prices = pd.read_csv(FIXTURES / "sp500_prices.csv", index_col=0,
                         parse_dates=True).loc[window]
    tested = 0
    for col in list(prices.columns)[:40]:
        s = prices[col].pct_change().dropna()
        if s.size < 100:
            continue
        excess = (s - rf_path.reindex(s.index)).dropna().rename("excess")
        ci = sg.bootstrap_excess_return_ci(excess, seed=7, n_resamples=400,
                                           label="cash")
        assert ci.extra["n_degenerate_resamples"] == 0, col
        assert ci.n_usable == 400, col
        tested += 1
    assert tested >= 30
    # including the two flattest real objects in the fixtures
    for name, series in (("rf_path", rf_path),
                         ("AGG", extras["AGG"].pct_change().dropna())):
        ci = sg.bootstrap_excess_return_ci(series.rename("excess"), seed=7,
                                           n_resamples=400)
        assert ci.extra["n_degenerate_resamples"] == 0, name


def test_a_one_sided_interval_is_refused_by_its_own_algebraic_bound():
    """PRINCIPLE 4, at the site that FORMS the interval, as a backstop under the
    survivorship gate: an interval has TWO endpoints, and neither pivot quantile
    may be rounding residue of the other. An endpoint that IS the point estimate
    turns ``excludes_zero`` into ``sign(point)``.

    :data:`MAGNITUDE_DUST_RATIO` is DERIVED (√eps is the length-ratio companion
    of a variance-share floor at eps) and the gap it sits in is measured here
    rather than asserted in a comment.
    """
    assert sg.MAGNITUDE_DUST_RATIO == pytest.approx(
        math.sqrt(sg.DISPERSION_SHARE_FLOOR), rel=1e-15)

    # THE ARTEFACT, built as the bootstrap builds it: an ATOM at pivot zero wide
    # enough to swallow the 2.5% quantile. 4% of the resamples reproduced the
    # sample's own multiset, so their pivot is not near zero, it IS zero.
    atom = np.concatenate([np.zeros(40), np.linspace(0.5, 3.0, 960)])
    with pytest.raises(sg.ResampleRefusal) as exc:
        sg._studentized_interval(-1.5e-3, 4.0e-4, atom, 0.95)
    assert "one endpoint" in str(exc.value)
    # ...and the direction of the fabricated verdict is exactly sign(point):
    # with the bound removed the interval's upper end is the point itself.
    q_lo = float(np.quantile(atom, 0.025))
    assert q_lo == 0.0
    assert -1.5e-3 - 4.0e-4 * q_lo == -1.5e-3

    # A NEARLY-zero quantile is caught too, which is the case equality alone
    # misses: the measured collapse produced q_lo = -5.5e-16 against q_hi ≈ 2,
    # i.e. an endpoint 128 ulps from the point estimate rather than 0 ulps.
    near = np.concatenate([np.full(40, -5.5e-16), np.linspace(0.5, 3.0, 960)])
    assert float(np.quantile(near, 0.025)) != 0.0
    with pytest.raises(sg.ResampleRefusal):
        sg._studentized_interval(-1.5e-3, 4.0e-4, near, 0.95)

    # THE REAL SIDE, and the margin, recomputed on genuine fixture series.
    window = slice("2022-12-01", "2026-01-30")
    prices = pd.read_csv(FIXTURES / "sp500_prices.csv", index_col=0,
                         parse_dates=True).loc[window]
    tightest, n = 1.0, 0
    for col in list(prices.columns)[:40]:
        s = prices[col].pct_change().dropna()
        if s.size < 100:
            continue
        ci = sg.bootstrap_excess_return_ci(s.rename("excess"), seed=3,
                                           n_resamples=400)
        near_half = min(ci.point - ci.ci_low, ci.ci_high - ci.point)
        tightest = min(tightest, near_half / (ci.ci_high - ci.ci_low))
        n += 1
    assert n >= 30
    assert tightest > 0.2                       # the tightest real interval
    assert tightest / sg.MAGNITUDE_DUST_RATIO > 1e6
    assert sg.MAGNITUDE_DUST_RATIO / 2.0e-16 > 1e6

    # AND IT IS NOT THE STRONGER FORM. ``ci_low < point < ci_high`` was written
    # first and measured to over-correct: it refuses the committed
    # non-positive-Mertens-numerator construction, whose pivots are all
    # positive, so its interval legitimately sits entirely below its own point.
    # Both of its quantiles are real numbers, so this bound accepts it.
    s = _mertens_numerator_negative_series(n=120, k=59)
    ci = sg.bootstrap_sharpe_ci(s, seed=1, n_resamples=400)
    assert ci.ci_high < ci.point                 # the strong form would refuse
    assert ci.ci_low < ci.ci_high
    assert ci.extra["se_method"] == "lo_iid_fallback"


def test_the_sharpe_bootstrap_sentence_is_not_sign_blind():
    """A NEGATIVE Sharpe whose interval excludes zero was told "the
    risk-adjusted edge survives resampling" — in the same report in which
    ``sharpe_standard_error`` said of the same input "that interval contains
    zero, so this Sharpe is not statistically distinguishable from no edge".
    Measured: point Sharpe -0.33, bootstrap interval [-1.67, -0.33].

    ``excludes_zero`` is one bit and the sentence needs two, so all three states
    are exercised here and the negative one must not use the word "edge" for
    what it found.
    """
    idx = pd.bdate_range("2015-01-02", periods=793)
    rng = np.random.default_rng(6401)
    cases = {}
    for name, mu in (("loser", -0.001), ("winner", 0.001), ("tie", 0.0)):
        r = pd.Series(rng.normal(mu, 0.005, 793), index=idx)
        cases[name] = sg.bootstrap_sharpe_ci(r, seed=5, n_resamples=400)

    lose = cases["loser"]
    assert lose.ci_high < 0.0 and lose.excludes_zero is True
    assert "SHORTFALL" in lose.interpretation
    assert "edge survives" not in lose.interpretation
    assert "underperforms" in lose.interpretation

    win = cases["winner"]
    assert win.ci_low > 0.0 and win.excludes_zero is True
    assert "edge survives resampling" in win.interpretation
    assert "SHORTFALL" not in win.interpretation

    tie = cases["tie"]
    assert tie.excludes_zero is False
    assert "contains zero" in tie.interpretation
    # the three sentences are three sentences, not one bit rendered twice
    assert len({c.interpretation.split(". ")[-1] for c in cases.values()}) == 3


def test_assess_excess_refuses_when_the_hac_statistic_is_not_measurable(
        monkeypatch):
    """A DEFENSIVE BRANCH, pinned because it is the only one that stands between
    an unmeasurable HAC standard error and ``classify_excess`` being handed a NaN
    endpoint — where ``ci_low > 0`` is False, ``ci_high < 0`` is False, and the
    reader is told the two are "not distinguishable".

    It is not reachable from data in this tree: over 3,000 constructed series
    (alternating signs, extreme scales, two-point laws) not one passed
    :func:`degeneracy_refusal` and then produced a non-finite t. So it is
    exercised by substituting the HAC site, which is the honest way to pin a
    guard whose input no input reaches — an untested guard is how the
    ``se > 0`` shape survived six rounds.
    """
    a = make_returns(300, seed=6402, mu=0.0005)
    b = make_returns(300, seed=6403, mu=0.0002)
    real = sg.newey_west_tstat
    assert real((a - b).rename("excess")).available is True

    def unmeasurable(excess, **kw):
        out = real(excess, **kw)
        # Built through HACResult.refuse — i.e. through the chokepoint, the way
        # _unavailable_hac builds one. Handing the available payload's fields
        # (and its available payload's PROSE) to a result marked available=False
        # is itself refused, by gate_prose, which is the point of the gate: a
        # refusal carrying a testable fit's sentences is the defect, not the fake.
        return sg.HACResult.refuse(
            **{**asdict(out),
               "autocorr_lag1": float("nan"),
               "available": False,
               "unavailable_reason": "the long-run variance came out zero",
               "interpretation": (
                   "No t-statistic, no p-value and no interval are reported "
                   "against SPY: the long-run variance came out zero.")})

    monkeypatch.setattr(sg, "newey_west_tstat", unmeasurable)
    out = sg.assess_excess(a, b, seed=1, n_resamples=100, label="SPY")
    assert out.available is False
    assert out.verdict == sg.TIES_WITHIN_NOISE       # pinned, not a verdict
    assert out.verdict_source == "none"
    assert out.ci_coverage_caveat is None
    assert "the long-run variance came out zero" in out.unavailable_reason
    assert "it is a refusal" in out.interpretation
    assert json.dumps(out.to_dict(), allow_nan=False)


def test_the_removed_studentizer_ratio_condition_was_behaviourally_inert():
    """WHY A CONDITION WAS DELETED RATHER THAN PINNED. ``_usable_pivots`` used to
    carry a third test — the row's studentizer not being dust relative to the
    point estimate's — and no input reached it: every row it would refuse was
    already refused by :func:`degenerate_rows`, so it was pinned only by
    re-asserting its own value and a mutation to it could not fail.

    Recomputed here over the rare-signal family rather than argued. The derived
    ratio itself was not deleted; it moved to :func:`_studentized_interval`,
    where the test above shows it load-bearing on both sides.
    """
    fired_alone = 0
    for n_events in (1, 2, 3, 5, 10):
        for move in (0.02, 0.10):
            strat, rf = _rare_signal_case(n_events=n_events, move=move)
            x = (strat - rf).to_numpy(dtype=float)
            lag = sg.hac_lag(x, 1)
            idx = sg.stationary_bootstrap_indices(
                len(x), 400, sg.bootstrap_block_days(len(x), 1), seed=7)
            boot = x[idx]
            se_b = np.sqrt(sg._hac_long_run_variance_matrix(boot, lag) / len(x))
            se_point = math.sqrt(sg._hac_long_run_variance(x, lag) / len(x))
            finite = np.isfinite(boot.mean(axis=1)) & np.isfinite(se_b) & (se_b > 0)
            degenerate = sg.degenerate_rows(boot)
            would_refuse = ~(se_b > se_point * sg.MAGNITUDE_DUST_RATIO)
            fired_alone += int((finite & would_refuse & ~degenerate).sum())
            # and the condition it was meant to be, the dust, IS caught
            assert bool(degenerate[se_b < se_point * 1e-10].all())
    assert fired_alone == 0


# ------------------------------------ the band where the two halves disagreed

def _one_tick_band_case(t: int = 793, ticks: int = 1):
    """An all-cash strategy against a cash path that moves by ONE real ^IRX
    quote tick. Nothing synthetic about the granularity: ^IRX is quoted to
    0.001 in percent, i.e. 3.968e-08 as a daily rate, asserted here.
    """
    irx = pd.read_csv(FIXTURES / "extras_prices.csv",
                      index_col=0, parse_dates=True)["^IRX"]
    grid = float(np.abs(np.diff(np.unique(irx.dropna().to_numpy()))).min())
    assert grid == pytest.approx(0.001, abs=1e-5), "fixture quote grain changed"
    tick = 0.001 / 100.0 / 252.0
    idx = _fixture_sessions()[:t]
    rf = np.full(t, 0.0002)
    rf[-ticks:] = 0.0002 + tick
    return pd.Series(0.0, index=idx), pd.Series(rf, index=idx)


def test_every_rendering_site_applies_both_halves_of_the_policy():
    """DEFECT 2 — the two gates disagreed by NINE ORDERS OF MAGNITUDE and the
    LOOSER one was the site that renders the claim.

    ``assess_excess`` applied the share floor AND the absolute floor;
    ``newey_west_tstat`` and ``bootstrap_excess_return_ci`` applied only the
    share floor. In the band between them (share > 2.2e-16 with sd <= 1e-7) the
    bundled entry REFUSED while both rendering sites FABRICATED:

        newey_west_tstat -> available=True, "a serial-correlation-robust
          t-statistic of -399678.09 ... That clears the conventional |t| > 2 bar"
        bootstrap        -> [-1239964.96%, -5.04%], excludes_zero=True,
          "The interval excludes zero."

    A gate at the entry that is missing at the renderer means the entry refuses
    while the renderer fabricates. Reachable at real ^IRX quote granularity.
    """
    strat, rf = _one_tick_band_case()
    excess = (strat - rf).rename("excess")
    x = excess.to_numpy(dtype=float)

    # The input really is IN THE BAND: the share half passes it by five orders
    # of magnitude, the absolute half refuses it.
    assert sg.dispersion_share(x) / sg.DISPERSION_SHARE_FLOOR > 1e5
    assert float(np.std(x, ddof=1)) < sg.ABSOLUTE_SD_FLOOR
    assert sg.degeneracy_refusal(x).kind == "absolute"

    # 1. the HAC site
    hac = sg.newey_west_tstat(excess, label="cash")
    assert hac.available is False
    assert math.isnan(hac.t_stat) and math.isnan(hac.ci_low_annualized)
    for text in (hac.interpretation, hac.unavailable_reason):
        assert "clears the conventional" not in text
        assert "not distinguishable from noise" not in text
        assert not re.search(r"t-statistic of\s*[-+]?[\d.]", text)
        assert "nan" not in text.lower()
    assert json.dumps(hac.to_dict(), allow_nan=False)

    # 2. the bootstrap site
    with pytest.raises(ValueError, match="dispersion"):
        sg.bootstrap_excess_return_ci(excess, seed=1, n_resamples=200,
                                      label="cash")

    # 3. the bundled entry — which already refused, and still does
    out = sg.assess_excess(strat, rf, seed=1, n_resamples=200, label="cash")
    assert out.available is False and out.verdict_source == "none"
    assert out.verdict == sg.TIES_WITHIN_NOISE

    # 4. the three sites agree about WHICH input they refuse, across the band.
    for t_obs in (30, 126, 253, 793):
        for ticks in (1, 2, 3):
            s, b = _one_tick_band_case(t=t_obs, ticks=ticks)
            e = (s - b).rename("excess")
            refused = sg.degeneracy_refusal(e.to_numpy(dtype=float)) is not None
            assert sg.newey_west_tstat(e, label="cash").available is not refused
            boot_refused = False
            try:
                sg.bootstrap_excess_return_ci(e, seed=1, n_resamples=100)
            except ValueError as exc:
                boot_refused = "dispersion" in str(exc) or "floor" in str(exc)
            assert boot_refused is refused, (t_obs, ticks)
            assert sg.assess_excess(s, b, seed=1, n_resamples=100,
                                    label="cash").available is not refused


def test_the_refusal_names_what_is_actually_true_of_the_data():
    """DEFECT 3 — the wrong-reason refusal, one input over.

    In the band the share branch did not fire, so control reached the
    absolute-floor branch, which returned the tracks-exactly text:

        "the strategy tracks cash exactly - the difference series has no
         dispersion, so there is nothing to test"

    about a series that differs from cash by -5.04% annualized on ALL 793
    sessions, and ``pipeline._significance_findings`` printed that sentence to
    the reader. 62 of 108 near-constant cases took that branch. The lead clause
    is now chosen by the MEAN and the explanation by the KIND, so neither can
    be chosen by which floor happened to fire.
    """
    strat, rf = _one_tick_band_case()
    out = sg.assess_excess(strat, rf, seed=1, n_resamples=200, label="cash")
    reason = out.unavailable_reason
    assert "tracks cash exactly" not in reason
    assert "differs from cash" in reason
    assert f"{-0.0002 * PERIODS:+.2%}" in reason          # the real -5.04%
    assert "793" in reason
    assert f"{sg.ABSOLUTE_SD_FLOOR:.1e}" in reason        # and WHICH bar bound
    assert "standard deviation" in reason

    # The same input at the HAC site tells the same story, from the same text.
    hac = sg.newey_west_tstat((strat - rf).rename("excess"), label="cash")
    assert hac.unavailable_reason == reason
    with pytest.raises(ValueError) as exc:
        sg.bootstrap_excess_return_ci((strat - rf).rename("excess"), seed=1,
                                      n_resamples=200, label="cash")
    assert reason in str(exc.value)

    # And "tracks exactly" is still SAID when it is true — the distinction the
    # lead clause draws is the mean, not the floor.
    idx = _fixture_sessions()
    same = pd.Series(0.0002, index=idx)
    zero = sg.assess_excess(same, same.copy(), seed=1, n_resamples=200,
                            label="cash")
    assert "tracks cash exactly" in zero.unavailable_reason
    # ...and a nonzero CONSTANT, refused by the other half, names its constant.
    const = sg.assess_excess(pd.Series(0.0, index=idx), same, seed=1,
                             n_resamples=200, label="cash")
    assert "tracks cash exactly" not in const.unavailable_reason
    assert f"{sg.DISPERSION_SHARE_FLOOR:.1e}" in const.unavailable_reason
    assert "no dispersion" in const.unavailable_reason


def test_every_site_in_the_module_reaches_the_predicate():
    """PRINCIPLE 2, proved by GREP rather than by assertion.

    Every function in this module that forms or renders a t-statistic, an
    interval or a verdict must reach :func:`degeneracy_refusal` — directly or
    through the one hop the module allows (``_degenerate_sharpe``, the
    ``ReturnMoments`` gate, and ``degenerate_rows``, the row form). This test
    reads the source and fails if a new site appears without one.
    """
    import ast

    src = Path(sg.__file__).read_text()
    tree = ast.parse(src)
    bodies: dict[str, str] = {}
    moments_gated: set[str] = set()
    sees_data: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            stmts = node.body
            if (stmts and isinstance(stmts[0], ast.Expr)
                    and isinstance(stmts[0].value, ast.Constant)
                    and isinstance(stmts[0].value.value, str)):
                stmts = stmts[1:]        # drop the docstring: prose is not code
            body = "\n".join(ast.unparse(s) for s in stmts)
            bodies[node.name] = body
            args = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
            anns = " ".join(ast.unparse(a.annotation) for a in args
                            if a.annotation is not None)
            # A function whose series argument is already a ReturnMoments is
            # gated at CONSTRUCTION (try_from_returns -> _degenerate_sharpe ->
            # the predicate); it never sees a raw series to test.
            if "ReturnMoments" in anns:
                moments_gated.add(node.name)
            # A function that can see raw data is one that converts a series or
            # is handed an array. Anything else is a pure formatter: it has no
            # series to apply a predicate to.
            if ("_series(" in body or "ndarray" in anns
                    or "pd.Series" in anns or "np.ndarray" in anns):
                sees_data.add(node.name)
    assert len(bodies) > 30
    assert {"sharpe_standard_error", "probabilistic_sharpe_ratio",
            "sharpe_difference_test"} <= moments_gated

    # The renderers, identified by what they REACH FOR, not by a hand-kept list,
    # so a NEW site that renders a claim about an interval fails this test — and
    # transitively closed over CALLERS, so moving a sentence into a helper does
    # not move its caller out of scope. These are the chokepoint functions: a
    # site cannot form a critical value, a three-state verdict or a sentence
    # about one without calling one of them.
    phrases = ("classify_excess(", "zero_relation_sentence(", "critical_z(")
    rendering = {n for n, b in bodies.items() if any(p in b for p in phrases)}
    for _ in range(len(bodies)):
        grown = {n for n, b in bodies.items()
                 if any(f"{r}(" in b for r in rendering)} | rendering
        if grown == rendering:
            break
        rendering = grown
    assert {"newey_west_tstat", "bootstrap_excess_return_ci",
            "bootstrap_sharpe_ci", "assess_excess", "_bar_sentence"} <= rendering

    # The predicate itself, its row form, the resample-mask helper that calls
    # the row form, and the ReturnMoments constructors (which go through
    # _degenerate_sharpe). Those are the only hops the module allows.
    allowed = ("degeneracy_refusal(", "_usable_pivots(", "degenerate_rows(",
               "ReturnMoments.try_from_returns(", "ReturnMoments.from_returns(")
    checked = 0
    for name in sorted(rendering & sees_data):
        assert (name in moments_gated
                or any(call in bodies[name] for call in allowed)), name
        checked += 1
    assert checked >= 4, rendering & sees_data

    # ============================ THE CHOKEPOINT ============================
    # ONE CRITICAL VALUE. The defect this round exists for is that there were
    # two, three lines apart: the interval at Phi-inverse(0.5 + conf/2) =
    # 1.9599639845 and the sentence beside it at a folklore 2.0. They disagree
    # for every |t| in [1.95996, 2.0), and that window is reachable. So the
    # quantile is DERIVED IN EXACTLY ONE BODY and nothing else spells a bar.
    code = "\n\n".join(bodies.values())
    derivers = sorted(n for n, b in bodies.items() if "norm_ppf(0.5 +" in b)
    assert derivers == ["critical_z"], derivers
    assert "|t|" not in code, "a |t| bar respelled outside critical_z"
    for name, body in bodies.items():
        if name == "critical_z":
            continue
        assert "1.96" not in body, name
        assert not re.search(r"abs\([a-z_]*t[a-z_]*\)\s*[<>]=?\s*2", body), name

    # ONE THREE-STATE COMPARISON. No site may keep a private ``ci_low > 0`` /
    # ``ci_high < 0`` pair — that is how ``excludes_zero`` and the sentence
    # beside it drifted apart, and how the sign-blind sentence survived being
    # fixed at one of its three sites.
    for name, body in bodies.items():
        if name == "classify_excess":
            continue
        assert not re.search(r"\b(lo|ci_low|ci_lo)\b\s*>\s*0", body), name
        assert not re.search(r"\b(hi|ci_high|ci_hi)\b\s*<\s*0", body), name

    # ONE FIELD GATE, and it is the ONLY way a payload is serialized: a
    # ``to_dict`` that builds its own dict is a payload outside the chokepoint.
    to_dicts = [b for n, b in bodies.items() if n == "to_dict"]
    assert to_dicts, "to_dict disappeared"
    for name, body in bodies.items():
        if name != "to_dict":
            continue
        assert "gated_dict(" in body, body
    # ...and nothing bypasses it by reaching for the JSON helper directly.
    for name, body in bodies.items():
        if name in ("gated_dict", "_jsonable"):
            continue
        assert "_jsonable(" not in body, name

    # ONE PROSE GATE, reached only through the field gate, so a payload cannot
    # be gated without its sentences being gated too.
    assert "gate_prose(" in bodies["_gate_leaf"]
    assert "_LEAKED_STATISTIC" in bodies["gate_prose"]
    # ...and the prose gate is NAME-BLIND: every string leaf goes through it, so
    # a prose field added tomorrow under a new name is gated too. A list of prose
    # field names is the shape that let ``method_note`` through.
    assert "PROSE_FIELDS" not in bodies["_gate_leaf"]

    # ...and the two hops the module allows are really wired to the predicate.
    assert "degeneracy_refusal(" in bodies["_degenerate_sharpe"]
    assert "degenerate_rows(" in bodies["_usable_pivots"]
    # No site may keep a private second opinion on either floor: the absolute
    # half is reachable ONLY through the predicate, never by re-testing
    # metrics._degenerate or by respelling a threshold.
    for name, body in bodies.items():
        if name in ("degeneracy_refusal", "_degenerate_sharpe"):
            continue
        assert "metrics._degenerate(" not in body, name
        assert "DISPERSION_SHARE_FLOOR" not in body or name in (
            "degenerate_rows", "degeneracy_reason"), name
    assert "1e-7" not in code and "np.finfo" not in code


def test_the_two_halves_of_the_policy_bracket_the_dust_and_the_real_series():
    """THE OVER-CORRECTION CHECK for the ABSOLUTE half, which is the half that
    can mis-calibrate — the engine's inverse-vol floor sat 250x above its threat
    and bound on a real asset class, and this project has paid for that once.

    Every margin below is RECOMPUTED from the committed fixtures. The honest
    summary is that this half is one order of magnitude clear of the flattest
    object the fixtures can produce, not "orders of magnitude clear of anything
    real" — and that number belongs in a test, not in a comment.

    CORRECTED EVIDENCE NOTE. An earlier round quoted "t = -9.54" as this
    module's figure for the all-zero strategy over the varying-rf 2016-2024
    window. That number is not from this module: -9.54 is ``attribution``'s FF6
    regression alpha t-statistic on that fit (CAPM -9.51, FF3 -9.52, FF5 -9.54,
    FF6 -9.54, from ``test_attribution.py``), whose denominator is a HAC
    standard error on regression residuals against six factors. THIS module's
    Newey-West t-statistic on the same difference series is t = -3.77 at lag
    154 over 2,264 sessions, recomputed below. Both agree the fit is testable,
    which is what the evidence was cited for; the number was the sibling's.
    """
    floor = sg.ABSOLUTE_SD_FLOOR
    assert floor == metrics._MIN_DAILY_SD, "the two modules must share one floor"

    window = slice("2022-12-01", "2026-01-30")
    irx = pd.read_csv(FIXTURES / "extras_prices.csv",
                      index_col=0, parse_dates=True)["^IRX"]
    rf_path = (irx / 100.0 / 252.0).loc[window]

    # --- REAL SIDE, flattest first.
    margins = {}
    margins["irx_rf_path"] = float(rf_path.std(ddof=1)) / floor
    # the tightest thing constructible from real quotes at all: the daily
    # DIFFERENCE of that path, which is not an input this module is ever given
    margins["irx_rf_path_diff"] = float(rf_path.diff().dropna().std(ddof=1)) / floor
    extras = pd.read_csv(FIXTURES / "extras_prices.csv", index_col=0,
                         parse_dates=True).loc[window]
    margins["flattest_etf"] = min(
        float(extras[c].pct_change().dropna().std(ddof=1)) / floor
        for c in ("AGG", "DIA", "QQQ"))
    prices = pd.read_csv(FIXTURES / "sp500_prices.csv", index_col=0,
                         parse_dates=True).loc[window]
    rets = prices.pct_change().dropna(how="all")
    margins["flattest_sp500_name"] = float(
        rets.std(ddof=1).min()) / floor
    margins["flattest_sp500_excess"] = float(
        rets.sub(rf_path, axis=0).std(ddof=1).min()) / floor

    assert margins["irx_rf_path_diff"] > 10.0, margins
    assert margins["irx_rf_path"] > 200.0, margins
    assert margins["flattest_etf"] > 1e4, margins
    assert margins["flattest_sp500_name"] > 1e4, margins
    assert margins["flattest_sp500_excess"] > 1e4, margins
    assert min(margins.values()) > 1.0, margins      # nothing real is refused

    # THE NUMBERS THE MODULE COMMENT STATES, pinned to the construction it names
    # rather than left to drift. An earlier version of that comment carried three
    # that belong to a different construction: "502" series, an excess-over-rf
    # minimum of 9.94e-03 ABOVE the raw minimum of 9.33e-03 (subtracting a
    # near-constant rf path cannot raise a standard deviation by 7%), and
    # 3.18e-03 for AGG. All four quantities are recomputed here.
    usable = rets.std(ddof=1).dropna()
    assert prices.shape[1] == 504 and usable.size == 503
    assert float(usable.min()) == pytest.approx(9.331e-03, rel=2e-3)
    assert usable.idxmin() == rets.sub(rf_path, axis=0).std(ddof=1).idxmin()
    assert float(rets.sub(rf_path, axis=0).std(ddof=1).min()) == pytest.approx(
        float(usable.min()), rel=1e-4)       # the SAME name, not 7% apart
    assert float(extras["AGG"].pct_change().dropna().std(ddof=1)) == (
        pytest.approx(3.744e-03, rel=2e-3))
    assert float(rf_path.std(ddof=1)) == pytest.approx(2.220e-05, rel=2e-3)
    assert float(rf_path.diff().dropna().std(ddof=1)) == pytest.approx(
        1.125e-06, rel=2e-3)
    # ...and the share-floor comment's own real-side minimum, same window.
    shares = {c: sg.dispersion_share(rets[c].dropna().to_numpy())
              for c in rets.columns if rets[c].notna().sum() >= 2}
    assert len(shares) == 503
    assert min(shares.values()) == pytest.approx(0.956, abs=5e-3)
    # ...and every one of them stays testable end to end, at both sites.
    for series in (rf_path, extras["AGG"].pct_change().dropna()):
        assert sg.degeneracy_refusal(series.to_numpy()) is None
        assert sg.newey_west_tstat(series, label="cash").available is True

    # --- ARTEFACT SIDE: the band this half covers reaches down to the 5e-21
    # dust a constant resample carries, so it spans 14 orders below the floor.
    const = np.full(793, -2e-4)
    dust_sd = float(np.std(const, ddof=1))
    # Not zero — an exact constant still carries rounding from the two-pass
    # mean, which is precisely why ``sd > 0`` and ``se > 0`` never refused it.
    assert 0.0 < dust_sd < 1e-15
    assert floor / dust_sd > 1e11, dust_sd
    assert sg.degeneracy_refusal(const).kind == "constant"

    # --- the corrected evidence note, recomputed rather than quoted.
    factors = pd.read_csv(FIXTURES / "factors_ff5_daily.csv", comment="#",
                          index_col=0, parse_dates=True).loc["2016-01-01":"2024-12-31"]
    rf = factors["RF"] if "RF" in factors else factors["rf"]
    assert rf.nunique() > 1, "the window must really have a varying rf"
    diff = (pd.Series(0.0, index=rf.index) - rf).rename("excess")
    hac = sg.newey_west_tstat(diff, label="cash")
    assert hac.available is True
    assert hac.t_stat == pytest.approx(-3.77, abs=0.05), hac.t_stat
    assert hac.n_obs == 2264 and hac.lag == 154
    assert float(diff.std(ddof=1)) / floor > 800.0     # nowhere near the floor
    assert sg.dispersion_share(diff.to_numpy()) > 0.5


# --------------------------------------------------- the algebraic bounds

def test_an_impossible_dispersion_share_is_refused_not_compared_to_a_floor():
    """PRINCIPLE 4. ``share = Σ(x−x̄)²/Σx²`` lies in [0, 1] by construction
    (``Σx² = Σ(x−x̄)² + T·x̄²``), so a value outside it is broken arithmetic, not
    a degenerate fit — and an impossible value is the cheapest degeneracy
    detector there is, needing no calibration at all. The sibling module makes
    the same check on ``resid_var_share <= 1.0``.
    """
    # the bound holds on everything real, including the pathological cases
    for x in (np.zeros(50), np.full(50, 3.0), np.linspace(-1, 1, 50),
              make_returns(500, seed=6202).to_numpy(),
              np.concatenate([np.zeros(49), [1e9]])):
        s = sg.dispersion_share(x)
        assert -1e-9 <= s <= 1.0 + 1e-9, s

    class _Impossible(np.ndarray):
        pass

    # driven through the predicate directly, since the arithmetic cannot
    # produce it: a share of 3.5 must be named impossible, not "constant".
    real = sg.dispersion_share
    try:
        sg.dispersion_share = lambda x: 3.5
        deg = sg.degeneracy_refusal(np.linspace(-1.0, 1.0, 50))
        assert deg is not None and deg.kind == "impossible"
        reason = sg.degeneracy_reason(deg, -2e-4, "cash")
        assert "algebraically possible" in reason and "3.5" in reason
        assert "tracks cash exactly" not in reason
        hac = sg.newey_west_tstat(pd.Series(np.linspace(-1.0, 1.0, 50)))
        assert hac.available is False
        assert "algebraically possible" in hac.unavailable_reason
    finally:
        sg.dispersion_share = real


def test_a_non_finite_or_inverted_interval_is_refused_not_classified_as_a_tie():
    """The VERDICT site is a rendering site. ``ci_low > 0.0`` is False for a
    NaN, so an unguarded NaN interval classifies as ``ties_within_noise`` — the
    reassuring branch, reached by a sample with no statistics at all."""
    assert sg.classify_excess(0.05, 0.01, 0.09) == sg.BEATS
    for bad in ((float("nan"), 0.01, 0.09), (0.05, float("nan"), 0.09),
                (0.05, 0.01, float("nan")), (float("inf"), 0.01, 0.09)):
        with pytest.raises(ValueError, match="non-finite"):
            sg.classify_excess(*bad)
    with pytest.raises(ValueError, match="inverted"):
        sg.classify_excess(0.05, 0.09, 0.01)


def test_the_studentized_interval_asserts_its_own_algebraic_bound():
    """``q_lo <= q_hi`` and ``se_point >= 0`` make ``ci_low <= ci_high``
    identical; a violated identity is broken arithmetic and is raised rather
    than handed on to be classified."""
    pivots = np.linspace(-2.0, 2.0, 101)
    lo, hi, straddles = sg._studentized_interval(0.001, 0.0005, pivots, 0.95)
    assert lo <= hi
    assert straddles is True                    # these pivots do straddle zero
    with pytest.raises(ValueError, match="algebraic bound"):
        sg._studentized_interval(0.001, -1.0, pivots, 0.95)
    with pytest.raises(ValueError, match="algebraic bound"):
        sg._studentized_interval(float("nan"), 0.0005, pivots, 0.95)


def test_a_resample_mean_outside_the_data_hull_is_refused():
    """Every bootstrap resample is an average of OBSERVED values, so its mean
    cannot leave ``[min x, max x]``. Outside it, the index matrix is wrong —
    and that is a data/plumbing problem, not a wider interval."""
    x = np.linspace(-0.02, 0.03, 200)
    boot = x[np.random.default_rng(6203).integers(0, 200, size=(50, 200))]
    means = boot.mean(axis=1)
    se_b = np.full(50, 1e-3)
    ok, _ = sg._usable_pivots(boot, means, se_b, "mean")
    assert ok.all()
    with pytest.raises(ValueError, match="hull"):
        sg._usable_pivots(boot, means + 1.0, se_b, "mean")


def test_the_bar_sentence_cannot_be_widened_to_absorb_a_non_finite_t():
    """The NaN-into-the-tie-branch code, pinned at the SENTENCE.

    ``if abs(tstat) >= 2.0`` is False for a NaN, so a NaN prints "That is short
    of the conventional |t| > 2 bar, so the gap is not distinguishable from
    noise" — a verdict. The reviewer's mutant is the other direction, widening
    the test to ``abs(tstat) >= 2.0 or not math.isfinite(tstat)``, which prints
    "That clears the conventional |t| > 2 bar" for a statistic that does not
    exist. Neither is reachable through a bare ``if``, which is why the
    implementation checks the EXCLUSIVITY of the two comparisons instead: for
    any real t exactly one holds, for a NaN neither does, and widening either
    one breaks the exclusivity and is caught.
    """
    # THE SENTENCE IS NOW READ OFF THE INTERVAL, so it is exercised the way the
    # module builds it: a symmetric interval at THE critical value.
    def sentence(t, *, conf=0.95, se=1.0):
        z = sg.critical_z(conf)
        mean = t * se
        return sg._bar_sentence(t, mean, mean - z * se, mean + z * se, conf)

    assert "reaches it" in sentence(3.0)
    assert "reaches it" in sentence(-7.5)
    assert "does not reach" in sentence(1.9)
    assert "does not reach" in sentence(0.0)
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="refusing to pick"):
            sentence(bad)
    # and the sentence really is the one newey_west_tstat renders
    hac = sg.newey_west_tstat(make_returns(400, seed=6205, mu=0.002),
                              label="cash")
    assert sg._bar_sentence(hac.t_stat, hac.mean_annualized,
                            hac.ci_low_annualized, hac.ci_high_annualized,
                            hac.confidence) in hac.interpretation


def test_a_non_finite_t_statistic_is_a_refusal_not_the_reassuring_branch():
    """``abs(t) >= 2.0`` is False for a NaN, so an unguarded NaN t picks "That
    is short of the conventional |t| > 2 bar, so the gap is not distinguishable
    from noise" — a fabricated TIE from a sample with no statistic at all.

    Forced by collapsing the long-run variance under a series that PASSES the
    predicate, which is the only way this branch is reachable at all.
    """
    series = make_returns(300, seed=6204, mu=0.0005)
    assert sg.degeneracy_refusal(series.to_numpy()) is None
    real = sg._hac_long_run_variance
    for collapsed in (0.0, float("nan")):
        try:
            sg._hac_long_run_variance = lambda x, lag: collapsed
            hac = sg.newey_west_tstat(series, label="cash")
        finally:
            sg._hac_long_run_variance = real
        assert hac.available is False, collapsed
        assert math.isnan(hac.t_stat)
        for text in (hac.interpretation, hac.unavailable_reason):
            assert "not distinguishable from noise" not in text
            assert "clears the conventional" not in text
            assert "nan" not in text.lower()
        assert json.dumps(hac.to_dict(), allow_nan=False)
    # and the guard did not swallow the ordinary case
    assert sg.newey_west_tstat(series, label="cash").available is True


# ---------------------------------------------------------------- bootstrap

def test_stationary_bootstrap_indices_are_in_range_and_form_blocks():
    idx = sg.stationary_bootstrap_indices(500, 200, 10.0, seed=1)
    assert idx.shape == (200, 500)
    assert idx.min() >= 0 and idx.max() < 500
    # With E[block]=10 roughly 90% of consecutive draws continue the block.
    cont = (np.diff(idx, axis=1) % 500) == 1
    assert 0.80 < cont.mean() < 0.97


def test_stationary_bootstrap_is_deterministic_given_a_seed():
    a = sg.stationary_bootstrap_indices(300, 50, 8.0, seed=123)
    b = sg.stationary_bootstrap_indices(300, 50, 8.0, seed=123)
    c = sg.stationary_bootstrap_indices(300, 50, 8.0, seed=124)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_bootstrap_sharpe_ci_is_reproducible_and_brackets_the_point_estimate():
    r = make_returns(1000, seed=51, mu=0.0005)
    a = sg.bootstrap_sharpe_ci(r, seed=2024, n_resamples=400)
    b = sg.bootstrap_sharpe_ci(r, seed=2024, n_resamples=400)
    assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)
    assert a.ci_low < a.point < a.ci_high
    assert a.point == pytest.approx(metrics.sharpe(r), rel=1e-12)
    assert sg.bootstrap_sharpe_ci(r, seed=7, n_resamples=400).ci_low != a.ci_low


def test_bootstrap_sharpe_ci_covers_the_true_sharpe_at_about_the_nominal_rate():
    """Monte-Carlo coverage check, fully seeded. A studentized stationary
    bootstrap at T=504 should land near the nominal 95%."""
    true_sr_annual = 1.0
    mu = true_sr_annual / math.sqrt(PERIODS) * DAILY_VOL
    idx = pd.bdate_range("2015-01-02", periods=504)
    covered = 0
    reps = 120
    for i in range(reps):
        rng = np.random.default_rng(1000 + i)
        r = pd.Series(rng.standard_normal(504) * DAILY_VOL + mu, index=idx)
        ci = sg.bootstrap_sharpe_ci(r, seed=5000 + i, n_resamples=250,
                                    expected_block_days=5.0)
        covered += ci.ci_low <= true_sr_annual <= ci.ci_high
    assert 0.88 <= covered / reps <= 0.99


def _mertens_numerator_negative_series(n: int = 60, k: int = 29,
                                       spread: float = 0.0007) -> pd.Series:
    """A sample whose Mertens variance term ``1 − g₃·SR + ((g₄−1)/4)·SR²`` is
    NEGATIVE, built rather than searched for.

    In the population kurtosis ≥ skew² + 1, so on the two-point law the term is
    exactly ``(1 − g₃·SR/2)²`` and can only touch zero, never cross it. Put a
    sample exactly on ``SR = 2/g₃`` and the term collapses to
    ``(ĝ₄ − 1)/ĝ₃² − 1``, which the BIAS-CORRECTED sample estimators make
    slightly negative — they correct the two moments by different factors. That
    is the only way into this branch, which is why the module documents it as
    reachable "only for extreme skew/Sharpe combinations": a near-frozen series
    (here 60 days near +0.99% with a 7bp two-point jitter, i.e. a per-period
    Sharpe of 29).
    """
    pattern = np.zeros(n)
    pattern[:k] = spread
    g3 = sg.sample_skew(pattern)
    sd = float(np.std(pattern, ddof=1))
    level = 2.0 / g3 * sd - float(pattern.mean())     # ⇒ SR̂ == 2/ĝ₃ exactly
    return pd.Series(pattern + level,
                     index=pd.bdate_range("2015-01-02", periods=n))


def test_bootstrap_sharpe_ci_labels_the_iid_fallback_instead_of_clamping():
    """The non-positive-numerator branch of ``bootstrap_sharpe_ci`` had no test.

    The old code clamped the numerator to 1e-12, which made the studentizer
    ``sqrt(1e-12/(T−1)) ≈ 1.3e-7`` and manufactured a near-zero-width interval
    that necessarily reported ``excludes_zero=True`` — a refusal condition
    turned into a confident claim. The fix falls back to Lo's IID form and says
    so in ``extra["se_method"]``; both halves are pinned here, because
    restoring the clamp AND dropping the label passed the whole file before.
    """
    # 120 sessions rather than 60: at 60, 4.0% of the resamples of a two-point
    # law reproduce the sample's own multiset exactly, so the 2.5% pivot quantile
    # IS zero and the interval is refused for having one endpoint (pinned in
    # test_a_one_sided_interval_is_refused_by_its_own_algebraic_bound). At 120
    # that atom is 0.25% and the α/2 quantile is a real 2.1e-03.
    s = _mertens_numerator_negative_series(n=120, k=59)
    m = sg.ReturnMoments.from_returns(s)
    assert sg._psr_variance_numerator(m) < 0.0        # the branch is reachable

    ci = sg.bootstrap_sharpe_ci(s, seed=1, n_resamples=400)
    assert ci.extra["se_method"] == "lo_iid_fallback"
    lo_iid = math.sqrt((1.0 + m.sharpe_periodic ** 2 / 2.0) / (ci.n_obs - 1))
    assert ci.extra["se_periodic"] == pytest.approx(lo_iid, rel=1e-12)
    # The clamp's signature was a collapsed interval: clamping to 1e-12 scales
    # this studentizer — and with it the whole width — by a factor of
    # sqrt(1e-12/numer) ≈ 4e-7, which is what turned a refusal into
    # "excludes_zero=True". Lo's SE keeps a real interval.
    assert ci.ci_high - ci.ci_low > 1.0

    # THE SAME FALLBACK PER ROW, because the alternative is a biased drop. Rows
    # whose own Mertens numerator is non-positive used to get se = NaN and be
    # discarded: 33.8% of 400 resamples of the 60-session version of this series,
    # thrown away on a branch the scalar site handles by falling back. They are
    # studentized by Lo's form instead, and the count is surfaced.
    assert ci.extra["n_lo_iid_fallback_resamples"] > 0
    assert ci.extra["n_degenerate_resamples"] == ci.n_resamples - ci.n_usable
    sr_b, se_b, fell_back = sg._sharpe_and_se_matrix(
        s.to_numpy(dtype=float)[sg.stationary_bootstrap_indices(
            len(s), 400, sg.bootstrap_block_days(len(s), 1), seed=1)])
    assert fell_back.any() and np.isfinite(se_b[fell_back]).all()
    assert (se_b[fell_back] > 0).all()

    # ...and an ordinary series is NOT labelled as a fallback, so the label
    # carries information rather than being a constant.
    ok = sg.bootstrap_sharpe_ci(make_returns(300, seed=777, mu=0.0005),
                                seed=1, n_resamples=200)
    assert ok.extra["se_method"] == "mertens_non_normal"
    assert ok.extra["n_lo_iid_fallback_resamples"] == 0
    assert json.dumps(ci.to_dict(), allow_nan=False)


def test_bootstrap_ci_narrows_as_the_sample_grows():
    short = sg.bootstrap_sharpe_ci(make_returns(300, seed=61, mu=0.0005),
                                   seed=1, n_resamples=400)
    long_ = sg.bootstrap_sharpe_ci(make_returns(2500, seed=61, mu=0.0005),
                                   seed=1, n_resamples=400)
    assert (long_.ci_high - long_.ci_low) < (short.ci_high - short.ci_low)


def test_bootstrap_refuses_a_sample_too_short_for_inference():
    with pytest.raises(ValueError, match="30"):
        sg.bootstrap_sharpe_ci(make_returns(20, seed=71), seed=1, n_resamples=50)


def test_bootstrap_excess_return_ci_is_seeded_and_annualized():
    a = make_returns(1000, seed=81, mu=0.0006)
    b = make_returns(1000, seed=82, mu=0.0002)
    ci = sg.bootstrap_excess_return_ci(a - b, seed=99, n_resamples=400,
                                       holding_period_days=5)
    same = sg.bootstrap_excess_return_ci(a - b, seed=99, n_resamples=400,
                                         holding_period_days=5)
    assert (ci.ci_low, ci.ci_high) == (same.ci_low, same.ci_high)
    assert ci.point == pytest.approx(float((a - b).mean()) * PERIODS, rel=1e-12)
    assert ci.ci_low < ci.point < ci.ci_high
    # The lag is data-driven now, but it can never fall below the published
    # T-only rule floored by the holding period.
    assert ci.extra["lag"] >= sg.newey_west_lag(1000, 5)


# ------------------------------------------------- block-length policy

def test_block_length_scales_with_the_sample_and_never_approaches_it():
    """The old fixed 10-day block was simultaneously too long at T=30 (three
    blocks per resample) and too short at T=2,520.

    Pins are hand-derived from the documented rule
    ``min(max(hold, ceil(T^(1/3))), T/20)``: ceil(30^(1/3))=4 but the cap is
    30/20=1.5; ceil(252^(1/3))=7 against a cap of 12.6; ceil(2520^(1/3))=14
    against a cap of 126.
    """
    assert sg.bootstrap_block_days(30) == pytest.approx(1.5)
    assert sg.bootstrap_block_days(252) == pytest.approx(7.0)
    assert sg.bootstrap_block_days(2520) == pytest.approx(14.0)
    prev = 0.0
    for t in (30, 60, 126, 252, 504, 1260, 2520):
        blk = sg.bootstrap_block_days(t)
        assert blk >= prev                       # scales with T
        assert blk <= t / sg.MIN_EFFECTIVE_BLOCKS + 1e-12   # never approaches T
        prev = blk


def test_holding_period_actually_sizes_the_blocks():
    """The integration note always claimed holding_period_days 'floors the HAC
    lag AND sizes the bootstrap blocks'. It used to do only the first: the
    block stayed at 10.0 whether the caller declared a 1-day or a 60-day hold.
    """
    assert sg.bootstrap_block_days(1000, holding_period_days=1) == pytest.approx(10.0)
    assert sg.bootstrap_block_days(1000, holding_period_days=21) == pytest.approx(21.0)
    a = make_returns(1000, seed=501, mu=0.0006)
    b = make_returns(1000, seed=502, mu=0.0002)
    short_hold = sg.assess_excess(a, b, seed=3, n_resamples=100,
                                  holding_period_days=1)
    long_hold = sg.assess_excess(a, b, seed=3, n_resamples=100,
                                 holding_period_days=21)
    assert short_hold.bootstrap.expected_block_days == pytest.approx(10.0)
    assert long_hold.bootstrap.expected_block_days == pytest.approx(21.0)
    assert long_hold.hac.lag >= 21


def test_an_oversized_block_is_refused_instead_of_collapsing_the_interval():
    """Measured on the old code: T=30 with expected_block_days=1000 was
    ACCEPTED and returned CI = [0.639, 0.639] — a ZERO-WIDTH interval — with
    excludes_zero=True and the text 'the risk-adjusted edge survives
    resampling'. Every resample was a circular shift of the whole sample, so
    the studentized pivots collapsed onto zero.
    """
    with pytest.raises(ValueError, match="blocks"):
        sg.stationary_bootstrap_indices(30, 100, 1000.0, seed=1)
    with pytest.raises(ValueError, match="blocks"):
        sg.stationary_bootstrap_indices(252, 100, 20.0, seed=1)   # cap is 12.6
    r = make_returns(30, seed=503, mu=0.002)
    with pytest.raises(ValueError, match="blocks"):
        sg.bootstrap_sharpe_ci(r, seed=1, n_resamples=100,
                               expected_block_days=1000.0)


def test_the_sharpe_interval_no_longer_shrinks_monotonically_with_the_block():
    """Reviewer's measurement on T=252: the Sharpe CI width fell monotonically
    3.40 -> 1.40 as the block went 5 -> 504, i.e. a caller "being careful"
    about a long hold bought a MORE confident verdict. Inside the legal range
    the width must no longer march downward, and everything past the cap is
    now a refusal rather than a narrower interval.
    """
    r = make_returns(252, seed=504, mu=0.0008)
    widths = {}
    for blk in (2.0, 5.0, 12.0):                  # cap at T=252 is 12.6
        ci = sg.bootstrap_sharpe_ci(r, seed=11, n_resamples=400,
                                    expected_block_days=blk)
        widths[blk] = ci.ci_high - ci.ci_low
    assert min(widths.values()) > 0.5 * max(widths.values())
    for blk in (13.0, 63.0, 252.0):
        with pytest.raises(ValueError, match="blocks"):
            sg.bootstrap_sharpe_ci(r, seed=11, n_resamples=100,
                                   expected_block_days=blk)


def test_a_long_hold_on_a_short_sample_is_refused_not_squeezed():
    a = make_returns(200, seed=505, mu=0.0006)
    b = make_returns(200, seed=506, mu=0.0001)
    out = sg.assess_excess(a, b, seed=3, n_resamples=100,
                           holding_period_days=60)     # needs >= 1200 days
    assert out.available is False
    assert out.bootstrap is None
    assert "independent holds" in out.unavailable_reason
    assert sg.bootstrap_feasibility(200, 60) is not None
    assert sg.bootstrap_feasibility(1200, 60) is None


# ------------------------------------------------------- measured coverage

def _ar1_series(n: int, *, seed: int, rho: float, mu: float = 0.0,
                vol: float = DAILY_VOL) -> pd.Series:
    """AR(1) daily series with a 200-step burn-in, scaled to `vol` uncondition-
    ally so coverage is comparable across rho."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n + 200)
    y = np.zeros(n + 200)
    for i in range(1, n + 200):
        y[i] = rho * y[i - 1] + e[i]
    x = y[200:] * vol * math.sqrt(1.0 - rho ** 2) + mu
    return pd.Series(x, index=pd.bdate_range("2015-01-02", periods=n))


def _excess_coverage(t: int, *, rho: float = 0.0, hold: int = 1,
                     reps: int = 200, n_resamples: int = 200) -> float:
    """Fraction of seeded replications whose 95% interval contains the TRUE
    mean of zero. Because the true mean is zero, 1 − coverage is exactly the
    rate at which the three-state verdict calls a genuine tie a win or a loss.
    """
    covered = 0
    for i in range(reps):
        x = _ar1_series(t, seed=20_000 + i, rho=rho)
        ci = sg.bootstrap_excess_return_ci(x, seed=90_000 + i,
                                           holding_period_days=hold,
                                           n_resamples=n_resamples)
        covered += ci.ci_low <= 0.0 <= ci.ci_high
    return covered / reps


@pytest.mark.parametrize("t", [30, 126, 504])
def test_excess_return_ci_coverage_is_near_nominal_at_short_and_long_samples(t):
    """THE load-bearing test for the three-state verdict.

    The verdict is nothing but this interval's relationship to zero, so its
    coverage IS the product claim — 1 − coverage is the rate at which a true
    tie is called a win or a loss. The reviewer measured the code that shipped
    before (fixed 10-day blocks, T-only HAC lag) at 0.857 / 0.880 / 0.910 /
    0.943 for T = 30 / 126 / 252 / 504 against a nominal 0.95.

    On THESE seeds, old policy: 0.830 (T=30), 0.935, 0.910.
    On these seeds, T-scaled blocks + Andrews bandwidth: 0.945, 0.920, 0.915.

    The pin is the DESIGN value, not the implementation: seeded Monte-Carlo
    replications of a series whose true mean is zero, nominal rate 0.95. The
    0.90 floor is ~2 standard errors (√(.95·.05/200) ≈ 0.015) below nominal,
    and restoring the fixed 10-day block fails it at T=30 (0.830).
    """
    cov = _excess_coverage(t)
    assert 0.90 <= cov <= 0.99, f"T={t}: measured coverage {cov:.3f}"


def test_excess_return_ci_coverage_holds_up_under_autocorrelation():
    """The reviewer measured 0.890–0.900 coverage under AR(1) against a
    nominal 0.95. The Andrews (1991) data-driven bandwidth is what fixes it:
    on ρ=0.8 the T-only rule picks 5–6 lags and understates the long-run
    variance badly. On these seeds, reverting to the T-only lag drops the first
    cell to 0.873; with the data-driven bandwidth it is 0.907 (and 0.933 for
    the ρ=0.5 cell)."""
    assert 0.88 <= _excess_coverage(504, rho=0.8, hold=5, reps=150) <= 0.99
    assert 0.88 <= _excess_coverage(252, rho=0.5, hold=5, reps=150) <= 0.99


def test_a_strongly_autocorrelated_excess_series_is_flagged_in_the_prose():
    a = _ar1_series(504, seed=606, rho=0.8)
    z = pd.Series(0.0, index=a.index)
    out = sg.assess_excess(a, z, seed=3, n_resamples=200, holding_period_days=5)
    assert out.available is True
    assert "strongly autocorrelated" in out.interpretation


# -------------------------------------------------------- three-state verdict

def test_classify_excess_has_three_states():
    assert sg.classify_excess(0.05, 0.01, 0.09) == sg.BEATS
    assert sg.classify_excess(-0.05, -0.09, -0.01) == sg.LOSES
    assert sg.classify_excess(0.05, -0.02, 0.12) == sg.TIES_WITHIN_NOISE


def test_assess_excess_calls_a_coin_flip_a_tie():
    """Two indistinguishable noise series must not produce a winner just
    because one point estimate is larger."""
    a = make_returns(1000, seed=91, mu=0.0002)
    b = make_returns(1000, seed=92, mu=0.0002)
    out = sg.assess_excess(a, b, seed=3, n_resamples=300)
    assert out.verdict == sg.TIES_WITHIN_NOISE
    assert "not distinguishable" in out.interpretation


def test_assess_excess_detects_a_real_gap():
    a = make_returns(2500, seed=93, mu=0.0016)
    b = make_returns(2500, seed=94, mu=0.0001)
    out = sg.assess_excess(a, b, seed=3, n_resamples=300, label="SPY")
    assert out.verdict == sg.BEATS
    assert out.hac.t_stat > 2.0


def test_the_verdict_carries_its_own_coverage_shortfall_as_a_field():
    """The residual liberality of the interval used to live only in this
    module's docstrings, which no reader of the report ever sees.

    The three-state verdict IS the interval's relationship to zero, and that
    interval's measured coverage is 0.90–0.94 against a nominal 0.95. So the
    shortfall rides out on the result as a field — the same way a data problem
    surfaces as a finding instead of being absorbed — and the two verdict
    sources carry DIFFERENT disclosures, because only the bootstrap's shortfall
    is measured here.
    """
    a = make_returns(700, seed=111, mu=0.0007)
    b = make_returns(700, seed=112, mu=0.0002)
    out = sg.assess_excess(a, b, seed=9, n_resamples=200)
    assert out.available is True and out.verdict_source == "bootstrap"
    assert out.ci_coverage_caveat == sg.CI_COVERAGE_CAVEAT_BOOTSTRAP
    d = out.to_dict()
    assert d["ci_coverage_caveat"] == sg.CI_COVERAGE_CAVEAT_BOOTSTRAP
    for token in ("liberal", "0.90", "0.94", "0.95"):
        assert token in d["ci_coverage_caveat"]
    assert json.dumps(d, allow_nan=False)

    # The HAC fallback is a weaker claim and must not borrow the bootstrap's
    # measured numbers.
    hac_only = sg.assess_excess(a, b, seed=9, bootstrap=False)
    assert hac_only.verdict_source == "hac"
    assert hac_only.ci_coverage_caveat == sg.CI_COVERAGE_CAVEAT_HAC
    assert hac_only.ci_coverage_caveat != out.ci_coverage_caveat
    assert "unquantified" in hac_only.ci_coverage_caveat

    # A refusal has no verdict, so there is nothing to qualify.
    refused = sg.assess_excess(make_returns(20, seed=113, mu=0.002),
                              make_returns(20, seed=114), seed=9,
                              n_resamples=100)
    assert refused.available is False
    assert refused.ci_coverage_caveat is None
    assert refused.to_dict()["ci_coverage_caveat"] is None

    # The verdict itself is unchanged by the disclosure.
    assert out.verdict in (sg.BEATS, sg.LOSES, sg.TIES_WITHIN_NOISE)


def test_assess_excess_aligns_on_the_common_index():
    """Comparing different periods is the silent-wrong-answer failure mode."""
    a = make_returns(600, seed=95, mu=0.0005)
    b = make_returns(600, seed=96, mu=0.0005).iloc[100:]
    out = sg.assess_excess(a, b, seed=3, bootstrap=False)
    assert out.n_obs == 500
    assert out.verdict_source == "hac"


@pytest.mark.parametrize("t", [5, 10, 20, 29])
def test_assess_excess_refuses_a_verdict_below_the_minimum_sample(t):
    """PSR and DSR self-suppress below 30 observations; the verdict did not.

    It fell back to a HAC interval that needs only 3 observations and uses
    normal critical values with no small-sample correction. Measured false
    non-tie rate under a TRUE TIE, 500 reps: T=5 -> 0.268, T=10 -> 0.162,
    T=20 -> 0.100, T=29 -> 0.094 against a nominal 0.05. A 5-day coin flip was
    called a win 27% of the time. Now there is simply no verdict.
    """
    a = make_returns(t, seed=700 + t, mu=0.002)
    b = make_returns(t, seed=800 + t, mu=0.0)
    out = sg.assess_excess(a, b, seed=3, n_resamples=100)
    assert out.available is False
    assert out.verdict_source == "none"
    assert out.verdict == sg.TIES_WITHIN_NOISE
    assert str(sg.MIN_OBS_FOR_INFERENCE) in out.unavailable_reason
    assert "refusal" in out.interpretation
    assert json.dumps(out.to_dict(), allow_nan=False)


def test_the_minimum_sample_gate_is_the_same_one_psr_uses():
    """One threshold, not two: a verdict that requires less evidence than the
    probability printed beside it is the inconsistency this gate closes."""
    assert sg.bootstrap_feasibility(sg.MIN_OBS_FOR_INFERENCE - 1, 1) is not None
    assert sg.bootstrap_feasibility(sg.MIN_OBS_FOR_INFERENCE, 1) is None


# ------------------------------------------------------------- bundled entry

def test_assess_sharpe_bundles_every_statistic_and_serializes_to_json():
    r = make_returns(1260, seed=101, mu=0.0006)
    out = sg.assess_sharpe(r, seed=17, n_trials=12,
                           trial_sharpes_annual=[0.2, 0.9, -0.3, 1.4, 0.5, 0.1],
                           n_resamples=300)
    d = out.to_dict()
    for key in ("moments", "standard_error", "psr", "dsr", "min_track_record",
                "min_backtest_length", "bootstrap", "interpretation"):
        assert key in d
    assert d["moments"]["sharpe_annual"] == pytest.approx(metrics.sharpe(r), rel=1e-12)
    assert d["dsr"]["dsr"] is not None
    # Must survive strict JSON: no NaN/Infinity leaking into the report.
    text = json.dumps(d, allow_nan=False)
    assert "NaN" not in text and "Infinity" not in text


def test_assess_sharpe_interpretations_are_deterministic_non_empty_prose():
    r = make_returns(800, seed=102, mu=0.0004)
    a = sg.assess_sharpe(r, seed=5, n_resamples=200)
    b = sg.assess_sharpe(r, seed=5, n_resamples=200)
    assert a.interpretation == b.interpretation
    assert len(a.interpretation) > 100
    for piece in (a.standard_error.interpretation, a.psr.interpretation,
                  a.dsr.interpretation, a.min_trl.interpretation,
                  a.min_btl.interpretation):
        assert isinstance(piece, str) and piece.strip()


def test_assess_sharpe_can_skip_the_bootstrap_for_short_samples():
    r = make_returns(40, seed=103, mu=0.001)
    out = sg.assess_sharpe(r, seed=5, bootstrap=False)
    assert out.bootstrap is None
    assert out.psr.psr is not None
    assert json.dumps(out.to_dict(), allow_nan=False)


def test_assess_excess_serializes_to_json():
    a = make_returns(700, seed=104, mu=0.0007)
    b = make_returns(700, seed=105, mu=0.0002)
    d = sg.assess_excess(a, b, seed=9, n_resamples=200).to_dict()
    assert d["verdict"] in (sg.BEATS, sg.LOSES, sg.TIES_WITHIN_NOISE)
    assert json.dumps(d, allow_nan=False)


# ============================================================================
# THE ENUMERATING TESTS — B AND C
#
# Six rounds of this module were lost to tests that PINNED the fields and the
# sentences whoever wrote them happened to think of. Round 8 pinned the
# t-statistic and the confidence interval three lines below it went on being
# published; the round before pinned the verdict and the decay finding went on
# being published. A pinning test cannot fail on a field that did not exist when
# it was written, so it certifies the author's enumeration rather than the
# module's behaviour.
#
# The two tests below do not enumerate. They WALK — the payload structure and
# the emitted prose, programmatically, on degenerate input — and require
# everything outside an explicit allow-list to be absent. A field added tomorrow
# and a sentence added tomorrow both fail them with no edit to this file. That
# property is the deliverable; the specific open items this round closed are
# pinned separately above, and are the cases these two would have caught.
# ============================================================================

#: THE TEST'S OWN ALLOW-LIST, spelled here on purpose instead of imported from
#: ``sg.HONEST_UNDER_REFUSAL``.
#:
#: If it were imported, widening the module's allow-list would widen this test
#: with it, and a newly-exempted field would pass silently — the test would move
#: with whatever the module happens to do, which is exactly the failure this
#: file's header warns about for the annualization constant. Two independent
#: lists means exempting a field is a two-file act with a reviewer in between.
#:
#: MEMBERSHIP IS A RULE, not a taste: the quantity must be honest arithmetic ON
#: THE SAMPLE AS GIVEN, computable with no sampling distribution at all.
#:   * a count, or a calendar fact about the window;
#:   * a configuration value the caller passed in and is being shown back;
#:   * the POINT ESTIMATE — what the error bars were going to be about, and true
#:     of the sample whether or not the sample can carry error bars;
#:   * the multiple-testing budget, which is a function of the trial LEDGER and
#:     not of the series being refused;
#:   * the refusal metadata itself, because nulling that would hide the refusal,
#:     which is the one thing worse than publishing the number.
#: Anything else is a claim about a distribution a degenerate series does not
#: have, and must be None or NaN.
HONEST_UNDER_DEGENERACY = frozenset({
    # counts and calendar facts
    "n_obs", "observed_n_obs", "observed_years", "n_resamples", "n_usable",
    "n_degenerate_resamples", "n_lo_iid_fallback_resamples", "n_trials",
    # caller configuration, echoed back
    "confidence", "alpha", "seed", "lag", "expected_block_days",
    "holding_period_days", "sr_star_periodic", "sr_star_annual",
    "target_sharpe_annual",
    # the point estimate, in every unit and at every site that carries one
    "point", "point_daily", "point_periodic", "mean_daily", "mean_annualized",
    "sharpe_periodic", "sharpe_annual", "difference_annual",
    "sharpe_annual_a", "sharpe_annual_b", "skew", "kurtosis_raw",
    # the search budget: a function of the trial ledger, not of this series
    "sr0_periodic", "sr0_annual", "variance_trial_sharpes_periodic",
    "min_years_required", "expected_lucky_sharpe_annual",
    # refusal metadata
    "available", "valid", "sufficient_sample", "attainable", "sufficient",
})

#: A formatted DERIVED statistic, as this module formats one — the test's own
#: copy, for the same reason the allow-list is. Not "any number": a refusal
#: sentence is REQUIRED to report the sample's own arithmetic (793 sessions, a
#: -5.04% annualized mean, a 5.1e-21 dispersion, the algebraically possible
#: [0, 1]) and must be able to do so without tripping this.
LEAKED_STATISTIC = re.compile(
    r"t-?statistic\s*(of|=|:)?\s*[-+(]*\d"
    r"|\|\s*t\s*\|"
    r"|\bt\s*=\s*[-+(]*\d"
    r"|\bz\s*=\s*[-+(]*\d"
    r"|p[-\s]?value\s*(of|=|:)?\s*[-+(]*\d"
    r"|\bp\s*[=<>]\s*[-+(]*\d"
    r"|R²\s*(of|=|:)?\s*[-+(]*\d"
    r"|±\s*[-+(]*\d"
    r"|\[[^\]]*\d[.%][^\]]*,[^\]]*\]"
    r"|interval\s+(excludes|contains|lies)"
    r"|critical\s+value"
    r"|(?:PSR|DSR)\s*(?:of|is|=|:)\s*[-+(]*\d"
    r"|probabilit(?:y|ies)\s+(?:of|is)\s*[-+(]*\d"
    r"|\b(?<!dispersion\s)(?:ratio|share)\s*(?:of|is|=|:)?\s*[-+(]*\d",
    re.IGNORECASE)


def _leaves(obj, path="$"):
    """Every leaf of a nested payload, as ``(path, key, value)``.

    Structural, not name-driven: dicts, lists, tuples and dataclasses are all
    descended into, so a new field, a new nested payload or a new ``extra``
    entry is walked without this function being told about it.
    """
    from dataclasses import fields, is_dataclass
    if is_dataclass(obj) and not isinstance(obj, type):
        obj = {f.name: getattr(obj, f.name) for f in fields(obj)}
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _leaves(v, f"{path}.{k}")
        return
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _leaves(v, f"{path}[{i}]")
        return
    yield path, path.rsplit(".", 1)[-1].split("[")[0], obj


def _is_null(v) -> bool:
    if v is None:
        return True
    return isinstance(v, float) and not math.isfinite(v)


def _degenerate_payloads():
    """Everything this module can hand a caller about an UNTESTABLE fit.

    Both forms of every one: the serialized ``to_dict`` (what the report and the
    API see) and the live object (what every caller in this repo reads). The
    object level is not optional — round 8's nulling lived in ``to_dict`` at one
    site and the dataclass field beside it still carried the number.
    """
    idx = _fixture_sessions()
    flat_rf = pd.Series(0.0002, index=idx)
    all_cash = pd.Series(0.0, index=idx)
    # A nonzero CONSTANT difference: sd 5e-20, dispersion share ~1e-31.
    constant = metrics._excess(all_cash, flat_rf).rename("excess")
    # A non-finite difference: dispersion unmeasurable rather than tiny.
    inf_vals = np.zeros(len(idx))
    inf_vals[7] = np.inf
    non_finite = pd.Series(inf_vals, index=idx).rename("excess")
    short = make_returns(20, seed=7301, mu=0.0004)
    loser = make_returns(400, seed=7302, mu=-0.0008)
    out = {}

    out["newey_west/constant"] = sg.newey_west_tstat(constant, label="cash")
    out["newey_west/non_finite"] = sg.newey_west_tstat(non_finite, label="cash")
    out["assess_excess/constant"] = sg.assess_excess(
        all_cash, flat_rf, seed=1, n_resamples=200, label="cash")
    out["assess_excess/non_finite"] = sg.assess_excess(
        non_finite, pd.Series(0.0, index=idx), seed=1, n_resamples=200,
        label="cash")
    out["assess_sharpe/constant"] = sg.assess_sharpe(
        all_cash, seed=1, rf_daily=flat_rf, n_resamples=200)
    out["psr/too_short"] = sg.probabilistic_sharpe_ratio(
        sg.ReturnMoments.from_returns(short), 0.0)
    out["psr/non_positive_variance"] = sg.probabilistic_sharpe_ratio(
        sg.ReturnMoments.from_returns(
            _mertens_numerator_negative_series(n=120, k=59)), 0.0)
    out["dsr/no_ledger"] = sg.deflated_sharpe_ratio(
        sg.ReturnMoments.from_returns(loser), n_trials=12)
    out["min_trl/below_threshold"] = sg.min_track_record_length(
        sg.ReturnMoments.from_returns(loser), 0.0, 0.05)
    # The zero-standard-error Sharpe comparison: a series against itself has a
    # zero difference AND a zero combined SE, so z does not exist.
    m = sg.ReturnMoments.from_returns(loser)
    zero_se = sg.sharpe_difference_test(m, m)
    assert zero_se["available"] is True, (
        "a series against itself still has a positive combined SE; the "
        "refusal branch below is reached by the substitution instead")
    flat = sg.ReturnMoments(sharpe_periodic=0.0, n_obs=10 ** 9, skew=0.0,
                            kurtosis_raw=3.0)
    out["sharpe_difference/no_se"] = _zero_se_difference(flat)
    return out


def _zero_se_difference(moments):
    """``sharpe_difference_test`` on a pair whose combined SE is exactly zero.

    Substituted rather than constructed: the Lo/Mertens SE is
    ``sqrt(numer/T)`` with ``numer >= 1`` for a zero Sharpe, so no real sample
    reaches a zero combined SE. The branch is still the only thing standing
    between ``z = nan`` and "The two are not distinguishable" — a fabricated tie
    — so it is pinned by substitution, the same way the unmeasurable-HAC branch
    above is.
    """
    real = sg.sharpe_standard_error

    def zero(m, **kw):
        out = real(m, **kw)
        return sg.SharpeSE(**{**asdict(out), "se_annual": 0.0,
                              "se_periodic": 0.0})
    sg.sharpe_standard_error = zero
    try:
        return sg.sharpe_difference_test(moments, moments)
    finally:
        sg.sharpe_standard_error = real


def test_every_numeric_leaf_outside_the_allow_list_is_null_on_a_degenerate_fit():
    """TEST B — THE ENUMERATING FIELD WALK.

    For every payload this module can hand a caller about an untestable fit, at
    BOTH the object level and the serialized level, every numeric leaf whose name
    is not in :data:`HONEST_UNDER_DEGENERACY` must be None or NaN. No field is
    named in the assertion; the walk finds them.

    WHAT THIS CATCHES THAT SIX ROUNDS OF PINNING DID NOT. A field added to any of
    these result types tomorrow — an appraisal ratio, a second p-value, an
    information coefficient — arrives in this walk automatically. If it went
    through the chokepoint it is already null and this passes. If its author
    bypassed the chokepoint, or exempted it in ``sg.HONEST_UNDER_REFUSAL``
    without exempting it here, this fails and names the path.
    """
    payloads = _degenerate_payloads()
    assert len(payloads) >= 9
    walked = 0
    for name, payload in payloads.items():
        forms = {"object": payload}
        if hasattr(payload, "to_dict"):
            forms["serialized"] = payload.to_dict()
        for level, form in forms.items():
            leaves = list(_leaves(form))
            assert leaves, f"{name}/{level} has no leaves to walk"
            numeric = [(p, k, v) for p, k, v in leaves
                       if isinstance(v, (bool, int, float, np.bool_,
                                         np.integer, np.floating))
                       or v is None]
            assert numeric, f"{name}/{level} has no numeric leaves"
            for path, key, value in numeric:
                if key in HONEST_UNDER_DEGENERACY:
                    continue
                assert _is_null(value), (
                    f"{name}/{level}: {path} = {value!r} is a derived quantity "
                    f"published on an untestable fit. Either route it through "
                    f"significance.gate_derived, or — if it is honest "
                    f"arithmetic on the sample as given — add it to BOTH "
                    f"significance.HONEST_UNDER_REFUSAL and this test's "
                    f"HONEST_UNDER_DEGENERACY, with the reason."
                )
                walked += 1
        # ...and it is JSON, which is the other thing a NaN breaks.
        if "serialized" in forms:
            assert json.dumps(forms["serialized"], allow_nan=False)
    # The walk has to actually reach nulled fields, or it is asserting nothing.
    assert walked >= 40, walked


def test_the_allow_list_is_the_modules_and_the_disagreement_is_the_point():
    """The two allow-lists are INDEPENDENT copies that must currently agree.

    Kept as its own assertion rather than by importing one into the other: if
    they disagree, one of two things happened and both want a human. Either the
    module exempted a field the test does not (a fabrication about to be
    published), or the test exempts one the module nulls (a needless refusal).
    Test B fails on the first case on its own; this one names the drift directly
    and in both directions.
    """
    only_module = sg.HONEST_UNDER_REFUSAL - HONEST_UNDER_DEGENERACY
    only_test = HONEST_UNDER_DEGENERACY - sg.HONEST_UNDER_REFUSAL
    assert not only_module, (
        f"the module exempts {sorted(only_module)} from the degeneracy gate and "
        f"this test does not: justify each in both places or gate them")
    assert not only_test, (
        f"this test exempts {sorted(only_test)} and the module nulls them: the "
        f"module is refusing honest arithmetic")
    # And the allow-list is a SHORT list of arithmetic, not a second payload
    # schema: if it ever holds most of the fields, the gate has stopped gating.
    all_keys = {k for p in _degenerate_payloads().values()
                for _, k, _ in _leaves(p)}
    assert len(sg.HONEST_UNDER_REFUSAL & all_keys) < len(all_keys) / 2


def _every_sentence(payload) -> list[tuple[str, str]]:
    """Every string leaf of a payload, by path. Name-blind on purpose.

    Not "every field named in a list of prose fields". That was the module's
    original shape and it was demonstrably wrong: a leaky sentence put in a NEW
    field — ``method_note``, say — is in no list written before that field
    existed, so the gate never saw it and a formatted t-statistic reached the
    payload. This walk caught that, the module was made name-blind in response
    (:data:`significance.FORMERLY_PROSE_FIELDS` records why), and the walk stays
    name-blind so the next such field is caught whichever gate sees it first.
    """
    return [(p, v) for p, _, v in _leaves(payload)
            if isinstance(v, str) and len(v) > 20]


def test_no_sentence_on_a_degenerate_fit_formats_a_derived_statistic():
    """TEST C — THE ENUMERATING PROSE WALK.

    Every sentence this module can emit about an untestable fit is walked and
    regexed. No list of known sentences: a leaky sentence written tomorrow, in a
    field with a name nobody has thought of yet, fails this without an edit.

    The pattern is deliberately narrow about WHAT is forbidden and broad about
    WHERE it looks. A refusal is REQUIRED to report the sample's own arithmetic —
    "differs from cash by the same -5.04% annualized on every one of its 793
    sessions", "dispersion share 1.6e-31, at or below the 2.2e-16 floor",
    "outside the algebraically possible [0, 1]" — and all of that passes. What
    may not appear is a formatted t, p, z, R-squared, interval bound, ratio or
    probability, because those exist only if the fit is testable.
    """
    payloads = _degenerate_payloads()
    checked = 0
    for name, payload in payloads.items():
        forms = {"object": payload}
        if hasattr(payload, "to_dict"):
            forms["serialized"] = payload.to_dict()
        for level, form in forms.items():
            sentences = _every_sentence(form)
            assert sentences, f"{name}/{level} emits no prose at all"
            for path, text in sentences:
                hit = LEAKED_STATISTIC.search(text)
                assert hit is None, (
                    f"{name}/{level}: {path} formats {hit.group(0)!r} on an "
                    f"untestable fit — {text!r}. A refusal may report the "
                    f"sample's own arithmetic and nothing else."
                )
                # ...and the two states a refusal is NOT allowed to imply.
                low = text.lower()
                assert "not distinguishable from noise" not in low, path
                assert "clears" not in low, path
                assert "nan" not in low, path
                checked += 1
    assert checked >= 20, checked
    # THE SIX REFUSAL REASONS, over every kind the predicate can report plus the
    # one newey_west_tstat constructs, at both signs of the mean — the sentence
    # generator itself, not only the sentences that happened to be reachable
    # through the payloads above.
    kinds = ("too_short", "non_finite", "impossible", "constant", "absolute",
             "unmeasurable")
    for kind in kinds:
        for mean in (0.0, -2e-4, +2e-4):
            deg = sg.Degeneracy(kind, 1.6e-31, 5.1e-21, 793)
            text = sg.degeneracy_reason(deg, mean, "cash")
            hit = LEAKED_STATISTIC.search(text)
            assert hit is None, (kind, mean, hit and hit.group(0), text)
            assert sg.gate_prose(text, available=False) == text


def test_the_prose_gate_and_the_field_gate_are_the_same_refusal():
    """The two halves of the chokepoint, shown to actually fire.

    A gate no input reaches is how the ``se > 0`` shape survived six rounds, so
    both halves are driven directly here as well as through the payloads.
    """
    # THE FIELD HALF: default-deny, so an unknown key is nulled without being
    # named anywhere. This is the property that gates a field added tomorrow.
    gated = sg.gate_derived(
        {"t_stat": -1.96, "appraisal_ratio": 0.42, "n_obs": 793,
         "mean_daily": -2e-4, "nested": {"information_coefficient": 0.1},
         "extra": {"some_future_share": 0.5}},
        available=False)
    assert gated["t_stat"] is None
    assert gated["appraisal_ratio"] is None          # never named, still gated
    assert gated["nested"]["information_coefficient"] is None
    assert gated["extra"]["some_future_share"] is None
    assert gated["n_obs"] == 793 and gated["mean_daily"] == -2e-4
    # ...and it is a pass-through when the fit IS testable, so the gate cannot
    # quietly become a second, stricter degeneracy policy.
    same = {"t_stat": -1.96, "appraisal_ratio": 0.42}
    assert sg.gate_derived(same, available=True) == same
    # ...and a nested payload that gated itself is not re-gated, because an
    # outer refusal must not null an inner result that is testable on its own.
    inner = {"available": True, "t_stat": -1.96}
    assert sg.gate_derived({"hac": inner}, available=False)["hac"] == inner

    # THE PROSE HALF: a leaky refusal sentence is raised, not printed.
    for leak in ("a t-statistic of -1.96",
                 "That clears the conventional |t| > 2 bar.",
                 "the 95% interval [-6.67%, -0.00%]",
                 "an appraisal ratio of 0.42",
                 "Sharpe 1.20 ± 0.31",
                 "against the two-sided 95% critical value of 1.96",
                 "the interval excludes zero"):
        with pytest.raises(ValueError, match="formats a derived statistic"):
            sg.gate_prose(leak, available=False)
        assert sg.gate_prose(leak, available=True) == leak
    # ...and the sentences a refusal MUST be able to say are not refused.
    for honest in (
        "the strategy differs from cash by the same -5.04% annualized on every "
        "one of its 793 sessions",
        "reports a dispersion share of 1.4, outside the algebraically possible "
        "[0, 1] — the arithmetic behind it is broken",
        "dispersion share 1.6e-31, at or below the 2.2e-16 floor",
        "a daily standard deviation of 5.09e-21, at or below the 1.0e-07 floor",
        "only 12 overlapping observations — fewer than the 30 below which a "
        "resampled confidence interval has no honest coverage",
    ):
        assert sg.gate_prose(honest, available=False) == honest


# ============================================================================
# THE REMAINING OPEN ITEMS, each subsumed by the chokepoint rather than patched
# ============================================================================

def test_the_sign_blind_sentence_is_subsumed_at_every_site_not_fixed_at_one():
    """OPEN ITEM 2. The sign-blind sentence was fixed at ONE site and left at the
    two others that render the same claim.

    Before: ``bootstrap_sharpe_ci`` had three branches, while
    ``sharpe_standard_error`` said "The interval excludes zero, so there is a
    measurable edge" about an interval lying entirely BELOW zero, and
    ``bootstrap_excess_return_ci`` said "The interval excludes zero." with no
    direction at all — one bit rendered where two are needed.

    Pinned BEHAVIOURALLY, at every site, in all three states — and structurally,
    by requiring each site to reach the one sentence function rather than to have
    grown its own third branch.
    """
    import ast
    tree = ast.parse(Path(sg.__file__).read_text())
    bodies = {}
    for n in ast.walk(tree):
        if not isinstance(n, ast.FunctionDef):
            continue
        stmts = n.body
        if (stmts and isinstance(stmts[0], ast.Expr)
                and isinstance(stmts[0].value, ast.Constant)
                and isinstance(stmts[0].value.value, str)):
            stmts = stmts[1:]        # drop the docstring: prose is not code
        bodies[n.name] = "\n".join(ast.unparse(x) for x in stmts)
    for site in ("sharpe_standard_error", "bootstrap_sharpe_ci",
                 "bootstrap_excess_return_ci", "_bar_sentence"):
        assert "zero_relation_sentence(" in bodies[site], site

    idx = pd.bdate_range("2015-01-02", periods=793)
    rng = np.random.default_rng(7401)
    series = {name: pd.Series(rng.normal(mu, 0.004, 793), index=idx)
              for name, mu in (("loser", -0.0012), ("winner", 0.0012),
                               ("tie", 0.0))}
    seen = {}
    for name, r in series.items():
        m = sg.ReturnMoments.from_returns(r)
        bench = pd.Series(0.0, index=idx)
        seen[name] = {
            "parametric": sg.sharpe_standard_error(m).interpretation,
            "boot_sharpe": sg.bootstrap_sharpe_ci(
                r, seed=5, n_resamples=400).interpretation,
            "boot_excess": sg.bootstrap_excess_return_ci(
                (r - bench).rename("excess"), seed=5, n_resamples=400,
                label="cash").interpretation,
            "hac": sg.newey_west_tstat(r, label="cash").interpretation,
        }
    for site in ("parametric", "boot_sharpe", "boot_excess", "hac"):
        lose, win, tie = (seen[k][site] for k in ("loser", "winner", "tie"))
        # THREE distinct claims, not one bit rendered twice.
        claims = {t.split(". ")[-1] for t in (lose, win, tie)}
        assert len(claims) == 3, (site, claims)
        # ...and the losing one never borrows the winning one's words.
        assert "lies entirely below zero" in lose, site
        assert "lies entirely above zero" in win, site
        assert "contains zero" in tie, site
        for word in ("edge survives", "in the strategy's favour",
                     "margin over cash survives"):
            assert word not in lose, (site, word)
    # And the phrase that carried the defect is gone from the module's CODE:
    # "excludes zero" cannot be rendered without a direction, because no branch
    # says it any more. (The module's prose still recounts it as history, which
    # is what a comment is for; this checks the executable part.)
    code = "\n\n".join(bodies.values())
    assert "excludes zero" not in code


def test_sharpe_difference_test_is_two_sided_in_the_region_and_in_the_sentence():
    """OPEN ITEM 3. ``significant = z >= norm_ppf(0.95)`` is a ONE-sided
    rejection region, and the sentence beside it — "larger than sampling noise
    comfortably explains", against "not distinguishable … in either direction" —
    is a TWO-sided claim. That advertises a 5% error rate as 2.5%, and it is
    directional on top: b beating a by four standard errors came back "not
    distinguishable".

    The region is now the interval at :func:`critical_z`, read by
    :func:`classify_excess`, so region, interval and sentence are one object.
    """
    hi = sg.ReturnMoments.from_returns(make_returns(2000, seed=7402, mu=0.0016))
    lo = sg.ReturnMoments.from_returns(make_returns(2000, seed=7403, mu=0.0000))

    forward = sg.sharpe_difference_test(hi, lo)
    reverse = sg.sharpe_difference_test(lo, hi)

    # 1. SYMMETRY. A two-sided test cannot see the order of its arguments.
    assert forward["significant_at_95"] is reverse["significant_at_95"]
    assert forward["z_stat"] == pytest.approx(-reverse["z_stat"], rel=1e-12)
    assert forward["p_value_two_sided"] == pytest.approx(
        reverse["p_value_two_sided"], rel=1e-12)
    # ...which the OLD one-sided region did not: it fired one way only.
    assert (forward["z_stat"] >= sg.norm_ppf(0.95)) is not (
        reverse["z_stat"] >= sg.norm_ppf(0.95))

    # 2. THE REGION IS THE INTERVAL. Not a second comparison beside it.
    for d in (forward, reverse):
        assert d["significant_at_95"] is (
            sg.classify_excess(d["difference_annual"], d["ci_low_annual"],
                               d["ci_high_annual"]) != sg.TIES_WITHIN_NOISE)
        assert d["verdict"] in (sg.BEATS, sg.LOSES, sg.TIES_WITHIN_NOISE)
        crit = sg.critical_z(d["confidence"])
        assert f"{crit:.2f}" in d["interpretation"]
        assert (abs(d["z_stat"]) >= crit) is d["significant_at_95"]

    # 3. THE DIRECTION IS IN THE SENTENCE, both ways round.
    assert forward["verdict"] == sg.BEATS
    assert "a is better than b" in forward["interpretation"]
    assert reverse["verdict"] == sg.LOSES
    assert "b is better than a" in reverse["interpretation"]

    # 4. BOTH p-values are reported and LABELLED, so neither is read as the other.
    assert forward["p_value_two_sided"] == pytest.approx(
        2.0 * forward["p_value_one_sided"], rel=1e-9)

    # 5. A tie is still a tie, in both directions.
    a = sg.ReturnMoments.from_returns(make_returns(600, seed=7404, mu=0.0002))
    b = sg.ReturnMoments.from_returns(make_returns(600, seed=7405, mu=0.0003))
    tie = sg.sharpe_difference_test(a, b)
    assert tie["significant_at_95"] is False
    assert "not distinguishable" in tie["interpretation"]


def test_a_one_sided_pivot_distribution_is_disclosed_rather_than_published_bare():
    """OPEN ITEM 4. A pivot distribution that never straddles zero was still
    publishing an ``excludes_zero`` interval with nothing said about it.

    The module's own measurements say this must NOT become a refusal — the strong
    form ``ci_low < point < ci_high`` refuses the committed
    non-positive-Mertens-numerator construction, whose pivots are all genuinely
    positive — so it is SURFACED instead: a field in ``extra`` and a sentence
    beside the verdict. Surfaced, not absorbed, and not over-corrected.
    """
    one_sided = sg.bootstrap_sharpe_ci(
        _mertens_numerator_negative_series(n=120, k=59), seed=1, n_resamples=400)
    assert one_sided.extra["pivots_straddle_zero"] is False
    assert one_sided.ci_high < one_sided.point      # entirely below its own point
    assert sg.ONE_SIDED_PIVOTS_CAVEAT in one_sided.interpretation
    # ...and the relationship to zero is still reported rather than withheld
    assert isinstance(one_sided.excludes_zero, bool)

    # THE OVER-CORRECTION CHECK: an ordinary series straddles, and gets no
    # caveat, so the disclosure keeps its meaning.
    ordinary = sg.bootstrap_sharpe_ci(
        make_returns(600, seed=7406, mu=0.0006), seed=1, n_resamples=400)
    assert ordinary.extra["pivots_straddle_zero"] is True
    assert sg.ONE_SIDED_PIVOTS_CAVEAT not in ordinary.interpretation
    assert ordinary.ci_low < ordinary.point < ordinary.ci_high
    # and the same field exists on the excess path, which shares the helper
    ex = sg.bootstrap_excess_return_ci(
        make_returns(600, seed=7407, mu=0.0004).rename("excess"), seed=1,
        n_resamples=400)
    assert ex.extra["pivots_straddle_zero"] is True


def test_the_two_documentation_drifts_are_recomputed_not_quoted():
    """OPEN ITEM 6. Two comments carried numbers their own measurements deny.

    Both are pure arithmetic on figures already in the comment, which is why they
    survived six rounds — nothing recomputed them. They are recomputed here, so
    the comment cannot drift again without this failing.
    """
    src = Path(sg.__file__).read_text()

    # DRIFT 1 — ABSOLUTE_SD_FLOOR. The comment's summary said the margin to the
    # flattest object the fixtures can produce is "11.3x"; the measurement three
    # lines above it is 1.125e-06 against a floor of 1e-7, i.e. 11.25x, which
    # rounds DOWN to 11.2x.
    margin = 1.125e-06 / sg.ABSOLUTE_SD_FLOOR
    assert margin == pytest.approx(11.25, rel=1e-12)
    assert f"{margin:.1f}x margin" in src
    assert "11.3x margin" not in src

    # DRIFT 2 — MAGNITUDE_DUST_RATIO. The comment said √eps "sits 7.4 orders
    # above the artefact and 7.4 orders below the tightest real interval". The
    # second is right and the first is the same number printed twice: the artefact
    # margin is 7.6 orders, not 7.4.
    to_artefact = math.log10(sg.MAGNITUDE_DUST_RATIO / 3.7e-16)
    to_real = math.log10(0.342 / sg.MAGNITUDE_DUST_RATIO)
    assert to_artefact == pytest.approx(7.6, abs=0.05)
    assert to_real == pytest.approx(7.4, abs=0.05)
    assert to_artefact != pytest.approx(to_real, abs=0.05), (
        "the two margins are different numbers; that is the drift")
    assert f"{to_artefact:.1f} orders above the artefact" in src
    assert f"{to_real:.1f} orders below the tightest real interval" in src
