"""Factor attribution: known-answer regressions, the refusal paths, and the
integrity of the committed Ken French fixture.

The load-bearing test is `test_known_answer_*`: a series constructed as exactly
1.5 × market + 2%/yr must come back as beta 1.5 and alpha 2.0%/yr. Everything
else in this module is downstream of that arithmetic being right — if the join,
the excess-return conversion or the annualization is off by anything, that test
fails first and loudly.
"""
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from audit import attribution as at
from audit.attribution import FactorData
from dsl.schema import (
    Costs, Selection, Signal, StrategySpec, Universe, Window,
)
from engine.backtest import run

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "factors_ff5_daily.csv"


# ------------------------------------------------------------- fixtures

@pytest.fixture(scope="module")
def factors() -> FactorData:
    return at.load_factors()


@pytest.fixture(scope="module")
def window(factors) -> pd.DataFrame:
    """A clean multi-year slice well inside the fixture's coverage."""
    return factors.frame.loc["2016-01-01":"2024-12-31"]


def withheld(x) -> bool:
    """Did the chokepoint withhold this value?

    `attribution._withhold_unsupported` leaves None, uniformly. NaN is accepted
    too because a few quantities are NaN for arithmetic reasons of their own
    (`t_two_sided_p` of a NaN t, a lag-1 autocorrelation of a constant residual)
    and a test that distinguished the two would be pinning the reason a number is
    missing rather than that it is missing. `np.isfinite(None)` raises, which is
    why this exists at all.
    """
    return x is None or (isinstance(x, float) and math.isnan(x))


def synthetic_copy(tmp_path: Path, frame: pd.DataFrame) -> Path:
    """Same numbers, a header that declares them synthetic."""
    path = tmp_path / "factors_synth.csv"
    with path.open("w") as fh:
        fh.write("# Placeholder factors for testing.\n")
        fh.write("# SYNTHETIC: yes\n")
        frame.to_csv(fh, float_format="%.6f")
    return path


# --------------------------------------------- the hand-constructed case

def test_known_answer_capm_recovers_beta_and_alpha(window):
    """r_t = RF_t + 1.5·MKT_t + 0.02/252 must give beta 1.5, alpha +2.0%/yr."""
    strat = window["rf"] + 1.5 * window["mkt_rf"] + 0.02 / 252.0
    payload, _ = at.attribution_report(strat, attempts=1)

    capm = payload["models"]["capm"]
    assert capm["betas"]["mkt_rf"]["beta"] == pytest.approx(1.5, abs=1e-9)
    assert capm["alpha_annual"] == pytest.approx(0.02, abs=1e-9)
    assert capm["alpha_per_period"] == pytest.approx(0.02 / 252.0, abs=1e-12)
    assert capm["r2"] == pytest.approx(1.0, abs=1e-9)
    assert capm["annualization"] == 252


def test_known_answer_survives_the_whole_ladder(window):
    """Adding orthogonal-ish factors must not move a true alpha: every model in
    the ladder recovers 2%/yr and a zero loading on every non-market factor."""
    strat = window["rf"] + 1.5 * window["mkt_rf"] + 0.02 / 252.0
    payload, _ = at.attribution_report(strat, attempts=1)

    for key in payload["ladder"]:
        model = payload["models"][key]
        assert model["alpha_annual"] == pytest.approx(0.02, abs=1e-8), key
        assert model["betas"]["mkt_rf"]["beta"] == pytest.approx(1.5, abs=1e-8), key
        for name, beta in model["betas"].items():
            if name != "mkt_rf":
                assert beta["beta"] == pytest.approx(0.0, abs=1e-8), (key, name)


def test_known_answer_geometric_alpha_and_appraisal_ratio(window):
    """The two annualizations of a known alpha, and the appraisal ratio — the
    last of which cannot be checked on this construction at all.

    `rf + 1.0*mkt_rf + 0.03/252` is an EXACT linear combination of the factors,
    so its residual scale is rounding dust and the appraisal ratio it used to
    publish was 1.8e+14: a real 3%/yr divided by 1.7e-16 of idiosyncratic
    volatility. This test asserted that ratio was internally consistent, which it
    was — dust divided by dust to fourteen digits. The chokepoint withholds it
    now, and the identity is asserted where there is a residual to divide by.
    """
    strat = window["rf"] + 1.0 * window["mkt_rf"] + 0.03 / 252.0
    payload, _ = at.attribution_report(strat, attempts=1)
    capm = payload["models"]["capm"]
    # arithmetic 252·α is the reported headline; geometric is (1+α)^252 − 1.
    # Both are POINT ESTIMATES and survive the refusal.
    assert capm["alpha_annual"] == pytest.approx(0.03, abs=1e-9)
    assert capm["alpha_annual_geometric"] == pytest.approx(
        (1 + 0.03 / 252) ** 252 - 1, abs=1e-12)
    assert capm["alpha_testable"] is False
    assert capm["appraisal_ratio"] is None
    assert "appraisal_ratio" in capm["withheld"]

    # the same alpha with a real residual: the ratio is published and is exactly
    # annual alpha per unit of annual idiosyncratic volatility
    rng = np.random.default_rng(23)
    noisy = strat + pd.Series(rng.normal(0.0, 0.004, len(window)),
                              index=window.index)
    capm2 = at.attribution_report(noisy, attempts=1)[0]["models"]["capm"]
    assert capm2["alpha_testable"] is True
    assert capm2["appraisal_ratio"] == pytest.approx(
        capm2["alpha_annual"] / capm2["resid_vol_annual"], rel=1e-9)
    assert capm2["withheld"] == []


def test_multi_factor_known_answer(window):
    """A series built from three factors recovers all three loadings."""
    strat = (window["rf"] + 0.9 * window["mkt_rf"] - 0.4 * window["smb"]
             + 0.6 * window["mom"] + 0.01 / 252.0)
    payload, _ = at.attribution_report(strat, attempts=1)
    b = payload["models"]["ff6"]["betas"]
    assert b["mkt_rf"]["beta"] == pytest.approx(0.9, abs=1e-8)
    assert b["smb"]["beta"] == pytest.approx(-0.4, abs=1e-8)
    assert b["mom"]["beta"] == pytest.approx(0.6, abs=1e-8)
    assert b["hml"]["beta"] == pytest.approx(0.0, abs=1e-8)
    assert payload["models"]["ff6"]["alpha_annual"] == pytest.approx(0.01, abs=1e-8)


def test_rolling_beta_tracks_a_regime_shift(window):
    """Beta 0.5 for the first half, 1.5 for the second: the rolling series must
    start near 0.5 and end near 1.5.

    The rolling level is the THIRD place this module fits a regression, and it
    asks the same degeneracy predicate as the other two. This construction is an
    exact combination inside each half, so the windows that sit entirely within a
    regime are degenerate and say so; the windows that straddle the break have a
    real residual and are testable. Both kinds appear here, which is why the
    per-window reason code is worth carrying.
    """
    half = len(window) // 2
    loading = pd.Series(0.5, index=window.index)
    loading.iloc[half:] = 1.5
    strat = window["rf"] + loading * window["mkt_rf"]
    payload, _ = at.attribution_report(strat, attempts=1)

    roll = payload["rolling_market_beta"]
    assert len(roll) > 20
    assert roll[0]["beta"] == pytest.approx(0.5, abs=1e-6)
    assert roll[-1]["beta"] == pytest.approx(1.5, abs=1e-6)
    assert all(set(p) == {"date", "beta", "alpha_annual", "alpha_testable",
                          "untestable_kind"} for p in roll)
    # the level really does ask the predicate, and really does get both answers
    kinds = {p["untestable_kind"] for p in roll}
    assert kinds == {None, "no_testable_intercept"}, kinds
    assert all((p["untestable_kind"] is None) is p["alpha_testable"] for p in roll)


# ---------------------------------------------------- the HAC machinery

def test_hac_matches_ols_on_iid_residuals():
    """With i.i.d. homoskedastic residuals the HAC sandwich has nothing to fix,
    so it must land close to the classical OLS standard error."""
    rng = np.random.default_rng(11)
    n = 4000
    x = rng.normal(0, 0.01, n)
    y = 0.001 + 2.0 * x + rng.normal(0, 0.005, n)
    X = np.column_stack([np.ones(n), x])
    fit = at.ols_hac(y, X, lags=21)
    assert fit.coefs[1] == pytest.approx(2.0, abs=0.05)
    for hac, ols in zip(fit.se_hac, fit.se_ols):
        assert hac == pytest.approx(ols, rel=0.30)


def test_hac_exceeds_ols_on_autocorrelated_residuals():
    """Persistent residuals are exactly the case OLS errors understate. The
    whole reason we pay for Newey-West is that the HAC error must be larger."""
    rng = np.random.default_rng(5)
    n = 3000
    x = rng.normal(0, 0.01, n)
    u = np.zeros(n)
    for t in range(1, n):
        u[t] = 0.8 * u[t - 1] + rng.normal(0, 0.004)
    y = 0.0005 + 1.0 * x + u
    X = np.column_stack([np.ones(n), x])
    fit = at.ols_hac(y, X, lags=21)
    assert fit.se_hac[0] > 2.0 * fit.se_ols[0]


def test_hac_standard_errors_match_an_independently_derived_reference():
    """Golden values, not a re-derivation.

    The three standard errors below were produced OUTSIDE this module by an
    explicit double loop over (t, lag) written straight from Newey & West
    (1987) — S = Σ_t u_t² x_t x_t' + Σ_{l=1..L} (1 − l/(L+1)) Σ_{t>l} u_t u_{t−l}
    (x_t x_{t−l}' + x_{t−l} x_t'), V = (X'X)⁻¹ S (X'X)⁻¹ · T/(T−k) — and then
    hard-coded here as literals. No line of `ols_hac` participates in producing
    the expected value, so this test can actually fail.

    It is also a mutation detector on the three pieces the sandwich is made of.
    Against se_hac[0] = 3.22798060916071e-04 the same data gives:
      * Bartlett weights dropped (w = 1):          3.43230e-04  (+6.3%)
      * small-sample T/(T−k) correction dropped:   3.22188e-04  (−0.19%)
      * lag truncation L = 0 (White, no HAC):      1.73150e-04  (−46%)
    all of which blow through the 1e-9 tolerance.
    """
    rng = np.random.default_rng(20240917)
    n = 800
    x1 = rng.normal(0.0, 0.01, n)
    x2 = rng.normal(0.0, 0.006, n)
    e = rng.normal(0.0, 0.004, n)
    u = np.zeros(n)
    u[0] = e[0]
    for t in range(1, n):
        u[t] = 0.6 * u[t - 1] + e[t]          # AR(1), the case HAC exists for
    y = 0.0004 + 1.2 * x1 - 0.7 * x2 + u
    X = np.column_stack([np.ones(n), x1, x2])

    fit = at.ols_hac(y, X, lags=21)

    assert fit.coefs == pytest.approx(
        [7.285008370636268e-06, 1.1729570651257712, -0.7533192980519604], rel=1e-9)
    assert fit.se_hac == pytest.approx(
        [3.22798060916071e-04, 1.677834087181931e-02, 2.8343769383902293e-02],
        rel=1e-9)
    # and the mutants those literals are chosen to kill
    assert fit.se_hac[0] != pytest.approx(3.43230e-04, rel=1e-3)   # no Bartlett
    assert fit.se_hac[0] != pytest.approx(3.22188e-04, rel=1e-4)   # no T/(T−k)
    assert fit.se_hac[0] != pytest.approx(1.73150e-04, rel=1e-3)   # no HAC at all


def test_hac_with_zero_lags_is_white_heteroskedasticity_robust():
    """L=0 collapses the Bartlett sum, leaving the White (HC0) sandwich, up to
    the T/(T−k) small-sample scaling this module applies."""
    rng = np.random.default_rng(2)
    n = 500
    x = rng.normal(0, 1, n)
    y = 1.0 + 0.5 * x + rng.normal(0, 1, n)
    X = np.column_stack([np.ones(n), x])
    fit = at.ols_hac(y, X, lags=0)

    xtx_inv = np.linalg.pinv(X.T @ X)
    coefs = xtx_inv @ (X.T @ y)
    u = y - X @ coefs
    meat = (X * u[:, None]).T @ (X * u[:, None])
    white = np.sqrt(np.diag(xtx_inv @ meat @ xtx_inv) * (n / (n - 2)))
    assert fit.se_hac == pytest.approx(white, rel=1e-12)


# --------------------------------------------------- the Student-t tail

def test_student_t_against_closed_forms():
    """df=1 is Cauchy and df=2 has a closed form — both are exact checks on the
    incomplete-beta implementation, no scipy required."""
    # Cauchy: P(|T| > 1) = 1 - 2·arctan(1)/π = 0.5
    assert at.t_two_sided_p(1.0, 1) == pytest.approx(0.5, abs=1e-10)
    # df=2: P(|T| > t) = 1 - t/sqrt(2 + t²)
    for t in (0.5, 1.0, 2.0, 4.0):
        assert at.t_two_sided_p(t, 2) == pytest.approx(
            1 - t / math.sqrt(2 + t * t), abs=1e-10)
    # large df converges on the normal: P(|Z| > 1.959964) = 0.05
    assert at.t_two_sided_p(1.959964, 5_000_000) == pytest.approx(0.05, abs=1e-4)
    # textbook critical value: df=10, two-tailed 5% is t = 2.228
    assert at.t_two_sided_p(2.228, 10) == pytest.approx(0.05, abs=5e-4)


def test_t_hurdle_inverts_the_tail():
    for df in (30, 250, 2500):
        for p in (0.05, 0.01, 0.00125):
            t = at.t_hurdle_for_p(p, df)
            assert at.t_two_sided_p(t, df) == pytest.approx(p, rel=1e-6)


def test_t_hurdle_rises_with_the_number_of_attempts():
    """The multiple-testing bar must get harder, never easier, as the user
    tries more variants."""
    df = 2000
    hurdles = [at.t_hurdle_for_p(0.05 / n, df) for n in (1, 5, 20, 40, 100)]
    assert hurdles == sorted(hurdles)
    # research calibration: ~40 variants puts the bar above 3
    assert hurdles[3] > 3.0


# ------------------------------------------------------ the ladder story

def constructed_factors(seed: int = 19, n: int = 2500) -> FactorData:
    """A controlled factor panel where RMW alone carries a high Sharpe.

    The real fixture cannot produce this test: over 2015-2026 no single style
    factor has a high enough in-sample Sharpe to give a CAPM intercept a t-stat
    above ~2, so the decay ratio the finding keys on is unreachable. Building
    the panel lets us pin the answer. It is marked synthetic, which is exactly
    how the module must treat any factor data that is not Ken French's.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2012-01-02", periods=n)
    frame = pd.DataFrame({
        "mkt_rf": rng.normal(0.0003, 0.010, n),
        "smb": rng.normal(0.0, 0.004, n),
        "hml": rng.normal(0.0, 0.004, n),
        "rmw": rng.normal(0.00035, 0.004, n),   # ~8.8%/yr at ~6.4% vol
        "cma": rng.normal(0.0, 0.004, n),
        "mom": rng.normal(0.0, 0.005, n),
        "rf": np.full(n, 0.00005),
    }, index=idx)
    frame.index.name = "date"
    return FactorData(frame=frame, synthetic=True, provenance="constructed in-test",
                      path="<constructed>", start=str(idx[0].date()),
                      end=str(idx[-1].date()))


def test_alpha_decay_names_the_factor_that_absorbed_it():
    """A series that is really an RMW (profitability) bet shows real CAPM alpha
    and loses it once the five factors are present — the finding must say so
    and name RMW."""
    fd = constructed_factors()
    f = fd.frame
    rng = np.random.default_rng(101)
    noise = pd.Series(rng.normal(0, 0.002, len(f)), index=f.index)
    strat = f["rf"] + 1.0 * f["mkt_rf"] + 1.0 * f["rmw"] + noise
    payload, findings = at.attribution_report(strat, attempts=1, factors=fd)

    capm_t = payload["models"]["capm"]["alpha_t_hac"]
    ff6_t = payload["models"]["ff6"]["alpha_t_hac"]
    assert payload["models"]["capm"]["alpha_annual"] > 0.05   # real CAPM alpha
    assert abs(capm_t) >= 2.0
    assert abs(ff6_t) < abs(capm_t) / 2.0                     # it decays away

    decay = [f for f in findings if f.check == "factor_alpha_decay"]
    assert decay, [f.check for f in findings]
    assert decay[0].severity == "warning"
    assert decay[0].evidence["absorbed_by"] == "rmw"
    assert "profitability" in decay[0].message
    assert "not skill" in decay[0].message


def test_no_alpha_finding_when_the_strategy_is_pure_beta(window):
    """Beta 1.0 and no skill must read as "no alpha survives".

    The idiosyncratic noise is load-bearing, not decoration. `rf + 1.0*mkt_rf`
    alone is an EXACT linear combination of the factors, so its t-statistic
    (-0.38, as it happened) is a rounding error rather than a measurement, and
    this branch is not the one such a series belongs in — see
    test_exact_factor_combination_refuses_the_alpha_verdict. A real pure-beta
    strategy has a residual, and that is the case asserted here.
    """
    rng = np.random.default_rng(11)
    noise = pd.Series(rng.normal(0, 0.002, len(window)), index=window.index)
    strat = window["rf"] + 1.0 * window["mkt_rf"] + noise
    payload, findings = at.attribution_report(strat, attempts=1)
    assert payload["models"]["ff6"]["alpha_testable"] is True
    alpha_f = [f for f in findings if f.check == "factor_alpha"][0]
    assert alpha_f.severity == "warning"
    assert "not statistically distinguishable from zero" in alpha_f.message
    assert "NOT statistically distinguishable" in payload["interpretation"]


def test_significant_alpha_is_still_held_to_the_multiple_testing_bar(window):
    """t clears 2.0 but the session has tried 60 variants — the finding must
    downgrade it and quote the Bonferroni hurdle."""
    rng = np.random.default_rng(23)
    noise = pd.Series(rng.normal(0, 0.002, len(window)), index=window.index)
    strat = window["rf"] + 1.0 * window["mkt_rf"] + 0.02 / 252.0 + noise
    payload, findings = at.attribution_report(strat, attempts=60)

    assert abs(payload["models"]["ff6"]["alpha_t_hac"]) >= 2.0
    alpha_f = [f for f in findings if f.check == "factor_alpha"][0]
    assert alpha_f.severity in ("warning", "pass")
    mt = payload["multiple_testing"]
    assert mt["attempts"] == 60
    assert mt["t_hurdle_bonferroni"] > 3.0
    assert mt["p_bonferroni"] >= mt["p_raw"]
    assert "60 strategy variants" in payload["interpretation"]


def test_the_bonferroni_haircut_is_applied_and_decides_the_verdict(window):
    """The multiple-testing correction itself, pinned where it CHANGES the answer.

    `p_bonferroni = min(1, attempts × p_raw)` was untested: the test above
    accepts severity "warning" OR "pass", so replacing the haircut with
    `min(1, p_raw)` — deleting the correction outright — left the whole module
    green. Multiple-testing is a headline honesty claim of this product; it needs
    a case where the corrected and the uncorrected verdicts DISAGREE.

    This is that case. Over 2,264 days the fitted alpha is +2.8%/yr (the series
    is built with 2.4%, and this noise draw's own mean adds the rest) with a HAC
    t of ~2.72, so the raw p is ~0.0065 — comfortably significant on its own.
    Twenty variants have been tried, so the honest p is ~0.131, comfortably NOT
    significant.
    The expected p-value is computed here from `math.erfc` (stdlib), not from
    this module: on 2,257 residual degrees of freedom the normal tail is within
    1% of the Student-t one, which is far tighter than the 20× the haircut
    itself is worth.
    """
    attempts = 20
    rng = np.random.default_rng(23)
    noise = pd.Series(rng.normal(0, 0.002, len(window)), index=window.index)
    strat = window["rf"] + 1.0 * window["mkt_rf"] + 0.024 / 252.0 + noise
    payload, findings = at.attribution_report(strat, attempts=attempts)

    head, mt = payload["models"]["ff6"], payload["multiple_testing"]
    t = head["alpha_t_hac"]
    assert head["n_obs"] == 2264 and head["df_resid"] == 2257
    assert head["alpha_annual"] == pytest.approx(0.0278, abs=0.002)   # POSITIVE
    assert t == pytest.approx(2.72, abs=0.05)
    assert mt["attempts"] == attempts

    # P(|Z| > t), independent of this module's incomplete-beta tail
    p_independent = math.erfc(abs(t) / math.sqrt(2.0))
    assert p_independent == pytest.approx(0.0065, abs=0.0005)
    assert mt["p_raw"] == pytest.approx(p_independent, rel=0.02)
    # the haircut: 20 × 0.0065 ≈ 0.131, which is NOT below 5%
    assert mt["p_bonferroni"] == pytest.approx(attempts * p_independent, rel=0.02)
    assert mt["p_bonferroni"] > 0.05 > mt["p_raw"]
    assert mt["survives_bonferroni"] is False
    assert mt["t_hurdle_bonferroni"] > abs(t)      # the bar it fails to clear

    # and the VERDICT follows the corrected p-value, not the raw one. Without
    # the haircut this alpha reads "Alpha survives the six-factor model", pass.
    alpha_f = [f for f in findings if f.check == "factor_alpha"][0]
    assert alpha_f.severity == "warning"
    assert alpha_f.name == (
        "Alpha is significant on its own, but not after multiple testing")
    assert "does not clear the bar for this session" in alpha_f.message
    assert not [f for f in findings
                if f.check == "factor_alpha" and f.severity == "pass"]


def test_significantly_negative_alpha_is_never_a_pass(window):
    """The verdict-inversion bug: every branch keyed on abs(t), so a strategy
    that reliably LOSES to its own factor exposures was reported as
    'Alpha survives the six-factor model', severity pass.

    Construction pins the sign independently of the module: the series IS the
    market minus 5%/yr, so the true six-factor alpha is −5%/yr by arithmetic.
    """
    rng = np.random.default_rng(7)
    noise = pd.Series(rng.normal(0, 0.002, len(window)), index=window.index)
    strat = window["rf"] + 1.0 * window["mkt_rf"] - 0.05 / 252.0 + noise
    payload, findings = at.attribution_report(strat, attempts=1)

    head = payload["models"]["ff6"]
    assert head["alpha_annual"] < -0.02              # genuinely negative
    assert head["alpha_t_hac"] <= -2.0               # and significantly so
    assert payload["multiple_testing"]["survives_bonferroni"] is True

    alpha_f = [f for f in findings if f.check == "factor_alpha"][0]
    assert alpha_f.severity == "warning"             # NOT "pass"
    assert "survives" not in alpha_f.name.lower()
    assert "NEGATIVE" in alpha_f.name or "NEGATIVE" in alpha_f.message
    assert "underperform" in alpha_f.message or "MORE than this strategy" in alpha_f.message
    assert "BELOW zero" in payload["interpretation"]
    # and no finding anywhere in this payload may read as a pass on alpha
    assert not [f for f in findings
                if f.check == "factor_alpha" and f.severity == "pass"]


def test_significantly_positive_alpha_can_still_pass(window):
    """The counterpart: the sign gate must not swallow a genuine positive."""
    rng = np.random.default_rng(7)
    noise = pd.Series(rng.normal(0, 0.002, len(window)), index=window.index)
    strat = window["rf"] + 1.0 * window["mkt_rf"] + 0.05 / 252.0 + noise
    payload, findings = at.attribution_report(strat, attempts=1)
    assert payload["models"]["ff6"]["alpha_t_hac"] >= 2.0
    alpha_f = [f for f in findings if f.check == "factor_alpha"][0]
    assert alpha_f.severity == "pass"
    assert "distinguishable from zero" in payload["interpretation"]


def test_degenerate_regression_refuses_instead_of_reporting_nan(window):
    """An all-cash series (the engine emits exactly this when nothing ever
    qualifies: excess return identically zero) gives se(alpha)=0, t=NaN, R²=NaN.

    Every threshold test — `abs(t) < 2`, `r2 > 0.99` — is False for NaN, so the
    old code routed it into the 'clears the conventional bar' branch AND the
    'the regression is well-posed' branch, then printed 'n/a' and 'nan%' in the
    same payload. NaN discipline means refuse.
    """
    cash = window["rf"].copy()                       # excess return ≡ 0
    payload, findings = at.attribution_report(cash, attempts=1)

    assert payload["available"] is False
    assert payload["reason"] == "degenerate_regression"
    refusal = [f for f in findings if f.check == "factor_attribution"]
    assert refusal and refusal[0].severity == "note"
    assert "degenerate" in refusal[0].message

    # no verdict of any kind may have been emitted
    assert not [f for f in findings if f.check in ("factor_alpha", "factor_reliability")]
    # and no rendered NaN can reach a reader
    for f in findings:
        assert "nan" not in f.message.lower()
    assert "nan" not in str(payload.get("message", "")).lower()


def test_absorbing_factor_is_ranked_by_contribution_not_by_t_stat():
    """A loading's t-stat says how PRECISELY the beta is estimated, not how much
    alpha it absorbed. Here momentum is demeaned to exactly zero mean, so a 2.5
    beta on it — estimated with an enormous t — absorbs exactly nothing, while
    RMW's 0.5 beta on a 30%/yr factor eats the whole CAPM alpha.

    Independent check: absorption is beta_k × mean(f_k) × 252, computed here
    from the constructed panel, not read back from the module.
    """
    n = 2500
    idx = pd.bdate_range("2012-01-02", periods=n)
    rng = np.random.default_rng(3)
    mom = rng.normal(0.0, 0.005, n)
    mom = mom - mom.mean()                           # EXACTLY zero mean
    frame = pd.DataFrame({
        "mkt_rf": rng.normal(0.0003, 0.010, n),
        "smb": rng.normal(0.0, 0.004, n),
        "hml": rng.normal(0.0, 0.004, n),
        "rmw": rng.normal(0.0012, 0.004, n),         # ~30%/yr
        "cma": rng.normal(0.0, 0.004, n),
        "mom": mom,
        "rf": np.full(n, 0.00005),
    }, index=idx)
    frame.index.name = "date"
    fd = FactorData(frame=frame, synthetic=True, provenance="constructed in-test",
                    path="<constructed>", start=str(idx[0].date()),
                    end=str(idx[-1].date()))
    strat = (frame["rf"] + 1.0 * frame["mkt_rf"] + 0.5 * frame["rmw"]
             + 2.5 * frame["mom"]
             + pd.Series(rng.normal(0, 0.002, n), index=idx))
    payload, findings = at.attribution_report(strat, attempts=1, factors=fd)

    betas = payload["models"]["ff6"]["betas"]
    # momentum is the most precisely estimated non-market loading, so the old
    # rule ("largest |t| among the non-market loadings") would name it ...
    by_t = {n: abs(b["t_hac"]) for n, b in betas.items() if n != "mkt_rf"}
    assert max(by_t, key=by_t.get) == "mom"
    assert by_t["mom"] > 5.0 * by_t["rmw"]
    # ... and absorbs nothing, because its factor mean is zero
    assert betas["mom"]["beta"] * frame["mom"].mean() * 252 == pytest.approx(0.0, abs=1e-9)
    hand_rmw = betas["rmw"]["beta"] * float(frame["rmw"].mean()) * 252

    decay = [f for f in findings if f.check == "factor_alpha_decay"]
    assert decay, [f.check for f in findings]
    assert decay[0].evidence["absorbed_by"] == "rmw"        # not "mom"
    assert decay[0].evidence["absorbed_annual"] == pytest.approx(hand_rmw, rel=1e-9)
    assert "profitability" in decay[0].message
    assert decay[0].evidence["factor_contributions_annual"]["mom"] == pytest.approx(
        0.0, abs=1e-9)


def test_absorbing_factor_follows_the_direction_the_alpha_moved():
    """The sign branch in `_absorbing_factor`, which nothing else reaches.

    "Which factor ate the alpha" is a question about the direction alpha MOVED.
    When the ladder cuts alpha (decay > 0) the culprit is the largest POSITIVE
    contribution; when adding factors RAISES alpha (decay < 0) it is the largest
    negative one. Hard-coding direction = 1.0 names the positive contributor in
    both cases — i.e. names a factor that pushed alpha the other way.

    Hand arithmetic: contribution_k = β_k × mean(f_k) × 252, so size is
    +1.0 × 0.001 × 252 = +0.252 and value is −1.0 × 0.002 × 252 = −0.504.
    """
    headline = {"betas": {"mkt_rf": {"beta": 1.0},
                          "smb": {"beta": 1.0},
                          "hml": {"beta": -1.0}}}
    f = pd.DataFrame({"mkt_rf": [0.001] * 4, "smb": [0.001] * 4,
                      "hml": [0.002] * 4})
    assert at._factor_contributions(headline, f) == {
        "smb": pytest.approx(0.252, rel=1e-12),
        "hml": pytest.approx(-0.504, rel=1e-12),
    }

    # alpha fell: the factor that absorbed it is the positive contributor
    name, absorbed, factor_ret = at._absorbing_factor(headline, f, decay=+0.10)
    assert name == "smb"
    assert absorbed == pytest.approx(0.252, rel=1e-12)
    assert factor_ret == pytest.approx(0.252, rel=1e-12)

    # alpha ROSE: the same call must name the negative contributor instead
    name, absorbed, factor_ret = at._absorbing_factor(headline, f, decay=-0.10)
    assert name == "hml"
    assert absorbed == pytest.approx(-0.504, rel=1e-12)
    assert factor_ret == pytest.approx(0.504, rel=1e-12)

    # no non-market loading at all: nothing to name, and no crash
    assert at._absorbing_factor({"betas": {"mkt_rf": {"beta": 1.0}}}, f,
                                decay=+0.10) is None


# ------------------------------------------------------- spanning test

def orthogonal_noise(index, mkt, sd=0.002, seed=19) -> pd.Series:
    """Genuine idiosyncratic noise with the market projected OUT of it.

    A spanning test on an EXACT benchmark clone has zero residual variance and
    therefore no testable intercept (see
    test_exact_benchmark_clone_refuses_the_spanning_verdict), so a test that
    wants the REAL 'spanned by the benchmark' verdict has to give the residual
    something to be. Projecting the constant and the market out of the noise
    keeps the construction exact anyway: the fitted beta is still 0.7 and the
    fitted intercept still 0 to 1e-9, but now against a real residual variance.
    """
    rng = np.random.default_rng(seed)
    raw = rng.normal(0.0, sd, len(index))
    m = np.asarray(mkt, dtype=float)
    X = np.column_stack([np.ones(len(m)), m])
    coefs = np.linalg.lstsq(X, raw, rcond=None)[0]
    return pd.Series(raw - X @ coefs, index=index)


def test_spanning_flags_a_levered_benchmark_clone(window):
    """0.7 × SPY plus idiosyncratic noise: beta 0.70, no intercept, and a finding
    that says it in the language a PM would use."""
    bench = window["rf"] + window["mkt_rf"]
    noise = orthogonal_noise(window.index, window["mkt_rf"])
    strat = window["rf"] + 0.7 * (bench - window["rf"]) + noise
    payload, findings = at.attribution_report(strat, bench_returns=bench,
                                              bench_label="SPY", attempts=1)
    sp = payload["spanning"]
    assert sp["betas"]["bench_excess"]["beta"] == pytest.approx(0.7, abs=1e-9)
    assert sp["alpha_annual"] == pytest.approx(0.0, abs=1e-9)
    # the intercept really is testable here — that is what makes the verdict a
    # verdict rather than an artefact
    assert sp["alpha_testable"] is True
    assert sp["verdict"]["available"] is True
    assert not sp["alpha_significant"]

    span_f = [f for f in findings if f.check == "factor_spanning"]
    assert span_f and span_f[0].severity == "warning"
    assert "70% SPY" in span_f[0].message
    assert "70% SPY" in sp["plain_english"]


def test_exact_benchmark_clone_refuses_the_spanning_verdict(window):
    """The regression THIS round was introduced by, from the other direction.

    `alpha_testable` was folded into `alpha_significant`, which is read by three
    unrelated call sites. The spanning branches have no untestable case, so an
    exact clone of the benchmark carrying a REAL +5%/yr constant — a fit with
    zero residual variance, intercept exactly +5.0%/yr and t = 4e14 — stopped
    being merely silent and started emitting a warning that said
    "+5.0%/yr (t = 397831687950515.81) ... no risk-adjusted value added", a
    sentence contradicted by its own first half. A fabricated PASS flatters; a
    fabricated REFUTATION misleads in the other direction while looking rigorous.

    Both clones below are degenerate and both must refuse in BOTH directions:
    one whose true intercept is zero, one whose true intercept is +5%/yr.
    """
    bench = window["rf"] + window["mkt_rf"]
    for label, strat, true_alpha in (
        ("zero intercept", window["rf"] + 1.0 * (bench - window["rf"]), 0.0),
        ("real +5%/yr intercept",
         window["rf"] + 1.0 * (bench - window["rf"]) + 0.05 / 252.0, 0.05),
    ):
        payload, findings = at.attribution_report(
            strat, bench_returns=bench, bench_label="SPY", attempts=1)
        sp = payload["spanning"]

        # the fit is the degenerate-without-NaN one, and the point estimate is
        # exactly the constructed constant — it is the STANDARD ERROR that is dust
        assert sp["resid_var_share"] <= at.RESID_VAR_SHARE_FLOOR, label
        assert sp["alpha_annual"] == pytest.approx(true_alpha, abs=1e-9), label
        assert sp["alpha_testable"] is False, label

        # the payload carries an UNAVAILABLE verdict with its reason, not a verdict
        assert sp["verdict"]["available"] is False, label
        assert sp["verdict"]["reason"] == "no_testable_intercept", label
        assert "UP TO A CONSTANT" in sp["verdict"]["message"], label

        span_f = [f for f in findings if f.check == "factor_spanning"]
        assert len(span_f) == 1, label
        assert span_f[0].severity == "note", label
        assert "could not be tested" in span_f[0].name, label

        # neither direction may be asserted, anywhere a reader looks
        for text in ([sp["plain_english"], payload["interpretation"]]
                     + [f.message for f in findings]):
            assert "no risk-adjusted value added" not in text, (label, text)
            assert "distinguishable from zero" not in text, (label, text)
            assert "value added" not in text or "neither" in text, (label, text)
            # and no t-statistic from a zero-residual fit reaches the page
            assert not re.search(r"\bt\s*[=>]\s*[-+]?\d", text), (label, text)
        # specifically not the 15-digit one, nor the +5.0%/yr beside it
        assert "397831687950515" not in sp["plain_english"], label
        assert f"{true_alpha:+.1%}" not in sp["plain_english"] or true_alpha == 0.0

        # the beta IS supported by this fit, and is still reported
        assert sp["betas"]["bench_excess"]["beta"] == pytest.approx(1.0, abs=1e-9)
        assert "beta of 1.00" in span_f[0].message, label
        assert "100% SPY" in span_f[0].message, label


def test_levered_clone_is_described_as_borrowing_not_negative_cash(window):
    """beta > 1 must read as leverage, not as '-21% cash', which reads as a bug."""
    bench = window["rf"] + window["mkt_rf"]
    strat = window["rf"] + 1.25 * (bench - window["rf"])
    payload, _ = at.attribution_report(strat, bench_returns=bench,
                                       bench_label="SPY", attempts=1)
    phrase = payload["spanning"]["replicating_portfolio"]
    assert "125% SPY" in phrase and "borrowed" in phrase
    assert "cash" not in phrase


def test_a_significantly_negative_spanning_intercept_is_reported(window):
    """The benchmark gate used to fire only when the intercept was INSIGNIFICANT,
    so 'you lose to SPY by 5%/yr at the same beta' produced silence."""
    rng = np.random.default_rng(7)
    noise = pd.Series(rng.normal(0, 0.002, len(window)), index=window.index)
    bench = window["rf"] + window["mkt_rf"]
    strat = window["rf"] + 1.0 * window["mkt_rf"] - 0.05 / 252.0 + noise
    payload, findings = at.attribution_report(strat, bench_returns=bench,
                                              bench_label="SPY", attempts=1)
    sp = payload["spanning"]
    assert sp["alpha_annual"] < -0.02 and sp["alpha_significant"]
    span_f = [f for f in findings if f.check == "factor_spanning"]
    assert span_f and span_f[0].severity == "warning"
    assert "BELOW zero" in span_f[0].message
    assert "significantly NEGATIVE" in sp["plain_english"]
    assert "WORSE than the benchmark" in sp["plain_english"]
    # the old text called this "NOT statistically distinguishable from zero"
    assert "NOT statistically distinguishable" not in sp["plain_english"]


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_benchmark_with_unusable_values_skips_spanning_instead_of_crashing(
        window, bad):
    """Index coverage was checked, values were not: a benchmark with NaN on
    covered dates reached pinv and raised LinAlgError out of the whole report.

    ±inf is the same failure and must take the same path. `isna()` is False for
    an infinity, so the NaN-only version of this guard let an inf through to
    pinv, which raised `LinAlgError: SVD did not converge` out of
    attribution_report — the crash the note exists to prevent.
    """
    bench = (window["rf"] + window["mkt_rf"]).copy()
    bench.iloc[100:110] = bad
    strat = window["rf"] + 1.1 * window["mkt_rf"]
    payload, _ = at.attribution_report(strat, bench_returns=bench, attempts=1)
    assert payload["available"] is True
    assert payload["spanning"] is None
    assert payload["notes"]
    assert "10 missing or non-finite value(s)" in payload["notes"][0]
    assert "spanning test was skipped" in payload["notes"][0]


def test_spanning_skipped_when_benchmark_does_not_cover_the_window(window):
    bench = (window["rf"] + window["mkt_rf"]).iloc[50:]
    strat = window["rf"] + window["mkt_rf"]
    payload, _ = at.attribution_report(strat, bench_returns=bench, attempts=1)
    assert payload["spanning"] is None
    assert payload["notes"] and "spanning test was skipped" in payload["notes"][0]


# --------------------------------------------------- refusals and guards

def test_nan_in_strategy_series_is_refused(window):
    strat = (window["rf"] + window["mkt_rf"]).copy()
    strat.iloc[10] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        at.attribution_report(strat, attempts=1)


@pytest.mark.parametrize("bad", [np.inf, -np.inf])
def test_infinity_in_strategy_series_is_refused_as_broken_data(window, bad):
    """An inf must take the NaN path, not the degenerate-regression one.

    `isna()` is False for ±inf, so an infinite return used to pass the entry
    check, propagate through the design matrix as NaN coefficients, and surface
    as `reason='degenerate_regression'` — whose message says the series "never
    left cash". That is the wrong diagnosis of an arithmetically broken series,
    and it is reported to the reader as prose.
    """
    strat = (window["rf"] + window["mkt_rf"]).copy()
    strat.iloc[10] = bad
    with pytest.raises(ValueError, match="non-finite"):
        at.attribution_report(strat, attempts=1)


def test_short_overlap_returns_a_finding_not_a_number(window):
    """Six months of data cannot support an alpha estimate; the module must say
    so rather than print one."""
    short = (window["rf"] + window["mkt_rf"]).iloc[:120]
    payload, findings = at.attribution_report(short, attempts=1)
    assert payload["available"] is False
    assert payload["reason"] == "insufficient_overlap"
    refusal = [f for f in findings if f.check == "factor_attribution"]
    assert refusal and "no alpha is reported" in refusal[0].message


def test_window_outside_factor_coverage_is_refused(factors):
    """A backtest that ends before the fixture starts has zero overlap."""
    idx = pd.bdate_range("2005-01-03", periods=600)
    strat = pd.Series(0.0004, index=idx)
    payload, findings = at.attribution_report(strat, attempts=1)
    assert payload["available"] is False
    assert payload["n_obs"] == 0
    assert any(f.check == "factor_attribution" for f in findings)


def test_percent_scaled_factor_file_is_rejected(tmp_path, window):
    """The classic silent corruption: factors left in percent. The units
    assertion must turn it into an exception, not a 100× smaller beta."""
    bad = tmp_path / "percent.csv"
    (window * 100.0).to_csv(bad)
    with pytest.raises(ValueError, match="percent rather than decimal"):
        at.load_factors(bad)


def test_nan_in_factor_file_is_rejected(tmp_path, window):
    frame = window.copy()
    frame.iloc[3, 0] = np.nan
    path = tmp_path / "holey.csv"
    frame.to_csv(path)
    with pytest.raises(ValueError, match="NaN"):
        at.load_factors(path)


def test_missing_factor_column_is_rejected(tmp_path, window):
    path = tmp_path / "partial.csv"
    window.drop(columns=["mom"]).to_csv(path)
    with pytest.raises(ValueError, match="missing columns"):
        at.load_factors(path)


def test_missing_fixture_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        at.load_factors(tmp_path / "nope.csv")


# ------------------------------------------- synthetic data can never hide

def test_synthetic_factors_are_labelled_everywhere(tmp_path, window):
    """If the fixture is ever replaced by a placeholder, the payload, a WARNING
    finding and the plain-English sentence must all say so unmistakably."""
    fd = at.load_factors(synthetic_copy(tmp_path, window))
    assert fd.synthetic is True

    strat = window["rf"] + window["mkt_rf"] + 0.05 / 252.0
    payload, findings = at.attribution_report(strat, attempts=1, factors=fd)

    assert payload["factors"]["synthetic"] is True
    assert payload["interpretation"].startswith("ILLUSTRATIVE ONLY (synthetic factors)")
    syn = [f for f in findings if f.check == "factor_data_synthetic"]
    assert syn and syn[0].severity == "warning"
    assert "SYNTHETIC" in syn[0].message and "ILLUSTRATIVE ONLY" in syn[0].message


def test_committed_fixture_is_not_synthetic(factors):
    assert factors.synthetic is False
    assert "ken.french" in factors.provenance.lower()


# ------------------------------------------------ diagnostics / reliability

def test_diagnostics_pass_on_a_well_posed_regression(window):
    rng = np.random.default_rng(31)
    noise = pd.Series(rng.normal(0, 0.005, len(window)), index=window.index)
    strat = window["rf"] + window["mkt_rf"] + noise
    payload, findings = at.attribution_report(strat, attempts=1)
    rel = [f for f in findings if f.check == "factor_reliability"][0]
    assert rel.severity == "pass"
    # the scaled condition index must stay small — an unscaled one would be
    # enormous for every regression and the check would mean nothing
    assert payload["diagnostics"]["condition_number"] < at.CONDITION_BAR
    assert all(v < at.VIF_BAR for v in payload["diagnostics"]["vif"].values())


def test_perfect_fit_is_flagged_as_a_factor_portfolio(window):
    """R² of 1.0 means the intercept carries no information about skill."""
    strat = window["rf"] + 1.2 * window["mkt_rf"]
    payload, findings = at.attribution_report(strat, attempts=1)
    rel = [f for f in findings if f.check == "factor_reliability"][0]
    assert rel.severity == "warning"
    assert any("factor portfolio" in i or "factor portfolio" in i
               for i in payload["diagnostics"]["issues"])


def test_exact_factor_combination_refuses_the_alpha_verdict(window):
    """The same construction as the test above, judged on ALPHA rather than R².

    `rf + 1.2*mkt_rf` is an exact linear combination of the factors, so SSR is
    rounding dust rather than zero. That leaves the NaN guard in
    attribution_report untripped (t and R² are both finite) while alpha = 1.6e-16
    and se(alpha) = 2.8e-19 divide into t = 2.32, which reached the reader as
    severity "pass" / "Alpha survives the six-factor model" and the prose
    "statistically distinguishable from zero" — in the SAME payload that says R²
    of 1.000 means the strategy IS a factor portfolio. One report, two
    contradicting verdicts, and the surviving one was fabricated.
    """
    strat = window["rf"] + 1.2 * window["mkt_rf"]
    payload, findings = at.attribution_report(strat, attempts=1)

    # The report is still produced — the R² finding above is the useful half.
    assert payload["available"] is True
    head = payload["models"]["ff6"]
    # The fit really is the degenerate-without-NaN one this test is about.
    assert head["r2"] == 1.0
    assert np.isfinite(head["alpha_t_hac"])
    assert head["resid_var_share"] <= at.RESID_VAR_SHARE_FLOOR

    # The verdict is refused, not passed.
    assert head["alpha_testable"] is False
    # `alpha_significant` is still NOT conjoined with the predicate — the flag is
    # computed from |t| alone, because three call sites read it and they need
    # different untestable behaviour (see
    # test_exact_benchmark_clone_refuses_the_spanning_verdict). What changed is
    # that the row no longer PUBLISHES it: on this fit it used to export
    # `alpha_significant: True` beside `alpha_testable: False`, a fourth rendering
    # of the verdict, in the same dict. None is "no verdict exists", which is the
    # shape `survives_bonferroni` and the spanning verdict already use — neither
    # True nor the false refutation False would be.
    assert head["alpha_significant"] is None
    assert "alpha_significant" in head["withheld"]
    alpha_f = [f for f in findings if f.check == "factor_alpha"]
    assert len(alpha_f) == 1
    assert alpha_f[0].severity == "note"
    assert not [f for f in findings
                if f.check == "factor_alpha" and f.severity == "pass"]
    assert "could not be tested" in alpha_f[0].name.lower()
    assert "never as a pass" in alpha_f[0].message

    # And the fabricated t-statistic does not reach the reader anywhere. The
    # check is the FORMATTED t as this module would print it, read back from the
    # payload — not a literal. A test that forbids the string "2.32" pins one
    # magic number and passes for every other c in the same family.
    assert "NOT TESTABLE" in payload["interpretation"]
    assert "distinguishable from zero" not in payload["interpretation"]
    printed = {f"{payload['models'][k]['alpha_t_hac']:.2f}" for k in payload["ladder"]}
    assert printed, "nothing to forbid means the check is vacuous"
    for text in [payload["interpretation"]] + [
            f.message for f in findings
            if f.check in ("factor_alpha", "factor_alpha_decay", "factor_spanning")]:
        assert not (printed & set(re.findall(r"-?\d+\.\d\d", text))), text
        assert not re.search(r"\bt\s*[=>]\s*[-+]?\d", text), text
    for text in [payload["interpretation"]] + [f.message for f in findings]:
        assert "nan" not in text.lower()

    # The contradiction is gone: the R² half still fires, the alpha half refuses.
    assert any("factor portfolio" in i for i in payload["diagnostics"]["issues"])


def test_ladder_decay_refuses_when_only_one_rung_is_testable(window, factors):
    """The decay gate's other side, driven directly.

    The gate needs a testable intercept on BOTH rungs. Ordinary data cannot
    produce (CAPM untestable, headline testable): adding regressors cannot raise
    SSR and SST is unchanged, so the headline's residual share is never the
    larger one. That is an algebraic argument about quantities computed in
    floating point, and two shares an ulp either side of the floor invert it, so
    the branch is a real guard rather than decoration — and an unexercised guard
    is a guess. This drives it with a real report's own regressions and one flag
    flipped, which is the only honest way to reach a float coincidence.
    """
    noise = orthogonal_noise(window.index, window["mkt_rf"], sd=0.004, seed=5)
    strat = window["rf"] + 1.0 * window["mkt_rf"] + 0.04 / 252.0 + noise
    j = at._join(strat, factors.frame, None)
    models = {}
    for key, (label, cols) in at.MODEL_LADDER.items():
        row = at.regress(j.excess, j.factors, cols, at.TRADING_DAYS,
                         at.HAC_LAGS_DAILY, key)
        row["label"] = label
        models[key] = row
    headline = models[at.HEADLINE_MODEL]
    assert headline["alpha_testable"] is True
    assert models["capm"]["alpha_testable"] is True        # as data alone gives it
    models["capm"]["alpha_testable"] = False               # the ulp inversion

    mt = {"attempts": 1, "alpha_testable": True, "p_raw": headline["alpha_p_hac"],
          "p_bonferroni": headline["alpha_p_hac"], "t_hurdle_bonferroni": 1.96,
          "t_hurdle_harvey_liu_zhu": 3.0, "survives_bonferroni": True}
    payload = {
        "window": {"backtest_start": str(j.excess.index[0].date()),
                   "backtest_end": str(j.excess.index[-1].date()),
                   "attribution_start": str(j.excess.index[0].date()),
                   "attribution_end": str(j.excess.index[-1].date()),
                   "coverage": j.coverage, "n_obs": len(j.excess)},
        "spanning": None,
        "diagnostics": {"issues": []},
    }
    out = at._findings(models, headline, mt, j, factors, payload, "SPY")

    decay = [f for f in out if f.check == "factor_alpha_decay"]
    assert len(decay) == 1
    assert decay[0].severity == "note"
    assert "could not be tested" in decay[0].name
    assert "BOTH" in decay[0].message
    assert not re.search(r"\bt\s*[=>]\s*[-+]?\d", decay[0].message)
    assert decay[0].evidence["capm_alpha_testable"] is False
    # the headline verdict itself is untouched: this refusal is scoped to the
    # ladder comparison, not widened into the alpha verdict beside it
    alpha_f = [f for f in out if f.check == "factor_alpha"]
    assert len(alpha_f) == 1 and alpha_f[0].severity in ("pass", "warning")


# The reviewer's grid: rf + c*mkt_rf for c in [0.05, 3.0) step 0.005. Every
# member is an exact linear combination of the factors, so every fit on it is
# degenerate, and the t-statistics along it wander over the whole plausible range
# because they are rounding errors divided by rounding errors.
#
# THIS AXIS IS ONE-SIDED, and that is why it could not catch the round-6 defect.
# `rf + c*mkt_rf` drives the NUMERATOR of SSR/SST to dust while SST stays ~1.0 by
# construction: y always carries the market's own variation. Its non-vacuity
# assertion (`resid_var_share <= FLOOR`) is satisfied BY CONSTRUCTION on this
# family, so no sweep of c can ever reach the regime where the DENOMINATOR
# collapses. DENOMINATOR_COLLAPSE_GRID below is the second axis, and
# test_degenerate_grid_has_two_axes_and_they_are_disjoint asserts the two families
# really do land in different regimes rather than both testing the same one.
DEGENERATE_GRID = [round(0.05 + 0.005 * i, 4)
                   for i in range(int(round((3.0 - 0.05) / 0.005)))]

# The SECOND axis: a constant excess return. `rf + k/252` has excess ≡ k/252, so
# SST is rounding dust while Σy² is not, SSR/SST comes out far ABOVE 1 and R² far
# below 0 — both arithmetically impossible for least squares with an intercept.
# 41 constants from -10%/yr to +10%/yr, including the exactly-zero case (which is
# the older NaN degeneracy and must refuse the whole report instead).
DENOMINATOR_COLLAPSE_GRID = [round(i * 0.005, 4) for i in range(-20, 21)]


def test_no_degenerate_fit_reaches_the_reader_anywhere_on_the_grid(window):
    """One scan, all three verdicts, 590 fits. This is the test the previous
    round should have written.

    What it replaces: a test that forbade the literal string "2.32" in the
    findings for the single case c = 1.2. That pinned one magic number instead of
    the behaviour, so it stayed green while `factor_alpha_decay` — ungated —
    went on emitting "CAPM alpha is +0.0%/yr (t = 2.64), but adding size, value,
    profitability, investment and momentum cuts it to +0.0%/yr (t = 0.53). The
    loading that absorbed it is value (HML)..." at severity WARNING for 215 of
    the 590 values on this grid. Every number in that sentence is one rounding
    error divided by another.

    The invariants, for EVERY member of the grid:
      * the fit is degenerate (residual share at or below the floor) and
        `alpha_testable` is False on the headline, on CAPM and on the spanning
        regression — so the scan cannot go quietly vacuous if the fixture or the
        arithmetic changes;
      * no finding from `factor_alpha`, `factor_alpha_decay` or
        `factor_spanning` carries a severity above "note" — the whole family of
        verdicts is refused, not just the one that was noticed;
      * no reader-facing text quotes a t-statistic, in the module's own print
        format, from a zero-residual fit;
      * the verdict-bearing booleans downstream of that same t — the spanning
        verdict and `survives_bonferroni` — are refused too, and
        `survives_bonferroni` is None rather than False: a bare False asserts
        "tested, did not survive", which is a negative verdict on a fit that was
        never testable;
      * the assertions that were MISSING in the other direction, added in round 6:
        `resid_var_share <= 1.0` and `r2 >= 0` on every fit this test touches.
        They hold trivially on this family — that is the point. They are the
        identity (SSR ≤ SST for least squares with an intercept) that the second
        axis violates, so having them here says explicitly which regime this axis
        does NOT reach, instead of leaving the gap unnamed.

    Re-exposing ANY ONE of the three findings fails this test, which is the
    property the old one lacked. It costs ~30s: 590 full reports on 2,264
    sessions, which is the price of the coverage.
    """
    assert len(DEGENERATE_GRID) == 590
    assert DEGENERATE_GRID[0] == 0.05 and DEGENERATE_GRID[-1] == 2.995
    bench = window["rf"] + window["mkt_rf"]
    gated = ("factor_alpha", "factor_alpha_decay", "factor_spanning")
    t_pattern = re.compile(r"\bt\s*[=>]\s*[-+]?\d")
    two_dp = re.compile(r"-?\d+\.\d\d")
    old_decay_gate_trips = 0
    ungated_ci_excluded_zero = 0

    for c in DEGENERATE_GRID:
        strat = window["rf"] + c * window["mkt_rf"]
        payload, findings = at.attribution_report(
            strat, bench_returns=bench, bench_label="SPY", attempts=1)
        head = payload["models"]["ff6"]
        capm = payload["models"]["capm"]
        sp = payload["spanning"]

        for model in (head, capm, sp):
            assert model["resid_var_share"] <= at.RESID_VAR_SHARE_FLOOR, (c, model["key"])
            assert model["alpha_testable"] is False, (c, model["key"])
            # the identity, asserted in the direction this axis cannot break: the
            # numerator is dust, so the ratio is small and R² pins to 1.0
            assert model["resid_var_share"] <= 1.0, (c, model["key"])
            assert model["r2"] >= 0.0, (c, model["key"], model["r2"])
            # Both shares are dust here because the NUMERATOR is dust, and the two
            # denominators agree to a fraction of a percent — SST ≈ Σy², i.e. the
            # dependent variable's variation is entirely intact on this family.
            # That equality is what the second axis breaks (there SST/Σy² ~ 1e-32).
            assert model["resid_uncentered_share"] == pytest.approx(
                model["resid_var_share"], rel=0.05), (c, model["key"])
            # the reason code and the predicate cannot disagree about whether a
            # verdict exists, or the payload says one thing and the prose another
            assert (at._untestable_kind(model) == "") is model["alpha_testable"], (
                c, model["key"])
            assert at._untestable_kind(model) == "no_testable_intercept", (
                c, model["key"])
            # A VERDICT HAS MORE THAN ONE RENDERING. The t was gated and the
            # interval and the p-value beside it were not, so this scan passed for
            # a round while every fit on the grid published a 95% CI of width
            # ~1e-15 and a p-value of 0.0. Each loading's t is the same claim per
            # factor. The point estimates — alpha and the betas — stay.
            assert model["alpha_annual_ci95"] == [None, None], (c, model["key"])
            assert withheld(model["alpha_p_hac"]), (c, model["key"])
            assert withheld(model["appraisal_ratio"]), (c, model["key"])
            assert model["alpha_significant"] is None, (c, model["key"])
            assert np.isfinite(model["alpha_annual"]), (c, model["key"])
            for name, b in model["betas"].items():
                assert withheld(b["t_hac"]), (c, model["key"], name)
                assert withheld(b["t_ols"]), (c, model["key"], name)
                assert np.isfinite(b["beta"]), (c, model["key"], name)
            if _would_be_ci(model)[0] > 0.0 or _would_be_ci(model)[1] < 0.0:
                ungated_ci_excluded_zero += 1

        # how often the OLD finite-only decay gate would have fired: this grid is
        # not a corner case, and if a future change makes it one this number says so
        t_capm, t_head = capm["alpha_t_hac"], head["alpha_t_hac"]
        if (np.isfinite(t_capm) and np.isfinite(t_head)
                and abs(t_capm) >= at.T_CONVENTIONAL
                and abs(t_head) < abs(t_capm) / 2.0):
            old_decay_gate_trips += 1

        for f in findings:
            if f.check in gated:
                assert f.severity == "note", (c, f.check, f.severity, f.name)

        printed_ts = {f"{m['alpha_t_hac']:.2f}" for m in (head, capm, sp)}
        prose = [payload["interpretation"], sp["plain_english"]]
        for text in prose + [f.message for f in findings]:
            assert not t_pattern.search(text), (c, text)
        for text in prose + [f.message for f in findings if f.check in gated]:
            # The spanning beta on this grid IS c and is legitimately quoted (the
            # fit does support it), so that one phrase is masked by its exact
            # wording rather than by dropping the value from the comparison — a
            # dropped value would also mask a t that happened to equal c.
            scrubbed = text.replace(f"beta of {c:.2f}", "beta of <c>")
            leaked = printed_ts & set(two_dp.findall(scrubbed))
            assert not leaked, (c, leaked, text)

        assert sp["verdict"]["available"] is False, c
        assert sp["verdict"]["reason"] == "no_testable_intercept", c
        mt = payload["multiple_testing"]
        assert mt["alpha_testable"] is False, c
        # None, not False: False is the sentence "this alpha was tested and did
        # not survive multiple testing", which is a negative VERDICT on a fit that
        # has no test at all — the false-refutation shape this module refuses.
        assert mt["survives_bonferroni"] is None, c
        assert mt["verdict"]["available"] is False, c
        assert mt["verdict"]["reason"] == "no_testable_intercept", c
        assert withheld(mt["p_raw"]) and withheld(mt["p_bonferroni"]), c
        # the hurdle is a property of df and the attempt count, not of this fit
        assert np.isfinite(mt["t_hurdle_bonferroni"]), c
        # The refusal findings' `evidence` is serialized into the exported report
        # JSON. Both messages promise the reader that no alpha, t-statistic or R²
        # is quoted, so the payload beside them may not carry those numbers either:
        # a dict that contradicts its own message is the same defect one layer down.
        alpha_note = [f for f in findings if f.check == "factor_alpha"][0]
        span_note = [f for f in findings if f.check == "factor_spanning"][0]
        for key in ("alpha_annual", "alpha_t_hac", "r2"):
            assert alpha_note.evidence[key] is None, (c, key)
        for key in ("spanning_alpha_annual", "spanning_t", "r2"):
            assert span_note.evidence[key] is None, (c, key)
        # the diagnostics that JUSTIFY the refusal are still carried
        assert alpha_note.evidence["resid_var_share"] == head["resid_var_share"], c
        assert span_note.evidence["beta"] == sp["betas"]["bench_excess"]["beta"], c

    # measured on the committed fixture over 2016-2024: 215 of 590
    assert old_decay_gate_trips == 215, old_decay_gate_trips
    # and how often the UNGATED interval was a false claim rather than a harmless
    # one: this many of the 1,770 fits on this axis published a 95% CI that
    # EXCLUDED ZERO, on widths of ~1e-17 to ~1e-14. A gate on an input where it
    # does no work proves nothing, so the number is pinned like the one above.
    assert ungated_ci_excluded_zero == 794, ungated_ci_excluded_zero


def test_alpha_testable_needs_every_one_of_its_five_conditions():
    """Each of the five conditions driven ALONE — because on real inputs they
    overlap, and overlap is what lets a load-bearing guard be deleted.

    Measured: removing EITHER `resid_var_share <= 1.0` OR the uncentered-share
    condition from `_alpha_is_testable` leaves all 69 end-to-end tests green,
    because the reachable flat-RF constant trips both tells at once (SSR/SST =
    3507 AND SSR/Σy² = 2.6e-28), so either one alone still refuses it. A guard no
    test can show to be load-bearing is a guard that goes in the next cleanup, so
    each is pinned here on an input that ONLY it can see:
      * share 3507 with a healthy uncentered share 0.534 — only condition 3;
      * share 0.996, a legal ratio, with an uncentered share of dust — only 4.
    Neither input has to be reachable; the predicate has to be right.
    """
    eps = at.RESID_VAR_SHARE_FLOOR
    # the real all-zero-strategy-over-varying-rf fit, which IS testable
    ok = dict(alpha_t_hac=-9.54, r2=0.0036, resid_var_share=0.996,
              resid_uncentered_share=0.534)
    assert at._alpha_is_testable(**ok) is True

    def without(**over):
        return at._alpha_is_testable(**{**ok, **over})

    # 1. NaN t or NaN R² (SST exactly zero) — and, condition 5, the same
    #    non-finite t behind shares that are all perfectly legal, which is the
    #    ONE condition `_untestable_kind` could not see before round 10
    assert without(alpha_t_hac=float("nan")) is False
    assert without(r2=float("nan")) is False
    assert at._untestable_kind_of(**{**ok, "alpha_t_hac": float("nan")}) == (
        "no_intercept_standard_error")
    assert at._untestable_kind_of(**{**ok, "alpha_t_hac": float("inf")}) == (
        "no_intercept_standard_error")
    assert at._untestable_kind_of(**{**ok, "r2": float("nan")}) == (
        "no_intercept_standard_error")
    # 2. the NUMERATOR collapses — an exact factor combination
    assert without(resid_var_share=9.3e-31, resid_uncentered_share=9.2e-31) is False
    assert without(resid_var_share=eps) is False          # at the floor, not above it
    assert without(resid_var_share=eps * 1.01) is True    # just above it, testable
    # 3. the IMPOSSIBLE ratio, with a healthy uncentered share, so only the
    #    `<= 1.0` condition can refuse this one
    assert without(resid_var_share=3507.5) is False
    assert without(resid_var_share=1.0 + 1e-12) is False
    assert without(resid_var_share=1.0) is True           # the bound itself is legal
    # 4. the DENOMINATOR collapses, with the ratio inside its legal range, so only
    #    the uncentered condition can refuse this one
    assert without(resid_uncentered_share=2.6e-28) is False
    assert without(resid_uncentered_share=eps) is False
    assert without(resid_uncentered_share=eps * 1.01) is True
    assert without(resid_uncentered_share=float("nan")) is False


def test_untestable_reason_never_claims_something_its_own_numbers_contradict():
    """The refusal prose is shared by the payload, the findings and the
    interpretation, so every branch of it is checked against the numbers that
    reach that branch.

    The pre-round-6 wording had exactly one non-NaN branch, so the flat-RF case
    was described as "residual variance 3.5e+03 of total, at or below the 2.2e-16
    floor" — a sentence contradicted by the number inside it. A refusal that
    misstates WHY it is refusing is a small lie in the same paragraph as an honest
    one, and it is the sentence a reader would quote back.

    FIVE regimes, each with the claim it is allowed to make and the claims it is
    not. None of them may quote a t-statistic, and none may render a NaN.

    The fifth — a non-finite t behind perfectly legal shares — is the one the
    predicate could not name until round 10, when `_untestable_kind` was made the
    SAME condition set as `alpha_testable` instead of a three-condition mirror of
    it. Before that it answered "" (testable) for that fit, so this function fell
    through to its last branch and described a residual share of 0.5 as "residual
    variance 5.0e-01 of total, at or below the 2.2e-16 floor" — the pre-round-6
    defect, reappearing through a different door. Its `must_not` list is that
    sentence.
    """
    t_pattern = re.compile(r"\bt\s*[=>]\s*[-+]?\d")
    nan = float("nan")
    cases = {
        # SST exactly zero: no ratio exists at all
        "no_variation": (dict(resid_var_share=nan, resid_uncentered_share=nan,
                              alpha_t_hac=nan, r2=nan),
                         ["no usable t-statistic"],
                         ["UP TO A CONSTANT", "ABOVE 1", "UNCENTERED"]),
        # an exact factor combination: the NUMERATOR is dust
        "no_testable_intercept": (
            dict(resid_var_share=9.3e-31, resid_uncentered_share=9.2e-31,
                 alpha_t_hac=2.32, r2=1.0),
            ["UP TO A CONSTANT", "at or below the"],
            ["ABOVE 1", "IMPOSSIBLE", "no variation OF ITS OWN"]),
        # the measured flat-RF collapse: the ratio itself is impossible
        "no_dependent_variation": (
            dict(resid_var_share=3507.5, resid_uncentered_share=2.6e-28,
                 alpha_t_hac=-3.7e14, r2=-3506.5),
            ["no variation OF ITS OWN", "ABOVE 1", "IMPOSSIBLE"],
            ["UP TO A CONSTANT", "of total, at or below"]),
        # the other way in: a legal-looking ratio over a collapsed denominator
        "uncentered_only": (
            dict(resid_var_share=0.5, resid_uncentered_share=1e-30,
                 alpha_t_hac=3.0e14, r2=0.5),
            ["no variation OF ITS OWN", "UNCENTERED"],
            ["UP TO A CONSTANT", "ABOVE 1"]),
        # legal shares, and the t itself is not a number: se(alpha) came back 0
        "no_intercept_standard_error": (
            dict(resid_var_share=0.5, resid_uncentered_share=0.5,
                 alpha_t_hac=nan, r2=0.5),
            ["no usable t-statistic", "division by zero"],
            ["UP TO A CONSTANT", "at or below the", "ABOVE 1",
             "no variation OF ITS OWN", "5.0e-01"]),
    }
    for label, (model, must, must_not) in cases.items():
        expected_kind = "no_dependent_variation" if label == "uncentered_only" else label
        assert at._untestable_kind(model) == expected_kind, label
        why = at._untestable_reason(model, "SPY alone")
        for phrase in must:
            assert phrase in why, (label, phrase, why)
        for phrase in must_not:
            assert phrase not in why, (label, phrase, why)
        assert "SPY alone" in why or label == "no_variation", label
        assert "nan" not in why.lower(), label
        assert not t_pattern.search(why), (label, why)


# ------------------------------------- the second axis: the DENOMINATOR collapses

def flat_rf_runs(frame: pd.DataFrame,
                 min_len: int = at.MIN_DAILY_OBS) -> list[tuple[int, int, float]]:
    """Every maximal run of consecutive sessions on which the fixture's RF is
    CONSTANT and which is long enough to be regressed at all.

    Derived from the committed file rather than hard-coded: this is the property
    that makes the defect reachable from the product (a strategy whose filter
    never qualifies returns 0.0 every day, so excess = -RF is a constant), and if
    a future fixture removes the flat runs this returns fewer of them and the test
    that consumes it says so instead of passing vacuously.
    """
    rf = frame["rf"].to_numpy(dtype=float)
    runs: list[tuple[int, int, float]] = []
    start = 0
    for i in range(1, len(rf) + 1):
        if i == len(rf) or rf[i] != rf[start]:
            if i - start >= min_len:
                runs.append((start, i, float(rf[start])))
            start = i
    return runs


def test_the_fixture_really_does_contain_long_flat_rf_runs(factors):
    """The reachability premise of the two tests below, asserted on its own.

    Four runs of at least a year, the longest 793 sessions (2022-12-01 to
    2026-01-30) at RF = 0.0002/day. Two of them are RF = 0 exactly — there
    excess ≡ 0, SST is exactly 0 and the OLDER NaN guard refuses the whole
    report — and two are NONZERO, which is the regime no guard covered.
    """
    runs = flat_rf_runs(factors.frame)
    assert len(runs) == 4, [(b - a, v) for a, b, v in runs]
    lengths = sorted((b - a) for a, b, v in runs)
    assert lengths == [565, 587, 755, 793], lengths
    nonzero = [(b - a, v, str(factors.frame.index[a].date())) for a, b, v in runs
               if v != 0.0]
    assert sorted(nonzero) == [(565, 0.0001, "2018-01-02"),
                              (793, 0.0002, "2022-12-01")], nonzero


def test_all_zero_strategy_over_a_flat_rf_run_refuses_every_verdict(factors):
    """THE REACHABLE DEFECT, reproduced from the committed fixture alone.

    The engine returns 0.0 — not RF — on a day when nothing qualifies, so a
    strategy whose filter never fires is the all-zero series. Over a flat-RF run
    its excess return is a nonzero CONSTANT: y has no variation of its own, SST is
    rounding dust, and SSR/SST comes out 3507 with R² -3506. Every guard in this
    module keyed on that ratio being SMALL, so `alpha_testable` was True and the
    reader was shown, on the live tree before this round:

        WARNING  Alpha is significantly NEGATIVE
                 "alpha is -5.0%/yr with a HAC t-statistic of -372352520723111.69
                  — significantly BELOW zero"
        WARNING  Beaten by SPY buy-and-hold on a risk-adjusted basis
                 "intercept is -5.0%/yr (t = -383213209391422.88) with beta 0.00
                  and R² -335126%"
        PASS     Factor regression diagnostics
                 "the regression is well-posed and the loadings are interpretable"

    Three confident findings, all fabricated, one of them a PASS. The t-statistics
    are not asserted as literals — the check is the module's OWN print format of
    whatever this fit produces, so it holds for every constant, not just this one.
    """
    runs = [r for r in flat_rf_runs(factors.frame) if r[2] != 0.0]
    assert runs, "the fixture no longer makes this defect reachable"
    t_pattern = re.compile(r"\bt\s*[=>]\s*[-+]?\d")

    for start, stop, rf_value in runs:
        idx = factors.frame.index[start:stop]
        label = f"{idx[0].date()}..{idx[-1].date()} rf={rf_value}"
        strat = pd.Series(0.0, index=idx)          # never traded: the engine's 0.0
        bench = factors.frame["rf"].loc[idx] + factors.frame["mkt_rf"].loc[idx]
        payload, findings = at.attribution_report(
            strat, bench_returns=bench, bench_label="SPY", attempts=1)

        head = payload["models"]["ff6"]
        sp = payload["spanning"]
        # the regime is real and is NOT the one the first grid covers: the ratio is
        # ABOVE 1 and the R² is NEGATIVE, each arithmetically impossible for OLS
        # with an intercept, while the uncentered share is dust
        assert head["resid_var_share"] > 1.0, label
        assert head["r2"] < 0.0, label
        assert head["resid_uncentered_share"] <= at.RESID_VAR_SHARE_FLOOR, label
        assert np.isfinite(head["alpha_t_hac"]), label     # finite and plausible
        assert abs(head["alpha_t_hac"]) > 1e6, label       # and pure dust over dust

        # every verdict is refused, and the reason names the denominator
        assert head["alpha_testable"] is False, label
        assert sp["alpha_testable"] is False, label
        # in EVERY rendering, not only as a t: the 95% interval and the p-value
        # this fit used to publish were [-5.0000000000%, -5.0000000000%] and 0.0
        for model in (head, payload["models"]["capm"], sp):
            assert model["alpha_annual_ci95"] == [None, None], (label, model["key"])
            assert withheld(model["alpha_p_hac"]), (label, model["key"])
            assert withheld(model["appraisal_ratio"]), (label, model["key"])
            assert all(withheld(b["t_hac"])
                       for b in model["betas"].values()), (label, model["key"])
        assert at._untestable_kind(head) == "no_dependent_variation", label
        assert sp["verdict"]["available"] is False, label
        assert sp["verdict"]["reason"] == "no_dependent_variation", label

        checks = {f.check: f for f in findings}
        assert checks["factor_alpha"].severity == "note", label
        assert "could not be tested" in checks["factor_alpha"].name.lower(), label
        assert checks["factor_spanning"].severity == "note", label
        assert "could not be tested" in checks["factor_spanning"].name, label
        # the PASS that said "well-posed" is now the impossible-R² caveat
        rel = checks["factor_reliability"]
        assert rel.severity == "warning", label
        assert "well-posed" not in rel.message, label
        assert any("IMPOSSIBLE" in i for i in payload["diagnostics"]["issues"]), label
        assert payload["diagnostics"]["issues"], label

        # multiple testing publishes a MISSING result, not a negative one
        mt = payload["multiple_testing"]
        assert mt["survives_bonferroni"] is None, label
        assert mt["verdict"]["reason"] == "no_dependent_variation", label
        assert withheld(mt["p_raw"]), label

        # and no fabricated number reaches a reader, in prose or in the exported
        # evidence: the forbidden strings are this fit's own numbers as this module
        # would print them, not literals copied from the bug report. The appraisal
        # ratio is now WITHHELD from the payload, so the string to forbid is the
        # one it WOULD have published, rebuilt from the two numbers that stay.
        printed = {f"{head['alpha_t_hac']:.2f}", f"{sp['alpha_t_hac']:.2f}",
                   f"{head['alpha_annual']:+.1%}", f"{sp['alpha_annual']:+.1%}",
                   f"{sp['r2']:.0%}", f"{_would_be_appraisal(head):.2f}"}
        prose = [payload["interpretation"], sp["plain_english"]] + [
            f.message for f in findings]
        for text in prose:
            assert not t_pattern.search(text), (label, text)
            assert "nan" not in text.lower(), (label, text)
            for bad in printed:
                assert bad not in text, (label, bad, text)
        assert "NOT TESTABLE" in payload["interpretation"], label
        for text in prose:
            assert "significantly BELOW zero" not in text, (label, text)
            assert "no risk-adjusted value added" not in text, (label, text)
        for key in ("alpha_annual", "alpha_t_hac", "r2"):
            assert checks["factor_alpha"].evidence[key] is None, (label, key)
        for key in ("spanning_alpha_annual", "spanning_t", "r2"):
            assert checks["factor_spanning"].evidence[key] is None, (label, key)


def test_no_degenerate_fit_reaches_the_reader_on_the_constant_excess_axis(window):
    """The second axis swept, the way DEGENERATE_GRID sweeps the first.

    41 constant excess returns from -10%/yr to +10%/yr. Every one of them has a
    dependent variable with no variation of its own, so every one must refuse —
    either at the report level (the older NaN path, when the constant divides
    exactly and SST comes out 0) or verdict by verdict. Nothing in between is
    acceptable, and nothing may quote a t.
    """
    bench = window["rf"] + window["mkt_rf"]
    gated = ("factor_alpha", "factor_alpha_decay", "factor_spanning")
    t_pattern = re.compile(r"\bt\s*[=>]\s*[-+]?\d")
    refused_report = collapsed = 0

    for c in DENOMINATOR_COLLAPSE_GRID:
        strat = window["rf"] + c / 252.0
        payload, findings = at.attribution_report(
            strat, bench_returns=bench, bench_label="SPY", attempts=1)

        if not payload["available"]:
            # the NaN path: SST exactly 0, the whole report refuses
            assert payload["reason"] == "degenerate_regression", c
            assert not [f for f in findings if f.check in gated], c
            refused_report += 1
        else:
            head = payload["models"]["ff6"]
            sp = payload["spanning"]
            assert head["resid_var_share"] > 1.0, c
            assert head["r2"] < 0.0, c
            assert head["resid_uncentered_share"] <= at.RESID_VAR_SHARE_FLOOR, c
            for model in (head, payload["models"]["capm"], sp):
                assert model["alpha_testable"] is False, (c, model["key"])
                assert (at._untestable_kind(model) == ""
                        ) is model["alpha_testable"], (c, model["key"])
                assert at._untestable_kind(model) == "no_dependent_variation", (
                    c, model["key"])
                # the interval and the p-value are the same verdict in two other
                # forms; on THIS axis the ungated interval excluded zero for every
                # single fit (111 of 111, widths 2.8e-17 to 1.5e-15), because a
                # constant excess return is exactly what pins α to its own dust
                assert model["alpha_annual_ci95"] == [None, None], (c, model["key"])
                assert withheld(model["alpha_p_hac"]), (c, model["key"])
                assert withheld(model["appraisal_ratio"]), (c, model["key"])
                lo, hi, _w = _would_be_ci(model)
                assert lo > 0.0 or hi < 0.0, (c, model["key"], lo, hi)
                assert all(withheld(b["t_hac"])
                           for b in model["betas"].values()), (c, model["key"])
            assert payload["multiple_testing"]["survives_bonferroni"] is None, c
            assert any("IMPOSSIBLE" in i
                       for i in payload["diagnostics"]["issues"]), c
            for f in findings:
                if f.check in gated:
                    assert f.severity == "note", (c, f.check, f.severity)
            collapsed += 1

        for text in [f.message for f in findings] + [payload.get("interpretation", "")]:
            assert not t_pattern.search(text), (c, text)
            assert "nan" not in text.lower(), (c, text)

    # measured on the committed fixture over 2016-2024: 37 constants collapse the
    # denominator, 4 (including c = 0) divide exactly into SST = 0 and take the
    # older NaN path. Both counts are asserted so the axis cannot go vacuous by
    # sliding entirely into one branch.
    assert (collapsed, refused_report) == (37, 4), (collapsed, refused_report)


def test_degenerate_grid_has_two_axes_and_they_are_disjoint(window):
    """The two sweeps must live in DIFFERENT regimes, or the second is decoration.

    Axis 1 (`rf + c*mkt_rf`) drives the NUMERATOR to dust: SSR/SST ≈ 0, R² = 1,
    SST ≈ Σy². Axis 2 (`rf + k/252`) drives the DENOMINATOR to dust: SSR/SST > 1,
    R² < 0, SST/Σy² ~ 1e-32. A guard that catches one is not thereby a guard that
    catches the other — that is the whole lesson of this round — so the test suite
    has to be able to tell them apart.
    """
    cols = list(at.FACTOR_COLUMNS)

    def fit(strat: pd.Series) -> at.OLSFit:
        excess = (strat - window["rf"]).to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(excess))]
                            + [window[c].to_numpy(dtype=float) for c in cols])
        return at.ols_hac(excess, X, at.HAC_LAGS_DAILY)

    numerator = fit(window["rf"] + 1.2 * window["mkt_rf"])
    denominator = fit(window["rf"] + 0.05 / 252.0)

    # axis 1: numerator dust, denominator healthy, R² pinned to 1
    assert numerator.resid_var_share <= at.RESID_VAR_SHARE_FLOOR
    assert numerator.resid_var_share <= 1.0
    assert numerator.r2 == 1.0
    assert numerator.resid_uncentered_share == pytest.approx(
        numerator.resid_var_share, rel=0.05)

    # axis 2: the ratio is impossible, the uncentered share is dust
    assert denominator.resid_var_share > 1.0
    assert denominator.r2 < 0.0
    assert denominator.resid_uncentered_share <= at.RESID_VAR_SHARE_FLOOR
    # and the two denominators differ by ~30 orders of magnitude, which IS the
    # collapse: the ratio of the two shares is Σy²/SST, measured 3.2e+31 here
    # (i.e. SST/Σy² = 3.2e-32) against ~1.00 on axis 1
    assert (denominator.resid_var_share / denominator.resid_uncentered_share
            > 1e25)
    # the OLD guard, on its own, would have passed the second axis as testable
    assert not (denominator.resid_var_share <= at.RESID_VAR_SHARE_FLOOR)


def test_the_denominator_guard_does_not_refuse_a_real_answer(factors, window):
    """The over-correction check: the new conditions must not swallow a fit that
    has a real sampling distribution.

    Three fits that MUST stay testable, all measured on the committed fixture:
      * the SAME all-zero strategy over 2016-2024, where RF VARIES — excess = -RF
        genuinely moves, SSR/SST = 0.996 and the uncentered share 0.534, and the
        honest answer is a real t of about -9.5. If the guard keyed on "excess is
        just -RF" or on the share being near 1 it would refuse this, which would be
        a false refusal of a measurable result;
      * a genuine residual of sd 1e-9 — seven orders of magnitude below the daily
        signal, far tighter than any real strategy — at 33x the floor;
      * an ordinary noisy strategy, share 0.15.
    """
    # 1. all-zero over a VARYING-rf window: legitimately testable
    strat = pd.Series(0.0, index=window.index)
    assert window["rf"].nunique() > 1                    # rf really does vary here
    payload, findings = at.attribution_report(strat, attempts=1)
    assert payload["available"] is True
    head = payload["models"]["ff6"]
    assert head["resid_var_share"] == pytest.approx(0.996, abs=0.01)
    assert head["resid_var_share"] <= 1.0 and head["r2"] >= 0.0
    assert head["resid_uncentered_share"] == pytest.approx(0.534, abs=0.01)
    assert head["alpha_testable"] is True
    assert head["alpha_t_hac"] == pytest.approx(-9.54, abs=0.05)
    assert at._untestable_kind(head) == ""
    alpha_f = [f for f in findings if f.check == "factor_alpha"][0]
    assert alpha_f.severity == "warning"                  # a real verdict, not a note
    assert "could not be tested" not in alpha_f.name.lower()
    assert payload["multiple_testing"]["survives_bonferroni"] is True
    assert payload["multiple_testing"]["verdict"]["available"] is True
    assert not any("IMPOSSIBLE" in i for i in payload["diagnostics"]["issues"])

    # 2. a genuinely tight but non-degenerate fit stays testable
    rng = np.random.default_rng(11)
    tight = (window["rf"] + window["mkt_rf"]
             + pd.Series(rng.normal(0.0, 1e-9, len(window)), index=window.index))
    tight_head = at.attribution_report(tight, attempts=1)[0]["models"]["ff6"]
    assert tight_head["resid_var_share"] / at.RESID_VAR_SHARE_FLOOR > 30.0
    assert tight_head["resid_uncentered_share"] > at.RESID_VAR_SHARE_FLOOR
    assert tight_head["alpha_testable"] is True

    # 3. and the ordinary case is untouched
    ordinary = (window["rf"] + window["mkt_rf"]
                + pd.Series(rng.normal(0.0, 0.005, len(window)), index=window.index))
    ord_head = at.attribution_report(ordinary, attempts=1)[0]["models"]["ff6"]
    assert ord_head["resid_var_share"] == pytest.approx(0.154, abs=0.01)
    assert ord_head["alpha_testable"] is True


# --------------------------- ONE predicate, asked at EVERY site that renders

def _condition_four_only(window, sd: float = 1e-16) -> pd.Series:
    """A fit the predicate refuses ONLY through its uncentered-share condition.

    A near-constant excess return — a 2e-4/day level plus noise of sd 1e-16, a
    coefficient of variation of 5e-13 — leaves SSR/SST a perfectly LEGAL ratio,
    measured 0.9998 with R² +0.00024, while SSR/Σy² is 2.5e-26, dust. Every
    degeneracy test that reads the RATIO says "fine"; only the denominator test
    sees it.

    NOT reachable from the product (no real price series has a CV that small);
    this is the regime the ratio-shaped derivation could not see, and the point is
    the derivation, not the input. The REACHABLE half of the same policy is
    `test_all_zero_strategy_over_a_flat_rf_run_refuses_every_verdict`.
    """
    rng = np.random.default_rng(7)
    return (window["rf"] + 0.0002
            + pd.Series(rng.normal(0.0, sd, len(window)), index=window.index))


def test_every_renderer_asks_the_predicate_not_its_own_re_derivation(window, factors):
    """THE POLICY, driven on all three degenerate regimes at once.

    Six rounds each fixed a named degeneracy case and left a sibling, because
    each site RE-DERIVED degeneracy from whatever quantity was in front of it.
    The reliability block derived it from the RATIO (`r2 < 0.0 or
    resid_var_share > 1.0`), so a fit refused ONLY by the predicate's fourth
    condition — the UNCENTERED share — had a legal ratio, fired no issue, left
    `issues` EMPTY, and reached the reader as

        [note]  factor_alpha: Alpha could not be tested
        [pass]  Factor regression diagnostics
                "... the regression is well-posed and the loadings are
                 interpretable"

    in the SAME findings list, on a fit whose t is 3.0e14 and whose residual sd
    is 3.1e-17. The entry point refused while the renderer fabricated.

    So the assertion here is not "case X is fixed" — it is that every site which
    renders a claim agrees with `_untestable_kind`, on all three regimes that
    reach it:
      * the NUMERATOR collapses (an exact factor combination);
      * the ratio is ARITHMETICALLY IMPOSSIBLE (the reachable flat-RF constant);
      * the DENOMINATOR collapses behind a legal ratio (condition 4 alone).
    """
    t_pattern = re.compile(r"\bt\s*[=>]\s*[-+]?\d")
    # the LONGEST nonzero flat-RF run (793 sessions), the one long enough for the
    # monthly pass to run at all
    a, b, _rf = [r for r in flat_rf_runs(factors.frame) if r[2] != 0.0][-1]
    flat_idx = factors.frame.index[a:b]

    cases = {
        "numerator": (window["rf"] + 1.2 * window["mkt_rf"],
                      "no_testable_intercept"),
        "impossible_ratio": (pd.Series(0.0, index=flat_idx),
                             "no_dependent_variation"),
        "denominator": (_condition_four_only(window), "no_dependent_variation"),
    }
    for label, (strat, expected_kind) in cases.items():
        payload, findings = at.attribution_report(strat, attempts=1)
        assert payload["available"] is True, label
        head = payload["models"]["ff6"]

        # the regime really is the one this case names
        assert head["alpha_testable"] is False, label
        assert at._untestable_kind(head) == expected_kind, label

        # EVERY site agrees with the predicate.
        issues = payload["diagnostics"]["issues"]
        assert any("NO TESTABLE INTERCEPT" in i for i in issues), (label, issues)
        rel = [f for f in findings if f.check == "factor_reliability"][0]
        assert rel.severity == "warning", label
        assert "well-posed" not in rel.message, label
        assert payload["multiple_testing"]["survives_bonferroni"] is None, label
        assert payload["multiple_testing"]["verdict"]["reason"] == expected_kind, label
        rd = payload["risk_decomposition"]
        assert rd["available"] is False, label
        assert rd["reason"] == expected_kind, label
        assert rd["factor_share"] is None and rd["specific_share"] is None, label
        assert withheld(head["resid_autocorr1"]), label
        assert at._rho_txt(head["resid_autocorr1"]) == "n/a", label
        # `regress` is a renderer too, and it publishes the verdict three ways:
        # as a t, as an interval and as a probability. All three, plus every
        # loading's t, or the entry refuses while the row fabricates.
        assert head["alpha_annual_ci95"] == [None, None], label
        assert withheld(head["alpha_p_hac"]), label
        assert withheld(head["appraisal_ratio"]), label
        assert all(withheld(b["t_hac"]) and withheld(b["t_ols"])
                   for b in head["betas"].values()), label
        # the point estimates the fit DOES support are untouched
        assert np.isfinite(head["alpha_annual"]), label
        assert all(np.isfinite(b["beta"]) for b in head["betas"].values()), label

        # and the same reason is named everywhere, with no fabricated t in sight
        for text in [f.message for f in findings] + [payload["interpretation"]]:
            assert not t_pattern.search(text), (label, text)
            assert "nan" not in text.lower(), (label, text)


def test_the_reliability_block_has_no_ratio_shaped_hole_left(window):
    """The condition-4-only fit, pinned against the derivation that missed it.

    This is the input the old branch could not see, stated as the arithmetic
    rather than as a story: its R² is POSITIVE, its residual share is BELOW 1 and
    BELOW the factor-portfolio bar, so `r2 < 0.0`, `resid_var_share > 1.0` and
    `r2 > R2_FACTOR_PORTFOLIO` are all False — and it is still dust. If the
    reliability block ever re-derives degeneracy from the ratio again, the three
    assertions below stay true and the fourth fails.
    """
    payload, findings = at.attribution_report(_condition_four_only(window),
                                              attempts=1)
    head = payload["models"]["ff6"]

    # none of the ratio-shaped tells fire on this fit
    assert head["r2"] > 0.0
    assert head["resid_var_share"] <= 1.0
    assert head["r2"] < at.R2_FACTOR_PORTFOLIO
    # but the denominator has collapsed and the intercept is dust over dust
    assert head["resid_uncentered_share"] <= at.RESID_VAR_SHARE_FLOOR
    assert abs(head["alpha_t_hac"]) > 1e6
    assert head["resid_vol_annual"] / math.sqrt(at.TRADING_DAYS) < 1e-12

    # the renderer refuses anyway, because it asks the predicate
    assert payload["diagnostics"]["issues"]
    rel = [f for f in findings if f.check == "factor_reliability"][0]
    assert rel.severity == "warning"
    assert "well-posed" not in rel.message
    assert "UNCENTERED" in " ".join(payload["diagnostics"]["issues"])


def test_both_disjuncts_of_the_shared_predicate_are_pinned_on_their_own():
    """HYGIENE, by this module's own standard: `_untestable_kind` is now the ONE
    thing every renderer asks, so each of its disjuncts has to be shown to be
    load-bearing — an unpinned branch is exactly what the reliability block's
    `r2 < 0.0` turned out to be (redundant with `share > 1.0` on every input ever
    shown to it, and therefore free to be deleted or to rot unnoticed).

    The two disjuncts overlap on every REACHABLE input — the flat-RF constant
    trips both at once — so each is driven here on an input only it can see.
    """
    # only the algebraic bound can see this one: the ratio is impossible while
    # the uncentered share is healthy
    ratio_only = dict(resid_var_share=3507.5, resid_uncentered_share=0.534,
                      alpha_t_hac=-3.7e14, r2=-3506.5)
    assert at._untestable_kind(ratio_only) == "no_dependent_variation"
    assert at._alpha_is_testable(**ratio_only) is False
    assert "ABOVE 1" in at._untestable_reason(ratio_only)

    # only the denominator test can see this one: the ratio is perfectly legal
    unc_only = dict(resid_var_share=0.9998, resid_uncentered_share=2.5e-26,
                    alpha_t_hac=3.0e14, r2=0.00024)
    assert at._untestable_kind(unc_only) == "no_dependent_variation"
    assert at._alpha_is_testable(**unc_only) is False
    assert "UNCENTERED" in at._untestable_reason(unc_only)

    # and neither fires on an ordinary fit, nor on the bound itself
    assert at._untestable_kind(
        dict(resid_var_share=0.15, resid_uncentered_share=0.10,
             alpha_t_hac=1.4, r2=0.85)) == ""
    assert at._untestable_kind(
        dict(resid_var_share=1.0, resid_uncentered_share=0.10,
             alpha_t_hac=1.4, r2=0.0)) == ""


def test_residual_autocorrelation_is_gated_by_the_same_dispersion_test(window):
    """`_lag1_autocorr` refuses only an EXACTLY constant residual (sd == 0), so on
    a degenerate fit it hands back a tidy lag-1 correlation of rounding dust —
    +0.01 on residuals of sd 3.1e-17 — which the diagnostics block compares
    against a bar and the reliability PASS prints verbatim.

    The check is not "it is NaN": it is that the SAME residuals still produce a
    finite number when the predicate is not consulted, so the NaN in the payload
    is the gate's doing and not an accident of the data.
    """
    strat = _condition_four_only(window)
    payload, _ = at.attribution_report(strat, attempts=1)
    head = payload["models"]["ff6"]

    cols = at.MODEL_LADDER[at.HEADLINE_MODEL][1]
    X = np.column_stack([np.ones(len(window))]
                        + [window[c].to_numpy(dtype=float) for c in cols])
    y = (strat - window["rf"]).to_numpy(dtype=float)
    fit = at.ols_hac(y, X, at.HAC_LAGS_DAILY)
    assert fit.resid.std() > 0.0                       # not exactly constant
    assert np.isfinite(at._lag1_autocorr(fit.resid))   # ungated, it IS a number
    assert abs(at._lag1_autocorr(fit.resid)) < at.RESID_AUTOCORR_BAR

    # gated, it is not a number, and nothing renders one
    assert head["alpha_testable"] is False
    assert withheld(head["resid_autocorr1"])
    assert withheld(payload["diagnostics"]["resid_autocorr1"])
    assert not any("autocorrelation is" in i
                   for i in payload["diagnostics"]["issues"])

    # an ordinary fit still reports it
    rng = np.random.default_rng(3)
    ordinary = (window["rf"] + window["mkt_rf"]
                + pd.Series(rng.normal(0.0, 0.004, len(window)), index=window.index))
    ord_head = at.attribution_report(ordinary, attempts=1)[0]["models"]["ff6"]
    assert ord_head["alpha_testable"] is True
    assert np.isfinite(ord_head["resid_autocorr1"])


def _would_be_ci(model: dict) -> tuple[float, float, float]:
    """The interval this fit WOULD have published before the gate, rebuilt from
    the numbers the payload still carries: α ± t_{0.975,df} × se(α) × P.

    Rebuilt rather than remembered, so the counterfactual is this fit's own
    arithmetic and not a literal copied from the bug report. Returns (lo, hi,
    width).
    """
    half = (model["alpha_ci95_t_critical"] * model["alpha_se_hac"]
            * model["annualization"])
    return (model["alpha_annual"] - half, model["alpha_annual"] + half, 2 * half)


def _would_be_appraisal(model: dict) -> float:
    """The appraisal ratio this fit WOULD have published before the chokepoint:
    α_annual / σ_resid,annual, rebuilt from the two numbers the payload still
    carries.

    Same discipline as `_would_be_ci`: the counterfactual is this fit's own
    arithmetic, so the forbidden string is whatever THIS input produces rather
    than a literal from a bug report — and if a future change made the ratio
    harmless (a real residual), the assertions built on it would be visibly
    checking a real number instead of silently checking nothing.
    """
    return model["alpha_annual"] / model["resid_vol_annual"]


def test_every_rendering_of_the_alpha_verdict_is_gated_not_just_the_t(factors, window):
    """PRINCIPLE 2 at the site that FORMS the numbers: a verdict has more than one
    RENDERING, and gating one of them is gating none.

    `regress` asked the predicate for alpha's t and for the residual
    autocorrelation, and then three lines later published the SAME significance
    verdict in INTERVAL form and in PROBABILITY form without asking. Measured on
    the live tree before this round:

      (1) a constant 0.1%/day strategy over the fixture's 793-session flat-RF run
          published `models.ff6.alpha_annual_ci95 = [+20.1600000000%,
          +20.1600000000%]` — width 2.1e-15 — with `alpha_p_hac = 0.0` and
          `alpha_testable` False, in the same payload whose interpretation reads
          "alpha is NOT TESTABLE" and whose factor_alpha note promises that "no
          alpha, t-statistic or R² is quoted";
      (2) the exact benchmark clone published `spanning.alpha_annual_ci95 =
          [+5.000000000000%, +5.000000000000%]`, se 5.0e-19, `alpha_p_hac = 0`,
          beside the factor_spanning note whose evidence sets `withheld =
          [spanning_alpha_annual, spanning_t, r2]`.

    A zero-width 95% interval that EXCLUDES ZERO is a stronger claim than the
    t-statistic the module refuses to print: it says the constant is pinned to
    1e-15 of a percent. So the assertions here are:

      * the point ESTIMATE survives — (0.001 − 0.0002) × 252 = 20.16%/yr is what
        the constant IS, and suppressing it would be a fabricated absence;
      * every PRECISION claim formed from se(α) is withheld: the interval, the
        p-value, and each loading's t;
      * the withheld interval WOULD have excluded zero, rebuilt from this fit's
        own published se — otherwise the gate could be passing on an input where
        it does no work;
      * `alpha_ci95_t_critical` is still published, deliberately: a Student-t
        quantile of df is not a claim about this fit, the same reason
        `t_hurdle_bonferroni` survives a refused multiple-testing verdict. Pinned
        so the decision cannot drift into an unnoticed one either way.
    """
    # ---- case (1): the reachable one, from the committed fixture alone
    a, b, rf_value = [r for r in flat_rf_runs(factors.frame) if r[2] != 0.0][-1]
    idx = factors.frame.index[a:b]
    assert len(idx) == 793 and rf_value == 0.0002
    payload, findings = at.attribution_report(pd.Series(0.001, index=idx),
                                              attempts=1)
    head = payload["models"]["ff6"]

    assert head["alpha_testable"] is False
    # the estimate is honest arithmetic and stays
    assert head["alpha_annual"] == pytest.approx((0.001 - rf_value) * 252, abs=1e-12)
    # every precision claim formed from se(alpha) is withheld
    assert head["alpha_annual_ci95"] == [None, None]
    assert withheld(head["alpha_p_hac"])
    assert all(withheld(bt["t_hac"]) and withheld(bt["t_ols"])
               for bt in head["betas"].values())
    # ...while the loadings themselves, which the fit does support, are published
    assert all(np.isfinite(bt["beta"]) for bt in head["betas"].values())
    # and the gate is doing real work here: ungated, this interval excluded zero
    lo, hi, width = _would_be_ci(head)
    assert lo > 0.0 and width < 1e-10, (lo, hi, width)
    # the deliberate exception: a quantile of df, not a claim about this fit
    assert head["alpha_ci95_t_critical"] == pytest.approx(
        at.t_hurdle_for_p(0.05, head["df_resid"]), rel=1e-12)
    # the payload no longer contradicts the prose beside it
    assert "NOT TESTABLE" in payload["interpretation"]
    note = [f for f in findings if f.check == "factor_alpha"][0]
    assert "t-statistic" in note.message and note.severity == "note"
    assert payload["multiple_testing"]["p_raw"] is None or not np.isfinite(
        payload["multiple_testing"]["p_raw"])

    # ---- case (2): the exact benchmark clone carrying a REAL +5%/yr
    bench = window["rf"] + window["mkt_rf"]
    clone = window["rf"] + 1.0 * (bench - window["rf"]) + 0.05 / 252.0
    payload, findings = at.attribution_report(clone, bench_returns=bench,
                                              bench_label="SPY", attempts=1)
    sp = payload["spanning"]
    assert sp["alpha_testable"] is False
    assert sp["alpha_annual"] == pytest.approx(0.05, abs=1e-9)     # the estimate
    assert sp["alpha_annual_ci95"] == [None, None]                 # the precision
    assert withheld(sp["alpha_p_hac"])
    beta = sp["betas"]["bench_excess"]
    assert beta["beta"] == pytest.approx(1.0, abs=1e-9)            # a real loading
    assert withheld(beta["t_hac"]) and withheld(beta["t_ols"])
    lo, hi, width = _would_be_ci(sp)
    assert lo > 0.0 and width < 1e-10, (lo, hi, width)
    span_note = [f for f in findings if f.check == "factor_spanning"][0]
    assert span_note.evidence["withheld"] == ["spanning_alpha_annual",
                                              "spanning_t", "r2"]
    # every model row in the whole payload agrees: no withheld-interval row
    # publishes a p, and no published-interval row is untestable
    rows = [*payload["models"].values(), sp,
            *payload["monthly"]["models"].values()]
    for m in rows:
        if m["alpha_testable"]:
            assert all(v is not None and np.isfinite(v)
                       for v in m["alpha_annual_ci95"]), m["key"]
            assert np.isfinite(m["alpha_p_hac"]), m["key"]
        else:
            assert m["alpha_annual_ci95"] == [None, None], m["key"]
            assert withheld(m["alpha_p_hac"]), m["key"]

    # ---- the CONTROL: an ordinary fit must still publish all three
    rng = np.random.default_rng(11)
    good = (window["rf"] + 0.9 * window["mkt_rf"] + 0.0004
            + pd.Series(rng.normal(0.0, 0.004, len(window)), index=window.index))
    payload, _ = at.attribution_report(good, bench_returns=bench,
                                       bench_label="SPY", attempts=1)
    head = payload["models"]["ff6"]
    assert head["alpha_testable"] is True
    g_lo, g_hi = head["alpha_annual_ci95"]
    assert g_lo < head["alpha_annual"] < g_hi
    # and it is the exact Student-t interval, not a widened or degenerate one
    assert (g_hi - g_lo) / 2.0 == pytest.approx(_would_be_ci(head)[2] / 2.0,
                                                rel=1e-12)
    assert 0.0 < head["alpha_p_hac"] < 1.0
    assert all(np.isfinite(bt["t_hac"]) and np.isfinite(bt["t_ols"])
               for bt in head["betas"].values())
    # THE IDENTITY between the three renderings, asserted where all three are
    # published: an interval excludes zero exactly when |t| exceeds the critical
    # value the interval was built from, and the two-sided p crosses 5% at the
    # same point. This is what "the same claim three ways" means arithmetically,
    # and it is what would break if one of the three were ever formed from the
    # OLS standard error while the others use HAC.
    excludes_zero = g_lo > 0.0 or g_hi < 0.0
    assert excludes_zero is (
        abs(head["alpha_t_hac"]) > head["alpha_ci95_t_critical"])
    assert excludes_zero is (head["alpha_p_hac"] < 0.05)
    assert head["alpha_significant"] is (abs(head["alpha_t_hac"]) >= 2.0)
    # and the p-value is the HAC one its key NAME promises, not the OLS one
    # sitting beside it: both clear the bar on this fit, so the identity above
    # cannot tell them apart and the value has to be pinned directly.
    assert head["alpha_p_hac"] == pytest.approx(
        at.t_two_sided_p(head["alpha_t_hac"], head["df_resid"]), rel=1e-12)
    assert head["alpha_p_hac"] != pytest.approx(
        at.t_two_sided_p(head["alpha_t_ols"], head["df_resid"]), rel=1e-6)
    assert payload["spanning"]["alpha_testable"] is True
    assert all(v is not None for v in payload["spanning"]["alpha_annual_ci95"])


def test_the_predicate_has_one_condition_set_not_two_spellings():
    """The docstring claimed ONE degeneracy predicate; there were TWO functions
    over DIFFERENT condition sets, and that is the same defect class as an ungated
    verdict — a false claim about which guard protects what.

    `_alpha_is_testable` checked five conditions (finite t, finite R², the share
    floor, the algebraic bound, the uncentered floor). `_untestable_kind` — the
    one every RENDERER asks, and the one that names the reason in the payload, the
    prose and the refusal evidence — checked three of them. So a fit with a
    healthy residual share and a non-finite t answered `alpha_testable = False` at
    the entry and `kind = ""` — testable — at every renderer, and
    `_untestable_reason` described it as "residual variance 5.0e-01 of total, at
    or below the 2.2e-16 floor", a sentence contradicted by its own number.
    Measured on the live tree before this round, with `risk_decomposition`
    publishing `available: True` on that same row.

    The property, not the instance: over the cross product of interesting values
    for the four quantities the predicate reads, the BOOL and the NAME must be the
    same answer, and the dict form must agree with the scalar form. 1,512 rows.
    """
    nan, inf = float("nan"), float("inf")
    eps = at.RESID_VAR_SHARE_FLOOR
    ts = [0.0, 1.4, -9.54, 3.0e14, nan, inf]
    r2s = [0.0, 0.35, 1.0, -3506.5, nan, -inf]
    shares = [0.15, 0.996, 1.0, 1.0 + 1e-12, 3507.5, eps, nan]
    uncs = [0.10, 0.534, eps, eps * 1.01, 1e-30, nan]

    assert len(ts) * len(r2s) * len(shares) * len(uncs) == 1512
    seen: set[str] = set()
    for t in ts:
        for r2 in r2s:
            for share in shares:
                for unc in uncs:
                    kind = at._untestable_kind_of(t, r2, share, unc)
                    row = dict(alpha_t_hac=t, r2=r2, resid_var_share=share,
                               resid_uncentered_share=unc)
                    assert at._untestable_kind(row) == kind, row
                    assert at._alpha_is_testable(**row) is (kind == ""), row
                    seen.add(kind)
                    if kind:
                        why = at._untestable_reason(row, "the factors")
                        assert "nan" not in why.lower(), row
                        # the branch that actually fired has to describe the
                        # numbers that got it there: only the centered-floor case
                        # may say the residual share "of total" is at or below the
                        # floor, because only there is it true
                        if kind != "no_testable_intercept":
                            assert "of total, at or below" not in why, (kind, row)

    # the cross product really does reach every branch, including the one the two
    # spellings used to disagree about — otherwise this is 1,296 vacuous rows
    assert seen == {"", "no_variation", "no_testable_intercept",
                    "no_dependent_variation", "no_intercept_standard_error"}, seen
    # and the disagreement itself, stated as the single row it was
    split = dict(alpha_t_hac=nan, r2=0.5, resid_var_share=0.5,
                 resid_uncentered_share=0.5)
    assert at._alpha_is_testable(**split) is False
    assert at._untestable_kind(split) == "no_intercept_standard_error"
    assert "division by zero" in at._untestable_reason(split)
    assert "of total, at or below" not in at._untestable_reason(split)
    assert "5.0e-01" not in at._untestable_reason(split)
    # every renderer that reads the NAME now refuses it too
    rd = at.risk_decomposition(
        {**split, "factors": ["mkt_rf"], "betas": {"mkt_rf": {"beta": 0.9}},
         "resid_vol_annual": 0.1},
        pd.DataFrame({"mkt_rf": [0.01, -0.01, 0.02, 0.0, -0.005]}), 252)
    assert rd["available"] is False
    assert rd["reason"] == "no_intercept_standard_error"
    assert rd["factor_share"] is None and rd["specific_share"] is None


# ============ THE CHOKEPOINT: gating by construction, not by memory =========
#
# Rounds 3-8 each pinned the fields the author knew about. Each one passed, and
# each one shipped a sibling — usually three lines below the new gate, because
# the enumeration was done by the reasoning that had just missed one. The two
# tests at the end of this section do not pin fields. They WALK the payload and
# they WALK the prose, and the allow-list below is the only way out. A field
# added anywhere in this module's output, by anyone, that carries a number on a
# degenerate fit fails them with no edit here.

# The fields this module may still publish about a fit with NO TESTABLE
# INTERCEPT. Declared here INDEPENDENTLY of the module's own
# `HONEST_UNDER_DEGENERACY`: the module's table is what the chokepoint obeys,
# this is what the reader is promised, and
# test_the_honest_table_is_declared_in_two_places_that_must_agree asserts the two
# are the same set. Widening one without the other fails.
#
# Two kinds, and the reason each is honest when the arithmetic is dust:
#   POINT ESTIMATES — the intercept IS the constant the factors cannot
#     reproduce, a loading IS the exposure; withholding them would replace a
#     fabricated number with a fabricated absence.
#   REFUSAL EVIDENCE — the quantities whose dustiness IS the reason for the
#     refusal, published only beside `alpha_testable: False` and a reason code,
#     quoted by no sentence (the prose walker proves it), and read by the
#     predicate itself, which is why nulling them would erase the reason code.
ROW_POINT_ESTIMATES = frozenset({
    "alpha_per_period", "alpha_annual", "alpha_annual_geometric",
    "resid_vol_annual",          # the residual SCALE: how much dust there is
})
ROW_REFUSAL_EVIDENCE = frozenset({
    "alpha_se_hac", "alpha_t_hac", "alpha_t_ols", "r2", "adj_r2",
    "resid_var_share", "resid_uncentered_share",
})
ROW_STRUCTURAL = frozenset({
    "n_obs", "df_resid", "hac_lags", "annualization", "condition_number",
    "alpha_ci95_t_critical",     # a Student-t quantile of df alone
    "alpha_testable",            # the predicate's own answer
})
ROW_HONEST = ROW_POINT_ESTIMATES | ROW_REFUSAL_EVIDENCE | ROW_STRUCTURAL
BETA_HONEST = frozenset({"beta", "se_hac", "vif"})


def _row_paths(prefix: str) -> set[str]:
    return ({f"{prefix}.{k}" for k in ROW_HONEST}
            | {f"{prefix}.betas.*.{k}" for k in BETA_HONEST})


PUBLISHABLE_ON_A_DEGENERATE_REPORT = frozenset(
    {
        # the run itself: counts and dates, nothing derived from a fit
        "available", "annualization", "window.coverage", "window.n_obs",
        "factors.synthetic",
        "monthly.available", "monthly.disagrees_with_daily",
        # the rolling level: two point estimates and the predicate's answer
        "rolling_market_beta.*.beta", "rolling_market_beta.*.alpha_annual",
        "rolling_market_beta.*.alpha_testable",
        # the variance decomposition: x'Fx and the residual scale squared are
        # properties of the exposures; every SHARE built from them is not
        "risk_decomposition.available", "risk_decomposition.factor_variance",
        "risk_decomposition.specific_variance",
        # multiple testing: the hurdles are properties of df and the attempt
        # count, the p-values and the survival flag are properties of the fit
        "multiple_testing.attempts", "multiple_testing.alpha_testable",
        "multiple_testing.t_hurdle_bonferroni",
        "multiple_testing.t_hurdle_harvey_liu_zhu",
        "multiple_testing.verdict.available",
        "spanning.verdict.available",
        # diagnostics: the design matrix's own properties
        "diagnostics.condition_number", "diagnostics.vif.*",
    }
    | _row_paths("models.*") | _row_paths("monthly.models.*")
    | _row_paths("spanning")
)


def _leaves(value, path: tuple[str, ...] = ()):
    """Every non-container leaf of a payload, with its dotted path.

    Structure-walking is the point: a test that lists the fields it knows about
    is a test written by the reasoning that missed the last one.
    """
    if isinstance(value, dict):
        for k, v in value.items():
            yield from _leaves(v, path + (str(k),))
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            yield from _leaves(v, path + (str(i),))
    else:
        yield ".".join(path), value


def _is_number(v) -> bool:
    return isinstance(v, (bool, int, float, np.integer, np.floating, np.bool_))


def _path_allowed(path: str, patterns=PUBLISHABLE_ON_A_DEGENERATE_REPORT) -> bool:
    got = path.split(".")
    for pattern in patterns:
        segs = pattern.split(".")
        if len(segs) == len(got) and all(s == "*" or s == g
                                        for s, g in zip(segs, got)):
            return True
    return False


def _degenerate_report(fd: FactorData, n: int = 300):
    """One report in which EVERY fit is degenerate, so the whole payload can be
    walked at once.

    `rf + 1.2*mkt_rf` against a benchmark of `rf + mkt_rf` makes the daily
    ladder, the spanning regression and every rolling window an exact linear
    combination; 300 sessions is over the 252-day daily floor and under the
    36-month monthly one, so the monthly pass declines instead of contributing
    the one legitimately-testable ladder on this input (which has its own test:
    test_the_two_ladders_never_disagree_in_silence).
    """
    f = fd.frame.iloc[:n]
    strat = f["rf"] + 1.2 * f["mkt_rf"]
    bench = f["rf"] + f["mkt_rf"]
    return at.attribution_report(strat, bench_returns=bench, bench_label="SPY",
                                 attempts=7, factors=fd)


def test_the_honest_table_is_declared_in_two_places_that_must_agree():
    """The module's chokepoint table and this file's allow-list are the same set.

    The chokepoint obeys `attribution.HONEST_UNDER_DEGENERACY`; the walker below
    obeys `PUBLISHABLE_ON_A_DEGENERATE_REPORT`. If the module's table alone were
    authoritative, adding a path to it would silently widen what the walker
    permits and the enumerating test would go quiet — the gate would be
    self-approving. So the two are declared independently and compared here: a
    new honest field costs its author an entry in both places and a written
    reason, which is the friction this whole round is about.
    """
    # the module's table is keyed by LEAF PATH within one block; the walker's is
    # keyed by path within the whole report. Compare the leaf names.
    module_leaves = {p.split(".")[-1] for p in at.HONEST_UNDER_DEGENERACY}
    walker_leaves = {p.split(".")[-1] for p in PUBLISHABLE_ON_A_DEGENERATE_REPORT}
    # `date`/`untestable_kind` are strings, `coverage`/`synthetic` never pass
    # through the chokepoint (they are not derived from a fit), so they are the
    # walker's alone; everything else must be in both.
    walker_only = {"coverage", "synthetic", "disagrees_with_daily"}
    assert walker_leaves - walker_only == module_leaves, (
        (walker_leaves - walker_only) ^ module_leaves)
    # every entry carries a real reason, not a placeholder
    for path, reason in at.HONEST_UNDER_DEGENERACY.items():
        assert len(reason) >= 25, (path, reason)
    # and the table names only quantities, never whole subtrees: a wildcard as
    # the LAST segment would exempt everything under it
    for path in at.HONEST_UNDER_DEGENERACY:
        assert not path.endswith("*") or path in ("vif.*",), path


def test_every_numeric_leaf_of_a_degenerate_report_is_null_unless_declared_honest(
        factors):
    """THE ENUMERATING TEST — and the one this module should have had at round 3.

    It does not know what fields exist. It walks the report payload, finds every
    numeric and boolean leaf, and requires each one to be either (a) on the
    declared allow-list above, with a written reason, or (b) None. A field added
    tomorrow — by anyone, anywhere in the payload, inside the chokepoint or
    beside it — is therefore gated BY CONSTRUCTION: if it goes through the
    chokepoint it comes out None and passes; if it bypasses the chokepoint it
    carries a number and fails THIS test, with no edit here.

    What the field-pinning tests it replaces could not see, in order: α's 95% CI
    and p-value (round 8 gated the t three lines above them), every loading's t,
    `appraisal_ratio` (round 9: alpha over a residual scale of rounding dust,
    still three lines below), `alpha_significant: True` exported beside
    `alpha_testable: False`, the rolling level's per-window alpha, and
    `risk_decomposition`'s shares.
    """
    payload, _ = _degenerate_report(factors)
    leaves = list(_leaves(payload))
    assert len(leaves) > 200, len(leaves)      # the walk reaches the whole tree

    offenders = [(p, v) for p, v in leaves
                 if _is_number(v) and not _path_allowed(p)]
    assert offenders == [], offenders

    # NON-VACUITY, both ways. The fit really is degenerate everywhere...
    assert payload["available"] is True
    assert payload["models"]["ff6"]["alpha_testable"] is False
    assert payload["spanning"]["alpha_testable"] is False
    assert payload["monthly"]["available"] is False
    assert all(p["alpha_testable"] is False
               for p in payload["rolling_market_beta"])
    # ...and the specific renderings six rounds discovered one at a time are all
    # in the withheld set, so this test cannot be passing because the payload is
    # empty or the walk is shallow.
    nulled = {p for p, v in leaves if v is None}
    for path in ("models.ff6.appraisal_ratio", "models.ff6.alpha_p_hac",
                 "models.ff6.alpha_significant", "models.ff6.alpha_annual_ci95.0",
                 "models.ff6.alpha_annual_ci95.1", "models.ff6.resid_autocorr1",
                 "models.ff6.betas.mom.t_hac", "models.ff6.betas.mom.t_ols",
                 "models.capm.appraisal_ratio", "spanning.alpha_p_hac",
                 "spanning.appraisal_ratio", "spanning.alpha_significant",
                 "spanning.betas.bench_excess.t_hac",
                 "risk_decomposition.factor_share",
                 "risk_decomposition.specific_share",
                 "risk_decomposition.total_vol_annual",
                 "risk_decomposition.per_factor_share.mom",
                 "multiple_testing.p_raw", "multiple_testing.p_bonferroni",
                 "multiple_testing.survives_bonferroni",
                 "diagnostics.resid_autocorr1"):
        assert path in nulled, path
    # every withheld path is INVENTORIED in the block that withheld it, so the
    # export says what is missing rather than leaving a reader to notice
    assert set(payload["models"]["ff6"]["withheld"]) == {
        p.split("models.ff6.")[1] for p in nulled if p.startswith("models.ff6.")}


def test_the_walk_is_per_fit_not_per_report(factors):
    """The same walk where the two ladders DISAGREE: a degenerate daily ladder
    beside a monthly one that is legitimately testable.

    Degeneracy is a property of a FIT, not of a run, and a chokepoint driven by
    the wrong scope would either fabricate on the daily side or silence the
    monthly side. The never-traded strategy over the fixture's 793-session
    flat-RF run is the reachable input where both happen at once (the monthly
    RF leg acquires month-length variation), so the daily subtree must be nulled
    to the same standard as above while the monthly rows publish everything.
    """
    a, b, _rf = [r for r in flat_rf_runs(factors.frame) if r[2] != 0.0][-1]
    idx = factors.frame.index[a:b]
    bench = factors.frame["rf"].loc[idx] + factors.frame["mkt_rf"].loc[idx]
    payload, _ = at.attribution_report(pd.Series(0.0, index=idx),
                                       bench_returns=bench, bench_label="SPY",
                                       attempts=3, factors=factors)
    daily = {k: payload[k] for k in ("models", "spanning", "risk_decomposition",
                                     "multiple_testing", "diagnostics")}
    offenders = [(p, v) for p, v in _leaves(daily)
                 if _is_number(v) and not _path_allowed(p)]
    assert offenders == [], offenders
    assert payload["models"]["ff6"]["alpha_testable"] is False
    # ...and the monthly ladder, which IS testable on this input, keeps every one
    # of the numbers the daily ladder just lost — the gate is scoped to the fit
    monthly = payload["monthly"]["models"]["ff6"]
    assert monthly["alpha_testable"] is True
    assert monthly["withheld"] == []
    for key in ("alpha_p_hac", "appraisal_ratio", "alpha_significant",
                "resid_autocorr1"):
        assert monthly[key] is not None, key
    assert all(v is not None for v in monthly["alpha_annual_ci95"])
    assert payload["monthly"]["disagrees_with_daily"] is True


def test_the_same_walk_finds_those_numbers_published_on_a_healthy_report(factors,
                                                                        window):
    """OVER-CORRECTION, caught by the same machinery. A chokepoint that nulls
    everything would pass the test above perfectly, so the walk is run again on
    an ordinary fit and every path it just required to be None must now be a
    number."""
    rng = np.random.default_rng(17)
    bench = window["rf"] + window["mkt_rf"]
    strat = (window["rf"] + 0.9 * window["mkt_rf"] + 0.0003
             + pd.Series(rng.normal(0.0, 0.004, len(window)), index=window.index))
    payload, _ = at.attribution_report(strat, bench_returns=bench,
                                       bench_label="SPY", attempts=7)
    leaves = dict(_leaves(payload))
    for path in ("models.ff6.appraisal_ratio", "models.ff6.alpha_p_hac",
                 "models.ff6.alpha_significant", "models.ff6.alpha_annual_ci95.0",
                 "models.ff6.resid_autocorr1", "models.ff6.betas.mom.t_hac",
                 "spanning.alpha_p_hac", "spanning.alpha_significant",
                 "risk_decomposition.factor_share",
                 "risk_decomposition.per_factor_share.mom",
                 "multiple_testing.p_raw", "multiple_testing.p_bonferroni",
                 "multiple_testing.survives_bonferroni",
                 "diagnostics.resid_autocorr1"):
        assert path in leaves, path
        assert leaves[path] is not None, path
    assert payload["monthly"]["available"] is True
    # nothing was withheld anywhere: every block's inventory is empty, on both
    # ladders and on the spanning fit
    for block in (*payload["models"].values(), payload["spanning"],
                  *payload["monthly"]["models"].values(),
                  payload["risk_decomposition"], payload["multiple_testing"],
                  payload["diagnostics"]):
        assert block["withheld"] == [], block.get("key", block.get("reason"))


# ---- the same discipline for RENDERED PROSE -------------------------------
#
# A number withheld from the payload and then printed in a sentence is not
# withheld. These are the shapes this module formats its verdict quantities in —
# `{:+.1%}`, `{:.1%}`, `{:.2f}`, `{:.3f}`, `{:.0%}` — reduced to two rules that
# do not need to know which field produced them.

_NUMBER_TOKEN = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?%?")
# The words a verdict quantity is introduced by. A digit within 15 characters
# after any of them is a quoted verdict whatever its format — this is what
# catches an R² rendered as an integer percent ("explains 100% of the variation"),
# which the decimal rule alone cannot see.
_VERDICT_WORD = re.compile(
    r"(\bt\s*[=><]|t-statistic|R²|\bR\^?2\b|\bp\s*[=><]|p-value|appraisal ratio"
    r"|confidence interval|explains|significance of)", re.I)
# The documented exemptions, each a quantity the fit DOES support or the
# refusal's own evidence. Anything not here has to be an integer or scientific
# dust.
_PROSE_EXEMPTIONS = (
    (re.compile(r"VIF \d+(?:\.\d+)?"),
     "a VIF is a property of the factor matrix, not of this fit"),
    (re.compile(r"condition number \d+(?:\.\d+)?"),
     "the scaled condition index is a property of the design matrix alone"),
    (re.compile(r"beta of \d+\.\d+"),
     "the loading is a point estimate this fit supports; the spanning refusal "
     "quotes it deliberately, in place of the intercept it will not quote"),
    (re.compile(r"\d+% (of the backtest|coverage)"),
     "the overlap share is a count of days, not a result"),
)


def _dust_scientific(tok: str) -> bool:
    """`2.2e-16` is the refusal's own evidence — a share or a standard error
    reported as dust. A mantissa with an exponent of -5 or smaller cannot be a
    verdict quantity this module would print; `2.32e+00` is NOT exempt, which is
    the point of reading the exponent rather than the notation.

    `0.00e+00` is exempt as well: a standard error of exactly zero is the tell
    the NaN-path refusal quotes, and zero is not a magnitude claim. Deliberately
    written as the scientific-zero shape only — a fabricated alpha printed as
    `+0.0%` is the round-6 defect and must stay caught.
    """
    if re.fullmatch(r"[-+]?0\.0+[eE][-+]?0+", tok):
        return True
    m = re.fullmatch(r"[-+]?\d+(?:\.\d+)?[eE]-(\d+)", tok)
    return bool(m) and int(m.group(1)) >= 5


def _plain_integer(tok: str) -> bool:
    """Counts, years, date fragments and the beta-derived integer percents:
    `252`, `1987`, `-12`, `36`, `120%`. A SIGNED percent is not here — nothing
    this module says honestly is a signed percent, and every alpha it formats
    is one."""
    return bool(re.fullmatch(r"[-+]?\d+", tok) or re.fullmatch(r"\d+%", tok))


def _leaked_numbers(text: str, extra_masks: tuple = ()) -> list[str]:
    masked = text
    for pattern in extra_masks:
        masked = pattern.sub(lambda m: "#" * len(m.group()), masked)
    for pattern, _why in _PROSE_EXEMPTIONS:
        masked = pattern.sub(lambda m: "#" * len(m.group()), masked)
    leaks = [tok for tok in (m.group() for m in _NUMBER_TOKEN.finditer(masked))
             if not (_dust_scientific(tok) or _plain_integer(tok))]
    leaks += [masked[m.start():m.end() + 15] for m in _VERDICT_WORD.finditer(masked)
              if re.search(r"\d", masked[m.end():m.end() + 15])]
    return leaks


def _prose(payload: dict, findings: list) -> list[tuple[str, str]]:
    """Every sentence this module can put in front of a reader for this run:
    every string leaf of the payload plus every finding's name and message.

    Two exclusions, both of them strings this module COPIES rather than composes:
    `factors.path` (a filesystem path, whose temp directories can contain any
    digits at all) and `factors.provenance` (the fixture's own header, verbatim —
    it documents the FILE, quotes the "-99.99" missing code and the "mapped to
    NaN" step, and says nothing whatever about this fit). Everything the module
    writes itself is in scope, including every note, message and verdict clause.
    """
    # by LEAF NAME, because the refusal payload splats the factor metadata at
    # top level while the full payload nests it under `factors`
    skip = ("path", "provenance")
    out = [(p, v) for p, v in _leaves(payload)
           if isinstance(v, str) and v and p.split(".")[-1] not in skip
           # the `withheld` inventories are lists of FIELD NAMES, and a field
           # name can contain digits (`alpha_annual_ci95`). They are the opposite
           # of a leak: they are the statement that a number is missing.
           and ".withheld." not in f".{p}."]
    for f in findings:
        out.append((f"finding[{f.check}].name", f.name))
        out.append((f"finding[{f.check}].message", f.message))
    return out


def test_no_prose_a_degenerate_report_can_emit_quotes_a_derived_number(factors,
                                                                      window):
    """THE PROSE TWIN of the enumerating test, and it walks sentences rather
    than listing them.

    Withholding a number from the payload and then printing it in a sentence
    withholds nothing, and this module has done exactly that: "R² of 1.000 means
    the strategy IS a factor portfolio" sat in the same findings list as a note
    promising the reader that "no alpha, t-statistic or R² is quoted" — for as
    long as the tests listed the sentences considered risky. This one asks every
    string the module emits, on four degenerate inputs, whether it contains a
    formatted t, p, R², interval bound, ratio or percentage; the exemptions are
    declared above with a reason each.

    The POSITIVE CONTROL at the end is part of the test: the detector is run on
    the sentences this module used to emit, rebuilt from the fit's own numbers,
    and must flag every one of them. A leak detector that cannot be shown to
    detect a leak is decoration.
    """
    a, b, _rf = [r for r in flat_rf_runs(factors.frame) if r[2] != 0.0][-1]
    flat_idx = factors.frame.index[a:b]
    bench = window["rf"] + window["mkt_rf"]
    cases = {
        "exact_combination": at.attribution_report(
            window["rf"] + 1.2 * window["mkt_rf"], bench_returns=bench,
            bench_label="SPY", attempts=7),
        "never_traded_flat_rf": at.attribution_report(
            pd.Series(0.0, index=flat_idx), attempts=7),
        "condition_four_only": at.attribution_report(
            _condition_four_only(window), attempts=7),
        # the NaN path: the whole report refuses, and its refusal message is
        # prose like any other
        "sst_exactly_zero": at.attribution_report(
            window["rf"] + 0.0 / 252.0, attempts=7),
    }
    for label, (payload, findings) in cases.items():
        sp = payload.get("spanning") or {}
        # the one dynamic exemption: the replicating portfolio is built from the
        # spanning BETA, a point estimate the refusal quotes on purpose, and its
        # wording ("120% SPY, funded with 20% borrowed") is percentages
        masks = ()
        if sp.get("replicating_portfolio"):
            masks = (re.compile(re.escape(sp["replicating_portfolio"]), re.I),)
        texts = _prose(payload, findings)
        assert len(texts) >= 8, (label, len(texts))
        for where, text in texts:
            # `\b`: the factor fixture's own provenance header contains the word
            # "provenance", which contains "nan". A rendered NaN is a data problem
            # shown as a number; a substring of an English word is not.
            assert not re.search(r"\bnan\b", text.lower()), (label, where, text)
            leaks = _leaked_numbers(text, masks)
            assert not leaks, (label, where, leaks, text)

        # POSITIVE CONTROL: the sentences this module used to print, on this fit
        if payload["available"]:
            head = payload["models"]["ff6"]
            for sentence in (
                f"the remaining alpha is {head['alpha_annual']:+.1%}/yr with a "
                f"HAC t-statistic of {head['alpha_t_hac']:.2f}",
                f"R² of {head['r2']:.3f} means the strategy IS a factor portfolio",
                f"the model explains {abs(head['r2']):.0%} of the variation",
                f"appraisal ratio {_would_be_appraisal(head):.2f}",
                f"the 95% interval is [{_would_be_ci(head)[0]:+.4%}, "
                f"{_would_be_ci(head)[1]:+.4%}]",
                f"Bonferroni p = {at.t_two_sided_p(head['alpha_t_hac'], head['df_resid']):.3f}",
            ):
                assert _leaked_numbers(sentence, masks), (label, sentence)


def test_the_variance_share_identities_are_asserted_not_assumed(window):
    """PRINCIPLE 4 at the site that renders shares, driven directly.

    Both bounds are algebraic identities, so no input can violate them — which is
    why they are free, and also why they would rot unnoticed as two bare
    `assert`s nobody can reach. They live in a named function, so the violation
    can be handed to them.
    """
    # they hold on a real decomposition
    rng = np.random.default_rng(43)
    strat = (window["rf"] + 0.8 * window["mkt_rf"]
             + pd.Series(rng.normal(0, 0.004, len(window)), index=window.index))
    rd = at.attribution_report(strat, attempts=1)[0]["risk_decomposition"]
    assert rd["available"] is True
    at._assert_variance_identities(rd["factor_share"], rd["specific_share"],
                                  rd["per_factor_share"])
    # and they are live: a partition that does not sum to one, and per-factor
    # contributions that do not sum to the factor share, each raise
    with pytest.raises(AssertionError, match="partition"):
        at._assert_variance_identities(0.6, 0.6, {"mkt_rf": 0.6})
    with pytest.raises(AssertionError, match="sum to the factor share"):
        at._assert_variance_identities(0.6, 0.4, {"mkt_rf": 0.5, "smb": 0.2})

    # and the SITE really calls them — a bound asserted in a function nobody
    # invokes is the same as an unasserted bound
    calls: list[tuple] = []
    real = at._assert_variance_identities
    try:
        at._assert_variance_identities = lambda *a: (calls.append(a),
                                                     real(*a))[1]
        at.attribution_report(strat, attempts=1)
    finally:
        at._assert_variance_identities = real
    assert len(calls) == 1 and calls[0][0] + calls[0][1] == pytest.approx(1.0)


def test_the_disagreement_note_names_a_mechanism_only_for_the_kind_it_explains():
    """Item 4: `elif daily_kind:` claimed NONLINEAR COMPOUNDING for ANY non-empty
    daily degeneracy.

    Nonlinear compounding explains exactly one kind — an exact linear combination
    stops being one at monthly frequency. For a fit whose intercept has no
    standard error at all, or whose excess return has no variation, curvature is
    not the mechanism, and a refusal that misstates its own reason is the defect
    class this module keeps returning to (the mechanism is the part a reader
    repeats). All four branches driven here, since only one is reachable from the
    fixture.
    """
    calendar = at._ladder_disagreement_note("no_dependent_variation", "")
    assert "CALENDAR ARTEFACT" in calendar and "NONLINEAR" not in calendar

    curvature = at._ladder_disagreement_note("no_testable_intercept", "")
    assert "NONLINEAR" in curvature and "CALENDAR ARTEFACT" not in curvature

    for kind in ("no_variation", "no_intercept_standard_error"):
        note = at._ladder_disagreement_note(kind, "")
        assert "NONLINEAR" not in note, kind
        assert "CALENDAR ARTEFACT" not in note, kind
        assert f"the daily fit's degeneracy is {kind}" in note, kind
        assert "NO mechanism is named" in note, kind
        # it still says WHAT disagrees and WHY the reader must not read the
        # monthly pass as a robustness check
        assert "DISAGREE" in note and "RESAMPLING" in note, kind

    monthly_side = at._ladder_disagreement_note("", "no_variation")
    assert "the MONTHLY fit is the degenerate one" in monthly_side
    assert "NONLINEAR" not in monthly_side


def test_the_two_ladders_never_disagree_in_silence(factors, window):
    """The predicate at the OTHER level: the monthly ladder fits its own
    regressions, and on the reachable degenerate input the two ladders reach
    OPPOSITE answers about the same strategy.

    A strategy that never traded over the fixture's 793-session flat-RF run has
    no variation of its own daily, so every daily verdict is refused. Compounding
    to calendar months makes the RF leg vary with the NUMBER OF TRADING DAYS in
    the month, so the monthly ff6 clears the predicate and publishes t = -104 —
    an artefact of the calendar, sitting beside a daily ladder that refuses to
    quote a t at all. The payload has to say so.
    """
    # the LONGEST nonzero flat-RF run (793 sessions), the one long enough for the
    # monthly pass to run at all
    a, b, _rf = [r for r in flat_rf_runs(factors.frame) if r[2] != 0.0][-1]
    idx = factors.frame.index[a:b]
    payload, _ = at.attribution_report(pd.Series(0.0, index=idx), attempts=1)

    monthly = payload["monthly"]
    assert monthly["available"] is True          # the disagreement is reachable
    assert at._untestable_kind(payload["models"]["ff6"]) == "no_dependent_variation"
    assert at._untestable_kind(monthly["models"]["ff6"]) == ""
    assert monthly["models"]["ff6"]["alpha_testable"] is True
    assert abs(monthly["models"]["ff6"]["alpha_t_hac"]) > 10.0

    assert monthly["disagrees_with_daily"] is True
    assert "DISAGREE" in monthly["note"]
    assert "CALENDAR ARTEFACT" in monthly["note"]

    # the OTHER daily degeneracy survives the resampling for a DIFFERENT reason,
    # and the note names that one instead of reusing the calendar story
    exact = at.attribution_report(window["rf"] + 1.2 * window["mkt_rf"],
                                  attempts=1)[0]
    assert at._untestable_kind(exact["models"]["ff6"]) == "no_testable_intercept"
    assert exact["monthly"]["disagrees_with_daily"] is True
    assert "NONLINEAR" in exact["monthly"]["note"]
    assert "CALENDAR ARTEFACT" not in exact["monthly"]["note"]

    # and the third branch — a well-posed daily fit beside a DEGENERATE monthly
    # one — driven directly, because no input built from this fixture reaches it
    # and a branch no test can drive is a branch that rots
    other_way = at._ladder_disagreement_note("", "no_variation")
    assert "the MONTHLY fit is the degenerate one (no_variation)" in other_way
    assert "the daily fit is well-posed" in other_way
    assert "CALENDAR ARTEFACT" not in other_way and "NONLINEAR" not in other_way

    # and an ordinary strategy says nothing of the kind
    rng = np.random.default_rng(5)
    ordinary = (window["rf"] + window["mkt_rf"]
                + pd.Series(rng.normal(0.0, 0.004, len(window)), index=window.index))
    ok = at.attribution_report(ordinary, attempts=1)[0]["monthly"]
    assert ok["available"] is True
    assert ok["disagrees_with_daily"] is False
    assert ok["note"] is None


def test_the_predicate_refuses_nothing_a_real_fit_needs(window):
    """OVER-CORRECTION, the other direction. A policy applied at every site is a
    policy that can silence every site, so the ordinary paths are pinned here:
    a noisy strategy, a tight-but-genuine fit seven orders below its signal, and
    the all-zero strategy over a VARYING-rf window (which is legitimately
    testable — its excess return is -RF and RF really moves)."""
    rng = np.random.default_rng(19)
    noisy = (window["rf"] + 0.8 * window["mkt_rf"] + 0.5 * window["hml"]
             + pd.Series(rng.normal(0.0, 0.004, len(window)), index=window.index))
    tight = (window["rf"] + window["mkt_rf"]
             + pd.Series(rng.normal(0.0, 1e-9, len(window)), index=window.index))
    assert window["rf"].nunique() > 1
    for label, strat in (("noisy", noisy), ("tight", tight),
                         ("all_zero_varying_rf", pd.Series(0.0, index=window.index))):
        payload, findings = at.attribution_report(strat, attempts=1)
        head = payload["models"]["ff6"]
        assert head["alpha_testable"] is True, label
        assert at._untestable_kind(head) == "", label
        assert np.isfinite(head["resid_autocorr1"]), label
        assert not any("NO TESTABLE INTERCEPT" in i
                       for i in payload["diagnostics"]["issues"]), label
        rd = payload["risk_decomposition"]
        assert rd["available"] is True, label
        assert rd["factor_share"] + rd["specific_share"] == pytest.approx(1.0), label
        assert payload["monthly"]["disagrees_with_daily"] is False, label
        assert payload["multiple_testing"]["verdict"]["available"] is True, label
    # and the healthy one still earns the PASS this round removed from the others
    rel = [f for f in at.attribution_report(noisy, attempts=1)[1]
           if f.check == "factor_reliability"][0]
    assert rel.severity == "pass"
    assert "well-posed" in rel.message


def test_uncentered_share_is_the_bounded_one(window):
    """Why Σy² is the denominator that works, as arithmetic rather than as prose.

    SST = Σy² - n·ȳ², so SST ≤ Σy² always, and therefore SSR/Σy² ≤ SSR/SST always:
    the uncentered share is the CONSERVATIVE one, and it can only fall below the
    floor when the fit really did reproduce y itself. The centered share has no
    such bound once SST stops being a real variance, which is the entire hole.
    """
    rng = np.random.default_rng(5)
    for sd, const in ((0.01, 0.0), (0.01, 0.05), (1e-6, 0.0), (0.0, 0.0002)):
        y = rng.normal(const, sd, 400) if sd > 0 else np.full(400, const)
        X = np.column_stack([np.ones(400), rng.normal(0, 0.01, 400)])
        f = at.ols_hac(y, X, 5)
        sst = float(((y - y.mean()) ** 2).sum())
        uss = float((y ** 2).sum())
        assert sst <= uss * (1 + 1e-12), (sd, const)
        assert f.resid_uncentered_share <= f.resid_var_share * (1 + 1e-9), (sd, const)
        # the identity the guard now asserts, on a fit that is NOT degenerate
        if sd > 0:
            assert f.resid_var_share <= 1.0 + 1e-12, (sd, const)
            assert f.r2 >= -1e-12, (sd, const)
        else:
            # a constant y: the ratio is impossible and the guard catches it
            assert f.resid_var_share > 1.0
            assert f.resid_uncentered_share <= at.RESID_VAR_SHARE_FLOOR


def test_exact_fit_residual_share_is_orders_of_magnitude_below_the_bar(window):
    """The tolerance is a measurement, not a round number — and this test makes
    the measurements the constant's comment cites.

    It used to make FOUR exact draws while the comment beside
    RESID_VAR_SHARE_FLOOR credited it with "worst of 160 draws: 9.1e-30" and with
    the 5e-15 noisy figure it never computed. A comment that attributes a
    measurement to a test which does not make it is an unverified claim wearing a
    citation, so the draws are here now:

      * 160 exact combinations, |coefficients| spanning ~1 to 1e6: SSR/SST stays
        under 1e-29, i.e. ~13 orders of magnitude below the bar, so the bar sits
        nowhere near the arithmetic noise it has to clear;
      * a genuine residual of sd 1e-9 against a ~1e-2 daily return — seven orders
        of magnitude below the signal, far tighter than any real strategy — lands
        at 5.2e-15, 23× ABOVE the bar, and stays testable, so the refusal cannot
        swallow a real regression.
    """
    cols = list(at.FACTOR_COLUMNS)

    def share(strat: pd.Series) -> float:
        excess = (strat - window["rf"]).to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(excess))]
                            + [window[c].to_numpy(dtype=float) for c in cols])
        return at.ols_hac(excess, X, at.HAC_LAGS_DAILY).resid_var_share

    rng = np.random.default_rng(0)
    worst, draws = 0.0, 0
    for scale in (1.0, 1e2, 1e4, 1e6):
        for _ in range(40):
            coefs = rng.normal(0.0, scale, len(cols))
            exact = window["rf"] + rng.normal(0.0, 0.01) + sum(
                coefs[i] * window[cols[i]] for i in range(len(cols))
            )
            s = share(exact)
            assert s < at.RESID_VAR_SHARE_FLOOR / 1e12, (scale, s)
            worst, draws = max(worst, s), draws + 1
    assert draws == 160                       # the number the comment cites
    assert worst < 1e-29, worst               # measured: 9.05e-30

    noisy = (window["rf"] + 1.2 * window["mkt_rf"]
             + pd.Series(rng.normal(0.0, 1e-9, len(window)), index=window.index))
    s = share(noisy)
    assert s == pytest.approx(5.2e-15, rel=0.05), s
    assert s / at.RESID_VAR_SHARE_FLOOR == pytest.approx(23.5, rel=0.05)
    payload, findings = at.attribution_report(noisy, attempts=1)
    assert payload["models"]["ff6"]["alpha_testable"] is True
    assert [f for f in findings if f.check == "factor_alpha"][0].severity != "note"


def test_r2_stops_distinguishing_a_tight_fit_at_eps_over_four():
    """The constant's OTHER stated justification, re-derived — it was wrong by 4×.

    The comment used to say the bar sits at machine epsilon "because that is
    precisely where the R² this module reports stops being able to tell a tight
    fit from an exact one — 1.0 - SSR/SST rounds to 1.0 below eps/2". Neither
    number is the threshold. `1.0 - s` is computed in round-to-nearest on a grid
    of spacing eps/2 below 1.0, so it returns exactly 1.0 for every s up to and
    including eps/4, and 1.0 - eps/2 immediately above it. The floor is therefore
    4× LARGER than the R²-indistinguishability point, which makes it
    conservative — it refuses every fit whose R² cannot be told from 1.0, plus a
    narrow band above — not coincident with it.
    """
    eps = float(np.finfo(float).eps)
    assert at.RESID_VAR_SHARE_FLOOR == eps

    # bisect for the largest s with 1.0 - s == 1.0; no library involved
    lo, hi = 0.0, eps
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 1.0 - mid == 1.0:
            lo = mid
        else:
            hi = mid
    assert lo == pytest.approx(eps / 4.0, rel=1e-12)
    assert 1.0 - eps / 4.0 == 1.0                      # still exactly 1.0
    assert 1.0 - eps / 4.0 * 1.01 != 1.0               # just above, it is not
    assert 1.0 - eps / 2.0 == 1.0 - eps / 2.0 and 1.0 - eps / 2.0 != 1.0
    # and the claim the comment now makes: the floor is 4× that threshold
    assert at.RESID_VAR_SHARE_FLOOR / lo == pytest.approx(4.0, rel=1e-12)


def test_short_attribution_window_is_disclosed(factors):
    """The factor library lags ~6 weeks. When the backtest runs past the end of
    the factor data, the report must say the two windows differ."""
    end = pd.Timestamp(factors.end)
    idx = pd.bdate_range(end - pd.Timedelta(days=900), end + pd.Timedelta(days=120))
    base = factors.frame["mkt_rf"].reindex(idx).fillna(0.0)
    strat = base + 0.0002
    payload, findings = at.attribution_report(strat, attempts=1)
    assert payload["available"] is True
    assert payload["window"]["attribution_end"] == factors.end
    assert payload["window"]["backtest_end"] > payload["window"]["attribution_end"]
    win_f = [f for f in findings if f.check == "factor_window"]
    assert win_f and win_f[0].severity == "note"
    assert "roughly six weeks" in win_f[0].message


# --------------------------------------------------------- monthly pass

def test_monthly_pass_agrees_with_the_daily_regression(window):
    """Same construction, two frequencies: alpha must annualize to the same
    2%/yr whether it is 252 × daily or 12 × monthly."""
    strat = window["rf"] + 1.1 * window["mkt_rf"] + 0.02 / 252.0
    payload, _ = at.attribution_report(strat, attempts=1)
    assert payload["monthly"]["available"] is True
    m = payload["monthly"]["models"]["capm"]
    assert m["annualization"] == 12
    assert m["n_obs"] >= at.MIN_MONTHLY_OBS
    assert m["alpha_annual"] == pytest.approx(0.02, abs=0.005)
    assert m["betas"]["mkt_rf"]["beta"] == pytest.approx(1.1, abs=0.05)


def test_monthly_pass_keeps_every_complete_calendar_month(factors, window):
    """A month is partial or not according to the TRADING calendar.

    January 2016 is fully covered — the 1st was a market holiday, so the first
    session is the 4th — but comparing that session to the 1st of the month
    dropped it, and the same arithmetic dropped the last month unless the window
    happened to end exactly on a calendar month end. Independent count: 2016-01
    through 2024-12 is 9 × 12 = 108 complete months.
    """
    daily = window["rf"] + window["mkt_rf"]
    assert str(daily.index[0].date()) == "2016-01-04"     # Jan 1 was a holiday
    assert str(daily.index[-1].date()) == "2024-12-31"

    monthly = at._to_monthly(daily, factors.frame.index)
    assert len(monthly) == 108
    assert str(monthly.index[0].date()) == "2016-01-31"   # January is NOT dropped
    assert str(monthly.index[-1].date()) == "2024-12-31"

    # a genuinely partial month still goes: stop one session early and December
    # is no longer complete
    truncated = at._to_monthly(daily.loc[:"2024-12-30"], factors.frame.index)
    assert len(truncated) == 107
    assert str(truncated.index[-1].date()) == "2024-11-30"

    # and the compounding itself is right: January 2016 is the product of its
    # 19 sessions, computed here from the raw series
    hand = float((1.0 + daily.loc["2016-01-01":"2016-01-31"]).prod() - 1.0)
    assert monthly.iloc[0] == pytest.approx(hand, rel=1e-12)


def test_monthly_pass_declines_when_there_are_too_few_months(window):
    strat = (window["rf"] + window["mkt_rf"]).iloc[:400]   # ~19 months
    payload, _ = at.attribution_report(strat, attempts=1)
    assert payload["monthly"]["available"] is False
    assert "36-month academic floor" in payload["monthly"]["note"]


# ---------------------------------------------------- risk decomposition

def test_risk_decomposition_shares_sum_to_one(window):
    rng = np.random.default_rng(41)
    noise = pd.Series(rng.normal(0, 0.004, len(window)), index=window.index)
    strat = window["rf"] + 0.8 * window["mkt_rf"] + 0.5 * window["hml"] + noise
    payload, _ = at.attribution_report(strat, attempts=1)
    rd = payload["risk_decomposition"]
    assert rd["available"] is True
    assert rd["factor_share"] + rd["specific_share"] == pytest.approx(1.0, abs=1e-12)
    assert sum(rd["per_factor_share"].values()) == pytest.approx(
        rd["factor_share"], abs=1e-10)
    assert 0.0 < rd["factor_share"] < 1.0


def test_risk_decomposition_annualizes_the_factor_covariance():
    """Hand-worked, and NOT via the shares.

    factor_share + specific_share == 1.0 holds by construction whatever scale the
    factor covariance is on, so the shares identity cannot detect a missing
    annualization. Drop the `× 252` and factor_variance is 252× too small — a
    factor share of 3.8% instead of 92.6% — and every shares test still passes.

    Closed form. Both columns have mean 0 and Σx² = 4e-4 over n = 5, so each has
    sample variance (ddof=1) 4e-4/4 = 1e-4, and Σ a_i b_i = 0 so they are exactly
    uncorrelated. Annualized F = diag(0.0252, 0.0252). With x = (2, 1):
        factor_variance  = (4 + 1) × 0.0252            = 0.126
        specific_variance = 0.1²                        = 0.010
        total_vol_annual = sqrt(0.136)                  = 0.36878177829...
        factor_share     = 0.126 / 0.136                = 0.9264705882...
    Un-annualized the same inputs give total_vol_annual = sqrt(0.0105) = 0.1025.
    """
    f = pd.DataFrame({
        "mkt_rf": [-0.01, -0.01, 0.0, 0.01, 0.01],
        "smb": [-0.01, 0.01, 0.0, -0.01, 0.01],
    })
    assert float(f["mkt_rf"].var(ddof=1)) == pytest.approx(1e-4, rel=1e-12)
    assert float(f["smb"].var(ddof=1)) == pytest.approx(1e-4, rel=1e-12)
    assert float((f["mkt_rf"] * f["smb"]).sum()) == pytest.approx(0.0, abs=1e-18)

    model = {
        "factors": ["mkt_rf", "smb"],
        "betas": {"mkt_rf": {"beta": 2.0}, "smb": {"beta": 1.0}},
        "resid_vol_annual": 0.1,
        # risk_decomposition asks the module's one degeneracy predicate before it
        # publishes a share, so a hand-built model row has to carry the four
        # quantities that predicate reads. All four are ordinary-fit values here —
        # this test is about the annualization, not about degeneracy.
        "resid_var_share": 0.40,
        "resid_uncentered_share": 0.40,
        "alpha_t_hac": 1.1,
        "r2": 0.60,
    }
    rd = at.risk_decomposition(model, f, 252)

    assert rd["available"] is True
    assert rd["factor_variance"] == pytest.approx(0.126, rel=1e-12)
    assert rd["specific_variance"] == pytest.approx(0.01, rel=1e-12)
    assert rd["total_vol_annual"] == pytest.approx(math.sqrt(0.136), rel=1e-12)
    assert rd["total_vol_annual"] != pytest.approx(math.sqrt(0.0105), rel=1e-3)
    assert rd["factor_share"] == pytest.approx(0.126 / 0.136, rel=1e-12)
    # the per-factor contributions x_k (Fx)_k / σ²_p, also annualized
    assert rd["per_factor_share"]["mkt_rf"] == pytest.approx(
        4 * 0.0252 / 0.136, rel=1e-12)
    assert rd["per_factor_share"]["smb"] == pytest.approx(
        0.0252 / 0.136, rel=1e-12)


def test_risk_decomposition_refusal_keeps_the_documented_key_set():
    """The failure return used to be a bare {'available': False}, so a consumer
    that had already passed the documented branch check KeyError'd on
    rd['factor_share']. Both shapes must carry the same keys."""
    model = {
        "factors": ["mkt_rf", "smb"],
        "betas": {"mkt_rf": {"beta": 0.0}, "smb": {"beta": 0.0}},
        "resid_vol_annual": 0.0,
        # Legal shares and a finite t: this row must reach the
        # ZERO-TOTAL-VARIANCE refusal, not the degeneracy refusal, or the test
        # stops covering the branch it names.
        "resid_var_share": 0.40,
        "resid_uncentered_share": 0.40,
        "alpha_t_hac": 1.1,
        "r2": 0.60,
    }
    f = pd.DataFrame({"mkt_rf": [0.01, -0.01, 0.02], "smb": [0.0, 0.01, -0.01]})
    rd = at.risk_decomposition(model, f, 252)
    assert rd["available"] is False
    assert rd["reason"] == "zero_total_variance"
    for key in ("factor_variance", "specific_variance", "factor_share",
                "specific_share", "total_vol_annual", "per_factor_share"):
        assert key in rd, key
    assert set(rd["per_factor_share"]) == {"mkt_rf", "smb"}


# --------------------------------------------- estimator conventions

def test_residual_volatility_uses_the_regression_degrees_of_freedom():
    """σ̂ = sqrt(SSR/(n−k)), not sqrt(SSR/(n−1)).

    Hand-worked, five points, closed form. x = (−2,−1,0,1,2), y = (1,−1,0,1,−1):
    Σy = 0 so the intercept is 0; Σxy = −2 and Σx² = 10 so the slope is −0.2.
    Fitted values −0.2x = (0.4, 0.2, 0, −0.2, −0.4), residuals
    (0.6, −1.2, 0, 1.2, −0.6), SSR = 0.36+1.44+0+1.44+0.36 = 3.6.
    With n = 5, k = 2: σ̂ = sqrt(3.6/3) = sqrt(1.2) = 1.0954451150103321.
    The ddof=1 version would be sqrt(3.6/4) = 0.9486832980505138 — 13.4% low,
    and the appraisal ratio 13.4% high.
    """
    idx = pd.bdate_range("2020-01-01", periods=5)
    f = pd.DataFrame({"x": [-2.0, -1.0, 0.0, 1.0, 2.0]}, index=idx)
    y = pd.Series([1.0, -1.0, 0.0, 1.0, -1.0], index=idx)
    m = at.regress(y, f, ("x",), periods_per_year=1, hac_lags=0, label="hand")

    assert m["betas"]["x"]["beta"] == pytest.approx(-0.2, abs=1e-12)
    assert m["alpha_per_period"] == pytest.approx(0.0, abs=1e-12)
    assert m["resid_vol_annual"] == pytest.approx(math.sqrt(1.2), rel=1e-12)
    assert m["resid_vol_annual"] != pytest.approx(math.sqrt(0.9), rel=1e-6)


def test_alpha_interval_uses_the_exact_student_t_not_1_96():
    """Same five points: df_resid = 3, and the textbook two-tailed 5% critical
    value on 3 degrees of freedom is 3.182 (Fisher's table), not 1.96. On the
    36-month monthly floor the difference is a 4% understatement of the
    interval, which is exactly the kind of quiet flattery this module exists to
    refuse."""
    idx = pd.bdate_range("2020-01-01", periods=5)
    f = pd.DataFrame({"x": [-2.0, -1.0, 0.0, 1.0, 2.0]}, index=idx)
    y = pd.Series([1.0, -1.0, 0.0, 1.0, -1.0], index=idx)
    m = at.regress(y, f, ("x",), periods_per_year=1, hac_lags=0, label="hand")

    assert m["df_resid"] == 3
    assert m["alpha_ci95_t_critical"] == pytest.approx(3.182446, abs=5e-6)
    half = (m["alpha_annual_ci95"][1] - m["alpha_annual_ci95"][0]) / 2.0
    assert half == pytest.approx(3.182446 * m["alpha_se_hac"], rel=1e-6)
    assert half != pytest.approx(1.96 * m["alpha_se_hac"], rel=1e-3)
    # published: t_{0.975, 29} = 2.045, the monthly-floor case
    assert at.t_hurdle_for_p(0.05, 29) == pytest.approx(2.045, abs=5e-4)


# --------------------------------------------------------- cost crossover

def test_breakeven_cost_bps_is_priced_on_traded_notional():
    """Hand arithmetic in the ENGINE's units, not the research note's.

    conventions.py: net_ret = gross_ret − traded_notional × bps/10⁴, and
    traded_notional = Σ|Δw| = 2 × the reported one-way turnover. So a strategy
    with 3%/yr gross alpha that turns over 2.5× one-way trades 5.0 of notional
    a year and pays 5.0 × c/10⁴ per year: c* = 0.03/5.0 × 10⁴ = 60 bps per
    side. The old code divided by the 2.5 and reported 120 — twice the truth.
    """
    assert at.breakeven_cost_bps(0.03, annual_traded_notional=5.0) == pytest.approx(60.0)
    assert at.breakeven_cost_bps(0.03, annual_turnover=2.5) == pytest.approx(60.0)
    assert at.breakeven_cost_bps(0.03, annual_turnover=2.5) != pytest.approx(120.0)
    # a churner: 50× one-way = 100 of notional a year → 3 bps
    assert at.breakeven_cost_bps(0.03, annual_turnover=50.0) == pytest.approx(3.0)
    # nothing ever traded: no cost rate can kill it
    assert at.breakeven_cost_bps(0.03, annual_turnover=0.0) == float("inf")
    # already under water before the first commission — negative, never a
    # comforting large positive number
    assert at.breakeven_cost_bps(-0.03, annual_turnover=2.5) == pytest.approx(-60.0)
    assert math.isnan(at.breakeven_cost_bps(float("nan"), annual_turnover=2.5))


def test_breakeven_cost_bps_refuses_an_unnamed_denominator():
    """The two denominators differ by exactly 2×, so the function will not let a
    caller pass one positionally and hope."""
    with pytest.raises(TypeError):
        at.breakeven_cost_bps(0.03, 2.5)
    with pytest.raises(ValueError, match="exactly one of"):
        at.breakeven_cost_bps(0.03)
    with pytest.raises(ValueError, match="exactly one of"):
        at.breakeven_cost_bps(0.03, annual_turnover=2.5, annual_traded_notional=5.0)


def _roundtrip_panel(factors: FactorData):
    """Two names with the same market beta and different idiosyncratic paths, so
    a 21-day momentum rank rotates between them and the book really trades."""
    f = factors.frame.loc["2015-01-02":"2024-12-31"]
    rng = np.random.default_rng(404)
    n = len(f)
    rets = {
        "A": f["rf"] + f["mkt_rf"] + rng.normal(0.0006, 0.012, n),
        "B": f["rf"] + f["mkt_rf"] + rng.normal(0.0000, 0.012, n),
        "SPY": f["rf"] + f["mkt_rf"],
    }
    panel = pd.DataFrame({k: 100.0 * (1.0 + v).cumprod() for k, v in rets.items()},
                         index=f.index)
    spec = StrategySpec(
        name="breakeven roundtrip",
        universe=Universe(preset="user_watchlist", resolved_tickers=["A", "B"]),
        selection=Selection(rank_by=Signal(type="return", lookback_days=21),
                            direction="top", n=1),
        weighting="equal",
        rebalance="monthly",
        costs=Costs(sweep_bps=[0, 5, 20]),
        benchmark="SPY",
        window=Window(start="2016-01-04", end="2024-12-31"),
    )
    return spec, panel, f["rf"]


def test_engine_still_charges_cost_on_two_times_one_way_turnover(factors):
    """Drift detector for NOTIONAL_PER_ONE_WAY_TURNOVER.

    The constant in attribution.py is a copy of an engine convention. This
    re-derives the ratio from a real engine run, so if engine/backtest.py ever
    changes what it charges against, this fails HERE rather than silently
    doubling a reported breakeven.
    """
    spec, panel, rf = _roundtrip_panel(factors)
    res = run(spec, panel, ["A", "B"], rf_daily=rf)
    assert res.turnover.sum() > 10.0                    # it really traded
    assert res.traded_notional.sum() == pytest.approx(
        at.NOTIONAL_PER_ONE_WAY_TURNOVER * res.turnover.sum(), rel=1e-12)
    years = len(res.accrual_index) / 252.0
    assert res.traded_notional.sum() / years == pytest.approx(
        at.NOTIONAL_PER_ONE_WAY_TURNOVER * res.annual_turnover, rel=1e-12)


def test_breakeven_cost_bps_is_where_the_engine_actually_kills_alpha(factors):
    """The only test of this function that cannot be fooled by a convention
    mismatch: take the reported breakeven, charge exactly that rate through the
    ENGINE's own cost model, re-run the factor regression on the resulting net
    series, and demand that alpha has landed on zero.

    This is what kills the 2× bug: at the old (doubled) answer the engine has
    taken the whole alpha away and then the same amount again.
    """
    spec, panel, rf = _roundtrip_panel(factors)
    res = run(spec, panel, ["A", "B"], rf_daily=rf)
    years = len(res.accrual_index) / 252.0
    annual_notional = float(res.traded_notional.sum() / years)

    gross = at.attribution_report(res.gross_returns, attempts=1)[0]
    alpha_gross = gross["models"]["ff6"]["alpha_annual"]
    assert alpha_gross > 0.02                           # there is something to kill

    bps = at.breakeven_cost_bps(alpha_gross, annual_traded_notional=annual_notional)
    assert 10.0 < bps < 500.0

    priced = spec.model_copy(update={"costs": Costs(
        sweep_bps=[0, 5, 20, round(bps, 6), round(2.0 * bps, 6)])})
    res2 = run(priced, panel, ["A", "B"], rf_daily=rf)

    at_breakeven = at.attribution_report(
        res2.net_returns[round(bps, 6)], attempts=1)[0]["models"]["ff6"]["alpha_annual"]
    at_double = at.attribution_report(
        res2.net_returns[round(2.0 * bps, 6)], attempts=1)[0]["models"]["ff6"]["alpha_annual"]

    # charged at the reported rate, alpha is gone — within 2% of the gross alpha
    assert abs(at_breakeven) < 0.02 * alpha_gross
    # charged at the rate the OLD formula reported, the engine has taken alpha
    # away twice over: net alpha ≈ −alpha_gross, nowhere near zero
    assert at_double == pytest.approx(-alpha_gross, rel=0.05)


# ---------------------------------------------- the committed fixture itself

def test_fixture_shape_and_known_values(factors):
    """Golden test on the checked-in Ken French data: exact columns, a known
    value on a fixed date, and a plausible units band."""
    frame = factors.frame
    assert list(frame.columns) == ["mkt_rf", "smb", "hml", "rmw", "cma", "mom", "rf"]
    assert not frame.isna().any().any()
    assert frame.index.is_monotonic_increasing and frame.index.is_unique
    assert str(frame.index[0].date()) == "2015-01-02"
    assert len(frame) > 2800

    # Published values for 2015-01-02 (FF5 daily / MOM daily, 202607 CRSP
    # build), converted from percent to decimal.
    row = frame.loc["2015-01-02"]
    assert row["mkt_rf"] == pytest.approx(-0.0012, abs=1e-9)
    assert row["smb"] == pytest.approx(-0.0056, abs=1e-9)
    assert row["hml"] == pytest.approx(0.0012, abs=1e-9)
    assert row["rmw"] == pytest.approx(-0.0028, abs=1e-9)
    assert row["cma"] == pytest.approx(0.0015, abs=1e-9)
    assert row["mom"] == pytest.approx(-0.0008, abs=1e-9)

    # units: daily factor moves are fractions of a percent, never percent
    assert 0.002 <= frame["mkt_rf"].abs().mean() <= 0.02
    assert frame.abs().to_numpy().max() < 0.25
    # RF is a daily rate, never negative
    assert (frame["rf"] >= 0).all()


def test_fixture_header_records_provenance():
    text = FIXTURE.read_text()
    header = "\n".join(ln for ln in text.splitlines() if ln.startswith("#"))
    assert "SYNTHETIC: no" in header
    assert "ken.french" in header
    assert "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip" in header
    assert "F-F_Momentum_Factor_daily_CSV.zip" in header
    assert "Fama" in header  # copyright attribution retained


def test_fixture_is_small_enough_to_commit():
    assert FIXTURE.stat().st_size < 1_000_000
