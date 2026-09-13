"""Statistical inference for the audit — the layer that turns point estimates
into claims with error bars.

WHY this module exists: every verdict in the audit used to reduce to a strict
inequality on a point estimate (``total_return_A > total_return_B``). A
strategy that beat the benchmark by 0.01% over an 18-month holdout scored
identically to one that beat it by 400%. That is not honesty, it is a coin
flip with a colour attached. Everything here exists so that the report can say
"and here is the sampling noise around that number".

WHAT IS IN HERE (all with citations on the individual functions):
  - Sharpe standard error — Lo (2002), plus the non-normal (skew/kurtosis)
    correction of Mertens (2002).
  - Probabilistic Sharpe Ratio — Bailey & López de Prado (2012/2014).
  - Deflated Sharpe Ratio + expected-maximum-Sharpe under N trials —
    Bailey & López de Prado (2014).
  - Minimum Backtest Length — Bailey, Borwein, López de Prado & Zhu (2014).
  - Minimum Track Record Length — inversion of PSR.
  - Newey–West (HAC) t-stat on the daily strategy-minus-benchmark series,
    with the Andrews (1991) data-driven Bartlett bandwidth.
  - Studentized stationary block bootstrap CIs — Politis & Romano (1994),
    Ledoit & Wolf (2008), with a block length that scales as T^(1/3) and is
    capped so it can never approach T (:func:`bootstrap_block_days`).

UNITS — READ THIS BEFORE TOUCHING ANYTHING.
``engine.metrics.sharpe()`` returns an ANNUALIZED Sharpe (×√252). Every
formula in the literature above is stated in the PER-PERIOD (daily) Sharpe.
A units slip here silently corrupts every downstream number — PSR, DSR,
MinTRL and the deflation threshold all move by a factor of ~15.9. So every
name in this module's public API carries its unit: ``sharpe_periodic`` vs
``sharpe_annual``, ``se_periodic`` vs ``se_annual``. Convert with
:func:`periodic_to_annual` / :func:`annual_to_periodic` and never by hand.

DETERMINISM: the bootstrap takes an explicit integer ``seed``. There is no
default and no unseeded randomness anywhere in this module — the engine is
reproducible by design and a report that changes between two runs of the same
spec would be a lie about its own precision.

NO SCIPY: the backend does not depend on scipy, so the normal CDF (``math.erf``)
and its inverse (Acklam's rational approximation + one Halley refinement
against ``math.erfc``, giving full double precision) are implemented below.
The skew/kurtosis estimators match ``scipy.stats.skew(bias=False)`` and
``scipy.stats.kurtosis(fisher=False, bias=False)`` exactly.

REFUSAL, NOT FABRICATION: a degenerate series (all cash, a filter that never
fires), a sample too short for the resampling to have honest coverage, or a
holding period too long for the sample all produce an ``available=False``
assessment carrying an explanation — never an exception out of an inline
pipeline step, and never a confident three-state verdict. See
:func:`bootstrap_block_days` for the measured coverage this policy buys.

THE CHOKEPOINT — WHY THIS IS NO LONGER A LIST OF SITES.
Rounds 3-8 each gated the site that had been named and each left a sibling,
usually in the same function three lines below the new gate: the t-statistic was
gated and the interval and the p-value beside it were not; the verdict was gated
and the decay finding was not; a critical value was spelled 1.9599639845 in the
interval and 2.0 in the sentence about it. Enumerating the sites by hand does
not converge, because the enumeration is done by the same reasoning that missed
them last time.

So there are now four functions, at the top of this file, and NOTHING here
publishes a derived quantity, a critical value, a verdict or a sentence about one
except through them:

  :func:`critical_z`              ONE critical value. The interval, the
                                  classifier and every sentence read it, so a
                                  mutation to it moves all three together.
  :func:`classify_excess`         ONE three-state comparison. No site spells its
                                  own ``ci_low > 0`` / ``ci_high < 0`` pair, and
                                  ``excludes_zero`` is read off this rather than
                                  computed beside it.
  :func:`zero_relation_sentence`  ONE sentence about an interval's relationship
                                  to zero. Sign-blindness is unreachable: the
                                  caller cannot render the excluding branch
                                  without supplying BOTH directions.
  :func:`gate_derived`            DEFAULT-DENY over every numeric leaf and every
                                  prose leaf of every payload, at object
                                  construction (:meth:`_Gated.refuse`) and at
                                  serialization (:func:`gated_dict`). A field
                                  added tomorrow is nulled because it is not on
                                  :data:`HONEST_UNDER_REFUSAL`; a refusal
                                  sentence that formats a statistic is raised by
                                  :func:`gate_prose`.

The tests that hold this shape are ENUMERATING, not pinning: they walk the
payload structure and the emitted prose programmatically on a degenerate input,
so a new ungated field and a new leaky sentence both fail without anyone editing
a test.

THE DEGENERACY POLICY — ONE PREDICATE, EVERY SITE, EVERY LEVEL.
Degeneracy does not always arrive as a zero, and it is not always small in
absolute terms: when the daily series is a nonzero CONSTANT (a strategy whose
filter never qualifies, differenced against a flat cash rate), its dispersion
about its own mean is floating-point rounding while its mean is a perfectly
ordinary number. ``se > 0`` is True there, so the HAC t-statistic came out as
that constant divided by a rounding error — measured ``t = -1.0e16``, ``p =
0.0``, "clears the conventional |t| > 2 bar", and a zero-width bootstrap
interval that "excludes zero".

Six rounds of this module each fixed one such instance and left a sibling one
LEVEL down or one SITE over. The siblings were not bad luck; they were one
policy applied in one place. So there is now exactly ONE predicate —
:func:`degeneracy_refusal`, checking BOTH the scale-free
:data:`DISPERSION_SHARE_FLOOR` and the absolute :data:`ABSOLUTE_SD_FLOOR`,
plus the algebraic bound on the share itself — and these are all the places it
is applied. Every site below either forms or renders a t-statistic, an
interval or a verdict, and NONE of them may be looser than any other:

  WHOLE-SERIES SITES
    * :func:`_degenerate_sharpe` — the gate on :class:`ReturnMoments`, hence on
      the Sharpe SE, :func:`sharpe_difference_test`, PSR, DSR, MinTRL and
      :func:`bootstrap_sharpe_ci`.
    * :func:`newey_west_tstat` — forms the t-statistic and picks the "|t| > 2"
      sentence. Also refuses a non-finite ``t`` rather than letting it fall
      into the "not distinguishable from noise" branch.
    * :func:`bootstrap_excess_return_ci` — forms the interval, ``excludes_zero``
      and "The interval excludes zero".
    * :func:`assess_excess` — the bundled entry and the verdict.
  RESAMPLE-LEVEL SITES (a whole-series guard cannot see one resample collapse)
    * :func:`bootstrap_sharpe_ci` and :func:`bootstrap_excess_return_ci`, via
      :func:`_usable_pivots` / :func:`degenerate_rows`.
  ALGEBRAIC BOUNDS (no calibration needed, so they are simply asserted)
    * ``0 <= dispersion_share <= 1``, in :func:`degeneracy_refusal`.
    * ``ci_low <= ci_high`` and both finite, in :func:`_studentized_interval`.
    * an interval has TWO endpoints: neither pivot quantile may be rounding
      residue of the other, also in :func:`_studentized_interval`. An endpoint
      that IS the point estimate turns ``excludes_zero`` into ``sign(point)``.
    * a resample mean lies inside the hull of the data, in :func:`_usable_pivots`.
    * :func:`classify_excess` refuses a non-finite or inverted interval instead
      of classifying it as a tie.

DROPPING A DEGENERATE RESAMPLE IS BIASED, SO IT IS REFUSED, NOT DROPPED.
The resamples :func:`_usable_pivots` excludes are the ones in which the rare part
of the series did not occur — measured, 100.0% of them sit on ONE side of the
point estimate — so an interval from the survivors is biased rather than merely
wider, and above a derived share of the distribution it is not reported at all.
That is a second named predicate, with the same shape as the first (one place,
both halves, every site): :func:`survivorship_refusal`, over
:func:`max_discarded_pivot_share` (relative) and :func:`min_usable_pivots`
(absolute). It raises :class:`ResampleRefusal`, which the bundled entries CATCH:
:func:`assess_excess` falls back to the unbiased HAC interval and labels it
``verdict_source="hac"``, and :func:`assess_sharpe` prints why no resampled
interval is there. See :func:`max_discarded_pivot_share` for the fabricated
verdict this closes and the measurements behind the bar.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from dsl.conventions import TRADING_DAYS_PER_YEAR
from engine import metrics

# ---------------------------------------------------------------- constants

#: Annualization constant — dsl/conventions.py is the single source of truth
#: for it (engine.metrics reads the same symbol). Hardcoding 252 here would
#: silently desynchronize every annualized number in this module from
#: ``metrics.sharpe`` the moment the convention changed.
PERIODS_PER_YEAR = float(TRADING_DAYS_PER_YEAR)
_SQRT_PERIODS = math.sqrt(PERIODS_PER_YEAR)

#: Euler–Mascheroni constant γ, used by the expected-maximum-Sharpe formula.
EULER_MASCHERONI = 0.5772156649015329

#: Below this many observations a Sharpe carries essentially no information
#: (SE of an annualized Sharpe at T=30 daily obs is ≈ 2.3). PSR/DSR refuse to
#: report rather than print a confident-looking number — Bailey & López de
#: Prado's own guidance is that the PSR normal approximation needs T ≳ 30.
MIN_OBS_FOR_INFERENCE = 30

#: Minimum number of expected blocks a stationary-bootstrap resample must
#: contain. A block length that approaches T turns every resample into a
#: circular shift of the whole sample: θ*_b → θ̂, s*_b → ŝ, the studentized
#: pivots collapse onto 0 and the interval degenerates onto the point estimate
#: while still reporting ``excludes_zero=True`` (measured: at T=30 with
#: ``expected_block_days=1000`` the old code returned a ZERO-WIDTH interval and
#: called it a win). The block length is therefore capped at
#: ``n_obs / MIN_EFFECTIVE_BLOCKS``, and a holding period too long to fit that
#: cap is a REFUSAL, not a silently clamped block.
#:
#: The cap is a BOUNDARY, not a comfort zone: measured at T=1,500 (200 seeded
#: replications, the protocol in :func:`bootstrap_block_days`), the mean
#: interval is nearly flat from block 1 to block 21 (0.2510 → 0.2469) and then
#: shrinks — 0.2389 at the cap of 75, coverage 0.92 there against 0.925 at
#: block 1.
#:
#: The rule :func:`bootstrap_block_days` derives does NOT stay clear of this
#: cap — an earlier version of this comment claimed it "never sits near the cap
#: in absolute terms", and that is false. It sits EXACTLY ON the cap whenever
#: ⌈T^(1/3)⌉ ≥ T/20, i.e. for every T ≤ 100 at a 1-day hold (T=30 → block 1.5 =
#: cap; T=100 → block 5 = cap), and at any T once the declared holding period
#: reaches the cap (T=1,000 with hold=50 → block 50 = cap, which is the longest
#: hold :func:`bootstrap_feasibility` allows there). What is measured is that
#: those on-the-cap cells are not a cliff: coverage 0.90–0.94 at T=30, 100,
#: 420/hold=21 and 1,000/hold=50 (400 seeded replications each, nominal 0.95),
#: the same band as the cells well inside the cap. The cap stops the
#: degeneracy above; it does not buy extra coverage, and the derived rule
#: leaning on it is normal rather than exceptional.
MIN_EFFECTIVE_BLOCKS = 20

#: Displayed ceiling for "years of track record required" — printing ``inf``
#: is useless to a reader, and any number past this is the same verdict.
MAX_DISPLAY_YEARS = 50.0


# ------------------------------------------------------- normal distribution

def norm_cdf(x: float) -> float:
    """Standard normal CDF Φ(x), via the error function (exact to ~1 ulp)."""
    return 0.5 * math.erfc(-float(x) / math.sqrt(2.0))


# Acklam's rational approximation coefficients (relative error < 1.15e-9),
# refined below to full double precision with one Halley step.
_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
      3.754408661907416e+00)
_P_LOW = 0.02425


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF Φ⁻¹(p) for 0 < p < 1.

    Acklam's rational approximation followed by one Halley refinement against
    ``math.erfc``; accurate to machine precision. Exists because the backend
    has no scipy dependency and every formula in this module needs Φ⁻¹.
    """
    p = float(p)
    if not (0.0 < p < 1.0):
        raise ValueError(f"norm_ppf needs 0 < p < 1, got {p!r}")
    if p < _P_LOW:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
            ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)
    elif p <= 1.0 - _P_LOW:
        q = p - 0.5
        r = q * q
        x = (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q / \
            (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
            ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)
    # One Halley step on Φ(x) − p = 0 takes ~1e-9 to ~1e-16.
    e = norm_cdf(x) - p
    u = e * math.sqrt(2.0 * math.pi) * math.exp(x * x / 2.0)
    return float(x - u / (1.0 + x * u / 2.0))


# -------------------------------------------------------------- unit helpers

def periodic_to_annual(sharpe_periodic: float,
                       periods_per_year: float = PERIODS_PER_YEAR) -> float:
    """Per-period Sharpe → annualized Sharpe (the √252 that metrics.sharpe applies)."""
    return float(sharpe_periodic * math.sqrt(periods_per_year))


def annual_to_periodic(sharpe_annual: float,
                       periods_per_year: float = PERIODS_PER_YEAR) -> float:
    """Annualized Sharpe → per-period Sharpe. Every formula here wants this one."""
    return float(sharpe_annual / math.sqrt(periods_per_year))


def _series(returns) -> pd.Series:
    """Coerce to float Series and refuse NaN.

    Delegates to ``metrics._check`` on purpose: the whole codebase has exactly
    one NaN policy (a NaN in a return series is an upstream alignment bug, not
    something to skip past), and inference must not be the place it gets a
    second one.
    """
    s = returns if isinstance(returns, pd.Series) else pd.Series(returns, dtype=float)
    return metrics._check(s.astype(float))


def _excess_series(returns, rf_daily: float | pd.Series = 0.0) -> pd.Series:
    """``returns − rf`` on the RETURN series' own index — via ``metrics._excess``.

    ``engine.metrics.sharpe`` accepts ``rf_daily`` as a scalar or as a daily
    path, and the pipeline is expected to pass the path. This module used to
    subtract a Series without aligning, which index-UNIONS the two: a 400-day
    return series minus a 500-day rf path reported ``n_obs = 500`` with NaN
    moments, and the NaN walked past the guard into the prose ("is nan%",
    valid=True, sufficient_sample=True).

    Delegating to ``metrics._excess`` is the point: alignment and the
    rf-path-with-holes refusal are ONE policy for the whole codebase, so
    inference cannot drift away from the metric it is putting error bars on.
    """
    return metrics._excess(_series(returns), rf_daily)


def _degenerate_sharpe(excess: np.ndarray) -> bool:
    """True when ``metrics.sharpe`` would return NaN on this excess series.

    Shares ``metrics._degenerate`` AND ``metrics._implausible`` rather than
    re-testing ``sd <= 0``: an all-cash series carries ~1e-18 of floating-point
    dust, and an exact-zero test let it through as a Sharpe of 1.6e8 while
    ``metrics.sharpe`` returned NaN on the identical input (confirmed audit
    finding). One degeneracy policy, one answer.

    This is the Sharpe family's entry to :func:`degeneracy_refusal` — it CALLS
    the module predicate rather than re-deriving a second opinion, so the
    Sharpe side of the module (:func:`sharpe_standard_error`,
    :func:`sharpe_difference_test`, PSR, DSR, MinTRL,
    :func:`bootstrap_sharpe_ci` and everything :class:`ReturnMoments` gates) is
    covered by the same policy as the excess side, plus the ``_implausible``
    residue on top. Routing it through the predicate rather than duplicating
    ``sd <= 1e-7`` is the point: one policy, one place to change it.
    """
    if excess.size < 2:
        return True
    if degeneracy_refusal(excess) is not None:
        return True
    sd = float(np.std(excess, ddof=1))
    return metrics._implausible(float(excess.mean() / sd * _SQRT_PERIODS), sd)


#: Below this share of its own total sum of squares, a series' dispersion about
#: its own mean IS floating-point rounding, and every statistic of the form
#: "mean over standard error" built on it is a constant divided by a rounding
#: artefact — finite, plausible-looking and meaningless. Same failure the
#: sibling module catches with ``attribution.RESID_VAR_SHARE_FLOOR``, same
#: constant, so the codebase has ONE dust policy rather than two.
#:
#: Deliberately SCALE-FREE (a share of the series' own size), not an absolute
#: floor on the standard error: an absolute floor has to be recalibrated for
#: every return scale, and the engine's inverse-vol floor earlier in this effort
#: was set 250x above its threat and bound on a real asset class. This bar
#: cannot bind on a series merely because its numbers are small — a 1bp/day edge
#: against 1bp/day of noise has a dispersion share of ~0.5, exactly like a 1%/day
#: one.
#:
#: WHERE THE BAR SITS, by measurement (all recomputed, not quoted, by
#: test_dispersion_share_floor_sits_between_the_dust_and_the_flattest_real_series):
#:   * ARTEFACT SIDE. An exact constant's share is pure arithmetic noise: worst
#:     of 84 constant series — lengths 30…5,000, |c| from 1e-6 to 1e6, including
#:     ones built the way the pipeline builds them (``metrics._excess`` of two
#:     flat series) — is 1.65e-31. eps sits 15.1 orders of magnitude above that.
#:   * REAL SIDE. The flattest genuinely-varying series in the committed fixture
#:     is the ^IRX-derived daily risk-free PATH over 2022-12-01..2026-01-30 (793
#:     sessions): share 0.0143, i.e. 13.8 orders of magnitude above the floor.
#:     Of the 504 S&P 500 fixture columns, 503 yield a return series with a
#:     measurable share and the minimum over those is 0.956 (SNDK); the 504th
#:     (FDXF) has too few quotes in the window to form one at all, which is a
#:     coverage fact about the fixture and not a dispersion this floor refuses.
#:   * SYNTHETIC TIGHT-BUT-REAL. A genuine dispersion five orders of magnitude
#:     below its own mean (sd 1e-9 about a mean of 2e-4 — far tighter than any
#:     real series) gives 2.22e-11: 1e5x ABOVE the floor, so it stays testable.
#: The floor only bites below sd/|mean| ~ 1.5e-8.
#:
#: The share is a monotone function of the t-statistic it is guarding, which is
#: why no legitimate fit can be refused by it: at lag 0,
#: ``share = (T-1)/(t^2 + T-1)``, so at T=793 the floor corresponds to
#: |t| ~ 1.9e9, and a fit at |t| = 1e8 still sits 357x above the floor.
DISPERSION_SHARE_FLOOR = float(np.finfo(float).eps)


def dispersion_share(x) -> float:
    """Σ(xₜ − x̄)² / Σxₜ² — the share of a series' size that is dispersion.

    1.0 for a mean-zero series, ~0 for a constant one, and — the point —
    unchanged when the whole series is rescaled. Compare against
    :data:`DISPERSION_SHARE_FLOOR` BEFORE forming any mean-over-standard-error
    statistic on ``x``.

    An all-zeros series has no total to take a share of; it returns 0.0, which
    is the correct verdict (there is nothing to test) rather than a NaN that
    every ``<=`` comparison would wave through.
    """
    arr = np.asarray(x, dtype=float)
    total = float((arr * arr).sum())
    if not (total > 0.0):
        return 0.0 if math.isfinite(total) else float("nan")
    with np.errstate(invalid="ignore"):
        dev = arr - arr.mean()
        return float((dev * dev).sum() / total)


#: The ABSOLUTE half of the policy, and deliberately NOT a new number: it is
#: ``engine.metrics._MIN_DAILY_SD`` (1e-7 daily, 1.6e-6 annualized vol), the
#: constant ``metrics.sharpe`` / ``metrics.information_ratio`` already refuse
#: on. Inventing a second absolute floor here is how a codebase ends up with
#: two thresholds nine orders of magnitude apart, which is exactly the defect
#: this section exists to close.
#:
#: WHY AN ABSOLUTE HALF IS NEEDED AT ALL. The share floor is scale-free, so it
#: cannot see a denominator that is dust in the units the series is actually
#: measured in. Reachable at real ^IRX quote granularity: an all-cash strategy
#: against a cash path that moves by ONE 0.001 tick (3.97e-8/day) over 793
#: sessions has sd 1.41e-08 — under this floor — while its dispersion share is
#: 4.96e-09, seven orders ABOVE the eps share floor. The share floor passes it;
#: the t-statistic that comes out is -399,678 and "clears the conventional
#: |t| > 2 bar", and the bootstrap interval is [-1239964.96%, -5.04%],
#: "excludes zero".
#:
#: WHERE IT SITS, by measurement (recomputed by
#: ``test_the_two_halves_of_the_policy_bracket_the_dust_and_the_real_series``).
#: An earlier version of this note carried three numbers that do NOT come from
#: the construction it names — "502" series, an excess-over-rf minimum of
#: 9.94e-03 ABOVE the raw minimum (subtracting a near-constant rf path cannot
#: raise a standard deviation by 7%), and 3.18e-03 for AGG. Re-measured on the
#: committed fixtures over 2022-12-01..2026-01-30:
#:   * REAL SIDE. 503 of the 504 S&P 500 fixture columns yield a return series;
#:     their minimum daily sd is 9.331e-03 (93,308x this floor, KO), and their
#:     excess-over-rf minimum is 9.331e-03 — the SAME name, 3e-6 relative apart,
#:     because the rf path's own sd is three orders smaller. AGG, the flattest of
#:     the three extras-fixture ETFs, is 3.744e-03 (37,439x). The flattest real
#:     series anywhere in the fixtures is the ^IRX-derived rf PATH at 2.220e-05
#:     (222x), and the tightest thing that can be constructed from real quotes at
#:     all — the daily DIFFERENCE of that path — is 1.125e-06, still 11.2x above.
#:   * ARTEFACT SIDE. The band this floor exists to cover runs from 1e-7 down
#:     to the 5.09e-21 dust a constant resample produces.
#: The 11.2x margin is the honest number: this half is NOT "orders of magnitude
#: clear of anything real", it is one order clear of the flattest object the
#: fixtures can produce, and that is the price of covering the band. (This line
#: said 11.3x while the measurement three lines above it said 1.125e-06, i.e.
#: 11.25x, which rounds DOWN — a drift small enough to survive six rounds and
#: exactly the kind a comment is allowed to carry and a test is not. Both
#: numbers are now recomputed by
#: ``test_the_two_halves_of_the_policy_bracket_the_dust_and_the_real_series``.)
ABSOLUTE_SD_FLOOR = float(metrics._MIN_DAILY_SD)

#: Tolerance on the ALGEBRAIC bound ``0 <= dispersion_share <= 1``. The bound is
#: exact in real arithmetic (Σ(x−x̄)² ≤ Σx², since Σx² = Σ(x−x̄)² + T·x̄²); only
#: floating-point cancellation can push the computed ratio past it, and only by
#: a few ulps. A share outside this is not a degenerate fit, it is broken
#: arithmetic, and it is reported as such rather than compared to a floor.
_SHARE_BOUND_TOL = 1e-9

#: The RATIO-form companion of the two floors above, and DERIVED rather than
#: chosen: a standard error is the square root of a variance, so the companion of
#: a variance-SHARE floor at eps is a LENGTH-ratio floor at √eps. Two lengths
#: whose ratio is below this are not a big one and a small one — the small one is
#: rounding residue of the big one.
#:
#: IT LIVES AT ONE SITE, and that site was chosen by MEASUREMENT rather than by
#: symmetry. It used to be a third condition inside :func:`_usable_pivots`
#: (``se_b > se_point × √eps``), where it was BEHAVIOURALLY INERT: searched over
#: the whole rare-signal family — T ∈ {253, 504, 793} × {1,2,3,5,10} event days ×
#: {2%, 5%, 10%} moves, 1,000 resamples each — exactly ZERO rows were refused by
#: it that :func:`degenerate_rows` did not already refuse, so no test could show
#: it load-bearing and it was pinned only by re-asserting its own value. The
#: reason is algebraic: on the mean path the Bartlett long-run variance is
#: positive semi-definite and bottoms out near ``γ₀/T`` (a perfectly alternating
#: row at lag 1 measures a ratio of 9e-2, not 1e-8); on the Sharpe path the
#: Mertens numerator ``1 − g₃·SR + ((g₄−1)/4)·SR²`` is a quadratic in SR whose
#: corrected-moment discriminant reached only +0.147 over a two-level-row search,
#: and its smallest positive value found there was 2.5e-03 — a studentizer ratio
#: of 5e-02. So it has been MOVED to :func:`_studentized_interval`, where the
#: same derived ratio IS load-bearing and the gap is measured on both sides:
#:   * ARTEFACT SIDE. The one-event rare-signal case (793 sessions, seed 7) forms
#:     an interval whose near half-width is 3.7e-16 of its width — the endpoint
#:     IS the point estimate, to 128 ulps.
#:   * REAL SIDE. The tightest genuinely two-sided interval over 200 S&P 500
#:     fixture names plus 40 seeded normal series has a near-half-width share of
#:     0.342 (FISV).
#: √eps = 1.49e-08 sits 7.6 orders above the artefact (1.49e-08 / 3.7e-16 =
#: 4.0e7) and 7.4 orders below the tightest real interval (0.342 / 1.49e-08 =
#: 2.3e7). The two margins are NOT the same number — this note said "7.4 and
#: 7.4", which is the real-side figure printed twice — pinned by
#: ``test_a_one_sided_interval_is_refused_by_its_own_algebraic_bound``.
MAGNITUDE_DUST_RATIO = math.sqrt(DISPERSION_SHARE_FLOOR)


@dataclass(frozen=True)
class Degeneracy:
    """Why a series cannot carry a mean-over-standard-error statistic.

    ``kind`` is what is TRUE OF THE DATA, not which floor happened to fire
    first — the refusal sentence is generated from it by
    :func:`degeneracy_reason`, at every site, so a refusal can never be
    explained by the wrong story. Five kinds:

    ``"too_short"``       fewer than 2 observations.
    ``"non_finite"``      an ±inf is present, so dispersion is unmeasurable
                          (``metrics._check`` refuses NaN, not ±inf, so this
                          reaches the module).
    ``"impossible"``      the dispersion share is outside its algebraic [0, 1].
    ``"constant"``        the share is at or below :data:`DISPERSION_SHARE_FLOOR`
                          — dust over dust, at any scale.
    ``"absolute"``        the daily sd is at or below :data:`ABSOLUTE_SD_FLOOR`
                          — dust in the units the series is measured in.

    A sixth kind, ``"unmeasurable"``, is not produced by the predicate: it is
    constructed by :func:`newey_west_tstat` when a series that PASSED the
    predicate still yields a non-finite t-statistic, so that the refusal path
    and its sentence are shared rather than duplicated.
    """
    kind: str
    share: float
    sd: float
    n_obs: int


def degeneracy_refusal(x) -> Degeneracy | None:
    """THE degeneracy predicate of this module. ``None`` means testable.

    ONE predicate, BOTH halves, applied at EVERY site that forms or renders a
    t-statistic, an interval or a verdict — see the module docstring for the
    enumerated list. Six rounds of this effort each fixed one degeneracy
    instance and left a sibling at a different site or a different level; the
    siblings were not bad luck, they were one policy applied in one place.

    THE TWO HALVES, AND WHY NEITHER ALONE IS THE POLICY:

      * :data:`DISPERSION_SHARE_FLOOR` is scale-free. It catches a nonzero
        CONSTANT at any magnitude — ``se > 0`` is True on the 5e-20 of rounding
        such a series carries — but it is blind to a denominator that is small
        in ABSOLUTE terms while its share is merely small: share 4.96e-09 is
        seven orders above the floor and still yields |t| = 4.0e5.
      * :data:`ABSOLUTE_SD_FLOOR` catches that, and is blind to the constant at
        large scale, and would mis-calibrate if it were set by taste rather
        than taken from the codebase's existing one.

    THEY CANNOT DISAGREE, because the policy is their UNION and the union is
    what every site applies. What was disagreeing before was not the floors but
    the SITES: ``assess_excess`` applied both while ``newey_west_tstat`` and
    ``bootstrap_excess_return_ci`` applied only the share half, so in the band
    between them (share > 2.2e-16 with sd <= 1e-7) the bundled entry REFUSED
    while the two rendering sites FABRICATED — t = -399,678 "clears the
    conventional |t| > 2 bar", interval [-1239964.96%, -5.04%] "excludes zero".
    Reachable at real ^IRX quote granularity with a single one-tick change.

    THE ALGEBRAIC BOUND IS CHECKED FIRST, because an impossible value is the
    cheapest degeneracy detector there is and it needs no calibration at all:
    ``dispersion_share`` is a ratio of a centred sum of squares to an uncentred
    one, so it lies in [0, 1] by construction. Outside that, the arithmetic is
    broken and nothing is reported from it.

    The tests are spelled ``not (share > FLOOR)`` rather than ``share <= FLOOR``
    ON PURPOSE. ``<=`` is False for a NaN, and a NaN that fails every ``if``
    lands in the reassuring branch: measured on a 300-day series with one inf,
    this module printed "a serial-correlation-robust t-statistic of nan … That
    is short of the conventional |t| > 2 bar, so the gap is not distinguishable
    from noise" — a fabricated TIE, from a series with no statistics at all.
    """
    arr = np.asarray(x, dtype=float).ravel()
    n = int(arr.size)
    if n < 2:
        return Degeneracy("too_short", float("nan"), float("nan"), n)
    share = dispersion_share(arr)
    if not math.isfinite(share):
        return Degeneracy("non_finite", share, float("nan"), n)
    if share < -_SHARE_BOUND_TOL or share > 1.0 + _SHARE_BOUND_TOL:
        return Degeneracy("impossible", share, float("nan"), n)
    with np.errstate(invalid="ignore"):
        sd = float(np.std(arr, ddof=1))
    if not (share > DISPERSION_SHARE_FLOOR):
        return Degeneracy("constant", share, sd, n)
    if not (sd > ABSOLUTE_SD_FLOOR):
        return Degeneracy("absolute", share, sd, n)
    return None


def degenerate_rows(m: np.ndarray) -> np.ndarray:
    """:func:`degeneracy_refusal` over the ROWS of a (B, T) resample matrix.

    PRINCIPLE 3 — the policy has to hold at every LEVEL the arithmetic happens,
    and a whole-series guard cannot see one resample collapse. Measured: a
    rare-signal strategy (in cash except 2 event days) against flat cash over
    793 sessions has whole-series sd 1.51e-3 and share 0.9932, so NOTHING
    refuses it — and ~14% of its 1,000 stationary-bootstrap resamples miss the
    event days, are exactly constant, and carry ``se_b = 5.09e-21``. That is
    positive dust, so ``se_b > 0`` passes it; the pivots are ~1.5e16 and the
    reported interval was [-4.97%, +19879096992716836.00%], printed verbatim
    into the reader-facing interpretation and into the pipeline's
    ``excess_significance`` finding — and it decided the verdict, turning a
    "loses" into a "ties_within_noise".

    Same two floors as the scalar predicate, so the levels cannot drift apart;
    pinned row-by-row against it by
    ``test_the_row_predicate_is_the_same_predicate_as_the_scalar_one``.
    Returns a boolean mask, True where the row is degenerate.
    """
    a = np.asarray(m, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"degenerate_rows needs a 2-D matrix, got {a.ndim}-D")
    n = a.shape[1]
    if n < 2:
        return np.ones(a.shape[0], dtype=bool)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        total = (a * a).sum(axis=1)
        dev = a - a.mean(axis=1, keepdims=True)
        ss = (dev * dev).sum(axis=1)
        ratio = ss / total
        share = np.where(total > 0.0,
                         ratio,
                         np.where(np.isfinite(total), 0.0, np.nan))
        sd = np.sqrt(ss / (n - 1.0))
    bad_bound = (share < -_SHARE_BOUND_TOL) | (share > 1.0 + _SHARE_BOUND_TOL)
    # ``~(a > b)`` rather than ``a <= b`` so a NaN is degenerate, not waved through.
    return bad_bound | ~(share > DISPERSION_SHARE_FLOOR) | ~(sd > ABSOLUTE_SD_FLOOR)


def degeneracy_reason(deg: Degeneracy, mean: float, label: str) -> str:
    """The ONE refusal sentence, generated from what is true of the data.

    Every refusing site renders this, so the wrong-reason failure cannot recur
    one input over. It did recur: in the band where only the absolute half
    fires, ``assess_excess`` fell through to the share branch's text and told
    the reader "the strategy tracks cash exactly — the difference series has no
    dispersion, so there is nothing to test" about a series that differs from
    cash by -5.04% annualized on all 793 of its sessions, and
    ``pipeline._significance_findings`` printed that sentence verbatim. 62 of
    108 near-constant cases took that branch. The lead clause is therefore
    chosen by the MEAN, and the explanation by the KIND, and neither is chosen
    by which floor happened to fire.
    """
    if deg.kind == "too_short":
        return (f"there are fewer than 2 daily differences against {label}, so "
                f"there is no dispersion to measure")
    if deg.kind == "non_finite":
        return (f"the daily difference against {label} contains a value that is "
                f"not finite, so its dispersion cannot be measured and there is "
                f"nothing to test")
    if deg.kind == "unmeasurable":
        return (f"the serial-correlation-robust standard error of the daily "
                f"difference against {label} came out non-finite or zero, so "
                f"the statistic it would divide does not exist")
    if deg.kind == "impossible":
        return (f"the daily difference against {label} reports a dispersion "
                f"share of {deg.share:.6g}, outside the algebraically possible "
                f"[0, 1] — the arithmetic behind it is broken, so no statistic "
                f"is formed from it")
    lead = (f"the strategy tracks {label} exactly"
            if mean == 0.0 else
            f"the strategy differs from {label} by the same "
            f"{mean * PERIODS_PER_YEAR:+.2%} annualized on every one of its "
            f"{deg.n_obs:,} sessions")
    if deg.kind == "constant":
        return (f"{lead} — the difference series has no dispersion about its "
                f"own mean (dispersion share {deg.share:.1e}, at or below the "
                f"{DISPERSION_SHARE_FLOOR:.1e} floor), so there is nothing to "
                f"test it against")
    return (f"{lead} — what dispersion the difference series has is a daily "
            f"standard deviation of {deg.sd:.2e}, at or below the "
            f"{ABSOLUTE_SD_FLOOR:.1e} floor below which a standard error is a "
            f"rounding artefact rather than a measurement, so there is nothing "
            f"to test it against")


def _finite(x) -> float | None:
    """JSON-safe scalar: NaN/±inf become None rather than invalid JSON."""
    if x is None:
        return None
    f = float(x)
    return f if math.isfinite(f) else None


def _jsonable(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, float):
            out[k] = _finite(v)
        elif isinstance(v, dict):
            out[k] = _jsonable(v)
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = _finite(v) if isinstance(v, np.floating) else int(v)
        else:
            out[k] = v
    return out


# ============================================================================
# THE CHOKEPOINT
#
# Rounds 3-8 each gated the named site and each left a sibling — most often in
# the SAME FUNCTION, three lines below the gate that had just been added. Round
# 8 gated the t-statistic and missed the confidence interval and the p-value;
# the round before gated the verdict and missed the decay finding. Six rounds,
# six siblings. Enumerating the sites by hand does not converge, because the
# enumeration is done by the same reasoning that missed them last time.
#
# So the gating is no longer a list of sites. There are exactly FOUR functions
# below, and nothing in this module publishes a derived quantity, a critical
# value, a verdict or a sentence about one except through them:
#
#   :func:`critical_z`            THE critical value. One derivation, quoted by
#                                 the classifier AND by every sentence. It used
#                                 to be derived twice — the HAC interval at
#                                 Φ⁻¹(½+conf/2) = 1.9599639845 and the sentence
#                                 three lines below at a folklore 2.0 — which
#                                 disagree for every |t| in [1.95996, 2.0).
#   :func:`classify_excess`       THE three-state comparison. No site spells its
#                                 own ``> 0`` / ``< 0`` pair.
#   :func:`zero_relation_sentence` THE sentence about an interval's relationship
#                                 to zero. Sign-aware BY CONSTRUCTION: the
#                                 caller cannot render the excluding branch
#                                 without supplying BOTH directions.
#   :func:`gate_derived`          DEFAULT-DENY over every numeric leaf and every
#                                 prose leaf of every payload, applied at object
#                                 construction AND at serialization. A field
#                                 added tomorrow is nulled because it is not on
#                                 :data:`HONEST_UNDER_REFUSAL`, not because its
#                                 author remembered to null it.
#
# The tests that hold this shape are the enumerating ones: they walk the payload
# and the prose PROGRAMMATICALLY on a degenerate input rather than pinning the
# fields and sentences anyone happened to think of.
# ============================================================================

def critical_z(confidence: float) -> float:
    """THE critical value of this module: Φ⁻¹(½ + conf/2), two-sided.

    ONE critical value, derived in ONE place, used by the interval AND by the
    sentence about it. It was derived twice before, three lines apart:
    :func:`newey_west_tstat` built its interval at ``norm_ppf(0.5 +
    confidence/2)`` = 1.9599639845 and ``_bar_sentence`` compared ``|t|``
    against a hardcoded 2.0. Those disagree for every ``|t|`` in
    [1.9599639845, 2.0), and that window is REACHABLE on a whole family:
    ``_rare_signal_case(n_events=1, move=0.05, t=738)`` refuses the bootstrap,
    falls back to HAC, measures ``t = -1.9600236384`` with an interval of
    [-6.665e-02, -1.014e-06] — so ``classify_excess`` says ``loses`` while the
    sentence beside it said "That is short of the conventional |t| > 2 bar, so
    the gap is not distinguishable from noise", in the same WARNING-severity
    paragraph the pipeline renders verbatim. Also at T=739, T=740, and on the
    ``bootstrap=False`` path at T=300 (t = +1.97, verdict ``beats``).

    The fix is not a second constant set to 1.96. It is that there is one
    function, and a mutation to it moves the interval and the sentence TOGETHER
    — which is the only arrangement in which they cannot contradict each other.
    """
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence!r}")
    return norm_ppf(0.5 + confidence / 2.0)


#: The three states a benchmark comparison can honestly take.
BEATS = "beats"
TIES_WITHIN_NOISE = "ties_within_noise"
LOSES = "loses"


def classify_excess(point: float, ci_low: float, ci_high: float) -> str:
    """Three-state verdict from an interval: beats / ties-within-noise / loses.

    The whole point of this module: a strict inequality on point estimates has
    two states and both of them can be noise. An interval that straddles zero
    means the honest answer is "these are not distinguishable", which is a
    verdict the report is allowed to give.

    THE VERDICT SITE IS A RENDERING SITE, so the policy reaches it too. A NaN
    endpoint fails both ``>`` tests and lands in ``ties_within_noise`` — the
    reassuring branch, reached by a series with no statistics at all — and an
    inverted interval is broken arithmetic. Both are refused here rather than
    classified; every caller in this module gates on
    :func:`degeneracy_refusal` first, so reaching this is a bug upstream, not a
    wide answer.

    THIS IS ALSO THE ONLY PLACE an interval's relationship to zero is computed.
    ``excludes_zero``, the bootstrap sentences, the HAC critical-value sentence
    and the headline all read it from here, so there is one comparison rather
    than five that can drift apart.
    """
    if not (math.isfinite(point) and math.isfinite(ci_low)
            and math.isfinite(ci_high)):
        raise ValueError(
            f"refusing to classify a non-finite interval: point {point!r}, "
            f"[{ci_low!r}, {ci_high!r}] — a NaN endpoint silently classifies "
            f"as ties_within_noise, which is a verdict this sample cannot make"
        )
    if ci_low > ci_high:
        raise ValueError(
            f"refusing to classify an inverted interval [{ci_low!r}, "
            f"{ci_high!r}] — ci_low <= ci_high is algebraic, so this is broken "
            f"arithmetic rather than a wide answer"
        )
    if ci_low > 0.0:
        return BEATS
    if ci_high < 0.0:
        return LOSES
    return TIES_WITHIN_NOISE


def zero_relation_sentence(point: float, ci_low: float, ci_high: float, *,
                           above: str, below: str, contains: str
                           ) -> tuple[str, str]:
    """THE sentence about an interval's relationship to zero. ``(verdict, text)``.

    SIGN-BLINDNESS IS IMPOSSIBLE HERE BY CONSTRUCTION, which is the point. The
    sign-blind sentence was fixed at ONE site — :func:`bootstrap_sharpe_ci`,
    which told the reader "the risk-adjusted edge survives resampling" about an
    interval of [-1.67, -0.33] — and left at the two siblings that render the
    same claim: :func:`sharpe_standard_error` ("The interval excludes zero, so
    there is a measurable edge") and :func:`bootstrap_excess_return_ci` ("The
    interval excludes zero."). Both are one bit rendered where two are needed.

    A caller here cannot reach the excluding branch without having written BOTH
    directions, and cannot reach any branch without going through
    :func:`classify_excess` — so the prose and the verdict are the same
    comparison rather than two that agree until they do not.
    """
    verdict = classify_excess(point, ci_low, ci_high)
    where, claim = {
        BEATS: ("lies entirely above zero", above),
        LOSES: ("lies entirely below zero", below),
        TIES_WITHIN_NOISE: ("contains zero", contains),
    }[verdict]
    return verdict, f"The interval {where}, so {claim}."


#: THE ALLOW-LIST — the only numeric field names that survive a refusal, and the
#: reason :func:`gate_derived` can be default-deny. Everything here is honest
#: arithmetic REGARDLESS of degeneracy; everything not here is nulled.
#:
#: The test of membership is not "is it useful" but "would this number be a
#: claim about a sampling distribution the sample does not have". A mean is not:
#: it is what the error bars were going to be ABOUT, and it is true of the
#: sample whether or not the sample can carry error bars. A t-statistic is:
#: it exists only if the denominator does.
HONEST_UNDER_REFUSAL = frozenset({
    # --- counts and calendar facts about the sample
    "n_obs", "observed_n_obs", "observed_years", "n_resamples", "n_usable",
    "n_degenerate_resamples", "n_lo_iid_fallback_resamples", "n_trials",
    # --- configuration the CALLER supplied, echoed back so a report can show
    #     what was asked for even when the answer is a refusal
    "confidence", "alpha", "seed", "lag", "expected_block_days",
    "holding_period_days", "sr_star_periodic", "sr_star_annual",
    "target_sharpe_annual",
    # --- the point estimate, in every unit and at every site that carries one
    "point", "point_daily", "point_periodic", "mean_daily", "mean_annualized",
    "sharpe_periodic", "sharpe_annual", "difference_annual",
    "sharpe_annual_a", "sharpe_annual_b", "skew", "kurtosis_raw",
    # --- the multiple-testing budget: a function of the TRIAL LEDGER, not of
    #     the series being refused, so a degenerate series does not make the
    #     luck threshold unknowable.
    "sr0_periodic", "sr0_annual", "variance_trial_sharpes_periodic",
    "min_years_required", "expected_lucky_sharpe_annual",
    # --- refusal metadata. Nulling THESE would hide the refusal itself, which
    #     is the one thing worse than publishing the number.
    "available", "valid", "sufficient_sample", "attainable", "sufficient",
})

#: How a NESTED payload announces "I have already been through the chokepoint at
#: my own level". A refusal one level up must not null a nested result that is
#: perfectly testable on its own terms — an :class:`ExcessAssessment` that
#: withholds a VERDICT because the sample is too short still carries a
#: :class:`HACResult` whose arithmetic is real, and nulling it would be
#: over-refusal rather than honesty. What stops that from becoming a loophole is
#: that the nested payload's own prose must then make no claim it cannot support:
#: see the ``MIN_OBS_FOR_INFERENCE`` branch in :func:`newey_west_tstat`.
SELF_GATED_MARKERS = frozenset({"available", "valid", "attainable"})

#: PROSE IS GATED BY NAME-BLIND WALK, NOT BY A LIST OF FIELD NAMES. This set is
#: kept only to record what the list USED to be, and why a list was the wrong
#: shape: a leaky sentence put in a NEW field — ``method_note``, say — is not in
#: any list written before that field existed, so :func:`gate_prose` never saw
#: it and the formatted t-statistic reached the payload. Demonstrated, not
#: supposed. So :func:`_gate_leaf` now sends EVERY string leaf through the prose
#: gate, on the same default-deny principle as the numeric half: a refusal's
#: sentences are the whole product of a refusal, so they are kept and CHECKED,
#: whatever they are called.
#:
#: Non-prose strings (``method``, ``statistic``, ``kind``, ``verdict``,
#: ``verdict_source``, ``years_required_display``) pass the check trivially —
#: they are labels, not claims — and if one ever stops passing, that is the
#: signal, not a reason for an exemption list.
FORMERLY_PROSE_FIELDS = frozenset({"interpretation", "unavailable_reason",
                                   "ci_coverage_caveat",
                                   "years_required_display"})

#: A DERIVED, verdict-shaped quantity AS THIS MODULE FORMATS ONE. Deliberately
#: not "any number": a refusal sentence legitimately reports the sample's own
#: arithmetic — 793 sessions, a -5.04% annualized mean, a 5.1e-21 dispersion,
#: the algebraically possible [0, 1]. What may never appear in a refusal is a
#: formatted statistic that exists only if the fit is testable.
#:
#: This is a pattern over PRODUCED STRINGS, not a list of known sentences, so a
#: leaky sentence written tomorrow is caught without anyone extending a list.
_LEAKED_STATISTIC = re.compile(
    r"""(
        t-?statistic \s* (of|=|:)? \s* [-+(]* \d      # "t-statistic of -1.96"
      | \| \s* t \s* \|                               # any |t| bar sentence
      | \b t \s* = \s* [-+(]* \d                      # "t = -1.96"
      | \b z \s* = \s* [-+(]* \d                      # "z = 2.13"
      | p [-\s]? value \s* (of|=|:)? \s* [-+(]* \d    # "p-value of 0.03"
      | \b p \s* [=<>] \s* [-+(]* \d                  # "p < 0.05"
      | R ² \s* (of|=|:)? \s* [-+(]* \d               # "R2 of 0.41"
      | ± \s* [-+(]* \d                               # "0.42 +/- 0.11"
      | \[ [^\]]* \d [.%] [^\]]* , [^\]]* \]          # "[-6.67%, -0.00%]"
      | interval \s+ (excludes|contains|lies)         # a claim about an interval
      | critical \s+ value                            # ...or about its own bar
      | (?:PSR|DSR) \s* (?:of|is|=|:) \s* [-+(]* \d   # "DSR of 0.87"
      | probabilit (?:y|ies) \s+ (?:of|is) \s* [-+(]* \d
      # ...and any other RATIO or SHARE presented as a measurement. "dispersion
      # share" is exempt because it is the sample's own arithmetic and the
      # refusal sentence is REQUIRED to quote it — that exemption is why this is
      # a lookbehind rather than a second, weaker pattern.
      | \b (?<! dispersion\  ) (?:ratio|share) \s* (?:of|is|=|:)? \s* [-+(]* \d
    )""", re.VERBOSE | re.IGNORECASE)


def gate_prose(text: str, *, available: bool) -> str:
    """THE prose chokepoint. Pass-through when testable; a REFUSAL when not.

    PRINCIPLE C: the field gate and the prose gate have to be the same policy,
    because six rounds of this module were lost to a nulled field sitting beside
    a sentence that still printed the number. So a refusal's sentences go
    through :data:`_LEAKED_STATISTIC` — a pattern over the produced string, not
    a list of the sentences anyone remembered — and a match RAISES.

    Raising is deliberate and is not the "never raises on data" contract being
    broken: a leaky refusal template is a PROGRAMMER error (the author wrote a
    sentence that formats a statistic the sample does not have), reachable only
    from the module's own source, and the house rule for a broken template is
    the same as for a broken number — surface it, do not absorb it. The
    enumerating prose test walks every sentence the module can emit on a
    degenerate input, so this fires in a test run rather than in a report.
    """
    if available:
        return text
    hit = _LEAKED_STATISTIC.search(text or "")
    if hit is not None:
        raise ValueError(
            f"a refusal sentence formats a derived statistic "
            f"({hit.group(0)!r}) that this sample cannot support: {text!r}. "
            f"A refusal may report the sample's own arithmetic and nothing "
            f"else — see HONEST_UNDER_REFUSAL for the same rule on fields"
        )
    return text


def gate_derived(fields: dict, *, available: bool, null=None) -> dict:
    """THE field chokepoint. DEFAULT-DENY over every leaf, by comprehension.

    ``available=True`` is a pass-through: a testable fit is entitled to its
    statistics. ``available=False`` walks every leaf of ``fields`` — nested
    dicts, lists and tuples included — and

      * replaces every numeric leaf (float, int, bool, and their numpy forms)
        whose KEY is not in :data:`HONEST_UNDER_REFUSAL` with ``null``;
      * sends every STRING leaf through :func:`gate_prose`, whatever it is
        called — see :data:`FORMERLY_PROSE_FIELDS` for why a list of prose field
        names was the wrong shape;
      * leaves everything else alone (nested dataclasses have already gated
        themselves at their own level).

    WHY DEFAULT-DENY AND NOT A LIST OF DERIVED FIELDS. A deny-list is a list of
    the fields whoever wrote it thought of, which is exactly the reasoning that
    produced six siblings in six rounds. An allow-list inverts the failure mode:
    a field added tomorrow and forgotten is NULLED, which is at worst an
    over-refusal that a test will name, rather than a fabricated number a reader
    will believe. Adding a field to the allow-list is the deliberate act, and
    the enumerating test keeps its own copy of the list so that act cannot be
    done silently.

    ``null`` is ``None`` at serialization and ``float("nan")`` where the
    dataclass field is annotated non-optional and the published contract is a
    NaN (:class:`HACResult`); ``to_dict`` maps the NaN to ``None`` either way.
    """
    if available:
        return dict(fields)
    return {k: _gate_leaf(k, v, null) for k, v in fields.items()}


def _gate_leaf(key: str, value, null):
    """One leaf of :func:`gate_derived`. Recursive; no site may call it alone."""
    if isinstance(value, dict):
        if SELF_GATED_MARKERS & value.keys():
            return value            # already gated at its own level
        return {k: _gate_leaf(k, v, null) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_gate_leaf(key, v, null) for v in value)
    if isinstance(value, str):
        return gate_prose(value, available=False)
    if isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating)):
        return value if key in HONEST_UNDER_REFUSAL else null
    return value


def gated_dict(fields: dict, *, available: bool) -> dict:
    """The serialized form of the chokepoint: gate, then make JSON-safe.

    Every ``to_dict`` in this module is exactly this call. The JSON step is not
    a second policy — a non-finite float is unrepresentable in JSON and becomes
    ``None`` whatever its availability.
    """
    return _jsonable(gate_derived(fields, available=available, null=None))


class _Gated:
    """Mixin making :func:`gate_derived` the only way a payload reaches a caller.

    Two hooks, and a subclass may override only the first:

      ``payload_available``  what "this fit is testable" means for this result
                             type. Defaults to the ``available`` field.
      ``refuse``             THE constructor for a refusing payload. Every
                             ``available=False`` / ``valid=False`` /
                             ``attainable=False`` result in this module is built
                             through it, so its derived fields are nulled at the
                             OBJECT level — not only in ``to_dict`` — without
                             its author listing them.

    The nulling itself is not overridable. That is the entire point: a new
    result type is gated by inheriting, and a new field on an existing one is
    gated by not being on :data:`HONEST_UNDER_REFUSAL`.
    """

    #: What a nulled numeric becomes at OBJECT level. See :func:`gate_derived`.
    _REFUSAL_NULL: float | None = None

    @property
    def payload_available(self) -> bool:
        return bool(getattr(self, "available", True))

    @classmethod
    def refuse(cls, **fields):
        return cls(**gate_derived(fields, available=False,
                                  null=cls._REFUSAL_NULL))

    def to_dict(self) -> dict:
        return gated_dict(asdict(self), available=self.payload_available)


# ------------------------------------------------------------------ moments

def sample_skew(x: np.ndarray) -> float:
    """Bias-corrected sample skewness g₃ (adjusted Fisher–Pearson).

    Matches ``scipy.stats.skew(x, bias=False)`` and ``pandas.Series.skew()``.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    d = x - x.mean()
    m2 = float((d ** 2).mean())
    if m2 <= 0.0 or n < 3:
        return 0.0
    g1 = float((d ** 3).mean()) / m2 ** 1.5
    return float(math.sqrt(n * (n - 1.0)) / (n - 2.0) * g1)


def sample_kurtosis_raw(x: np.ndarray) -> float:
    """Bias-corrected RAW (non-excess) sample kurtosis g₄; 3.0 for a normal.

    Matches ``scipy.stats.kurtosis(x, fisher=False, bias=False)`` and
    ``pandas.Series.kurt() + 3``. The PSR/DSR formulas want RAW kurtosis —
    passing excess kurtosis here is the single easiest way to get a wrong
    probability that still looks plausible.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    d = x - x.mean()
    m2 = float((d ** 2).mean())
    if m2 <= 0.0:
        return 3.0
    raw = float((d ** 4).mean()) / m2 ** 2
    if n < 4:
        return raw
    corrected = ((n ** 2 - 1.0) * raw - 3.0 * (n - 1.0) ** 2) / ((n - 2.0) * (n - 3.0))
    return float(corrected + 3.0)


@dataclass(frozen=True)
class ReturnMoments(_Gated):
    """The four numbers every Bailey/López de Prado formula needs.

    ``sharpe_periodic`` is deliberately not called ``sharpe``: it is the
    per-period (daily) Sharpe, NOT the ×√252 number that
    ``engine.metrics.sharpe`` returns and the UI displays.
    """
    sharpe_periodic: float
    n_obs: int
    skew: float
    kurtosis_raw: float

    @property
    def sharpe_annual(self) -> float:
        return periodic_to_annual(self.sharpe_periodic)

    @classmethod
    def from_returns(cls, returns,
                     rf_daily: float | pd.Series = 0.0) -> "ReturnMoments":
        """Moments of ``returns − rf``.

        Degeneracy is judged by :func:`_degenerate_sharpe`, i.e. by the SAME
        policy ``metrics.sharpe`` uses, not by ``sd <= 0``.
        """
        excess_s = _excess_series(returns, rf_daily)
        if len(excess_s) < 2:
            raise ValueError("need at least 2 observations to estimate moments")
        excess = excess_s.to_numpy(dtype=float)
        sd = float(np.std(excess, ddof=1))
        if _degenerate_sharpe(excess):
            raise ValueError(
                "return series has zero dispersion — Sharpe is undefined, "
                "refusing to fabricate an inference on it"
            )
        return cls(
            sharpe_periodic=float(excess.mean() / sd),
            n_obs=int(excess.size),
            skew=sample_skew(excess),
            kurtosis_raw=sample_kurtosis_raw(excess),
        )

    @classmethod
    def try_from_returns(cls, returns, rf_daily: float | pd.Series = 0.0
                         ) -> "ReturnMoments | None":
        """:meth:`from_returns`, or ``None`` when the series is degenerate.

        The bundled entry points use this: an all-cash strategy is a DATA
        condition that must surface as an unavailable-with-explanation result,
        not as an exception out of an inline pipeline step (that would be a 500
        where the house rules demand a finding). NaN still raises — a NaN is an
        upstream alignment bug, and the whole codebase has one NaN policy.
        """
        try:
            return cls.from_returns(returns, rf_daily)
        except ValueError as exc:
            if "zero dispersion" in str(exc) or "at least 2 observations" in str(exc):
                return None
            raise

    def to_dict(self) -> dict:
        # ``sharpe_annual`` is a property, so the field list is spelled out —
        # and it still goes through the chokepoint like every other payload.
        return gated_dict({
            "sharpe_periodic": self.sharpe_periodic,
            "sharpe_annual": self.sharpe_annual,
            "n_obs": self.n_obs,
            "skew": self.skew,
            "kurtosis_raw": self.kurtosis_raw,
        }, available=self.payload_available)


def _psr_variance_numerator(moments: ReturnMoments) -> float:
    """1 − g₃·SR + ((g₄−1)/4)·SR² — the numerator of V[SR̂] in PSR.

    Bailey & López de Prado (2012), eq. (2). For a normal (g₃=0, g₄=3) this
    collapses to 1 + SR²/2, i.e. Lo (2002). Negative skew and fat tails both
    inflate it, which is exactly why a "great" Sharpe on a tail-selling
    strategy is less trustworthy than the same Sharpe on a symmetric one.
    """
    sr = moments.sharpe_periodic
    return 1.0 - moments.skew * sr + ((moments.kurtosis_raw - 1.0) / 4.0) * sr * sr


# ---------------------------------------------------- Sharpe standard errors

@dataclass(frozen=True)
class SharpeSE(_Gated):
    """Sharpe point estimate with its standard error and a normal CI."""
    sharpe_periodic: float
    sharpe_annual: float
    se_periodic: float
    se_annual: float
    ci_low_annual: float
    ci_high_annual: float
    n_obs: int
    confidence: float
    method: str            # "lo_iid" | "mertens_non_normal"
    interpretation: str


def sharpe_standard_error(moments: ReturnMoments, *, non_normal: bool = True,
                          confidence: float = 0.95) -> SharpeSE:
    """Standard error of the Sharpe ratio.

    Lo, A. (2002), "The Statistics of Sharpe Ratios", *Financial Analysts
    Journal* 58(4): under IID normal returns the asymptotic standard error of
    the per-period Sharpe is

        SE(SR) = sqrt( (1 + SR²/2) / T )

    Mertens, E. (2002) generalizes this to non-normal returns by replacing the
    numerator with the PSR variance term ``1 − g₃·SR + ((g₄−1)/4)·SR²``, which
    is what ``non_normal=True`` (the default) uses — the strategies this DSL
    generates (drawdown-from-high entries, RSI thresholds) produce exactly the
    negatively-skewed, fat-tailed streams where the IID form is too optimistic.

    Note PSR itself divides by (T−1) rather than T; that finite-sample
    refinement lives in :func:`probabilistic_sharpe_ratio`. The difference is
    immaterial at the sample sizes this product runs on and both are cited as
    published.

    Returns SEs and the CI in BOTH units; the annualized CI is the one the
    report shows next to the headline Sharpe.
    """
    sr = moments.sharpe_periodic
    t = moments.n_obs
    if t < 2:
        raise ValueError("Sharpe standard error needs at least 2 observations")
    numer = _psr_variance_numerator(moments) if non_normal else 1.0 + sr * sr / 2.0
    method = "mertens_non_normal" if non_normal else "lo_iid"
    if numer <= 0.0:
        # Possible only for extreme skew/Sharpe combinations; fall back to the
        # IID form and say so rather than returning a NaN standard error.
        numer = 1.0 + sr * sr / 2.0
        method = "lo_iid_fallback"
    se_p = math.sqrt(numer / t)
    se_a = periodic_to_annual(se_p)
    z = critical_z(confidence)
    lo_a, hi_a = moments.sharpe_annual - z * se_a, moments.sharpe_annual + z * se_a
    # THE CHOKEPOINT, not a local ``crosses_zero`` bit. This site used to render
    # "The interval excludes zero, so there is a measurable edge" for an
    # interval lying entirely BELOW zero — the same sign-blind sentence that was
    # fixed at bootstrap_sharpe_ci and left here. It cannot recur: the excluding
    # branch is unreachable without both directions being written.
    _, claim = zero_relation_sentence(
        moments.sharpe_annual, lo_a, hi_a,
        above=("there is a measurable risk-adjusted edge — its size, not its "
               "existence, is what the interval is uncertain about"),
        below=("what this sample measures is a risk-adjusted SHORTFALL rather "
               "than an edge: the strategy underperforms its own cash rate by "
               "more than this standard error explains"),
        contains=("this Sharpe is not statistically distinguishable from no "
                  "edge at all"))
    interp = (
        f"Annualized Sharpe {moments.sharpe_annual:.2f} ± {se_a:.2f} "
        f"(standard error), {confidence:.0%} confidence interval "
        f"[{lo_a:.2f}, {hi_a:.2f}] on {t:,} observations. " + claim
    )
    return SharpeSE(
        sharpe_periodic=sr, sharpe_annual=moments.sharpe_annual,
        se_periodic=se_p, se_annual=se_a,
        ci_low_annual=lo_a, ci_high_annual=hi_a,
        n_obs=t, confidence=confidence, method=method, interpretation=interp,
    )


def sharpe_difference_test(a: ReturnMoments, b: ReturnMoments, *,
                           non_normal: bool = True,
                           confidence: float = 0.95) -> dict:
    """Test the DIFFERENCE of two independent Sharpes, not their ratio.

    The audit's overfit detector compared in-sample Sharpe to holdout Sharpe
    against a round 2.0× ratio. Two estimates whose standard errors are ±0.65
    and ±1.00 (annualized) produce a 2× ratio by sampling noise a large
    fraction of the time. The correct comparison is on the difference with
    SE = sqrt(SE_a² + SE_b²) — valid when the two segments are disjoint, which
    an in-sample/holdout split is.

    TWO-SIDED, AND THE SENTENCE MATCHES. ``significant`` used to be
    ``z >= norm_ppf(0.95)`` — a ONE-sided test at 5% — while the sentence beside
    it read "The gap is larger than sampling noise comfortably explains" and its
    complement read "not distinguishable … in either direction", which are
    two-sided claims. A one-sided rejection region rendered as a two-sided claim
    is an error rate of 5% advertised as 2.5%, and it is directional on top:
    ``b`` beating ``a`` by four standard errors came back "not
    distinguishable". The interval is now built at :func:`critical_z` and the
    verdict read off it by :func:`classify_excess`, so the region, the interval
    and the sentence are one object. Both p-values are reported, each labelled.

    A NON-MEASURABLE ``z`` IS A REFUSAL, not the reassuring branch. With
    ``se == 0`` the old code produced ``z = nan``, ``significant = False`` and
    printed "The two are not distinguishable" — a fabricated tie out of a
    statistic that does not exist. It now goes through the chokepoint like every
    other refusal in this module.
    """
    se_a = sharpe_standard_error(a, non_normal=non_normal)
    se_b = sharpe_standard_error(b, non_normal=non_normal)
    diff = se_a.sharpe_annual - se_b.sharpe_annual
    se = math.sqrt(se_a.se_annual ** 2 + se_b.se_annual ** 2)
    z = diff / se if se > 0 else float("nan")
    base = {
        "sharpe_annual_a": se_a.sharpe_annual, "se_annual_a": se_a.se_annual,
        "sharpe_annual_b": se_b.sharpe_annual, "se_annual_b": se_b.se_annual,
        "difference_annual": diff, "se_difference_annual": se,
        "confidence": confidence,
    }
    if not math.isfinite(z):
        return gated_dict({
            **base, "z_stat": z, "ci_low_annual": float("nan"),
            "ci_high_annual": float("nan"), "p_value_two_sided": float("nan"),
            "p_value_one_sided": float("nan"), "significant_at_95": None,
            "available": False,
            "unavailable_reason": (
                "the two Sharpe standard errors sum to zero, so the difference "
                "has no sampling distribution and no statistic is formed from it"
            ),
            "interpretation": (
                f"No comparison is reported between the two Sharpes "
                f"({se_a.n_obs:,} and {se_b.n_obs:,} observations): the two "
                f"standard errors sum to zero, so the difference has no "
                f"sampling distribution. This is a refusal in both directions, "
                f"not a tie."
            ),
        }, available=False)

    crit = critical_z(confidence)
    lo, hi = diff - crit * se, diff + crit * se
    p_two = 2.0 * (1.0 - norm_cdf(abs(z)))
    p_one = 1.0 - norm_cdf(z)
    verdict, claim = zero_relation_sentence(
        diff, lo, hi,
        above="a is better than b by more than sampling noise explains",
        below="b is better than a by more than sampling noise explains",
        contains=("the two are not distinguishable — the gap is inside the "
                  "noise, so no conclusion should be drawn from it in either "
                  "direction"))
    interp = (
        f"Sharpe {se_a.sharpe_annual:.2f} ± {se_a.se_annual:.2f} versus "
        f"{se_b.sharpe_annual:.2f} ± {se_b.se_annual:.2f}: a difference of "
        f"{diff:.2f} ± {se:.2f} (z = {z:.2f} against the two-sided "
        f"{confidence:.0%} critical value of {crit:.2f}; "
        f"{confidence:.0%} interval [{lo:.2f}, {hi:.2f}]). " + claim
    )
    return gated_dict({
        **base, "z_stat": z, "ci_low_annual": lo, "ci_high_annual": hi,
        "p_value_two_sided": p_two, "p_value_one_sided": p_one,
        "significant_at_95": verdict != TIES_WITHIN_NOISE,
        "verdict": verdict, "available": True, "unavailable_reason": None,
        "interpretation": interp,
    }, available=True)


# ------------------------------------------------- Probabilistic Sharpe Ratio

@dataclass(frozen=True)
class PSRResult(_Gated):
    """PSR(SR*) — probability the TRUE Sharpe exceeds a threshold."""
    psr: float | None
    sr_star_periodic: float
    sr_star_annual: float
    sharpe_periodic: float
    sharpe_annual: float
    n_obs: int
    skew: float
    kurtosis_raw: float
    sufficient_sample: bool
    valid: bool
    interpretation: str

    #: ``valid`` is what "testable" means for a PSR: an insufficient sample or a
    #: non-positive variance term both mean the normal approximation this result
    #: rests on does not exist here.
    @property
    def payload_available(self) -> bool:
        return bool(self.valid)


def probabilistic_sharpe_ratio(moments: ReturnMoments,
                               sr_star_periodic: float = 0.0,
                               *, label: str = "zero") -> PSRResult:
    """Probabilistic Sharpe Ratio — Bailey & López de Prado (2012), eq. (2).

        PSR(SR*) = Z[ (SR̂ − SR*)·√(T−1) / √(1 − g₃·SR̂ + ((g₄−1)/4)·SR̂²) ]

    where Z is the standard normal CDF, SR̂ is the observed PER-PERIOD Sharpe,
    T the number of observations, g₃ the sample skewness and g₄ the RAW sample
    kurtosis (3 for a normal, not excess). Source:
    https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf

    Guards, both of which surface as findings rather than as silent numbers:
      * ``T < MIN_OBS_FOR_INFERENCE`` → ``psr=None``, ``sufficient_sample=False``.
        The normal approximation is not usable and printing 0.87 from 20 days
        of data would be worse than printing nothing.
      * non-positive variance term → ``valid=False``. Reachable only for
        extreme skew/Sharpe combinations, but it must not become a NaN that
        propagates into the verdict.
    """
    sr = moments.sharpe_periodic
    t = moments.n_obs
    star_annual = periodic_to_annual(sr_star_periodic)
    base = dict(
        sr_star_periodic=sr_star_periodic, sr_star_annual=star_annual,
        sharpe_periodic=sr, sharpe_annual=moments.sharpe_annual, n_obs=t,
        skew=moments.skew, kurtosis_raw=moments.kurtosis_raw,
    )
    if t < MIN_OBS_FOR_INFERENCE:
        return PSRResult.refuse(
            psr=None, sufficient_sample=False, valid=False, **base,
            interpretation=(
                f"Only {t:,} return observations — fewer than the "
                f"{MIN_OBS_FOR_INFERENCE} needed before a Sharpe probability "
                f"means anything. No probability is reported for this sample."
            ),
        )
    numer = _psr_variance_numerator(moments)
    if numer <= 0.0:
        return PSRResult.refuse(
            psr=None, sufficient_sample=True, valid=False, **base,
            interpretation=(
                f"The Sharpe variance term is non-positive at skew "
                f"{moments.skew:.2f} and kurtosis {moments.kurtosis_raw:.2f} — "
                f"the PSR approximation does not apply to this return "
                f"distribution, so no probability is reported."
            ),
        )
    z = (sr - sr_star_periodic) * math.sqrt(t - 1.0) / math.sqrt(numer)
    psr = norm_cdf(z)
    interp = (
        f"Probability that the true Sharpe exceeds {label} "
        f"(annualized {star_annual:.2f}) is {psr:.1%}, from {t:,} observations "
        f"with skew {moments.skew:.2f} and kurtosis {moments.kurtosis_raw:.2f}. "
        + ("That clears the conventional 95% bar."
           if psr >= 0.95 else
           "That is below the conventional 95% bar — on this much data, this "
           "Sharpe is not distinguishable from the threshold.")
    )
    return PSRResult(psr=psr, sufficient_sample=True, valid=True, **base,
                     interpretation=interp)


# ------------------------------------------------------ Deflated Sharpe Ratio

def expected_max_sharpe(variance_trial_sharpes_periodic: float, n_trials: int,
                        mean_trial_sharpe_periodic: float = 0.0) -> float:
    """E[max SR] over N independent trials — Bailey & López de Prado (2014), eq. (1).

        E[max{SRₙ}] ≈ E[{SRₙ}] + √V[{SRₙ}] · ( (1−γ)·Z⁻¹[1 − 1/N]
                                               + γ·Z⁻¹[1 − 1/(N·e)] )

    with γ the Euler–Mascheroni constant. This is the Sharpe you expect to see
    purely by luck after N trials, and it is the threshold the Deflated Sharpe
    Ratio measures against. All Sharpes PER-PERIOD. Source:
    https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf

    Requires ``n_trials >= 2``: Z⁻¹[1 − 1/1] = Z⁻¹[0] = −∞.
    """
    if n_trials < 2:
        raise ValueError("expected_max_sharpe needs n_trials >= 2")
    if variance_trial_sharpes_periodic < 0.0:
        raise ValueError("variance of trial Sharpes cannot be negative")
    z1 = norm_ppf(1.0 - 1.0 / n_trials)
    z2 = norm_ppf(1.0 - 1.0 / (n_trials * math.e))
    gate = (1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2
    return float(mean_trial_sharpe_periodic
                 + math.sqrt(variance_trial_sharpes_periodic) * gate)


def trial_sharpe_variance_periodic(trial_sharpes_annual) -> float:
    """Variance of a trial ledger's annualized Sharpes, converted to per-period.

    V[SR_periodic] = V[SR_annual] / 252 — the deflation formula wants
    per-period units, and handing it annualized variance inflates the
    threshold by 252×.
    """
    arr = np.asarray(list(trial_sharpes_annual), dtype=float)
    if arr.size < 2:
        raise ValueError("need at least 2 trial Sharpes to estimate a variance")
    if not np.isfinite(arr).all():
        raise ValueError("trial Sharpe ledger contains non-finite values")
    return float(np.var(arr, ddof=1) / PERIODS_PER_YEAR)


@dataclass(frozen=True)
class DSRResult(_Gated):
    """Deflated Sharpe Ratio — PSR measured against the luck threshold."""
    dsr: float | None
    n_trials: int
    sr0_periodic: float | None
    sr0_annual: float | None
    variance_trial_sharpes_periodic: float | None
    sharpe_annual: float
    n_obs: int
    available: bool
    interpretation: str
    psr: PSRResult | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["psr"] = self.psr.to_dict() if self.psr is not None else None
        return gated_dict(d, available=self.payload_available)


def deflated_sharpe_ratio(moments: ReturnMoments, *, n_trials: int,
                          variance_trial_sharpes_periodic: float | None = None,
                          trial_sharpes_annual=None) -> DSRResult:
    """Deflated Sharpe Ratio — Bailey & López de Prado (2014), *JPM* 40(5).

        SR₀ = √V[{SRₙ}] · ( (1−γ)·Z⁻¹[1 − 1/N] + γ·Z⁻¹[1 − 1/(N·e)] )
        DSR = PSR(SR₀)

    i.e. the probability that the true Sharpe beats what N trials of pure luck
    would have produced. This is the formula that prices the search: after
    N=100 trials on noise you expect a maximum Sharpe around 3.26σ, so an
    un-deflated "Sharpe 2.5" from a hundred variants is not evidence of
    anything. Source: https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf

    Requires a TRIAL LEDGER, not just a counter: V[{SRₙ}] is the variance of
    the Sharpes actually observed across trials. Pass either
    ``variance_trial_sharpes_periodic`` directly (per-period units) or
    ``trial_sharpes_annual`` (the ledger's annualized Sharpes, converted here).
    With neither, DSR is unavailable and says so — inventing a variance would
    be exactly the kind of confident fabrication this product exists to stop.

    With ``n_trials < 2`` there is nothing to deflate and DSR degenerates to
    PSR(0); that is reported explicitly rather than silently.
    """
    if variance_trial_sharpes_periodic is None and trial_sharpes_annual is not None:
        variance_trial_sharpes_periodic = trial_sharpe_variance_periodic(
            trial_sharpes_annual)

    if n_trials < 2:
        psr = probabilistic_sharpe_ratio(moments, 0.0)
        return DSRResult(
            dsr=psr.psr, n_trials=max(n_trials, 1), sr0_periodic=0.0,
            sr0_annual=0.0, variance_trial_sharpes_periodic=None,
            sharpe_annual=moments.sharpe_annual, n_obs=moments.n_obs,
            available=psr.valid, psr=psr,
            interpretation=(
                "Only one strategy was tried against this data, so there is no "
                "search to deflate: the deflated Sharpe equals the plain "
                f"probabilistic Sharpe. {psr.interpretation}"
            ),
        )

    if variance_trial_sharpes_periodic is None:
        return DSRResult.refuse(
            dsr=None, n_trials=n_trials, sr0_periodic=None, sr0_annual=None,
            variance_trial_sharpes_periodic=None,
            sharpe_annual=moments.sharpe_annual, n_obs=moments.n_obs,
            available=False, psr=None,
            interpretation=(
                f"{n_trials} strategies were tried against this data, but the "
                f"deflated Sharpe also needs the spread of those trials' "
                f"Sharpes, which was not recorded. The trial count alone "
                f"cannot set the bar, so no deflated Sharpe is reported."
            ),
        )

    sr0 = expected_max_sharpe(variance_trial_sharpes_periodic, n_trials)
    psr = probabilistic_sharpe_ratio(
        moments, sr0, label=f"the {n_trials}-trial luck threshold")
    sr0_annual = periodic_to_annual(sr0)
    if psr.psr is None:
        interp = psr.interpretation
    else:
        interp = (
            f"After {n_trials} trials on this data, luck alone is expected to "
            f"produce an annualized Sharpe of {sr0_annual:.2f}. Against that "
            f"bar — not against zero — the probability this strategy's true "
            f"Sharpe is higher is {psr.psr:.1%}. "
            + ("It clears the 95% bar even after deflation."
               if psr.psr >= 0.95 else
               f"The observed Sharpe of {moments.sharpe_annual:.2f} does not "
               f"clear it: this result is inside the range that searching "
               f"{n_trials} strategies produces from noise.")
        )
    return DSRResult(
        dsr=psr.psr, n_trials=n_trials, sr0_periodic=sr0, sr0_annual=sr0_annual,
        variance_trial_sharpes_periodic=variance_trial_sharpes_periodic,
        sharpe_annual=moments.sharpe_annual, n_obs=moments.n_obs,
        available=psr.valid, psr=psr, interpretation=interp,
    )


# ---------------------------------------------------- minimum lengths

@dataclass(frozen=True)
class MinBTLResult(_Gated):
    """How many years of data a given trial count actually needs."""
    n_trials: int
    target_sharpe_annual: float
    min_years_required: float
    observed_years: float | None
    expected_lucky_sharpe_annual: float | None
    sufficient: bool | None
    interpretation: str


def expected_lucky_sharpe_annual(n_trials: int, years: float) -> float:
    """E[max SR] in ANNUALIZED units from N trials over ``years`` of data.

    Bailey, Borwein, López de Prado & Zhu (2014), Proposition 2.1: with y
    years of data the expected in-sample maximum annualized Sharpe from pure
    luck is ≈ y^(-1/2)·[ (1−γ)Z⁻¹[1−1/N] + γ·Z⁻¹[1−1/(N·e)] ], bounded above
    by √(2·ln N). Source:
    https://www.davidhbailey.com/dhbpapers/backtest-pseudo.pdf

    This is the "noise line": a strategy whose realized Sharpe sits below it
    is indistinguishable from the best of N coin flips.
    """
    if n_trials < 2:
        return 0.0
    if years <= 0.0:
        raise ValueError("years must be positive")
    z1 = norm_ppf(1.0 - 1.0 / n_trials)
    z2 = norm_ppf(1.0 - 1.0 / (n_trials * math.e))
    gate = (1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2
    return float(gate / math.sqrt(years))


def min_backtest_length(n_trials: int, target_sharpe_annual: float = 1.0,
                        observed_years: float | None = None) -> MinBTLResult:
    """Minimum Backtest Length — Bailey, Borwein, López de Prado & Zhu (2014).

    Theorem 3.1: the backtest length in YEARS needed so that N independent
    trials with true out-of-sample Sharpe zero do not routinely produce an
    in-sample annualized Sharpe of ``target_sharpe_annual``:

        MinBTL ≈ ( [ (1−γ)·Z⁻¹[1−1/N] + γ·Z⁻¹[1−1/(N·e)] ] / E[max_N] )²
               < 2·ln(N) / E[max_N]²

    Calibration from the paper: with 5 years of data no more than ~45
    independent configurations should be tried; with 7 trials, 2 years of data
    already suffices to manufacture an in-sample Sharpe of 1 out of noise.
    Source: https://www.davidhbailey.com/dhbpapers/backtest-pseudo.pdf

    The paper's own caveat is part of the interpretation text: MinBTL is
    necessary, not sufficient — a backtest longer than MinBTL can still be
    overfit.
    """
    if target_sharpe_annual <= 0.0:
        raise ValueError("target_sharpe_annual must be positive")
    if n_trials < 2:
        return MinBTLResult(
            n_trials=max(n_trials, 1), target_sharpe_annual=target_sharpe_annual,
            min_years_required=0.0, observed_years=observed_years,
            expected_lucky_sharpe_annual=0.0, sufficient=True,
            interpretation=("A single trial needs no multiple-testing budget: "
                            "minimum backtest length does not bind here."),
        )
    z1 = norm_ppf(1.0 - 1.0 / n_trials)
    z2 = norm_ppf(1.0 - 1.0 / (n_trials * math.e))
    gate = (1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2
    years_needed = float((gate / target_sharpe_annual) ** 2)

    lucky = None
    sufficient = None
    if observed_years is not None and observed_years > 0.0:
        lucky = expected_lucky_sharpe_annual(n_trials, observed_years)
        sufficient = observed_years >= years_needed

    if observed_years is None:
        interp = (
            f"Supporting {n_trials} independent trials at an annualized Sharpe "
            f"of {target_sharpe_annual:.2f} requires about "
            f"{years_needed:.1f} years of data."
        )
    elif sufficient:
        interp = (
            f"{n_trials} trials over {observed_years:.1f} years: the minimum "
            f"backtest length for a Sharpe of {target_sharpe_annual:.2f} at "
            f"this trial count is {years_needed:.1f} years, so the sample is "
            f"long enough. Luck alone would be expected to produce a Sharpe of "
            f"{lucky:.2f} here. This is a necessary condition, not a "
            f"sufficient one — a long backtest can still be overfit."
        )
    else:
        interp = (
            f"{n_trials} trials were run on {observed_years:.1f} years of data, "
            f"but supporting that many trials at an annualized Sharpe of "
            f"{target_sharpe_annual:.2f} needs about {years_needed:.1f} years. "
            f"Over this window, searching {n_trials} strategies is expected to "
            f"produce a best-of Sharpe around {lucky:.2f} from pure noise."
        )
    return MinBTLResult(
        n_trials=n_trials, target_sharpe_annual=target_sharpe_annual,
        min_years_required=years_needed, observed_years=observed_years,
        expected_lucky_sharpe_annual=lucky, sufficient=sufficient,
        interpretation=interp,
    )


@dataclass(frozen=True)
class MinTRLResult(_Gated):
    """How long a track record must be for this Sharpe to be credible."""
    n_obs_required: float | None
    years_required: float | None
    years_required_display: str
    observed_n_obs: int
    observed_years: float
    sr_star_periodic: float
    sr_star_annual: float
    alpha: float
    attainable: bool
    interpretation: str

    #: ``attainable`` is what "testable" means for an inverted PSR: below the
    #: threshold, or with a non-positive variance term, there is no length to
    #: report and every number derived from one is nulled by the chokepoint.
    @property
    def payload_available(self) -> bool:
        return bool(self.attainable)


def min_track_record_length(moments: ReturnMoments,
                            sr_star_periodic: float = 0.0,
                            alpha: float = 0.05,
                            *, label: str = "zero") -> MinTRLResult:
    """Minimum Track Record Length — PSR inverted for T.

    Bailey & López de Prado (2012); the number of observations at which
    PSR(SR*) would reach 1−α, holding the observed Sharpe, skew and kurtosis
    fixed:

        MinTRL(SR*) = 1 + ( 1 − g₃·SR̂ + ((g₄−1)/4)·SR̂² )
                          · ( z_{1−α} / (SR̂ − SR*) )²

    Undefined (infinite) when SR̂ ≤ SR*: no amount of data makes a Sharpe
    credibly exceed a bar it does not currently reach. Also refused, with
    ``attainable=False``, when the variance term above is non-positive — the
    same condition :func:`probabilistic_sharpe_ratio` reports as
    ``valid=False``, since this is that formula inverted. Result is in
    observations at the sampling frequency; ``years_required`` divides by 252.
    Reference derivation:
    https://portfoliooptimizer.io/blog/the-probabilistic-sharpe-ratio-bias-adjustment-confidence-intervals-hypothesis-testing-and-minimum-track-record-length/

    It answers the user's actual next question — "so how much data would I
    need for this to be credible?" — and makes the cost of negative skew
    visceral: two strategies with the same Sharpe can differ threefold in
    MinTRL purely because one sells tails.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    sr = moments.sharpe_periodic
    observed_years = moments.n_obs / PERIODS_PER_YEAR
    star_annual = periodic_to_annual(sr_star_periodic)
    if sr <= sr_star_periodic:
        return MinTRLResult.refuse(
            n_obs_required=None, years_required=None,
            years_required_display="never at this Sharpe",
            observed_n_obs=moments.n_obs, observed_years=observed_years,
            sr_star_periodic=sr_star_periodic, sr_star_annual=star_annual,
            alpha=alpha, attainable=False,
            interpretation=(
                f"The observed annualized Sharpe of {moments.sharpe_annual:.2f} "
                f"is at or below {label} (annualized {star_annual:.2f}). More "
                f"data cannot make it credibly exceed a bar it does not reach — "
                f"there is no track record length that fixes this."
            ),
        )
    numer = _psr_variance_numerator(moments)
    if numer <= 0.0:
        # Same refusal :func:`probabilistic_sharpe_ratio` already makes on this
        # condition. Without it the negative variance term went straight into
        # ``1 + numer·(z/ΔSR)²`` and produced a NEGATIVE required length
        # reported as attainable — the inversion cannot be valid where the PSR
        # it inverts is not.
        return MinTRLResult.refuse(
            n_obs_required=None, years_required=None,
            years_required_display="not defined for this distribution",
            observed_n_obs=moments.n_obs, observed_years=observed_years,
            sr_star_periodic=sr_star_periodic, sr_star_annual=star_annual,
            alpha=alpha, attainable=False,
            interpretation=(
                f"The Sharpe variance term is non-positive at skew "
                f"{moments.skew:.2f} and kurtosis {moments.kurtosis_raw:.2f} — "
                f"the PSR inversion this length comes from does not apply to "
                f"this return distribution, so no required track record length "
                f"is reported."
            ),
        )
    z = norm_ppf(1.0 - alpha)
    n_needed = 1.0 + numer * (z / (sr - sr_star_periodic)) ** 2
    years = n_needed / PERIODS_PER_YEAR
    display = (f">{MAX_DISPLAY_YEARS:.0f} years" if years > MAX_DISPLAY_YEARS
               else f"{years:.1f} years")
    have_enough = moments.n_obs >= n_needed
    interp = (
        f"At this Sharpe, skew and kurtosis, {display} of daily data would be "
        f"needed for {(1 - alpha):.0%} confidence that the true Sharpe exceeds "
        f"{label} (annualized {star_annual:.2f}). This backtest covers "
        f"{observed_years:.1f} years. "
        + ("The record is already long enough."
           if have_enough else
           f"It is {(years - observed_years):.1f} years short.")
    )
    return MinTRLResult(
        n_obs_required=n_needed, years_required=years,
        years_required_display=display, observed_n_obs=moments.n_obs,
        observed_years=observed_years, sr_star_periodic=sr_star_periodic,
        sr_star_annual=star_annual, alpha=alpha, attainable=True,
        interpretation=interp,
    )


# ------------------------------------------------- Newey-West / HAC inference

def newey_west_lag(n_obs: int, holding_period_days: int = 1) -> int:
    """HAC truncation lag: the automatic rule, floored by the holding period.

    The automatic bandwidth ``int(4·(T/100)^(2/9))`` (Newey & West 1994) is the
    conventional default. It is floored by ``holding_period_days`` because this
    engine's cohort mode holds each book for ``hold_days``, mechanically
    inducing serial correlation out to that lag — callers should pass
    ``max(hold_days, rebalance interval in days)``. Taking the max is the
    conservative choice: a longer lag widens the standard error.
    """
    auto = int(4.0 * (max(n_obs, 1) / 100.0) ** (2.0 / 9.0))
    lag = max(auto, int(holding_period_days))
    return int(max(0, min(lag, max(n_obs - 2, 0))))


def andrews_bartlett_lag(x: np.ndarray, holding_period_days: int = 1) -> int:
    """Andrews (1991) AR(1) plug-in bandwidth for the Bartlett kernel.

    Andrews, D. (1991), "Heteroskedasticity and Autocorrelation Consistent
    Covariance Matrix Estimation", *Econometrica* 59(3), §6:

        α(1) = 4ρ̂² / ((1−ρ̂)²(1+ρ̂)²),   S_T = 1.1447 · (α(1)·T)^(1/3)

    with ρ̂ the fitted lag-1 autoregressive coefficient of the series. The
    T-only rule in :func:`newey_west_lag` cannot see dependence: on an AR(1)
    with ρ=0.8 it picks 6 lags at T=1,260 and the long-run variance comes out
    far too small, which is why the excess-return interval measured only 88.5%
    coverage there. This rule picks ~33 lags on the same series and lifts it to
    ~92%. On IID data α(1) → 0 and the bandwidth collapses to ~0, so nothing is
    widened that does not need widening.

    Floored by ``holding_period_days`` (the engine's cohort holds induce
    dependence out to that lag by construction) and clamped to T−2.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    floor = max(int(holding_period_days), 0)
    if n < 3:
        return int(max(0, min(floor, max(n - 2, 0))))
    d = x - x.mean()
    denom = float((d[:-1] * d[:-1]).sum())
    rho = float((d[:-1] * d[1:]).sum() / denom) if denom > 0.0 else 0.0
    # The bandwidth explodes as |ρ| → 1; cap the fitted root well inside the
    # unit circle rather than emitting an absurd lag from a near-unit-root fit.
    rho = min(max(rho, -0.97), 0.97)
    alpha = 4.0 * rho * rho / ((1.0 - rho) ** 2 * (1.0 + rho) ** 2)
    s_t = 1.1447 * (alpha * n) ** (1.0 / 3.0) if alpha > 0.0 else 0.0
    return int(max(0, min(max(int(round(s_t)), floor), max(n - 2, 0))))


def hac_lag(x: np.ndarray, holding_period_days: int = 1) -> int:
    """The truncation lag actually used: the LONGER of the two rules.

    ``max(newey_west_lag(T, hold), andrews_bartlett_lag(x, hold))``. Taking the
    max is the conservative direction — a longer lag can only widen the
    standard error — and it means this never returns a shorter lag than the
    published T-only rule that shipped before.
    """
    x = np.asarray(x, dtype=float)
    n = int(x.size)
    return int(max(newey_west_lag(n, holding_period_days),
                   andrews_bartlett_lag(x, holding_period_days)))


def _hac_long_run_variance(x: np.ndarray, lag: int) -> float:
    """Newey–West long-run variance with the Bartlett kernel.

        S = γ₀ + 2·Σ_{k=1..L} (1 − k/(L+1))·γ_k,   γ_k = Σ d_t d_{t−k} / (T−1)

    Bartlett weights guarantee a non-negative estimate, which is why the
    kernel is used rather than a plain truncated sum.

    The autocovariances divide by (T−1), not T: one degree of freedom goes to
    the estimated mean, and dividing by T biased the standard error low by
    √((T−1)/T) — 1.7% at T=30, always in the anti-conservative direction
    (confirmed audit finding). The studentized bootstrap is unaffected either
    way (the factor cancels in the pivot), but the HAC t-stat and its interval
    are reported on their own and must not be flattered.
    """
    n = x.size
    d = x - x.mean()
    denom = float(n - 1) if n > 1 else 1.0
    s = float((d * d).sum() / denom)
    for k in range(1, lag + 1):
        if k >= n:
            break
        gamma = float((d[:-k] * d[k:]).sum() / denom)
        s += 2.0 * (1.0 - k / (lag + 1.0)) * gamma
    return max(s, 0.0)


def _hac_long_run_variance_matrix(x: np.ndarray, lag: int) -> np.ndarray:
    """Vectorized :func:`_hac_long_run_variance` over rows of a (B, T) matrix."""
    n = x.shape[1]
    d = x - x.mean(axis=1, keepdims=True)
    denom = float(n - 1) if n > 1 else 1.0
    s = (d * d).sum(axis=1) / denom
    for k in range(1, lag + 1):
        if k >= n:
            break
        gamma = (d[:, :-k] * d[:, k:]).sum(axis=1) / denom
        s = s + 2.0 * (1.0 - k / (lag + 1.0)) * gamma
    return np.maximum(s, 0.0)


@dataclass(frozen=True)
class HACResult(_Gated):
    """Newey–West t-stat on a daily excess-return series."""
    #: This is the one payload whose published contract is a NaN rather than a
    #: ``None`` at object level: the fields are annotated ``float`` and every
    #: consumer already branches on ``available``. ``to_dict`` still serializes
    #: them as ``None``.
    _REFUSAL_NULL = float("nan")
    mean_daily: float
    mean_annualized: float
    se_daily: float
    se_annualized: float
    t_stat: float
    p_value_two_sided: float
    ci_low_annualized: float
    ci_high_annualized: float
    n_obs: int
    lag: int
    autocorr_lag1: float
    confidence: float
    interpretation: str
    #: False on a series whose dispersion is rounding dust: ``t_stat``,
    #: ``p_value_two_sided``, the standard errors and the interval are all NaN
    #: and ``unavailable_reason`` explains why. Consumers MUST branch on this —
    #: every ``if abs(t) >= 2`` test is False for NaN, so an unchecked NaN
    #: routes into the "short of the bar" branch, which is a verdict.
    available: bool = True
    unavailable_reason: str | None = None


def _bar_sentence(tstat: float, mean_annualized: float, ci_low: float,
                  ci_high: float, confidence: float) -> str:
    """The critical-value sentence — from THE interval, at THE critical value.

    TWO CRITICAL VALUES, THREE LINES APART, is what used to be here. The HAC
    interval was built at ``norm_ppf(0.5 + confidence/2)`` = 1.9599639845 and
    this function compared ``|t|`` against a hardcoded 2.0. They disagree for
    every ``|t|`` in [1.9599639845, 2.0), and round 8 made that window reachable
    on a whole family by routing the rare-signal case through HAC:
    ``_rare_signal_case(n_events=1, move=0.05, t=738)`` measures
    ``t = -1.9600236384`` with an interval of [-6.665e-02, -1.014e-06], so
    ``classify_excess`` returned ``loses`` and this sentence said "That is short
    of the conventional |t| > 2 bar, so the gap is not distinguishable from
    noise" — in one WARNING-severity paragraph titled "Trails DIA by more than
    noise". Also at T=739, T=740 and on the ``bootstrap=False`` path at T=300.

    THE FIX IS STRUCTURAL, not a 1.96 literal. There is now exactly ONE
    comparison: the claim comes from :func:`classify_excess` on the interval the
    verdict is read off, and the number quoted beside it comes from
    :func:`critical_z`. The two cannot disagree because there is no second test
    to disagree with — and a mutation to ``critical_z`` moves the interval, the
    classification and this sentence together.

    (``|t| >= z`` and "the interval excludes zero" are the same statement for a
    symmetric interval built at the same ``z``; they can differ by at most a
    rounding ulp exactly at the boundary, and the INTERVAL is the authority
    because it is what the verdict is read from.)

    A non-finite ``t`` is refused rather than described. ``abs(t) >= 2.0`` was
    False for a NaN, so an unguarded NaN picked the "short of the bar, not
    distinguishable from noise" branch, which is a fabricated TIE; ±inf
    satisfies any bar honestly and is still nothing to say "clears" about.
    """
    if not math.isfinite(tstat):
        raise ValueError(
            f"refusing to pick a critical-value sentence for t = {tstat!r}: a "
            f"statistic that is not finite is neither at nor under any bar, so "
            f"any sentence about the bar would be fabricated"
        )
    z = critical_z(confidence)
    verdict, claim = zero_relation_sentence(
        mean_annualized, ci_low, ci_high,
        above="this sample separates the two in the strategy's favour",
        below="this sample separates the two against the strategy",
        contains="the gap is not distinguishable from noise")
    reached = ("does not reach" if verdict == TIES_WITHIN_NOISE
               else "reaches")
    return (f"Measured against the two-sided {confidence:.0%} critical value "
            f"of {z:.2f}, the statistic {reached} it: "
            f"{claim[0].lower()}{claim[1:]}")


def _unavailable_hac(mean: float, n_obs: int, lag: int, confidence: float,
                     deg: Degeneracy, label: str) -> HACResult:
    """A HAC result that REFUSES the t-statistic instead of fabricating one.

    Both directions are refused. A fabricated ``t = -1e16`` reads as "loses" and
    a NaN ``t`` falls through ``abs(t) >= 2`` into "not distinguishable from
    noise" — a false tie is as wrong as a false win, and this sample supports
    neither. Nothing that depends on the collapsed denominator is reported, and
    on a non-finite series not even the mean is formatted: "+inf% annualized" is
    a data problem printed as a performance figure.

    The sentence comes from :func:`degeneracy_reason`, shared with
    :func:`assess_excess` and :func:`bootstrap_excess_return_ci`, so all three
    sites tell the reader the same true story about the same input.

    NOTHING HERE IS NULLED BY HAND. The fields are handed to
    :meth:`HACResult.refuse`, i.e. to :func:`gate_derived`, which nulls every
    numeric leaf that is not on :data:`HONEST_UNDER_REFUSAL`. That is the whole
    difference from round 8, which nulled the t-statistic on this line and left
    the interval and the p-value three lines below: a field added to
    :class:`HACResult` tomorrow is nulled here whether or not anyone edits this
    function.
    """
    nan = float("nan")
    reason = degeneracy_reason(deg, mean, label)
    if deg.kind in ("non_finite", "impossible", "unmeasurable"):
        interp = (
            f"No t-statistic, no p-value and no interval are reported for the "
            f"daily difference series against {label} ({n_obs:,} sessions): "
            f"{reason}. This is a data problem to fix upstream, not a verdict "
            f"in either direction."
        )
    else:
        interp = (
            f"No t-statistic, no p-value and no interval are reported against "
            f"{label}: {reason}. A series whose dispersion this sample cannot "
            f"measure has no sampling distribution, so a "
            f"serial-correlation-robust standard error on it is a rounding "
            f"artefact. This is a refusal in BOTH directions — not a win, and "
            f"not a tie either. Whichever series is larger, this sample cannot "
            f"say whether the difference is real."
        )
    return HACResult.refuse(
        mean_daily=mean, mean_annualized=mean * PERIODS_PER_YEAR,
        se_daily=nan, se_annualized=nan, t_stat=nan, p_value_two_sided=nan,
        ci_low_annualized=nan, ci_high_annualized=nan, n_obs=n_obs, lag=lag,
        autocorr_lag1=nan, confidence=confidence,
        available=False, unavailable_reason=reason, interpretation=interp,
    )


def newey_west_tstat(excess_returns, *, holding_period_days: int = 1,
                     lag: int | None = None, confidence: float = 0.95,
                     label: str = "the benchmark") -> HACResult:
    """HAC t-statistic on the daily (strategy − benchmark) series.

    Newey, W. & West, K. (1987), "A Simple, Positive Semi-Definite,
    Heteroskedasticity and Autocorrelation Consistent Covariance Matrix",
    *Econometrica* 55(3). The mean daily excess return is tested against zero
    with a Bartlett-kernel long-run variance:

        SE(mean) = sqrt( S / T ),   t = mean / SE(mean)

    The IID standard error is wrong here by construction: positions persist
    across days (cohorts hold for ``hold_days``, rebalances are monthly), so
    the daily excess series is autocorrelated and an IID SE is too narrow —
    i.e. too flattering to the strategy. ``lag`` defaults to
    :func:`newey_west_lag`.

    ``mean_annualized`` is the arithmetic 252× scaling of the mean daily
    excess, not a compounded return — it is a test statistic in readable units,
    not a performance figure.

    REFUSES (``available=False``, every statistic NaN) on
    :func:`degeneracy_refusal` — BOTH halves of it. The gate is HERE, at the
    site that formats the t-statistic and picks the "|t| > 2" sentence, not
    only in the bundled entry point: ``se > 0`` cannot see its own denominator
    collapse, and on a constant series this function returned ``t = -1.0e16``
    with ``p = 0.0`` and the verdict "That clears the conventional |t| > 2
    bar." It applied only the share half until the absolute half joined it, and
    in the band between them it rendered ``t = -399678.09`` — "That clears the
    conventional |t| > 2 bar" — on an input the bundled entry point was
    refusing at the same moment. No bandwidth is fitted on a refused series
    either — the Andrews plug-in's ρ̂ would itself be dust over dust — so
    ``lag`` reports the caller's explicit lag or the holding-period floor, not
    a fitted number.

    The non-finite ``t`` is ALSO a refusal, not a sentence: ``abs(t) >= 2.0``
    is False for a NaN, so an unguarded NaN picks the "short of the bar, not
    distinguishable from noise" branch, which is a fabricated TIE.
    """
    x = _series(excess_returns).to_numpy(dtype=float)
    t = x.size
    if t < 3:
        raise ValueError("Newey–West needs at least 3 observations")
    fallback_lag = (int(lag) if lag is not None
                    else max(int(holding_period_days), 0))
    deg = degeneracy_refusal(x)
    if deg is not None:
        return _unavailable_hac(float(x.mean()), t, fallback_lag, confidence,
                                deg, label)
    if lag is None:
        lag = hac_lag(x, holding_period_days)
    s = _hac_long_run_variance(x, lag)
    se = math.sqrt(s / t)
    mean = float(x.mean())
    tstat = mean / se if se > 0 else float("nan")
    if not math.isfinite(tstat):
        return _unavailable_hac(
            mean, t, lag, confidence,
            Degeneracy("unmeasurable", dispersion_share(x), se, t), label)
    p = 2.0 * (1.0 - norm_cdf(abs(tstat)))
    z = critical_z(confidence)
    lo = (mean - z * se) * PERIODS_PER_YEAR
    hi = (mean + z * se) * PERIODS_PER_YEAR
    d = x - x.mean()
    ac1 = float((d[:-1] * d[1:]).sum() / (d * d).sum()) if t > 2 and (d * d).sum() > 0 else 0.0
    interp = (
        f"Average daily excess return over {label} is "
        f"{mean * PERIODS_PER_YEAR:+.2%} annualized, with a "
        f"serial-correlation-robust t-statistic of {tstat:.2f} "
        f"(Newey–West, {lag} lags, {t:,} days; "
        f"{confidence:.0%} interval {lo:+.2%} to {hi:+.2%} annualized). "
        # BELOW THE INFERENCE MINIMUM THE NUMBERS ARE ARITHMETIC AND THE CLAIM IS
        # WITHHELD. This function is called on short samples by design —
        # ``assess_excess`` attaches it "for the record" alongside a refusal —
        # and it used to attach the critical-value sentence too, so a result
        # whose own headline read "No verdict is reported … it is a refusal"
        # carried "the interval contains zero, so the gap is not distinguishable
        # from noise" three lines below. That is the SAME self-contradicting
        # paragraph as the folklore-2.0 one, from a different cause, and no
        # gating of FIELDS could have caught it.
        + (_bar_sentence(tstat, mean * PERIODS_PER_YEAR, lo, hi, confidence)
           if t >= MIN_OBS_FOR_INFERENCE else
           f"No claim is made about where that statistic sits relative to any "
           f"critical value: on fewer than {MIN_OBS_FOR_INFERENCE} observations "
           f"the normal critical values this interval uses have no honest "
           f"coverage — measured, they call a true tie a win or a loss 26.8% of "
           f"the time at 5 observations and 9.4% at 29, against a nominal 5%. "
           f"The numbers above are the sample's own arithmetic, not evidence.")
    )
    return HACResult(
        mean_daily=mean, mean_annualized=mean * PERIODS_PER_YEAR,
        se_daily=se, se_annualized=se * PERIODS_PER_YEAR, t_stat=tstat,
        p_value_two_sided=p, ci_low_annualized=lo, ci_high_annualized=hi,
        n_obs=t, lag=lag, autocorr_lag1=ac1, confidence=confidence,
        interpretation=interp,
    )


# ------------------------------------------- stationary block bootstrap

def max_block_days(n_obs: int) -> float:
    """The longest expected block this sample can honestly carry.

    ``n_obs / MIN_EFFECTIVE_BLOCKS`` — see :data:`MIN_EFFECTIVE_BLOCKS` for the
    degeneracy it prevents. Never below 1 (one observation is always a legal
    block).
    """
    return max(1.0, float(n_obs) / float(MIN_EFFECTIVE_BLOCKS))


def bootstrap_block_days(n_obs: int, holding_period_days: int = 1) -> float:
    """Expected stationary-bootstrap block length, in trading days.

        block = min( max(holding_period_days, ⌈T^(1/3)⌉),  T / 20 )

    Two literatures meet here. Politis & White (2004) give the T^(1/3) rate for
    the stationary bootstrap's optimal expected block length, so the block MUST
    scale with the sample — the old fixed 10 days was simultaneously far too
    long at T=30 (three blocks in a resample) and too short at T=2,520.
    Separately, the engine's cohort mode holds each book for ``hold_days``,
    which induces dependence out to that lag by construction, so the block is
    floored by the holding period — this is the parameter the integration note
    always claimed sized the blocks and, until now, did not.

    MEASURED COVERAGE of :func:`bootstrap_excess_return_ci` under this policy
    together with the Andrews bandwidth of :func:`hac_lag`. Protocol: nominal
    95%, a series whose TRUE mean is zero, 400 seeded replications per cell at
    200 resamples each; row A uses series seeds ``20_000+i`` and bootstrap seeds
    ``90_000+i``, row B the same offsets shifted by +500,000, because one seed
    set is only worth about ±0.02 here:

        T:          30    60   126   252   504
        seed set A .932  .935  .920  .900  .917
        seed set B .917  .920  .927  .935  .927

    Under AR(1) dependence, with the holding period declared honestly (same
    protocol, both seed sets): ρ=0.5/hold=5 → .907/.932 at T=252 and .922/.927
    at T=504; ρ=0.8/hold=5 → .915/.917 at T=504; ρ=0.9/hold=20 → .902/.900 at
    T=504. Shorter samples than that are REFUSED by
    :func:`bootstrap_feasibility` rather than reported.

    So the interval is LIBERAL in every cell of that protocol — worst .900,
    best .935, call it 90–94% rather than 95%, and read any single cell as
    ±0.02 (a 200-replication run of the same T=30 seeds reads .945). An earlier
    version of this table advertised ".955 at T=30" and a
    ".932–.948" row; those numbers do not reproduce on either seed set above
    and have been replaced by what does. (The fixed-10-day, T-only-lag policy
    that shipped before measured .857–.943 on the original reviewer's own
    seeds; that code is gone, so the row is not reproducible from this tree and
    is not restated as if it were.) The user-facing wording of this shortfall is
    :data:`CI_COVERAGE_CAVEAT_BOOTSTRAP`, carried as a field on every
    :class:`ExcessAssessment` verdict so a reader of the report sees it too.
    """
    auto = math.ceil(max(int(n_obs), 1) ** (1.0 / 3.0))
    want = float(max(int(holding_period_days), 1, auto))
    return min(want, max_block_days(n_obs))


def bootstrap_feasibility(n_obs: int, holding_period_days: int = 1
                          ) -> str | None:
    """``None`` if the bootstrap can run honestly here, else the reason it cannot.

    STRUCTURAL only — it looks at T and the declared holding period, never at
    the realized variance of this particular sample. Gating on an ESTIMATED
    effective sample size was measured and rejected: it accepts exactly the
    draws whose long-run variance came out small, and conditional coverage of
    the survivors was WORSE than the unconditional coverage it was meant to
    rescue (IID T=30: .915 unconditional → .838 among the accepted). A gate
    that selects on the statistic it is protecting is not a gate.
    """
    t = int(n_obs)
    hold = max(int(holding_period_days), 1)
    if t < MIN_OBS_FOR_INFERENCE:
        return (f"only {t:,} overlapping observations — fewer than the "
                f"{MIN_OBS_FOR_INFERENCE} below which a resampled confidence "
                f"interval has no honest coverage")
    if hold > max_block_days(t):
        return (f"a {hold}-day holding period leaves about {t / hold:.0f} "
                f"independent holds in {t:,} days, and at least "
                f"{MIN_EFFECTIVE_BLOCKS} are needed before resampling can "
                f"separate an edge from serial correlation")
    return None


def stationary_bootstrap_indices(n_obs: int, n_resamples: int,
                                 expected_block_days: float, *,
                                 seed: int) -> np.ndarray:
    """Resampling indices for the stationary bootstrap of Politis & Romano (1994).

    Blocks have geometrically distributed length: index 0 of each resample is
    drawn uniformly from 0..T−1, and thereafter the index continues the
    previous block with probability Q = 1 − 1/E[block length] (wrapping around
    the end of the sample), else jumps to a fresh uniform draw. Expected block
    length is 1/(1−Q).

    Unlike the fixed-block bootstrap this keeps the resampled series
    stationary, which matters because the statistic is a Sharpe — a ratio of
    moments that a non-stationary resampling scheme biases.

    ``seed`` is mandatory: the engine is reproducible by design, and a
    confidence interval that moves between two runs of the same spec would
    misrepresent its own precision.

    ``expected_block_days`` is validated at BOTH ends: ≥ 1, and no longer than
    :func:`max_block_days`. Only the lower bound existed before, and the
    unbounded upper end was the confirmed hole — see :data:`MIN_EFFECTIVE_BLOCKS`.
    Use :func:`bootstrap_block_days` to derive a legal one.
    """
    if n_obs < 2:
        raise ValueError("bootstrap needs at least 2 observations")
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1")
    if expected_block_days < 1.0:
        raise ValueError("expected_block_days must be >= 1")
    cap = max_block_days(n_obs)
    if expected_block_days > cap:
        # Blocks approaching T make every resample a circular shift of the
        # sample: the studentized pivots collapse to 0 and the interval
        # degenerates onto the point estimate while still claiming to exclude
        # zero. Refuse the parameter instead of quietly returning a confident
        # zero-width interval.
        raise ValueError(
            f"expected_block_days={expected_block_days:g} leaves fewer than "
            f"{MIN_EFFECTIVE_BLOCKS} blocks in {n_obs} observations "
            f"(cap {cap:g}); a block that long collapses the bootstrap onto "
            f"the point estimate"
        )
    rng = np.random.default_rng(seed)
    q_continue = 1.0 - 1.0 / float(expected_block_days)
    idx = np.empty((n_resamples, n_obs), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n_obs, n_resamples)
    cont = rng.random((n_resamples, n_obs)) < q_continue
    fresh = rng.integers(0, n_obs, (n_resamples, n_obs))
    for t in range(1, n_obs):
        idx[:, t] = np.where(cont[:, t], (idx[:, t - 1] + 1) % n_obs, fresh[:, t])
    return idx


@dataclass(frozen=True)
class BootstrapCI(_Gated):
    """A studentized stationary-bootstrap confidence interval."""
    statistic: str                 # "sharpe_annual" | "excess_return_annualized"
    point: float
    ci_low: float
    ci_high: float
    confidence: float
    n_resamples: int
    n_usable: int
    expected_block_days: float
    seed: int
    n_obs: int
    excludes_zero: bool
    interpretation: str
    extra: dict = field(default_factory=dict)


def _sharpe_and_se_matrix(x: np.ndarray
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-period Sharpe and its plug-in SE for each row of a (B, T) matrix.

    The in-resample standard error is the PSR/Mertens plug-in
    ``sqrt((1 − g₃·SR + ((g₄−1)/4)·SR²)/(T−1))`` — the same variance term as
    :func:`sharpe_standard_error`, so the studentization is consistent with the
    parametric interval reported beside it.

    A NON-POSITIVE NUMERATOR FALLS BACK TO LO'S IID FORM, PER ROW, exactly as
    :func:`sharpe_standard_error` and :func:`bootstrap_sharpe_ci` already do at
    the whole-series level. It used to return NaN for such a row, which dropped
    it — and dropping is the biased operation this round exists to remove.
    Measured on the committed two-point construction that reaches this branch (60
    sessions on ``SR = 2/ĝ₃``, the series behind
    ``test_bootstrap_sharpe_ci_labels_the_iid_fallback_instead_of_clamping``):
    33.8% of 400 resamples had a non-positive numerator, so 33.8% of the pivot
    distribution was being discarded on a branch the scalar site handles by
    falling back and LABELLING it. One rule, every level.

    Returns ``(sharpe, se, fell_back)``; the mask is carried out so the count can
    be reported rather than absorbed.
    """
    n = x.shape[1]
    mean = x.mean(axis=1)
    d = x - mean[:, None]
    m2 = (d ** 2).mean(axis=1)
    m3 = (d ** 3).mean(axis=1)
    m4 = (d ** 4).mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sd = np.sqrt(m2 * n / (n - 1.0))
        sr = np.where(sd > 0, mean / sd, np.nan)
        g1 = np.where(m2 > 0, m3 / m2 ** 1.5, 0.0)
        g2 = np.where(m2 > 0, m4 / m2 ** 2, 3.0)
    if n > 2:
        g1 = math.sqrt(n * (n - 1.0)) / (n - 2.0) * g1
    if n > 3:
        g2 = ((n ** 2 - 1.0) * g2 - 3.0 * (n - 1.0) ** 2) / ((n - 2.0) * (n - 3.0)) + 3.0
    numer = 1.0 - g1 * sr + ((g2 - 1.0) / 4.0) * sr ** 2
    fell_back = np.isfinite(numer) & ~(numer > 0.0)
    with np.errstate(invalid="ignore"):
        lo_iid = 1.0 + sr ** 2 / 2.0
        se = np.sqrt(np.where(fell_back, lo_iid, numer) / (n - 1.0))
    return sr, se, fell_back


class ResampleRefusal(ValueError):
    """The bootstrap refuses this sample: its pivot set is not a smaller sample.

    Separate from the plain ``ValueError`` the module raises for a programmer
    error (an oversized block, a wrong index matrix) because the two want
    different handling: a programmer error must surface, while THIS one has an
    unbiased fallback — :func:`newey_west_tstat` — and the bundled entries take
    it and LABEL it (``verdict_source="hac"``), rather than either crashing or
    silently keeping the biased number.
    """


def max_discarded_pivot_share(confidence: float) -> float:
    """The RELATIVE half of the resample-survivorship policy: ``(1−conf)/2``.

    DERIVED, not chosen, and the derivation is the whole argument for refusing
    instead of dropping. The endpoint of a studentized interval is the α/2
    quantile of the pivot distribution. The resamples :func:`_usable_pivots`
    removes are NOT a random thinning: on a rare-signal series they are exactly
    the resamples that missed the rare day, so their mean sits on ONE side of the
    point estimate. Measured on the committed fixture window (793 sessions, seed
    7, 1,000 resamples): 100.0% of the discarded resamples have a mean BELOW the
    point estimate, at every event count tested (1, 2, 3 and 5 events, i.e.
    discarded shares of 38.2%, 14.3%, 4.2% and 0.3%). One-sidedness does not
    decay as the share shrinks — only the size of the damage does.

    So if a share ``d`` of the distribution is removed and all of it lies below
    the point, the survivors' α/2 quantile is the FULL distribution's
    ``d + (1−d)·α/2`` quantile: the lower endpoint is computed at an effective
    tail probability inflated by up to ``d``. Requiring ``d < α/2`` bounds that
    inflation by α/2, i.e. the one-sided error rate at this endpoint can at most
    double, from α/2 to α — which keeps the truncation's contribution inside the
    liberality :data:`CI_COVERAGE_CAVEAT_BOOTSTRAP` already discloses (a measured
    6–10% two-sided error rate against a nominal 5%). Above that share the
    interval is a different question than the one asked and is refused.

    THE BAR IS CONSERVATIVE, AND FREE. It sits far below where the fabrication
    becomes visible — the endpoint does not collapse onto the point estimate
    until the discarded share reaches ~34% — and far above anything real: over
    120 S&P 500 fixture names differenced against the ^IRX rf path, plus that rf
    path itself and AGG, the discarded count is ZERO out of 400 at every one. So
    nothing in the fixtures is refused by it and no threshold here has to be
    defended on a real asset class.

    WHY A BIASED SUBSAMPLE IS NOT A SMALLER SAMPLE — the regression this exists
    to close. An earlier round DROPPED the degenerate resamples and reported a
    narrower interval from the survivors. Measured at one event day in 504
    sessions: 33.9% of resamples are exactly constant and were dropped; of the
    survivors, 57% have the identical multiset as the sample so their pivot is
    rounding noise; the empirical 2.5% pivot quantile landed inside that atom at
    -5.5e-16, and ``_studentized_interval`` returned ``ci_high == point`` to 128
    ulps. ``excludes_zero`` then degenerated to ``sign(point)`` and the report
    said "The strategy loses to cash by more than sampling noise explains …
    t-statistic of -0.01 … That is short of the conventional |t| > 2 bar" in one
    paragraph. Stable across seeds 1/2/3/5/7/11/42/99 and sign-symmetric.
    """
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence!r}")
    return (1.0 - confidence) / 2.0


def min_usable_pivots(confidence: float) -> int:
    """The ABSOLUTE half of the same policy: ``1 + ceil(2/(1−conf))`` resamples.

    PRINCIPLE 1 — the relative half above cannot see a pivot set that is small
    in absolute terms, and this half cannot see a big one that is biased, so the
    policy is their UNION and :func:`survivorship_refusal` is the only place
    either is spelled. They are not allowed to disagree because neither is ever
    consulted alone.

    Also DERIVED rather than chosen, and STRICTER than the bare ``< 20`` literal
    it replaced: the α/2 quantile has to be MEASURED rather than extrapolated
    from the smallest pivot drawn.

    OFF BY ONE, NOW FIXED, and the off-by-one was in the derivation rather than
    in the arithmetic. The old form solved ``n·α/2 ≥ 1`` and returned 40 at the
    95% default. But the quantile is taken by ``np.quantile`` with its default
    linear method, whose α/2 quantile sits at fractional INDEX ``(n−1)·α/2``, not
    at ``n·α/2``: at n=40 that index is 0.975, i.e. still between the smallest
    and second-smallest pivot, which is precisely the "read off the smallest
    pivot drawn" failure the bar exists to prevent. The condition is therefore
    ``(n−1)·α/2 ≥ 1``, i.e. ``n ≥ 1 + 2/α`` — 41 at the default, and the
    endpoint is then the second-smallest pivot or an interpolation above it.

    It binds only on a tiny ``n_resamples``: at the module's 1,000 default the
    relative half already forces at least 975 survivors.
    """
    ratio = 1.0 / max_discarded_pivot_share(confidence)
    # ``1 - 0.90`` is 0.09999999999999998 in binary, so 2/α comes out as
    # 20.000000000000004 and a bare ``ceil`` returns 21 — an off-by-one from
    # REPRESENTATION on top of the one in the derivation. Snap to the integer
    # when the ratio is within a few ulps of it; the tolerance is far below the
    # 1-resample granularity of the answer, so it cannot change a real case.
    return 1 + int(math.ceil(ratio - 1e-9))


def survivorship_refusal(n_usable: int, n_dropped: int, n_resamples: int,
                         confidence: float) -> str | None:
    """THE resample-survivorship predicate. ``None`` means the pivot set stands.

    One named predicate, BOTH halves (:func:`max_discarded_pivot_share` and
    :func:`min_usable_pivots`), applied at BOTH bootstrap sites — the same shape
    as :func:`degeneracy_refusal` one level up. What was here before was a pure
    absolute floor (``< 20 usable``) with no relative companion, which is exactly
    the asymmetry principle 1 forbids: at 1,000 resamples it permitted 98% of the
    distribution to be discarded, one-sidedly, and still rendered a verdict.
    """
    share = float(n_dropped) / float(n_resamples) if n_resamples > 0 else 1.0
    cap = max_discarded_pivot_share(confidence)
    if share > cap:
        return (f"{n_dropped:,} of {n_resamples:,} resamples ({share:.1%}) carry "
                f"no usable studentizer, above the {cap:.1%} bar. Those are not "
                f"a random thinning — they are the resamples in which the rare "
                f"part of the series did not occur, so they all sit on one side "
                f"of the point estimate and an interval from the survivors is "
                f"BIASED rather than merely wider. A biased subsample is not a "
                f"smaller sample, so no resampled interval is reported")
    floor = min_usable_pivots(confidence)
    if n_usable < floor:
        return (f"only {n_usable:,} of {n_resamples:,} resamples carry a usable "
                f"studentizer, fewer than the {floor} at which the lowest "
                f"{cap:.1%} of the pivot distribution reaches past the smallest "
                f"pivot drawn, so the endpoint would be extrapolated from that "
                f"one draw rather than measured")
    return None


def _usable_pivots(boot: np.ndarray, stat_b: np.ndarray, se_b: np.ndarray,
                   point_of: str) -> tuple[np.ndarray, int]:
    """The mask of resamples whose pivot is a statistic rather than an artefact.

    THE SAME POLICY, ONE LEVEL DOWN. ``se_b > 0`` is the ``se > 0`` shape the
    whole-series sites were fixed for, living inside the resampling loop where
    a whole-series guard cannot reach it. Two conditions, and the first is
    :func:`degeneracy_refusal` verbatim on the resample itself:

      1. the resample is not degenerate by either half of the policy
         (:func:`degenerate_rows`);
      2. the statistic and its studentizer are finite and the studentizer is
         positive.

    There used to be a third — the studentizer not being dust relative to the
    point estimate's own standard error — and it was removed because it was
    BEHAVIOURALLY INERT: over the whole rare-signal family it refused not one row
    that :func:`degenerate_rows` did not already refuse, and both paths bound the
    ratio away from dust algebraically. See :data:`MAGNITUDE_DUST_RATIO` for the
    measurements and for the site the derived ratio moved TO, which is strictly
    more general: a bound on the INTERVAL catches any pivot-set pathology that
    collapses an endpoint, whichever row produced it.

    ALGEBRAIC BOUND, checked here because it costs nothing: every resample of a
    mean is an average of observed values, so it cannot leave ``[min x, max x]``.
    A resample mean outside that hull means the index matrix is wrong, and that
    is raised rather than quietly widening an interval.

    Returns ``(mask, n_dropped)``, and ``n_dropped`` is EVERY row the mask
    excludes, so ``n_usable + n_dropped == n_resamples`` identically. It used to
    count only the rows that were finite-but-degenerate, which made it ZERO on
    the Sharpe path of the rare-signal case — 382 of 1,000 resamples were
    dropped, ``n_usable`` said 618, ``n_degenerate_resamples`` said 0 and the
    caveat sentence never printed. A count that disagrees with the mask it
    describes is an absorbed data problem wearing the label of a surfaced one.
    """
    if point_of == "mean":
        lo_hull, hi_hull = float(np.min(boot)), float(np.max(boot))
        if np.isfinite(lo_hull) and np.isfinite(hi_hull):
            out = stat_b[np.isfinite(stat_b)]
            if out.size and (out.min() < lo_hull - 1e-12 * max(1.0, abs(lo_hull))
                             or out.max() > hi_hull + 1e-12 * max(1.0, abs(hi_hull))):
                raise ValueError(
                    f"a bootstrap resample mean fell outside the algebraically "
                    f"possible hull of the data [{lo_hull!r}, {hi_hull!r}] — "
                    f"the resampling index matrix is wrong"
                )
    degenerate = degenerate_rows(boot)
    finite = np.isfinite(stat_b) & np.isfinite(se_b) & (se_b > 0)
    ok = finite & ~degenerate
    return ok, int((~ok).sum())


def _degenerate_resample_note(n_dropped: int, n_resamples: int,
                              confidence: float) -> str:
    """Surfaced, never absorbed — the honest caveat on a thinned pivot set.

    Only reachable BELOW :func:`max_discarded_pivot_share`; above it the interval
    is refused outright rather than qualified, because the survivors are a biased
    subsample and no sentence makes a biased number readable.
    """
    if n_dropped <= 0:
        return ""
    return (f"{n_dropped:,} of {n_resamples:,} resamples "
            f"({n_dropped / n_resamples:.1%}) were discarded because their own "
            f"studentizer is not a usable standard error — a constant or "
            f"near-constant resample carries rounding dust rather than "
            f"dispersion. Those resamples are the ones in which the rare part of "
            f"the series did not occur, so they sit on one side of the point "
            f"estimate and what is left is a biased subsample, not a smaller "
            f"one. It is under the "
            f"{max_discarded_pivot_share(confidence):.1%} share above which this "
            f"interval is refused outright, but read it as a lower bound on the "
            f"uncertainty rather than a measurement of it.")


def _studentized_interval(point: float, se_point: float,
                          pivots: np.ndarray, confidence: float
                          ) -> tuple[float, float, bool]:
    """Ledoit & Wolf (2008) studentized interval from bootstrap pivots.

    Returns ``(ci_low, ci_high, pivots_straddle_zero)``. The third value is
    carried OUT rather than absorbed: see the note on the two quantile signs
    below.

        CI = [ θ̂ − ŝ·q_{1−α/2},  θ̂ − ŝ·q_{α/2} ]

    where q are empirical quantiles of d*_b = (θ*_b − θ̂)/s*_b. The reversed
    quantile order is not a typo — the pivot distribution is being inverted.

    TWO ALGEBRAIC BOUNDS, asserted rather than assumed.

    FIRST: ``q_lo <= q_hi`` and ``se_point >= 0``, so ``ci_low <= ci_high``
    identically. An inverted or non-finite interval is broken arithmetic, not a
    wide answer, and it is raised rather than handed to ``classify_excess`` —
    where ``ci_low > 0`` is False for a NaN and the reader would be told the two
    are "not distinguishable".

    SECOND (PRINCIPLE 4, and the one that catches the fabricated verdict): an
    interval has TWO endpoints, and each is ``se_point`` times a quantile of the
    same pivot sample. If one of those two quantiles is rounding residue of the
    other, that endpoint is not an endpoint — it IS the point estimate, and
    ``excludes_zero`` degenerates to ``sign(point)``, so the direction of the
    verdict is decided by the sign of a number the interval was supposed to put
    error bars on. Measured, on one event day in 504 sessions: ``q_lo`` = -5.5e-16
    against ``q_hi`` ≈ 2, ``ci_high − point`` = 6.9e-18 against a point of
    -3.5e-02, and the report said "loses … t-statistic of -0.01 … short of the
    conventional |t| > 2 bar" in one paragraph. The mechanism is an ATOM at zero:
    a mass of resamples reproduces the sample's own multiset, so their pivot is
    pure rounding and the α/2 quantile lands inside the atom.

    EQUALITY ALONE IS NOT THE TEST, which is worth saying because it looks like it
    should be: 6.9e-18 added to -3.5e-02 is 128 ulps, perfectly representable, and
    ``ci_high > point`` was True throughout. The bound is therefore stated as a
    dust bound on ``min(|q_lo|, |q_hi|) / max(|q_lo|, |q_hi|)``, which needs no
    calibration of its own — it reuses the module's derived
    :data:`MAGNITUDE_DUST_RATIO`.

    AND IT IS STATED ON THE QUANTILES, NOT ON THE HALF-WIDTHS ABOUT ``point``,
    because the stronger form — ``ci_low < point < ci_high``, i.e. ``q_lo < 0 <
    q_hi`` — was written first and MEASURED TO OVER-CORRECT. Take the committed
    non-positive-Mertens-numerator construction (a two-point law placed exactly on
    ``SR = 2/ĝ₃``) at 120 sessions: every one of its pivots is POSITIVE, so its
    interval sits entirely below its own point estimate, and the strong form
    refuses it. Its two quantiles are nonetheless both real — ``q_lo`` = 2.1e-03
    against ``q_hi`` = 2.69, a ratio of 7.8e-04 — so the form used here accepts it.
    A pivot distribution that never straddles zero is its own thing to be uneasy
    about; it is not the defect this bound is for, and conflating them would cost
    a labelled branch for nothing.

    SO IT IS SURFACED INSTEAD OF REFUSED, which is the open item this closes. An
    interval whose two pivot quantiles share a sign lies entirely on one side of
    its own point estimate, and an ``excludes_zero`` read off it was being
    published with nothing said about that. ``pivots_straddle_zero`` is returned
    to the caller, carried in ``extra`` and, when False, given its own sentence
    beside the verdict. Surfaced, not absorbed, and not over-corrected into a
    refusal the measurements above say it must not be.

    THE MARGINS, all recomputed: 2.0e-16 for the artefact (the atom case above),
    7.8e-04 for that two-point construction at 120 sessions, and ~1 for an ordinary
    series. √eps = 1.49e-08 sits eight orders above the first and nearly five below
    the second. The same construction at 60 sessions is REFUSED, and rightly: 4.0%
    of its resamples reproduce the sample's own multiset exactly, so its α/2
    quantile is not near zero but IS zero, to the last bit.

    This refusal is a :class:`ResampleRefusal`: the HAC path is unbiased on these
    inputs, so the bundled entries fall back to it and LABEL the source rather
    than crashing.
    """
    alpha = 1.0 - confidence
    q_lo = float(np.quantile(pivots, alpha / 2.0))
    q_hi = float(np.quantile(pivots, 1.0 - alpha / 2.0))
    lo, hi = point - se_point * q_hi, point - se_point * q_lo
    if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
        raise ValueError(
            f"studentized interval violates its own algebraic bound: "
            f"[{lo!r}, {hi!r}] from point {point!r}, se {se_point!r}, "
            f"pivot quantiles [{q_lo!r}, {q_hi!r}]"
        )
    small, large = min(abs(q_lo), abs(q_hi)), max(abs(q_lo), abs(q_hi))
    if not (small > MAGNITUDE_DUST_RATIO * large):
        raise ResampleRefusal(
            f"the resampled interval [{lo:.6g}, {hi:.6g}] has only one endpoint: "
            f"its pivot quantiles are [{q_lo:.3g}, {q_hi:.3g}], and {small:.3g} "
            f"is rounding residue of {large:.3g} rather than a second quantile, "
            f"so that endpoint is the point estimate {point:.6g} itself. An "
            f"excludes-zero claim would then carry nothing but the sign of the "
            f"point estimate, so no resampled interval is reported"
        )
    return lo, hi, bool(q_lo < 0.0 < q_hi)


#: The disclosure for an interval whose pivot distribution never straddles zero.
#: Deterministic template text, and a FIELD-plus-sentence rather than a comment,
#: because "the interval excludes zero" read off such a pivot set is a weaker
#: claim than it looks: every resample fell on one side of the point estimate, so
#: the interval is displaced rather than centred and its coverage is not the
#: measured band :data:`CI_COVERAGE_CAVEAT_BOOTSTRAP` describes.
ONE_SIDED_PIVOTS_CAVEAT = (
    "One-sided resamples: every resampled value of this statistic fell on the "
    "same side of the point estimate, so both ends of this interval are "
    "displaced in one direction rather than straddling the estimate. Its "
    "relationship to zero is still reported, but read it as a statement about a "
    "displaced interval, not a centred one."
)


def bootstrap_sharpe_ci(returns, *, seed: int,
                        rf_daily: float | pd.Series = 0.0,
                        holding_period_days: int = 1,
                        n_resamples: int = 1000,
                        expected_block_days: float | None = None,
                        confidence: float = 0.95) -> BootstrapCI:
    """Studentized stationary-bootstrap CI on the ANNUALIZED Sharpe.

    Ledoit, O. & Wolf, M. (2008), "Robust performance hypothesis testing with
    the Sharpe ratio", *Journal of Empirical Finance* 15(5), combined with the
    stationary bootstrap of Politis & Romano (1994). The studentized bootstrap
    has better finite-sample coverage than HAC asymptotics, which Ledoit & Wolf
    note "is often liberal when sample sizes are small to moderate". Source:
    https://www.econ.uzh.ch/dam/jcr:ffffffff-961c-1dd9-0000-000044710066/RobustVariance.pdf

    The interval is reported in annualized units because that is the unit the
    report displays; internally everything is per-period.

    ``expected_block_days=None`` (the default) derives the block length from T
    and the holding period via :func:`bootstrap_block_days`; an explicit value
    is validated against :func:`max_block_days`.
    """
    excess_s = _excess_series(returns, rf_daily)
    x = excess_s.to_numpy(dtype=float)
    t = x.size
    reason = bootstrap_feasibility(t, holding_period_days)
    if reason is not None:
        raise ValueError(f"bootstrap cannot run on this sample: {reason}")
    if expected_block_days is None:
        expected_block_days = bootstrap_block_days(t, holding_period_days)
    moments = ReturnMoments.from_returns(excess_s)
    # Non-positive Mertens numerator: fall back to Lo's IID form and LABEL it,
    # exactly as sharpe_standard_error does. The old code clamped it to 1e-12,
    # which manufactured a near-zero-width interval that necessarily reported
    # excludes_zero=True — a refusal condition turned into a confident claim.
    numer = _psr_variance_numerator(moments)
    se_method = "mertens_non_normal"
    if numer <= 0.0:
        numer = 1.0 + moments.sharpe_periodic ** 2 / 2.0
        se_method = "lo_iid_fallback"
    se_point = math.sqrt(numer / (t - 1.0))

    idx = stationary_bootstrap_indices(t, n_resamples, expected_block_days, seed=seed)
    boot = x[idx]
    sr_b, se_b, fell_back = _sharpe_and_se_matrix(boot)
    ok, n_dropped = _usable_pivots(boot, sr_b, se_b, "sharpe")
    refusal = survivorship_refusal(int(ok.sum()), n_dropped, n_resamples,
                                   confidence)
    if refusal is not None:
        raise ResampleRefusal(
            f"stationary bootstrap cannot produce an honest Sharpe interval "
            f"here: {refusal}")
    pivots = (sr_b[ok] - moments.sharpe_periodic) / se_b[ok]
    lo_p, hi_p, straddles = _studentized_interval(
        moments.sharpe_periodic, se_point, pivots, confidence)
    lo, hi = periodic_to_annual(lo_p), periodic_to_annual(hi_p)
    note = _degenerate_resample_note(n_dropped, n_resamples, confidence)
    # SIGN-AWARE THROUGH THE CHOKEPOINT, not by a local three-branch ``if``.
    # This site was the one place the sign blindness was fixed three rounds ago
    # and the fix did not travel; it now shares one function with every other
    # site, and ``excludes_zero`` is read off the same classification rather than
    # being a second comparison beside it.
    verdict, verdict_sentence = zero_relation_sentence(
        moments.sharpe_annual, lo, hi,
        above="the risk-adjusted edge survives resampling",
        below=("what survives resampling is a risk-adjusted SHORTFALL, not an "
               "edge: this strategy underperforms its own cash rate by more "
               "than resampling explains"),
        contains=("after allowing for serial correlation and fat tails, this "
                  "Sharpe is not distinguishable from no edge"))
    excludes_zero = verdict != TIES_WITHIN_NOISE
    interp = (
        f"Bootstrap {confidence:.0%} confidence interval for the annualized "
        f"Sharpe is [{lo:.2f}, {hi:.2f}] around a point estimate of "
        f"{moments.sharpe_annual:.2f} "
        f"({int(ok.sum()):,} usable resamples, expected block "
        f"{expected_block_days:.0f} days, seed {seed}). "
        + verdict_sentence
        + ("" if straddles else f" {ONE_SIDED_PIVOTS_CAVEAT}")
        + (f" {note}" if note else "")
    )
    return BootstrapCI(
        statistic="sharpe_annual", point=moments.sharpe_annual,
        ci_low=lo, ci_high=hi, confidence=confidence,
        n_resamples=n_resamples, n_usable=int(ok.sum()),
        expected_block_days=expected_block_days, seed=seed, n_obs=t,
        excludes_zero=excludes_zero, interpretation=interp,
        extra={"point_periodic": moments.sharpe_periodic,
               "ci_low_periodic": lo_p, "ci_high_periodic": hi_p,
               "se_periodic": se_point, "se_method": se_method,
               "pivots_straddle_zero": straddles,
               "n_degenerate_resamples": n_dropped,
               # Surfaced, not absorbed: how many resamples were studentized by
               # Lo's IID form because their own Mertens numerator was
               # non-positive. Dropping them instead is a biased thinning.
               "n_lo_iid_fallback_resamples": int((fell_back & ok).sum()),
               "holding_period_days": int(holding_period_days)},
    )


def bootstrap_excess_return_ci(excess_returns, *, seed: int,
                               holding_period_days: int = 1,
                               n_resamples: int = 1000,
                               expected_block_days: float | None = None,
                               confidence: float = 0.95,
                               label: str = "the benchmark") -> BootstrapCI:
    """Studentized stationary-bootstrap CI on the mean daily excess return.

    Same machinery as :func:`bootstrap_sharpe_ci` (Politis & Romano 1994 +
    Ledoit & Wolf 2008), with the statistic being the mean of the daily
    strategy-minus-benchmark series and the in-resample studentizer being the
    Newey–West long-run standard error at the same lag. This is the interval
    that lets "beats the benchmark" become a three-state answer instead of a
    strict inequality on two point estimates.

    Reported in arithmetic annualized units (mean × 252) so it can sit next to
    the excess-return headline; the per-period values are in ``extra``.

    ``holding_period_days`` does two jobs here, and BOTH are real: it floors
    the HAC lag used to studentize, and it floors the expected block length
    (:func:`bootstrap_block_days`). It used to do only the first while the
    integration note claimed it did both.

    Coverage is T- and dependence-dependent — see :func:`bootstrap_block_days`
    for the measured table. It is liberal: 0.90–0.94 at a nominal 0.95 in every
    cell measured there, never the single "93.3%" figure the module once
    advertised from one T.
    """
    x = _series(excess_returns).to_numpy(dtype=float)
    t = x.size
    reason = bootstrap_feasibility(t, holding_period_days)
    if reason is not None:
        raise ValueError(f"bootstrap cannot run on this sample: {reason}")
    if expected_block_days is None:
        expected_block_days = bootstrap_block_days(t, holding_period_days)
    # The same refusal :func:`newey_west_tstat` makes, at the second site that
    # renders a claim from this denominator, and now with BOTH halves of the
    # policy rather than only the share half. ``se_point <= 0`` below is not
    # enough: on a nonzero CONSTANT series the long-run variance is rounding
    # dust rather than zero, every resample is the same constant, and the
    # studentized interval came out as the ZERO-WIDTH [-5.04%, -5.04%] with
    # ``excludes_zero=True``. With only the share half, the band underneath it
    # rendered [-1239964.96%, -5.04%] and "The interval excludes zero" on an
    # input ``assess_excess`` was refusing at the same moment.
    deg = degeneracy_refusal(x)
    if deg is not None:
        raise ValueError(
            f"excess series cannot support a studentized interval: "
            + degeneracy_reason(deg, float(x.mean()), label)
        )
    lag = hac_lag(x, holding_period_days)
    point = float(x.mean())
    se_point = math.sqrt(_hac_long_run_variance(x, lag) / t)
    if se_point <= 0.0:
        raise ValueError("excess series has zero long-run variance")

    idx = stationary_bootstrap_indices(t, n_resamples, expected_block_days, seed=seed)
    boot = x[idx]
    means = boot.mean(axis=1)
    se_b = np.sqrt(_hac_long_run_variance_matrix(boot, lag) / t)
    ok, n_dropped = _usable_pivots(boot, means, se_b, "mean")
    refusal = survivorship_refusal(int(ok.sum()), n_dropped, n_resamples,
                                   confidence)
    if refusal is not None:
        raise ResampleRefusal(
            f"stationary bootstrap cannot produce an honest excess-return "
            f"interval against {label}: {refusal}")
    pivots = (means[ok] - point) / se_b[ok]
    lo_p, hi_p, straddles = _studentized_interval(point, se_point, pivots,
                                                 confidence)
    lo, hi = lo_p * PERIODS_PER_YEAR, hi_p * PERIODS_PER_YEAR
    note = _degenerate_resample_note(n_dropped, n_resamples, confidence)
    # THE SIBLING OF THE SIGN-BLIND SENTENCE. "The interval excludes zero." is
    # one bit rendered where two are needed: it said the same thing about an
    # interval entirely above zero and one entirely below it, and the reader had
    # to infer the direction from a number two sentences earlier. Same chokepoint
    # as every other site now, so the direction cannot go missing again.
    verdict, verdict_sentence = zero_relation_sentence(
        point * PERIODS_PER_YEAR, lo, hi,
        above=f"the strategy's margin over {label} survives resampling",
        below=(f"what survives resampling is a SHORTFALL against {label}, not a "
               f"margin: the strategy trails it by more than resampling "
               f"explains"),
        contains=(f"on this sample the strategy and {label} are not "
                  f"distinguishable"))
    excludes_zero = verdict != TIES_WITHIN_NOISE
    interp = (
        f"Bootstrap {confidence:.0%} confidence interval for the annualized "
        f"excess return over {label} is [{lo:+.2%}, {hi:+.2%}] around "
        f"{point * PERIODS_PER_YEAR:+.2%} "
        f"({int(ok.sum()):,} usable resamples, expected block "
        f"{expected_block_days:.0f} days, seed {seed}). "
        + verdict_sentence
        + ("" if straddles else f" {ONE_SIDED_PIVOTS_CAVEAT}")
        + (f" {note}" if note else "")
    )
    return BootstrapCI(
        statistic="excess_return_annualized", point=point * PERIODS_PER_YEAR,
        ci_low=lo, ci_high=hi, confidence=confidence, n_resamples=n_resamples,
        n_usable=int(ok.sum()), expected_block_days=expected_block_days,
        seed=seed, n_obs=t, excludes_zero=excludes_zero, interpretation=interp,
        extra={"point_daily": point, "ci_low_daily": lo_p,
               "ci_high_daily": hi_p, "se_daily": se_point, "lag": lag,
               "pivots_straddle_zero": straddles,
               "n_degenerate_resamples": n_dropped,
               "holding_period_days": int(holding_period_days)},
    )


# ---------------------------------------------------- three-state comparisons
#
# :data:`BEATS` / :data:`TIES_WITHIN_NOISE` / :data:`LOSES` and
# :func:`classify_excess` live in THE CHOKEPOINT section at the top of this
# module, because every sentence in it needs them: a three-state constant defined
# below the functions that render three-state prose is how a site ends up with
# its own private ``lo > 0.0`` pair.

#: The residual liberality of the interval the verdict rests on, in the
#: READER's words. It used to live only in the docstrings of this module, which
#: no user ever sees: the report printed a three-state verdict whose interval is
#: measurably narrower than 95% coverage and said nothing about it. Like a data
#: problem becoming a finding, this is carried out to the caller as a FIELD on
#: :class:`ExcessAssessment` rather than absorbed — the verdict is not weakened,
#: its uncertainty is made visible. Deterministic template text; the numbers in
#: it are the measured table in :func:`bootstrap_block_days`.
CI_COVERAGE_CAVEAT_BOOTSTRAP = (
    "Coverage caveat: this verdict is nothing more than the bootstrap "
    "interval's relationship to zero, and that interval is mildly liberal. Its "
    "measured coverage runs about 0.90 to 0.94 against a nominal 0.95, and "
    "falls short in every cell tested — sample lengths from 30 to 504 days, "
    "independent and autocorrelated alike, 400 seeded replications per cell, "
    "each cell worth about plus or minus 0.02. So the true "
    "uncertainty is wider than the interval shown, and on a genuine tie a "
    "\"beats\" or \"loses\" call happens roughly 6% to 10% of the time rather "
    "than the nominal 5%. Read the interval as a lower bound on the "
    "uncertainty, not an upper one."
)

#: The same disclosure for the ``bootstrap=False`` path, which is a DIFFERENT
#: and weaker claim: this module measures coverage for the bootstrap interval
#: only, so the HAC shortfall is unquantified and must not borrow the
#: bootstrap's numbers.
CI_COVERAGE_CAVEAT_HAC = (
    "Coverage caveat: the resampled interval was skipped, so this verdict "
    "comes from the Newey-West interval, which uses normal critical values with "
    "no small-sample correction. HAC asymptotics are liberal at small to "
    "moderate sample sizes (Ledoit & Wolf 2008), and this interval's coverage "
    "is not measured here — the shortfall is unquantified rather than small. "
    "Read the interval as a lower bound on the uncertainty."
)


# ------------------------------------------------------------ bundled entries

@dataclass(frozen=True)
class SharpeAssessment(_Gated):
    """Everything the report needs to say about one Sharpe number.

    Every statistic is optional: on a degenerate series (all cash, a filter
    that never fires) ``available`` is False, the statistics are None and
    ``unavailable_reason`` explains it. Consumers must branch on ``available``
    and render an em-dash — the same contract ``metrics.sharpe``'s NaN carries.
    """
    moments: ReturnMoments | None
    standard_error: SharpeSE | None
    psr: PSRResult | None
    dsr: DSRResult | None
    min_trl: MinTRLResult | None
    min_trl_vs_deflated: MinTRLResult | None
    min_btl: MinBTLResult | None
    bootstrap: BootstrapCI | None
    interpretation: str
    n_obs: int = 0
    available: bool = True
    unavailable_reason: str | None = None

    def to_dict(self) -> dict:
        # Hand-assembled because the published key names differ from the field
        # names — and still through the chokepoint, which is the only thing that
        # matters about an assembly point.
        return gated_dict({
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
            "n_obs": self.n_obs,
            "moments": self.moments.to_dict() if self.moments else None,
            "standard_error": (self.standard_error.to_dict()
                               if self.standard_error else None),
            "psr": self.psr.to_dict() if self.psr else None,
            "dsr": self.dsr.to_dict() if self.dsr else None,
            "min_track_record": self.min_trl.to_dict() if self.min_trl else None,
            "min_track_record_vs_deflated": (
                self.min_trl_vs_deflated.to_dict()
                if self.min_trl_vs_deflated is not None else None),
            "min_backtest_length": self.min_btl.to_dict() if self.min_btl else None,
            "bootstrap": self.bootstrap.to_dict() if self.bootstrap else None,
            "interpretation": self.interpretation,
        }, available=self.available)


def _unavailable_sharpe(n_obs: int, reason: str) -> SharpeAssessment:
    return SharpeAssessment.refuse(
        moments=None, standard_error=None, psr=None, dsr=None, min_trl=None,
        min_trl_vs_deflated=None, min_btl=None, bootstrap=None,
        n_obs=n_obs, available=False, unavailable_reason=reason,
        interpretation=(
            f"No Sharpe inference is reported for this strategy: {reason}. "
            f"Reporting a probability here would be a confident statement "
            f"about a number that does not exist."
        ),
    )


def assess_sharpe(returns, *, seed: int, rf_daily: float | pd.Series = 0.0,
                  n_trials: int = 1, trial_sharpes_annual=None,
                  alpha: float = 0.05, n_resamples: int = 1000,
                  holding_period_days: int = 1,
                  expected_block_days: float | None = None,
                  bootstrap: bool = True) -> SharpeAssessment:
    """One call producing every Sharpe inference the report needs.

    Composes: moments → Lo/Mertens standard error → PSR(0) → DSR against the
    N-trial luck threshold → MinTRL (against zero and, when available, against
    the deflated threshold) → MinBTL → studentized bootstrap CI.

    ``seed`` is required and threads into the bootstrap; pass something derived
    from the spec hash so the same spec always yields the same interval.
    Set ``bootstrap=False`` to skip the expensive part. ``holding_period_days``
    sizes the bootstrap blocks (see :func:`bootstrap_block_days`); a sample too
    short for that holding period simply carries no bootstrap interval, it does
    not raise.

    NEVER RAISES on data: a degenerate series returns ``available=False`` with
    an explanation. It still raises on NaN (an upstream alignment bug) and on
    programmer errors such as an oversized explicit ``expected_block_days``.
    """
    moments = ReturnMoments.try_from_returns(returns, rf_daily)
    if moments is None:
        n = len(_series(returns))
        return _unavailable_sharpe(
            n,
            "the return series has no dispersion at all — every day is "
            "identical, so a Sharpe ratio is undefined rather than large"
            if n >= 2 else
            "there are fewer than 2 return observations")
    se = sharpe_standard_error(moments)
    psr = probabilistic_sharpe_ratio(moments, 0.0)
    dsr = deflated_sharpe_ratio(
        moments, n_trials=n_trials,
        trial_sharpes_annual=trial_sharpes_annual)
    trl_zero = min_track_record_length(moments, 0.0, alpha)
    trl_deflated = (
        min_track_record_length(moments, dsr.sr0_periodic, alpha,
                                label="your own search threshold")
        if dsr.sr0_periodic else None)
    # Judge the trial budget against the Sharpe actually achieved; for a
    # non-positive Sharpe there is nothing to have got lucky about, so fall
    # back to the paper's own calibration target of 1.0.
    btl_target = moments.sharpe_annual if moments.sharpe_annual > 0 else 1.0
    btl = min_backtest_length(
        n_trials, target_sharpe_annual=btl_target,
        observed_years=moments.n_obs / PERIODS_PER_YEAR)

    boot = None
    boot_refusal = bootstrap_feasibility(moments.n_obs, holding_period_days)
    if bootstrap and boot_refusal is None:
        try:
            boot = bootstrap_sharpe_ci(
                returns, seed=seed, rf_daily=rf_daily,
                holding_period_days=holding_period_days, n_resamples=n_resamples,
                expected_block_days=expected_block_days,
                confidence=1.0 - alpha)
        except ResampleRefusal as exc:
            # Same biased-subsample refusal as on the excess path. There is no
            # HAC interval for a Sharpe, so nothing replaces it — the parametric
            # Lo/Mertens interval above stands on its own and the reason the
            # resampled one is missing is printed rather than absorbed.
            boot_refusal = str(exc)

    parts = [se.interpretation, psr.interpretation]
    if dsr.interpretation:
        parts.append(dsr.interpretation)
    if boot is not None:
        parts.append(boot.interpretation)
    elif bootstrap:
        parts.append(f"No resampled interval is reported: {boot_refusal}.")
    parts.append(trl_zero.interpretation)
    return SharpeAssessment(
        moments=moments, standard_error=se, psr=psr, dsr=dsr,
        min_trl=trl_zero, min_trl_vs_deflated=trl_deflated, min_btl=btl,
        bootstrap=boot, n_obs=moments.n_obs, available=True,
        unavailable_reason=None, interpretation=" ".join(parts),
    )


@dataclass(frozen=True)
class ExcessAssessment(_Gated):
    """Three-state benchmark comparison with the evidence behind it.

    ``available=False`` is the fourth, non-verdict state: the sample cannot
    support ANY of the three answers. ``verdict`` is then pinned to
    ``TIES_WITHIN_NOISE`` so no consumer can read a winner out of it, but
    consumers MUST branch on ``available`` and print ``unavailable_reason``
    rather than the tie headline — "we cannot tell" is not the same claim as
    "they are indistinguishable", and the audit is not allowed to blur them.

    ``ci_coverage_caveat`` is the other half of that honesty: whenever there IS
    a verdict it carries the measured shortfall of the interval the verdict
    rests on, so the report can show it instead of leaving it in this module's
    docstrings where no reader ever looks.
    """
    verdict: str                   # beats | ties_within_noise | loses
    hac: HACResult | None
    bootstrap: BootstrapCI | None
    n_obs: int
    interpretation: str
    verdict_source: str = "bootstrap"    # "bootstrap" | "hac" | "none"
    available: bool = True
    unavailable_reason: str | None = None
    #: How far short of nominal the interval behind this verdict actually
    #: covers — :data:`CI_COVERAGE_CAVEAT_BOOTSTRAP` or
    #: :data:`CI_COVERAGE_CAVEAT_HAC` depending on ``verdict_source``, and None
    #: when ``available`` is False because a refusal has no verdict to qualify.
    #: The report is expected to RENDER this next to the verdict.
    ci_coverage_caveat: str | None = None

    def to_dict(self) -> dict:
        return gated_dict({
            "verdict": self.verdict,
            "verdict_source": self.verdict_source,
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
            "ci_coverage_caveat": self.ci_coverage_caveat,
            "hac": self.hac.to_dict() if self.hac else None,
            "bootstrap": self.bootstrap.to_dict() if self.bootstrap else None,
            "n_obs": self.n_obs,
            "interpretation": self.interpretation,
        }, available=self.available)


def _unavailable_excess(n_obs: int, reason: str, label: str,
                        hac: HACResult | None = None) -> ExcessAssessment:
    return ExcessAssessment.refuse(
        verdict=TIES_WITHIN_NOISE, hac=hac, bootstrap=None, n_obs=n_obs,
        verdict_source="none", available=False, unavailable_reason=reason,
        interpretation=(
            f"No verdict is reported against {label}: {reason}. This is not a "
            f"tie — it is a refusal. Whichever total return is larger, this "
            f"sample cannot tell you whether the difference is real."
        ),
    )


def assess_excess(strategy, benchmark, *, seed: int,
                  holding_period_days: int = 1, alpha: float = 0.05,
                  n_resamples: int = 1000,
                  expected_block_days: float | None = None,
                  bootstrap: bool = True,
                  label: str = "the benchmark") -> ExcessAssessment:
    """Is "beats the benchmark" real? Returns a three-state answer, or refuses.

    Aligns the two series on their common index (a misalignment here would be
    a silent comparison of different periods), forms the daily
    strategy-minus-benchmark series, and reports both a Newey–West t-stat and a
    studentized stationary-bootstrap interval. The verdict comes from the
    bootstrap interval (better finite-sample coverage per Ledoit & Wolf 2008),
    or from the HAC interval when the caller passed ``bootstrap=False``.

    MINIMUM-SAMPLE GATE, mirroring the one PSR/DSR already apply: below
    ``MIN_OBS_FOR_INFERENCE`` observations — or when the declared holding
    period leaves fewer than ``MIN_EFFECTIVE_BLOCKS`` independent holds — the
    result is ``available=False`` and NO verdict. The HAC fallback used to run
    at any T ≥ 3 on normal critical values with no small-sample correction, and
    measurement showed it calling a true tie a win or a loss 26.8% of the time
    at T=5, 16.2% at T=10 and 9.4% at T=29, against a nominal 5%.

    NEVER RAISES on data: too few shared dates, a strategy identical to its
    benchmark, or a strategy whose difference from it is a nonzero CONSTANT (an
    all-cash strategy against a flat cash rate) come back as ``available=False``
    with an explanation.
    """
    s, b = _series(strategy), _series(benchmark)
    common = s.index.intersection(b.index)
    if len(common) < 3:
        return _unavailable_excess(
            len(common),
            f"the strategy and {label} share only {len(common)} dates",
            label)
    excess = (s.loc[common] - b.loc[common]).rename("excess")
    x = excess.to_numpy(dtype=float)
    # ONE predicate, BOTH halves, and the sentence generated from what is TRUE
    # of the data rather than from which half fired. This used to be two
    # separate gates with two separate texts, and the second one said "the
    # strategy tracks the benchmark exactly — the difference series has no
    # dispersion" about a series differing from cash by -5.04% annualized on
    # all 793 sessions. Gated HERE as well as at both rendering sites: this
    # entry is not allowed to be the only place the policy holds, and it is not
    # allowed to be stricter than the sites it calls either.
    deg = degeneracy_refusal(x)
    if deg is not None:
        return _unavailable_excess(
            len(excess), degeneracy_reason(deg, float(x.mean()), label), label)

    reason = bootstrap_feasibility(len(excess), holding_period_days)
    if reason is not None:
        # Report the HAC evidence for the record, but NOT as a verdict.
        hac_only = newey_west_tstat(excess,
                                    holding_period_days=holding_period_days,
                                    confidence=1.0 - alpha, label=label)
        return _unavailable_excess(len(excess), reason, label, hac=hac_only)

    hac = newey_west_tstat(excess, holding_period_days=holding_period_days,
                           confidence=1.0 - alpha, label=label)
    if not hac.available:
        # The predicate above passed and the t-statistic still came out
        # unmeasurable. There is then no interval at either site: the bootstrap
        # would studentize by the same collapsed denominator and classify_excess
        # refuses a NaN endpoint, so this is a refusal rather than a traceback.
        return _unavailable_excess(
            len(excess), hac.unavailable_reason or "the Newey-West standard "
            "error of the daily difference is not measurable", label, hac=hac)
    boot = None
    boot_refusal: str | None = None
    if bootstrap:
        try:
            boot = bootstrap_excess_return_ci(
                excess, seed=seed, holding_period_days=holding_period_days,
                n_resamples=n_resamples, expected_block_days=expected_block_days,
                confidence=1.0 - alpha, label=label)
        except ResampleRefusal as exc:
            # THE BIASED-SUBSAMPLE PATH. The bootstrap's pivot set is a
            # one-sided subsample, not a smaller sample, so its interval is
            # refused. The HAC interval is unbiased on exactly these inputs (it
            # never resamples, so a rare day cannot go missing from it), so the
            # verdict comes from there — and ``verdict_source`` says so, carrying
            # CI_COVERAGE_CAVEAT_HAC rather than the bootstrap's measured band.
            # Silently keeping the bootstrap number here is what reported "loses"
            # beside "t-statistic of -0.01 … short of the |t| > 2 bar".
            boot_refusal = str(exc)

    if boot is not None:
        verdict = classify_excess(boot.point, boot.ci_low, boot.ci_high)
        source = "bootstrap"
    else:
        verdict = classify_excess(hac.mean_annualized, hac.ci_low_annualized,
                                  hac.ci_high_annualized)
        source = "hac"

    headline = {
        BEATS: f"The strategy beats {label} by more than sampling noise explains.",
        LOSES: f"The strategy loses to {label} by more than sampling noise explains.",
        TIES_WITHIN_NOISE: (
            f"The strategy and {label} are not distinguishable on this sample — "
            f"whichever number is larger, the difference is inside the noise."),
    }[verdict]
    parts = [headline, hac.interpretation]
    if boot is not None:
        parts.append(boot.interpretation)
    elif boot_refusal is not None:
        parts.append(
            f"No resampled interval stands behind this verdict: {boot_refusal}. "
            f"The verdict above is therefore the Newey-West interval's "
            f"relationship to zero, which does not resample and so cannot lose "
            f"the rare part of this series.")
    if boot is not None:
        # SURFACED, NOT ABSORBED. The two intervals are two different estimators
        # of the same quantity and they can land on opposite sides of zero; the
        # report used to print the resampled verdict as the headline and the HAC
        # sentence underneath it with nothing to say they disagreed, which reads
        # as one paragraph contradicting itself. It is not a contradiction — it
        # is a sample sitting on the boundary — and saying so is the difference.
        hac_verdict = classify_excess(hac.mean_annualized,
                                      hac.ci_low_annualized,
                                      hac.ci_high_annualized)
        if hac_verdict != verdict:
            parts.append(
                f"The two intervals do not agree: the resampled interval reads "
                f"\"{verdict.replace('_', ' ')}\" and the Newey-West interval "
                f"reads \"{hac_verdict.replace('_', ' ')}\". The verdict above "
                f"is the resampled one, for the finite-sample reason in the "
                f"method note; the disagreement itself is evidence that this "
                f"sample sits on the boundary, and neither answer should be "
                f"read as settled."
            )
    if abs(hac.autocorr_lag1) >= 0.3:
        # Surfaced, not absorbed: the measured coverage of this interval
        # degrades as the daily excess series becomes strongly persistent.
        parts.append(
            f"The daily excess series is strongly autocorrelated "
            f"(lag-1 {hac.autocorr_lag1:+.2f}), so treat this interval as a "
            f"lower bound on the true uncertainty — resampled intervals are "
            f"measurably liberal under this much persistence."
        )
    return ExcessAssessment(
        verdict=verdict, hac=hac, bootstrap=boot, n_obs=len(excess),
        verdict_source=source, available=True, unavailable_reason=None,
        ci_coverage_caveat=(CI_COVERAGE_CAVEAT_BOOTSTRAP if source == "bootstrap"
                            else CI_COVERAGE_CAVEAT_HAC),
        interpretation=" ".join(parts),
    )
