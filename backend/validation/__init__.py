"""Out-of-sample validation primitives.

Pure, deterministic index arithmetic that the audit consumes to build splits
which are genuinely out-of-sample — purged and embargoed per López de Prado
(AFML ch. 7 & 12) — instead of the single unpurged 70/30 cut the audit shipped
with. Nothing in here runs a backtest, touches the network or reads the
filesystem, so it stays fast enough to run inside every audit; it does read the
engine's rebalance calendar and the dsl/conventions constants, because a purge
sized from a private copy of those numbers goes stale the day the engine moves.

Sizing the purge: call `purge_length(spec, index=..., execution_lag_days=...)`
with the calendar and the lag the run ACTUALLY used, and hand the result to
`purged_split` / `walk_forward` / `combinatorial_purged_cv` as `lookback`.
`purge_components` returns the same number broken into its terms so the report
can disclose where the holdout bars went.

The package exists as a separate layer (rather than inside audit/) because
three callers want the same primitive: the holdout check, walk-forward, and
CPCV. One implementation, three callers.
"""
from validation.splits import (  # noqa: F401
    CPCVDesign,
    CPCVPath,
    DEFAULT_N_FOLDS,
    DEFAULT_N_GROUPS,
    DEFAULT_N_TEST_GROUPS,
    DEFAULT_TRAIN_FRACTION,
    PathSegment,
    Split,
    SplitError,
    calendar_rebalance_carry,
    combinatorial_purged_cv,
    default_embargo,
    n_cpcv_paths,
    n_cpcv_splits,
    purge_components,
    purge_length,
    purged_split,
    rsi_reach,
    signal_reach,
    walk_forward,
)

__all__ = [
    "CPCVDesign",
    "CPCVPath",
    "DEFAULT_N_FOLDS",
    "DEFAULT_N_GROUPS",
    "DEFAULT_N_TEST_GROUPS",
    "DEFAULT_TRAIN_FRACTION",
    "PathSegment",
    "Split",
    "SplitError",
    "calendar_rebalance_carry",
    "combinatorial_purged_cv",
    "default_embargo",
    "n_cpcv_paths",
    "n_cpcv_splits",
    "purge_components",
    "purge_length",
    "purged_split",
    "rsi_reach",
    "signal_reach",
    "walk_forward",
]
