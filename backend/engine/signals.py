"""Signal computation. Every primitive is lag-safe by construction:

- only backward-looking operations (shift with non-negative lags, rolling
  windows, ewm) are used, so the value at date t depends exclusively on
  closes at dates ≤ t. A property test perturbs future rows and asserts
  signals at t are unchanged (audit check #2's impossibility proof).
- rolling windows use min_periods = full window, so a ticker with partial
  history is NaN (ineligible) until its lookback actually exists — partial
  data never silently produces a value.

Exact formulas are documented in dsl/conventions.py (the single source of
truth shared with the LLM prompt and the glossary).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dsl.schema import Signal

# Lookback used when weighting == "inverse_vol" (see conventions table)
INVERSE_VOL_LOOKBACK = 63


def compute_signal(prices: pd.DataFrame, signal: Signal) -> pd.DataFrame:
    """Signal value per (date, ticker). NaN where history is insufficient."""
    t = signal.type
    if t in ("return", "daily_return"):
        lb = signal.lookback_days
        if lb is None:
            raise ValueError(f"signal '{t}' requires lookback_days")
        skip = signal.skip_days
        return prices.shift(skip) / prices.shift(skip + lb) - 1.0
    if t == "volatility":
        lb = signal.lookback_days
        if lb is None:
            raise ValueError("signal 'volatility' requires lookback_days")
        daily = prices.pct_change(fill_method=None)
        return daily.rolling(lb, min_periods=lb).std() * np.sqrt(252)
    if t == "sma_cross":
        if signal.fast is None or signal.slow is None:
            raise ValueError("signal 'sma_cross' requires fast and slow")
        fast = prices.rolling(signal.fast, min_periods=signal.fast).mean()
        slow = prices.rolling(signal.slow, min_periods=signal.slow).mean()
        return fast / slow - 1.0
    if t == "drawdown_from_high":
        lb = signal.lookback_days
        if lb is None:
            raise ValueError("signal 'drawdown_from_high' requires lookback_days")
        rolling_max = prices.rolling(lb, min_periods=lb).max()
        return prices / rolling_max - 1.0
    if t == "rsi":
        lb = signal.lookback_days or 14
        delta = prices.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        # Wilder smoothing = EWM with alpha = 1/lb
        avg_gain = gain.ewm(alpha=1.0 / lb, adjust=False, min_periods=lb).mean()
        avg_loss = loss.ewm(alpha=1.0 / lb, adjust=False, min_periods=lb).mean()
        rs = avg_gain / avg_loss
        rsi = 100.0 - 100.0 / (1.0 + rs)
        # avg_loss == 0 → RSI 100 by convention
        return rsi.where(avg_loss != 0, 100.0).where(avg_gain.notna())
    if t == "fundamental":
        raise ValueError(
            "fundamental signals need licensed data — rejected in V0; "
            "the validator should have caught this before the engine"
        )
    raise ValueError(f"unknown signal type: {t}")


def apply_filter_op(values: pd.Series, op: str, value) -> pd.Series:
    """Boolean eligibility mask for one filter at one date. NaN → False."""
    if op == "gt":
        return values > value
    if op == "gte":
        return values >= value
    if op == "lt":
        return values < value
    if op == "lte":
        return values <= value
    if op == "between":
        lo, hi = value
        return (values >= lo) & (values <= hi)
    raise ValueError(f"unknown filter op: {op}")
