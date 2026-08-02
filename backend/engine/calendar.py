"""Rebalance-date selection on the trading calendar.

The calendar is the price panel's own DatetimeIndex (SPY's trading days —
see dsl/conventions.py). Signal dates are the LAST trading day of each
calendar period; positions earn returns from the NEXT trading day.
"""
from __future__ import annotations

import pandas as pd


def rebalance_signal_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """Last trading day of each period present in `index` (daily = every day)."""
    if len(index) == 0:
        return index
    if freq == "daily":
        return index
    grouper = {
        "weekly": index.to_period("W"),
        "monthly": index.to_period("M"),
        "quarterly": index.to_period("Q"),
        "annual": index.to_period("Y"),
    }.get(freq)
    if grouper is None:
        raise ValueError(f"unknown rebalance frequency: {freq}")
    s = pd.Series(index, index=grouper)
    last_days = s.groupby(level=0).max()
    return pd.DatetimeIndex(last_days.values)


def clip_window(
    index: pd.DatetimeIndex, start: str | None, end: str | None
) -> pd.DatetimeIndex:
    """Trading days within [start, end] (inclusive)."""
    mask = pd.Series(True, index=index)
    if start is not None:
        mask &= index >= pd.Timestamp(start)
    if end is not None:
        mask &= index <= pd.Timestamp(end)
    return index[mask.values]
