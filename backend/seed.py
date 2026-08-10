"""Seed the parquet price cache from the frozen fixture CSVs — fully offline.

Lives in the backend package (not scripts/) so it can run on container startup:
the Dockerfile ships backend/ and fixtures/ but not scripts/. A deployed backend
seeds itself once from the fixtures, so the public demo runs backtests
deterministically with no yfinance call (which also sidesteps the yfinance ToS
question for the hosted instance).
"""
from __future__ import annotations

import pandas as pd

from config import settings
from engine.data import PriceCache

# Fixtures were frozen from the founding session (downloaded 2026-06); the
# fetched_at is pinned so provenance stays honest.
FIXTURE_FETCHED_AT = "2026-06-01T00:00:00+00:00"


def _fixture_files() -> list:
    d = settings.fixtures_dir
    return [
        d / "sp500_prices.csv",      # loaded first …
        d / "war_list_prices.csv",   # … war list wins for overlaps (fresher tail)
        d / "extras_prices.csv",     # QQQ/AGG/^IRX etc., if present
    ]


def seed_cache() -> int:
    """Load every fixture CSV into the price cache. Returns the ticker count."""
    settings.ensure_dirs()
    cache = PriceCache()
    for path in _fixture_files():
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        for col in df.columns:
            series = df[col].dropna()
            if series.empty:
                continue
            existing = cache.get_series(col)
            if existing is not None:
                merged = pd.concat([existing, series])
                series = merged[~merged.index.duplicated(keep="last")].sort_index()
            cache.put_series(col, series, source="fixture",
                             fetched_at=FIXTURE_FETCHED_AT)
    return len(cache.tickers())


def seed_if_empty() -> int:
    """Seed only when the cache is empty (first boot); a no-op afterward, so it
    costs nothing on warm restarts and never overwrites a real cache."""
    settings.ensure_dirs()
    if PriceCache().tickers():
        return 0
    return seed_cache()
