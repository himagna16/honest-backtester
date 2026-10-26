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


# Removed/renamed S&P 500 members' closes for point-in-time membership, built by
# scripts/build_sp500_membership.py (downloaded 2026-09-17, so pinned apart).
PIT_FIXTURE = "sp500_pit_prices.parquet"
PIT_FETCHED_AT = "2026-09-17T00:00:00+00:00"


def _fixture_files() -> list:
    d = settings.fixtures_dir
    return [
        d / PIT_FIXTURE,             # loaded first, so the CSVs below win overlaps …
        d / "sp500_prices.csv",      # … then the S&P 500 …
        d / "war_list_prices.csv",   # … war list wins for overlaps (fresher tail)
        d / "extras_prices.csv",     # QQQ/AGG/^IRX etc., if present
    ]


def _read_fixture(path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
        df.index = pd.DatetimeIndex(df.index)
        return df
    return pd.read_csv(path, index_col=0, parse_dates=True)


def seed_cache() -> int:
    """Load every fixture into the price cache. Returns the ticker count."""
    settings.ensure_dirs()
    cache = PriceCache()
    for path in _fixture_files():
        if not path.exists():
            continue
        df = _read_fixture(path)
        fetched_at = PIT_FETCHED_AT if path.name == PIT_FIXTURE else FIXTURE_FETCHED_AT
        for col in df.columns:
            series = df[col].dropna()
            if series.empty:
                continue
            existing = cache.get_series(col)
            if existing is not None:
                merged = pd.concat([existing, series])
                series = merged[~merged.index.duplicated(keep="last")].sort_index()
            cache.put_series(col, series, source="fixture",
                             fetched_at=fetched_at)
    return len(cache.tickers())


def backfill_cache(path=None) -> dict[str, int]:
    """Add a fixture's history to an EXISTING cache without disturbing it.

    Dates the cache already holds keep the cached close. Earlier dates the
    cache lacks are spliced on, rescaled so the fixture's close matches the
    cache's on the first shared date — adjusted closes downloaded on different
    days differ by a constant factor, and an unscaled splice would print a
    fake one-day jump. Returns {ticker: sessions added}.
    """
    settings.ensure_dirs()
    cache = PriceCache()
    df = _read_fixture(path or settings.fixtures_dir / PIT_FIXTURE)
    added: dict[str, int] = {}
    for col in df.columns:
        series = df[col].dropna()
        if series.empty:
            continue
        existing = cache.get_series(col)
        if existing is None or existing.dropna().empty:
            cache.put_series(col, series, source="fixture", fetched_at=PIT_FETCHED_AT)
            added[col] = len(series)
            continue
        existing = existing.dropna()
        older = series[series.index < existing.index[0]]
        shared = series.index.intersection(existing.index)
        if older.empty or shared.empty:
            continue
        scale = float(existing.loc[shared[0]]) / float(series.loc[shared[0]])
        merged = pd.concat([older * scale, existing]).sort_index()
        info = cache.info(col) or {}
        cache.put_series(col, merged, source=info.get("source", "cache"),
                         fetched_at=info.get("fetched_at"))
        added[col] = len(older)
    return added


def seed_if_empty() -> int:
    """Seed only when the cache is empty (first boot); a no-op afterward, so it
    costs nothing on warm restarts and never overwrites a real cache."""
    settings.ensure_dirs()
    if PriceCache().tickers():
        return 0
    return seed_cache()
