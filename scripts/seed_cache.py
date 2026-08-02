"""Seed the parquet price cache from the frozen fixture CSVs, fully offline.

Usage: .venv/bin/python scripts/seed_cache.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backend"))

from config import settings  # noqa: E402
from engine.data import PriceCache  # noqa: E402

# The fixtures were frozen from the original momentum_backtest session
# (downloaded 2026-06); fetched_at is pinned so provenance is honest.
FIXTURE_FETCHED_AT = "2026-06-01T00:00:00+00:00"

FIXTURE_FILES = [
    REPO / "fixtures" / "sp500_prices.csv",      # loaded first …
    REPO / "fixtures" / "war_list_prices.csv",   # … war list wins for overlaps (fresher tail)
    REPO / "fixtures" / "extras_prices.csv",     # QQQ/AGG/^IRX etc., if present
]


def main() -> None:
    settings.ensure_dirs()
    cache = PriceCache()
    seeded: dict[str, int] = {}
    for path in FIXTURE_FILES:
        if not path.exists():
            print(f"skip (not found): {path.name}")
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
            seeded[col] = len(series)
        print(f"seeded {len(df.columns)} tickers from {path.name}")
    print(f"cache now holds {len(cache.tickers())} tickers at {cache.dir}")


if __name__ == "__main__":
    main()
