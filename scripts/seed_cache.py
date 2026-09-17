"""Seed the parquet price cache from the frozen fixture CSVs, fully offline.

Usage: .venv/bin/python scripts/seed_cache.py                 # seed from fixtures
       .venv/bin/python scripts/seed_cache.py --backfill-pit  # add PIT S&P 500 names

Thin CLI over backend/seed.py, which the app also calls on first boot.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backend"))

from seed import backfill_cache, seed_cache  # noqa: E402


if __name__ == "__main__":
    if "--backfill-pit" in sys.argv:
        # An existing cache: add the point-in-time S&P 500 names' history
        # without overwriting anything already cached.
        added = backfill_cache()
        print(f"backfilled {len(added)} tickers "
              f"({sum(added.values())} sessions added)")
    else:
        n = seed_cache()
        print(f"cache now holds {n} tickers")
