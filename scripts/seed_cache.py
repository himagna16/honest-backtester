"""Seed the parquet price cache from the frozen fixture CSVs, fully offline.

Usage: .venv/bin/python scripts/seed_cache.py

Thin CLI over backend/seed.py, which the app also calls on first boot.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backend"))

from seed import seed_cache  # noqa: E402


if __name__ == "__main__":
    n = seed_cache()
    print(f"cache now holds {n} tickers")
