"""Refresh the checked-in universe seed lists from Wikipedia (Gap #3).

Wikipedia scraping is fragile and has its own ToS considerations — that is
exactly why the lists are CHECKED IN as seeds and this script is a manual,
occasional refresh (not a runtime dependency). Each output file records its
source and snapshot date on the first lines.

Honesty caveat (disclosed by the audit on every run): these are *current*
constituent lists, and backtesting past windows on today's members is
survivorship-biased — on this project's own S&P 500 momentum runs, by an order
of magnitude. S&P 500 windows from 2015 use point-in-time membership instead
(scripts/build_sp500_membership.py); these lists remain the universe for the
Nasdaq-100, the Dow 30 and earlier S&P 500 windows, and the audit caps a
ranked strategy on them at amber.

Usage: .venv/bin/python scripts/refresh_universes.py
"""
from __future__ import annotations

import io
from datetime import date
from pathlib import Path

import pandas as pd
import requests

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "fixtures"

SOURCES = {
    "sp500": ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "Symbol"),
    "nasdaq100": ("https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies", "Ticker"),
    "dow30": ("https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average", "Symbol"),
}


def scrape(url: str, symbol_col: str) -> list[str]:
    resp = requests.get(url, timeout=30, headers={"User-Agent": "honest-backtester/0.1 (personal research tool)"})
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    for table in tables:
        if symbol_col in table.columns:
            syms = [str(s).strip() for s in table[symbol_col].dropna()]
            syms = [s for s in syms if s and len(s) <= 6 and s.upper() == s]
            if len(syms) >= 25:
                return syms
    raise RuntimeError(f"no table with column {symbol_col!r} found at {url}")


def main() -> None:
    today = date.today().isoformat()
    for name, (url, col) in SOURCES.items():
        try:
            syms = scrape(url, col)
        except Exception as exc:
            print(f"{name}: FAILED ({exc}) — keeping existing seed file")
            continue
        path = FIXTURES / f"{name}_tickers.txt"
        lines = [f"# source: {url}", f"# snapshot: {today}", *sorted(set(syms))]
        path.write_text("\n".join(lines) + "\n")
        print(f"{name}: {len(set(syms))} tickers → {path.name}")


if __name__ == "__main__":
    main()
