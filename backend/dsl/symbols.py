"""Symbol master (Gap #4): validate tickers at spec time so hallucinated or
renamed symbols are rejected with suggestions BEFORE anything runs.

The master is the union of every source of truth we have locally: the seeded
price cache, the checked-in constituent lists, sector ETFs, benchmark tickers,
and a small set of common ETFs/renames. It deliberately errs toward rejecting
unknowns: a typo caught at validation beats an empty yfinance frame mid-run.
"""
from __future__ import annotations

import difflib
from functools import lru_cache

from dsl.universes import SECTOR_ETFS, PRESET_FILES, _read_list, normalize_ticker
from config import settings

COMMON_ETFS = [
    "SPY", "QQQ", "DIA", "IWM", "AGG", "BND", "TLT", "IEF", "SHY", "GLD",
    "SLV", "VTI", "VOO", "VEA", "VWO", "EFA", "EEM", "LQD", "HYG", "XBI",
]

KNOWN_RENAMES = {
    "FB": "META",
    "TWTR": "X (delisted 2022 — not backtestable via yfinance)",
    "GOOG.L": "GOOGL",
}


@lru_cache(maxsize=1)
def symbol_master() -> frozenset[str]:
    symbols: set[str] = set(COMMON_ETFS) | set(SECTOR_ETFS)
    for fname in PRESET_FILES.values():
        path = settings.fixtures_dir / fname
        if path.exists():
            symbols.update(_read_list(path))
    # everything already in the price cache is known-good
    try:
        from engine.data import PriceCache
        symbols.update(PriceCache().tickers())
    except Exception:
        pass
    # constituents CSV carries names too (Symbol column)
    con = settings.fixtures_dir / "sp500_constituents.csv"
    if con.exists():
        import csv
        with con.open() as f:
            for row in csv.DictReader(f):
                if row.get("Symbol"):
                    symbols.add(normalize_ticker(row["Symbol"]))
    return frozenset(s for s in symbols if s)


def check_tickers(tickers: list[str]) -> tuple[list[str], list[dict]]:
    """→ (valid_normalized, problems). Each problem: {ticker, suggestion?}."""
    master = symbol_master()
    valid: list[str] = []
    problems: list[dict] = []
    for raw in tickers:
        sym = normalize_ticker(raw)
        if sym in master:
            valid.append(sym)
            continue
        problem: dict = {"ticker": raw}
        if sym in KNOWN_RENAMES:
            problem["suggestion"] = KNOWN_RENAMES[sym]
        else:
            close = difflib.get_close_matches(sym, master, n=3, cutoff=0.6)
            if close:
                problem["suggestion"] = " or ".join(close)
        problems.append(problem)
    return valid, problems
