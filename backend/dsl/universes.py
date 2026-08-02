"""Universe presets (Gap #3).

Seed constituent lists are checked into fixtures/ with their source URL and
snapshot date in the header; scripts/refresh_universes.py refreshes them on
demand. The audit layer discloses on every run that current-membership lists
are still mildly survivorship biased — "less biased, not unbiased".
"""
from __future__ import annotations

from pathlib import Path

from config import settings

SECTOR_ETFS = [
    "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY",
]

PRESET_FILES = {
    "sp500": "sp500_tickers.txt",
    "nasdaq100": "nasdaq100_tickers.txt",
    "dow30": "dow30_tickers.txt",
}

PRESET_LABELS = {
    "sp500": "S&P 500 (current members — seed list, see fixtures/sp500_tickers.txt)",
    "nasdaq100": "Nasdaq-100 (current members)",
    "dow30": "Dow 30 (current members)",
    "sector_etfs": "11 SPDR sector ETFs",
    "user_watchlist": "user-supplied watchlist",
}


def normalize_ticker(sym: str) -> str:
    """yfinance uses '-' where index lists use '.' (BRK.B → BRK-B)."""
    return sym.strip().upper().replace(".", "-")


def _read_list(path: Path) -> list[str]:
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(normalize_ticker(line))
    return sorted(set(out))


def preset_tickers(preset: str) -> list[str]:
    if preset == "sector_etfs":
        return list(SECTOR_ETFS)
    if preset in PRESET_FILES:
        path = settings.fixtures_dir / PRESET_FILES[preset]
        if not path.exists():
            raise FileNotFoundError(
                f"seed list missing: {path} — run scripts/refresh_universes.py"
            )
        return _read_list(path)
    raise ValueError(f"preset '{preset}' has no ticker list (user_watchlist "
                     f"resolves from the uploaded file instead)")


def read_watchlist_file(name: str) -> list[str]:
    """Read an uploaded watchlist from data/watchlists/ (basename only —
    no path traversal)."""
    safe = Path(name).name
    path = settings.db_path.parent / "watchlists" / safe
    if not path.exists():
        raise FileNotFoundError(f"watchlist file not found: {safe}")
    return _read_list(path)
