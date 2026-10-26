"""Universe presets (Gap #3).

Seed constituent lists are checked into fixtures/ with their source URL and
snapshot date in the header; scripts/refresh_universes.py refreshes them on
demand.

Index presets resolve POINT-IN-TIME where dated membership exists (the S&P 500
from 2015-01-01, see dsl/membership.py): the universe is every company that was
a member at some point in the window, and the engine only lets a name be picked
on dates it was actually in the index. Where it does not exist (Nasdaq-100, Dow
30, or an S&P 500 window starting before coverage), the preset falls back to
TODAY's members, and the audit treats that as survivorship-biased: a warning,
and a ranked strategy cannot earn a green verdict on it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config import settings
from dsl.membership import PointInTimeMembership, load_membership

SECTOR_ETFS = [
    "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY",
]

PRESET_FILES = {
    "sp500": "sp500_tickers.txt",
    "nasdaq100": "nasdaq100_tickers.txt",
    "dow30": "dow30_tickers.txt",
}

PRESET_LABELS = {
    "sp500": "the S&P 500",
    "nasdaq100": "the Nasdaq-100 (today's members)",
    "dow30": "the Dow 30 (today's members)",
    "sector_etfs": "the 11 SPDR sector ETFs",
    "user_watchlist": "user-supplied watchlist",
}

INDEX_PRESETS = ("sp500", "nasdaq100", "dow30")
INDEX_NAMES = {"sp500": "S&P 500", "nasdaq100": "Nasdaq-100", "dow30": "Dow 30"}


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


@dataclass
class ResolvedUniverse:
    tickers: list[str]
    # Set only when the universe is point-in-time for the requested window.
    membership: PointInTimeMembership | None = None
    # For an index preset that fell back to today's members: why, in words.
    fallback_reason: str | None = None

    @property
    def point_in_time(self) -> bool:
        return self.membership is not None


def resolve_preset(preset: str, start: str | None, end: str | None) -> ResolvedUniverse:
    """The tickers a preset stands for over [start, end], point-in-time when
    the dated membership covers the window's start."""
    if preset not in INDEX_PRESETS:
        return ResolvedUniverse(preset_tickers(preset))
    name = INDEX_NAMES[preset]
    membership = load_membership(preset)
    if membership is None:
        return ResolvedUniverse(
            preset_tickers(preset),
            fallback_reason=f"there is no dated membership history for the {name} here",
        )
    if not membership.covers(start):
        return ResolvedUniverse(
            preset_tickers(preset),
            fallback_reason=(
                f"dated {name} membership starts {membership.coverage_start.date()}, "
                f"and this window starts {start}"
            ),
        )
    return ResolvedUniverse(membership.symbols_between(start, end), membership=membership)


def read_watchlist_file(name: str) -> list[str]:
    """Read an uploaded watchlist from data/watchlists/ (basename only —
    no path traversal)."""
    safe = Path(name).name
    path = settings.db_path.parent / "watchlists" / safe
    if not path.exists():
        raise FileNotFoundError(f"watchlist file not found: {safe}")
    return _read_list(path)
