"""Point-in-time index membership: who was in the index on day t.

A ranked strategy on TODAY's constituent list buys companies before they were
in the index, and those companies are the ones that grew into it. That is how
this project's own "full S&P 500" momentum runs reported +534% to +3560% over
2023-2026: AppLovin, Carvana, Vertiv and Palantir were eligible years before
S&P added them. Membership as of each signal date removes that.

The data is fixtures/<index>_membership/membership.csv, one row per membership
spell, built by scripts/build_sp500_membership.py from pinned, cited sources
(see that folder's README.md). Semantics:
  * `start` inclusive, `end` exclusive, both effective dates; blank start =
    before coverage, blank end = still a member at the anchor date.
  * `price_symbol` is the column the engine prices the member with; blank when
    no usable free history exists. Such members are COUNTED (coverage) but can
    never be picked — the gap is disclosed, not hidden.
  * Nothing is claimed before `coverage_start` (meta.json).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from config import settings

MEMBERSHIP_DIRS = {"sp500": "sp500_membership"}


@dataclass(frozen=True)
class MembershipCoverage:
    """How much of the index a run could actually see at its signal dates."""
    dates: int
    mean_priced_share: float     # members with a usable close ÷ members, averaged
    min_priced_share: float
    min_date: str | None
    mean_members: float
    mean_unpriced: float

    def to_dict(self) -> dict:
        return {
            "signal_dates": self.dates,
            "mean_priced_share": self.mean_priced_share,
            "min_priced_share": self.min_priced_share,
            "min_date": self.min_date,
            "mean_members": self.mean_members,
            "mean_unpriced": self.mean_unpriced,
        }


class PointInTimeMembership:
    def __init__(self, index: str, spells: pd.DataFrame, meta: dict):
        self.index = index
        self.meta = meta
        s = spells.copy()
        s["start"] = pd.to_datetime(s["start"])
        s["end"] = pd.to_datetime(s["end"])
        s["price_symbol"] = s["price_symbol"].fillna("").astype(str)
        self.spells = s
        self.coverage_start = pd.Timestamp(meta["coverage_start"])
        self.known_through = pd.Timestamp(meta["changes_through"])

    # ---- windows ----
    def covers(self, start: str | pd.Timestamp | None) -> bool:
        """Membership is only claimed from coverage_start. A window that starts
        earlier gets no point-in-time universe at all — never a half-and-half
        one, because the early rebalances are where survivors do the damage."""
        return start is not None and pd.Timestamp(start) >= self.coverage_start

    def _overlapping(self, start, end) -> pd.DataFrame:
        s = self.spells
        lo = pd.Timestamp(start)
        hi = pd.Timestamp(end) if end is not None else pd.Timestamp.max
        began = s["start"].isna() | (s["start"] <= hi)
        not_over = s["end"].isna() | (s["end"] > lo)
        return s[began & not_over]

    def symbols_between(self, start, end) -> list[str]:
        """Every price symbol that was a member at any point in [start, end] —
        what a run must fetch. Members with no usable history are left out
        here (there is nothing to fetch) and reappear in coverage()."""
        spells = self._overlapping(start, end)
        return sorted({p for p in spells["price_symbol"] if p})

    def members_on(self, day) -> pd.DataFrame:
        t = pd.Timestamp(day)
        s = self.spells
        on = (s["start"].isna() | (s["start"] <= t)) & (s["end"].isna() | (s["end"] > t))
        return s[on]

    # ---- the engine's view ----
    def eligibility(self, index: pd.DatetimeIndex, symbols: list[str]) -> pd.DataFrame:
        """dates × symbols, True where the symbol was a member at that close.
        A symbol with several spells (FOX before and after 2019, a name removed
        and re-added) is a member on the union of them."""
        out = pd.DataFrame(False, index=index, columns=list(symbols))
        wanted = set(symbols)
        for r in self.spells.itertuples():
            if r.price_symbol not in wanted:
                continue
            on = pd.Series(True, index=index)
            if pd.notna(r.start):
                on &= index >= r.start
            if pd.notna(r.end):
                on &= index < r.end
            out[r.price_symbol] |= on.to_numpy()
        return out

    def coverage(self, dates, prices: pd.DataFrame) -> MembershipCoverage:
        """Share of members with a usable close on each date — the part of the
        index a strategy could actually have picked from."""
        idx = pd.DatetimeIndex(dates)
        if len(idx) == 0:
            return MembershipCoverage(0, 0.0, 0.0, None, 0.0, 0.0)
        s = self.spells
        t = idx.to_numpy()[None, :]
        start = s["start"].to_numpy()[:, None]
        end = s["end"].to_numpy()[:, None]
        on = (pd.isna(start) | (start <= t)) & (pd.isna(end) | (end > t))  # spells × dates

        # A member counts as priced only if its symbol has a close on that date.
        closes = prices.reindex(index=idx)
        priced_on = np.zeros(on.shape, dtype=bool)
        for i, p in enumerate(s["price_symbol"]):
            if p and p in closes.columns:
                priced_on[i, :] = closes[p].notna().to_numpy()

        n_members = on.sum(axis=0)
        n_priced = (on & priced_on).sum(axis=0)
        keep = n_members > 0
        if not keep.any():
            return MembershipCoverage(0, 0.0, 0.0, None, 0.0, 0.0)
        share = n_priced[keep] / n_members[keep]
        kept_dates = idx[keep]
        worst = int(share.argmin())
        return MembershipCoverage(
            dates=int(keep.sum()),
            mean_priced_share=float(share.mean()),
            min_priced_share=float(share[worst]),
            min_date=str(kept_dates[worst].date()),
            mean_members=float(n_members[keep].mean()),
            mean_unpriced=float((n_members[keep] - n_priced[keep]).mean()),
        )

    def provenance(self) -> dict:
        return {
            "index": self.index,
            "coverage_start": str(self.coverage_start.date()),
            "known_through": str(self.known_through.date()),
            "anchor_source": self.meta.get("anchor_source"),
            "changes_source": self.meta.get("changes_source"),
            "membership_sha256": self.meta.get("membership_sha256"),
        }


def _load(folder: Path, index: str) -> PointInTimeMembership | None:
    csv, meta_path = folder / "membership.csv", folder / "meta.json"
    if not csv.exists() or not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    blob = csv.read_bytes()
    if meta.get("membership_sha256") and hashlib.sha256(blob).hexdigest() != meta["membership_sha256"]:
        # A hand-edited membership file is not the one the validation ran on.
        raise RuntimeError(
            f"{csv} does not match the hash in {meta_path.name} — rebuild it with "
            "scripts/build_sp500_membership.py instead of editing it by hand"
        )
    spells = pd.read_csv(csv, dtype=str)
    return PointInTimeMembership(index, spells, meta)


@lru_cache(maxsize=None)
def load_membership(index: str) -> PointInTimeMembership | None:
    """The point-in-time table for an index preset, or None when none exists
    (Nasdaq-100 and Dow 30 have no free dated history here)."""
    folder = MEMBERSHIP_DIRS.get(index)
    if folder is None:
        return None
    return _load(settings.fixtures_dir / folder, index)
