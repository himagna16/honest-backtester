"""Data layer: yfinance + local parquet cache + per-run frozen snapshots.

Honesty rules (Section 3 + Gap #2/#4):
- Data problems are surfaced, never absorbed: missing tickers, short history,
  and stale cache all land in PricePanel.notes / missing / stale for the audit
  layer to turn into findings.
- Never silently mix stale and fresh: the panel's as_of is the OLDEST fetch
  date across its tickers, and stale=True if any ticker missed its refresh.
- Every run freezes its exact panel to parquet with a SHA-256 content hash;
  re-runs replay the snapshot by default. "Refresh data" is the only path to
  new numbers.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from config import settings

META_FILE = "_meta.json"
CACHE_STALE_HOURS = 24


def _safe_name(ticker: str) -> str:
    return ticker.replace("^", "%5E").replace("/", "-") + ".parquet"


@dataclass
class PricePanel:
    prices: pd.DataFrame                 # dates × tickers, adjusted closes
    as_of: str                           # oldest fetch date across tickers (ISO)
    sources: dict[str, str]              # ticker -> cache|network|fixture|snapshot
    missing: list[str] = field(default_factory=list)   # requested, no data at all
    first_dates: dict[str, str] = field(default_factory=dict)  # ticker -> first close
    stale: bool = False
    notes: list[str] = field(default_factory=list)


class PriceCache:
    def __init__(self, cache_dir: Path | None = None):
        self.dir = Path(cache_dir or settings.cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._meta_path = self.dir / META_FILE

    # ---------- metadata ----------
    def _load_meta(self) -> dict:
        if self._meta_path.exists():
            return json.loads(self._meta_path.read_text())
        return {}

    def _save_meta(self, meta: dict) -> None:
        self._meta_path.write_text(json.dumps(meta, indent=1, sort_keys=True))

    # ---------- per-ticker series ----------
    def get_series(self, ticker: str) -> pd.Series | None:
        path = self.dir / _safe_name(ticker)
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        s = df["close"]
        s.index = pd.DatetimeIndex(s.index)
        s.name = ticker
        return s

    def put_series(
        self, ticker: str, series: pd.Series, source: str,
        fetched_at: str | None = None,
    ) -> None:
        series = series.dropna().sort_index()
        if series.empty:
            return
        df = pd.DataFrame({"close": series.values},
                          index=pd.DatetimeIndex(series.index).tz_localize(None))
        df.to_parquet(self.dir / _safe_name(ticker))
        meta = self._load_meta()
        meta[ticker] = {
            "fetched_at": fetched_at or datetime.now(timezone.utc).isoformat(),
            "source": source,
            "first": str(df.index[0].date()),
            "last": str(df.index[-1].date()),
        }
        self._save_meta(meta)

    def info(self, ticker: str) -> dict | None:
        return self._load_meta().get(ticker)

    def tickers(self) -> list[str]:
        return sorted(self._load_meta().keys())


def _fetch_yfinance(tickers: list[str], start: str, end: str | None) -> pd.DataFrame:
    """Chunked yfinance download of dividend/split-adjusted closes."""
    import yfinance as yf

    frames = []
    for i in range(0, len(tickers), 50):
        chunk = tickers[i : i + 50]
        raw = yf.download(
            chunk, start=start, end=end, auto_adjust=True,
            progress=False, threads=True, group_by="column",
        )
        if raw is None or raw.empty:
            continue
        closes = raw["Close"] if "Close" in raw else raw
        if isinstance(closes, pd.Series):
            closes = closes.to_frame(chunk[0])
        frames.append(closes)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1)
    out.index = pd.DatetimeIndex(out.index).tz_localize(None)
    return out.sort_index()


def offline_mode() -> bool:
    return os.environ.get("HONEST_OFFLINE", "") == "1"


def get_panel(
    tickers: list[str],
    start: str,
    end: str | None = None,
    refresh: bool = False,
    lead_in_days: int = 1200,
    cache: PriceCache | None = None,
) -> PricePanel:
    """Assemble a price panel with lead-in history for signal lookbacks.

    `tickers` must already include the benchmark ticker(s). `start`/`end`
    bound the *simulation* window; the panel itself reaches back lead_in_days
    calendar days for lookbacks.
    """
    cache = cache or PriceCache()
    tickers = list(dict.fromkeys(tickers))
    fetch_start = (pd.Timestamp(start) - timedelta(days=lead_in_days)).date().isoformat()

    meta = cache._load_meta()
    now = datetime.now(timezone.utc)
    to_fetch: list[str] = []
    for tk in tickers:
        info = meta.get(tk)
        if info is None:
            to_fetch.append(tk)
            continue
        fetched_at = datetime.fromisoformat(info["fetched_at"])
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=timezone.utc)
        if refresh or (now - fetched_at) > timedelta(hours=CACHE_STALE_HOURS):
            to_fetch.append(tk)

    notes: list[str] = []
    fetch_failed = False
    if to_fetch and not offline_mode():
        try:
            fresh = _fetch_yfinance(to_fetch, fetch_start, end)
            for tk in to_fetch:
                if tk in fresh.columns and fresh[tk].dropna().size > 0:
                    existing = cache.get_series(tk)
                    series = fresh[tk].dropna()
                    if existing is not None:
                        # keep older history the new fetch didn't cover
                        series = pd.concat(
                            [existing[existing.index < series.index[0]], series]
                        )
                    cache.put_series(tk, series, source="network")
        except Exception as exc:  # network/rate-limit → fall back to cache
            fetch_failed = True
            notes.append(
                f"yfinance fetch failed ({type(exc).__name__}); using cached "
                f"data where available"
            )
    elif to_fetch and offline_mode():
        fetch_failed = any(meta.get(tk) is None for tk in to_fetch)

    meta = cache._load_meta()
    columns: dict[str, pd.Series] = {}
    missing: list[str] = []
    sources: dict[str, str] = {}
    first_dates: dict[str, str] = {}
    fetch_dates: list[str] = []
    stale = False
    for tk in tickers:
        s = cache.get_series(tk)
        if s is None or s.dropna().empty:
            missing.append(tk)
            continue
        s = s[(s.index >= pd.Timestamp(fetch_start))
              & ((end is None) | (s.index <= pd.Timestamp(end or s.index[-1])))]
        if s.dropna().empty:
            missing.append(tk)
            continue
        columns[tk] = s
        info = meta.get(tk, {})
        sources[tk] = info.get("source", "cache")
        fetched = info.get("fetched_at", "")
        fetch_dates.append(fetched)
        first = s.dropna().index[0]
        first_dates[tk] = str(first.date())
        fetched_dt = datetime.fromisoformat(fetched) if fetched else None
        if fetched_dt is not None and fetched_dt.tzinfo is None:
            fetched_dt = fetched_dt.replace(tzinfo=timezone.utc)
        if fetched_dt is None or (now - fetched_dt) > timedelta(hours=CACHE_STALE_HOURS):
            stale = True

    if not columns:
        raise RuntimeError(
            "no price data available for any requested ticker — "
            "run `make seed` to load the offline fixtures, or check the network"
        )

    prices = pd.DataFrame(columns).sort_index()
    prices = prices.dropna(how="all")
    # The trading calendar is SPY's own sessions (dsl/conventions.py). A
    # spurious weekend/holiday row from one bad ticker would otherwise hijack
    # month-end rebalances and inflate day counts (confirmed audit finding).
    if "SPY" in prices.columns:
        spy_valid = prices["SPY"].notna()
        dropped = int((~spy_valid).sum())
        if dropped:
            notes.append(
                f"{dropped} non-SPY-session row(s) dropped — the calendar is "
                f"SPY's trading days"
            )
        prices = prices[spy_valid]
    as_of = min(fetch_dates)[:10] if fetch_dates else "unknown"
    if fetch_failed:
        stale = True
    if missing:
        notes.append(
            f"{len(missing)} ticker(s) had no price data and were excluded: "
            + ", ".join(sorted(missing)[:20])
        )
    return PricePanel(
        prices=prices, as_of=as_of, sources=sources, missing=missing,
        first_dates=first_dates, stale=stale, notes=notes,
    )


# ---------------- snapshots (Gap #2) ----------------

def panel_content_hash(prices: pd.DataFrame) -> str:
    """Deterministic content hash of the exact panel used by a run."""
    csv_bytes = prices.to_csv(float_format="%.10g").encode()
    return hashlib.sha256(csv_bytes).hexdigest()


def save_snapshot(panel: PricePanel, run_dir: Path) -> tuple[str, str]:
    """Freeze the panel; returns (path, sha256).

    Panel BYTES are stored content-addressed in `data/panels/<hash>.parquet`
    and shared by every run that froze an identical panel; the run directory
    keeps only its own metadata pointing at that blob. Most users test the
    same universe and window (everyone backtests S&P 500 momentum), so this is
    the difference between one stored panel and one per run — without weakening
    reproducibility at all, since the hash IS the identity of the bytes.

    Runs written before this change keep a real `panel.parquet` inside their
    own directory; `load_snapshot` still reads those, so old journals replay.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    content_hash = panel_content_hash(panel.prices)

    blob_dir = Path(settings.panels_dir)
    blob_dir.mkdir(parents=True, exist_ok=True)
    blob = blob_dir / f"{content_hash}.parquet"
    if not blob.exists():
        # Write then rename so a crash mid-write can never leave a truncated
        # blob that later runs would trust on the strength of its filename.
        tmp = blob.with_suffix(f".{os.getpid()}.tmp")
        panel.prices.to_parquet(tmp)
        tmp.replace(blob)

    meta = {
        "as_of": panel.as_of,
        "sources": panel.sources,
        "missing": panel.missing,
        "first_dates": panel.first_dates,
        "stale": panel.stale,
        "notes": panel.notes,
        "content_hash": content_hash,
        "panel_blob": str(blob),
    }
    (run_dir / "panel_meta.json").write_text(json.dumps(meta, indent=1))
    return str(blob), content_hash


def load_snapshot(run_dir: Path) -> PricePanel:
    meta_path = run_dir / "panel_meta.json"
    legacy = run_dir / "panel.parquet"
    # Legacy runs stored the parquet in the run directory itself; prefer that
    # when present so journals written before content-addressing still replay.
    if legacy.exists():
        path = legacy
    else:
        ref = json.loads(meta_path.read_text()).get("panel_blob")
        if not ref:
            raise RuntimeError(f"run snapshot at {run_dir} has no panel to replay")
        path = Path(ref)
        if not path.exists():
            raise RuntimeError(f"frozen panel {path} is missing — cannot replay")
    prices = pd.read_parquet(path)
    prices.index = pd.DatetimeIndex(prices.index)
    meta = json.loads((run_dir / "panel_meta.json").read_text())
    stored = meta.get("content_hash")
    actual = panel_content_hash(prices)
    notes = list(meta.get("notes", []))
    if stored and stored != actual:
        raise RuntimeError(
            "snapshot content hash mismatch — the frozen panel was modified"
        )
    return PricePanel(
        prices=prices,
        as_of=meta.get("as_of", "unknown"),
        sources={tk: "snapshot" for tk in prices.columns},
        missing=meta.get("missing", []),
        first_dates=meta.get("first_dates", {}),
        stale=False,  # a snapshot is exact by definition; staleness was disclosed at run time
        notes=notes,
    )
