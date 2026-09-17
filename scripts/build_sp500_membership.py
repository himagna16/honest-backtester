"""Rebuild point-in-time S&P 500 membership from pinned, cited sources.

Why this exists: a backtest over past years on TODAY's constituent list buys
companies before they were in the index. The biggest momentum winners of
2023-2026 (APP, CVNA, VRT, PLTR…) joined the S&P 500 only AFTER their run-ups,
so a "full S&P 500" momentum run quietly harvested them from day one.

Sources (free, citable, pinned to exact revisions so a rebuild is reproducible):
  * Current constituents: Wikipedia "List of S&P 500 companies", revision
    LIST_REVID — the anchor we walk backward from.
  * Dated changes: Wikipedia "Historical components of the S&P 500", revision
    CHANGES_REVID. Every row cites an S&P Dow Jones Indices announcement; the
    citation URLs are carried into fixtures/sp500_membership/changes.csv.
  Wikipedia text is CC BY-SA 4.0 — see fixtures/sp500_membership/README.md for
  attribution. Prices for removed names come from yfinance (the same source as
  every other price fixture in this repo).

What it writes (fixtures/sp500_membership/):
  anchor.csv       the pinned constituent list (Symbol, Security, Date added)
  changes.csv      the pinned change table, one row per change, with citations
  membership.csv   derived: one row per membership spell, with the price
                   symbol the engine should use (blank = no usable free history)
  meta.json        revisions, coverage start, content hash of membership.csv
  VALIDATION.txt   the consistency checks below, regenerated on every build
and fixtures/sp500_pit_prices.parquet (removed/renamed names' closes) with
--fetch-prices.

corrections.csv in the same folder is the ONLY hand-curated input: ticker
aliases where the change table uses a symbol the anchor no longer does, and
price-symbol overrides where a removed company's history lives under another
ticker (or under a ticker that now belongs to a different company). Every row
carries a note saying why.

Usage:
  .venv/bin/python scripts/build_sp500_membership.py              # membership only
  .venv/bin/python scripts/build_sp500_membership.py --fetch-prices
  .venv/bin/python scripts/build_sp500_membership.py --crosscheck path/to/fja.csv
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "fixtures" / "sp500_membership"
PRICES_OUT = REPO / "fixtures" / "sp500_pit_prices.parquet"
HTML_CACHE = REPO / "data" / "membership_src"   # gitignored; raw pages
UA = {"User-Agent": "honest-backtester/0.1 (personal research tool)"}

LIST_TITLE = "List_of_S%26P_500_companies"
LIST_REVID = 1370105675          # 2026-08-19T03:43:37Z
CHANGES_TITLE = "Historical_components_of_the_S%26P_500"
CHANGES_REVID = 1370891103       # 2026-08-23T16:26:12Z

# Membership is only CLAIMED from this date. Before ~2014 the change table is
# explicitly "selected" changes and disagrees materially with the independent
# fja05680/sp500 history (see VALIDATION.txt), so earlier windows are reported
# as not covered rather than silently trusted.
COVERAGE_START = "2015-01-01"
PRICE_FETCH_START = "2014-01-01"
# A price series that begins more than this long after a spell starts belongs
# to a later listing (a reused ticker), not to the member.
LATE_START_TOLERANCE_DAYS = 10


def permalink(title: str, revid: int) -> str:
    return f"https://en.wikipedia.org/w/index.php?title={title}&oldid={revid}"


def fetch_html(title: str, revid: int) -> str:
    HTML_CACHE.mkdir(parents=True, exist_ok=True)
    path = HTML_CACHE / f"{revid}.html"
    if path.exists():
        return path.read_text()
    resp = requests.get(permalink(title, revid), headers=UA, timeout=60)
    resp.raise_for_status()
    path.write_text(resp.text)
    return resp.text


def clean_ticker(raw: str | None) -> str | None:
    """'ALLE |' → 'ALLE', 'BRK.B' → 'BRK.B' (dots kept here; the price symbol
    is normalized separately), '' → None."""
    if raw is None:
        return None
    t = re.sub(r"[^A-Za-z0-9.]", "", str(raw)).upper()
    return t or None


def day(x) -> str | None:
    """A spell bound as 'YYYY-MM-DD', or None when open-ended (None/NaN/'')."""
    if x is None or (isinstance(x, float) and pd.isna(x)) or x == "":
        return None
    return str(x)[:10]


def price_symbol_for(label: str) -> str:
    """yfinance spelling: BRK.B → BRK-B (dsl/universes.normalize_ticker)."""
    return label.replace(".", "-")


# ---------------------------------------------------------------- parsing

def parse_anchor(html: str) -> pd.DataFrame:
    table = pd.read_html(io.StringIO(html))[0]
    need = {"Symbol", "Security", "Date added"}
    if not need.issubset(table.columns):
        raise RuntimeError(f"anchor table columns changed: {list(table.columns)}")
    out = table[["Symbol", "Security", "Date added"]].copy()
    out["Symbol"] = out["Symbol"].map(clean_ticker)
    if out["Symbol"].duplicated().any():
        raise RuntimeError("anchor list has duplicate symbols")
    return out.sort_values("Symbol").reset_index(drop=True)


def parse_changes(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", class_="wikitable")
    refs: dict[str, str] = {}
    for li in soup.select("ol.references > li"):
        links = [a.get("href") for a in li.find_all("a", class_="external")]
        if li.get("id") and links:
            refs[li["id"]] = links[0]

    rows = []
    for tr in table.find_all("tr"):
        tds = tr.find_all("td", recursive=False)
        if not tds:
            continue                       # header rows
        if any(td.get("rowspan") or td.get("colspan") for td in tds):
            raise RuntimeError("change table uses row/colspan in its body — "
                               "the parser assumes one cell per column")
        # Two rows (2015-03-23 SLG/NBR, 2016-11-01 ARNC/AA) have no Refs cell.
        if len(tds) not in (6, 7):
            raise RuntimeError(f"unexpected change-table row width {len(tds)}")
        text = [td.get_text(" ", strip=True) for td in tds]
        cites = []
        for a in (tds[6].select("sup a[href^='#cite_note']") if len(tds) == 7 else []):
            url = refs.get(a["href"][1:])
            if url and url not in cites:
                cites.append(url)
        rows.append({
            "effective_date": datetime.strptime(text[0], "%B %d, %Y").date().isoformat(),
            "added_ticker": clean_ticker(text[1]),
            "added_security": text[2].rstrip(" |") or None,
            "removed_ticker": clean_ticker(text[3]),
            "removed_security": text[4].rstrip(" |") or None,
            "reason": text[5] or None,
            "citations": " ".join(cites) or None,
        })
    df = pd.DataFrame(rows)
    return df.sort_values("effective_date", ascending=False, kind="mergesort") \
             .reset_index(drop=True)


# ---------------------------------------------------------- reconstruction

def load_corrections() -> tuple[dict, list[dict], list[dict]]:
    """→ (row aliases, price-symbol rules, price splits). See README.md."""
    corr = pd.read_csv(OUT / "corrections.csv", dtype=str).fillna("")
    aliases = {}
    price_rules = []
    splits = []
    for r in corr.to_dict("records"):
        if not r["note"].strip():
            raise RuntimeError(f"correction without a note: {r}")
        if r["kind"] == "row_alias":
            aliases[(r["label"], r["date"])] = r["target"]
        elif r["kind"] == "price_symbol":
            price_rules.append(r)
        elif r["kind"] == "price_split":
            splits.append(r)
        else:
            raise RuntimeError(f"unknown correction kind: {r['kind']}")
    return aliases, price_rules, splits


def reconstruct(anchor: pd.DataFrame, changes: pd.DataFrame, aliases: dict,
                anchor_date: str) -> tuple[pd.DataFrame, list[str]]:
    """Walk backward from the anchor list, undoing each dated change.

    A change effective on day d means the added name is a member FROM d and the
    removed name is a member UNTIL d (exclusive). Walking backward, undo every
    addition on d before undoing any removal on d, so a same-day
    remove-and-re-add of one ticker (FOX/FOXA on 2019-03-19) becomes two spells.
    """
    security = dict(zip(anchor["Symbol"], anchor["Security"]))
    open_end: dict[str, str | None] = {s: None for s in anchor["Symbol"]}
    spells: list[dict] = []
    problems: list[str] = []
    used_aliases = set()

    for d in sorted(changes["effective_date"].unique(), reverse=True):
        day = changes[changes["effective_date"] == d]
        for r in day.itertuples():
            if pd.isna(r.added_ticker):
                continue
            key = (r.added_ticker, d)
            label = aliases.get(key, r.added_ticker)
            if key in aliases:
                used_aliases.add(key)
            if label in open_end:
                spells.append({"label": label, "security": security.get(label) or r.added_security,
                               "start": d, "end": open_end.pop(label)})
            else:
                problems.append(f"{d}: added {r.added_ticker} ({r.added_security}) "
                                f"but it is not a member afterwards")
        for r in day.itertuples():
            if pd.isna(r.removed_ticker):
                continue
            key = (r.removed_ticker, d)
            label = aliases.get(key, r.removed_ticker)
            if key in aliases:
                used_aliases.add(key)
            if label in open_end:
                problems.append(f"{d}: removed {r.removed_ticker} ({r.removed_security}) "
                                f"but it is already a member before this date")
                spells.append({"label": label, "security": security.get(label),
                               "start": d, "end": open_end[label]})
            open_end[label] = d
            security[label] = r.removed_security or security.get(label)

    for label, end in open_end.items():
        spells.append({"label": label, "security": security.get(label), "start": None, "end": end})

    unused = set(aliases) - used_aliases
    if unused:
        problems.append(f"corrections.csv aliases never matched a change row: {sorted(unused)}")

    for sp in spells:           # None (not NaN) for open-ended bounds
        sp["start"] = sp["start"] or None
        sp["end"] = sp["end"] or None
    return pd.DataFrame(spells, dtype=object), problems


def apply_price_splits(spells: pd.DataFrame, splits: list[dict]) -> pd.DataFrame:
    """Cut a spell in two at `date` when its price symbol only becomes the
    member's own history from that date (e.g. a series that carries a
    different predecessor company before a merger). The earlier piece takes
    `target` as its price symbol — blank means no usable history. Membership
    itself is unchanged: both pieces are still member days."""
    rows = spells.to_dict("records")
    for rule in splits:
        d = rule["date"]
        hits = [i for i, r in enumerate(rows) if r["label"] == rule["label"]
                and (day(r["start"]) is None or day(r["start"]) < d)
                and (day(r["end"]) is None or day(r["end"]) > d)]
        if len(hits) != 1:
            raise RuntimeError(f"price split must match exactly one spell: {rule}")
        r = rows[hits[0]]
        before = {**r, "end": d, "price_symbol": rule["target"], "price_note": rule["note"],
                  "split": True}
        after = {**r, "start": d}
        rows[hits[0]] = after
        rows.append(before)
    return pd.DataFrame(rows)


def apply_price_rules(spells: pd.DataFrame, rules: list[dict]) -> pd.DataFrame:
    """Default price symbol = the label; a price_symbol rule matches the one
    spell with that label that ENDS on `date` (a rename or merger date). An
    empty target means the history is known to be unusable (no free series, or
    a different company's series)."""
    out = spells.copy()
    out["price_symbol"] = out["label"].map(price_symbol_for)
    out["price_note"] = ""
    out["split"] = False
    for rule in rules:
        hit = (out["label"] == rule["label"]) & (out["end"].map(day) == rule["date"])
        if int(hit.sum()) != 1:
            raise RuntimeError(f"price rule must match exactly one spell: {rule}")
        out.loc[hit, "price_symbol"] = rule["target"]
        out.loc[hit, "price_note"] = rule["note"]
    return out


def fetch_prices(symbols: list[str]) -> pd.DataFrame:
    import yfinance as yf

    frames = []
    for i in range(0, len(symbols), 50):
        chunk = symbols[i:i + 50]
        raw = yf.download(chunk, start=PRICE_FETCH_START, auto_adjust=True,
                          progress=False, threads=True, group_by="column")
        if raw is None or raw.empty:
            continue
        closes = raw["Close"] if "Close" in raw else raw
        if isinstance(closes, pd.Series):
            closes = closes.to_frame(chunk[0])
        frames.append(closes)
    px = pd.concat(frames, axis=1)
    px = px.loc[:, ~px.columns.duplicated()]
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    return px.sort_index()


def check_price_identity(spells: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    """Blank the price symbol where the series cannot be the member's own
    history: no data at all, or a series that starts well after the spell began
    (the ticker was reused by a later listing — SE, DOW, FOX, BTU, Q …)."""
    out = spells.copy()
    out["price_check"] = ""
    cov = pd.Timestamp(COVERAGE_START)
    for i, r in out.iterrows():
        sym = r["price_symbol"]
        end = pd.Timestamp(day(r["end"])) if day(r["end"]) else None
        if end is not None and end <= cov:
            out.at[i, "price_check"] = "before_coverage"
            out.at[i, "price_symbol"] = ""
            continue
        if not sym:
            out.at[i, "price_check"] = "unusable_by_correction"
            continue
        s = px[sym].dropna() if sym in px.columns else pd.Series(dtype=float)
        if s.empty:
            out.at[i, "price_check"] = "no_free_history"
            out.at[i, "price_symbol"] = ""
            continue
        need_from = max(pd.Timestamp(day(r["start"])) if day(r["start"]) else cov, cov)
        if s.index[0] > need_from + pd.Timedelta(days=LATE_START_TOLERANCE_DAYS):
            out.at[i, "price_check"] = f"reused_ticker_series_starts_{s.index[0].date()}"
            out.at[i, "price_symbol"] = ""
            continue
        out.at[i, "price_check"] = "ok"
    return out


def load_fixture_prices() -> pd.DataFrame:
    """The checked-in price fixtures. They were frozen in June 2026, when some
    names since removed (AVB) still had full histories on Yahoo; a fresh
    download can come back truncated for them, so fixtures win."""
    frames = []
    for name in ("sp500_prices.csv", "war_list_prices.csv", "extras_prices.csv"):
        path = REPO / "fixtures" / name
        if path.exists():
            frames.append(pd.read_csv(path, index_col=0, parse_dates=True))
    px = pd.concat(frames, axis=1, sort=True)
    return px.loc[:, ~px.columns.duplicated()]


def jump_screen(spells: pd.DataFrame, px: pd.DataFrame, limit: float = 0.40) -> list[str]:
    """One-day moves beyond ±limit on member days — the signature of a series
    that switches company (or misses a split) mid-spell."""
    out = []
    cov = pd.Timestamp(COVERAGE_START)
    for r in spells.itertuples():
        if not r.price_symbol or r.price_symbol not in px.columns:
            continue
        lo = max(pd.Timestamp(day(r.start)) if day(r.start) else cov, cov)
        hi = pd.Timestamp(day(r.end)) if day(r.end) else px.index[-1]
        ret = px[r.price_symbol].dropna().pct_change().loc[lo:hi]
        for when, v in ret[ret.abs() > limit].items():
            out.append(f"  {r.label} ({r.price_symbol}) {when.date()}: {v:+.0%}")
    return out


# ------------------------------------------------------------- validation

def members_on(spells: pd.DataFrame, when: str) -> pd.DataFrame:
    starts, ends = spells["start"].map(day), spells["end"].map(day)
    started = starts.isna() | (starts.fillna("") <= when)
    not_ended = ends.isna() | (ends.fillna("9999") > when)
    return spells[started & not_ended]


def validation_report(spells: pd.DataFrame, anchor: pd.DataFrame, problems: list[str],
                      crosscheck: Path | None, jumps: list[str]) -> str:
    lines = [f"Built {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
             f"anchor: {permalink(LIST_TITLE, LIST_REVID)}",
             f"changes: {permalink(CHANGES_TITLE, CHANGES_REVID)}",
             f"coverage_start: {COVERAGE_START}", ""]

    in_cov = [p for p in problems if p[:10] >= COVERAGE_START or not p[:4].isdigit()]
    lines.append(f"Reconstruction inconsistencies since coverage start: {len(in_cov)}")
    lines += [f"  {p}" for p in in_cov]
    lines.append(f"Inconsistencies before coverage start (not relied on): "
                 f"{len(problems) - len(in_cov)}")
    lines.append("")

    lines.append("Member count (securities) and usable price history, per year-end:")
    lines.append(f"  {'date':10}  {'members':>7}  {'priced':>6}  {'share':>5}")
    for y in range(int(COVERAGE_START[:4]), 2027):
        when = f"{y}-12-31" if y < 2026 else "2026-08-01"
        m = members_on(spells, when)
        priced = int(m["price_symbol"].fillna("").astype(bool).sum())
        lines.append(f"  {when:10}  {len(m):7d}  {priced:6d}  {priced / len(m):5.1%}")
    lines.append("")

    # Anchor "Date added" vs the reconstructed join date, for joins in coverage.
    cur = spells[spells["end"].map(day).isna()].drop_duplicates("label").set_index("label")
    mism = []
    checked = 0
    for r in anchor.itertuples():
        added = str(r._3)[:10]
        if not re.match(r"\d{4}-\d{2}-\d{2}", added) or added < COVERAGE_START:
            continue
        checked += 1
        got = day(cur.loc[r.Symbol, "start"]) if r.Symbol in cur.index else None
        if got != added:
            mism.append(f"  {r.Symbol}: list says added {added}, change table gives {got}")
    lines.append(f"'Date added' cross-check (current members added since coverage start): "
                 f"{checked - len(mism)}/{checked} agree")
    lines += mism
    lines.append("")

    lines.append(f"One-day moves beyond ±40% on member days with usable prices: {len(jumps)}")
    lines += jumps
    lines.append("")

    if crosscheck is not None:
        lines += crosscheck_fja(spells, crosscheck)
    return "\n".join(lines) + "\n"


def crosscheck_fja(spells: pd.DataFrame, path: Path) -> list[str]:
    """Compare against fja05680/sp500 (MIT; its pre-2019 history comes from
    Andreas Clenow's Norgate-derived list, independent of Wikipedia). Its
    tickers are as-of each date, so renames show up as complementary pairs;
    what is left after pairing is a real disagreement."""
    fja = pd.read_csv(path, parse_dates=["date"])
    fja = fja[fja["date"] >= COVERAGE_START].reset_index(drop=True)
    only_f: dict[str, set] = {}
    only_w: dict[str, set] = {}
    for i, r in fja.iterrows():
        f = {t.strip().replace(".", "-") for t in r["tickers"].split(",")}
        w = {s.replace(".", "-") for s in members_on(spells, str(r["date"].date()))["label"]}
        for t in f - w:
            only_f.setdefault(t, set()).add(i)
        for t in w - f:
            only_w.setdefault(t, set()).add(i)
    lines = [f"fja05680/sp500 cross-check over {len(fja)} dated snapshots since {COVERAGE_START}:"]
    by_rows: dict[frozenset, list[str]] = {}
    for t, s in only_w.items():
        by_rows.setdefault(frozenset(s), []).append(t)
    paired = []
    for t, s in list(only_f.items()):
        match = by_rows.get(frozenset(s))
        if match and len(match) == 1:
            paired.append(f"{t}->{match[0]}")
            only_w.pop(match[0], None)
            only_f.pop(t)
    lines.append(f"  same-dates ticker pairs (renames / labelling): {len(paired)}")
    lines.append("    " + ", ".join(sorted(paired)))
    for name, d in (("only in fja05680", only_f), ("only in this build", only_w)):
        short = {t: s for t, s in d.items() if len(s) <= 3}
        long = {t: s for t, s in d.items() if len(s) > 3}
        lines.append(f"  {name}: {len(short)} ticker(s) on <=3 snapshots "
                     f"(effective-date convention), {len(long)} longer:")
        for t, s in sorted(long.items(), key=lambda kv: min(kv[1])):
            lo, hi = fja.loc[min(s), "date"].date(), fja.loc[max(s), "date"].date()
            lines.append(f"    {t:6} {lo}..{hi} ({len(s)} snapshots)")
    lines.append("")
    return lines


# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch-prices", action="store_true",
                    help="download closes for every spell's price symbol and "
                         "write fixtures/sp500_pit_prices.parquet")
    ap.add_argument("--crosscheck", type=Path, default=None,
                    help="fja05680/sp500 'Historical Components & Changes (Updated).csv'")
    args = ap.parse_args()

    anchor_html = fetch_html(LIST_TITLE, LIST_REVID)
    changes_html = fetch_html(CHANGES_TITLE, CHANGES_REVID)
    anchor = parse_anchor(anchor_html)
    changes = parse_changes(changes_html)
    OUT.mkdir(parents=True, exist_ok=True)
    anchor.to_csv(OUT / "anchor.csv", index=False)
    changes.to_csv(OUT / "changes.csv", index=False)

    aliases, price_rules, splits = load_corrections()
    anchor_date = "2026-08-19"
    spells, problems = reconstruct(anchor, changes, aliases, anchor_date)
    spells = apply_price_rules(spells, price_rules)
    spells = apply_price_splits(spells, splits)

    candidates = sorted({s for s in spells["price_symbol"] if s})
    px_path = HTML_CACHE / "prices.parquet"
    if args.fetch_prices or not px_path.exists():
        fetch_prices(candidates).to_parquet(px_path)
    fresh = pd.read_parquet(px_path)
    fixture = load_fixture_prices()
    # Fixture columns win; fresh downloads fill only symbols the fixtures lack.
    px = pd.concat([fixture, fresh[[c for c in fresh.columns if c not in fixture.columns]]],
                   axis=1).sort_index()
    spells = check_price_identity(spells, px)
    jumps = jump_screen(spells, px)

    spells = spells.sort_values(["label", "start"], na_position="first", kind="mergesort")
    cols = ["label", "security", "start", "end", "price_symbol", "price_check", "price_note"]
    spells[cols].to_csv(OUT / "membership.csv", index=False)
    blob = (OUT / "membership.csv").read_bytes()

    # Only what the checked-in fixtures do not already carry. Rewritten from the
    # download cache on every build so it always matches membership.csv.
    need = sorted({s for s in spells["price_symbol"].fillna("") if s} - set(fixture.columns))
    extra = fresh[need].loc[PRICE_FETCH_START:].dropna(how="all")
    extra.to_parquet(PRICES_OUT)
    print(f"wrote {len(need)} price series → {PRICES_OUT.relative_to(REPO)}")

    meta = {
        "index": "sp500",
        "anchor_source": permalink(LIST_TITLE, LIST_REVID),
        "anchor_date": anchor_date,
        "changes_source": permalink(CHANGES_TITLE, CHANGES_REVID),
        "changes_through": str(changes["effective_date"].max()),
        "coverage_start": COVERAGE_START,
        "license": "Wikipedia text CC BY-SA 4.0 (see README.md)",
        "membership_sha256": hashlib.sha256(blob).hexdigest(),
        "built": date.today().isoformat(),
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    report = validation_report(spells, anchor, problems, args.crosscheck, jumps)
    (OUT / "VALIDATION.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    sys.exit(main())
