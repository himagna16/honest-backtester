"""Re-run the landing page's two S&P 500 figures on point-in-time membership.

The landing and README quote two results from the founding session's scripts
(~/Downloads/momentum_backtest, not part of this repo), both run on TODAY's
S&P 500 list:
  * multifactor.py — 12-1 momentum + low volatility, top 25, monthly, 10 bps
  * reversal.py    — buy yesterday's losers for one day, at 0 / 5 / 20 bps

This ports each script's logic line for line, with one change: on every date
the cross-section is restricted to the names that were S&P 500 members that
day (ranks are taken within members only). It runs twice — on today's list,
where it must reproduce the scripts' printed numbers exactly, and on
point-in-time membership. fixtures/sp500_prices.csv is the same file the
scripts read; former members' closes come from fixtures/sp500_pit_prices.parquet.

Writes docs/founding_story_pit.md and .json; every number in them is computed here.

Usage: .venv/bin/python scripts/founding_story_pit.py
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backend"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dsl.membership import load_membership  # noqa: E402

FIX = REPO / "fixtures"
TOP_N, COST_BPS, BENCH = 25, 10, "SPY"
COST_LEVELS = [0.0, 0.0005, 0.002]


def load_prices() -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    today = pd.read_csv(FIX / "sp500_prices.csv", index_col=0, parse_dates=True)
    extra = pd.read_parquet(FIX / "sp500_pit_prices.parquet")
    extra.index = pd.DatetimeIndex(extra.index)
    extra = extra[[c for c in extra.columns if c not in today.columns]].reindex(today.index)
    today_list = [c for c in today.columns if c != BENCH]
    return today, today_list, pd.concat([today, extra], axis=1)


# ------------------------------------------------------------ multifactor.py

def multifactor(closes: pd.DataFrame, eligible) -> dict:
    """multifactor.py with a membership mask. `eligible(index, cols)` → bool frame."""
    spy_px = closes[BENCH]
    universe = closes[[c for c in closes.columns if c != BENCH]]
    m = universe.resample("ME").last()
    m_spy = spy_px.resample("ME").last()
    mom = m.shift(1) / m.shift(12) - 1
    vol_daily = universe.pct_change().rolling(252).std()
    vol_m = vol_daily.resample("ME").last()

    member = eligible(m.index, list(m.columns))
    mom_rank = mom.where(member).rank(axis=1, pct=True)
    vol_rank = vol_m.where(member).rank(axis=1, pct=True, ascending=False)
    composite = (mom_rank + vol_rank) / 2
    fwd = m.pct_change().shift(-1)

    rows, prev = [], set()
    for t in m.index:
        scores = composite.loc[t].dropna()
        if len(scores) < TOP_N:
            rows.append((t, np.nan, 0.0))
            continue
        picks = list(scores.sort_values(ascending=False).head(TOP_N).index)
        gross = fwd.loc[t, picks].mean()
        turnover = len(set(picks) ^ prev)
        cost = turnover * COST_BPS / 10_000 / TOP_N
        rows.append((t, gross, cost))
        prev = set(picks)
    res = pd.DataFrame(rows, columns=["date", "gross", "cost"]).set_index("date")
    res["net"] = res["gross"] - res["cost"]
    res = res.dropna(subset=["net"])
    spy_ret = m_spy.pct_change().shift(-1).reindex(res.index)

    def stats(r):
        r = r.dropna()
        eq = (1 + r).cumprod()
        years = len(r) / 12
        vol = r.std() * np.sqrt(12)
        return {"total": float(eq.iloc[-1] - 1), "cagr": float(eq.iloc[-1] ** (1 / years) - 1),
                "vol": float(vol), "sharpe": float((r.mean() * 12) / vol),
                "max_dd": float((eq / eq.cummax() - 1).min())}

    return {"window": [str(res.index[0].date()), str(res.index[-1].date())],
            "months": len(res), "net": stats(res["net"]), "gross": stats(res["gross"]),
            "spy": stats(spy_ret)}


# --------------------------------------------------------------- reversal.py

def reversal(closes: pd.DataFrame, eligible) -> dict:
    u = closes[[c for c in closes.columns if c != BENCH]]
    spy = closes[BENCH]
    ret = u.pct_change()
    member = eligible(ret.index, list(ret.columns))
    nxt = ret.shift(-1)
    down, up = (ret < 0) & member, (ret > 0) & member
    baskets = {
        "BuyLosers": nxt.where(down).mean(axis=1),
        "BuyWinners": nxt.where(up).mean(axis=1),
        "BuyAll": nxt.where(member).mean(axis=1),
    }
    res = pd.DataFrame(baskets).dropna()

    def stats(r):
        r = r.dropna()
        eq = (1 + r).cumprod()
        years = len(r) / 252
        vol = r.std() * np.sqrt(252)
        return {"total": float(eq.iloc[-1] - 1), "cagr": float(eq.iloc[-1] ** (1 / years) - 1),
                "sharpe": float((r.mean() * 252) / vol)}

    out = {"window": [str(res.index[0].date()), str(res.index[-1].date())], "days": len(res)}
    for name in baskets:
        out[name] = {f"{int(c * 10000)}bps": stats(res[name] - c) for c in COST_LEVELS}
    out["SPY"] = stats(spy.pct_change().reindex(res.index))
    return out


def main() -> None:
    today_px, today_list, pit_px = load_prices()
    membership = load_membership("sp500")
    today_set = set(today_list)

    def today_mask(index, cols):
        return pd.DataFrame({c: c in today_set for c in cols}, index=index)

    def pit_mask(index, cols):
        return membership.eligibility(pd.DatetimeIndex(index), cols)

    # The months before coverage cannot be claimed point-in-time; multifactor's
    # first scored month is 2016-01 (12 months of lookback), inside coverage.
    results = {
        "generated": date.today().isoformat(),
        "membership": membership.provenance(),
        "multifactor": {"today": multifactor(today_px, today_mask),
                        "pit": multifactor(pit_px, pit_mask)},
        "reversal": {"today": reversal(today_px, today_mask),
                     "pit": reversal(pit_px.loc[membership.coverage_start:], pit_mask)},
    }
    # The port must reproduce the original scripts before its PIT leg means anything.
    mf, rv = results["multifactor"]["today"], results["reversal"]["today"]
    checks = {
        "multifactor net total +163.1%": round(mf["net"]["total"], 3) == 1.631,
        "multifactor net CAGR 9.8%": round(mf["net"]["cagr"], 3) == 0.098,
        "multifactor SPY total +362.0%": round(mf["spy"]["total"], 3) == 3.620,
        "reversal losers 0bps 748%/1.05": (round(rv["BuyLosers"]["0bps"]["total"], 2) == 7.48
                                            and round(rv["BuyLosers"]["0bps"]["sharpe"], 2) == 1.05),
        "reversal losers 5bps 103%/0.41": (round(rv["BuyLosers"]["5bps"]["total"], 2) == 1.03
                                            and round(rv["BuyLosers"]["5bps"]["sharpe"], 2) == 0.41),
        "reversal SPY 344%/0.83": (round(rv["SPY"]["total"], 2) == 3.44
                                   and round(rv["SPY"]["sharpe"], 2) == 0.83),
    }
    results["reproduces_original_scripts"] = checks
    for k, ok in checks.items():
        print(("ok   " if ok else "FAIL ") + k)

    (REPO / "docs" / "founding_story_pit.json").write_text(json.dumps(results, indent=1) + "\n")
    (REPO / "docs" / "founding_story_pit.md").write_text(render(results))
    mp, rp = results["multifactor"]["pit"], results["reversal"]["pit"]
    print(f"multifactor PIT: net {mp['net']['total']:+.1%} CAGR {mp['net']['cagr']:.1%} "
          f"vs SPY {mp['spy']['total']:+.1%} CAGR {mp['spy']['cagr']:.1%}")
    for bps in ("0bps", "5bps", "20bps"):
        print(f"reversal PIT losers {bps}: {rp['BuyLosers'][bps]['total']:+.0%} / "
              f"{rp['BuyLosers'][bps]['sharpe']:.2f}")
    print(f"reversal PIT SPY: {rp['SPY']['total']:+.0%} / {rp['SPY']['sharpe']:.2f}")


def render(r: dict) -> str:
    mf, rv = r["multifactor"], r["reversal"]
    L = [
        "# The landing page's S&P 500 figures, on point-in-time membership",
        "",
        f"Generated {r['generated']} by `scripts/founding_story_pit.py`, which ports the founding "
        "session's `multifactor.py` and `reversal.py` line for line and restricts each date's "
        "cross-section to that day's S&P 500 members (ranks taken within members). The *today* "
        "column must reproduce the original scripts' printed output:",
        "",
    ]
    L += [f"- {'✓' if ok else '✗'} {k}" for k, ok in r["reproduces_original_scripts"].items()]
    L += [
        "",
        "## Momentum + low volatility, top 25, monthly, 10 bps (`multifactor.py`)",
        "",
        "| | Today's list | Point-in-time |",
        "|---|---:|---:|",
        f"| Window | {mf['today']['window'][0]} → {mf['today']['window'][1]} | "
        f"{mf['pit']['window'][0]} → {mf['pit']['window'][1]} |",
    ]
    for label, key in (("Total return (net)", "total"), ("CAGR (net)", "cagr"),
                       ("Sharpe (net)", "sharpe"), ("Max drawdown (net)", "max_dd")):
        fmt = (lambda v: f"{v:.2f}") if key == "sharpe" else (lambda v: f"{v:+.1%}")
        L.append(f"| {label} | {fmt(mf['today']['net'][key])} | {fmt(mf['pit']['net'][key])} |")
    L.append(f"| SPY total / CAGR | {mf['today']['spy']['total']:+.1%} / {mf['today']['spy']['cagr']:.1%} "
             f"| {mf['pit']['spy']['total']:+.1%} / {mf['pit']['spy']['cagr']:.1%} |")
    L += [
        "",
        "## Buy yesterday's losers for one day (`reversal.py`)",
        "",
        f"Point-in-time runs from {rv['pit']['window'][0]} (membership coverage starts "
        f"{r['membership']['coverage_start']}); today's list from {rv['today']['window'][0]}. "
        "Each cell is total return / Sharpe.",
        "",
        "| Basket | Today 0 bps | Today 5 bps | Today 20 bps | PIT 0 bps | PIT 5 bps | PIT 20 bps |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("BuyLosers", "BuyWinners", "BuyAll"):
        cells = [f"{rv[k][name][b]['total']:+.0%} / {rv[k][name][b]['sharpe']:.2f}"
                 for k in ("today", "pit") for b in ("0bps", "5bps", "20bps")]
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L.append(f"| SPY buy-and-hold | {rv['today']['SPY']['total']:+.0%} / {rv['today']['SPY']['sharpe']:.2f} "
             f"| | | {rv['pit']['SPY']['total']:+.0%} / {rv['pit']['SPY']['sharpe']:.2f} | | |")
    L += [
        "",
        "Caveats carried over from the original scripts: a name that stops trading mid-month "
        "drops out of that month's average; members with no free price history can't be "
        "picked (see `fixtures/sp500_membership/README.md`).",
        "",
    ]
    return "\n".join(L)


if __name__ == "__main__":
    main()
