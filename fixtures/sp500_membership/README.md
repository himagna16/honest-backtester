# Point-in-time S&P 500 membership

Who was actually in the S&P 500 on each day since **2015-01-01**, rebuilt from
free, cited sources, so a backtest only picks from companies that were
in the index on that date.

## Why

A momentum run over 2023–2026 on *today's* constituent list bought AppLovin,
Carvana, Vertiv and Palantir before they joined the index, at the start of the
run-ups that later got them added. Using today's members to test the past is
survivorship bias, and it produced +534% to +3560% "S&P 500" backtests in this
project.

## Sources

| What | Source | Pinned revision |
|---|---|---|
| Constituents on 2026-08-19 (the anchor) | Wikipedia, *List of S&P 500 companies* | [oldid 1370105675](https://en.wikipedia.org/w/index.php?title=List_of_S%26P_500_companies&oldid=1370105675) |
| Dated additions and removals | Wikipedia, *Historical components of the S&P 500* | [oldid 1370891103](https://en.wikipedia.org/w/index.php?title=Historical_components_of_the_S%26P_500&oldid=1370891103) |
| Prices for removed and renamed names | yfinance (the same source as the other price fixtures) | downloaded 2026-09-17 |

391 of the 407 change rows cite an S&P Dow Jones Indices announcement. Those
URLs are kept in the `citations` column of `changes.csv`. Both Wikipedia tables
are © Wikipedia contributors, licensed CC BY-SA 4.0. `anchor.csv` and
`changes.csv` are adapted from those revisions and carry the same license.

Independent cross-check: [fja05680/sp500](https://github.com/fja05680/sp500)
(MIT). Its pre-2019 history comes from Andreas Clenow's Norgate-derived list,
not from Wikipedia. It was used only to validate; none of its data is checked
in.

## Files

- `anchor.csv`, `changes.csv`: parsed straight from the pinned revisions.
- `corrections.csv`: the only hand-edited input. Each row has a note.
  - `row_alias`: a change row uses a ticker the anchor no longer uses
    (Everest Re `RE` → `EG`).
  - `price_symbol`: a membership spell ending on `date` takes its prices from
    `target`, because the company was renamed (Fleetcor `FLT` → `CPAY`).
    A blank target means its free price history is unusable.
  - `price_split`: the price symbol only becomes the member's own history from
    `date`. The earlier part of the spell stays a member but has no usable
    price.
- `membership.csv`: derived output, one row per membership spell.
  `start` is inclusive (blank = before coverage). `end` is exclusive
  (blank = still a member). `price_symbol` is blank when no usable free
  history exists, and `price_check` says why.
- `meta.json`: sources, coverage start, and the SHA-256 of `membership.csv`
  (reports record it).
- `VALIDATION.txt`: regenerated on every build.
- `../sp500_pit_prices.parquet`: closes for spell price symbols that the other
  fixtures don't carry.

Rebuild: `.venv/bin/python scripts/build_sp500_membership.py [--fetch-prices]`

## Conventions

- A change effective on day *d* counts from the close of *d*. On a signal
  date *t*, the eligible names are those whose spell contains *t*. This is the
  effective date, not the earlier announcement date, so knowing an addition a
  few days early is never assumed.
- Membership only decides who can be **picked** at a rebalance. A name's
  earlier price history still feeds its signals, because that history was
  public at the time. A held name that leaves the index mid-period is held
  until the next rebalance, like a real portfolio that rebalances on schedule.

## What this does not fix (disclosed on every report)

Some members have no free price history: mostly acquired companies, plus a
few that went bankrupt (Twitter, Activision, Xilinx, SVB Financial, First
Republic, Chesapeake and others). Those members can't be picked, and the
report says how many were missing at each rebalance. From
`VALIDATION.txt`, the share of members with usable prices is 80% at end-2015,
92.5% at end-2020 and 99.4% at end-2025. Missing losers flatter a universe, so
early-window results still lean optimistic. Closing that gap needs licensed
delisted-stock data (Norgate, CRSP).

Known labeling quirks that don't change who was a member:
- ACE Ltd/Chubb (2016) and Tyco/Johnson Controls (2016): the table labels the
  merger removal with the acquirer's old ticker. Both companies count as members
  until the merger either way.
- DuPont `DD` before 2017-09-01: Yahoo's series is continuous across the
  DowDuPont merger. It may carry Dow Chemical's history rather than E.I.
  du Pont's, and both were members.
