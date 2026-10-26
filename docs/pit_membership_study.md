# Point-in-time S&P 500: does today's list explain the huge momentum numbers?

Generated 2026-09-17 by `scripts/pit_membership_study.py`. Every figure below is computed by that script; rerun it to regenerate this file.

**Method.** Same engine, same prices, same rules; only the universe changes. *Today* is the S&P 500 list the app shipped with. *Point-in-time* is the index as it stood at each rebalance ([membership](https://en.wikipedia.org/w/index.php?title=Historical_components_of_the_S%26P_500&oldid=1370891103), coverage from 2015-01-01). Returns are net of 5 bps per side, the number the verdict uses.

## 1. Saved S&P 500 runs, replayed from their frozen price snapshots

*Today* and *Point-in-time* are both replays with the current engine on the run's frozen prices, so they differ only in who was eligible. *Saved* is what the stored report said. All six were produced before commit 35b370b (2026-09-13) landed, which added a one-day execution lag, charged costs on traded notional, credited T-bill yield to cash and screened stale feeds. The August runs replay lower than they were saved; the 2026-09-12 run, made while that work was in progress, replays to within 0.3 percentage points of a +18,693% total.

| Run | Saved name | Rule | Window | Saved | Today (replay) | Point-in-time | SPY | Members priced |
|---|---|---|---|---:|---:|---:|---:|---:|
| `r_723dddda0a` | War list 12mo momentum top-5 (S&P 500 re-run) | top 5 by 252d return, monthly | 2023-02-01 → 2026-05-28 | +3,553% | +3,525% | +286% | +94% | 98% |
| `r_299575a09e` | CAS check | top 5 by 126d return, monthly | 2024-07-01 → 2026-05-28 | +950% | +915% | +359% | +42% | 98% |
| `r_610285e740` | CAS check 2 | top 5 by 126d return, monthly | 2024-07-01 → 2026-05-28 | +950% | +915% | +359% | +42% | 98% |
| `r_910ac2227b` | Tech momentum fast-growers (SP500 tech) | top 5 by 63d return, monthly | 2016-09-01 → 2026-05-28 | +11,920% | +10,835% | +889% | +305% | 93% |
| `r_354a4c2fd0` | Least volatile SPY stocks (quarterly) | top 20 by 20d volatility, quarterly | 2016-10-03 → 2026-05-28 | +4,127% | +3,405% | +357% | +305% | 93% |
| `r_7ff6b5113e` | 6mo momentum top-5 | top 5 by 126d return, monthly | 2016-02-01 → 2026-01-02 | +18,693% | +18,693% | +211% | +316% | 92% |

Names the *today* runs bought on dates they were not S&P 500 members (first such pick → date they joined, or blank if they never were one in the window):

- `r_723dddda0a` — 14 names: TPL (2023-01-31→2024-11-26), SMCI (2023-02-28→2024-03-18), FIX (2023-03-31→2025-12-22), AXON (2023-03-31→2023-05-04), VRT (2023-06-30→2026-03-23), BLDR (2023-08-31→2023-12-18), APP (2023-10-31→2025-09-22), CVNA (2023-11-30→2025-12-22), COIN (2023-12-29→2025-05-19), COHR (2024-09-30→2026-03-23), HOOD (2024-11-29→2025-09-22), ECHO (2025-08-29→2026-03-23) …
- `r_299575a09e` — 9 names: CVNA (2024-06-28→2025-12-22), APP (2024-06-28→2025-09-22), HOOD (2024-07-31→2025-09-22), TPL (2024-08-30→2024-11-26), ECHO (2024-09-30→2026-03-23), FIX (2025-08-29→2025-12-22), LITE (2025-08-29→2026-03-23), SNDK (2025-10-31→2025-11-28), CIEN (2025-11-28→2026-02-09)
- `r_610285e740` — 9 names: CVNA (2024-06-28→2025-12-22), APP (2024-06-28→2025-09-22), HOOD (2024-07-31→2025-09-22), TPL (2024-08-30→2024-11-26), ECHO (2024-09-30→2026-03-23), FIX (2025-08-29→2025-12-22), LITE (2025-08-29→2026-03-23), SNDK (2025-10-31→2025-11-28), CIEN (2025-11-28→2026-02-09)
- `r_910ac2227b` — 54 names: AMD (2016-08-31→2017-03-20), LITE (2016-08-31→2026-03-23), TPL (2016-10-31→2024-11-26), STLD (2016-11-30→2022-12-22), COHR (2016-11-30→2026-03-23), TTD (2017-02-28→2025-07-18), DXCM (2017-03-31→2020-05-12), ANET (2017-03-31→2018-08-28), ALGN (2017-04-28→2017-06-19), BLDR (2017-04-28→2023-12-18), VEEV (2017-05-31→2026-05-07), FSLR (2017-06-30→2022-12-19) …
- `r_354a4c2fd0` — 53 names: TPL (2016-09-30→2024-11-26), NXPI (2016-09-30→2021-03-22), AMD (2016-09-30→2017-03-20), FIX (2016-09-30→2025-12-22), PODD (2016-09-30→2023-03-15), LULU (2016-09-30→2023-10-18), TRGP (2016-09-30→2022-10-12), CIEN (2016-12-30→2026-02-09), LITE (2016-12-30→2026-03-23), TTD (2016-12-30→2025-07-18), LVS (2016-12-30→2019-10-03), ARES (2016-12-30→2025-12-11) …
- `r_7ff6b5113e` — 46 names: AMD (2016-03-31→2017-03-20), LITE (2016-04-29→2026-03-23), TRGP (2016-06-30→2022-10-12), TPL (2016-10-31→2024-11-26), COHR (2017-01-31→2026-03-23), ANET (2017-04-28→2018-08-28), BLDR (2017-04-28→2023-12-18), XYZ (2017-04-28→2025-07-23), TTD (2017-05-31→2025-07-18), TSLA (2017-05-31→2020-12-21), TTWO (2017-05-31→2018-03-19), CVNA (2017-11-30→2025-12-22) …

## 2. Momentum lookbacks, top 5, monthly, 2023-01-02 → 2026-05-29

| Lookback | Today | Point-in-time | SPY | Today's non-member picks |
|---:|---:|---:|---:|---|
| 21d | +534% | +94% | +94% | 27: CVNA, COIN, SMCI, FIX, PANW, TKO, PLTR, APP … |
| 63d | +1,859% | +395% | +94% | 21: COIN, CVNA, SMCI, APP, PLTR, VRT, TPL, ERIE … |
| 126d | +2,724% | +418% | +94% | 20: AXON, SMCI, PLTR, BLDR, CVNA, APP, VRT, COIN … |
| 189d | +3,158% | +527% | +94% | 18: FICO, AXON, SMCI, BLDR, PLTR, VRT, CVNA, APP … |
| 252d | +3,561% | +290% | +94% | 14: TPL, SMCI, FIX, AXON, VRT, BLDR, APP, CVNA … |

## What this does not remove

Members with no free price history (mostly acquired companies, a few failures) can't be picked in either column; the *members priced* share says how much of the index each point-in-time run could see. See `fixtures/sp500_membership/README.md`.
