# The landing page's S&P 500 figures, on point-in-time membership

Generated 2026-09-17 by `scripts/founding_story_pit.py`, which ports the founding session's `multifactor.py` and `reversal.py` line for line and restricts each date's cross-section to that day's S&P 500 members (ranks taken within members). The *today* column must reproduce the original scripts' printed output:

- ✓ multifactor net total +163.1%
- ✓ multifactor net CAGR 9.8%
- ✓ multifactor SPY total +362.0%
- ✓ reversal losers 0bps 748%/1.05
- ✓ reversal losers 5bps 103%/0.41
- ✓ reversal SPY 344%/0.83

## Momentum + low volatility, top 25, monthly, 10 bps (`multifactor.py`)

| | Today's list | Point-in-time |
|---|---:|---:|
| Window | 2016-01-31 → 2026-04-30 | 2016-01-31 → 2026-04-30 |
| Total return (net) | +163.1% | +159.1% |
| CAGR (net) | +9.8% | +9.6% |
| Sharpe (net) | 0.74 | 0.75 |
| Max drawdown (net) | -20.5% | -19.3% |
| SPY total / CAGR | +362.0% / 16.0% | +362.0% / 16.0% |

## Buy yesterday's losers for one day (`reversal.py`)

Point-in-time runs from 2015-01-05 (membership coverage starts 2015-01-01); today's list from 2015-01-05. Each cell is total return / Sharpe.

| Basket | Today 0 bps | Today 5 bps | Today 20 bps | PIT 0 bps | PIT 5 bps | PIT 20 bps |
|---|---:|---:|---:|---:|---:|---:|
| BuyLosers | +748% / 1.05 | +103% / 0.41 | -97% / -1.51 | +413% / 0.81 | +22% / 0.19 | -98% / -1.68 |
| BuyWinners | +527% / 0.92 | +50% / 0.28 | -98% / -1.66 | +247% / 0.67 | -17% / 0.00 | -99% / -2.00 |
| BuyAll | +485% / 0.94 | +40% / 0.25 | -98% / -1.82 | +259% / 0.70 | -14% / 0.02 | -99% / -2.03 |
| SPY buy-and-hold | +344% / 0.83 | | | +344% / 0.83 | | |

Caveats carried over from the original scripts: a name that stops trading mid-month drops out of that month's average; members with no free price history can't be picked (see `fixtures/sp500_membership/README.md`).
