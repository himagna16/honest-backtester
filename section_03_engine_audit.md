## Backtest Engine & Honesty Audit

This is the product. Competitors sell dopamine; we sell the audit. Every design decision below exists because Sai personally produced a fake +1650% result in one afternoon and only caught it by re-running honestly.

### Engine Architecture

**Core loop (pandas, vectorized).** One `prices` DataFrame (dates × tickers, dividend-adjusted closes via yfinance `auto_adjust=True`), one `signals` DataFrame, one `weights` DataFrame. No per-day Python loops except rebalance bookkeeping.

- **Frequencies:** daily and monthly rebalance. Monthly = last trading day signal, first trading day execution. Daily strategies (dip-buying) rebalance every bar.
- **Point-in-time discipline, by construction:** the engine computes `weights[t]` from data through `t` only, then earns `returns[t+1]`. Concretely: `portfolio_ret = (weights.shift(1) * asset_returns).sum(axis=1)`. The `.shift(1)` is applied inside the engine, not by strategy code — strategy specs literally cannot request same-day fills because the DSL has no "execute at signal price" primitive. This is how the 12-1 momentum backtest was kept honest (skip the most recent month; signal at month-end, trade next open).
- **Position & turnover accounting:** `turnover[t] = 0.5 * Σ|w[t] - w_drifted[t-1]|` where `w_drifted` is last period's weights drifted by returns. Turnover is a first-class output because it drives the cost sweep.
- **Cost application:** `net_ret[t] = gross_ret[t] - turnover[t] * cost_bps/10000`, applied per rebalance. Costs are a parameter of the *run*, not the strategy — every strategy is automatically run at all three cost levels.
- **Total returns:** dividend-adjusted throughout; raw close prices are never used for return math (a documented invariant, tested).

### Data Layer

**V0:** yfinance with a local parquet cache in `~/.honest_backtester/cache/` keyed by `(ticker, start, end)`. Staleness rule: cache older than 24h for daily data triggers a refresh; refresh failures fall back to cache with a visible "data as of YYYY-MM-DD" stamp on results. Never silently mix stale and fresh tickers in one run — refresh all or stamp all.

**V1 upgrade path:** licensed survivorship-free data (Norgate, Sharadar, or Polygon) with delisted tickers and point-in-time index constituents. **Why this matters:** a "current S&P 500 members" list only contains companies that *survived and grew* into the index. Backtesting 2016–2026 on today's members means you bought 2016-era NVDA because you secretly knew it would win. That's the same sin as Sai's hand-picked 44-ticker watchlist, just laundered through an index. Honest S&P 500 backtests need the membership list *as of each rebalance date*, including companies that later went to zero. yfinance cannot provide this; it's the single biggest reason V1 needs paid data.

### The Honesty Audit

Each check produces a finding: `severity` (FATAL / WARNING / NOTE), detection logic, and a plain-English message.

**1. Selection/survivorship bias.** *Detect:* universe is user-supplied tickers, or an index list without point-in-time membership. *Action:* auto re-run the identical spec on a neutral universe (V0: full current S&P 500 with a disclosed "still biased, but less" caveat; V1: true point-in-time universe). *Message:* "You picked these 44 tickers knowing how they performed. On a neutral universe your strategy returned +163%, not +1650%. The difference is hindsight, not skill." Severity: FATAL if user universe outperforms neutral by >2x.

**2. Lookahead guards (structural).** Not a check — an impossibility proof. The DSL exposes only `lag`-safe primitives (`momentum(12,1)`, `sma(200)`, `vol(60)`); the engine applies `shift(1)` centrally; monthly signals use completed months only. Audit line shown on every result: "Lookahead bias: impossible by construction — signals at t, returns from t+1." One unit test per primitive asserts zero dependence on future rows (perturb future data, confirm identical signal).

**3. Transaction-cost sweep.** Every run reports 0 / 5 / 20 bps side-by-side, cost scaled by measured turnover. *Detect:* strategy beats benchmark at 0bps but not at 5bps → WARNING: "Your edge is smaller than retail trading costs. Gross +748% became −97% at 20bps — turnover is eating you alive." Additional wide-spread warning when universe includes tickers under ~$2B market cap or ADV below a threshold: "Small-cap spreads often exceed 20bps; even the pessimistic column may be optimistic."

**4. Benchmark truth.** SPY total return over the identical window, always: final return, CAGR, annualized vol, Sharpe, max drawdown, plus rolling 1-year excess return so a strategy that won only in 2020 can't hide. Headline framing is always relative: "vs. doing nothing (SPY)."

**5. Regime coverage.** *Detect:* window vs a hardcoded regime table (2020 COVID crash, 2022 rate shock, 2018 Q4, etc.). If window contains no ≥15% SPY drawdown: WARNING "Your test window contains no bear market. Strategies that look great in bull markets often die in the first real drawdown."

**6. Out-of-sample + parameter sensitivity.** Default 70/30 date split; verdict privileges the validation segment. Sensitivity: automatically re-run with each numeric parameter perturbed ±1 step (top-5 → top-4 and top-6; 12-month lookback → 11 and 13). If validation Sharpe or any perturbation flips the strategy from winner to loser: WARNING "Your result depends on a specific parameter choice. Top-5 beat SPY; top-6 didn't. That's noise, not edge."

**7. Overfitting / multiple testing.** Persist a per-universe attempt counter (SQLite). After N variants on the same data: "This is your 14th strategy tested against this data. With enough tries, something beats SPY by luck. A Bonferroni-adjusted bar for significance is now ~N× higher." Never blocks — informs.

**8. Capacity/realism.** User supplies account size (default $10k). *Detect:* any position under one share without fractional-share support; weights implying sub-dollar trades; NOTE if daily rebalancing is required but user has a taxable account (short-term gains drag, V1). Message: "At $10k, a 5% position in BRK.A is untradeable."

### Metrics Glossary (shown in-product)

- **CAGR** = `(end/start)^(252/days) − 1` — smoothed annual growth.
- **Sharpe** = `mean(daily_ret − rf) / std(daily_ret) × √252` — return per unit of pain; rf from 13-week T-bill (^IRX), fallback 0 with disclosure.
- **Max drawdown** = `min(equity/cummax(equity) − 1)` — worst peak-to-trough loss.
- **Turnover** = `0.5 × Σ|Δw|` per rebalance, annualized — how much you trade; cost multiplier.
- **Hit rate** = share of periods beating benchmark — consistency, not magnitude.

### Verdict Logic

Findings roll up to one headline: any FATAL → **"Result not trustworthy"** with the top reason; no FATAL but net-of-5bps validation-segment return trails SPY → **"Honest result: loses to buy-and-hold"**; survives costs, out-of-sample, and sensitivity → **"Survived the audit"** — plus permanent framing: "Surviving a backtest is necessary, not sufficient. This is not investment advice." The verdict always uses the *pessimistic* cell: net-of-costs, validation-segment, neutral-universe numbers. Optimistic numbers are visible but never the headline.

### Testing the Engine

The auditor must itself be audited:

- **Golden fixtures:** tiny hand-computed cases checked into the repo — 3 tickers × 10 days with known momentum ranks, weights, turnover, and net returns computed by hand in a spreadsheet; asserted to the penny.
- **Property tests:** (a) zero-cost 100% SPY buy-and-hold must reproduce SPY total return exactly; (b) costs ≥ 0 ⇒ net ≤ gross, always; (c) shuffling *future* rows never changes today's weights (lookahead detector); (d) weights sum to ≤ 1.0 every day; (e) doubling costs at fixed turnover doubles the cost drag.
- **Regression fixtures from real history:** Sai's three existing backtests (`backtest.py`, `multifactor.py`, `reversal.py` results) become frozen fixtures — the engine must reproduce the honest +163% and the cost-collapse of dip-buying, and must flag the 44-ticker run as FATAL. If the new engine can't catch the bug its author already fell for, it ships nothing.