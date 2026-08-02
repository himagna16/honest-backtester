# The Honest Backtester — Product Brief

**Version:** 1.0 · **Date:** 2026-08-02 · **Owner:** Sai Maruvada · **Status:** V0 approved for build

---

## Origin (why this product exists)

This product was born from a real session. The founder backtested his first momentum strategy and got **+1650%**. Fixing the survivorship-biased universe dropped it to **+163%** — which *lost* to SPY buy-and-hold (+362%). A dip-buying strategy beat SPY gross (Sharpe 1.05 vs 0.83) but died at even 5 bps/day of transaction costs. Every existing retail backtesting tool would have shown the first number and none of the corrections.

**Thesis:** every competitor is incentivized to flatter the user, because flattering numbers sell subscriptions. This product wins by being the one that doesn't. It is *the backtester that tries to kill your strategy before the market does.*

Reference implementations from that session live in `/Users/saimaruvada/Downloads/momentum_backtest/` (`backtest.py`, `multifactor.py`, `reversal.py`, `allocations.py`, plus cached price CSVs). They are the seed of the engine and the source of golden-test fixtures.

## Non-negotiable product rules

1. **The audit is not a feature — it is the product.** There is no way to run a backtest without the full honesty audit. No toggle, ever.
2. **SPY (or an audited-appropriate benchmark) appears on every result.** Never omittable.
3. **No arbitrary user code execution.** The LLM translates English into a constrained JSON DSL; only the DSL runs.
4. **Educational/research tool only.** No personalized investment advice, no trade execution, no autotrading, no "signals" — permanently. Disclaimers on every report and every export.
5. **The LLM never silently guesses on material parameters.** Ambiguity → targeted clarifying questions. The parsed spec is confirmed by the user before anything runs.
6. **Honest failure reporting.** Data problems, shortened windows, and excluded tickers are surfaced as findings, never silently absorbed.

## Scope at a glance

| | V0 (build now) | V1 (only if validated) |
|---|---|---|
| Users | Sai only, local-first, one command to run | Hosted multi-user |
| Auth/payments | None | Auth + Stripe |
| Data | yfinance + pinned local snapshots | Licensed survivorship-free point-in-time data |
| Assets | US equities + ETFs, daily bars | Expanded DSL blocks |
| Never | — | Trade execution, advice, signals, code upload |

## How to read this brief

- **Section 1 — Product Spec & UX:** flows, the Truth Report, journal, edge states.
- **Section 2 — Strategy DSL & LLM Layer:** the JSON spec, translation, validation, guardrails.
- **Section 3 — Backtest Engine & Honesty Audit:** the core differentiator, check-by-check.
- **Section 4 — Technical Architecture & Build Plan:** stack, repo, API contract, milestones M0–M6.
- **Section 5 — Path to Market, Risks & Compliance:** validation plan, V1 deltas, legal.
- **Section 6 — Gap Audit (first-class requirements):** 12 gaps found by adversarial review of sections 1–5. **These are requirements, not suggestions** — the build prompt maps each to a milestone. Where a gap conflicts with an earlier section, the gap audit wins.

---



# Section 1 — Product Spec & UX

## Product Spec & UX

### One-liner

**The Honest Backtester** — describe a trading strategy in plain English, get a backtest that actively tries to kill it: bias audits, cost sweeps, and a benchmark comparison on every single result. The backtester that tries to kill your strategy before the market does.

### Goals & Non-Goals

**V0 goals (personal tool, local-first):**
- Sai can type a strategy idea ("buy the 5 strongest S&P stocks by 12-month momentum, rebalance monthly") and get a Truth Report in under 2 minutes on cached/free yfinance data.
- Every result ships with the full honesty audit — no way to run a backtest without it. The audit is not a feature toggle; it is the product.
- Encode today's hard-won lessons as automated checks: hand-picked universe detection (the +1650% trap), cost-sweep at 0/5/20 bps (the buy-the-dip killer), always-on SPY benchmark with Sharpe (the 163% vs 362% reality check).
- Constrained JSON strategy DSL — LLM translates English to spec, engine executes spec. No arbitrary code execution.

**V0 non-goals:** auth, payments, multi-user, hosting, licensed data, intraday data, options/futures/crypto, live signals, broker connections.

**V1 goals (if V0 validates):** hosted multi-user app, auth + Stripe, licensed survivorship-free point-in-time universe data, strategy sharing, expanded DSL blocks.

**V1 non-goals (permanent):** trade execution of any kind, personalized advice, autotrading integrations, "signals" subscriptions. These are never on the roadmap.

### Personas

**V0 — Sai (the builder-user):** beginner dev, self-teaching quant methods, has personally been burned by a fake +1650% backtest. Needs a tool that argues with him. Success = he trusts a result enough to decide *not* to trade a strategy.

**V1 — the quant-curious retail trader:** uses Robinhood/Composer/TradingView, has strategy ideas from YouTube and Reddit, can't code, doesn't know what survivorship bias is. Currently served by tools that flatter them with unaudited equity curves. Success = "this tool saved me from losing money," told to a friend.

### Core User Stories

1. As a user, I describe a strategy in plain English and see it translated into an explicit, editable spec before anything runs — so I know the machine tested what I meant.
2. As a user, I get a one-glance verdict on whether the strategy beats SPY after realistic costs, before I see any flattering numbers.
3. As a user, every bias in my test is caught, named, and explained in plain English so I learn *why* the number is inflated.
4. As a user, I can save strategies, re-run them, and compare versions over time in a journal.
5. As a user, I am never told what to trade — only what the evidence shows.

### Primary Flow: Idea → Spec → Run → Truth Report

**1. Describe.** A single large text box: *"Describe your strategy. Be specific about what you buy, when, and when you sell."* Ghost-text example inside. Below it, 3–4 starter chips ("12-1 momentum on S&P 500," "Buy the dip," "Low-vol tilt") so the empty box is never intimidating.

**2. Confirm the spec.** The LLM returns a **parsed strategy card**, not a wall of JSON: universe, signal, ranking/filter, position count, weighting, rebalance frequency, holding rules — each as a labeled, human-readable row with an edit affordance (dropdowns/inputs bound to the DSL's legal values). A collapsed "View raw spec (JSON)" toggle for learning. Any assumption the LLM made is flagged inline in amber: *"You didn't specify rebalance frequency — I assumed monthly. Correct?"* **Nothing runs until the user clicks "Run the gauntlet."** This confirmation step is a hard gate: it prevents silent misinterpretation and doubles as education (users see their vague idea forced into precise terms).

**3. Run.** Loading state is themed as the audit working: a checklist that fills in sequence — "Fetching prices… Checking your universe for selection bias… Simulating with costs… Comparing against SPY… Stress-testing regimes." Honest progress, not a fake spinner; typical run 10–60s.

**4. Truth Report.** See below.

### The Truth Report (detailed spec)

**Verdict-first design.** The top of the page is a full-width verdict banner — the single sentence a friend would tell you, before any chart:

- Red: **"Loses to SPY after costs."**
- Amber: **"Beats SPY gross, but dies at 5 bps. Your edge is smaller than your costs."**
- Amber: **"Beats SPY — but your test window has no bear market."**
- Green (rare, deliberately hard to earn): **"Survived the gauntlet. Beats SPY net of 20 bps across regimes."**

Below the banner, one row of four stats, strategy vs SPY side by side: total return, CAGR, Sharpe, max drawdown. SPY is never omittable.

**Audit findings list.** A stacked list of finding cards, ordered by severity (blockers red, warnings amber, passes green-checked but collapsed). Each card: icon + name ("Selection bias detected"), what was found ("Your universe is 44 hand-picked tickers, all current large-cap winners"), why it inflates results (2–3 sentences, plain English: "You picked these because they already went up. In 2016 you couldn't have known. This is the same trap that produces fake +1650% backtests."), and what the tool did about it ("Re-run available on the full S&P 500 constituent list"). Every finding links to a short explainer page — the audit is a curriculum.

**Cost-sweep table.** Four columns — 0 bps / 5 bps / 20 bps / SPY buy-and-hold — with total return and Sharpe in each. Cells that flip from beating SPY to losing are highlighted, so the exact break-even cost level is visible at a glance. Caption: "5 bps is optimistic for retail. 20 bps is realistic for frequent trading."

**Charts.** (a) Equity curve, strategy vs SPY, log scale, with the net-of-costs curve as the default line and the gross curve rendered thinner/dashed — the flattering number is visually subordinate. Out-of-sample period shaded and labeled. (b) Drawdown chart, strategy vs SPY underwater plot. (c) **Regime coverage bar:** a horizontal timeline of the test window segmented into bull/bear/sideways/high-vol regimes, with untested regime types called out ("Your window contains no sustained bear market — results may not survive one").

**Out-of-sample panel.** Train vs holdout stats side by side; a large train/holdout performance gap triggers an overfitting warning ("Your strategy performs 3x better in-sample — classic overfit signature"). A run counter per strategy family feeds a multiple-testing warning: "This is your 9th variant of this idea. One of them will look good by luck."

### Strategy Library & Journal

Left-nav list of saved strategies. Each strategy has: name, current spec, **version history** (every spec edit = new version), and **run history** (timestamped Truth Reports with their verdict badges). The list view shows verdict badges, not returns — the library visually rewards honesty, not big numbers. A "compare runs" view diffs two versions' specs and stats. V0 storage: local SQLite/JSON files.

### Empty, Error, and Edge States

- **Empty library:** "No strategies yet. Most ideas don't survive the audit — that's the point. Describe your first one." + starter chips.
- **Ambiguous text:** LLM never guesses silently on material parameters; it returns targeted questions as inline form fields ("'Strong stocks' — ranked by what? [momentum ▾ / volatility / volume]").
- **Unsupported idea:** honest scope refusal listing which DSL blocks exist: "Options strategies aren't supported in v1. Here's what the engine can test: …" Log the request for roadmap.
- **Data failure:** "yfinance rate-limited — retry in ~60s. Cached data available through [date]" with a "run on cache" option, clearly labeled as stale.
- **Insufficient history:** "Only 3 years of data for 6 of your tickers. Backtest shortened to [range]" — shown *before* the run, on the spec card.

### Tone & Voice

- **Honest over kind, but never smug.** "Your strategy loses to SPY after costs" — stated flat, then explained. The user is never mocked; the *bias* is the villain.
- **Educational by default.** Every warning teaches the concept behind it in one short paragraph. Jargon always gets a plain-English gloss on first use.
- **Never advice.** No "you should," no "consider buying," no forward-looking recommendations. Only past-tense, evidence-scoped statements: "over this period, this rule underperformed." Persistent footer disclaimer on every report: *"Educational research tool. Past performance does not predict future results. Not investment advice."* Any user request phrased as "what should I buy" gets a fixed response redirecting to backtesting the idea instead.
- **No celebration of returns.** Green verdicts say "survived," not "winner." The product's emotional register is a good editor, not a cheerleader.



# Section 2 — Strategy DSL & LLM Layer

## Strategy DSL & LLM Layer

### Design philosophy

The DSL is a **closed vocabulary of building blocks**, not a programming language. Every strategy the engine can run is expressible as a single JSON document validated against a fixed schema. The LLM's only job is translation from English to that document — it never generates code, never touches the backtest engine, and never sees price data. This is what makes v0 safe to build alone: the worst a bad translation can do is produce a spec the validator rejects or a backtest that answers the wrong question (which the confirmation step catches).

### The strategy spec (JSON Schema sketch)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "additionalProperties": false,
  "required": ["name", "universe", "selection", "weighting", "rebalance", "costs", "benchmark"],
  "properties": {
    "name":        { "type": "string", "maxLength": 80 },
    "universe": {
      "type": "object", "additionalProperties": false,
      "properties": {
        "preset":  { "enum": ["sp500", "nasdaq100", "sector_etfs", "dow30", "user_watchlist"] },
        "watchlist_file": { "type": "string" }
      },
      "required": ["preset"]
    },
    "filters": {
      "type": "array", "maxItems": 5,
      "items": {
        "type": "object", "additionalProperties": false,
        "required": ["signal", "op", "value"],
        "properties": {
          "signal": { "$ref": "#/$defs/signal" },
          "op":     { "enum": ["gt", "lt", "gte", "lte", "between"] },
          "value":  { "type": ["number", "array"] }
        }
      }
    },
    "selection": {
      "type": "object", "additionalProperties": false,
      "properties": {
        "rank_by":   { "$ref": "#/$defs/signal" },
        "direction": { "enum": ["top", "bottom"] },
        "n":         { "type": "integer", "minimum": 1, "maximum": 50 },
        "static_allocation": {
          "type": "array",
          "items": { "type": "object", "properties": {
            "ticker": {"type": "string"}, "weight": {"type": "number"} } }
        }
      }
    },
    "weighting":  { "enum": ["equal", "rank", "inverse_vol"] },
    "rebalance":  { "enum": ["daily", "weekly", "monthly", "quarterly", "annual"] },
    "entry_exit": {
      "type": "object", "additionalProperties": false,
      "properties": {
        "hold_days":     { "type": "integer", "minimum": 1, "maximum": 252 },
        "stop_loss_pct": { "type": "number" },
        "take_profit_pct": { "type": "number" }
      }
    },
    "costs": {
      "type": "object", "additionalProperties": false,
      "properties": {
        "sweep_bps": { "type": "array", "default": [0, 5, 20] }
      }
    },
    "benchmark":  { "enum": ["SPY", "QQQ", "sixty_forty"] },
    "window":     { "type": "object", "properties": {
        "start": {"type": "string", "format": "date"},
        "end":   {"type": "string", "format": "date"} } }
  },
  "$defs": {
    "signal": {
      "type": "object", "additionalProperties": false,
      "required": ["type"],
      "properties": {
        "type": { "enum": [
          "return",            
          "sma_cross",         
          "drawdown_from_high",
          "volatility",        
          "rsi",               
          "daily_return",      
          "fundamental"        
        ]},
        "lookback_days": { "type": "integer", "minimum": 1, "maximum": 756 },
        "skip_days":     { "type": "integer", "minimum": 0, "maximum": 63 },
        "fast": { "type": "integer" }, "slow": { "type": "integer" },
        "field": { "enum": ["pe", "pb", "div_yield"] }
      }
    }
  }
}
```

Notes: `skip_days` exists specifically so "12-1 momentum" (12-month return skipping the last month) is first-class — the exact construct from today's honest backtest. `fundamental` is a **V1 placeholder**: the validator rejects it in v0 with a message ("fundamental data requires licensed sources — coming in V1"), so the vocabulary is stable but the capability is gated. `costs.sweep_bps` is not optional and not user-tunable below the default — every result ships with the 0/5/20bps sweep; that's a product rule, not a preference.

### Three worked examples

**1. "Buy the 5 best 6-month performers in the S&P 500, rebalanced monthly"**

```json
{
  "name": "6mo momentum top-5",
  "universe": { "preset": "sp500" },
  "selection": { "rank_by": { "type": "return", "lookback_days": 126, "skip_days": 0 },
                 "direction": "top", "n": 5 },
  "weighting": "equal",
  "rebalance": "monthly",
  "costs": { "sweep_bps": [0, 5, 20] },
  "benchmark": "SPY",
  "window": { "start": "2016-01-01", "end": "2026-01-01" }
}
```

**2. "Buy stocks that dropped 5%+ yesterday, hold 5 days"**

```json
{
  "name": "Dip buyer, 5-day hold",
  "universe": { "preset": "sp500" },
  "filters": [ { "signal": { "type": "daily_return", "lookback_days": 1 },
                 "op": "lte", "value": -0.05 } ],
  "selection": { "rank_by": { "type": "daily_return", "lookback_days": 1 },
                 "direction": "bottom", "n": 20 },
  "weighting": "equal",
  "rebalance": "daily",
  "entry_exit": { "hold_days": 5 },
  "costs": { "sweep_bps": [0, 5, 20] },
  "benchmark": "SPY"
}
```

**3. "60/40 stocks and bonds, rebalanced once a year"**

```json
{
  "name": "60/40 annual rebalance",
  "universe": { "preset": "user_watchlist", "watchlist_file": "sixty_forty.txt" },
  "selection": { "static_allocation": [
    { "ticker": "SPY", "weight": 0.60 },
    { "ticker": "AGG", "weight": 0.40 } ] },
  "weighting": "equal",
  "rebalance": "annual",
  "costs": { "sweep_bps": [0, 5, 20] },
  "benchmark": "sixty_forty"
}
```

(`static_allocation` bypasses ranking entirely — the validator enforces that a spec has *either* `rank_by` *or* `static_allocation`, never both, and that static weights sum to 1.0 ± 0.001.)

### LLM translation layer

**Flow:** text box → Claude API call with the schema → validation → **spec echo** → user confirms → engine runs. Two hard rules:

1. **The parsed spec is always shown back before running** — rendered as both the raw JSON and a plain-English readback ("Every month, rank S&P 500 stocks by 6-month return and hold the top 5, equally weighted"). The readback is generated in the same API call. The user clicks "Run" or edits — the LLM never triggers a backtest itself.
2. **Ambiguity → questions, never guesses.** If the input underdetermines a required field, the model returns `{"status": "needs_clarification", "questions": [...]}` instead of a spec. "Buy good momentum stocks" should yield "Over what lookback — 3, 6, or 12 months? How many holdings? Rebalance how often?" — at most 3 targeted questions per round.

**Validation layer (deterministic, not LLM):** the returned JSON is checked against the schema with `jsonschema` in Python, plus semantic rules the schema can't express (weights sum to 1, `fast < slow`, `window.start < window.end`, v0 capability gates like the `fundamental` rejection). Schema failures are fed back to the model once for self-repair; a second failure surfaces an error to the user. Use the API's structured outputs (`output_config.format` with this JSON Schema, or `client.messages.parse()` with a Pydantic model mirroring it) so malformed JSON is impossible at the transport level — validation then only catches semantic issues. Note the schema above uses `between`/array values and `additionalProperties: false` throughout, which is structured-outputs-compatible; keep numeric range checks (`minimum`/`maximum`) in the Python validator since the API's schema enforcement doesn't support them.

**Guardrails:** no code execution path exists (there is no field whose value is ever `eval`'d); asset classes outside the preset universes (options, crypto, futures, shorting, leverage) are out-of-vocabulary — the system prompt instructs the model to decline them explicitly with a "not supported in v0" message rather than approximating.

### Draft system prompt for the translator

```
You translate plain-English investment strategy descriptions into a JSON
strategy spec. You are a translator, not an advisor.

Rules:
- Output ONLY a spec matching the provided schema, or a needs_clarification
  object with at most 3 questions. Never both.
- Never guess a required parameter the user did not state or clearly imply.
  Lookback windows, holding counts, and rebalance frequency are the usual
  gaps — ask.
- Convention: "6-month" = 126 trading days, "12-1 momentum" = lookback 252,
  skip 21. State conversions in the readback.
- If the strategy needs anything outside the schema (options, shorting,
  leverage, intraday, fundamentals), return needs_clarification explaining
  it is unsupported in this version. Do not approximate it silently.
- Never comment on whether the strategy is good, likely to work, or
  suitable for the user. Never suggest trades.
- Alongside the spec, produce a one-paragraph plain-English readback of
  exactly what will be tested, including every defaulted value.
```

Prompt-engineering notes: put the full JSON Schema and 4–6 few-shot examples (including one `needs_clarification` example and one "unsupported asset class" refusal) in the system prompt, in that order, and keep them byte-stable; the volatile user text goes only in the user turn. Few-shots matter more than instructions for the trading-day conversions and for teaching the model *when* to ask vs. translate.

### Cost control

- **Model:** default to `claude-opus-5` ($5/$25 per MTok) — translation quality is the product's front door, and a single translation is ~4K cached input + ~500 output tokens, well under a cent per call at cache-read rates. If Sai wants to cut costs after measuring accuracy, `claude-haiku-4-5` ($1/$5) is the step-down to A/B against a fixed test set of ~30 descriptions — but that's his call, made on data, not a premature optimization.
- **Prompt caching:** mark the last block of the system prompt (schema + few-shots) with `cache_control: {"type": "ephemeral"}`. Everything stable lives before the breakpoint; the user's strategy text comes after. Repeat translations then bill the ~4K-token prefix at ~0.1× — effectively the whole cost becomes the 500 output tokens.
- **Token caps:** `max_tokens: 2000` per translation (a full spec + readback fits in ~800); clarification rounds are cheaper still. Cap clarification loops at 2 rounds before telling the user to rephrase.
- **No retry storms:** one self-repair attempt on validation failure, then surface the error. Log every (input, spec, validation result) triple locally — that corpus becomes the eval set for any future model-tier or prompt change.



# Section 3 — Backtest Engine & Honesty Audit

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



# Section 4 — Technical Architecture & Build Plan

## Technical Architecture & Build Plan

### Recommended Stack (V0)

| Layer | Choice | Why |
|---|---|---|
| Frontend | Next.js 14 (App Router, TypeScript) | Sai has shipped Next.js on Vercel twice (Dual Admit, party RSVP). Zero new learning for the UI layer. |
| Backend | Python FastAPI | The backtest engine already exists as Python/pandas in `momentum_backtest/`. FastAPI adds a thin HTTP layer over code Sai wrote today. Auto-generated OpenAPI docs at `/docs` double as a debugging tool. |
| Persistence | SQLite (single file, `data/honest.db`) | No server to run, trivially backed up, adequate for one user. Accessed via SQLAlchemy so the V1 Postgres swap is a connection-string change. |
| Price data | yfinance + local CSV cache (already built) | Free, and the caching pattern from today's session carries over directly. |
| LLM | Claude API via `anthropic` Python SDK, called **only from the backend** | Key never touches the browser. |

**Why not all-Streamlit:** faster week one, but the Truth Report UI (audit cards, cost-sweep tables, regime timelines) will outgrow Streamlit's layout model, and none of it transfers to a market-ready V1 — you'd rewrite the frontend anyway. **Why not all-Next.js:** the engine is pandas; porting to JS throws away today's working code and the entire Python quant ecosystem, which is also the ecosystem Sai is learning for interviews. **V1 migration path:** SQLite → Postgres (SQLAlchemy makes this mechanical), FastAPI → hosted on Fly.io/Railway, Next.js → Vercel, add auth (Clerk/NextAuth) and Stripe. Nothing in V0 blocks this.

### Local-First V0

One command: `make dev` (or `./dev.sh`) starts FastAPI on `:8000` and Next.js on `:3000` concurrently. No auth, no payments, no Docker required (optional Dockerfile later). Everything — DB, price cache, run artifacts — lives inside the repo's gitignored `data/` directory.

### Repo Structure (monorepo)

```
honest-backtester/
├── Makefile                  # dev, test, seed, clean
├── .env.example              # ANTHROPIC_API_KEY=  (documented, empty)
├── frontend/                 # Next.js app
│   ├── app/                  # pages: /, /strategy/[id], /journal
│   ├── components/           # StrategyInput, SpecReview, TruthReport, AuditCard, EquityChart
│   └── lib/api.ts            # typed fetch client for the backend
├── backend/
│   ├── main.py               # FastAPI app + routes
│   ├── dsl/
│   │   ├── schema.py         # Pydantic models = the strategy spec DSL
│   │   └── validator.py      # semantic checks beyond schema
│   ├── engine/
│   │   ├── data.py           # yfinance fetch + CSV cache (port from today)
│   │   ├── backtest.py       # vectorized engine
│   │   └── metrics.py        # CAGR, Sharpe, max drawdown, turnover
│   ├── audit/
│   │   ├── universe.py       # survivorship/selection-bias detection
│   │   ├── costs.py          # 0/5/20 bps sweep
│   │   ├── regimes.py        # bear/bull/flat coverage check
│   │   ├── lookahead.py      # signal-timing guards
│   │   └── oos.py            # out-of-sample split + overfit warnings
│   ├── llm/
│   │   ├── translate.py      # Claude call: English → spec JSON
│   │   └── prompts/          # versioned prompt files
│   ├── store/                # SQLAlchemy models: Strategy, Run, Report
│   └── tests/
│       ├── golden/           # fixture prices + expected metrics JSON
│       └── fixtures/rigged_universe/   # hand-picked-winners universe
└── data/                     # gitignored: honest.db, price_cache/
```

### API Contract

All JSON over HTTP; errors return `{"error": {"code", "message"}}`.

**`POST /api/translate-strategy`** — `{ "text": "buy the 5 biggest gainers each month..." }` → `{ "spec": {...}, "assumptions": ["interpreted 'biggest gainers' as 21-day return"], "unresolved": [] }`. Never runs anything; translation only.

**`POST /api/validate-spec`** — `{ "spec": {...} }` → `{ "valid": true, "warnings": ["universe is hand-picked: survivorship audit will flag this"], "errors": [] }`. Pure function, no side effects.

**`POST /api/run-backtest`** — `{ "spec": {...}, "strategy_id": "optional" }` → `202` with `{ "run_id": "r_abc123" }`, then progress via **`GET /api/run-backtest/{run_id}/events`** (Server-Sent Events): `{"stage": "fetching_data", "pct": 20}` ... `{"stage": "done", "report_id": "rep_1"}`.

**`GET /api/report/{report_id}`** → `{ "metrics": { "total_return", "cagr", "sharpe", "max_drawdown", "turnover" }, "benchmark": { same shape, for SPY }, "cost_sweep": [{ "bps": 0|5|20, "total_return", "sharpe" }], "audit": [{ "check": "survivorship", "severity": "fail|warn|pass", "message", "evidence" }], "oos": { "in_sample": {...}, "out_of_sample": {...} }, "equity_curve": [[date, strat, spy], ...] }`.

**`GET /api/strategies`** → `[{ "id", "name", "created_at", "last_run": { "report_id", "sharpe", "audit_fail_count" } }]`. Plus `GET /api/strategies/{id}` for the journal detail view.

Spec JSON travels frontend↔backend but is only ever *executed* by the engine after `validate-spec` passes — the DSL (fixed enums for universe, signal, ranking, schedule, position sizing) is the security boundary; no arbitrary code in V1.

### Background Jobs (V0: keep it simple)

No Celery, no Redis. `run-backtest` executes in a FastAPI `BackgroundTask`; progress events are pushed to an in-process queue and streamed over SSE. Typical runs (10y daily, ≤500 tickers, cached CSVs) finish in seconds. If the tab closes, the run completes and lands in the journal. V1 swap: same endpoints, worker moves to a real queue.

### Build Order: M0–M6

Each milestone is a self-contained agentic coding run with its own acceptance test.

- **M0 — Scaffold.** Monorepo, Makefile, FastAPI hello + Next.js page calling it, SQLite migration, `.env.example`, CI running `pytest` + `next build`. *Accept:* `make dev` serves both; `make test` green.
- **M1 — Engine + golden tests.** Port data cache and vectorized backtest from `momentum_backtest/`; metrics module. Golden tests: fixed 2016–2026 fixture prices for SPY-momentum and buy-the-dip with expected metrics pinned to today's hand-verified numbers (±0.1%). *Accept:* golden tests pass; engine callable as a pure function `run(spec, prices) -> results`.
- **M2 — DSL + validation.** Pydantic spec schema, semantic validator, `validate-spec` endpoint. *Accept:* 10 valid + 10 invalid spec fixtures classified correctly; invalid specs can never reach the engine.
- **M3 — LLM translate.** `translate-strategy` endpoint; prompt returns spec JSON + stated assumptions; retry-on-invalid-JSON loop (max 2); refusal path for untranslatable input. *Accept:* the three worked examples (rigged momentum, honest S&P momentum, buy-the-dip) each translate from plain English to a spec that passes validation.
- **M4 — Truth Report UI.** All five audit modules wired into `run-backtest`; report page with equity chart, metrics vs SPY, cost-sweep table, audit cards with severity colors, SSE progress bar. *Accept:* rigged-universe fixture produces a red survivorship FAIL; honest S&P spec produces pass; disclaimers visible on every report.
- **M5 — Journal + cache.** Strategy list/detail pages, re-run button, spec version history, price-cache reuse across runs. *Accept:* second run of the same universe makes zero network fetches; journal survives restart.
- **M6 — Polish + docs.** README with 5-minute quickstart, `.env` setup docs, error states (API key missing, ticker not found, yfinance down), empty states, loading states. *Accept:* fresh clone → running first backtest in under 5 minutes following only the README.

### Definition of Done (V0 checklist)

- [ ] `make dev` starts everything with one command on a clean Mac clone
- [ ] All golden tests pass (`make test`)
- [ ] The three worked-example strategies run end-to-end: English → spec → backtest → Truth Report
- [ ] Rigged-universe fixture triggers a survivorship-bias FAIL; honest S&P universe does not
- [ ] Every report shows: SPY benchmark + Sharpe, 0/5/20bps cost sweep, regime coverage, OOS split, audit cards
- [ ] Cost sweep on buy-the-dip reproduces today's finding (gross win → net death)
- [ ] No spec can execute without passing validation; no code-execution path exists
- [ ] Educational-use disclaimer on every report page; no advice language anywhere
- [ ] `git grep -i "sk-ant"` returns nothing; repo is clean of secrets

### Env & Config

Single secret in V0: `ANTHROPIC_API_KEY`, loaded from `backend/.env` via `pydantic-settings`, read only by the backend (never `NEXT_PUBLIC_*`, never shipped to the browser). `.env` is gitignored; `.env.example` documents every variable with comments. Non-secret config (cache dir, default cost assumptions, SQLite path) lives in `backend/config.py` with env overrides. Backend fails fast at startup with a plain-English message if the key is missing — but only endpoints touching `/translate-strategy` require it, so the engine and audits work fully offline.



# Section 5 — Path to Market, Risks & Compliance

## Path to Market, Risks & Compliance

### Validation Plan (Do This Before Writing Any V1 Code)

The V0 already exists as working Python in `/Users/saimaruvada/Downloads/momentum_backtest/`. Before investing in auth, payments, or hosted infrastructure, run a 2-3 week validation sprint:

1. **Ship the story, not the product.** Post the momentum saga to X and r/algotrading / r/quant / r/investing (respecting each sub's self-promo rules): "My backtest said +1650%. Then I removed the cheating and it lost to SPY. So I built a backtester that tries to kill your strategy before the market does." The +1650% → +163% → cost-sweep-death arc is the marketing asset — it is concrete, verifiable, and embarrassing in a relatable way. Drafts exist; polish and post.
2. **Waitlist landing page.** One page (Vercel, same stack as party-rsvp): tagline, one screenshot of a real Honesty Audit report, email capture. No pricing yet. Ship in a weekend.
3. **Success threshold:** 10+ genuine "I'd use this" replies with specifics (a strategy they'd test, a bias that bit them) OR 50+ waitlist emails within 3 weeks. Upvotes and "cool idea" comments do not count.
4. **Kill criterion:** If posts get crickets — under 10 substantive replies and under 25 signups after posting in 3+ venues — stop V1 planning. V0 remains valuable as Sai's personal tool and a quant-interview portfolio piece, which is the original goal anyway. That is a soft landing, not a failure.
5. **Secondary signal:** DM 5-10 engaged repliers and ask "what did your last backtest lie to you about?" Their answers are the V1 audit-feature backlog.

### V1 Delta List (What Public Launch Actually Requires)

| Item | Why | Realistic cost/effort |
|---|---|---|
| Auth (Clerk or Supabase Auth) | Multi-user; Sai has not built auth before — budget learning time | ~$0-25/mo at small scale |
| Stripe subscriptions | He has used Payment Links; subscriptions + webhooks are a step up | 2.9% + 30¢, no fixed cost |
| Hosted deploy | Next.js front on Vercel; Python engine on a worker (Railway/Fly/DO droplet, pattern known from Kalshi box) | $20-50/mo |
| **Licensed survivorship-free data** | yfinance is against Yahoo ToS for a commercial product and is itself survivorship-biased — a fatal irony for an "honesty" product. Options: Norgate (~$30-60/mo, delisted-stock coverage, the standard hobbyist-pro choice), EODHD (~$20-80/mo), Polygon (~$29-199/mo) | **$30-200/mo — the single biggest V1 cost and the moat** |
| Rate limiting + queueing | Backtests are CPU-bound; one user must not starve others | Redis queue, minimal $ |
| LLM cost caps | Per-user daily cap on Claude API calls (e.g. 20 spec-generations/day free tier); prompt-cache the DSL system prompt | Engineering, not $ |
| Monitoring | Sentry + uptime ping; know when the engine silently returns garbage | ~$0-26/mo |
| Legal review | See below; before charging money, not after | $1-3k one-time |

### Business Risks, Honestly

**The perverse-incentive problem is the core risk.** Users emotionally want to be told their strategy is genius; the product's entire value is telling them it is not. Competitors (Composer, TradeZella, TrendSpider, QuantConnect) implicitly sell flattery via unaudited equity curves. Mitigations:

- **Framing:** the audit is a *gauntlet*, not a report card. "Your strategy survived 4 of 6 kill-tests" is a challenge, not an insult. Gamify survival, not returns.
- **Shareable Truth Report cards** as the viral loop: a clean image card ("Claimed: +748%. After 20bps costs: -97%. Verdict: KILLED") is inherently screenshot-able on FinTwit, where dunking on fake backtests is already a genre.
- **Graduation features:** when a strategy survives the audit, unlock forward-testing (paper-tracking it out-of-sample from today). Gives survivors a payoff and creates retention for the honest minority.
- Accept that churn will be high among users seeking validation. The retained cohort — people who actually want truth — is smaller but pays and evangelizes.

**Crowded market:** the wedge is not "backtesting" (commodity) but "audited backtesting" (unoccupied). If a big player copies the audit, the moat is data quality + audit depth + brand trust, all of which cost money to replicate and hurt incumbents' flattery-based engagement metrics.

**Data moat is a treadmill:** survivorship-free data is a permanent recurring cost that scales with asset-class coverage. This is why V1 stays US-equities-daily only.

### Legal & Compliance

- **Positioning: educational and research tool, period.** No personalized recommendations, no "you should buy X," no discretionary or autonomous trading — ever (this is already a standing product rule). The LLM layer must be prompt-constrained to never emit advice, only strategy specs and audit findings.
- **Why the constrained DSL matters for compliance too, not just security:** no arbitrary code execution means no user-supplied code exfiltrating data, attacking the host, or turning the platform into an execution venue. It also keeps the product legibly a *calculator*, not a trading system.
- **Draft disclaimer (every results page and Truth Report card):**
  > "This is a historical simulation for educational purposes only. It is not investment advice, and [product] is not a registered investment adviser or broker-dealer. Simulated performance does not reflect actual trading and has inherent limitations. Past performance does not guarantee future results. You can lose money trading securities."
- **ToS/Privacy basics for V1:** ToS covering acceptable use, no-advice disclaimer, data-provider passthrough restrictions (Norgate/Polygon licenses restrict redistribution of raw data — audits/derived stats are fine, raw price dumps are not); privacy policy covering email, strategy specs, and Stripe-held payment data. Use a standard SaaS template initially.
- **Non-negotiable:** paid legal review (~$1-3k, a fintech-adjacent startup attorney) before the first paid subscription. Investment-adviser regulation (SEC/state RIA rules) has real edges around "personalized" outputs; get a professional to bless the line the product walks. Do not skip this to save money.

### Security (V1)

- **No code upload, recommendation: keep it that way indefinitely.** The DSL is the security boundary. If code upload is ever reconsidered, it requires full sandboxing (gVisor/Firecracker, no network egress, CPU/memory/time limits) — months of hardening for marginal user value. Don't.
- **API keys:** Claude and data-provider keys live server-side in environment secrets only; never shipped to the client; rotate on any suspicion.
- **User data:** strategy specs are users' ideas — treat as confidential, encrypt at rest, never train on them, never show cross-user. Stripe holds all payment data; the app never touches card numbers.
- **LLM input handling:** user text-box input goes to the spec-generator prompt — treat output as untrusted, validate the JSON spec against a strict schema before the engine ever sees it (this also blocks prompt-injection from producing rogue specs).

### Cost Model

- **V0 (now):** ~$5-20/mo — Claude API calls for spec generation only (cacheable system prompt keeps this low); yfinance free; runs on Sai's Mac. Effectively pocket change.
- **V1 (launched):** ~$100-350/mo baseline — data license $30-200, hosting $20-50, auth/monitoring $0-50, LLM $20-50 (capped per user) — plus one-time legal $1-3k. Breakeven at roughly 10-25 subscribers at $15-29/mo, which is a sane first milestone.

### Explicitly Excluded from V0

- **Auth** — single user; it's Sai's laptop.
- **Payments** — nothing to sell until validation passes.
- **Live trading / brokerage integration** — permanently against product rules; also a regulatory cliff.
- **Intraday data** — cost and complexity explode; daily bars prove the concept.
- **Options / crypto / futures** — each is a new data license, cost model, and audit ruleset; US equities daily only.
- **User code upload** — the DSL *is* the product's security and compliance boundary.
- **Licensed data** — yfinance is acceptable for personal research; swap before anyone else touches it.



# Section 6 — Gap Audit: Additional Requirements (adversarial review)

- **Entry/exit semantics undefined in the engine** — Backtest Engine & Honesty Audit — The DSL exposes `hold_days`, `stop_loss_pct`, and `take_profit_pct`, but the engine spec only describes vectorized rebalance-weight math; stops on daily bars need explicit fill rules (checked at close, exited at next open? gap-through-stop handling?) and `hold_days` conflicts with `rebalance` frequency (dip-buyer: daily rebalance + 5-day hold = overlapping cohorts — how are weights combined?). Specify these semantics and add golden fixtures for them, or cut stops/TP from the V0 schema; as written, M1/M2 will discover the engine can't execute specs the validator accepts.

- **Per-run reproducibility: pinned data snapshots + determinism rules** — Backtest Engine (new subsection) — yfinance silently restates adjusted prices (dividends, splits, corrections), so re-running an identical spec next month yields different numbers, which poisons the journal's "compare runs" feature and the multiple-testing counter. Persist with every Run a content hash and a frozen parquet snapshot of the exact price panel used, and re-run against the snapshot by default with an explicit "refresh data" option. Also define deterministic tie-breaking in `rank_by` (equal momentum values → alphabetical) so top-N selection is stable across runs and platforms.

- **Where the S&P 500 constituent list comes from in V0** — Data Layer — The `sp500` preset and the survivorship audit's "re-run on full S&P 500" both require a ticker list the brief never sources. Specify: a checked-in seed list, refresh mechanism (Wikipedia scrape is fragile and has its own ToS questions), update cadence, and how the ~500-ticker initial fetch behaves under yfinance rate limits (first cold run could take many minutes — the "10–60s typical run" claim depends on this being pre-warmed). This is a hard M1 blocker.

- **Ticker validation and missing-data policy** — DSL & LLM Layer + Engine — Nothing validates that tickers exist: the LLM can hallucinate symbols, watchlist files can contain typos or renamed tickers (FB→META), and yfinance returns empty frames rather than errors. Add a symbol-master check at `validate-spec` time. Separately, define what happens when a universe ticker lacks data for part of the window — silent dropping is itself survivorship bias and must instead surface as an audit finding ("6 of 44 tickers excluded for missing data; results biased toward survivors").

- **Benchmark-appropriateness audit check** — Honesty Audit (check #9) — The DSL offers SPY/QQQ/sixty_forty benchmarks but nothing decides or audits the choice: a 60/40 or bond-heavy strategy compared to SPY produces a misleading verdict in both directions (looks bad in bulls, fake-good in bears). Add a check that compares the strategy's realized beta/asset mix to the chosen benchmark and emits a WARNING with a suggested re-run ("Your portfolio is 40% bonds; SPY is the wrong yardstick — compared against 60/40, your excess return drops to X"). Also define who picks the benchmark (LLM default? spec-card dropdown?).

- **Trading-calendar and date-convention spec** — Backtest Engine — "126 trading days," "last trading day signal," "monthly rebalance," and the 70/30 split all assume a calendar the brief never defines: NYSE holidays, half days, and month-end when the last day is a holiday. Specify use of an explicit trading calendar (e.g., `exchange_calendars` or derive from SPY's own index) and document the day-count conventions in one table shared by the engine, the LLM prompt, and the glossary — otherwise the LLM's "6 months = 126 days" and the engine's actual windows will silently disagree.

- **LLM translation quality harness (semantically-wrong-but-valid specs)** — LLM Layer — Validation catches malformed specs but not correct-schema/wrong-meaning ones ("6-month" translated as 26 days passes every check and quietly answers the wrong question). Build the ~30-description eval set now, in M3, with expected specs as fixtures asserted in CI — not later as an optional A/B tool. Also generate the plain-English readback deterministically from the validated JSON (template code), not in the same LLM call, so the readback can never describe a different strategy than the one that runs.

- **Spec schema versioning and migration** — Technical Architecture — Strategies and runs persist in SQLite, and V1 explicitly plans "expanded DSL blocks," but there is no `spec_version` field or migration policy. Add a version integer to every stored spec, a rule for loading old versions (auto-migrate vs. read-only), and a note that Run records must store the resolved spec verbatim (with all defaults expanded) so old Truth Reports remain interpretable after schema changes.

- **Forward-testing ("graduation") is a one-liner but is the product's natural retention loop** — Path to Market, promote to its own subsection — Specify at least the V1 shape: freezing the spec at graduation (immutable, hash-referenced), a daily scheduled evaluation job, a live-vs-backtest drift report ("forward Sharpe 0.4 vs backtest 1.1"), and the compliance framing that keeps it a tracking journal rather than a signals product (no notifications phrased as "buy now"). Deciding this now affects V0 data-snapshot and spec-immutability design.

- **Export and import** — Product Spec & UX (Library section) — V0 has no way to get anything out: no spec JSON export/import (needed for backup, bug reports, and sharing during the validation sprint), and no report export, yet the marketing plan depends on posting Honesty Audit screenshots. Add: download spec as JSON, import spec file, and export Truth Report as a static image/PDF with the disclaimer baked into the exported artifact (a shared image detached from the app must still carry the legal footer).

- **Telemetry and product metrics** — Path to Market — The validation plan defines external signals (replies, waitlist) but no in-product instrumentation for V1 decision-making: translation success/clarification/refusal rates, spec-confirm abandonment, verdict distribution, re-run rate after a FATAL. Define the event list now and log it locally in V0 (it doubles as the LLM eval corpus already mentioned); for V1 pick a privacy-respecting analytics stance, since strategy text is confidential user IP and must never go to third-party analytics.

- **Accessibility and small-screen behavior of the Truth Report** — Product Spec & UX — Verdict severity is communicated almost entirely by red/amber/green, which fails colorblind users; require icon + text-label redundancy on every badge and finding card, and text summaries/table fallbacks for the equity, drawdown, and regime charts. Also specify how the cost-sweep table and regime timeline degrade on narrow viewports — the FinTwit share loop means many first impressions of a Truth Report will happen on phones.
