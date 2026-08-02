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