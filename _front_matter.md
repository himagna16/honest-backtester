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
