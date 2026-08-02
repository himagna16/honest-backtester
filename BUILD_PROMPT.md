# BUILD PROMPT — The Honest Backtester V0 (one-shot ultracode run)

Copy everything below the line into Claude Code with this folder as the working directory.

---

ultracode

Build **V0 of "The Honest Backtester"** — a local-first, single-user web app — by executing the spec in `PRODUCT_BRIEF.md` in this directory. Read the entire brief before writing any code. Section 6 (Gap Audit) contains first-class requirements; where it conflicts with sections 1–5, Section 6 wins.

## What you are building

Plain-English strategy in a text box → Claude API translates it into a constrained JSON strategy spec (the DSL in Section 2) → user confirms/edits the parsed spec card → Python engine backtests it on daily yfinance data (Section 3) → a **Truth Report** (Section 1) with a verdict-first banner, always-on SPY benchmark, honesty-audit findings, cost sweep (0/5/20 bps), equity/drawdown charts, regime coverage, and out-of-sample panel. Strategies and runs persist in a local journal.

## Non-negotiables (from the brief — verify each before calling anything done)

1. No backtest can run without the full honesty audit. No toggle exists.
2. Benchmark comparison appears on every result and every export.
3. No arbitrary code execution — only validated DSL specs run. Reject anything else.
4. Educational-tool disclaimer on every report and baked into every exported artifact.
5. The LLM never silently guesses material parameters — it must return targeted clarifying questions, and the parsed spec card requires explicit user confirmation before running.
6. The plain-English readback of a spec is generated deterministically from the validated JSON by template code — never by the same LLM call that produced the spec.
7. Data problems (missing tickers, shortened windows, exclusions) surface as audit findings, never silently.

## Stack (Section 4)

Next.js (App Router) frontend · Python 3.11+ FastAPI backend · SQLite persistence · pandas/numpy engine · yfinance with **pinned parquet snapshots per run** (Gap #2: content-hashed; re-runs replay the snapshot by default, explicit "refresh data" fetches new) · Anthropic SDK for translation (`ANTHROPIC_API_KEY` from `.env`, never committed; use the current Claude Sonnet-class model; cache translations). One command starts everything (a `Makefile` or `dev.sh` that launches backend + frontend).

## Existing assets — use them

- `/Users/saimaruvada/Downloads/momentum_backtest/backtest.py`, `multifactor.py`, `reversal.py` — reference engine logic (momentum ranking, cost application, `shift(-1)` lookahead guard, stats functions). Port the *logic*, upgrade the structure.
- `/Users/saimaruvada/Downloads/momentum_backtest/prices.csv` (44-ticker "War" list) and `sp500_prices.csv` — copy into `fixtures/` as frozen test data. The War-list panel is your **rigged-universe fixture**: the selection-bias audit MUST fire on it (it reproduces the famous fake +1650%).
- `/Users/saimaruvada/Downloads/momentum_backtest/tickers.txt` — the hand-picked watchlist for universe-upload testing.

## Build order — milestones with acceptance criteria

Execute M0→M6 sequentially. Commit at every milestone boundary with a descriptive message. Do not advance past a milestone with failing tests.

**M0 — Scaffold.** Repo per Section 4 layout; FastAPI + Next.js talk to each other; SQLite migrations run; `dev.sh` boots both; README with setup steps. ✅ `curl localhost:8000/health` and the Next.js shell page both work.

**M1 — Engine + golden tests.** Vectorized engine (Section 3): point-in-time discipline, monthly + daily rebalance, equal/rank/inverse-vol weighting, turnover-scaled costs, dividend-adjusted total returns, NYSE trading calendar via `exchange_calendars` or SPY's own index (Gap #6 — document day-count conventions in one table used by engine, LLM prompt, and glossary). Deterministic tie-breaking in ranks (Gap #2). Missing-data policy per Gap #4 (exclusions become audit findings, never silent drops). Resolve Gap #1 by **cutting stop-loss/take-profit from the V0 DSL** and defining `hold_days` overlapping-cohort semantics explicitly (equal-weight across active cohorts) with a golden fixture proving it. ✅ Golden tests pass: (a) zero-cost buy-and-hold SPY equals SPY total return exactly; (b) hand-computed 3-ticker momentum fixture matches to 6 decimals; (c) overlapping-cohort dip-buy fixture matches hand-computed result; (d) a deliberately lookahead-biased variant is impossible to express through the public API.

**M2 — DSL + validation.** JSON Schema per Section 2 with `spec_version` field and defaults-expanded storage (Gap #8). Universe presets: `sp500` (checked-in seed constituent list with documented source + refresh script — Gap #3), `nasdaq100`, `watchlist_upload`. Symbol-master validation at spec time — hallucinated/renamed tickers rejected with suggestions (Gap #4). ✅ The three worked-example specs from Section 2 validate; a spec with ticker "FAKECO" is rejected with a useful error; cold-start S&P fetch is chunked, cached, and resumable.

**M3 — LLM translation + eval harness.** Translator per Section 2's system prompt: NL → spec JSON or clarifying questions. Build the ~30-description eval set NOW as CI fixtures (Gap #7): each case = input text → expected spec (or expected clarifying question). Include semantic traps ("6-month momentum" must be 126 trading days, not 26). Deterministic template readback (non-negotiable #6). ✅ Eval suite ≥90% exact-match on specs, 100% on refuse/clarify cases; ambiguous input ("buy strong stocks") returns clarifying questions, never a guess.

**M4 — Truth Report UI.** Verdict-first banner (red/amber/green WITH icon + text labels — Gap #12), side-by-side stat row, audit finding cards ordered by severity with plain-English explanations, cost-sweep table with break-even highlighting, log-scale equity curve (net-of-costs as the primary line, gross visually subordinate), drawdown chart, regime-coverage bar, out-of-sample panel with overfitting warning, multiple-testing counter. Benchmark-appropriateness check per Gap #5 (realized-beta mismatch → warning + suggested re-run against 60/40). Responsive on mobile widths; charts have text/table fallbacks (Gap #12). ✅ Running the rigged-universe fixture produces a red/amber verdict with the selection-bias finding card and a working "re-run on S&P 500" action; all three worked examples render complete reports.

**M5 — Journal + snapshots + export.** Strategy library with version history and run history showing verdict badges (not returns); compare-runs diff view; per-run frozen parquet snapshot + content hash with replay-by-default (Gap #2); spec JSON export/import; Truth Report export as PNG/PDF with the disclaimer baked into the artifact (Gap #10). Local event log (translation success/clarify/refusal rates, spec-confirm abandonment, verdict distribution) written to SQLite — doubles as the future eval corpus (Gap #11). ✅ Re-running a saved strategy from its snapshot reproduces the prior report byte-for-byte on metrics; export/import round-trips a spec; exported PNG contains the disclaimer.

**M6 — Polish + docs.** First-run onboarding (empty-state with starter chips), error/edge states from Section 1 (yfinance rate-limit fallback to labeled cached data, unsupported-idea refusal listing supported blocks, insufficient-history warning BEFORE running), loading checklist themed as the audit working, glossary page with metric formulas, README covering setup/architecture/how to add a DSL block. ✅ Full Definition of Done below passes.

## Definition of Done (verify every line, honestly report any that fail)

- [ ] All golden, unit, and LLM-eval tests pass (`make test` runs everything)
- [ ] The three worked-example strategies run end-to-end from text box to Truth Report
- [ ] The rigged-universe fixture triggers the selection-bias audit (the fake +1650% is caught)
- [ ] Cost sweep flips the dip-buying example from beating SPY (gross) to losing (20 bps) — reproducing the session's real result
- [ ] Snapshot replay is deterministic; "refresh data" is the only path to new numbers
- [ ] No route exists that runs a backtest without the audit; no export lacks the disclaimer
- [ ] `ANTHROPIC_API_KEY` only via `.env`; `.env` gitignored; no secrets in repo
- [ ] App boots with one command on macOS; README verified by following it verbatim

## Working style

Fix root causes, not symptoms. If a brief requirement is ambiguous, choose the interpretation consistent with the product thesis (honesty > flattery) and log the decision in `DECISIONS.md`. If something in the brief cannot be met, say so explicitly in your final report — do not quietly ship a weaker version. Final report: what was built, test results (including failures), known limitations, and suggested next steps.
