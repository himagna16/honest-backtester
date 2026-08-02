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