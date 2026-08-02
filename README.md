# The Honest Backtester

Describe a trading strategy in plain English → get a backtest that actively
tries to kill it: bias audits, cost sweeps, and an always-on SPY benchmark on
every single result. *The backtester that tries to kill your strategy before
the market does.*

**Educational research tool only. Not investment advice.**

## Quickstart (macOS)

Prereqs: Python 3.11 (`brew install python@3.11`), Node 18+ (`brew install node`).

```bash
# 1. One-time: seed the offline price cache from the frozen fixtures
make seed

# 2. Optional (only needed for plain-English → spec translation):
cp .env.example .env   # then put your ANTHROPIC_API_KEY in .env

# 3. Run everything (backend :8000*, frontend :3000)
make dev
```

Open http://localhost:3000. Backend API docs at http://localhost:8000/docs.

\* If port 8000 is taken, `dev.sh` automatically falls back to 8010/8020/8030
and points the frontend at the right port. Override with `HONEST_BACKEND_PORT`.

The engine, honesty audits, journal, and exports all work **fully offline** on
the seeded fixture data. Only the translation endpoint needs the API key.

## Tests

```bash
make test    # engine golden tests, DSL validation, API tests, frontend typecheck
make eval    # live LLM translation eval harness (requires ANTHROPIC_API_KEY)
```

## Architecture

- `frontend/` — Next.js (App Router, TypeScript). Pages: describe → confirm
  spec → Truth Report; journal; glossary.
- `backend/` — Python FastAPI.
  - `dsl/` — the strategy spec (constrained JSON DSL), semantic validator,
    day-count conventions (single source of truth), deterministic readback.
  - `engine/` — vectorized pandas backtest, data layer with per-run parquet
    snapshots, metrics.
  - `audit/` — the honesty audit: selection bias, lookahead, cost sweep,
    benchmark truth, regimes, out-of-sample, multiple testing, capacity,
    benchmark appropriateness.
  - `llm/` — Claude translation (English → spec JSON or clarifying questions).
  - `store/` — SQLite via SQLAlchemy; plain-SQL migrations.
- `fixtures/` — frozen price panels + constituent lists (offline test data).
- `data/` — gitignored: SQLite DB, parquet price cache, run snapshots, exports.

See `DECISIONS.md` for ambiguity resolutions and `PRODUCT_BRIEF.md` for the
full spec. This README is expanded with full docs in milestone M6.
