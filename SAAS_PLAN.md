# Free tier, paid tier, and the road to SaaS

Status: **landing page + waitlist shipped. Auth and billing deliberately not
built yet.** This document says why, and what order the rest goes in.

---

## Two blockers that must clear before you charge anyone

These aren't process theatre — each one can sink the product if you take money
first and discover them second.

### 1. yfinance forbids commercial use, and it's survivorship-biased

Your own brief calls this out: *"yfinance is against Yahoo ToS for a commercial
product and is itself survivorship-biased — a fatal irony for an 'honesty'
product."*

Charging subscriptions while the data underneath is (a) not licensed for
resale and (b) exactly the bias your product exists to expose is the one
failure mode that kills this specific business. A competitor, a journalist, or
a single user reading the audit's own survivorship disclaimer can make that
point for you.

**Cost to clear:** $30–200/month.

| Provider | ~Cost | Notes |
|---|---|---|
| Norgate | $30–60/mo | Delisted-stock coverage; the hobbyist-pro standard |
| EODHD | $20–80/mo | Broad coverage, decent API |
| Polygon | $29–199/mo | Good infra, verify point-in-time constituents |

**This is not just a licence swap — it's the paid tier's actual product.**
Point-in-time index membership is what lets you say "we tested the companies
that existed *then*, including the ones that went to zero." Nobody else in the
retail tier can say that. It's simultaneously your biggest cost, your moat, and
the honest answer to the caveat your app currently prints on every run.

### 2. Legal review before the first paid subscription

Also from the brief: *"paid legal review (~$1–3k, a fintech-adjacent startup
attorney) before the first paid subscription. Do not skip this to save money."*

The line between "educational calculator" and "investment adviser" has real
regulatory edges (SEC/state RIA rules), and the edges live exactly where
personalised output lives. You are currently well clear — no advice language,
no signals, no execution — but a paid product with a "your strategy survived"
verdict deserves a professional read of that line before money changes hands.

**Sequencing consequence:** build billing *after* the legal review, not before.
Otherwise you may have to change the product after already charging for it.

---

## The tier split, and the one thing that can't be paywalled

**The audit is never behind the paywall.** Charging someone to discover their
strategy is broken inverts the entire product thesis, and it violates your own
non-negotiable #1 (no way to run a backtest without the full audit). If the
free tier shows a flattering number and asks for money to reveal the truth,
you have built the thing you set out to replace.

So the split is not *audit vs no audit*. It's **data quality and forward
tracking** — which happens to be exactly where your real marginal cost sits.

| | Free | Pro |
|---|---|---|
| Full honesty audit | ✅ every check | ✅ |
| Benchmark on every result | ✅ | ✅ |
| English → spec translation | ✅ | ✅ |
| Truth Report export (PNG/PDF) | ✅ — this is the viral loop, never gate it | ✅ |
| Journal, versions, snapshot replay | ✅ | ✅ |
| Universe data | Current index membership (survivorship disclosed) | **Point-in-time membership incl. delisted** |
| History depth | ~10 years | Full licensed history |
| Backtests | Metered (see below) | Unlimited |
| Forward testing / graduation | — | ✅ freeze a survivor, track live vs backtest |

**Why export stays free:** the brief identifies shareable Truth Report cards as
the growth loop — *"Claimed +748%. After 20bps costs: −97%. Verdict: KILLED"*
is inherently screenshot-able on FinTwit. Gating your own distribution channel
to protect a $15/mo tier is a bad trade.

**Forward testing is the retention product.** Gap #9 in your brief flagged it:
survivors get frozen (immutable, hash-referenced spec), evaluated daily, and
reported as live-vs-backtest drift ("forward Sharpe 0.4 vs backtest 1.1").
That's a reason to keep paying after the audit has done its job — and the
snapshot/spec-immutability design in V0 already supports it.

---

## What the free tier actually costs you

Measured against the current architecture, per free user:

| Cost driver | Reality |
|---|---|
| **LLM translation** | Negligible — ~$0.0002–0.0003 per translation on a cheap tier, once per *strategy idea* (re-runs, variants, and S&P re-runs are free). 1,000 users × 20 ideas ≈ $6/month. |
| **Backtest compute** | The real cost. A 500-ticker S&P run is seconds of CPU, and the selection-bias audit runs a *second* full backtest. This is what to meter. |
| **Price data** | Fixed monthly licence, not per-user. Scales with *coverage*, not traffic. |
| **Run snapshots** | ⚠️ **The scaling landmine — see below.** |

### The snapshot design doesn't survive multi-user as-is

Every run freezes its full price panel to `data/runs/<run_id>/panel.parquet`.
That's what makes V0 byte-for-byte reproducible, and it must stay — but at
multi-user scale it's ruinous: a 500-ticker × 10-year panel is a few MB, and
1,000 users × 10 runs is tens of gigabytes of near-identical data.

**Fix before launch, not after:** the panel is already content-hashed
(`panel_content_hash`). Store panels **content-addressed** — one blob per
unique hash, with runs referencing it. Users running the same
universe+window (which will be the overwhelming majority — everyone tests
S&P 500 momentum) share a single stored panel. Same reproducibility
guarantee, a fraction of the storage. This is a contained change to
`engine/data.py::save_snapshot` and the `runs` table.

**Metering suggestion:** count backtests, not translations. Something like
20 runs/month free — enough to genuinely test several ideas including the
audit's re-runs, cheap enough to serve at scale. Meter *after* you know real
usage; the local event log already records `run_started` / `run_completed`.

---

## Order of operations

**Now → 3 weeks: validate. No auth, no billing, no data licence.**

1. ✅ Landing page + waitlist (`/landing`, shipped).
2. Post the +1650% story to X and r/algotrading, r/quant, r/investing —
   respecting each sub's self-promo rules. The arc is the asset: concrete,
   verifiable, and embarrassing in a relatable way.
3. **Go/no-go bar (from the brief):** 50+ waitlist emails, or 10+ substantive
   "I'd use this" replies naming a strategy or a bias that bit them, within
   3 weeks. Upvotes don't count. `GET /api/waitlist/stats` tracks it.
4. **Kill criterion is real:** under 10 substantive replies and under 25
   signups after posting in 3+ venues → stop V1 planning. V0 stays a
   genuinely valuable personal tool and quant-interview portfolio piece.
   That's a soft landing, not a failure.
5. DM 5–10 engaged repliers: *"what did your last backtest lie to you about?"*
   Their answers are the V1 audit-feature backlog. The waitlist note field
   collects this passively.

**If validated → build in this order:**

6. **Data licence first.** It's the paid tier's actual value, the ToS blocker,
   and the longest lead time. Swap `engine/data.py` behind the existing
   `PricePanel` interface — the audit and engine shouldn't need to change.
7. **Content-addressed snapshots** (above) before any real traffic.
8. **Auth.** Clerk or Supabase Auth (~$0–25/mo at small scale). Multi-tenancy
   is mostly a `user_id` column on `strategies` / `runs` / `waitlist` plus
   scoping in `store/repo.py` — the schema is already close.
9. **Legal review** (~$1–3k). Before billing, not after.
10. **Stripe subscriptions + metering.** You've used Payment Links; webhooks
    are the step up. Gate on the data tier, not on the audit.
11. **Hosting.** Next.js → Vercel; FastAPI → Fly.io/Railway. Backtests are
    CPU-bound, so move runs to a real queue (the SSE endpoints stay the same
    — `routes/runs.py` was written with that swap in mind).
12. **Rate limiting + monitoring.** One user must not starve others. Sentry
    (~$0–26/mo) so you know when the engine silently returns garbage.

**Realistic V1 running cost:** ~$100–350/month, plus one-time legal $1–3k.
Breakeven around 10–25 subscribers at $15–29/mo.

---

## What's already SaaS-ready

Worth knowing how much of this you don't have to redo:

- **SQLAlchemy + plain-SQL migrations** — the Postgres swap is a connection
  string plus dialect review.
- **The DSL is the security boundary.** No arbitrary code execution means
  multi-tenancy doesn't open an RCE surface. Keep it that way permanently.
- **Snapshots + content hashes** already give per-run reproducibility, which
  is what forward-testing needs.
- **Spec versioning** (`strategy_versions`) already supports the immutable
  hash-referenced freeze that graduation requires.
- **The local event log** is your analytics — and strategy text is
  confidential user IP that must never go to a third-party analytics vendor.
  Keep that stance when you add hosted analytics.
- **Provider-agnostic LLM layer** with a cost bench, so translation cost per
  free user is a measured number you can control.

---

## What is deliberately NOT built yet

Auth, Stripe, hosted deploy, usage metering, and the data-source swap. Not
because they're hard, but because building them before the waitlist validates
is the specific mistake the brief was written to prevent — and because two of
them are gated on a licence and a lawyer.
