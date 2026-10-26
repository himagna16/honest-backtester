# The Honest Backtester — build & launch checklist

**You are here:** ✅ **FULLY LIVE end-to-end.** Frontend on Vercel, backend on
Render (free), wired together. **Private** repo, live **site**:
https://honest-backtester-six.vercel.app · repo:
https://github.com/himagna16/honest-backtester (private — the source is closed
so the engine can become a paid tier later; the live site stays public).
Translate → run →
Truth Report → PNG export and the waitlist all work on the live site. Only
caveat: a full S&P 500 run is too heavy for Render's free 512 MB (light
universes like Dow 30 run fine); the hosted instance also sleeps after ~15 min
idle. Portfolio-ready today.

This file is the running to-do list. Rationale for the paid-tier items lives in
[SAAS_PLAN.md](SAAS_PLAN.md); this is the "what to do and where to go" version.

Legend:  ☐ = to do   ✅ = done   🔑 = needs an account/key you create (I can't)

---

## Now — point-in-time S&P 500 membership (branch `pit-sp500-membership`)

Built and tested locally, **not deployed**. See DECISIONS.md → "Point-in-time
S&P 500 membership" and `docs/pit_membership_study.md`.

- ✅ Membership rebuilt from cited sources, engine + audit use it, 8 saved
  reports re-run, landing demo/export/README regenerated from the re-runs
- ☐ **Sai: review the numbers, then OK a deploy** (Vercel frontend + Render
  backend). The new fixture `fixtures/sp500_pit_prices.parquet` ships with the
  backend and seeds on first boot. Note that S&P 500 runs now load ~90 more
  names, and a full S&P 500 run was already too heavy for Render's free tier.
- ☐ Existing local caches: `.venv/bin/python scripts/seed_cache.py --backfill-pit`
- ☐ Decide: should "margin inside the noise" / "no factor alpha" block a green
  verdict? (Four point-in-time S&P 500 re-runs are green without it.)

---

## Phase 0 — Run it and look at it  (now · free · no signup)

- ✅ App runs locally
- **Do this to start it:**
  ```bash
  ./dev.sh
  ```
  First run makes the Python venv and installs everything (a few minutes).
  Then open:
  - App (workbench): **http://localhost:3000**
  - New landing page: **http://localhost:3000/landing**
  - Backend API docs: **http://localhost:8010/docs** (port is printed by dev.sh)
- Every edit hot-reloads — no push or restart needed to see changes.

---

## Phase 1 — Turn on live AI  ✅ DONE

This enables the "describe a strategy in plain English" box. Everything else
already works without it.

- ✅ OpenAI key created, credits added, `.env` written (gitignored)
- ✅ `HONEST_LLM_MODEL=gpt-5-nano` — the cheapest model in the catalogue
- ✅ Verified with a live translation: "buy the 5 S&P 500 stocks with the
  highest 12-month return" → correct spec, 252-day lookback, monthly, equal weight
- ✅ Fixed `/health`, which reported "no API key" for non-Anthropic providers
- ☐ *(optional)* compare models on cost + accuracy before committing:
  ```bash
  make bench
  ```

---

## Phase 2 — One design system across the app  ✅ DONE

- ✅ `/` workbench · `/report/[id]` · `/journal` · `/compare` · `/strategy/[id]` · `/glossary`
- ✅ Shared palette/fonts/geometry with the landing, but a denser type ramp —
  one system, two densities. Fonts load once in the root layout.
- ✅ Measured: 7 type sizes, one 2px radius, zero shadows, zero WCAG failures,
  no horizontal overflow at 414px.
- ✅ Bugs found and fixed: `/health` provider check, a test that silently
  stopped testing once a real key existed, and a 2px mobile overflow from the
  result tab bar.

---

## Phase 3 — Code backed up to GitHub  ✅ DONE

- ✅ **https://github.com/himagna16/honest-backtester** — **private**
- ✅ Verified before pushing: `.env` untracked and gitignored, no key patterns
  anywhere in history or the working tree, and `data/` (which holds real
  waitlist email addresses) excluded.
- To push future work: `git push`

---

## Phase 4 — Validate: put the landing online + collect signups  🔑

The goal here is a shareable link and **50 waitlist signups** (the go/no-go bar).
Don't build the paid tier before this clears.

**Configs are written and validated — see [deploy/README.md](deploy/README.md)
for the exact steps.** I stopped short of deploying: it needs your Vercel and
Fly accounts (interactive sign-in), and putting the app on the public internet
is your call, not a build side-effect.

- ✅ **Frontend → Vercel — LIVE at https://honest-backtester-six.vercel.app**
  (account `himagna16` / `sai-maruvadas-projects`). `/` redirects to `/landing`.
  Redeploy after changes: `cd frontend && npx vercel deploy --prod`.
  - ⚠️ **The waitlist form does not work yet** — it needs the backend (next
    item). The demo, charts and copy all work. Do NOT post the link publicly
    until signups actually record, or you waste the launch.
- 🔑 **Backend → Fly.io** (needed only so the waitlist form works)
  - `deploy/Dockerfile` + `deploy/fly.toml` are ready; volume-backed so the
    journal survives redeploys. `fly auth login && fly launch --no-deploy`.
  - The landing demo, charts and copy work *without* the backend — you can ship
    frontend-only first and wire the API later.
  - ⚠️ Set the key as a secret (`fly secrets set OPENAI_API_KEY=...`), never as
    a `NEXT_PUBLIC_*` var, and rate-limit `/api/translate-strategy` before
    exposing it — a public LLM endpoint is a public wallet.
- ☐ Post the +1650% story where your audience is (r/algotrading, r/quant,
  FinTwit) — respecting each venue's self-promo rules. The story is the asset.
- ☐ Track progress: `GET /api/waitlist/stats` (or the export CSV).
- **Kill criterion (real):** under ~25 signups + under 10 substantive replies
  after posting in 3+ venues in 3 weeks → stop here; it stays a great personal
  tool. That's a fine outcome, not a failure.

---

## Phase 5 — Before you charge ANYONE  🔑 (gated on Phase 4 passing)

These are the blockers that can sink the business if skipped. Details and
reasoning in [SAAS_PLAN.md](SAAS_PLAN.md).

- 🔑 **Licensed price data** — the single most important one. yfinance is not
  licensed for commercial use *and* has no delisted-stock prices. Point-in-time
  S&P 500 *membership* is now rebuilt from free cited sources, but members with
  no free price history still can't be picked. This is also the paid tier's
  actual value.
  - **https://norgatedata.com** ($30–60/mo, delisted-stock coverage) ·
    **https://eodhd.com** · **https://polygon.io**
- 🔑 **Legal review** (~$1–3k) — a fintech-focused startup attorney reads the
  "educational tool vs. investment adviser" line *before* you take money.
  Find one via a startup-attorney referral or your state bar's referral service.
  (I can't give legal advice — this is where to go, not what they'll say.)
- ✅ **Content-addressed snapshots** — DONE. Panel bytes now live once in
  `data/panels/<sha256>.parquet`, shared by every run that froze an identical
  panel; run directories keep metadata only. Verified end-to-end: a second full
  backtest of the same spec added **0 bytes** (was ~22 MB). Old runs still
  replay, and reproducibility is unchanged — the hash *is* the identity.

---

## Phase 6 — Monetize  🔑 (only after Phase 5)

- 🔑 **Auth** — **https://clerk.com** or **https://supabase.com** (~$0–25/mo).
  Multi-tenancy is mostly a `user_id` column; schema is already close.
- 🔑 **Billing** — **https://dashboard.stripe.com**. Subscriptions + webhooks.
  Gate the paid tier on *data quality*, never on the audit.
- 🔑 **Monitoring** — **https://sentry.io** (~$0–26/mo) so you know when the
  engine silently returns garbage.
- ☐ **Rate limiting** so one user can't starve others.

---

## Quick reference — where to go

| Need | Where | Cost |
|---|---|---|
| OpenAI key (live AI) | platform.openai.com/api-keys | ~cents/use |
| Code backup | github.com (gh already authed) | free |
| Host the landing | vercel.com | free tier |
| Host the backend | fly.io / railway.app | ~$0–5/mo |
| Licensed data | norgatedata.com / eodhd.com / polygon.io | $30–200/mo |
| Auth | clerk.com / supabase.com | $0–25/mo |
| Payments | dashboard.stripe.com | % of revenue |
| Error monitoring | sentry.io | $0–26/mo |

**Realistic running cost once live:** ~$100–350/mo + one-time legal $1–3k.
Break-even ≈ 10–25 subscribers at $15–29/mo.

---

## What I can do for you vs. what only you can do

- **I did:** the app-wide redesign (Phase 2), the private GitHub repo
  (Phase 3), the Vercel/Fly/Docker configs, and content-addressed snapshots.
- **I can still:** rate limiting, the auth `user_id` scaffolding, Stripe
  webhook handlers — all buildable before you hold the accounts.
- **Only you can (🔑):** create accounts and paste keys into `.env`, add
  billing to OpenAI/Stripe, sign the data licence, hire the attorney. I never
  handle keys or credentials.
