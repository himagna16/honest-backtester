# The Honest Backtester — build & launch checklist

**You are here:** the product is functionally complete (M0–M6). The `/landing`
page has the new "Broadsheet" design; the *rest of the app is still on the old
look*. Nothing is deployed. No LLM key is set, so live translation is off (but
everything else works). Waitlist: 2 / 50 signups.

This file is the running to-do list. Rationale for the paid-tier items lives in
[SAAS_PLAN.md](SAAS_PLAN.md); this is the "what to do and where to go" version.

Legend:  ☐ = to do   ✅ = done   🔑 = needs an account/key you create (I can't)

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

## Phase 1 — Turn on live AI  🔑 (≈10 min · usage-based, cents)

This enables the "describe a strategy in plain English" box. Everything else
already works without it.

- 🔑 **Get an OpenAI API key**
  - Go to **https://platform.openai.com/api-keys** → *Create new secret key*
  - You must add a payment method first at **https://platform.openai.com/settings/organization/billing**
    (put a small cap, e.g. $5/month — nano is ~$0.0003 per translation)
- ☐ **Create your `.env`** (in the repo root):
  ```bash
  cp .env.example .env
  ```
  Then edit `.env` and set:
  ```
  OPENAI_API_KEY=sk-...your key...
  HONEST_LLM_MODEL=gpt-5-nano
  ```
  `.env` is gitignored — the key never gets committed.
- ☐ **Restart the backend** (Ctrl-C `dev.sh`, run it again).
- ☐ **Test it:** open the app, type a strategy in plain English, confirm it
  returns an editable spec.
- ☐ *(optional)* compare models on cost + accuracy before committing:
  ```bash
  make bench
  ```

---

## Phase 2 — Make the whole app look like the landing  (local build · free)

Right now `/landing` is premium and the inner pages are the old blue-on-slate
look. These pages still need the Broadsheet treatment:

- ☐ `/` — the workbench (home / strategy input)  ← most visible, do first
- ☐ `/report/[id]` — the Truth Report
- ☐ `/journal` — run history
- ☐ `/compare` — strategy vs benchmark
- ☐ `/strategy/[id]` — a saved strategy
- ☐ `/glossary` — the terms page

This is a "just ask me" item — no accounts, no cost. I'll do it whenever you say.

---

## Phase 3 — Back up the code to GitHub  🔑 (≈5 min · free)

Doesn't make the site public — just saves the source safely.

- 🔑 Your `gh` CLI is already logged in (as `himagna16`).
- ☐ Decide **private** (recommended) or public.
- ☐ Say the word and I'll run `gh repo create` + push. Or do it yourself:
  ```bash
  gh repo create honest-backtester --private --source=. --push
  ```

---

## Phase 4 — Validate: put the landing online + collect signups  🔑

The goal here is a shareable link and **50 waitlist signups** (the go/no-go bar).
Don't build the paid tier before this clears.

- 🔑 **Frontend → Vercel** (free tier, gives a public URL)
  - Requires Phase 3 first (Vercel deploys from a GitHub repo).
  - Go to **https://vercel.com** → sign in with GitHub → *Add New Project* →
    import the repo → set **Root Directory = `frontend`** → Deploy.
  - Add env var `NEXT_PUBLIC_API_BASE` = your backend URL (see next step).
- 🔑 **Backend → a host** (needed only so the waitlist form works)
  - **https://fly.io** or **https://railway.app** — deploy the FastAPI app.
  - The demo, charts, and all visuals work *without* the backend; only the
    waitlist POST needs it. (You could launch frontend-only first and wire the
    backend later.)
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
  licensed for commercial use *and* is survivorship-biased (the exact bias this
  product exists to expose). This is also the paid tier's actual value.
  - **https://norgatedata.com** ($30–60/mo, delisted-stock coverage) ·
    **https://eodhd.com** · **https://polygon.io**
- 🔑 **Legal review** (~$1–3k) — a fintech-focused startup attorney reads the
  "educational tool vs. investment adviser" line *before* you take money.
  Find one via a startup-attorney referral or your state bar's referral service.
  (I can't give legal advice — this is where to go, not what they'll say.)
- ☐ **Content-addressed snapshots** — a code change so 1,000 users testing the
  same universe don't each store a copy of the price panel. Do before real
  traffic. (Contained change in `engine/data.py`; ask me.)

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

- **I can:** finish the redesign (Phase 2), create + push the GitHub repo
  (Phase 3), write the Vercel/Fly config and the content-addressed snapshot
  change, wire any catalogue model.
- **Only you can (🔑):** create accounts and paste keys into `.env`, add
  billing to OpenAI/Stripe, sign the data licence, hire the attorney. I never
  handle keys or credentials.
