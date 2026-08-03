# Deploying — ready when you are

Everything in this folder is **prepared but not executed**. Nothing is public.
Deploying is deliberately a decision you make, not something that happened as a
side effect of a build: see `SAAS_PLAN.md` for the two blockers that gate
*charging* (data licence, legal review). Putting the **landing page** online to
collect waitlist signups is fine before those — that is the validation step.

## What is where

| File | Purpose |
|---|---|
| `../frontend/vercel.json` | Vercel config for the Next.js frontend |
| `Dockerfile` | Container for the FastAPI backend (Fly.io / Railway / anywhere) |
| `fly.toml` | Fly.io app config |
| `.dockerignore` | Keeps secrets, the local DB, and caches out of the image |

## Order

**1. Frontend → Vercel** (free tier, ~5 min)

The repo is already on GitHub (`himagna16/honest-backtester`, private).

- https://vercel.com → sign in with GitHub → **Add New Project** → import the repo
- **Root Directory: `frontend`** ← the one setting people miss
- Deploy. You get a `*.vercel.app` URL.

The landing page works fully without a backend — the demo, charts, and copy are
all client-side. Only the **waitlist form** needs the API. So you can ship the
landing first and wire the backend after.

**2. Backend → Fly.io** (~$0–5/mo)

```bash
brew install flyctl          # if needed
fly auth login               # opens a browser — your account, your call
fly launch --no-deploy       # reads fly.toml; pick a region near you
fly volumes create honest_data --size 1   # persistent disk for SQLite + panels
fly deploy
```

**3. Point the frontend at it**

In Vercel → Project → Settings → Environment Variables:

```
NEXT_PUBLIC_API_BASE = https://<your-app>.fly.dev
```

Redeploy. Then in `fly.toml`, set `HONEST_CORS_ORIGINS` to your Vercel URL so
the browser is allowed to call the API.

## Before you flip it on — read this

- **The LLM key is server-side only.** Never put a provider key in a
  `NEXT_PUBLIC_*` variable; that ships it to every visitor's browser. Set
  `OPENAI_API_KEY` with `fly secrets set OPENAI_API_KEY=...` (encrypted at
  rest, never in the repo).
- **Translation costs real money once public.** `gpt-5-nano` is ~$0.0002 per
  translation ($5 ≈ 27,000). `/api/translate-strategy` is now **rate-limited**
  (backend/ratelimit.py): per-IP burst + daily, plus a global daily budget
  guard tuned to keep a busy month under ~$5. Tune via `HONEST_RL_*`. Local
  calls are exempt. The real hard ceiling stays your OpenAI billing cap — set
  a low one and the app now fails gracefully (clear "unavailable") when it's hit.
- **yfinance is not licensed for commercial use.** Running the backtester
  publicly is a different posture than running it locally. See `SAAS_PLAN.md`.
- **The waitlist DB holds real email addresses.** It lives on the Fly volume,
  is gitignored, and must stay out of any image or repo.
