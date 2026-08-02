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