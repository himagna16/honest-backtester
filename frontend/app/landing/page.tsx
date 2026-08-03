"use client";

// Public landing page for the validation sprint (brief Section 5, step 2).
// The founding story IS the marketing asset — lead with the number that was
// wrong, not with a feature list.

import { useState } from "react";
import Link from "next/link";
import { api } from "@/lib/api";
import { DISCLAIMER } from "@/lib/constants";
import {
  IconAudit, IconCheck, IconRisk, IconSnapshot, IconTimeline, IconX,
} from "@/components/ui/Icons";
import { Notice } from "@/components/ui/Primitives";

const CHECKS = [
  { Icon: IconRisk, title: "Selection-bias detection",
    body: "Hand-picked a watchlist? Your exact rules are re-run on the full S&P 500 and compared. Hindsight gets named, not rewarded." },
  { Icon: IconTimeline, title: "Cost sweep at 0 / 5 / 20 bps",
    body: "Scaled by your strategy's measured turnover. The break-even cost level is shown, so you see exactly where the edge dies." },
  { Icon: IconAudit, title: "Train / holdout split",
    body: "The verdict is computed on the 30% you didn't get to look at. A big in-sample advantage is flagged as the overfit signature it is." },
  { Icon: IconCheck, title: "Lookahead impossible by construction",
    body: "Signals at day t, returns from t+1 — enforced inside the engine, not by convention. Altering future prices cannot change past weights." },
  { Icon: IconTimeline, title: "Regime coverage",
    body: "If your window contains no sustained bear market, the report says so. A strategy tested only in a bull run has not been tested." },
  { Icon: IconSnapshot, title: "Reproducible by default",
    body: "Every run freezes its exact price panel with a content hash. Re-runs replay it byte-for-byte, so your journal can't drift under you." },
];

export default function Landing() {
  const [email, setEmail] = useState("");
  const [note, setNote] = useState("");
  const [state, setState] = useState<"idle" | "sending" | "done" | "error">("idle");
  const [message, setMessage] = useState("");

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setState("sending");
    try {
      const r = await api<{ ok: boolean; already: boolean }>("/api/waitlist", {
        method: "POST", json: { email, note, source: "landing" },
      });
      setState("done");
      setMessage(r.already ? "You're already on the list." : "You're on the list.");
    } catch (err) {
      setState("error");
      setMessage((err as Error).message);
    }
  };

  return (
    <div style={{ background: "var(--bg)", minHeight: "100vh" }}>
      <nav className="landing-nav">
        <span className="brand">
          <svg width="16" height="16" viewBox="0 0 24 24" aria-hidden="true">
            <rect x="2" y="2" width="20" height="20" rx="5" fill="#0f172a" />
            <path d="M6 16.5 10 11l3 3 5-7.5" stroke="#7dd3fc" strokeWidth="2"
              fill="none" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          The Honest Backtester
        </span>
        <span className="spacer" />
        <a href="#how">How it works</a>
        <a href="#pricing">Pricing</a>
        <Link className="btn" href="/">Open the app</Link>
      </nav>

      <main className="landing">
        {/* ---------------- hero ---------------- */}
        <section className="hero">
          <span className="kicker"><IconRisk size={12} /> Educational research tool</span>
          <h1>My backtest said +1650%. Then I removed the cheating.</h1>
          <p className="lede">
            Every retail backtester would have shown me that first number and
            none of the corrections. So I built the one that argues back — it
            runs a full honesty audit on every result, and there is no way to
            turn it off.
          </p>

          <div className="storystrip">
            <div className="step fake">
              <div className="lbl">What the tool said</div>
              <div className="n">+1650%</div>
              <div className="why">
                44 hand-picked tickers I chose in 2026, tested from 2016. I was
                buying winners I already knew had won.
              </div>
            </div>
            <div className="step real">
              <div className="lbl">On a neutral universe</div>
              <div className="n">+163%</div>
              <div className="why">
                Same rules, honest universe. Still sounds great — until you
                notice SPY buy-and-hold returned +362% over the same window.
              </div>
            </div>
            <div className="step dead">
              <div className="lbl">After realistic costs</div>
              <div className="n">Killed</div>
              <div className="why">
                A second strategy beat SPY gross on Sharpe 1.05 vs 0.83 — and
                died completely at 5 bps of trading costs.
              </div>
            </div>
          </div>

          <p className="lede" style={{ marginBottom: 0 }}>
            Three numbers, one afternoon. The audit that caught them is now the
            product.
          </p>
        </section>

        {/* ---------------- what it does ---------------- */}
        <section className="section" id="how">
          <h2>Describe a strategy. It tries to kill it.</h2>
          <p className="sub">
            Plain English goes in. A precise, editable spec comes back for you
            to confirm — then every result ships with the full audit and an
            always-on benchmark. Nothing runs until you approve exactly what
            will be tested.
          </p>
          <div className="checks">
            {CHECKS.map((c) => (
              <article className="check" key={c.title}>
                <h3><c.Icon size={14} /> {c.title}</h3>
                <p>{c.body}</p>
              </article>
            ))}
          </div>
        </section>

        {/* ---------------- pricing ---------------- */}
        <section className="section" id="pricing">
          <h2>Free forever, for the part that matters</h2>
          <p className="sub">
            The audit is never paywalled. Charging you to find out your strategy
            is broken would invert the entire point of the product — so the
            honesty is free, and the paid tier is better <em>data</em> and
            forward tracking, which is where the real cost sits.
          </p>
          <div className="tiers">
            <div className="tier">
              <span className="name">Free</span>
              <div className="price">$0<small> / forever</small></div>
              <p className="blurb">
                The complete honesty audit, unpaywalled and unabridged.
              </p>
              <ul>
                <li><IconCheck size={13} /> Every audit check — selection bias, cost sweep, regimes, holdout, sensitivity</li>
                <li><IconCheck size={13} /> Benchmark on every result</li>
                <li><IconCheck size={13} /> Plain-English → spec translation</li>
                <li><IconCheck size={13} /> Shareable Truth Report exports</li>
                <li><IconCheck size={13} /> Strategy journal + version history</li>
                <li className="off"><IconX size={13} /> Current index membership only</li>
                <li className="off"><IconX size={13} /> Monthly backtest allowance</li>
              </ul>
              <div className="foot">
                <Link className="btn" href="/">Use it now, no account</Link>
              </div>
            </div>

            <div className="tier featured">
              <span className="name">Pro — planned</span>
              <div className="price">TBD<small> / month</small></div>
              <p className="blurb">
                Survivorship-free data and forward tracking. Priced once the
                data licence is signed — not before.
              </p>
              <ul>
                <li><IconCheck size={13} /> <strong>Point-in-time index membership</strong> — including companies that went to zero</li>
                <li><IconCheck size={13} /> Delisted-stock coverage back through 2008</li>
                <li><IconCheck size={13} /> Forward-testing: freeze a survivor and track it live vs its backtest</li>
                <li><IconCheck size={13} /> Unlimited backtests</li>
                <li><IconCheck size={13} /> Longer history windows</li>
                <li><IconCheck size={13} /> Everything in Free</li>
              </ul>
              <div className="foot">
                <a className="btn primary" href="#waitlist">Join the waitlist</a>
              </div>
            </div>
          </div>
          <p className="faint" style={{ marginTop: 12 }}>
            No pricing is committed yet, and nothing is for sale on this page.
            The paid tier ships only after a licensed data source and a legal
            review are in place.
          </p>
        </section>

        {/* ---------------- waitlist ---------------- */}
        <section className="section" id="waitlist">
          <h2>Tell me what your last backtest lied to you about</h2>
          <p className="sub">
            I&apos;m validating whether this is worth building for anyone but me.
            Leave an email and, if you feel like it, the strategy you&apos;d test
            first — that answer shapes what gets built next.
          </p>

          {state === "done" ? (
            <Notice kind="info">
              <strong>{message}</strong> No drip sequence, no newsletter — one
              email when there&apos;s something real to try.
            </Notice>
          ) : (
            <form className="waitform" onSubmit={submit}>
              <input type="email" required value={email} aria-label="Email address"
                placeholder="you@example.com"
                onChange={(e) => setEmail(e.target.value)} />
              <button className="primary" type="submit" disabled={state === "sending"}>
                {state === "sending" ? "Adding…" : "Join the waitlist"}
              </button>
              <textarea className="waitnote" value={note} aria-label="What did your last backtest lie to you about?"
                placeholder="Optional: what did your last backtest lie to you about?"
                onChange={(e) => setNote(e.target.value)} />
            </form>
          )}
          {state === "error" && <Notice kind="err">{message}</Notice>}
        </section>

        <footer className="landing-foot">
          <p style={{ marginTop: 0 }}>{DISCLAIMER}</p>
          <p>
            No trade execution, no signals, no personalised advice — permanently
            off the roadmap. Your strategy text is confidential and is never
            sent to third-party analytics.
          </p>
        </footer>
      </main>
    </div>
  );
}
