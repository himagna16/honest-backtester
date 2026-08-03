"use client";

// Public landing page for the validation sprint (brief Section 5, step 2).
// Broadsheet treatment: the founding story is reported, not pitched. Editorial
// rules and whitespace replace bordered cards; the only image on the page is a
// real Truth Report exported from the running app.

import { useState } from "react";
import Link from "next/link";
import Image from "next/image";
import { api } from "@/lib/api";
import { DISCLAIMER } from "@/lib/constants";
import { Notice } from "@/components/ui/Primitives";

const ENTRIES = [
  ["01", "Selection bias, named out loud",
    "Hand-pick a watchlist and your exact rules are re-run on the full S&P 500 for comparison. Hindsight gets reported, not rewarded."],
  ["02", "Costs swept at 0, 5 and 20 bps",
    "Scaled by your strategy's measured turnover. The report states the cost level at which the edge dies — often lower than you'd guess."],
  ["03", "A verdict you didn't get to see",
    "Thirty per cent of the window is held out. A large in-sample advantage that vanishes there is flagged as the overfit signature it is."],
  ["04", "Lookahead made impossible",
    "Signals at day t, returns from t+1, enforced inside the engine rather than by convention. Altering future prices cannot change past weights."],
  ["05", "Regime coverage, stated plainly",
    "If your window contains no sustained bear market, the report says so. A strategy tested only through a bull run has not been tested."],
  ["06", "Reproducible to the byte",
    "Every run freezes its price panel under a content hash and replays it exactly. Your journal cannot quietly drift underneath you."],
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
    <>
      <header className="bs-masthead-outer">
        <div className="bs-wrap">
          <div className="bs-masthead">
            <Link className="title" href="/landing">
              <svg width="15" height="15" viewBox="0 0 24 24" aria-hidden="true">
                <path d="M3 20 9 11l4 4 8-12" stroke="#7a2e2e" strokeWidth="2.6"
                  fill="none" strokeLinecap="square" />
              </svg>
              The Honest Backtester
            </Link>
            <nav>
              <a href="#audit">The audit</a>
              <a href="#terms">Terms</a>
              <a href="#waitlist">Waitlist</a>
              <Link href="/">Open the app →</Link>
            </nav>
          </div>
        </div>
      </header>

      <main>
        {/* ---------------- hero ---------------- */}
        <section className="bs-hero">
          <div className="bs-wrap">
            <p className="bs-kicker">Educational research tool · Nothing here is advice</p>
            <h1>
              My backtest said <em>+1650%</em>. Then I removed the cheating.
            </h1>
            <hr className="rule-under" />
            <div className="lede-row">
              <p className="standfirst">
                Every retail backtester would have shown me that first number and
                none of the corrections. So I built the one that <em>argues back</em>
                {" "}— it runs a full honesty audit on every result, and there is
                no way to turn it off.
              </p>
              <p className="byline">
                <span className="drop">T</span>he number came from forty-four
                tickers I picked in 2026 and tested from 2016. I was buying
                winners I already knew had won. Running the identical rules on a
                neutral universe cut the return by a factor of ten — and the
                benchmark still beat it. A second strategy looked better than
                SPY on Sharpe, right up until five basis points of trading cost
                erased it entirely. Three numbers, one afternoon. The audit that
                caught them is now the product.
              </p>
            </div>

            <div className="bs-figures">
              <div className="bs-fig a">
                <div className="cap">What the tool said</div>
                <div className="num">+1650%</div>
                <p>Forty-four hand-picked tickers, chosen with the benefit of a decade of hindsight.</p>
              </div>
              <div className="bs-fig b">
                <div className="cap">On a neutral universe</div>
                <div className="num">+163%</div>
                <p>Same rules, honest universe. Still sounds good — until you notice SPY returned +362%.</p>
              </div>
              <div className="bs-fig c">
                <div className="cap">After realistic costs</div>
                <div className="num">Killed</div>
                <p>Beat SPY gross on Sharpe, 1.05 against 0.83. Died completely at five basis points.</p>
              </div>
            </div>
          </div>
        </section>

        {/* ---------------- the product shot ---------------- */}
        <section className="bs-section">
          <div className="bs-wrap">
            <p className="bs-kicker">Every run ends here</p>
            <h2>The verdict comes first. The flattering number comes second.</h2>
            <p className="intro">
              Plain English goes in; a precise, editable spec comes back for you
              to approve. Nothing runs until you confirm exactly what will be
              tested — and what comes out is this.
            </p>
            <figure className="bs-plate">
              <div className="frame">
                <div className="chrome"><i /><i /><i /></div>
                <div className="shot">
                  <Image src="/truth-report-sample.png" width={1500} height={1875}
                    alt="An exported Truth Report: an amber verdict reading “Beats the benchmark — but your universe is hand-picked”, above the strategy's return and Sharpe shown next to the SPY benchmark, a cost sweep table, and the equity curve."
                    priority />
                </div>
              </div>
              <figcaption>
                A real export from the running application, not a mock-up. The
                benchmark row and the disclaimer are baked into the image file
                itself, so a screenshot on social media carries them too.
              </figcaption>
            </figure>
          </div>
        </section>

        {/* ---------------- the audit ---------------- */}
        <section className="bs-section" id="audit">
          <div className="bs-wrap">
            <p className="bs-kicker">Six ways it tries to kill your idea</p>
            <h2>The audit is not a feature. It is the only path to a result.</h2>
            <div className="bs-entries">
              {ENTRIES.map(([n, title, body]) => (
                <article className="bs-entry" key={n}>
                  <div className="n">{n}</div>
                  <h3>{title}</h3>
                  <p>{body}</p>
                </article>
              ))}
            </div>
          </div>
        </section>

        {/* ---------------- terms ---------------- */}
        <section className="bs-section" id="terms">
          <div className="bs-wrap">
            <p className="bs-kicker">Terms</p>
            <h2>Free forever, for the part that matters</h2>
            <p className="intro">
              The audit is never paywalled. Charging you to discover that your
              strategy is broken would invert the point of the product — so the
              honesty is free, and the paid tier is better <em>data</em>, which
              is where the real cost actually sits.
            </p>
            <div className="bs-terms">
              <div className="bs-term">
                <div className="name">Free</div>
                <div className="fee">$0<span> / forever</span></div>
                <p className="blurb">The complete honesty audit, unabridged.</p>
                <ul>
                  <li><span className="mark">✓</span> Every audit check — selection bias, cost sweep, regimes, holdout, sensitivity</li>
                  <li><span className="mark">✓</span> Benchmark on every result</li>
                  <li><span className="mark">✓</span> Plain-English to spec translation</li>
                  <li><span className="mark">✓</span> Shareable Truth Report exports</li>
                  <li><span className="mark">✓</span> Strategy journal and version history</li>
                  <li className="off"><span className="mark">—</span> Current index membership only</li>
                  <li className="off"><span className="mark">—</span> Monthly backtest allowance</li>
                </ul>
                <Link className="bs-btn quiet" href="/">Use it now — no account</Link>
              </div>

              <div className="bs-term featured">
                <div className="name">Pro — planned</div>
                <div className="fee">TBD<span> / month</span></div>
                <p className="blurb">
                  Survivorship-free data and forward tracking. Priced once the
                  data licence is signed, not before.
                </p>
                <ul>
                  <li><span className="mark">✓</span> <strong>Point-in-time index membership</strong>, including companies that went to zero</li>
                  <li><span className="mark">✓</span> Delisted-stock coverage back through 2008</li>
                  <li><span className="mark">✓</span> Forward testing — freeze a survivor, track it live against its backtest</li>
                  <li><span className="mark">✓</span> Unlimited backtests and longer history</li>
                  <li><span className="mark">✓</span> Everything in Free</li>
                </ul>
                <a className="bs-btn" href="#waitlist">Join the waitlist</a>
              </div>
            </div>
            <p className="bs-footnote">
              No pricing is committed, and nothing on this page is for sale. The
              paid tier ships only after a licensed data source and a legal
              review are in place.
            </p>
          </div>
        </section>
      </main>

      {/* ---------------- waitlist ---------------- */}
      <section className="bs-band" id="waitlist">
        <div className="bs-wrap">
          <p className="bs-kicker kicker-light">Validation, in the open</p>
          <h2>Tell me what your last backtest lied to you about</h2>
          <p className="intro">
            I&apos;m testing whether this is worth building for anyone but me.
            Leave an email and, if you like, the strategy you&apos;d test first —
            that answer shapes what gets built next.
          </p>

          {state === "done" ? (
            <p className="bs-said">
              {message}
              <small>No drip sequence, no newsletter — one email when there&apos;s something real to try.</small>
            </p>
          ) : (
            <form className="bs-form" onSubmit={submit}>
              <div className="line">
                <input type="email" required value={email} aria-label="Email address"
                  placeholder="you@example.com"
                  onChange={(e) => setEmail(e.target.value)} />
                <button className="bs-btn" type="submit" disabled={state === "sending"}>
                  {state === "sending" ? "Adding…" : "Join the waitlist"}
                </button>
              </div>
              <textarea value={note}
                aria-label="What did your last backtest lie to you about?"
                placeholder="Optional: what did your last backtest lie to you about?"
                onChange={(e) => setNote(e.target.value)} />
            </form>
          )}
          {state === "error" && (
            <div style={{ marginTop: 16 }}><Notice kind="err">{message}</Notice></div>
          )}
        </div>
      </section>

      <footer className="bs-colophon">
        <div className="bs-wrap">
          <p>{DISCLAIMER}</p>
          <p>
            No trade execution, no signals, no personalised advice — permanently
            off the roadmap. Your strategy text is confidential and is never sent
            to third-party analytics.
          </p>
        </div>
      </footer>
    </>
  );
}
