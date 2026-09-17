"use client";

// Public landing page for the validation sprint (brief Section 5, step 2).
// Broadsheet treatment: the founding story is reported, not pitched. Editorial
// rules and whitespace replace bordered cards; the imagery is a self-playing
// replay of a real run plus a real exported Truth Report — nothing mocked up.
//
// Motion is progressive enhancement throughout: the server HTML is complete
// and fully visible; Reveal/FigureCount (components/broadsheet/Reveal.tsx)
// only animate below-the-fold content after hydration, and the hero entrance
// is CSS-only so it can never flash or depend on JS.

import { useState } from "react";
import Link from "next/link";
import Image from "next/image";
import { api } from "@/lib/api";
import { DISCLAIMER } from "@/lib/constants";
import { Notice } from "@/components/ui/Primitives";
import { Reveal, FigureCount } from "@/components/broadsheet/Reveal";
import LiveDemo from "@/components/broadsheet/LiveDemo";

const ENTRIES = [
  ["01", "Selection bias, named out loud",
    "Hand-pick a watchlist and your exact rules are re-run on the S&P 500 as it stood at each rebalance, not on today's winners. Hindsight gets reported, not rewarded."],
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
              <Link href="/workbench">Open the app →</Link>
            </nav>
          </div>
        </div>
      </header>

      <main>
        {/* ---------------- hero ---------------- */}
        <section className="bs-hero">
          <div className="bs-wrap">
            <p className="bs-kicker bs-enter">Educational research tool · Nothing here is advice</p>
            <h1 className="bs-enter" style={{ "--enter-delay": "60ms" } as React.CSSProperties}>
              My backtest said <em>+1650%</em>. I{"\u00a0"}picked the tickers with hindsight.
            </h1>
            <hr className="rule-under bs-enter-rule" style={{ "--enter-delay": "120ms" } as React.CSSProperties} />
            <div className="lede-row">
              <p className="standfirst bs-enter" style={{ "--enter-delay": "120ms" } as React.CSSProperties}>
                Every retail backtester would have shown me that first number and
                none of the corrections. So I built the one that <em>argues back</em>
                {" "}— it runs a full honesty audit on every result, and there is
                no way to turn it off.
              </p>
              <p className="byline bs-enter" style={{ "--enter-delay": "180ms" } as React.CSSProperties}>
                <span className="drop">T</span>he number came from forty-four
                tickers I picked in 2026 and tested on prices from February 2023
                to May 2026. I was buying winners I already knew had won. A
                stricter strategy on the S&amp;P 500, tested on the companies
                actually in the index each month, made 9.6% a year from 2016 to
                2026, behind SPY&apos;s 16%. A dip-buying strategy trailed SPY
                on Sharpe before costs and fell far behind at five basis points
                a day. Three numbers, one afternoon. The audit that caught them
                is now the product.
              </p>
            </div>

            <Reveal as="div" className="bs-figures" variant="rule">
              <Reveal as="div" className="bs-fig a" delay={0}>
                <div className="cap">What my backtest said</div>
                <div className="num"><FigureCount value="+1650%" strike /></div>
                <p>136% a year over 40 months, on 44 tickers picked with hindsight. Max drawdown 45%, against SPY&apos;s 8%.</p>
              </Reveal>
              <Reveal as="div" className="bs-fig b" delay={90}>
                <div className="cap">A stricter strategy, the S&amp;P 500 as it stood</div>
                <div className="num"><FigureCount value="+159%" delay={90} /></div>
                <p>Different rules: momentum plus low volatility, top 25 names, 2016 to 2026, picking only from that month&apos;s index members. SPY returned +362% over the same years.</p>
              </Reveal>
              <Reveal as="div" className="bs-fig c" delay={180}>
                <div className="cap">After trading costs</div>
                <div className="num"><FigureCount value="Lost" delay={420} /></div>
                <p>A dip-buyer on the S&amp;P 500 as it stood each day had a Sharpe of 0.81 before costs, against SPY&apos;s 0.83. At five basis points a day it was up 22% (Sharpe 0.19), against SPY&apos;s 344%.</p>
              </Reveal>
            </Reveal>
          </div>
        </section>

        {/* ---------------- the live demo: the product, replayed ---------------- */}
        <section className="bs-section" id="replay">
          <div className="bs-wrap">
            <Reveal as="p" className="bs-kicker">Watch it argue back</Reveal>
            <Reveal as="h2" delay={60}>One sentence goes in. An argument comes back.</Reveal>
            <LiveDemo />
            <p className="bs-demo-caption">
              A replay of report rep_9534f1bae0, the same report as the export
              further down this page. It is a separate test on the same 44
              tickers, ranked by 6-month return from mid-2024 to May 2026, so
              its numbers differ from the backtest above. Its selection-bias
              check compares against the S&amp;P 500 as it stood each month.
              Every number is from the real report.
            </p>
          </div>
        </section>

        {/* ---------------- the product shot ---------------- */}
        <section className="bs-section">
          <div className="bs-wrap">
            <Reveal as="p" className="bs-kicker">Every run ends here</Reveal>
            <Reveal as="h2" delay={60}>The verdict comes first. The flattering number comes second.</Reveal>
            <Reveal as="p" className="intro" delay={120}>
              Plain English goes in; a precise, editable spec comes back for you
              to approve. Nothing runs until you confirm exactly what will be
              tested — and what comes out is this.
            </Reveal>
            <Reveal as="figure" className="bs-plate">
              <div className="frame">
                <div className="chrome"><i /><i /><i /></div>
                <div className="shot">
                  {/* Below the fold (the demo now precedes it) — lazy-load
                      rather than compete with the hero and fonts.
                      quality=95: this is a page of fine 8pt type, and the
                      default q=75 re-encode visibly softens the glyph edges —
                      the whole point of the shot is that it reads crisply. */}
                  <Image src="/truth-report-sample.png" width={3000} height={3750}
                    quality={95}
                    sizes="(max-width: 760px) 100vw, 1060px"
                    alt="An exported Truth Report: a red verdict reading “Result not trustworthy”, above the strategy's return and Sharpe shown next to the SPY benchmark and the full cost sweep table." />
                </div>
              </div>
              <figcaption>
                A real export from the running application, not a mock-up. The
                benchmark row and the disclaimer are baked into the image file
                itself, so a screenshot on social media carries them too.
              </figcaption>
            </Reveal>
          </div>
        </section>

        {/* ---------------- the audit ---------------- */}
        <section className="bs-section" id="audit">
          <div className="bs-wrap">
            <Reveal as="p" className="bs-kicker">Six ways it tries to kill your idea</Reveal>
            <Reveal as="h2" delay={60}>The audit is not a feature. It is the only path to a result.</Reveal>
            <Reveal as="div" className="bs-entries" variant="rule">
              {ENTRIES.map(([n, title, body], i) => (
                <Reveal as="article" className="bs-entry" key={n} delay={i * 60}>
                  <div className="n">{n}</div>
                  <h3>{title}</h3>
                  <p>{body}</p>
                </Reveal>
              ))}
            </Reveal>
          </div>
        </section>

        {/* ---------------- terms ---------------- */}
        <section className="bs-section" id="terms">
          <div className="bs-wrap">
            <Reveal as="p" className="bs-kicker">Terms</Reveal>
            <Reveal as="h2" delay={60}>Free forever, for the part that matters</Reveal>
            <Reveal as="p" className="intro" delay={120}>
              The audit is never paywalled. Charging you to discover that your
              strategy is broken would invert the point of the product — so the
              honesty is free, and the paid tier is better <em>data</em>, which
              is where the real cost actually sits.
            </Reveal>
            <Reveal as="div" className="bs-terms" variant="rule">
              <Reveal as="div" className="bs-term" delay={0}>
                <div className="name">Free</div>
                <div className="fee">$0<span> / forever</span></div>
                <p className="blurb">The complete honesty audit, unabridged.</p>
                <ul>
                  <li><span className="mark">✓</span> Every audit check — selection bias, cost sweep, regimes, holdout, sensitivity</li>
                  <li><span className="mark">✓</span> Benchmark on every result</li>
                  <li><span className="mark">✓</span> Plain-English to spec translation</li>
                  <li><span className="mark">✓</span> Shareable Truth Report exports</li>
                  <li><span className="mark">✓</span> Strategy journal and version history</li>
                  <li><span className="mark">✓</span> Point-in-time S&amp;P 500 membership from 2015, on free price data</li>
                  <li className="off"><span className="mark">—</span> Monthly backtest allowance</li>
                </ul>
                <Link className="bs-btn quiet" href="/workbench">Use it now — no account</Link>
              </Reveal>

              <Reveal as="div" className="bs-term featured" delay={90}>
                <div className="name">Pro — planned</div>
                <div className="fee">TBD<span> / month</span></div>
                <p className="blurb">
                  Survivorship-free data and forward tracking. Priced once the
                  data licence is signed, not before.
                </p>
                <ul>
                  <li><span className="mark">✓</span> <strong>Point-in-time index membership with every member priced</strong>, including companies that went to zero</li>
                  <li><span className="mark">✓</span> Delisted-stock coverage back through 2008</li>
                  <li><span className="mark">✓</span> Forward testing — freeze a survivor, track it live against its backtest</li>
                  <li><span className="mark">✓</span> Unlimited backtests and longer history</li>
                  <li><span className="mark">✓</span> Everything in Free</li>
                </ul>
                <a className="bs-btn" href="#waitlist">Join the waitlist</a>
              </Reveal>
            </Reveal>
            <Reveal as="p" className="bs-footnote">
              No pricing is committed, and nothing on this page is for sale. The
              paid tier ships only after a licensed data source and a legal
              review are in place.
            </Reveal>
          </div>
        </section>
      </main>

      {/* ---------------- waitlist ---------------- */}
      <section className="bs-band" id="waitlist">
        <div className="bs-wrap">
          <Reveal as="p" className="bs-kicker kicker-light">Validation, in the open</Reveal>
          <Reveal as="h2" delay={60}>Tell me what your last backtest lied to you about</Reveal>
          <Reveal as="p" className="intro" delay={120}>
            I&apos;m testing whether this is worth building for anyone but me.
            Leave an email and, if you like, the strategy you&apos;d test first —
            that answer shapes what gets built next.
          </Reveal>

          {/* One persistent wrapper around the whole done/form swap, so a
              submit can never re-trigger a scroll reveal. */}
          <Reveal as="div" delay={180}>
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
          </Reveal>
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
