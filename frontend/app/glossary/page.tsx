"use client";

// The audit is a curriculum (Section 1): every finding links here. Metric
// formulas match backend/engine/metrics.py; the conventions table is served
// from the single source of truth (backend/dsl/conventions.py).

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { PageDisclaimer, TopBar } from "@/components/Shell";
import { Panel } from "@/components/ui/Primitives";

const METRICS: { name: string; formula: string; gloss: string }[] = [
  { name: "CAGR", formula: "(1 + total_return)^(252/days) − 1",
    gloss: "Smoothed annual growth rate — what the ride averaged, not what it felt like." },
  { name: "Sharpe ratio", formula: "mean(daily − rf) / std(daily) × √252",
    gloss: "Return per unit of pain. The risk-free rate comes from the 13-week T-bill (^IRX); if unavailable we use 0 and say so." },
  { name: "Max drawdown", formula: "min(equity / peak − 1), peak includes your initial capital",
    gloss: "Worst peak-to-trough loss. Losing 30% right after funding counts — the peak includes day 0." },
  { name: "Turnover", formula: "0.5 × Σ|w_target − w_drifted| per rebalance, annualized",
    gloss: "How much you trade. It is the multiplier on every cost assumption, which is why the cost sweep scales with it." },
  { name: "Hit rate", formula: "share of calendar months beating the benchmark",
    gloss: "Consistency, not magnitude — one lucky year can hide in a big total return but not here." },
];

const EXPLAINERS: { id: string; title: string; body: string }[] = [
  {
    id: "selection-bias",
    title: "Selection bias (the +1650% trap)",
    body: "If you pick tickers you already know did well, any backtest on them is partly a memory of that knowledge. In 2016 you couldn't have known which 44 names would win — picking them in 2026 and testing from 2016 is hindsight wearing a lab coat. This tool re-runs your exact rules on the full S&P 500 and compares; a ranked strategy on a hand-picked list can never earn a green verdict.",
  },
  {
    id: "survivorship-bias",
    title: "Survivorship bias (index lists have it too)",
    body: "A CURRENT S&P 500 member list only contains companies that survived and grew into the index. Backtesting the past on today's members means you secretly bought the winners before they won. It's smaller than a hand-picked list's bias, but real — honest index backtests need the membership as of each rebalance date, which requires licensed point-in-time data (V1).",
  },
  {
    id: "lookahead-bias",
    title: "Lookahead bias",
    body: "Using information from after the trade to decide the trade. Here it is impossible by construction: signals at day t use closes through t only, positions earn returns from t+1, and the shift lives inside the engine — no spec can ask for a same-day fill. The test suite proves that altering future prices cannot change past weights.",
  },
  {
    id: "costs",
    title: "Transaction costs (the buy-the-dip killer)",
    body: "Every trade pays a spread and slippage. Costs scale with turnover, so strategies that churn die first: the original session's dip-buyer beat SPY gross and lost after costs. 5 bps per unit of turnover is optimistic for retail; 20 bps is realistic for frequent trading. The verdict always uses costed numbers.",
  },
  {
    id: "regime-coverage",
    title: "Regime coverage",
    body: "A strategy tested only in a bull market has never been tested. If the window contains no ≥15% benchmark drawdown, the report says so — strategies that look great in bull markets often die in their first real drawdown.",
  },
  {
    id: "overfitting",
    title: "Overfitting (the in-sample mirage)",
    body: "Tuning until the past looks great fits the noise, not the signal. The last 30% of the window is held out; performing far better on the 70% you could see is the classic overfit signature. The verdict privileges the holdout segment.",
  },
  {
    id: "parameter-sensitivity",
    title: "Parameter sensitivity",
    body: "If top-5 beats the benchmark but top-6 doesn't, you didn't find an edge — you found a coincidence with a parameter attached. Every numeric knob is re-run one step in each direction; flips are flagged.",
  },
  {
    id: "multiple-testing",
    title: "Multiple testing",
    body: "Test enough variants against the same data and something will 'work' by luck. The per-universe counter never resets; after N attempts, a Bonferroni-adjusted significance bar is roughly N× higher. It never blocks — it informs how much to trust a winner.",
  },
  {
    id: "benchmark-fit",
    title: "Benchmark appropriateness",
    body: "A bond-heavy portfolio judged against pure stocks looks bad in bulls and fake-good in bears. If the strategy's realized beta to its benchmark is far from 1, the report suggests re-running against 60/40 for a fair yardstick.",
  },
  {
    id: "capacity",
    title: "Capacity & realism",
    body: "Weights are theory; shares are reality. At a $10k account, a 5% position in a $600,000-per-share stock is untradeable without fractional shares. The audit checks recent positions against actual prices.",
  },
  {
    id: "missing-data",
    title: "Missing data & exclusions",
    body: "Silently dropping tickers that lack data is survivorship bias through the back door — the names that vanish are disproportionately the losers. Exclusions, late joiners, shortened windows, and price gaps are always disclosed as findings. Signals require a complete gap-free lookback.",
  },
  {
    id: "data-freshness",
    title: "Data freshness & snapshots",
    body: "yfinance restates adjusted prices over time, so re-running 'the same' backtest next month can give different numbers. Every run freezes its exact price panel with a content hash; re-runs replay the snapshot byte-for-byte by default. 'Refresh data' is the only path to new numbers.",
  },
  {
    id: "sharpe",
    title: "Risk-free rate in Sharpe",
    body: "Sharpe measures excess return over cash. When the 13-week T-bill series is unavailable the tool uses 0% and says so — which makes Sharpe look slightly better than reality, hence the disclosure.",
  },
];

interface ConventionRow { convention: string; rule: string }

export default function Glossary() {
  const [conventions, setConventions] = useState<ConventionRow[]>([]);

  useEffect(() => {
    api<{ conventions: ConventionRow[] }>("/api/conventions")
      .then((r) => setConventions(r.conventions))
      .catch(() => setConventions([]));
  }, []);

  return (
    <>
      <TopBar crumb="Glossary" />
      <main className="content" style={{ maxWidth: 900 }}>
      <p className="muted" style={{ marginTop: 0 }}>
        The audit is a curriculum: every finding on a Truth Report links to an
        entry here.
      </p>

      <h2>Metrics</h2>
      {METRICS.map((m) => (
        <div className="gloss-entry" key={m.name}>
          <h3 style={{ marginTop: 0 }}>{m.name}</h3>
          <p className="mono formula">{m.formula}</p>
          <p className="muted">{m.gloss}</p>
        </div>
      ))}

      <h2>The biases the audit hunts</h2>
      {EXPLAINERS.map((e) => (
        <div className="gloss-entry" key={e.id} id={e.id}>
          <h3 style={{ marginTop: 0 }}>{e.title}</h3>
          <p className="muted">{e.body}</p>
        </div>
      ))}

      <h2 id="conventions">Day-count &amp; calendar conventions</h2>
      <p className="faint">
        One table, one source of truth — the engine, the LLM translator, and
        this page all render from <code>backend/dsl/conventions.py</code>.
      </p>
      <div className="scroll-x">
        <table>
          <thead><tr><th>Convention</th><th>Rule</th></tr></thead>
          <tbody>
            {conventions.map((c) => (
              <tr key={c.convention}>
                <td style={{ whiteSpace: "nowrap", verticalAlign: "top" }}>
                  {c.convention}
                </td>
                <td style={{ textAlign: "left" }}>{c.rule}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <PageDisclaimer />
      </main>
    </>
  );
}
