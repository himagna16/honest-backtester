"use client";

// Truth Report building blocks: verdict banner, stat row, audit cards,
// cost-sweep table, OOS panel. Severity is always icon + text label,
// never color alone (Gap #12).

import { useState } from "react";
import type {
  AuditFinding, CostSweepRow, Metrics, Report, Verdict,
} from "@/lib/types";
import { num, pct, SEVERITY_META, times } from "@/lib/format";

export function VerdictBanner({ verdict }: { verdict: Verdict }) {
  const label =
    verdict.color === "red" ? "Killed" :
    verdict.color === "amber" ? "Wounded" : "Survived";
  return (
    <section className={`verdict verdict-${verdict.color}`} role="status">
      <div className="badge">
        <span aria-hidden="true" style={{ fontSize: "1rem" }}>{verdict.icon}</span>
        <span>{label} — {verdict.color} verdict</span>
      </div>
      <div className="headline">{verdict.headline}</div>
      <div className="sub">{verdict.sub}</div>
    </section>
  );
}

export function StatRow({
  metrics, benchmark, benchLabel,
}: {
  metrics: Metrics; benchmark: Metrics; benchLabel: string;
}) {
  const rows: {
    k: string; v: string; b: string; better: boolean | null;
  }[] = [
    { k: "Total return (net 5 bps)", v: pct(metrics.total_return),
      b: pct(benchmark.total_return),
      better: metrics.total_return > benchmark.total_return },
    { k: "CAGR", v: pct(metrics.cagr), b: pct(benchmark.cagr),
      better: metrics.cagr > benchmark.cagr },
    { k: "Sharpe", v: num(metrics.sharpe), b: num(benchmark.sharpe),
      better: metrics.sharpe > benchmark.sharpe },
    { k: "Max drawdown", v: pct(metrics.max_drawdown),
      b: pct(benchmark.max_drawdown),
      better: metrics.max_drawdown > benchmark.max_drawdown },
  ];
  return (
    <div className="statgrid">
      {rows.map((r) => (
        <div className="stat" key={r.k}>
          <div className="k">{r.k}</div>
          <div className={`v ${r.better === null ? "" : r.better ? "better" : "worse"}`}>
            {r.v}
          </div>
          <div className="b">{benchLabel}: {r.b}</div>
        </div>
      ))}
    </div>
  );
}

export function AuditCard({
  finding, onAction,
}: {
  finding: AuditFinding; onAction?: (finding: AuditFinding) => void;
}) {
  const meta = SEVERITY_META[finding.severity];
  const [open, setOpen] = useState(finding.severity !== "pass");
  return (
    <article className={`finding ${meta.cls}`}>
      <div className="head" style={{ cursor: finding.severity === "pass" ? "pointer" : "default" }}
        onClick={() => finding.severity === "pass" && setOpen(!open)}>
        <span className="sev">
          <span aria-hidden="true">{meta.icon}</span> {meta.label}
        </span>
        <span className="title">{finding.name}</span>
        {finding.severity === "pass" && (
          <span className="faint">{open ? "▾" : "▸"}</span>
        )}
      </div>
      {open && <p className="msg">{finding.message}</p>}
      {open && finding.action && onAction && (
        <div className="action">
          <button onClick={() => onAction(finding)}>{finding.action}</button>
        </div>
      )}
      {open && finding.explainer && (
        <div className="action">
          <a href={`/glossary#${finding.explainer}`} className="faint">
            Why this matters →
          </a>
        </div>
      )}
    </article>
  );
}

export function AuditList({
  findings, onAction,
}: {
  findings: AuditFinding[]; onAction?: (finding: AuditFinding) => void;
}) {
  const active = findings.filter((f) => f.severity !== "pass");
  const passes = findings.filter((f) => f.severity === "pass");
  return (
    <section>
      <h2>Audit findings</h2>
      {active.length === 0 && (
        <p className="muted">No blockers or warnings — read the passes below anyway.</p>
      )}
      {active.map((f) => (
        <AuditCard key={f.check + f.name} finding={f} onAction={onAction} />
      ))}
      {passes.length > 0 && (
        <>
          <h3 className="muted">{passes.length} checks passed (collapsed)</h3>
          {passes.map((f) => (
            <AuditCard key={f.check + f.name} finding={f} onAction={onAction} />
          ))}
        </>
      )}
    </section>
  );
}

export function CostSweepTable({ rows }: { rows: CostSweepRow[] }) {
  // Highlight the cell where beating flips to losing
  let flipAt: number | null = null;
  let prevBeats: boolean | null = null;
  for (const r of rows) {
    if (r.bps === null) continue;
    if (prevBeats === true && r.beats_benchmark === false) flipAt = r.bps;
    prevBeats = r.beats_benchmark;
  }
  return (
    <section>
      <h2>Cost sweep</h2>
      <div className="scroll-x">
        <table>
          <thead>
            <tr>
              <th>Costs</th><th>Total return</th><th>CAGR</th>
              <th>Sharpe</th><th>Max DD</th><th>vs benchmark</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => {
              const isBench = r.bps === null;
              const flip = r.bps !== null && flipAt !== null && r.bps >= flipAt;
              return (
                <tr key={i} className={isBench ? "bench-row" : ""}>
                  <td>{isBench ? r.label : `${r.bps} bps`}</td>
                  <td className={flip ? "flip" : ""}>{pct(r.total_return)}</td>
                  <td>{pct(r.cagr)}</td>
                  <td className={flip ? "flip" : ""}>{num(r.sharpe)}</td>
                  <td>{pct(r.max_drawdown)}</td>
                  <td className={r.beats_benchmark ? "beats" : flip ? "flip" : ""}>
                    {isBench ? "—" : r.beats_benchmark ? "✓ beats" : "✕ loses"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="faint">
        Costs scale with measured turnover. 5 bps is optimistic for retail;
        20 bps is realistic for frequent trading.
        {flipAt !== null && ` Break-even sits below ${flipAt} bps — highlighted cells are where the edge dies.`}
      </p>
    </section>
  );
}

export function OOSPanel({ report }: { report: Report }) {
  const { oos, multiple_testing, sensitivity } = report;
  const seg = (label: string, s: typeof oos.in_sample) => (
    <div className="card" style={{ flex: 1, minWidth: 240 }}>
      <h3>{label}</h3>
      <p className="faint">{s.start} → {s.end}</p>
      <table>
        <tbody>
          <tr><td>Total (net 5 bps)</td><td>{pct(s.total_return)}</td></tr>
          <tr><td>Benchmark</td><td>{pct(s.benchmark_total_return)}</td></tr>
          <tr><td>Sharpe</td><td>{num(s.sharpe)} vs {num(s.benchmark_sharpe)}</td></tr>
          <tr>
            <td>Verdict input</td>
            <td>{s.beats_benchmark ? "✓ beats" : "✕ trails"}</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
  return (
    <section>
      <h2>Out-of-sample &amp; robustness</h2>
      <div style={{ display: "flex", gap: "0.8rem", flexWrap: "wrap" }}>
        {seg("Train (first 70%)", oos.in_sample)}
        {seg("Holdout (last 30%) — the verdict uses this", oos.out_of_sample)}
      </div>
      {sensitivity.length > 0 && (
        <>
          <h3>Parameter sensitivity (±1 step)</h3>
          <div className="scroll-x">
            <table>
              <thead><tr><th>Variant</th><th>Total (net 5 bps)</th><th>vs benchmark</th></tr></thead>
              <tbody>
                {sensitivity.map((s) => (
                  <tr key={s.variant}>
                    <td>{s.variant}</td>
                    <td>{pct(s.net5_total_return)}</td>
                    <td>{s.beats_benchmark ? "✓ beats" : "✕ loses"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
      <p className="warnbox">
        Multiple-testing counter: this is run <strong>#{multiple_testing.attempts}</strong> against
        this universe. With enough tries, something beats the benchmark by luck.
      </p>
      {typeof report.metrics.annual_turnover === "number" && (
        <p className="faint">Annualized turnover: {times(report.metrics.annual_turnover)}</p>
      )}
    </section>
  );
}
