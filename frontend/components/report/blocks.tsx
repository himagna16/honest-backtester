"use client";

// Truth Report building blocks. Severity and verdict are always icon + text,
// never colour alone.

import { useState } from "react";
import type { AuditFinding, CostSweepRow, Metrics, Report, Verdict } from "@/lib/types";
import { num, pct } from "@/lib/format";
import {
  evidenceStrength, noteBuckets, riskBadges, stressCells,
} from "@/lib/insights";
import { Confidence, Panel } from "@/components/ui/Primitives";
import {
  IconAudit, IconCheck, IconInfo, IconRisk, IconTimeline, IconWarning, IconX,
} from "@/components/ui/Icons";

/* ---------------- sparkline ---------------- */

export function Sparkline({
  values, tone = "accent", height = 26,
}: {
  values: number[];
  tone?: "accent" | "pos" | "neg";
  height?: number;
}) {
  const clean = values.filter((v) => typeof v === "number" && !Number.isNaN(v));
  if (clean.length < 2) return null;
  const w = 120;
  const lo = Math.min(...clean);
  const hi = Math.max(...clean);
  const span = hi - lo || 1;
  const step = w / (clean.length - 1);
  const d = clean
    .map((v, i) => `${i === 0 ? "M" : "L"}${(i * step).toFixed(1)},${(height - ((v - lo) / span) * (height - 2) - 1).toFixed(1)}`)
    .join("");
  const stroke =
    tone === "pos" ? "var(--pos)" : tone === "neg" ? "var(--neg)" : "var(--accent)";
  return (
    <svg className="spark" viewBox={`0 0 ${w} ${height}`} preserveAspectRatio="none"
      aria-hidden="true">
      <path d={d} fill="none" stroke={stroke} strokeWidth="1.4"
        vectorEffect="non-scaling-stroke" strokeLinejoin="round" />
    </svg>
  );
}

/* ---------------- verdict ---------------- */

const VERDICT_META = {
  red: { label: "Killed", Icon: IconX },
  amber: { label: "Wounded", Icon: IconWarning },
  green: { label: "Survived", Icon: IconCheck },
} as const;

export function VerdictBanner({ verdict, report }: { verdict: Verdict; report: Report }) {
  const meta = VERDICT_META[verdict.color];
  const ev = evidenceStrength(report);
  return (
    <section className={`verdict verdict-${verdict.color}`} role="status">
      <div className="badge">
        <meta.Icon size={13} />
        {meta.label} — {verdict.color} verdict
      </div>
      <div className="headline">{verdict.headline}</div>
      <div className="sub">{verdict.sub}</div>
      <div style={{ marginTop: 12 }}>
        <Confidence level={ev.level} label={ev.label} />
      </div>
    </section>
  );
}

/* ---------------- risk badges ---------------- */

export function RiskBadges({ report }: { report: Report }) {
  const badges = riskBadges(report);
  if (badges.length === 0) return null;
  return (
    <div className="badges">
      {badges.map((b) => (
        <span key={b.label} className={`badge-risk ${b.tone}`} title={b.why}>
          <IconRisk size={11} />
          {b.label}
        </span>
      ))}
    </div>
  );
}

/* ---------------- metric cards ---------------- */

export function MetricCards({
  metrics, benchmark, benchLabel, equity, drawdown,
}: {
  metrics: Metrics;
  benchmark: Metrics;
  benchLabel: string;
  equity: number[];
  drawdown: number[];
}) {
  const cards = [
    {
      k: "Total return · net 5 bps", v: pct(metrics.total_return),
      b: `${benchLabel}: ${pct(benchmark.total_return)}`,
      better: metrics.total_return > benchmark.total_return,
      spark: equity, tone: "accent" as const,
    },
    {
      k: "CAGR", v: pct(metrics.cagr), b: `${benchLabel}: ${pct(benchmark.cagr)}`,
      better: metrics.cagr > benchmark.cagr, spark: equity, tone: "accent" as const,
    },
    {
      k: "Sharpe", v: num(metrics.sharpe), b: `${benchLabel}: ${num(benchmark.sharpe)}`,
      better: metrics.sharpe > benchmark.sharpe, spark: null, tone: "accent" as const,
    },
    {
      k: "Max drawdown", v: pct(metrics.max_drawdown),
      b: `${benchLabel}: ${pct(benchmark.max_drawdown)}`,
      better: metrics.max_drawdown > benchmark.max_drawdown,
      spark: drawdown, tone: "neg" as const,
    },
    {
      k: "Annual turnover",
      v: metrics.annual_turnover != null ? `${metrics.annual_turnover.toFixed(1)}×` : "—",
      b: "drives every cost column", better: null, spark: null, tone: "accent" as const,
    },
  ];
  return (
    <div className="metrics">
      {cards.map((c) => (
        <div className="metric" key={c.k}>
          <span className="k">{c.k}</span>
          <span className={`metric-value ${c.better === null ? "" : c.better ? "pos" : "neg"}`}>
            {c.v}
          </span>
          <span className="b">{c.b}</span>
          {c.spark && <Sparkline values={c.spark} tone={c.tone} />}
        </div>
      ))}
    </div>
  );
}

/* ---------------- stress test ---------------- */

export function StressPanel({ report }: { report: Report }) {
  const cells = stressCells(report);
  return (
    <Panel title="Stress test" icon={<IconRisk size={15} />} tight>
      <div className="stress">
        {cells.map((c) => (
          <div key={c.k} className={`cell ${c.state}`}>
            <div className="k">{c.k}</div>
            <div className="v">
              {c.state === "survived" ? <IconCheck size={12} />
                : c.state === "failed" ? <IconX size={12} />
                : <IconInfo size={12} />}
              {c.v}
            </div>
          </div>
        ))}
      </div>
      <p className="faint" style={{ marginTop: 10 }}>
        Each cell is a real column from this run — cost levels, the holdout
        segment, regime coverage, and one-step parameter variants.
      </p>
    </Panel>
  );
}

/* ---------------- research note ---------------- */

export function ResearchNote({
  report, onAction,
}: {
  report: Report;
  onAction?: (check: string) => void;
}) {
  const b = noteBuckets(report);
  return (
    <section className="note">
      <div className="note-head">
        <div className="kicker">Audit note · deterministic, not model-generated</div>
        <h2>What the gauntlet found</h2>
      </div>
      <div className="note-body">
        <div className="insights">
          <article className="insight worked">
            <h4><IconCheck size={12} /> What worked</h4>
            <ul>{b.worked.map((t, i) => <li key={i}>{t}</li>)}</ul>
          </article>
          <article className="insight failed">
            <h4><IconX size={12} /> What failed</h4>
            <ul>{b.failed.map((t, i) => <li key={i}>{t}</li>)}</ul>
          </article>
          <article className="insight risks">
            <h4><IconWarning size={12} /> Main risks</h4>
            <ul>{b.risks.map((t, i) => <li key={i}>{t}</li>)}</ul>
          </article>
          <article className="insight next">
            <h4><IconAudit size={12} /> Suggested next tests</h4>
            <ul>
              {b.next.map((n, i) => (
                <li key={i}>
                  {n.check && onAction ? (
                    <button className="ghost" style={{ padding: 0, minHeight: 0, color: "var(--accent)" }}
                      onClick={() => onAction(n.check)}>
                      {n.text}
                    </button>
                  ) : n.text}
                </li>
              ))}
            </ul>
          </article>
        </div>
      </div>
    </section>
  );
}

/* ---------------- audit findings ---------------- */

const SEV_META: Record<string, { label: string; Icon: typeof IconCheck }> = {
  fatal: { label: "BLOCKER", Icon: IconX },
  warning: { label: "WARNING", Icon: IconWarning },
  note: { label: "NOTE", Icon: IconInfo },
  pass: { label: "PASS", Icon: IconCheck },
};

export function AuditCard({
  finding, onAction,
}: {
  finding: AuditFinding;
  onAction?: (finding: AuditFinding) => void;
}) {
  const meta = SEV_META[finding.severity];
  const [open, setOpen] = useState(finding.severity !== "pass");
  const collapsible = finding.severity === "pass";
  return (
    <article className={`finding sev-${finding.severity}`}>
      <div className="head">
        <span className="sev"><meta.Icon size={11} /> {meta.label}</span>
        {collapsible ? (
          <button className="ghost" style={{ padding: 0, minHeight: 0, font: "inherit" }}
            aria-expanded={open} onClick={() => setOpen(!open)}>
            <span className="title">{finding.name}</span>
          </button>
        ) : (
          <span className="title">{finding.name}</span>
        )}
      </div>
      {open && <p className="msg">{finding.message}</p>}
      {open && (finding.action || finding.explainer) && (
        <div className="action">
          {finding.action && onAction && (
            <button onClick={() => onAction(finding)}>{finding.action}</button>
          )}
          {finding.explainer && (
            <a className="faint" href={`/glossary#${finding.explainer}`}>Why this matters →</a>
          )}
        </div>
      )}
    </article>
  );
}

export function AuditList({
  findings, onAction,
}: {
  findings: AuditFinding[];
  onAction?: (finding: AuditFinding) => void;
}) {
  const active = findings.filter((f) => f.severity !== "pass");
  const passes = findings.filter((f) => f.severity === "pass");
  return (
    <div>
      {active.map((f) => <AuditCard key={f.check + f.name} finding={f} onAction={onAction} />)}
      {passes.length > 0 && (
        <>
          <p className="eyebrow" style={{ margin: "20px 0 8px" }}>
            {passes.length} checks passed
          </p>
          {passes.map((f) => <AuditCard key={f.check + f.name} finding={f} onAction={onAction} />)}
        </>
      )}
    </div>
  );
}

/* ---------------- cost sweep ---------------- */

export function CostSweepTable({ rows }: { rows: CostSweepRow[] }) {
  let flipAt: number | null = null;
  let prev: boolean | null = null;
  for (const r of rows) {
    if (r.bps === null) continue;
    if (prev === true && r.beats_benchmark === false) flipAt = r.bps;
    prev = r.beats_benchmark;
  }
  return (
    <Panel title="Cost sweep" icon={<IconTimeline size={15} />} tight>
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
                    {isBench ? "—" : r.beats_benchmark ? "beats" : "loses"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="faint" style={{ marginTop: 10 }}>
        Costs scale with measured turnover. 5 bps is optimistic for retail; 20 bps
        is realistic for frequent trading.
        {flipAt !== null && ` Break-even sits below ${flipAt} bps.`}
      </p>
    </Panel>
  );
}

/* ---------------- out-of-sample ---------------- */

export function OOSPanel({ report }: { report: Report }) {
  const { oos, sensitivity } = report;
  const seg = (label: string, s: typeof oos.in_sample, emphasis?: boolean) => (
    <div style={{
      flex: 1, minWidth: 220, border: "1px solid var(--line)",
      borderRadius: "var(--r-sm)", padding: "12px 14px",
      background: emphasis ? "var(--accent-soft)" : "var(--surface)",
      borderColor: emphasis ? "var(--accent-line)" : "var(--line)",
    }}>
      <div className="eyebrow">{label}</div>
      <div className="faint" style={{ marginBottom: 8 }}>{s.start} → {s.end}</div>
      <table>
        <tbody>
          <tr><td>Total · net 5 bps</td><td>{pct(s.total_return)}</td></tr>
          <tr><td>Benchmark</td><td>{pct(s.benchmark_total_return)}</td></tr>
          <tr><td>Sharpe</td><td>{num(s.sharpe)} vs {num(s.benchmark_sharpe)}</td></tr>
          <tr>
            <td>Outcome</td>
            <td className={s.beats_benchmark ? "beats" : "flip"}>
              {s.beats_benchmark ? "beats" : "trails"}
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  );
  return (
    <div className="stack">
      <Panel title="Train vs holdout" icon={<IconTimeline size={15} />} tight>
        <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
          {seg("Train · first 70%", oos.in_sample)}
          {seg("Holdout · last 30% (verdict uses this)", oos.out_of_sample, true)}
        </div>
      </Panel>
      {sensitivity.length > 0 && (
        <Panel title="Parameter sensitivity" icon={<IconRisk size={15} />} tight>
          <div className="scroll-x">
            <table>
              <thead>
                <tr><th>Variant</th><th>Total · net 5 bps</th><th>vs benchmark</th></tr>
              </thead>
              <tbody>
                {sensitivity.map((s) => (
                  <tr key={s.variant}>
                    <td>{s.variant}</td>
                    <td>{pct(s.net5_total_return)}</td>
                    <td className={s.beats_benchmark ? "beats" : "flip"}>
                      {s.beats_benchmark ? "beats" : "loses"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      )}
    </div>
  );
}

/* ---------------- holdings log ---------------- */

export function HoldingsLog({ report }: { report: Report }) {
  const rows = report.holdings_recent ?? [];
  if (rows.length === 0) return null;
  return (
    <Panel title="Rebalance log" icon={<IconAudit size={15} />} tight>
      <div className="scroll-x" style={{ maxHeight: 420, overflowY: "auto" }}>
        <table>
          <thead>
            <tr><th>Signal date</th><th>Eligible</th><th>Held</th><th style={{ textAlign: "left" }}>Positions</th></tr>
          </thead>
          <tbody>
            {rows.map((h) => (
              <tr key={h.date}>
                <td>{h.date}</td>
                <td>{h.eligible}</td>
                <td>{h.picks.length}</td>
                <td style={{ textAlign: "left", whiteSpace: "normal" }}>
                  {h.picks.map((p) => (
                    <span key={p.ticker} style={{ marginRight: 10 }}>
                      <span className="mono">{p.ticker}</span>{" "}
                      <span className="faint">{(p.weight * 100).toFixed(1)}%</span>
                    </span>
                  ))}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="faint" style={{ marginTop: 10 }}>
        Last {rows.length} rebalances. Weights are targets at the signal date;
        positions drift between rebalances.
      </p>
    </Panel>
  );
}
