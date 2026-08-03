"use client";

// Hand-rolled SVG charts, light analyst theme. Log-scale equity (net-of-costs
// primary, gross visually subordinate), underwater drawdown, regime timeline.
// Every chart ships a table/list fallback and an aria description.

import type { RegimeSegment } from "@/lib/types";
import { pct } from "@/lib/format";
import { IconChart, IconTimeline } from "@/components/ui/Icons";

const W = 760;
const H = 250;
const PAD = { l: 44, r: 10, t: 10, b: 22 };

const xScale = (n: number) => (i: number) =>
  PAD.l + (i / Math.max(n - 1, 1)) * (W - PAD.l - PAD.r);

function buildPath(values: (number | null)[], x: (i: number) => number, y: (v: number) => number) {
  let d = "";
  values.forEach((v, i) => {
    if (v === null || Number.isNaN(v)) return;
    d += (d === "" ? "M" : "L") + `${x(i).toFixed(1)},${y(v).toFixed(1)}`;
  });
  return d;
}

function sample<T>(rows: T[], every: number): T[] {
  const out = rows.filter((_, i) => i % every === 0);
  if (rows.length && out[out.length - 1] !== rows[rows.length - 1]) out.push(rows[rows.length - 1]);
  return out;
}

/* ---------------- equity ---------------- */

export function EquityChart({
  points, oosStart, benchLabel, showGross = true,
}: {
  points: [string, number | null, number | null, number | null][];
  oosStart: string;
  benchLabel: string;
  showGross?: boolean;
}) {
  if (points.length < 2) return null;
  const dates = points.map((p) => p[0]);
  const all = points.flatMap((p) => [p[1], showGross ? p[2] : null, p[3]])
    .filter((v): v is number => v !== null && v > 0);
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  const y = (v: number) =>
    PAD.t + (1 - (Math.log(v) - Math.log(lo)) / (Math.log(hi) - Math.log(lo) || 1)) * (H - PAD.t - PAD.b);
  const x = xScale(points.length);
  const ticks = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]
    .filter((t) => t >= lo * 0.95 && t <= hi * 1.05);
  const oosIdx = dates.findIndex((d) => d >= oosStart);

  return (
    <figure className="chartbox" style={{ margin: 0 }}>
      <header><IconChart size={15} /><h3>Growth of $1 · log scale</h3></header>
      <div className="body">
        <svg viewBox={`0 0 ${W} ${H}`} role="img"
          aria-label={`Equity curve. Strategy net of 5 basis points ends at ${points[points.length - 1][1]?.toFixed(2)} times capital; ${benchLabel} ends at ${points[points.length - 1][3]?.toFixed(2)} times.`}>
          {oosIdx > 0 && (
            <rect x={x(oosIdx)} y={PAD.t} width={x(points.length - 1) - x(oosIdx)}
              height={H - PAD.t - PAD.b} fill="var(--warn-soft)" />
          )}
          {ticks.map((t) => (
            <g key={t}>
              <line x1={PAD.l} x2={W - PAD.r} y1={y(t)} y2={y(t)} stroke="var(--line)" />
              <text x={PAD.l - 7} y={y(t) + 3.5} textAnchor="end" fontSize="10"
                fill="var(--ink-3)">{t}×</text>
            </g>
          ))}
          {showGross && (
            <path d={buildPath(points.map((p) => p[2]), x, y)} fill="none"
              stroke="var(--line-2)" strokeWidth="1" strokeDasharray="3 3" />
          )}
          <path d={buildPath(points.map((p) => p[3]), x, y)} fill="none"
            stroke="var(--ink-3)" strokeWidth="1.3" />
          <path d={buildPath(points.map((p) => p[1]), x, y)} fill="none"
            stroke="var(--ink)" strokeWidth="1.9" strokeLinejoin="round" />
          {oosIdx > 0 && (
            <text x={x(oosIdx) + 5} y={PAD.t + 11} fontSize="10" fill="var(--warn)">
              holdout →
            </text>
          )}
          <text x={PAD.l} y={H - 5} fontSize="10" fill="var(--ink-3)">{dates[0]}</text>
          <text x={W - PAD.r} y={H - 5} fontSize="10" textAnchor="end"
            fill="var(--ink-3)">{dates[dates.length - 1]}</text>
        </svg>
        <div className="legend">
          <span><span className="sw" style={{ background: "var(--ink)", height: 2 }} />Strategy · net 5 bps</span>
          {showGross && <span><span className="sw" style={{ background: "var(--line-2)" }} />Gross</span>}
          <span><span className="sw" style={{ background: "var(--ink-3)" }} />{benchLabel}</span>
          <span><span className="sw" style={{ background: "var(--warn-soft)", height: 9, border: "1px solid var(--warn-line)" }} />Holdout</span>
        </div>
        <details className="fallback">
          <summary>View as table</summary>
          <div className="scroll-x">
            <table>
              <thead><tr><th>Date</th><th>Net 5 bps</th><th>Gross</th><th>Benchmark</th></tr></thead>
              <tbody>
                {sample(points, Math.ceil(points.length / 20)).map((p) => (
                  <tr key={p[0]}>
                    <td>{p[0]}</td>
                    <td>{p[1] === null ? "—" : `${p[1].toFixed(2)}×`}</td>
                    <td>{p[2] === null ? "—" : `${p[2].toFixed(2)}×`}</td>
                    <td>{p[3] === null ? "—" : `${p[3].toFixed(2)}×`}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      </div>
    </figure>
  );
}

/* ---------------- drawdown ---------------- */

export function DrawdownChart({
  points, benchLabel,
}: {
  points: [string, number | null, number | null][];
  benchLabel: string;
}) {
  if (points.length < 2) return null;
  const dates = points.map((p) => p[0]);
  const vals = points.flatMap((p) => [p[1], p[2]]).filter((v): v is number => v !== null);
  const lo = Math.min(...vals, -0.02);
  const h = 190;
  const y = (v: number) => PAD.t + (v / lo) * (h - PAD.t - PAD.b);
  const x = xScale(points.length);
  const grid = [-0.1, -0.2, -0.3, -0.5, -0.7].filter((t) => t > lo);
  const area = buildPath(points.map((p) => p[1]), x, y) +
    `L${x(points.length - 1).toFixed(1)},${y(0).toFixed(1)}L${x(0).toFixed(1)},${y(0).toFixed(1)}Z`;

  return (
    <figure className="chartbox" style={{ margin: 0 }}>
      <header><IconChart size={15} /><h3>Drawdown · underwater</h3></header>
      <div className="body">
        <svg viewBox={`0 0 ${W} ${h}`} role="img"
          aria-label={`Drawdown chart. Strategy worst drawdown ${pct(Math.min(...points.map((p) => p[1] ?? 0)))}.`}>
          {grid.map((t) => (
            <g key={t}>
              <line x1={PAD.l} x2={W - PAD.r} y1={y(t)} y2={y(t)} stroke="var(--line)" />
              <text x={PAD.l - 7} y={y(t) + 3.5} textAnchor="end" fontSize="10"
                fill="var(--ink-3)">{pct(t, 0)}</text>
            </g>
          ))}
          <path d={area} fill="var(--neg-soft)" stroke="none" />
          <path d={buildPath(points.map((p) => p[2]), x, y)} fill="none"
            stroke="var(--ink-3)" strokeWidth="1.2" />
          <path d={buildPath(points.map((p) => p[1]), x, y)} fill="none"
            stroke="var(--neg)" strokeWidth="1.6" />
          <line x1={PAD.l} x2={W - PAD.r} y1={y(0)} y2={y(0)} stroke="var(--line-2)" />
          <text x={PAD.l} y={h - 5} fontSize="10" fill="var(--ink-3)">{dates[0]}</text>
          <text x={W - PAD.r} y={h - 5} fontSize="10" textAnchor="end"
            fill="var(--ink-3)">{dates[dates.length - 1]}</text>
        </svg>
        <div className="legend">
          <span><span className="sw" style={{ background: "var(--neg)" }} />Strategy</span>
          <span><span className="sw" style={{ background: "var(--ink-3)" }} />{benchLabel}</span>
        </div>
        <details className="fallback">
          <summary>View as table</summary>
          <div className="scroll-x">
            <table>
              <thead><tr><th>Date</th><th>Strategy</th><th>Benchmark</th></tr></thead>
              <tbody>
                {sample(points, Math.ceil(points.length / 20)).map((p) => (
                  <tr key={p[0]}><td>{p[0]}</td><td>{pct(p[1])}</td><td>{pct(p[2])}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      </div>
    </figure>
  );
}

/* ---------------- regime timeline ---------------- */

const KIND_LABEL: Record<string, string> = {
  bull: "bull", bear: "bear market", correction: "correction",
  sideways: "sideways", down: "declining",
};

export function RegimeBar({
  segments, hasBear,
}: {
  segments: RegimeSegment[];
  hasBear: boolean;
}) {
  if (!segments.length) return null;
  const t0 = Date.parse(segments[0].start);
  const t1 = Date.parse(segments[segments.length - 1].end);
  const span = Math.max(t1 - t0, 1);
  return (
    <figure className="chartbox" style={{ margin: 0 }}>
      <header><IconTimeline size={15} /><h3>Market periods covered</h3></header>
      <div className="body">
        <div className="regimebar" role="img"
          aria-label={`Regime timeline: ${segments.map((s) => `${s.label} from ${s.start} to ${s.end}`).join("; ")}.`}>
          {segments.map((s, i) => (
            <div key={i} className={`rg-${s.kind}`}
              style={{ width: `${(100 * (Date.parse(s.end) - Date.parse(s.start))) / span}%` }}
              title={`${s.label}: ${s.start} → ${s.end}`} />
          ))}
        </div>
        <div className="timeline-axis">
          <span>{segments[0].start}</span>
          <span>{segments[segments.length - 1].end}</span>
        </div>
        <div className="legend">
          <span><span className="sw" style={{ background: "#cddbd0", height: 9 }} />bull</span>
          <span><span className="sw" style={{ background: "#e8dcb8", height: 9 }} />correction</span>
          <span><span className="sw" style={{ background: "#e0c2c2", height: 9 }} />bear</span>
          <span><span className="sw" style={{ background: "var(--surface-3)", height: 9 }} />sideways</span>
        </div>
        {!hasBear && (
          <div className="notice notice-warn" style={{ marginTop: 12 }}>
            <span>
              This window contains no sustained bear market — results may not
              survive one.
            </span>
          </div>
        )}
        <details className="fallback">
          <summary>View as list</summary>
          <ul style={{ margin: "8px 0 0", paddingLeft: 18, fontSize: 12, color: "var(--ink-2)" }}>
            {segments.map((s, i) => (
              <li key={i}>
                {s.start} → {s.end}: {s.label}
                {KIND_LABEL[s.kind] !== s.label ? ` (${KIND_LABEL[s.kind]})` : ""}
              </li>
            ))}
          </ul>
        </details>
      </div>
    </figure>
  );
}
