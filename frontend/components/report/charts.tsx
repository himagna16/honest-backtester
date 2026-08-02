"use client";

// Hand-rolled SVG charts: log-scale equity (net-of-costs primary, gross
// visually subordinate), underwater drawdown, and the regime timeline.
// Every chart ships a table fallback (Gap #12).

import type { RegimeSegment } from "@/lib/types";
import { pct } from "@/lib/format";

const W = 720;
const H = 300;
const PAD = { l: 52, r: 12, t: 12, b: 24 };

function xScale(n: number) {
  return (i: number) => PAD.l + (i / Math.max(n - 1, 1)) * (W - PAD.l - PAD.r);
}

function buildPath(
  values: (number | null)[],
  x: (i: number) => number,
  y: (v: number) => number,
): string {
  let d = "";
  values.forEach((v, i) => {
    if (v === null || Number.isNaN(v)) return;
    d += (d === "" ? "M" : "L") + `${x(i).toFixed(1)},${y(v).toFixed(1)}`;
  });
  return d;
}

function sample<T>(rows: T[], every: number): T[] {
  const out = rows.filter((_, i) => i % every === 0);
  if (rows.length && out[out.length - 1] !== rows[rows.length - 1])
    out.push(rows[rows.length - 1]);
  return out;
}

// ---------------- equity curve ----------------

export function EquityChart({
  points,
  oosStart,
}: {
  points: [string, number | null, number | null, number | null][];
  oosStart: string;
}) {
  if (points.length < 2) return null;
  const dates = points.map((p) => p[0]);
  const all = points.flatMap((p) => [p[1], p[2], p[3]]).filter(
    (v): v is number => v !== null && v > 0,
  );
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  const yLog = (v: number) =>
    PAD.t +
    (1 - (Math.log(v) - Math.log(lo)) / (Math.log(hi) - Math.log(lo) || 1)) *
      (H - PAD.t - PAD.b);
  const x = xScale(points.length);

  // Log-axis ticks at 1-2-5 wealth multiples
  const ticks: number[] = [];
  for (const base of [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]) {
    if (base >= lo * 0.95 && base <= hi * 1.05) ticks.push(base);
  }
  const oosIdx = dates.findIndex((d) => d >= oosStart);

  const netPath = buildPath(points.map((p) => p[1]), x, yLog);
  const grossPath = buildPath(points.map((p) => p[2]), x, yLog);
  const benchPath = buildPath(points.map((p) => p[3]), x, yLog);

  return (
    <figure className="chartbox" role="img"
      aria-label="Equity curve, log scale: strategy net of 5 basis points, gross, and benchmark.">
      <h3>Growth of $1 (log scale)</h3>
      <svg viewBox={`0 0 ${W} ${H}`} aria-hidden="true">
        {oosIdx > 0 && (
          <rect x={x(oosIdx)} y={PAD.t} width={x(points.length - 1) - x(oosIdx)}
            height={H - PAD.t - PAD.b} fill="var(--amber-bg)" opacity="0.7" />
        )}
        {ticks.map((t) => (
          <g key={t}>
            <line x1={PAD.l} x2={W - PAD.r} y1={yLog(t)} y2={yLog(t)}
              stroke="var(--line)" strokeWidth="1" />
            <text x={PAD.l - 6} y={yLog(t) + 4} textAnchor="end" fontSize="11"
              fill="var(--ink-faint)">{t < 1 ? `${t}x` : `${t}x`}</text>
          </g>
        ))}
        <path d={grossPath} fill="none" stroke="var(--ink-faint)"
          strokeWidth="1" strokeDasharray="4 4" />
        <path d={benchPath} fill="none" stroke="var(--accent)" strokeWidth="1.8" />
        <path d={netPath} fill="none" stroke="var(--ink)" strokeWidth="2.4" />
        {oosIdx > 0 && (
          <text x={x(oosIdx) + 4} y={PAD.t + 14} fontSize="11" fill="var(--amber)">
            out-of-sample →
          </text>
        )}
        <text x={PAD.l} y={H - 6} fontSize="11" fill="var(--ink-faint)">{dates[0]}</text>
        <text x={W - PAD.r} y={H - 6} fontSize="11" textAnchor="end"
          fill="var(--ink-faint)">{dates[dates.length - 1]}</text>
      </svg>
      <div className="legend">
        <span><span className="sw" style={{ background: "var(--ink)" }} />Strategy, net of 5 bps (the honest line)</span>
        <span><span className="sw" style={{ background: "var(--ink-faint)", height: 1 }} />Gross (flattering — dashed on purpose)</span>
        <span><span className="sw" style={{ background: "var(--accent)" }} />Benchmark</span>
        <span><span className="sw" style={{ background: "var(--amber-bg)", border: "1px solid var(--amber-line)", height: 10 }} />Holdout (30%)</span>
      </div>
      <details className="fallback">
        <summary>View as table (accessible fallback)</summary>
        <div className="scroll-x">
          <table>
            <thead><tr><th>Date</th><th>Net of 5 bps</th><th>Gross</th><th>Benchmark</th></tr></thead>
            <tbody>
              {sample(points, Math.ceil(points.length / 24)).map((p) => (
                <tr key={p[0]}>
                  <td>{p[0]}</td>
                  <td>{p[1] === null ? "—" : `${p[1].toFixed(2)}x`}</td>
                  <td>{p[2] === null ? "—" : `${p[2].toFixed(2)}x`}</td>
                  <td>{p[3] === null ? "—" : `${p[3].toFixed(2)}x`}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </figure>
  );
}

// ---------------- drawdown ----------------

export function DrawdownChart({
  points,
}: {
  points: [string, number | null, number | null][];
}) {
  if (points.length < 2) return null;
  const dates = points.map((p) => p[0]);
  const values = points.flatMap((p) => [p[1], p[2]]).filter(
    (v): v is number => v !== null,
  );
  const lo = Math.min(...values, -0.02);
  const y = (v: number) => PAD.t + (v / lo) * (H - PAD.t - PAD.b);
  const x = xScale(points.length);
  const stratPath = buildPath(points.map((p) => p[1]), x, y);
  const benchPath = buildPath(points.map((p) => p[2]), x, y);
  const gridTicks = [-0.1, -0.2, -0.3, -0.5, -0.7].filter((t) => t > lo);

  return (
    <figure className="chartbox" role="img"
      aria-label="Drawdown chart: peak-to-trough losses for strategy and benchmark.">
      <h3>Drawdown (underwater plot)</h3>
      <svg viewBox={`0 0 ${W} ${H}`} aria-hidden="true">
        <line x1={PAD.l} x2={W - PAD.r} y1={y(0)} y2={y(0)} stroke="var(--ink-faint)" />
        {gridTicks.map((t) => (
          <g key={t}>
            <line x1={PAD.l} x2={W - PAD.r} y1={y(t)} y2={y(t)} stroke="var(--line)" />
            <text x={PAD.l - 6} y={y(t) + 4} textAnchor="end" fontSize="11"
              fill="var(--ink-faint)">{pct(t, 0)}</text>
          </g>
        ))}
        <path d={benchPath} fill="none" stroke="var(--accent)" strokeWidth="1.6" />
        <path d={stratPath} fill="none" stroke="var(--red)" strokeWidth="2" />
        <text x={PAD.l} y={H - 6} fontSize="11" fill="var(--ink-faint)">{dates[0]}</text>
        <text x={W - PAD.r} y={H - 6} fontSize="11" textAnchor="end"
          fill="var(--ink-faint)">{dates[dates.length - 1]}</text>
      </svg>
      <div className="legend">
        <span><span className="sw" style={{ background: "var(--red)" }} />Strategy (net of 5 bps)</span>
        <span><span className="sw" style={{ background: "var(--accent)" }} />Benchmark</span>
      </div>
      <details className="fallback">
        <summary>View as table (accessible fallback)</summary>
        <div className="scroll-x">
          <table>
            <thead><tr><th>Date</th><th>Strategy</th><th>Benchmark</th></tr></thead>
            <tbody>
              {sample(points, Math.ceil(points.length / 24)).map((p) => (
                <tr key={p[0]}>
                  <td>{p[0]}</td><td>{pct(p[1])}</td><td>{pct(p[2])}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </figure>
  );
}

// ---------------- regime bar ----------------

const KIND_LABEL: Record<string, string> = {
  bull: "bull", bear: "bear market", correction: "correction",
  sideways: "sideways", down: "declining",
};

export function RegimeBar({
  segments,
  hasBear,
}: {
  segments: RegimeSegment[];
  hasBear: boolean;
}) {
  if (!segments.length) return null;
  const t0 = Date.parse(segments[0].start);
  const t1 = Date.parse(segments[segments.length - 1].end);
  const span = Math.max(t1 - t0, 1);

  return (
    <figure className="chartbox">
      <h3>Regime coverage</h3>
      <div className="regimebar" role="img"
        aria-label={`Market regimes across the test window: ${segments
          .map((s) => `${s.label} (${s.start} to ${s.end})`).join("; ")}.`}>
        {segments.map((s, i) => (
          <div key={i} className={`rg-${s.kind}`}
            style={{ width: `${(100 * (Date.parse(s.end) - Date.parse(s.start))) / span}%` }}
            title={`${s.label}: ${s.start} → ${s.end}`} />
        ))}
      </div>
      <div className="legend">
        <span><span className="sw" style={{ background: "var(--green-line)", height: 10 }} />bull</span>
        <span><span className="sw" style={{ background: "var(--amber-line)", height: 10 }} />correction</span>
        <span><span className="sw" style={{ background: "var(--red)", height: 10 }} />bear</span>
        <span><span className="sw" style={{ background: "var(--line)", height: 10 }} />sideways</span>
      </div>
      {!hasBear && (
        <p className="warnbox">
          ▲ This window contains <strong>no sustained bear market</strong> —
          results may not survive one.
        </p>
      )}
      <details className="fallback">
        <summary>View as list (accessible fallback)</summary>
        <ul>
          {segments.map((s, i) => (
            <li key={i}>
              {s.start} → {s.end}: {s.label}
              {KIND_LABEL[s.kind] !== s.label ? ` (${KIND_LABEL[s.kind]})` : ""}
            </li>
          ))}
        </ul>
      </details>
    </figure>
  );
}
