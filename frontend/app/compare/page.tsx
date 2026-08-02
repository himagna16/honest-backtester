"use client";

// Compare-runs view: spec diff + side-by-side stats for two Truth Reports.

import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { api } from "@/lib/api";
import type { ReportEnvelope } from "@/lib/types";
import { num, pct } from "@/lib/format";

function flatten(obj: unknown, prefix = ""): Record<string, string> {
  const out: Record<string, string> = {};
  if (obj === null || typeof obj !== "object") {
    out[prefix] = JSON.stringify(obj);
    return out;
  }
  for (const [key, value] of Object.entries(obj as Record<string, unknown>)) {
    const path = prefix ? `${prefix}.${key}` : key;
    if (value !== null && typeof value === "object" && !Array.isArray(value)) {
      Object.assign(out, flatten(value, path));
    } else {
      out[path] = JSON.stringify(value);
    }
  }
  return out;
}

function CompareInner() {
  const searchParams = useSearchParams();
  const a = searchParams.get("a");
  const b = searchParams.get("b");
  const [reports, setReports] = useState<[ReportEnvelope, ReportEnvelope] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!a || !b) { setError("pick two runs to compare"); return; }
    Promise.all([
      api<ReportEnvelope>(`/api/report/${a}`),
      api<ReportEnvelope>(`/api/report/${b}`),
    ]).then((pair) => setReports(pair as [ReportEnvelope, ReportEnvelope]))
      .catch((e) => setError(String(e.message ?? e)));
  }, [a, b]);

  if (error) return <p className="errbox">✕ {error}</p>;
  if (!reports) return <p className="faint">Loading comparison…</p>;
  const [ra, rb] = reports;

  const specA = flatten(ra.report.spec);
  const specB = flatten(rb.report.spec);
  const keys = [...new Set([...Object.keys(specA), ...Object.keys(specB)])].sort();
  const diffs = keys.filter((k) => specA[k] !== specB[k]);

  const statRows: [string, (r: ReportEnvelope) => string][] = [
    ["Verdict", (r) => `${r.report.verdict.icon} ${r.report.verdict.headline}`],
    ["Total return (net 5 bps)", (r) => pct(r.report.metrics.total_return)],
    ["CAGR", (r) => pct(r.report.metrics.cagr)],
    ["Sharpe", (r) => num(r.report.metrics.sharpe)],
    ["Max drawdown", (r) => pct(r.report.metrics.max_drawdown)],
    ["Benchmark total", (r) => pct(r.report.benchmark.total_return)],
    ["Holdout beats benchmark",
      (r) => (r.report.oos.out_of_sample.beats_benchmark ? "yes" : "no")],
    ["Annual turnover", (r) =>
      `${(r.report.metrics.annual_turnover ?? 0).toFixed(1)}×`],
    ["Data as of", (r) => r.report.data.as_of],
    ["Snapshot", (r) => r.report.run?.snapshot_hash?.slice(0, 12) ?? "—"],
  ];

  return (
    <div>
      <h1>Compare runs</h1>
      <div className="scroll-x">
        <table>
          <thead>
            <tr>
              <th></th>
              <th><a href={`/report/${ra.id}`}>{ra.id}</a><br />
                <span className="faint">{ra.created_at}</span></th>
              <th><a href={`/report/${rb.id}`}>{rb.id}</a><br />
                <span className="faint">{rb.created_at}</span></th>
            </tr>
          </thead>
          <tbody>
            {statRows.map(([label, get]) => (
              <tr key={label}>
                <td>{label}</td><td>{get(ra)}</td><td>{get(rb)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h2>Spec differences</h2>
      {diffs.length === 0 ? (
        <p className="muted">
          The specs are identical — any metric difference comes from the data
          (check the snapshot hashes and &ldquo;data as of&rdquo; above).
        </p>
      ) : (
        <div className="scroll-x">
          <table>
            <thead><tr><th>Field</th><th>Run A</th><th>Run B</th></tr></thead>
            <tbody>
              {diffs.map((k) => (
                <tr key={k}>
                  <td className="mono">{k}</td>
                  <td>{specA[k] ?? "—"}</td>
                  <td>{specB[k] ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default function ComparePage() {
  return (
    <Suspense fallback={<p className="faint">Loading…</p>}>
      <CompareInner />
    </Suspense>
  );
}
