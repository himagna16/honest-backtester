"use client";

// Strategy detail: version history, run history with verdict badges,
// replay-by-default re-runs (Gap #2), spec export, compare-runs diff.

import { use, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { API_BASE, api } from "@/lib/api";
import { startRun } from "@/lib/run";
import RunProgress from "@/components/RunProgress";

interface RunRow {
  id: string; version: number; status: string; created_at: string;
  data_as_of: string | null; data_source: string | null; error: string | null;
  report_id: string | null; verdict_color: string | null;
  verdict_headline: string | null;
}

interface Detail {
  id: string; name: string; created_at: string;
  versions: { id: string; version: number; spec: Record<string, unknown>;
              created_at: string }[];
  runs: RunRow[];
}

const BADGE: Record<string, string> = { red: "✕ Killed", amber: "▲ Wounded", green: "✓ Survived" };

export default function StrategyPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const router = useRouter();
  const [detail, setDetail] = useState<Detail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [compare, setCompare] = useState<string[]>([]);

  const load = () =>
    api<Detail>(`/api/strategies/${id}`)
      .then(setDetail)
      .catch((e) => setError(String(e.message ?? e)));
  useEffect(() => { load(); /* eslint-disable-line */ }, [id]);

  if (error) return <p className="errbox">✕ {error}</p>;
  if (!detail) return <p className="faint">Loading…</p>;

  const latestSpec = detail.versions[0]?.spec;
  const lastDoneRun = detail.runs.find((r) => r.status === "done");

  const rerun = async (mode: "replay" | "refresh") => {
    try {
      const body =
        mode === "replay" && lastDoneRun
          ? { rerun_of: lastDoneRun.id }
          : { spec: latestSpec, strategy_id: detail.id, refresh_data: mode === "refresh" };
      const { run_id } = await startRun(body);
      setRunId(run_id);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const toggleCompare = (reportId: string) => {
    setCompare((prev) =>
      prev.includes(reportId)
        ? prev.filter((x) => x !== reportId)
        : [...prev.slice(-1), reportId],
    );
  };

  if (runId) {
    return (
      <RunProgress runId={runId}
        onDone={(reportId) => router.push(`/report/${reportId}`)}
        onError={(message) => { setRunId(null); setError(message); }} />
    );
  }

  return (
    <div>
      <h1>{detail.name}</h1>
      <div style={{ display: "flex", gap: "0.6rem", flexWrap: "wrap" }}>
        {lastDoneRun && (
          <button className="primary" onClick={() => rerun("replay")}>
            Re-run from frozen snapshot
          </button>
        )}
        <button onClick={() => rerun("refresh")}>
          Refresh data &amp; run (new numbers)
        </button>
        <a className="btn" href={`${API_BASE}/api/strategies/${detail.id}/spec.json`}>
          Export spec JSON
        </a>
      </div>
      <p className="faint">
        Re-runs replay the run&apos;s frozen, content-hashed data snapshot by
        default — identical numbers, byte for byte. &ldquo;Refresh data&rdquo;
        is the only path to new numbers.
      </p>

      <h2>Run history</h2>
      {detail.runs.length === 0 && <p className="muted">No runs yet.</p>}
      <div className="scroll-x">
        <table>
          <thead>
            <tr><th>When</th><th>Spec v</th><th>Data</th><th>Verdict</th>
                <th>Report</th><th>Compare</th></tr>
          </thead>
          <tbody>
            {detail.runs.map((r) => (
              <tr key={r.id}>
                <td>{r.created_at}</td>
                <td>v{r.version}</td>
                <td className="faint">
                  {r.status === "error" ? `error: ${r.error?.slice(0, 60)}` :
                    `${r.data_source ?? ""} ${r.data_as_of ? `as of ${r.data_as_of}` : ""}`}
                </td>
                <td>
                  {r.verdict_color ? (
                    <span style={{
                      color: `var(--${r.verdict_color === "amber" ? "amber" : r.verdict_color})`,
                      fontWeight: 650 }}>
                      {BADGE[r.verdict_color]}
                    </span>
                  ) : r.status}
                </td>
                <td>{r.report_id &&
                  <a href={`/report/${r.report_id}`}>open</a>}</td>
                <td>{r.report_id && (
                  <input type="checkbox" aria-label="select for comparison"
                    checked={compare.includes(r.report_id)}
                    onChange={() => toggleCompare(r.report_id!)} />
                )}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {compare.length === 2 && (
        <p>
          <a className="btn" href={`/compare?a=${compare[0]}&b=${compare[1]}`}>
            Compare selected runs →
          </a>
        </p>
      )}

      <h2>Version history</h2>
      {detail.versions.map((v) => (
        <details key={v.id} className="card">
          <summary>
            v{v.version} · {v.created_at}
            {v.version === detail.versions[0].version ? " (current)" : ""}
          </summary>
          <pre className="scroll-x" style={{ fontSize: "0.78rem" }}>
            {JSON.stringify(v.spec, null, 2)}
          </pre>
        </details>
      ))}
    </div>
  );
}
