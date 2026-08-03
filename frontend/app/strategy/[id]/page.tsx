"use client";

// Strategy detail: version history, run history with verdict badges,
// replay-by-default re-runs, spec export, compare selection.

import { use, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { API_BASE, api } from "@/lib/api";
import { startRun } from "@/lib/run";
import RunProgress from "@/components/RunProgress";
import { PageDisclaimer, TopBar } from "@/components/Shell";
import { Notice, Panel } from "@/components/ui/Primitives";
import {
  IconCheck, IconCompare, IconDownload, IconRefresh, IconSnapshot,
  IconWarning, IconX,
} from "@/components/ui/Icons";

interface RunRow {
  id: string; version: number; status: string; created_at: string;
  data_as_of: string | null; data_source: string | null; error: string | null;
  report_id: string | null; verdict_color: string | null; verdict_headline: string | null;
}
interface Detail {
  id: string; name: string; created_at: string;
  versions: { id: string; version: number; spec: Record<string, unknown>; created_at: string }[];
  runs: RunRow[];
}

const BADGE = {
  red: { label: "Killed", Icon: IconX },
  amber: { label: "Wounded", Icon: IconWarning },
  green: { label: "Survived", Icon: IconCheck },
} as const;

export default function StrategyPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const router = useRouter();
  const [detail, setDetail] = useState<Detail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [compare, setCompare] = useState<string[]>([]);

  useEffect(() => {
    api<Detail>(`/api/strategies/${id}`)
      .then(setDetail)
      .catch((e) => setError(String(e.message ?? e)));
  }, [id]);

  if (error && !detail) {
    return (
      <>
        <TopBar crumb="Journal" />
        <main className="content"><Notice kind="err">{error}</Notice></main>
      </>
    );
  }
  if (!detail) {
    return (
      <>
        <TopBar crumb="Journal" />
        <main className="content"><p className="faint">Loading…</p></main>
      </>
    );
  }

  const latestSpec = detail.versions[0]?.spec;
  const lastDone = detail.runs.find((r) => r.status === "done");

  const rerun = async (mode: "replay" | "refresh") => {
    try {
      const body = mode === "replay" && lastDone
        ? { rerun_of: lastDone.id }
        : { spec: latestSpec, strategy_id: detail.id, refresh_data: true };
      const { run_id } = await startRun(body);
      setRunId(run_id);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const toggle = (reportId: string) =>
    setCompare((prev) =>
      prev.includes(reportId) ? prev.filter((x) => x !== reportId) : [...prev.slice(-1), reportId]);

  if (runId) {
    return (
      <>
        <TopBar crumb="Journal" strategy={detail.name} />
        <main className="content" style={{ maxWidth: 620 }}>
          <RunProgress runId={runId}
            onDone={(reportId) => router.push(`/report/${reportId}`)}
            onError={(m) => { setRunId(null); setError(m); }} />
        </main>
      </>
    );
  }

  return (
    <>
      <TopBar
        crumb="Journal"
        strategy={detail.name}
        actions={
          <>
            {lastDone && (
              <button onClick={() => rerun("replay")}>
                <IconSnapshot size={13} /> Replay snapshot
              </button>
            )}
            <button onClick={() => rerun("refresh")}>
              <IconRefresh size={13} /> Refresh &amp; run
            </button>
            <a className="btn" href={`${API_BASE}/api/strategies/${detail.id}/spec.json`}>
              <IconDownload size={13} /> Spec
            </a>
          </>
        }
      />
      <main className="content">
        <div className="stack">
          {error && <Notice kind="err">{error}</Notice>}
          <Notice kind="info">
            Re-runs replay the frozen, content-hashed snapshot by default —
            identical numbers. &ldquo;Refresh &amp; run&rdquo; is the only path to new data.
          </Notice>

          <Panel title="Run history" tight
            right={compare.length === 2 ? (
              <Link className="btn primary" href={`/compare?a=${compare[0]}&b=${compare[1]}`}>
                <IconCompare size={13} /> Compare
              </Link>
            ) : <span className="faint">select two to compare</span>}>
            <div className="scroll-x">
              <table>
                <thead>
                  <tr>
                    <th>When</th><th>Spec</th><th>Data</th><th>Verdict</th>
                    <th>Report</th><th>Compare</th>
                  </tr>
                </thead>
                <tbody>
                  {detail.runs.map((r) => {
                    const badge = r.verdict_color
                      ? BADGE[r.verdict_color as keyof typeof BADGE] : null;
                    return (
                      <tr key={r.id}>
                        <td className="faint">{r.created_at.slice(0, 16)}</td>
                        <td>v{r.version}</td>
                        <td className="faint">
                          {r.status === "error"
                            ? `error: ${r.error?.slice(0, 40)}`
                            : `${r.data_source ?? ""}${r.data_as_of ? ` · ${r.data_as_of}` : ""}`}
                        </td>
                        <td>
                          {badge ? (
                            <span className={`pill pill-${r.verdict_color}`}>
                              <badge.Icon size={11} /> {badge.label}
                            </span>
                          ) : <span className="faint">{r.status}</span>}
                        </td>
                        <td>{r.report_id && <Link href={`/report/${r.report_id}`}>open</Link>}</td>
                        <td>
                          {r.report_id && (
                            <input type="checkbox" aria-label={`Select run ${r.created_at} for comparison`}
                              checked={compare.includes(r.report_id)}
                              onChange={() => toggle(r.report_id!)} />
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </Panel>

          <Panel title="Version history" tight>
            {detail.versions.map((v) => (
              <details key={v.id} style={{ borderBottom: "1px solid var(--line)", padding: "8px 0" }}>
                <summary style={{ cursor: "pointer", fontSize: 13 }}>
                  v{v.version}
                  <span className="faint" style={{ marginLeft: 8 }}>
                    {v.created_at.slice(0, 16)}
                    {v.version === detail.versions[0].version ? " · current" : ""}
                  </span>
                </summary>
                <pre className="scroll-x mono" style={{
                  background: "var(--surface-2)", border: "1px solid var(--line)",
                  borderRadius: "var(--r-sm)", padding: 12, fontSize: 11.5, marginTop: 8,
                }}>
                  {JSON.stringify(v.spec, null, 2)}
                </pre>
              </details>
            ))}
          </Panel>
        </div>
        <PageDisclaimer />
      </main>
    </>
  );
}
