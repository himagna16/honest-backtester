"use client";

// The Truth Report (Section 1): verdict first, benchmark always, findings
// ordered by severity, cost sweep, log-scale equity, drawdown, regimes, OOS.

import { use, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { API_BASE, api } from "@/lib/api";
import type { AuditFinding, ReportEnvelope } from "@/lib/types";
import { startRun } from "@/lib/run";
import RunProgress from "@/components/RunProgress";
import {
  AuditList, CostSweepTable, OOSPanel, StatRow, VerdictBanner,
} from "@/components/report/blocks";
import { DrawdownChart, EquityChart, RegimeBar } from "@/components/report/charts";

export default function ReportPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const router = useRouter();
  const [envelope, setEnvelope] = useState<ReportEnvelope | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [rerunId, setRerunId] = useState<string | null>(null);

  useEffect(() => {
    api<ReportEnvelope>(`/api/report/${id}`)
      .then(setEnvelope)
      .catch((e) => setError(String(e.message ?? e)));
  }, [id]);

  if (error) return <p className="errbox">✕ {error}</p>;
  if (!envelope) return <p className="faint">Loading the Truth Report…</p>;
  const r = envelope.report;

  const handleAction = async (finding: AuditFinding) => {
    const spec = structuredClone(r.spec) as Record<string, any>;
    if (finding.check === "selection_bias") {
      spec.universe = { preset: "sp500" };
      spec.name = `${spec.name} (S&P 500 re-run)`.slice(0, 80);
    } else if (finding.check === "benchmark_fit") {
      spec.benchmark = "sixty_forty";
      spec.name = `${spec.name} (vs 60/40)`.slice(0, 80);
    } else {
      return;
    }
    try {
      const { run_id } = await startRun({ spec });
      setRerunId(run_id);
    } catch (e: any) {
      setError(String(e.message ?? e));
    }
  };

  if (rerunId) {
    return (
      <RunProgress
        runId={rerunId}
        onDone={(reportId) => router.push(`/report/${reportId}`)}
        onError={(message) => { setRerunId(null); setError(message); }}
      />
    );
  }

  return (
    <div>
      <p className="faint" style={{ display: "flex", gap: "0.8rem",
                                    flexWrap: "wrap", alignItems: "center" }}>
        <span>
          {r.spec.name as string} · {r.data.effective_start} → {r.data.effective_end} ·
          data as of {r.data.as_of}{r.data.stale ? " (cached)" : ""} ·
          run {r.run?.id}{r.run?.replay_of ? ` (snapshot replay of ${r.run.replay_of})` : ""}
        </span>
        <a className="btn" href={`${API_BASE}/api/report/${id}/export.png`}>
          Export PNG
        </a>
        <a className="btn" href={`${API_BASE}/api/report/${id}/export.pdf`}>
          Export PDF
        </a>
      </p>
      <VerdictBanner verdict={r.verdict} />
      <StatRow metrics={r.metrics} benchmark={r.benchmark} benchLabel={r.benchmark.label} />
      <p className="faint" style={{ fontStyle: "italic" }}>“{r.readback}”</p>

      <AuditList findings={r.audit} onAction={handleAction} />
      <CostSweepTable rows={r.cost_sweep} />
      <EquityChart points={r.equity_curve} oosStart={r.oos.out_of_sample.start} />
      <DrawdownChart points={r.drawdown_curve} />
      <RegimeBar segments={r.regimes.segments} hasBear={r.regimes.has_bear_market} />
      <OOSPanel report={r} />

      <section className="card">
        <h3>Data provenance</h3>
        <p className="muted">
          {r.data.universe_size} universe tickers · snapshot{" "}
          <code>{r.run?.snapshot_hash?.slice(0, 12)}</code> · source{" "}
          {r.run?.data_source} · risk-free: {r.data.rf_source === "^IRX"
            ? "13-week T-bill (^IRX)" : "0% (disclosed)"}
        </p>
        {r.data.notes.map((n) => <p key={n} className="faint">· {n}</p>)}
      </section>

      <p className="faint" style={{ marginTop: "1.5rem" }}>{r.disclaimer}</p>
    </div>
  );
}
