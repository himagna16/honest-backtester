"use client";

// Results workstation. Verdict first, benchmark always, then segmented tabs:
// Overview · Charts · Cost & stress · Holdings · Audit note.

import { use, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { API_BASE, api } from "@/lib/api";
import type { AuditFinding, ReportEnvelope } from "@/lib/types";
import { startRun } from "@/lib/run";
import RunProgress from "@/components/RunProgress";
import { PageDisclaimer, TopBar } from "@/components/Shell";
import { Notice, Panel, Segmented } from "@/components/ui/Primitives";
import {
  IconAudit, IconChart, IconCompare, IconDownload, IconRisk, IconSnapshot,
} from "@/components/ui/Icons";
import {
  AuditList, AuditSummary, CostSweepTable, HoldingsLog, MetricCards, OOSPanel,
  PurgePanel, ResearchNote, RiskBadges, SignificancePanel, StressPanel,
  VerdictBanner,
} from "@/components/report/blocks";
import { DrawdownChart, EquityChart, RegimeBar } from "@/components/report/charts";

type Tab = "overview" | "charts" | "cost" | "stats" | "holdings" | "note";

export default function ReportPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const router = useRouter();
  const [envelope, setEnvelope] = useState<ReportEnvelope | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [rerunId, setRerunId] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("overview");
  const [compare, setCompare] = useState(true);

  useEffect(() => {
    api<ReportEnvelope>(`/api/report/${id}`)
      .then(setEnvelope)
      .catch((e) => setError(String(e.message ?? e)));
  }, [id]);

  if (error) {
    return (
      <>
        <TopBar crumb="Report" />
        <main className="content"><Notice kind="err">{error}</Notice></main>
      </>
    );
  }
  if (!envelope) {
    return (
      <>
        <TopBar crumb="Report" />
        <main className="content"><p className="faint">Loading report…</p></main>
      </>
    );
  }

  const r = envelope.report;
  const benchLabel = r.benchmark.label;

  const runVariant = async (check: string) => {
    const spec = structuredClone(r.spec) as Record<string, any>;
    if (check === "selection_bias") {
      spec.universe = { preset: "sp500" };
      spec.name = `${spec.name} · S&P 500 re-run`.slice(0, 80);
    } else if (check === "benchmark_fit") {
      spec.benchmark = "sixty_forty";
      spec.name = `${spec.name} · vs 60/40`.slice(0, 80);
    } else return;
    try {
      const { run_id } = await startRun({ spec });
      setRerunId(run_id);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  if (rerunId) {
    return (
      <>
        <TopBar crumb="Report" strategy={String(r.spec.name ?? "")} />
        <main className="content" style={{ maxWidth: 620 }}>
          <RunProgress runId={rerunId}
            onDone={(reportId) => router.push(`/report/${reportId}`)}
            onError={(m) => { setRerunId(null); setError(m); }} />
        </main>
      </>
    );
  }

  const equitySeries = r.equity_curve.map((p) => p[1]).filter((v): v is number => v !== null);
  const ddSeries = r.drawdown_curve.map((p) => p[1]).filter((v): v is number => v !== null);

  return (
    <>
      <TopBar
        crumb="Report"
        strategy={String(r.spec.name ?? "")}
        actions={
          <>
            <a className="btn" href={`${API_BASE}/api/report/${id}/export.png`}>
              <IconDownload size={13} /> PNG
            </a>
            <a className="btn" href={`${API_BASE}/api/report/${id}/export.pdf`}>
              <IconDownload size={13} /> PDF
            </a>
          </>
        }
      />
      <main className="content report">
        {/* The answer, then the evidence — one calm group, generous space
            before the tabbed detail below. */}
        <div className="report-lede">
          <div className="row" style={{ justifyContent: "space-between" }}>
            <span className="faint">
              {r.data.effective_start} → {r.data.effective_end} · {r.data.universe_size} tickers ·
              data as of {r.data.as_of}{r.data.stale ? " (cached)" : ""}
            </span>
            <span className="faint mono" title={`snapshot ${r.run?.snapshot_hash}`}>
              <IconSnapshot size={12} /> {r.run?.snapshot_hash?.slice(0, 10)}
            </span>
          </div>

          <VerdictBanner verdict={r.verdict} report={r} />
          <RiskBadges report={r} />
          <MetricCards metrics={r.metrics} benchmark={r.benchmark} benchLabel={benchLabel}
            equity={equitySeries} drawdown={ddSeries} />
        </div>

        <div className="stack report-detail">
          <div className="row">
            <Segmented<Tab>
              label="Report sections"
              value={tab}
              onChange={setTab}
              options={[
                { value: "overview", label: "Overview" },
                { value: "charts", label: "Charts", icon: <IconChart size={12} /> },
                { value: "cost", label: "Cost & stress", icon: <IconRisk size={12} /> },
                { value: "stats", label: "Significance" },
                { value: "holdings", label: "Holdings" },
                { value: "note", label: "Audit note", icon: <IconAudit size={12} /> },
              ]}
            />
            {tab === "charts" && (
              <button className="ghost" onClick={() => setCompare(!compare)}
                aria-pressed={compare}>
                <IconCompare size={13} /> {compare ? "Hide" : "Show"} gross line
              </button>
            )}
          </div>

          {tab === "overview" && (
            <div className="stack">
              <p className="readback">{r.readback}</p>
              <AuditSummary report={r} onSeeAll={() => setTab("note")} />
            </div>
          )}

          {tab === "charts" && (
            <div className="stack">
              <EquityChart points={r.equity_curve} oosStart={r.oos.out_of_sample.start}
                benchLabel={benchLabel} showGross={compare} />
              <DrawdownChart points={r.drawdown_curve} benchLabel={benchLabel} />
              <RegimeBar segments={r.regimes.segments} hasBear={r.regimes.has_bear_market} />
            </div>
          )}

          {tab === "cost" && (
            <div className="stack">
              <CostSweepTable rows={r.cost_sweep} />
              <StressPanel report={r} />
              <OOSPanel report={r} />
            </div>
          )}

          {tab === "stats" && (
            <div className="stack">
              <SignificancePanel report={r} />
              <PurgePanel report={r} />
            </div>
          )}

          {tab === "holdings" && (
            <div className="stack">
              <HoldingsLog report={r} />
              <Panel title="Data provenance" icon={<IconSnapshot size={15} />} tight>
                <table>
                  <tbody>
                    <tr><td>Universe size</td><td>{r.data.universe_size}</td></tr>
                    <tr><td>Snapshot hash</td><td className="mono">{r.run?.snapshot_hash?.slice(0, 16)}</td></tr>
                    <tr><td>Source</td><td>{r.run?.data_source}</td></tr>
                    <tr><td>Risk-free rate</td><td>{r.data.rf_source === "^IRX" ? "13-week T-bill" : "0% (disclosed)"}</td></tr>
                    <tr><td>Excluded tickers</td><td>{r.data.missing.length}</td></tr>
                  </tbody>
                </table>
                {r.data.notes.map((n) => (
                  <p key={n} className="faint" style={{ marginTop: 8 }}>{n}</p>
                ))}
              </Panel>
            </div>
          )}

          {tab === "note" && (
            <div className="stack">
              <ResearchNote report={r} onAction={runVariant} />
              <Panel title="All findings" icon={<IconAudit size={15} />} tight>
                <AuditList findings={r.audit} onAction={(f) => runVariant(f.check)} />
              </Panel>
            </div>
          )}
        </div>
        <PageDisclaimer />
      </main>
    </>
  );
}
