"use client";

// Strategy library — verdict badges, never returns. The journal rewards
// honesty, not big numbers.

import { useEffect, useState } from "react";
import Link from "next/link";
import { api } from "@/lib/api";
import { PageDisclaimer, TopBar } from "@/components/Shell";
import { Notice, Panel } from "@/components/ui/Primitives";
import { IconCheck, IconEmptyChart, IconWarning, IconX } from "@/components/ui/Icons";

interface StrategyRow {
  id: string; name: string; updated_at: string; versions: number; runs: number;
  last_verdict_color: string | null; last_verdict_headline: string | null;
  last_report_id: string | null;
}

const BADGE = {
  red: { label: "Killed", Icon: IconX },
  amber: { label: "Wounded", Icon: IconWarning },
  green: { label: "Survived", Icon: IconCheck },
} as const;

export default function Journal() {
  const [rows, setRows] = useState<StrategyRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api<StrategyRow[]>("/api/strategies")
      .then(setRows)
      .catch((e) => setError(String(e.message ?? e)));
  }, []);

  const counts = rows?.reduce<Record<string, number>>((acc, r) => {
    if (r.last_verdict_color) acc[r.last_verdict_color] = (acc[r.last_verdict_color] ?? 0) + 1;
    return acc;
  }, {}) ?? {};

  return (
    <>
      <TopBar
        crumb="Journal"
        actions={
          rows && rows.length > 0 ? (
            <span className="faint">
              {counts.green ?? 0} survived · {counts.amber ?? 0} wounded · {counts.red ?? 0} killed
            </span>
          ) : null
        }
      />
      <main className="content">
        {error && <Notice kind="err">{error}</Notice>}
        {!rows && !error && <p className="faint">Loading…</p>}

        {rows && rows.length === 0 && (
          <Panel>
            <div className="empty">
              <IconEmptyChart size={34} />
              <p>No strategies yet.</p>
              <p className="faint">
                Most ideas don&apos;t survive the audit — that&apos;s the point.
              </p>
              <Link className="btn primary" href="/workbench" style={{ marginTop: 12 }}>
                Describe your first
              </Link>
            </div>
          </Panel>
        )}

        {rows && rows.length > 0 && (
          <Panel tight>
            <div className="scroll-x">
              <table>
                <thead>
                  <tr>
                    <th>Strategy</th><th>Verdict</th><th>Versions</th>
                    <th>Runs</th><th>Updated</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((s) => {
                    const badge = s.last_verdict_color
                      ? BADGE[s.last_verdict_color as keyof typeof BADGE]
                      : null;
                    return (
                      <tr key={s.id}>
                        <td style={{ textAlign: "left", whiteSpace: "normal" }}>
                          <Link href={`/strategy/${s.id}`} style={{ fontWeight: 550 }}>
                            {s.name}
                          </Link>
                          {s.last_verdict_headline && (
                            <div className="faint">{s.last_verdict_headline}</div>
                          )}
                        </td>
                        <td>
                          {badge ? (
                            <span className={`pill pill-${s.last_verdict_color}`}>
                              <badge.Icon size={11} /> {badge.label}
                            </span>
                          ) : <span className="faint">—</span>}
                        </td>
                        <td>{s.versions}</td>
                        <td>{s.runs}</td>
                        <td className="faint">{s.updated_at.slice(0, 16)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </Panel>
        )}
        <PageDisclaimer />
      </main>
    </>
  );
}
