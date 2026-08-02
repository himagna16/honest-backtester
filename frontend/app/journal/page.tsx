"use client";

// The strategy library (Section 1): verdict badges, never returns — the
// journal visually rewards honesty, not big numbers.

import { useEffect, useState } from "react";
import Link from "next/link";
import { api } from "@/lib/api";

interface StrategyRow {
  id: string;
  name: string;
  updated_at: string;
  versions: number;
  runs: number;
  last_verdict_color: string | null;
  last_verdict_headline: string | null;
  last_report_id: string | null;
}

const BADGE: Record<string, { icon: string; label: string }> = {
  red: { icon: "✕", label: "Killed" },
  amber: { icon: "▲", label: "Wounded" },
  green: { icon: "✓", label: "Survived" },
};

export default function Journal() {
  const [rows, setRows] = useState<StrategyRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api<StrategyRow[]>("/api/strategies")
      .then(setRows)
      .catch((e) => setError(String(e.message ?? e)));
  }, []);

  if (error) return <p className="errbox">✕ {error}</p>;
  if (rows === null) return <p className="faint">Loading the journal…</p>;

  return (
    <div>
      <h1>Journal</h1>
      {rows.length === 0 ? (
        <div className="card">
          <p className="muted">
            No strategies yet. Most ideas don&apos;t survive the audit —
            that&apos;s the point.
          </p>
          <p>
            <Link href="/">Describe your first one →</Link>
          </p>
        </div>
      ) : (
        rows.map((s) => {
          const badge = s.last_verdict_color ? BADGE[s.last_verdict_color] : null;
          return (
            <div className="card" key={s.id}
              style={{ display: "flex", justifyContent: "space-between",
                       gap: "1rem", flexWrap: "wrap", alignItems: "center" }}>
              <div>
                <Link href={`/strategy/${s.id}`} style={{ fontWeight: 650 }}>
                  {s.name}
                </Link>
                <div className="faint">
                  {s.versions} version{s.versions === 1 ? "" : "s"} ·{" "}
                  {s.runs} run{s.runs === 1 ? "" : "s"} · updated {s.updated_at}
                </div>
                {s.last_verdict_headline && (
                  <div className="muted" style={{ fontSize: "0.9rem" }}>
                    {s.last_verdict_headline}
                  </div>
                )}
              </div>
              {badge && (
                <span className={`verdict verdict-${s.last_verdict_color}`}
                  style={{ padding: "0.4rem 0.8rem", margin: 0 }}>
                  <span className="badge" style={{ margin: 0 }}>
                    {badge.icon} {badge.label}
                  </span>
                </span>
              )}
            </div>
          );
        })
      )}
    </div>
  );
}
