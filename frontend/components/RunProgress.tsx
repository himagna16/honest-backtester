"use client";

// Loading state = analysis in progress, not a generic spinner. Each stage is a
// real backend event streamed over SSE.

import { useEffect, useRef, useState } from "react";
import { subscribeRun, type RunEvent } from "@/lib/run";
import { Panel } from "@/components/ui/Primitives";
import { IconAudit, IconCheck } from "@/components/ui/Icons";

const STEPS = [
  { key: "queued", label: "Queued" },
  { key: "fetching", label: "Loading price panel" },
  { key: "snapshot", label: "Freezing data snapshot" },
  { key: "simulating", label: "Simulating with costs" },
  { key: "auditing", label: "Running the honesty audit" },
  { key: "finalizing", label: "Composing the report" },
];

export default function RunProgress({
  runId, onDone, onError,
}: {
  runId: string;
  onDone: (reportId: string) => void;
  onError: (message: string) => void;
}) {
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [elapsed, setElapsed] = useState(0);
  const doneRef = useRef(false);

  // A long stage must look like work, not a hang: the neutral-universe re-run
  // can fetch hundreds of tickers.
  useEffect(() => {
    const t = setInterval(() => setElapsed((s) => s + 1), 1000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    const unsubscribe = subscribeRun(runId, (event) => {
      setEvents((prev) => [...prev, event]);
      if (event.stage === "done" && event.report_id && !doneRef.current) {
        doneRef.current = true;
        onDone(event.report_id);
      }
      if (event.stage === "error" && !doneRef.current) {
        doneRef.current = true;
        onError(event.detail);
      }
    });
    return unsubscribe;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId]);

  const current = events[events.length - 1];
  const currentIndex = STEPS.findIndex((s) => s.key === current?.stage);
  const auditDetails = events.filter((e) => e.stage === "auditing").map((e) => e.detail);
  const pct = current?.pct ?? 0;

  return (
    <Panel title="Analysis in progress" icon={<IconAudit size={15} />}>
      <ul className="checklist" aria-live="polite">
        {STEPS.map((step, i) => {
          const done = currentIndex > i;
          const active = currentIndex === i;
          return (
            <li key={step.key} className={active ? "active" : done ? "done" : ""}>
              <span className="dot">
                {done ? <IconCheck size={13} /> : active ? <span className="pulse" /> : null}
              </span>
              {step.label}
            </li>
          );
        })}
      </ul>
      <div className="progresstrack"><i style={{ width: `${pct}%` }} /></div>
      {auditDetails.length > 0 && (
        <p className="progressnote">{auditDetails[auditDetails.length - 1]}</p>
      )}
      <p className="faint" style={{ marginTop: 8, display: "flex", justifyContent: "space-between" }}>
        <span>Every run is fully audited — there is no fast path.</span>
        <span className="mono">
          {Math.floor(elapsed / 60)}:{String(elapsed % 60).padStart(2, "0")}
        </span>
      </p>
      {elapsed > 45 && (
        <p className="faint" style={{ marginTop: 4 }}>
          Still working — a neutral-universe re-run can fetch several hundred
          tickers on first use.
        </p>
      )}
    </Panel>
  );
}
