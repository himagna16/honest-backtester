"use client";

// The loading state IS the audit working (Section 1): an honest checklist,
// not a fake spinner.

import { useEffect, useRef, useState } from "react";
import { subscribeRun, type RunEvent } from "@/lib/run";

const STEPS: { key: string; label: string }[] = [
  { key: "queued", label: "Queued" },
  { key: "fetching", label: "Fetching prices" },
  { key: "snapshot", label: "Freezing the data snapshot (content-hashed)" },
  { key: "simulating", label: "Simulating with costs (0 / 5 / 20 bps)" },
  { key: "auditing", label: "Running the honesty audit" },
  { key: "finalizing", label: "Writing the Truth Report" },
];

export default function RunProgress({
  runId,
  onDone,
  onError,
}: {
  runId: string;
  onDone: (reportId: string) => void;
  onError: (message: string) => void;
}) {
  const [events, setEvents] = useState<RunEvent[]>([]);
  const doneRef = useRef(false);

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

  const seen = new Set(events.map((e) => e.stage));
  const current = events[events.length - 1];
  const auditDetails = events
    .filter((e) => e.stage === "auditing")
    .map((e) => e.detail);

  return (
    <div className="card" aria-live="polite">
      <h2>Running the gauntlet…</h2>
      <ul className="checklist">
        {STEPS.map((step, i) => {
          const done =
            seen.has(step.key) &&
            (current?.stage !== step.key ||
              STEPS.findIndex((s) => s.key === current?.stage) > i);
          const active = current?.stage === step.key;
          return (
            <li key={step.key} className={active ? "active" : done ? "done" : ""}>
              {done ? "✓" : active ? "▸" : "○"} {step.label}
            </li>
          );
        })}
      </ul>
      {auditDetails.length > 0 && (
        <p className="progressnote">{auditDetails[auditDetails.length - 1]}</p>
      )}
      <p className="faint">
        The audit always runs — there is no toggle. Typical run: 10–60&nbsp;s.
      </p>
    </div>
  );
}
