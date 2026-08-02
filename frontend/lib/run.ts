import { API_BASE, api } from "@/lib/api";

export interface RunEvent {
  stage: string;
  detail: string;
  pct: number;
  report_id?: string;
}

export async function startRun(body: {
  spec?: Record<string, unknown>;
  rerun_of?: string;
  refresh_data?: boolean;
  strategy_id?: string;
}): Promise<{ run_id: string; strategy_id: string; warnings: string[] }> {
  return api("/api/run-backtest", { method: "POST", json: body });
}

/** Subscribe to a run's SSE progress. Returns an unsubscribe function. */
export function subscribeRun(
  runId: string,
  onEvent: (e: RunEvent) => void,
): () => void {
  const source = new EventSource(`${API_BASE}/api/run-backtest/${runId}/events`);
  source.onmessage = (msg) => {
    try {
      const event = JSON.parse(msg.data) as RunEvent;
      onEvent(event);
      if (event.stage === "done" || event.stage === "error") source.close();
    } catch {
      /* ignore malformed keepalives */
    }
  };
  source.onerror = () => {
    // The stream closes itself on done; a transport error mid-run surfaces
    // as a stalled checklist and the journal still records the outcome.
    source.close();
  };
  return () => source.close();
}
