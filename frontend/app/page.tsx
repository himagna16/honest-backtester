"use client";

// Describe → confirm the spec → run the gauntlet → Truth Report.

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { api, ApiError } from "@/lib/api";
import type { TranslateResult as TR } from "@/lib/types";
import SpecReview from "@/components/SpecReview";
import RunProgress from "@/components/RunProgress";
import { startRun } from "@/lib/run";

const CHIPS = [
  "12-1 momentum, top 10 in the S&P 500, monthly",
  "Buy S&P stocks that dropped 5%+ yesterday, hold 5 days",
  "Hold the 20 least volatile S&P stocks, quarterly",
  "60/40 stocks and bonds, rebalanced yearly",
];

type Phase =
  | { kind: "idle" }
  | { kind: "translating" }
  | { kind: "clarify"; result: TR }
  | { kind: "spec"; result: TR }
  | { kind: "unsupported"; reason: string }
  | { kind: "error"; message: string; canPasteJson?: boolean }
  | { kind: "running"; runId: string };

export default function Home() {
  const router = useRouter();
  const [text, setText] = useState("");
  const [answers, setAnswers] = useState<Record<number, string>>({});
  const [phase, setPhase] = useState<Phase>({ kind: "idle" });
  const [llmConfigured, setLlmConfigured] = useState<boolean | null>(null);
  const [jsonDraft, setJsonDraft] = useState("");

  useEffect(() => {
    api<{ status: string; llm_configured: boolean }>("/health")
      .then((h) => setLlmConfigured(h.llm_configured))
      .catch(() => setLlmConfigured(null));
  }, []);

  const translate = async (input: string) => {
    setPhase({ kind: "translating" });
    try {
      const result = await api<TR>("/api/translate-strategy", {
        method: "POST",
        json: { text: input },
      });
      if (result.status === "spec") setPhase({ kind: "spec", result });
      else if (result.status === "needs_clarification") {
        setAnswers({});
        setPhase({ kind: "clarify", result });
      } else if (result.status === "unsupported")
        setPhase({ kind: "unsupported", reason: result.unsupported_reason ?? "" });
      else
        setPhase({
          kind: "error",
          message: result.message ?? (result.errors ?? []).join("; "),
        });
    } catch (e) {
      const err = e as ApiError;
      setPhase({
        kind: "error",
        message: err.message,
        canPasteJson: err.code === "llm_not_configured",
      });
    }
  };

  const submitAnswers = (result: TR) => {
    const lines = (result.questions ?? []).map(
      (q, i) => `${q.question} ${answers[i] ?? ""}`,
    );
    translate(`${text}\n\nClarifications: ${lines.join(" | ")}`);
  };

  const runSpec = async (spec: Record<string, unknown>) => {
    try {
      const { run_id } = await startRun({ spec });
      setPhase({ kind: "running", runId: run_id });
    } catch (e) {
      setPhase({ kind: "error", message: (e as Error).message });
    }
  };

  const runPastedJson = () => {
    try {
      runSpec(JSON.parse(jsonDraft));
    } catch {
      setPhase({ kind: "error", message: "that isn't valid JSON", canPasteJson: true });
    }
  };

  return (
    <div className="describe">
      <h1>Describe your strategy</h1>
      <p className="muted">
        Be specific about what you buy, when, and when you sell. The backtest
        will try to kill it — that&apos;s the point.
      </p>

      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder='e.g. "Buy the 5 strongest S&P 500 stocks by 12-month momentum, rebalance monthly"'
        aria-label="Strategy description"
      />
      <div className="chips" role="list">
        {CHIPS.map((chip) => (
          <button key={chip} className="chip" role="listitem"
            onClick={() => setText(chip)}>
            {chip}
          </button>
        ))}
      </div>
      <div style={{ display: "flex", gap: "0.7rem", alignItems: "center", flexWrap: "wrap" }}>
        <button className="primary" disabled={!text.trim() || phase.kind === "translating"}
          onClick={() => translate(text)}>
          {phase.kind === "translating" ? "Translating…" : "Translate to a spec"}
        </button>
        {llmConfigured === false && (
          <span className="faint">
            No ANTHROPIC_API_KEY configured — you can still paste a spec JSON below.
          </span>
        )}
      </div>

      {phase.kind === "clarify" && (
        <div className="card">
          <h2>A few things you didn&apos;t specify</h2>
          <p className="faint">
            The translator never guesses material parameters — answer these and
            it will re-translate.
          </p>
          {(phase.result.questions ?? []).map((q, i) => (
            <div key={i} style={{ margin: "0.7rem 0" }}>
              <p style={{ margin: "0 0 0.3rem" }}>{q.question}</p>
              <div className="chips">
                {q.options.map((option) => (
                  <button key={option}
                    className="chip"
                    style={answers[i] === option
                      ? { borderColor: "var(--accent)", color: "var(--accent)" }
                      : undefined}
                    onClick={() => setAnswers({ ...answers, [i]: option })}>
                    {option}
                  </button>
                ))}
              </div>
            </div>
          ))}
          <button className="primary"
            disabled={Object.keys(answers).length < (phase.result.questions ?? []).length}
            onClick={() => submitAnswers(phase.result)}>
            Re-translate with answers
          </button>
        </div>
      )}

      {phase.kind === "spec" && phase.result.spec && (
        <SpecReview
          initialSpec={phase.result.spec}
          assumptions={phase.result.assumptions ?? []}
          onRun={runSpec}
        />
      )}

      {phase.kind === "unsupported" && (
        <div className="card">
          <h2>Not supported in V0 — honestly</h2>
          <p className="muted">{phase.reason}</p>
          <p className="faint">The request was logged for the roadmap.</p>
        </div>
      )}

      {phase.kind === "error" && (
        <div>
          <p className="errbox">✕ {phase.message}</p>
        </div>
      )}

      {(phase.kind === "error" && phase.canPasteJson) || llmConfigured === false ? (
        <details className="card" open={phase.kind === "error" && phase.canPasteJson}>
          <summary>Paste a spec JSON directly (offline mode)</summary>
          <textarea value={jsonDraft} onChange={(e) => setJsonDraft(e.target.value)}
            placeholder='{"name": "...", "universe": {"preset": "sp500"}, ...}'
            style={{ fontFamily: "var(--mono)", fontSize: "0.85rem" }} />
          <button onClick={runPastedJson} disabled={!jsonDraft.trim()}>
            Validate &amp; run
          </button>
        </details>
      ) : null}

      {phase.kind === "running" && (
        <RunProgress
          runId={phase.runId}
          onDone={(reportId) => router.push(`/report/${reportId}`)}
          onError={(message) => setPhase({ kind: "error", message })}
        />
      )}
    </div>
  );
}
