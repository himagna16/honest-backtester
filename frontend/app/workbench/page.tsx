"use client";

// Workbench: strategy input (left) + configuration (right). No marketing hero —
// the first screen is the actual backtesting workflow.

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { api, ApiError } from "@/lib/api";
import type { TranslateResult as TR, ValidateResult } from "@/lib/types";
import SpecReview from "@/components/SpecReview";
import RunProgress from "@/components/RunProgress";
import { startRun } from "@/lib/run";
import { PageDisclaimer, TopBar } from "@/components/Shell";
import { Notice, Panel, Segmented } from "@/components/ui/Primitives";
import {
  IconDescribe, IconEmptyChart, IconSettings, IconUpload,
} from "@/components/ui/Icons";

const CHIPS = [
  "12-1 momentum, top 10 in the S&P 500, monthly",
  "Buy S&P stocks that dropped 5%+ yesterday, hold 5 days",
  "Hold the 20 least volatile S&P stocks, quarterly",
  "60/40 stocks and bonds, rebalanced yearly",
];

type Mode = "describe" | "import";

export default function Workbench() {
  const router = useRouter();
  const [mode, setMode] = useState<Mode>("describe");
  const [text, setText] = useState("");
  const [jsonDraft, setJsonDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [spec, setSpec] = useState<Record<string, unknown> | null>(null);
  const [assumptions, setAssumptions] = useState<string[]>([]);
  const [questions, setQuestions] = useState<TR["questions"]>([]);
  const [answers, setAnswers] = useState<Record<number, string>>({});
  const [unsupported, setUnsupported] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [llmReady, setLlmReady] = useState<boolean | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api<{ llm_configured: boolean }>("/health")
      .then((h) => {
        setLlmReady(h.llm_configured);
        if (!h.llm_configured) setMode("import");
      })
      .catch(() => setLlmReady(null));
  }, []);

  const reset = () => {
    setSpec(null); setAssumptions([]); setQuestions([]);
    setUnsupported(null); setError(null);
  };

  const translate = async (input: string) => {
    reset(); setBusy(true);
    try {
      const r = await api<TR>("/api/translate-strategy", {
        method: "POST", json: { text: input },
      });
      if (r.status === "spec" && r.spec) {
        setSpec(r.spec); setAssumptions(r.assumptions ?? []);
      } else if (r.status === "needs_clarification") {
        setQuestions(r.questions ?? []); setAnswers({});
      } else if (r.status === "unsupported") {
        setUnsupported(r.unsupported_reason ?? "");
      } else {
        setError(r.message ?? (r.errors ?? []).join("; "));
      }
    } catch (e) {
      setError((e as ApiError).message);
    } finally {
      setBusy(false);
    }
  };

  const loadJson = async (raw: string) => {
    reset(); setBusy(true);
    try {
      const parsed = JSON.parse(raw);
      const r = await api<ValidateResult>("/api/import-spec", {
        method: "POST", json: { spec: parsed },
      });
      if (r.valid && r.expanded_spec) setSpec(r.expanded_spec);
      else setError(r.errors.join(" · "));
    } catch (e) {
      setError(e instanceof SyntaxError ? "That file isn't valid JSON." : (e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const onFile = (file: File) => {
    const reader = new FileReader();
    reader.onload = () => {
      const raw = String(reader.result ?? "");
      setJsonDraft(raw);
      loadJson(raw);
    };
    reader.readAsText(file);
  };

  const run = async (finalSpec: Record<string, unknown>) => {
    try {
      const { run_id } = await startRun({ spec: finalSpec });
      setRunId(run_id);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const answered = questions && Object.keys(answers).length >= questions.length;

  return (
    <>
      <TopBar
        crumb="Workbench"
        strategy={spec ? String(spec.name ?? "") : undefined}
        actions={
          spec && !runId ? (
            <button className="ghost" onClick={reset}>Clear</button>
          ) : null
        }
      />
      <main className="content">
        <div className="grid-2">
          {/* ---------------- strategy input ---------------- */}
          <Panel
            title="Strategy"
            icon={<IconDescribe size={15} />}
            right={
              <Segmented<Mode>
                label="Strategy input mode"
                value={mode}
                onChange={setMode}
                options={[
                  { value: "describe", label: "Describe", icon: <IconDescribe size={13} /> },
                  { value: "import", label: "Import spec", icon: <IconUpload size={13} /> },
                ]}
              />
            }
          >
            {mode === "describe" ? (
              <div className="describe">
                <textarea
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  placeholder='Describe what you buy, when, and when you sell — e.g. "Buy the 5 strongest S&P 500 stocks by 12-month momentum, rebalance monthly"'
                  aria-label="Strategy description"
                />
                <div className="row" style={{ marginTop: 10 }}>
                  <button className="primary" disabled={!text.trim() || busy}
                    onClick={() => translate(text)}>
                    {busy ? "Translating…" : "Translate to spec"}
                  </button>
                  {llmReady === false && (
                    <span className="faint">
                      No API key — use Import spec, or set ANTHROPIC_API_KEY.
                    </span>
                  )}
                </div>
                <p className="eyebrow" style={{ marginTop: 18 }}>Starting points</p>
                <div className="chips">
                  {CHIPS.map((c) => (
                    <button key={c} className="chip" type="button"
                      aria-pressed={text === c} onClick={() => setText(c)}>
                      {c}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <div className="describe">
                <div className="row" style={{ marginBottom: 10 }}>
                  <button onClick={() => fileRef.current?.click()}>
                    <IconUpload size={14} /> Choose .json file
                  </button>
                  <input ref={fileRef} type="file" accept=".json,application/json"
                    style={{ display: "none" }}
                    onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])} />
                  <span className="faint">or paste below</span>
                </div>
                <textarea
                  value={jsonDraft}
                  onChange={(e) => setJsonDraft(e.target.value)}
                  spellCheck={false}
                  className="mono"
                  placeholder='{"name": "...", "universe": {"preset": "sp500"}, ...}'
                  aria-label="Strategy spec JSON"
                />
                <div className="row" style={{ marginTop: 10 }}>
                  <button className="primary" disabled={!jsonDraft.trim() || busy}
                    onClick={() => loadJson(jsonDraft)}>
                    Validate spec
                  </button>
                  <span className="faint">
                    Exported specs round-trip exactly.
                  </span>
                </div>
              </div>
            )}

            {/* clarification is part of the input conversation */}
            {questions && questions.length > 0 && (
              <div style={{ marginTop: 20, borderTop: "1px solid var(--line)", paddingTop: 16 }}>
                <p className="eyebrow">Needs clarification</p>
                <p className="faint" style={{ marginTop: 0 }}>
                  The translator never guesses a material parameter.
                </p>
                {questions.map((q, i) => (
                  <div key={i} style={{ marginTop: 12 }}>
                    <p style={{ margin: "0 0 6px", fontSize: 13 }}>{q.question}</p>
                    <div className="chips">
                      {q.options.map((o) => (
                        <button key={o} className="chip" type="button"
                          aria-pressed={answers[i] === o}
                          onClick={() => setAnswers({ ...answers, [i]: o })}>
                          {o}
                        </button>
                      ))}
                    </div>
                  </div>
                ))}
                <button className="primary" style={{ marginTop: 14 }} disabled={!answered}
                  onClick={() =>
                    translate(
                      `${text}\n\nClarifications: ${questions
                        .map((q, i) => `${q.question} ${answers[i] ?? ""}`)
                        .join(" | ")}`,
                    )
                  }>
                  Re-translate with answers
                </button>
              </div>
            )}

            {unsupported && (
              <div style={{ marginTop: 16 }}>
                <Notice kind="warn">
                  <strong>Not supported in V0.</strong> {unsupported}
                </Notice>
              </div>
            )}
            {error && (
              <div style={{ marginTop: 16 }}>
                <Notice kind="err">{error}</Notice>
              </div>
            )}
          </Panel>

          {/* ---------------- configuration / run ---------------- */}
          {runId ? (
            <RunProgress
              runId={runId}
              onDone={(reportId) => router.push(`/report/${reportId}`)}
              onError={(m) => { setRunId(null); setError(m); }}
            />
          ) : spec ? (
            <SpecReview initialSpec={spec} assumptions={assumptions} onRun={run} />
          ) : (
            <Panel title="Configuration" icon={<IconSettings size={15} />}>
              <div className="empty">
                <IconEmptyChart size={34} />
                <p>No strategy loaded.</p>
                <p className="faint">
                  Describe one or import a spec — settings appear here for review
                  before anything runs.
                </p>
              </div>
            </Panel>
          )}
        </div>
        <PageDisclaimer />
      </main>
    </>
  );
}
