"use client";

// The parsed strategy card (Section 1, step 2): labeled human-readable rows
// with edit affordances bound to the DSL's legal values, LLM assumptions in
// amber, a collapsed raw-JSON view, and the hard gate: NOTHING runs until
// "Run the gauntlet".

import { useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";
import type { ValidateResult } from "@/lib/types";

type Spec = Record<string, any>;

const REBALANCE = ["daily", "weekly", "monthly", "quarterly", "annual"];
const WEIGHTING = ["equal", "rank", "inverse_vol"];
const BENCHMARK = ["SPY", "QQQ", "sixty_forty"];
const DIRECTION = ["top", "bottom"];
const UNIVERSE_LABEL: Record<string, string> = {
  sp500: "S&P 500 (current members)",
  nasdaq100: "Nasdaq-100 (current members)",
  dow30: "Dow 30 (current members)",
  sector_etfs: "11 sector ETFs",
  user_watchlist: "your watchlist",
};

export default function SpecReview({
  initialSpec,
  assumptions,
  onRun,
}: {
  initialSpec: Spec;
  assumptions: string[];
  onRun: (spec: Spec) => void;
}) {
  const [spec, setSpec] = useState<Spec>(initialSpec);
  const [validation, setValidation] = useState<ValidateResult | null>(null);
  const [showJson, setShowJson] = useState(false);
  const debounce = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Local telemetry (Gap #11): card views minus confirms = abandonment rate.
  useEffect(() => {
    api("/api/event", { method: "POST", json: { type: "ui_spec_card_shown" } })
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    if (debounce.current) clearTimeout(debounce.current);
    debounce.current = setTimeout(() => {
      api<ValidateResult>("/api/validate-spec", {
        method: "POST",
        json: { spec },
      })
        .then(setValidation)
        .catch(() => setValidation(null));
    }, 350);
    return () => {
      if (debounce.current) clearTimeout(debounce.current);
    };
  }, [spec]);

  const set = (path: string[], value: unknown) => {
    setSpec((prev) => {
      const next = structuredClone(prev);
      let node: any = next;
      for (const key of path.slice(0, -1)) {
        node[key] = node[key] ?? {};
        node = node[key];
      }
      node[path[path.length - 1]] = value;
      return next;
    });
  };

  const sel = spec.selection ?? {};
  const isStatic = Array.isArray(sel.static_allocation);
  const rankBy = sel.rank_by ?? {};
  const valid = validation?.valid ?? false;

  return (
    <div className="card">
      <h2>Confirm what will be tested</h2>
      {validation?.readback && (
        <p className="muted" style={{ fontStyle: "italic" }}>
          “{validation.readback}”
        </p>
      )}
      <p className="faint">
        This readback is generated from the validated spec by template code —
        never by the model that produced it.
      </p>

      {assumptions.map((a) => (
        <p key={a} className="assumption">▲ Assumption — {a}. Correct?</p>
      ))}

      <div className="specrow">
        <span className="k">Name</span>
        <span className="v">
          <input type="text" value={spec.name ?? ""} maxLength={80}
            onChange={(e) => set(["name"], e.target.value)} size={28} />
        </span>
      </div>
      <div className="specrow">
        <span className="k">Universe</span>
        <span className="v">
          {UNIVERSE_LABEL[spec.universe?.preset] ?? spec.universe?.preset}
          {spec.universe?.preset === "user_watchlist" &&
            Array.isArray(spec.universe?.resolved_tickers) && (
              <span className="faint">
                {spec.universe.resolved_tickers.length} tickers:{" "}
                {spec.universe.resolved_tickers.slice(0, 6).join(", ")}
                {spec.universe.resolved_tickers.length > 6 ? "…" : ""}
              </span>
            )}
        </span>
      </div>

      {isStatic ? (
        <div className="specrow">
          <span className="k">Allocation</span>
          <span className="v">
            {sel.static_allocation
              .map((p: any) => `${Math.round(p.weight * 100)}% ${p.ticker}`)
              .join(" · ")}
          </span>
        </div>
      ) : (
        <>
          <div className="specrow">
            <span className="k">Signal</span>
            <span className="v">
              {rankBy.type ?? "—"}
              {"lookback_days" in rankBy && rankBy.lookback_days != null && (
                <>
                  <label className="faint">lookback (days)</label>
                  <input type="number" min={1} max={756}
                    value={rankBy.lookback_days}
                    onChange={(e) =>
                      set(["selection", "rank_by", "lookback_days"],
                        Number(e.target.value))}
                    style={{ width: 76 }} />
                </>
              )}
              {rankBy.skip_days > 0 && (
                <span className="faint">skip {rankBy.skip_days}d</span>
              )}
            </span>
          </div>
          <div className="specrow">
            <span className="k">Selection</span>
            <span className="v">
              <select value={sel.direction ?? "top"}
                onChange={(e) => set(["selection", "direction"], e.target.value)}>
                {DIRECTION.map((d) => <option key={d}>{d}</option>)}
              </select>
              <label className="faint">n =</label>
              <input type="number" min={1} max={50} value={sel.n ?? 5}
                onChange={(e) => set(["selection", "n"], Number(e.target.value))}
                style={{ width: 64 }} />
            </span>
          </div>
          <div className="specrow">
            <span className="k">Weighting</span>
            <span className="v">
              <select value={spec.weighting ?? "equal"}
                onChange={(e) => set(["weighting"], e.target.value)}>
                {WEIGHTING.map((w) => <option key={w}>{w}</option>)}
              </select>
            </span>
          </div>
        </>
      )}

      <div className="specrow">
        <span className="k">Rebalance</span>
        <span className="v">
          <select value={spec.rebalance ?? "monthly"}
            onChange={(e) => set(["rebalance"], e.target.value)}>
            {REBALANCE.map((r) => <option key={r}>{r}</option>)}
          </select>
          {spec.entry_exit?.hold_days != null && (
            <>
              <label className="faint">hold (days)</label>
              <input type="number" min={1} max={252}
                value={spec.entry_exit.hold_days}
                onChange={(e) =>
                  set(["entry_exit", "hold_days"], Number(e.target.value))}
                style={{ width: 64 }} />
            </>
          )}
        </span>
      </div>
      <div className="specrow">
        <span className="k">Window</span>
        <span className="v">
          <input type="date" value={spec.window?.start ?? ""}
            onChange={(e) => set(["window", "start"], e.target.value)} />
          →
          <input type="date" value={spec.window?.end ?? ""}
            onChange={(e) => set(["window", "end"], e.target.value)} />
        </span>
      </div>
      <div className="specrow">
        <span className="k">Benchmark</span>
        <span className="v">
          <select value={spec.benchmark ?? "SPY"}
            onChange={(e) => set(["benchmark"], e.target.value)}>
            {BENCHMARK.map((b) => (
              <option key={b} value={b}>
                {b === "sixty_forty" ? "60/40 SPY+AGG" : b}
              </option>
            ))}
          </select>
          <span className="faint">always shown — never omittable</span>
        </span>
      </div>
      <div className="specrow">
        <span className="k">Costs</span>
        <span className="v">
          0 / 5 / 20 bps sweep <span className="faint">(fixed product rule)</span>
        </span>
      </div>

      {validation?.warnings?.map((w) => (
        <p key={w} className="warnbox">▲ {w}</p>
      ))}
      {validation && !validation.valid &&
        validation.errors.map((e) => <p key={e} className="errbox">✕ {e}</p>)}

      <div style={{ display: "flex", gap: "0.7rem", alignItems: "center",
                    marginTop: "0.9rem", flexWrap: "wrap" }}>
        <button className="primary" disabled={!valid}
          onClick={() => onRun(validation?.expanded_spec ?? spec)}>
          Run the gauntlet
        </button>
        <button onClick={() => setShowJson(!showJson)}>
          {showJson ? "Hide" : "View"} raw spec (JSON)
        </button>
        {validation?.universe_size ? (
          <span className="faint">{validation.universe_size} tickers in universe</span>
        ) : null}
      </div>
      {showJson && (
        <pre className="scroll-x" style={{ fontSize: "0.78rem" }}>
          {JSON.stringify(validation?.expanded_spec ?? spec, null, 2)}
        </pre>
      )}
    </div>
  );
}
