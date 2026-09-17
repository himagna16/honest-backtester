"use client";

// LiveDemo — a self-playing replay of report rep_2c6332f417, a 126-day-lookback
// War-list run (not the founding-story backtest; the landing caption says so).
// Four scenes on a loop: the strategy is typed, read back as a spec,
// audited, and stamped with the amber verdict. Pure React + CSS; every number
// comes from ./demoData, which is transcribed verbatim from the exported run.
//
// Motion contract: one rAF clock drives a declarative timeline; everything on
// screen is derived from a single elapsed-ms value, so pause, scene jumps and
// tab-visibility loss cannot drift. All reveals are opacity/transform only and
// are gated on data-ready="1" (set by JS after mount) — with no JS, and under
// prefers-reduced-motion, the component renders Scene 4's final verdict frame,
// which is the information-complete state.

import { useCallback, useEffect, useRef, useState } from "react";
import {
  APPROVAL_NOTE,
  AUDIT_CHECKS_TOTAL,
  AUDIT_ROWS,
  END_LINE,
  EQUITY_CURVE,
  EQUITY_WINDOW,
  FINAL_MULTIPLES,
  METRIC_PAIRS,
  NEUTRAL_RERUN,
  READBACK_EXCERPT,
  REPORT_ID,
  SPEC_CHIPS,
  TYPED_STRATEGY,
  VERDICT,
} from "./demoData";

/* ---------------------------------------------------------------- timeline */

// Per-character reveal times with deterministic jitter (45–60 ms, a longer
// beat after punctuation). A seeded LCG keeps server and client output
// identical, so there is nothing random to mismatch on hydration.
const CHAR_AT: readonly number[] = (() => {
  let seed = 0x2c63;
  const rand = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x7fffffff;
  };
  const out: number[] = [];
  let t = 0;
  for (const ch of TYPED_STRATEGY) {
    t += 45 + Math.round(rand() * 15);
    out.push(t);
    if (ch === "." || ch === ",") t += 160;
  }
  return out;
})();

const TYPE_END = CHAR_AT[CHAR_AT.length - 1];
const ROW_STEP = 480; // audit rows land every 480 ms
const ROW_LEAD = 350; // first row after the scene settles
const ROW_COUNT = AUDIT_ROWS.length + 1; // + the neutral re-run sub-row

// Scene lengths (ms). Holds are the reading pauses between scenes; the
// verdict holds ~5 s before the loop restarts.
const LEN = [
  TYPE_END + 2300,
  4300,
  ROW_LEAD + (ROW_COUNT - 1) * ROW_STEP + 2150,
  6800,
] as const;
const START = [0, LEN[0], LEN[0] + LEN[1], LEN[0] + LEN[1] + LEN[2]] as const;
const TOTAL = START[3] + LEN[3];

// Scene 4 beat offsets.
const V_STAMP = 60;
const V_METRICS = 340;
const V_SPARK = 700;
const V_ENDLINE = 1700;

const SCENE_LABELS = ["Describe", "Readback", "Audit", "Verdict"] as const;

interface Frame {
  scene: 0 | 1 | 2 | 3;
  typed: number; // characters visible in scene 1
  cardIn: boolean; // scene 2
  chips: number; // scene 2, 0..SPEC_CHIPS.length
  noteIn: boolean; // scene 2
  rows: number; // scene 3, 0..ROW_COUNT
  stamp: boolean; // scene 4 …
  metrics: boolean;
  spark: boolean;
  endline: boolean;
}

const FINAL_FRAME: Frame = {
  scene: 3,
  typed: TYPED_STRATEGY.length,
  cardIn: true,
  chips: SPEC_CHIPS.length,
  noteIn: true,
  rows: ROW_COUNT,
  stamp: true,
  metrics: true,
  spark: true,
  endline: true,
};

function typedCount(t: number): number {
  let n = 0;
  while (n < CHAR_AT.length && CHAR_AT[n] <= t) n += 1;
  return n;
}

function frameAt(t: number): Frame {
  if (t >= START[3]) {
    const r = t - START[3];
    return {
      ...FINAL_FRAME,
      stamp: r >= V_STAMP,
      metrics: r >= V_METRICS,
      spark: r >= V_SPARK,
      endline: r >= V_ENDLINE,
    };
  }
  if (t >= START[2]) {
    const r = t - START[2];
    const rows =
      r < ROW_LEAD
        ? 0
        : Math.min(ROW_COUNT, 1 + Math.floor((r - ROW_LEAD) / ROW_STEP));
    return { ...FINAL_FRAME, scene: 2, rows, stamp: false, metrics: false, spark: false, endline: false };
  }
  if (t >= START[1]) {
    const r = t - START[1];
    const chips =
      r < 640
        ? 0
        : Math.min(SPEC_CHIPS.length, 1 + Math.floor((r - 640) / 110));
    return {
      scene: 1,
      typed: TYPED_STRATEGY.length,
      cardIn: r >= 80,
      chips,
      noteIn: r >= 1500,
      rows: 0,
      stamp: false,
      metrics: false,
      spark: false,
      endline: false,
    };
  }
  return {
    scene: 0,
    typed: typedCount(t),
    cardIn: false,
    chips: 0,
    noteIn: false,
    rows: 0,
    stamp: false,
    metrics: false,
    spark: false,
    endline: false,
  };
}

// Complete (end-of-scene) frames, used for reduced-motion scene browsing —
// each scene's finished state is static and information-complete.
function completeFrame(scene: Frame["scene"]): Frame {
  const f: Frame = {
    ...FINAL_FRAME,
    scene,
    stamp: scene === 3,
    metrics: scene === 3,
    spark: scene === 3,
    endline: scene === 3,
  };
  if (scene < 1) {
    f.cardIn = false;
    f.chips = 0;
    f.noteIn = false;
  }
  if (scene < 2) f.rows = 0;
  return f;
}

function sameFrame(a: Frame, b: Frame): boolean {
  return (
    a.scene === b.scene &&
    a.typed === b.typed &&
    a.cardIn === b.cardIn &&
    a.chips === b.chips &&
    a.noteIn === b.noteIn &&
    a.rows === b.rows &&
    a.stamp === b.stamp &&
    a.metrics === b.metrics &&
    a.spark === b.spark &&
    a.endline === b.endline
  );
}

/* ------------------------------------------------------------ equity spark */

// Both series drawn on a log scale so each keeps its true shape — a linear
// axis at ×19 would flatten SPY into a lie of a line. Geometry is computed
// once at module scope from the transcribed curve; it is fully deterministic.
const SPARK_W = 640;
const SPARK_H = 150;
const PAD = { top: 10, right: 10, bottom: 10, left: 10 } as const;

interface Spark {
  strategy: string;
  spy: string;
  baselineY: number;
}

const SPARK: Spark = (() => {
  let min = Infinity;
  let max = -Infinity;
  for (const [, s, b] of EQUITY_CURVE) {
    min = Math.min(min, s, b);
    max = Math.max(max, s, b);
  }
  const lnMin = Math.log(min);
  const lnMax = Math.log(max);
  const x = (i: number) =>
    PAD.left + (i / (EQUITY_CURVE.length - 1)) * (SPARK_W - PAD.left - PAD.right);
  const y = (v: number) =>
    SPARK_H -
    PAD.bottom -
    ((Math.log(v) - lnMin) / (lnMax - lnMin)) * (SPARK_H - PAD.top - PAD.bottom);
  const path = (pick: (p: (typeof EQUITY_CURVE)[number]) => number) =>
    EQUITY_CURVE.map(
      (p, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)} ${y(pick(p)).toFixed(1)}`,
    ).join(" ");
  return {
    strategy: path((p) => p[1]),
    spy: path((p) => p[2]),
    baselineY: y(1),
  };
})();

/* -------------------------------------------------------------- component */

export default function LiveDemo() {
  // SSR and the first client render show the finished verdict frame; the
  // animation only takes over once JS flips `ready` — progressive
  // enhancement, not hide-then-hope.
  const [frame, setFrame] = useState<Frame>(FINAL_FRAME);
  const [ready, setReady] = useState(false);
  const [reduced, setReduced] = useState(false);
  const [paused, setPaused] = useState(false);

  const elapsedRef = useRef(0);
  const lastRef = useRef(0);
  const frameRef = useRef<Frame>(FINAL_FRAME);
  const pausedRef = useRef(false);
  const hoverRef = useRef(false);
  const hiddenRef = useRef(false);
  const hoverableRef = useRef(false);
  const railRef = useRef<Array<HTMLElement | null>>([]);

  const commit = useCallback((f: Frame) => {
    if (!sameFrame(frameRef.current, f)) {
      frameRef.current = f;
      setFrame(f);
    }
  }, []);

  const paintRail = useCallback((scene: number, frac: number) => {
    for (let i = 0; i < SCENE_LABELS.length; i += 1) {
      const el = railRef.current[i];
      if (el) {
        const v = i < scene ? 1 : i > scene ? 0 : Math.min(1, Math.max(0, frac));
        el.style.transform = `scaleX(${v})`;
      }
    }
  }, []);

  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    setReduced(mq.matches);
    const onChange = (e: MediaQueryListEvent) => setReduced(e.matches);
    mq.addEventListener("change", onChange);
    hoverableRef.current = window.matchMedia("(hover: hover)").matches;
    setReady(true);
    return () => mq.removeEventListener("change", onChange);
  }, []);

  useEffect(() => {
    if (!ready) return undefined;

    if (reduced) {
      // Land instantly in the final state; the rail still works as a static
      // scene switcher via jump().
      elapsedRef.current = TOTAL - 1;
      commit(FINAL_FRAME);
      paintRail(3, 1);
      return undefined;
    }

    elapsedRef.current = 0;
    lastRef.current = performance.now();
    hiddenRef.current = document.hidden;

    const onVisibility = () => {
      hiddenRef.current = document.hidden;
      // Re-baseline the clock so time hidden is not replayed on return.
      lastRef.current = performance.now();
    };
    document.addEventListener("visibilitychange", onVisibility);

    let raf = 0;
    const step = (now: number) => {
      // Clamp dt so a throttled background tab cannot leap the timeline.
      const dt = Math.min(Math.max(now - lastRef.current, 0), 100);
      lastRef.current = now;
      if (!pausedRef.current && !hoverRef.current && !hiddenRef.current) {
        elapsedRef.current = (elapsedRef.current + dt) % TOTAL;
      }
      const t = elapsedRef.current;
      const f = frameAt(t);
      commit(f);
      paintRail(f.scene, (t - START[f.scene]) / LEN[f.scene]);
      raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);

    return () => {
      cancelAnimationFrame(raf);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [ready, reduced, commit, paintRail]);

  const jump = (scene: Frame["scene"]) => {
    if (reduced) {
      commit(completeFrame(scene));
      paintRail(scene, 1);
      return;
    }
    // Jumping means "show me that scene", so it also resumes playback —
    // a paused jump would land on the scene's empty first frame.
    pausedRef.current = false;
    setPaused(false);
    elapsedRef.current = START[scene];
    commit(frameAt(START[scene])); // immediate, not next tick
  };

  const togglePause = () => {
    const next = !pausedRef.current;
    pausedRef.current = next;
    setPaused(next);
  };

  const sceneHidden = (i: number) => frame.scene !== i;

  return (
    <section
      className="bs-demo"
      data-ready={ready ? "1" : "0"}
      aria-label={`Self-playing replay of backtest run ${REPORT_ID}: a plain-English strategy is read back as a spec, audited, and stamped with an amber verdict — beats the benchmark, but the universe is hand-picked.`}
    >
      {/* Everything the animation shows, in one static sentence for screen
          readers; the ticking scenes below are decorative repetition. */}
      <p className="bs-demo-sr">
        The strategy “{TYPED_STRATEGY}” is read back as a spec —{" "}
        {SPEC_CHIPS.join(", ").toLowerCase()} — and audited: five checks pass,
        one warns that the 44-ticker universe is hand-picked.{" "}
        {NEUTRAL_RERUN.lead}
        {NEUTRAL_RERUN.sp500}
        {NEUTRAL_RERUN.mid}
        {NEUTRAL_RERUN.user}
        {NEUTRAL_RERUN.tail} Verdict: amber — {VERDICT.headline} Strategy
        versus SPY buy-and-hold:{" "}
        {METRIC_PAIRS.map((m) => `${m.label} ${m.strategy} vs ${m.spy}`).join(
          ", ",
        )}
        .
      </p>

      <div className="chrome" aria-hidden="true">
        <i />
        <i />
        <i />
        <span className="title">Replay · {REPORT_ID}</span>
      </div>

      <div className="screen">
        <div
          className="stage"
          aria-live="off"
          onMouseEnter={() => {
            if (hoverableRef.current) hoverRef.current = true;
          }}
          onMouseLeave={() => {
            hoverRef.current = false;
          }}
        >
          {/* ------------------------------------------------ 1 · describe */}
          <div
            className={`ld-scene${frame.scene === 0 ? " is-active" : ""}`}
            aria-hidden={sceneHidden(0)}
          >
            <p className="ld-kicker">01 · Describe</p>
            <div className="ld-typeline">
              {/* Invisible full text reserves the wrapped height, so typing
                  never reflows anything. */}
              <span className="ghost" aria-hidden="true">
                {TYPED_STRATEGY}
              </span>
              <p className="live" aria-hidden="true">
                {TYPED_STRATEGY.slice(0, frame.typed)}
                <span className="ld-caret" />
              </p>
            </div>
          </div>

          {/* ------------------------------------------------ 2 · readback */}
          <div
            className={`ld-scene${frame.scene === 1 ? " is-active" : ""}`}
            aria-hidden={sceneHidden(1)}
          >
            <p className="ld-kicker">02 · Readback</p>
            <div className={`ld-card${frame.cardIn ? " is-in" : ""}`}>
              <p className="ld-card-label">
                Deterministic spec — exactly what will run
              </p>
              <p className="ld-readback">{READBACK_EXCERPT}</p>
              <ul className="ld-chips">
                {SPEC_CHIPS.map((chip, i) => (
                  <li key={chip} className={i < frame.chips ? "is-in" : ""}>
                    {chip}
                  </li>
                ))}
              </ul>
            </div>
            <p className={`ld-note${frame.noteIn ? " is-in" : ""}`}>
              {APPROVAL_NOTE}
            </p>
          </div>

          {/* --------------------------------------------------- 3 · audit */}
          <div
            className={`ld-scene${frame.scene === 2 ? " is-active" : ""}`}
            aria-hidden={sceneHidden(2)}
          >
            <p className="ld-kicker">
              03 · The audit argues — {AUDIT_ROWS.length} of{" "}
              {AUDIT_CHECKS_TOTAL} checks shown
            </p>
            <ul className="ld-audit">
              {AUDIT_ROWS.map((row, i) => (
                <li
                  key={row.name}
                  className={`${row.severity}${i < frame.rows ? " is-in" : ""}`}
                >
                  <span className="mark" aria-hidden="true">
                    {row.severity === "warning" ? "▲" : "✓"}
                  </span>
                  <span className="name">{row.name}</span>
                  <span className="tag">
                    {row.severity === "warning" ? "Warning" : "Pass"}
                  </span>
                </li>
              ))}
              <li
                className={`sub${frame.rows === ROW_COUNT ? " is-in" : ""}`}
              >
                {NEUTRAL_RERUN.lead}
                <strong>{NEUTRAL_RERUN.sp500}</strong>
                {NEUTRAL_RERUN.mid}
                <strong>{NEUTRAL_RERUN.user}</strong>
                {NEUTRAL_RERUN.tail}
              </li>
            </ul>
          </div>

          {/* ------------------------------------------------- 4 · verdict */}
          <div
            className={`ld-scene${frame.scene === 3 ? " is-active" : ""}`}
            aria-hidden={sceneHidden(3)}
          >
            <p className="ld-kicker">04 · Verdict</p>
            <p className={`ld-stamp${frame.stamp ? " is-in" : ""}`}>
              <span aria-hidden="true">▲ </span>
              {VERDICT.stamp}
            </p>
            <p className={`ld-verdict${frame.stamp ? " is-in" : ""}`}>
              {VERDICT.headline}
            </p>

            <dl className={`ld-metrics${frame.metrics ? " is-in" : ""}`}>
              {METRIC_PAIRS.map((m) => (
                <div className="m" key={m.label}>
                  <dt>{m.label}</dt>
                  <dd className="v">{m.strategy}</dd>
                  <dd className="vs">SPY {m.spy}</dd>
                </div>
              ))}
            </dl>

            <figure className={`ld-spark${frame.spark ? " is-drawn" : ""}`}>
              <svg
                viewBox={`0 0 ${SPARK_W} ${SPARK_H}`}
                role="img"
                aria-label={`Equity curves on a log scale, ${EQUITY_WINDOW}: the strategy ends at ${FINAL_MULTIPLES.strategy} starting wealth, SPY buy-and-hold at ${FINAL_MULTIPLES.spy}.`}
              >
                <line
                  className="base"
                  x1={PAD.left}
                  y1={SPARK.baselineY}
                  x2={SPARK_W - PAD.right}
                  y2={SPARK.baselineY}
                  vectorEffect="non-scaling-stroke"
                />
                {/* No vector-effect here: non-scaling-stroke makes Chrome
                    compute dashes in screen space, which breaks the
                    pathLength=1 draw-on normalisation (verified). */}
                <path className="spy" d={SPARK.spy} pathLength={1} />
                <path className="strat" d={SPARK.strategy} pathLength={1} />
              </svg>
              <figcaption className="ld-legend">
                <span>
                  <i className="sw strat" aria-hidden="true" />
                  Strategy {FINAL_MULTIPLES.strategy}
                </span>
                <span>
                  <i className="sw spy" aria-hidden="true" />
                  SPY {FINAL_MULTIPLES.spy}
                </span>
                <span className="dim">log scale · {EQUITY_WINDOW}</span>
              </figcaption>
            </figure>

            <p className={`ld-endline${frame.endline ? " is-in" : ""}`}>
              {END_LINE}
            </p>
          </div>
        </div>

        {/* ------------------------------------------------------ controls */}
        <div className="ld-controls">
          <ol className="ld-rail">
            {SCENE_LABELS.map((label, i) => (
              <li key={label}>
                <button
                  type="button"
                  onClick={() => jump(i as Frame["scene"])}
                  aria-current={frame.scene === i ? "step" : undefined}
                  aria-label={`Go to scene ${i + 1}: ${label}`}
                >
                  <span className="row">
                    <span className="n" aria-hidden="true">
                      0{i + 1}
                    </span>
                    <span className="lbl">{label}</span>
                  </span>
                  <span className="track" aria-hidden="true">
                    <i
                      ref={(el) => {
                        railRef.current[i] = el;
                      }}
                    />
                  </span>
                </button>
              </li>
            ))}
          </ol>
          {!reduced && (
            <button
              type="button"
              className="ld-pp"
              onClick={togglePause}
              aria-label={paused ? "Play the replay" : "Pause the replay"}
            >
              {paused ? (
                <svg viewBox="0 0 12 12" aria-hidden="true">
                  <path d="M3 1.5 10.5 6 3 10.5Z" fill="currentColor" />
                </svg>
              ) : (
                <svg viewBox="0 0 12 12" aria-hidden="true">
                  <path
                    d="M3.5 1.5v9M8.5 1.5v9"
                    stroke="currentColor"
                    strokeWidth="2"
                  />
                </svg>
              )}
            </button>
          )}
        </div>
      </div>
    </section>
  );
}
