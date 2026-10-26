// Derives risk badges, evidence strength, stress cells, and the research-note
// buckets — ALL from real report fields. Nothing here is inferred or invented;
// each item cites the finding or metric it came from.

import type { AuditFinding, Report } from "./types";

export interface RiskBadge {
  label: string;
  tone: "warn" | "neg" | "info";
  why: string;
}

const sev = (r: Report, check: string) =>
  r.audit.find((f) => f.check === check)?.severity;

export function riskBadges(r: Report): RiskBadge[] {
  const out: RiskBadge[] = [];
  const dd = r.metrics.max_drawdown;
  if (dd <= -0.3)
    out.push({
      label: "High drawdown risk",
      tone: dd <= -0.5 ? "neg" : "warn",
      why: `Worst peak-to-trough loss was ${(dd * 100).toFixed(0)}%.`,
    });
  if (!r.regimes.has_bear_market)
    out.push({
      label: "Regime-sensitive",
      tone: "warn",
      why: "The test window contains no sustained bear market.",
    });
  if (sev(r, "parameter_sensitivity") === "warning")
    out.push({
      label: "Parameter-fragile",
      tone: "warn",
      why: "A one-step parameter change flips the result.",
    });
  if (r.metrics.days < 504)
    out.push({
      label: "Low sample size",
      tone: "warn",
      why: `Only ${r.metrics.days} trading days of simulation.`,
    });
  if (sev(r, "overfitting") === "warning")
    out.push({
      label: "Overfit risk",
      tone: "warn",
      why: "Performs materially better in-sample than on the holdout.",
    });
  if (sev(r, "cost_sweep") === "warning")
    out.push({
      label: "Cost-fragile",
      tone: "warn",
      why: "The edge does not survive realistic trading costs.",
    });
  if (sev(r, "survivorship_universe") === "warning")
    out.push({
      label: r.data.universe_membership?.point_in_time ? "Index history gaps" : "Today's index list",
      tone: "warn",
      why: r.data.universe_membership?.point_in_time
        ? "Some past index members have no free price history and could not be picked."
        : "The index universe is today's members, including companies that joined after their run-ups.",
    });
  const bias = sev(r, "selection_bias");
  if (bias === "fatal" || bias === "warning")
    out.push({
      label: "Hindsight universe",
      tone: bias === "fatal" ? "neg" : "warn",
      why: "Tickers were hand-picked knowing how they performed.",
    });
  if (r.multiple_testing.attempts > 3)
    out.push({
      label: `Attempt #${r.multiple_testing.attempts}`,
      tone: "info",
      why: "Repeated tests against the same data inflate luck.",
    });
  return out;
}

/** Evidence strength — sample size and regime coverage, not model confidence. */
export function evidenceStrength(r: Report): { level: 1 | 2 | 3; label: string } {
  const days = r.metrics.days;
  const bear = r.regimes.has_bear_market;
  if (days >= 1260 && bear)
    return { level: 3, label: `Strong evidence · ${Math.round(days / 252)}y incl. a bear market` };
  if (days >= 504)
    return {
      level: 2,
      label: `Moderate evidence · ${Math.round(days / 252)}y${bear ? " incl. a bear market" : ", no bear market"}`,
    };
  return { level: 1, label: `Thin evidence · ${days} trading days` };
}

export interface StressCell {
  k: string;
  v: string;
  state: "survived" | "failed" | "untested";
}

export function stressCells(r: Report): StressCell[] {
  const byBps = new Map(r.cost_sweep.filter((c) => c.bps !== null).map((c) => [c.bps, c]));
  const at = (bps: number) => byBps.get(bps);
  const cell = (k: string, ok: boolean | null | undefined): StressCell =>
    ok === null || ok === undefined
      ? { k, v: "not tested", state: "untested" }
      : { k, v: ok ? "survived" : "failed", state: ok ? "survived" : "failed" };

  return [
    cell("5 bps costs", at(5)?.beats_benchmark),
    cell("20 bps costs", at(20)?.beats_benchmark),
    cell("30% holdout", r.oos.out_of_sample.beats_benchmark),
    r.regimes.has_bear_market
      ? { k: "Bear market", v: "in window", state: "survived" }
      : { k: "Bear market", v: "never tested", state: "untested" },
    r.sensitivity.length === 0
      ? { k: "Parameter shift", v: "not tested", state: "untested" }
      : cell(
          "Parameter shift",
          r.sensitivity.every((s) => s.beats_benchmark === r.sensitivity[0].beats_benchmark),
        ),
  ];
}

export interface NoteBuckets {
  worked: string[];
  failed: string[];
  risks: string[];
  next: { text: string; check: string }[];
}

/** Structures the deterministic audit into a research-note shape. */
export function noteBuckets(r: Report): NoteBuckets {
  const worked: string[] = [];
  const failed: string[] = [];
  const risks: string[] = [];
  const next: { text: string; check: string }[] = [];

  const pct = (v: number) => `${v >= 0 ? "+" : ""}${(v * 100).toFixed(1)}%`;
  const gross = r.cost_sweep.find((c) => c.bps === 0);
  const net20 = r.cost_sweep.find((c) => c.bps === 20);

  if (gross?.beats_benchmark)
    worked.push(
      `Beat ${r.benchmark.label} before costs: ${pct(gross.total_return)} vs ${pct(r.benchmark.total_return)}.`,
    );
  if (net20?.beats_benchmark)
    worked.push(`Still ahead at 20 bps — the pessimistic cost column: ${pct(net20.total_return)}.`);
  if (r.oos.out_of_sample.beats_benchmark)
    worked.push(
      `Held up on the 30% holdout (${r.oos.out_of_sample.start} →): ${pct(r.oos.out_of_sample.total_return)}.`,
    );
  if (r.metrics.sharpe > r.benchmark.sharpe)
    worked.push(
      `Better risk-adjusted return: Sharpe ${r.metrics.sharpe.toFixed(2)} vs ${r.benchmark.sharpe.toFixed(2)}.`,
    );

  for (const f of r.audit) {
    if (f.severity === "fatal") failed.push(`${f.name}. ${f.message}`);
    else if (f.severity === "warning") risks.push(`${f.name}. ${f.message}`);
    if (f.action) next.push({ text: f.action, check: f.check });
  }
  if (!gross?.beats_benchmark)
    failed.push(
      `Trailed ${r.benchmark.label} even before costs (${pct(gross?.total_return ?? 0)} vs ${pct(r.benchmark.total_return)}) — costs were never the problem.`,
    );
  if (!r.oos.out_of_sample.beats_benchmark && gross?.beats_benchmark)
    failed.push(
      `Lost to the benchmark on the holdout segment, which is the cell the verdict uses.`,
    );

  if (worked.length === 0) worked.push("No test in the gauntlet came out in the strategy's favour.");
  if (failed.length === 0) failed.push("No blocker-level finding fired.");
  if (risks.length === 0) risks.push("No warning-level finding fired.");
  if (next.length === 0)
    next.push({
      text: "Re-run on a longer window, or vary one parameter to probe robustness.",
      check: "",
    });
  return { worked, failed, risks, next };
}

export function findingIsActionable(f: AuditFinding): boolean {
  return Boolean(f.action);
}
