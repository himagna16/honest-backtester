// Shapes returned by the backend (see backend/audit/pipeline.py).

export interface Metrics {
  total_return: number;
  cagr: number;
  vol: number;
  sharpe: number;
  max_drawdown: number;
  days: number;
  hit_rate?: number;
  annual_turnover?: number;
}

export interface Verdict {
  color: "red" | "amber" | "green";
  icon: string;
  headline: string;
  sub: string;
}

export interface AuditFinding {
  check: string;
  severity: "fatal" | "warning" | "note" | "pass";
  name: string;
  message: string;
  evidence: Record<string, unknown>;
  action: string | null;
  explainer: string | null;
}

export interface CostSweepRow {
  bps: number | null;
  label?: string;
  total_return: number;
  cagr: number;
  sharpe: number;
  max_drawdown: number;
  beats_benchmark: boolean | null;
}

export interface OOSSegment extends Metrics {
  benchmark_total_return: number;
  benchmark_sharpe: number;
  beats_benchmark: boolean;
  start: string;
  end: string;
}

export interface RegimeSegment {
  label: string;
  start: string;
  end: string;
  kind: "bull" | "bear" | "correction" | "sideways" | "down";
}

/** Every inference payload can refuse. `available: false` always travels with
 *  a reason — a refusal rendered as an omission reads as a pass. */
interface Refusable {
  available: boolean;
  unavailable_reason: string | null;
}

export interface SharpeSignificance extends Refusable {
  n_obs: number;
  interpretation: string;
  /** Probabilistic Sharpe: P(true Sharpe > 0). */
  psr: { psr: number | null; sharpe_annual: number } | null;
  /** Deflated Sharpe: P(true Sharpe beats what `n_trials` of luck produce).
   *  Needs the SPREAD of prior trial Sharpes, not just the count — without a
   *  trial ledger it reports itself unavailable rather than guessing. */
  dsr: {
    dsr: number | null;
    n_trials: number;
    sr0_annual: number | null;
    available: boolean;
    interpretation: string;
  } | null;
  min_trl: Record<string, unknown> | null;
  min_btl: Record<string, unknown> | null;
  bootstrap: Record<string, unknown> | null;
}

export interface ExcessSignificance extends Refusable {
  /** Three states plus refusal — "ties_within_noise" is a real answer. */
  verdict: "beats" | "ties_within_noise" | "loses";
  verdict_source: "bootstrap" | "hac" | "none";
  n_obs: number;
  interpretation: string;
  /** How far short of nominal the interval behind the verdict actually
   *  covers. Render it NEXT TO the verdict — it is the honest half. */
  ci_coverage_caveat: string | null;
}

/** Bars by which a training observation can still reach a holdout one. The
 *  70/30 cut is a plain date split, so this is how much of it leaks. */
export interface HoldoutPurge {
  signal_reach: number;
  staleness_reach: number;
  extra_reach: number;
  reach: number;
  carry: number;
  execution_lag: number;
  cost_charge: number;
  total: number;
}

export interface Report {
  verdict: Verdict;
  metrics: Metrics;
  gross_metrics: Metrics;
  net20_metrics: Metrics;
  benchmark: Metrics & { name: string; label: string };
  cost_sweep: CostSweepRow[];
  audit: AuditFinding[];
  oos: { split_date: string; in_sample: OOSSegment; out_of_sample: OOSSegment };
  sensitivity: { variant: string; net5_total_return: number; beats_benchmark: boolean }[];
  regimes: { segments: RegimeSegment[]; has_bear_market: boolean };
  neutral_rerun: {
    universe: string; net5_total_return: number; user_net5_total_return: number;
    wealth_ratio: number; start: string; end: string;
  } | null;
  multiple_testing: { universe_key: string; attempts: number };
  significance?: {
    sharpe: SharpeSignificance;
    excess: ExcessSignificance;
    seed: number;
  };
  attribution?: Record<string, unknown> & {
    available: boolean;
    unavailable_reason?: string | null;
  };
  holdout_purge?: HoldoutPurge | null;
  equity_curve: [string, number | null, number | null, number | null][];
  drawdown_curve: [string, number | null, number | null][];
  data: {
    as_of: string; stale: boolean; missing: string[]; notes: string[];
    universe_size: number; effective_start: string; effective_end: string;
    requested_start: string | null; requested_end: string | null; rf_source: string;
  };
  spec: Record<string, unknown>;
  readback: string;
  disclaimer: string;
  holdings_recent: {
    date: string;
    eligible: number;
    picks: { ticker: string; weight: number; score: number | null }[];
  }[];
  run?: { id: string; snapshot_hash: string; data_source: string; replay_of: string | null };
}

export interface ReportEnvelope {
  id: string;
  run_id: string;
  created_at: string;
  report: Report;
}

export interface TranslateResult {
  status: "spec" | "needs_clarification" | "unsupported" | "error";
  spec?: Record<string, unknown>;
  readback?: string;
  assumptions?: string[];
  warnings?: string[];
  questions?: { question: string; options: string[] }[];
  unsupported_reason?: string;
  errors?: string[];
  message?: string;
  cached?: boolean;
}

export interface ValidateResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
  expanded_spec?: Record<string, unknown>;
  readback?: string;
  universe_size?: number;
}
