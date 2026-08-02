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
