// Data for the landing-page live demo. Every value below is read from report
// rep_9534f1bae0 (War list, 126-day lookback, mid-2024 to May 2026; a separate run
// from the founding-story backtest) and nothing else. It is the 2026-09-17
// re-run of rep_2c6332f417 against the point-in-time S&P 500
// (docs/pit_rerun_log.md): the old report compared the list with TODAY's
// S&P 500 members, a 1.81× gap that stayed under the 2× bar. Raw JSON values
// are quoted in comments next to each display string so drift is auditable at
// a glance. If a number here cannot be traced to that report, it is a defect.

export const REPORT_ID = "rep_9534f1bae0";

/** Scene 1 — the plain-English strategy, typed character by character. */
export const TYPED_STRATEGY =
  "Each month, buy the 5 strongest stocks in my 44-ticker watchlist by " +
  "6-month return. Equal weight.";

/**
 * Scene 2 — the deterministic readback, first two sentences verbatim from the
 * report's `readback` field.
 */
export const READBACK_EXCERPT =
  "At the end of every month, rank your 44-ticker watchlist (COHR, HLIT, " +
  "APLD, QUBT, BE…) by total return over the past 126 trading days " +
  "(≈6 months) and hold the 5 with the highest values, equally weighted. " +
  "Test window: 2024-06-03 to 2026-05-29.";

/** Spec chips: spec_selection {direction top, n 5, lookback_days 126}, spec_rebalance "monthly", equal weight, benchmark SPY. */
export const SPEC_CHIPS = [
  "Top 5",
  "126-day return",
  "Monthly rebalance",
  "Equal weight",
  "vs SPY",
] as const;

export const APPROVAL_NOTE = "Nothing runs until this is approved.";

/**
 * Scene 3 — audit rows shown in the replay, names and severities verbatim
 * from the report's `audit` array (6 of its 17 checks; passes first, the
 * fatal finding lands last).
 */
export type AuditSeverity = "pass" | "warning" | "fatal";
export interface AuditRow {
  name: string;
  severity: AuditSeverity;
}
export const AUDIT_ROWS: readonly AuditRow[] = [
  { name: "Lookahead bias: impossible by construction", severity: "pass" },
  { name: "Survives the cost sweep", severity: "pass" },
  { name: "Train/holdout comparison", severity: "pass" },
  { name: "Bear-market coverage", severity: "pass" },
  { name: "Parameter sensitivity", severity: "pass" },
  { name: "Selection bias detected", severity: "fatal" },
] as const;

export const AUDIT_CHECKS_TOTAL = 17; // length of the report's `audit` array

/**
 * The neutral re-run sub-row under the fatal finding. From `neutral_rerun`
 * (point_in_time true):
 * net5_total_return 3.635296264392645 -> "+363.5%"
 * user_net5_total_return 16.465923708174397 -> "+1646.6%"
 * wealth_ratio 3.7680274812947534 -> "3.77×"
 */
export const NEUTRAL_RERUN = {
  lead: "Same rules, the S&P 500 as it stood each month: ",
  sp500: "+363.5%",
  mid: " vs your ",
  user: "+1646.6%",
  tail: ". Hand-picking alone: 3.77× in final wealth.",
} as const;

/** Scene 4 — verdict, from the report's `verdict` block (color "red", icon "✕"). */
export const VERDICT = {
  stamp: "Killed — red verdict",
  headline: "Result not trustworthy.",
  sub: "Selection bias detected — Re-run this spec on the full S&P 500",
} as const;

/**
 * The metrics row: strategy (metrics_net5) vs SPY buy-and-hold (benchmark).
 * total_return 16.465923708174397 vs 0.4192764536479263
 * cagr         3.489009953608134 vs 0.20180823015722082
 * sharpe       2.027315112087373 vs 0.9238968729961726
 * max_drawdown -0.612254748042193 vs -0.18755225458981672
 */
export interface MetricPair {
  label: string;
  strategy: string;
  spy: string;
}
export const METRIC_PAIRS: readonly MetricPair[] = [
  { label: "Total return", strategy: "+1646.6%", spy: "+41.9%" },
  { label: "CAGR", strategy: "+348.9%", spy: "+20.2%" },
  { label: "Sharpe", strategy: "2.03", spy: "0.92" },
  { label: "Max drawdown", strategy: "−61.2%", spy: "−18.8%" },
] as const;

export const END_LINE =
  "Benchmark shown on every result. No way to turn the audit off.";

/**
 * Equity spark labels — final wealth multiples from the last equity_curve
 * point ["2026-05-29", 17.465924, 17.629524, 1.419276].
 */
export const FINAL_MULTIPLES = { strategy: "×17.5", spy: "×1.42" } as const;

/** First and last dates of the equity curve itself. */
export const EQUITY_WINDOW = "2024-07-01 → 2026-05-29";

/**
 * The report's equity curve sampled every 4th session plus the last one
 * (121 of 480 points), as [date, strategy wealth multiple (net 5 bps),
 * SPY wealth multiple], rounded to 4 decimals — display precision for a
 * sparkline, far below one on-screen pixel. Column 1 is the report's net-of-5-bps
 * series: its last value, 17.4659, is exactly 1 + 16.465924 over the
 * report's window (the +1646.6% headline number).
 */
export type EquityPoint = readonly [date: string, strategy: number, spy: number];
export const EQUITY_CURVE: readonly EquityPoint[] = [
  ["2024-07-01", 1.0002, 1.0021],
  ["2024-07-08", 1.0539, 1.0203],
  ["2024-07-12", 1.1078, 1.0290],
  ["2024-07-18", 1.0506, 1.0155],
  ["2024-07-24", 0.9938, 0.9945],
  ["2024-07-30", 0.9318, 0.9959],
  ["2024-08-05", 0.8262, 0.9507],
  ["2024-08-09", 0.8589, 0.9794],
  ["2024-08-15", 0.8718, 1.0163],
  ["2024-08-21", 0.9077, 1.0301],
  ["2024-08-27", 0.8978, 1.0319],
  ["2024-09-03", 0.7981, 1.0144],
  ["2024-09-09", 0.8075, 1.0040],
  ["2024-09-13", 0.8919, 1.0327],
  ["2024-09-19", 0.8864, 1.0492],
  ["2024-09-25", 1.0035, 1.0507],
  ["2024-10-01", 1.0057, 1.0480],
  ["2024-10-07", 1.0314, 1.0465],
  ["2024-10-11", 1.0531, 1.0682],
  ["2024-10-17", 1.1717, 1.0733],
  ["2024-10-23", 1.1677, 1.0653],
  ["2024-10-29", 1.2349, 1.0723],
  ["2024-11-04", 1.0876, 1.0502],
  ["2024-11-08", 1.3694, 1.1025],
  ["2024-11-14", 1.4633, 1.0936],
  ["2024-11-20", 1.7377, 1.0884],
  ["2024-11-26", 1.7516, 1.1071],
  ["2024-12-03", 1.8061, 1.1131],
  ["2024-12-09", 1.8414, 1.1145],
  ["2024-12-13", 1.7733, 1.1136],
  ["2024-12-19", 2.2415, 1.0803],
  ["2024-12-26", 2.8017, 1.1121],
  ["2025-01-02", 2.5575, 1.0812],
  ["2025-01-08", 1.7706, 1.0902],
  ["2025-01-15", 1.8792, 1.0962],
  ["2025-01-22", 2.0961, 1.1215],
  ["2025-01-28", 1.9724, 1.1180],
  ["2025-02-03", 1.9953, 1.1055],
  ["2025-02-07", 1.9309, 1.1110],
  ["2025-02-13", 1.9149, 1.1276],
  ["2025-02-20", 1.8559, 1.1288],
  ["2025-02-26", 1.5616, 1.0995],
  ["2025-03-04", 1.2937, 1.0668],
  ["2025-03-10", 1.1176, 1.0367],
  ["2025-03-14", 1.7418, 1.0408],
  ["2025-03-20", 1.5646, 1.0458],
  ["2025-03-26", 1.6062, 1.0547],
  ["2025-04-01", 1.4545, 1.0405],
  ["2025-04-07", 1.3911, 0.9356],
  ["2025-04-11", 1.4972, 0.9904],
  ["2025-04-17", 1.4084, 0.9764],
  ["2025-04-24", 1.5646, 1.0141],
  ["2025-04-30", 1.5047, 1.0286],
  ["2025-05-06", 1.5756, 1.0365],
  ["2025-05-12", 1.8808, 1.0814],
  ["2025-05-16", 2.2269, 1.1022],
  ["2025-05-22", 2.5809, 1.0816],
  ["2025-05-29", 2.4751, 1.0945],
  ["2025-06-04", 2.5718, 1.1054],
  ["2025-06-10", 2.6108, 1.1186],
  ["2025-06-16", 2.7076, 1.1179],
  ["2025-06-23", 2.4714, 1.1165],
  ["2025-06-27", 2.6769, 1.1440],
  ["2025-07-03", 2.8423, 1.1634],
  ["2025-07-10", 2.8178, 1.1643],
  ["2025-07-16", 3.2003, 1.1613],
  ["2025-07-22", 3.1348, 1.1699],
  ["2025-07-28", 3.4606, 1.1849],
  ["2025-08-01", 3.1968, 1.1566],
  ["2025-08-07", 3.2791, 1.1762],
  ["2025-08-13", 3.2428, 1.1997],
  ["2025-08-19", 2.8567, 1.1903],
  ["2025-08-25", 2.9277, 1.1952],
  ["2025-08-29", 2.9028, 1.2000],
  ["2025-09-05", 2.7862, 1.2041],
  ["2025-09-11", 3.0724, 1.2234],
  ["2025-09-17", 3.5002, 1.2263],
  ["2025-09-23", 4.0042, 1.2372],
  ["2025-09-29", 3.8687, 1.2381],
  ["2025-10-03", 4.1946, 1.2484],
  ["2025-10-09", 4.5777, 1.2521],
  ["2025-10-15", 5.3537, 1.2409],
  ["2025-10-21", 4.5060, 1.2523],
  ["2025-10-27", 4.8925, 1.2783],
  ["2025-10-31", 5.0424, 1.2724],
  ["2025-11-06", 4.6918, 1.2505],
  ["2025-11-12", 4.6601, 1.2749],
  ["2025-11-18", 4.0334, 1.2314],
  ["2025-11-24", 3.9444, 1.2475],
  ["2025-12-01", 4.0234, 1.2691],
  ["2025-12-05", 4.2866, 1.2792],
  ["2025-12-11", 4.7020, 1.2857],
  ["2025-12-17", 3.7092, 1.2525],
  ["2025-12-23", 4.4498, 1.2872],
  ["2025-12-30", 4.3809, 1.2854],
  ["2026-01-06", 5.3860, 1.2944],
  ["2026-01-12", 5.9269, 1.3007],
  ["2026-01-16", 6.2550, 1.2941],
  ["2026-01-23", 6.1632, 1.2896],
  ["2026-01-29", 6.6213, 1.2986],
  ["2026-02-04", 6.8077, 1.2839],
  ["2026-02-10", 7.4229, 1.2950],
  ["2026-02-17", 7.6421, 1.2776],
  ["2026-02-23", 8.6349, 1.2768],
  ["2026-02-27", 9.2426, 1.2835],
  ["2026-03-05", 9.2137, 1.2748],
  ["2026-03-11", 10.5594, 1.2654],
  ["2026-03-17", 9.9871, 1.2551],
  ["2026-03-23", 10.9211, 1.2296],
  ["2026-03-27", 10.2175, 1.1897],
  ["2026-04-02", 10.7616, 1.2304],
  ["2026-04-09", 12.6691, 1.2756],
  ["2026-04-15", 12.9969, 1.3132],
  ["2026-04-21", 14.0332, 1.3210],
  ["2026-04-27", 14.0783, 1.3418],
  ["2026-05-01", 16.6485, 1.3521],
  ["2026-05-07", 16.4669, 1.3726],
  ["2026-05-13", 19.2651, 1.3927],
  ["2026-05-19", 17.0152, 1.3766],
  ["2026-05-26", 18.9347, 1.4082],
  ["2026-05-29", 17.4659, 1.4193]
] as const;
