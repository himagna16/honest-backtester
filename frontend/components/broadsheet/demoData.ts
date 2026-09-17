// Data for the landing-page live demo. Every value below is transcribed from
// report rep_2c6332f417 (War list, 126-day lookback, mid-2024 to May 2026; a
// separate run from the founding-story backtest) and nothing else. Raw JSON
// values are quoted in comments next to each display string so drift is
// auditable at a glance. If a number here cannot be traced to that report,
// it is a defect.

export const REPORT_ID = "rep_2c6332f417";

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
 * from the report's `audit` array (6 of its 13 checks; passes first, the
 * warning lands last).
 */
export type AuditSeverity = "pass" | "warning";
export interface AuditRow {
  name: string;
  severity: AuditSeverity;
}
export const AUDIT_ROWS: readonly AuditRow[] = [
  { name: "Lookahead bias: impossible by construction", severity: "pass" },
  { name: "Survives the cost sweep", severity: "pass" },
  { name: "Train/holdout comparison", severity: "pass" },
  { name: "Bear-market coverage", severity: "pass" },
  { name: "Multiple-testing counter", severity: "pass" },
  { name: "Hand-picked universe", severity: "warning" },
] as const;

export const AUDIT_CHECKS_TOTAL = 13; // length of the report's `audit` array

/**
 * The neutral re-run sub-row under the warning. From `neutral_rerun`:
 * net5_total_return 9.601898964610958  -> "+960.2%"
 * user_net5_total_return 18.165438429621275 -> "+1816.5%"
 * wealth_ratio 1.8077363775673898 -> "1.81×"
 */
export const NEUTRAL_RERUN = {
  lead: "Same rules, full S&P 500: ",
  sp500: "+960.2%",
  mid: " vs your ",
  user: "+1816.5%",
  tail: ". Hand-picking alone: 1.81× in final wealth.",
} as const;

/** Scene 4 — verdict, from the report's `verdict` block (color "amber", icon "▲"). */
export const VERDICT = {
  stamp: "Wounded — amber verdict",
  headline: "Beats the benchmark — but your universe is hand-picked.",
} as const;

/**
 * The metrics row: strategy (metrics_net5) vs SPY buy-and-hold (benchmark).
 * total_return 18.165438429621275 vs 0.4192764536479263
 * cagr         3.7132703172087735 vs 0.20180823015722082
 * sharpe       2.0722102680388366 vs 0.9239006969867338
 * max_drawdown -0.601155401325071 vs -0.18755225458981672
 */
export interface MetricPair {
  label: string;
  strategy: string;
  spy: string;
}
export const METRIC_PAIRS: readonly MetricPair[] = [
  { label: "Total return", strategy: "+1816.5%", spy: "+41.9%" },
  { label: "CAGR", strategy: "+371.3%", spy: "+20.2%" },
  { label: "Sharpe", strategy: "2.07", spy: "0.92" },
  { label: "Max drawdown", strategy: "−60.1%", spy: "−18.8%" },
] as const;

export const END_LINE =
  "Benchmark shown on every result. No way to turn the audit off.";

/**
 * Equity spark labels — final wealth multiples from the last equity_curve
 * point ["2026-05-29", 19.165438, …, 1.419276].
 */
export const FINAL_MULTIPLES = { strategy: "×19.2", spy: "×1.42" } as const;

/** First and last dates of the equity curve itself. */
export const EQUITY_WINDOW = "2024-07-01 → 2026-05-29";

/**
 * The full 121-point weekly equity curve from the report, transcribed as
 * [date, strategy wealth multiple (net 5 bps), SPY wealth multiple] and
 * rounded to 4 decimals — display precision for a sparkline, far below one
 * on-screen pixel. Column 1 is the report's `metrics_net5` series: its last
 * value, 19.1654, is exactly 1 + 18.165438 (the +1816.5% headline number).
 */
export type EquityPoint = readonly [date: string, strategy: number, spy: number];
export const EQUITY_CURVE: readonly EquityPoint[] = [
  ["2024-07-01", 0.9913, 1.0021],
  ["2024-07-08", 1.0426, 1.0203],
  ["2024-07-12", 1.0944, 1.0290],
  ["2024-07-18", 1.0394, 1.0155],
  ["2024-07-24", 0.9845, 0.9945],
  ["2024-07-30", 0.9238, 0.9959],
  ["2024-08-05", 0.8152, 0.9507],
  ["2024-08-09", 0.8481, 0.9794],
  ["2024-08-15", 0.8615, 1.0163],
  ["2024-08-21", 0.8976, 1.0301],
  ["2024-08-27", 0.8887, 1.0319],
  ["2024-09-03", 0.8166, 1.0144],
  ["2024-09-09", 0.8265, 1.0040],
  ["2024-09-13", 0.9113, 1.0327],
  ["2024-09-19", 0.9050, 1.0492],
  ["2024-09-25", 1.0226, 1.0507],
  ["2024-10-01", 1.0341, 1.0480],
  ["2024-10-07", 1.0619, 1.0465],
  ["2024-10-11", 1.0841, 1.0682],
  ["2024-10-17", 1.2075, 1.0733],
  ["2024-10-23", 1.2024, 1.0653],
  ["2024-10-29", 1.2736, 1.0723],
  ["2024-11-04", 1.1090, 1.0502],
  ["2024-11-08", 1.3958, 1.1025],
  ["2024-11-14", 1.4956, 1.0936],
  ["2024-11-20", 1.7699, 1.0884],
  ["2024-11-26", 1.7973, 1.1071],
  ["2024-12-03", 1.8076, 1.1131],
  ["2024-12-09", 1.8341, 1.1145],
  ["2024-12-13", 1.7675, 1.1136],
  ["2024-12-19", 2.2082, 1.0803],
  ["2024-12-26", 2.7590, 1.1121],
  ["2025-01-02", 2.6828, 1.0812],
  ["2025-01-08", 1.8116, 1.0902],
  ["2025-01-15", 1.9277, 1.0962],
  ["2025-01-22", 2.1538, 1.1215],
  ["2025-01-28", 2.0241, 1.1180],
  ["2025-02-03", 2.0431, 1.1055],
  ["2025-02-07", 1.9776, 1.1110],
  ["2025-02-13", 1.9597, 1.1276],
  ["2025-02-20", 1.8955, 1.1288],
  ["2025-02-26", 1.5955, 1.0995],
  ["2025-03-04", 1.3133, 1.0668],
  ["2025-03-10", 1.1351, 1.0367],
  ["2025-03-14", 1.7677, 1.0408],
  ["2025-03-20", 1.5862, 1.0458],
  ["2025-03-26", 1.6304, 1.0547],
  ["2025-04-01", 1.4876, 1.0405],
  ["2025-04-07", 1.4243, 0.9356],
  ["2025-04-11", 1.5336, 0.9904],
  ["2025-04-17", 1.4444, 0.9764],
  ["2025-04-24", 1.6045, 1.0141],
  ["2025-04-30", 1.5434, 1.0286],
  ["2025-05-06", 1.5840, 1.0365],
  ["2025-05-12", 1.8926, 1.0814],
  ["2025-05-16", 2.2430, 1.1022],
  ["2025-05-22", 2.5967, 1.0816],
  ["2025-05-29", 2.4910, 1.0945],
  ["2025-06-04", 2.5953, 1.1054],
  ["2025-06-10", 2.6448, 1.1186],
  ["2025-06-16", 2.7303, 1.1179],
  ["2025-06-23", 2.4898, 1.1165],
  ["2025-06-27", 2.6829, 1.1440],
  ["2025-07-03", 2.7980, 1.1634],
  ["2025-07-10", 2.7742, 1.1643],
  ["2025-07-16", 3.1422, 1.1613],
  ["2025-07-22", 3.0819, 1.1699],
  ["2025-07-28", 3.3967, 1.1849],
  ["2025-08-01", 3.2299, 1.1566],
  ["2025-08-07", 3.2991, 1.1762],
  ["2025-08-13", 3.2769, 1.1997],
  ["2025-08-19", 2.8882, 1.1903],
  ["2025-08-25", 2.9522, 1.1952],
  ["2025-08-29", 2.9238, 1.2000],
  ["2025-09-05", 2.8223, 1.2041],
  ["2025-09-11", 3.1212, 1.2234],
  ["2025-09-17", 3.5591, 1.2263],
  ["2025-09-23", 4.0678, 1.2372],
  ["2025-09-29", 3.9423, 1.2381],
  ["2025-10-03", 4.3948, 1.2484],
  ["2025-10-09", 4.7891, 1.2521],
  ["2025-10-15", 5.6160, 1.2409],
  ["2025-10-21", 4.7308, 1.2523],
  ["2025-10-27", 5.1302, 1.2783],
  ["2025-10-31", 5.2947, 1.2724],
  ["2025-11-06", 5.0107, 1.2505],
  ["2025-11-12", 4.9758, 1.2749],
  ["2025-11-18", 4.3028, 1.2314],
  ["2025-11-24", 4.1948, 1.2475],
  ["2025-12-01", 4.2739, 1.2691],
  ["2025-12-05", 4.5416, 1.2792],
  ["2025-12-11", 5.0029, 1.2857],
  ["2025-12-17", 3.9479, 1.2525],
  ["2025-12-23", 4.7378, 1.2872],
  ["2025-12-30", 4.6677, 1.2854],
  ["2026-01-06", 5.7926, 1.2944],
  ["2026-01-12", 6.3953, 1.3007],
  ["2026-01-16", 6.7649, 1.2941],
  ["2026-01-23", 6.6938, 1.2896],
  ["2026-01-29", 7.2045, 1.2986],
  ["2026-02-04", 7.4411, 1.2839],
  ["2026-02-10", 8.1203, 1.2950],
  ["2026-02-17", 8.3614, 1.2776],
  ["2026-02-23", 9.4595, 1.2768],
  ["2026-02-27", 10.1485, 1.2835],
  ["2026-03-05", 10.1594, 1.2748],
  ["2026-03-11", 11.6959, 1.2654],
  ["2026-03-17", 10.9085, 1.2551],
  ["2026-03-23", 12.1022, 1.2296],
  ["2026-03-27", 11.3725, 1.1897],
  ["2026-04-02", 11.9410, 1.2304],
  ["2026-04-09", 14.0366, 1.2756],
  ["2026-04-15", 14.4050, 1.3132],
  ["2026-04-21", 15.4701, 1.3210],
  ["2026-04-27", 15.6002, 1.3418],
  ["2026-05-01", 18.2472, 1.3521],
  ["2026-05-07", 18.1218, 1.3726],
  ["2026-05-13", 21.2129, 1.3927],
  ["2026-05-19", 18.7415, 1.3766],
  ["2026-05-26", 20.8797, 1.4082],
  ["2026-05-29", 19.1654, 1.4193]
] as const;
