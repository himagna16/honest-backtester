export const pct = (v: number | null | undefined, digits = 1): string =>
  v === null || v === undefined || Number.isNaN(v)
    ? "—"
    : `${v >= 0 ? "+" : ""}${(v * 100).toFixed(digits)}%`;

export const num = (v: number | null | undefined, digits = 2): string =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : v.toFixed(digits);

export const times = (v: number | null | undefined): string =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : `${v.toFixed(1)}×`;

export const SEVERITY_META: Record<
  string,
  { icon: string; label: string; cls: string }
> = {
  fatal: { icon: "✕", label: "FATAL", cls: "sev-fatal" },
  warning: { icon: "▲", label: "WARNING", cls: "sev-warning" },
  note: { icon: "●", label: "NOTE", cls: "sev-note" },
  pass: { icon: "✓", label: "PASS", cls: "sev-pass" },
};
