"use client";

import type { ReactNode } from "react";
import { IconInfo, IconWarning, IconX } from "./Icons";

/* ---------- segmented control (modes, result tabs) ---------- */

export function Segmented<T extends string>({
  options, value, onChange, label,
}: {
  options: { value: T; label: string; icon?: ReactNode }[];
  value: T;
  onChange: (v: T) => void;
  label: string;
}) {
  return (
    <div className="segmented" role="tablist" aria-label={label}>
      {options.map((o) => (
        <button
          key={o.value}
          role="tab"
          aria-selected={value === o.value}
          onClick={() => onChange(o.value)}
          type="button"
        >
          {o.icon}
          {o.label}
        </button>
      ))}
    </div>
  );
}

/* ---------- gentle notices (never aggressive error styling) ---------- */

export function Notice({
  kind = "info", children,
}: {
  kind?: "info" | "warn" | "err";
  children: ReactNode;
}) {
  const Icon = kind === "info" ? IconInfo : kind === "warn" ? IconWarning : IconX;
  return (
    <div className={`notice notice-${kind}`} role={kind === "err" ? "alert" : undefined}>
      <Icon size={14} />
      <span>{children}</span>
    </div>
  );
}

/* ---------- panel ---------- */

export function Panel({
  title, icon, right, children, tight,
}: {
  title?: string;
  icon?: ReactNode;
  right?: ReactNode;
  children: ReactNode;
  tight?: boolean;
}) {
  return (
    <section className="panel">
      {title && (
        <header>
          {icon}
          <h2>{title}</h2>
          {right && <div className="right">{right}</div>}
        </header>
      )}
      <div className={`body${tight ? " tight" : ""}`}>{children}</div>
    </section>
  );
}

/* ---------- evidence-strength indicator ----------
   Deliberately NOT a model-confidence score: it is a deterministic read of how
   much data backs the conclusion (trading days, rebalances, regime coverage). */

export function Confidence({ level, label }: { level: 1 | 2 | 3; label: string }) {
  return (
    <span className="confidence" title={label}>
      <span className="confbar" aria-hidden="true">
        {[1, 2, 3].map((i) => (
          <i key={i} className={i <= level ? "on" : ""} />
        ))}
      </span>
      {label}
    </span>
  );
}
