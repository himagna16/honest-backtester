"use client";

// Workstation shell: left sidebar (nav) + top bar (context + primary actions).
// On narrow screens the sidebar collapses to a 3-item bottom nav.

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState, type ReactNode } from "react";
import { api } from "@/lib/api";
import { DISCLAIMER } from "@/lib/constants";
import { IconGlossary, IconJournal, IconWorkbench } from "@/components/ui/Icons";

const NAV = [
  { href: "/", label: "Workbench", Icon: IconWorkbench },
  { href: "/journal", label: "Journal", Icon: IconJournal },
  { href: "/glossary", label: "Glossary", Icon: IconGlossary },
];

function isActive(pathname: string, href: string) {
  return href === "/" ? pathname === "/" : pathname.startsWith(href);
}

// Public marketing routes render bare — no sidebar, no app chrome.
const BARE_PREFIXES = ["/landing"];

export default function Shell({ children }: { children: ReactNode }) {
  const pathname = usePathname() ?? "/";
  const [health, setHealth] = useState<{ llm_configured: boolean } | null>(null);
  const [offline, setOffline] = useState(false);

  useEffect(() => {
    api<{ status: string; llm_configured: boolean }>("/health")
      .then(setHealth)
      .catch(() => setOffline(true));
  }, []);

  if (BARE_PREFIXES.some((p) => pathname.startsWith(p))) {
    return <>{children}</>;
  }

  return (
    <div className="shell">
      <aside className="sidebar">
        <Link href="/" className="brand">
          <svg width="16" height="16" viewBox="0 0 24 24" aria-hidden="true">
            <rect x="2" y="2" width="20" height="20" rx="5" fill="#0f172a" />
            <path d="M6 16.5 10 11l3 3 5-7.5" stroke="#7dd3fc" strokeWidth="2"
              fill="none" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          Honest Backtester
        </Link>
        <nav aria-label="Primary">
          {NAV.map(({ href, label, Icon }) => (
            <Link key={href} href={href} className="navlink"
              aria-current={isActive(pathname, href) ? "page" : undefined}>
              <Icon size={15} />
              {label}
            </Link>
          ))}
        </nav>
        <div className="foot">
          <div>
            <span className={`statusdot ${offline ? "off" : "ok"}`} />
            {offline ? "Backend unreachable" : "Engine ready · offline data"}
          </div>
          <div style={{ marginTop: 4 }}>
            {health
              ? health.llm_configured
                ? "Translation: configured"
                : "Translation: no API key"
              : " "}
          </div>
        </div>
      </aside>

      <div style={{ minWidth: 0 }}>
        {children}
        <nav className="mobilenav" aria-label="Primary">
          {NAV.map(({ href, label, Icon }) => (
            <Link key={href} href={href}
              aria-current={isActive(pathname, href) ? "page" : undefined}>
              <Icon size={18} />
              {label}
            </Link>
          ))}
        </nav>
      </div>
    </div>
  );
}

/* Top bar — pages supply their own context + actions. */
export function TopBar({
  crumb, strategy, actions,
}: {
  crumb: string;
  strategy?: string;
  actions?: ReactNode;
}) {
  return (
    <header className="topbar">
      <div className="crumb">
        <span>{crumb}</span>
        {strategy && (
          <>
            <span aria-hidden="true">/</span>
            <strong title={strategy}>{strategy}</strong>
          </>
        )}
      </div>
      {actions && <div className="actions">{actions}</div>}
    </header>
  );
}

export function PageDisclaimer() {
  return <p className="disclaimer">{DISCLAIMER}</p>;
}
