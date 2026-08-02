import type { Metadata } from "next";
import Link from "next/link";
import { DISCLAIMER } from "@/lib/constants";
import "./globals.css";

export const metadata: Metadata = {
  title: "The Honest Backtester",
  description:
    "The backtester that tries to kill your strategy before the market does.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <div className="shell">
          <nav className="topnav" aria-label="Main">
            <Link href="/" className="brand">
              The <span>Honest</span> Backtester
            </Link>
            <Link href="/journal" className="navlink">
              Journal
            </Link>
            <Link href="/glossary" className="navlink">
              Glossary
            </Link>
          </nav>
          <main className="content">{children}</main>
          <footer className="disclaimer-footer">{DISCLAIMER}</footer>
        </div>
      </body>
    </html>
  );
}
