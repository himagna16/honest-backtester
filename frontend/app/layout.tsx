import type { Metadata } from "next";
import Shell from "@/components/Shell";
import { fontVars } from "@/lib/fonts";
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
    <html lang="en" className={fontVars}>
      <body>
        <Shell>{children}</Shell>
      </body>
    </html>
  );
}
