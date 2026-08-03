import { Inter, Montserrat, Newsreader } from "next/font/google";
import type { ReactNode } from "react";

// Fonts are scoped to the marketing route: the app itself keeps the system
// stack so it renders instantly and works fully offline.
const display = Montserrat({
  subsets: ["latin"],
  weight: ["600", "700", "800"],
  variable: "--font-display",
  display: "swap",
});

const serif = Newsreader({
  subsets: ["latin"],
  weight: ["300", "400", "500"],
  style: ["normal", "italic"],
  variable: "--font-serif",
  display: "swap",
});

const ui = Inter({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-ui",
  display: "swap",
});

export default function LandingLayout({ children }: { children: ReactNode }) {
  return (
    <div className={`${display.variable} ${serif.variable} ${ui.variable} broadsheet`}>
      {children}
    </div>
  );
}
