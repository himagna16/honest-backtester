// One font definition for the whole product, so the marketing page and the
// workstation cannot drift into looking like two different applications.
//
// next/font downloads these at BUILD time and serves them from our own origin —
// there is no runtime request to Google, so the local-first promise holds and
// the app still works with no network.

import { Inter, Montserrat, Newsreader } from "next/font/google";

/** Headlines, metric values, verdicts — anything that should feel set, not typed. */
export const display = Montserrat({
  subsets: ["latin"],
  weight: ["600", "700", "800"],
  variable: "--font-display",
  display: "swap",
});

/** The reading voice: standfirsts, explanatory prose, captions. */
export const serif = Newsreader({
  subsets: ["latin"],
  weight: ["300", "400", "500"],
  style: ["normal", "italic"],
  variable: "--font-serif",
  display: "swap",
});

/** UI, labels, tables, forms. Tabular figures matter more here than character. */
export const ui = Inter({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-ui",
  display: "swap",
});

/** Convenience: every font variable, for the <html> class. */
export const fontVars = `${display.variable} ${serif.variable} ${ui.variable}`;
