import type { ReactNode } from "react";

// Fonts now load once in the root layout (lib/fonts.ts) and are shared with the
// app, so this layout only opts the marketing route into the editorial scale —
// bigger type, wider measure — on top of the same palette the workstation uses.
export default function LandingLayout({ children }: { children: ReactNode }) {
  return <div className="broadsheet">{children}</div>;
}
