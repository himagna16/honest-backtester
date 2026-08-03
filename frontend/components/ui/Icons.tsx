// Inline SVG icon set (Lucide-style geometry, 1.6px strokes). No emoji as
// icons. Every icon is aria-hidden — the adjacent text is the label.

type P = { size?: number; className?: string };

const base = (size: number) => ({
  width: size, height: size, viewBox: "0 0 24 24",
  fill: "none", stroke: "currentColor", strokeWidth: 1.6,
  strokeLinecap: "round" as const, strokeLinejoin: "round" as const,
  "aria-hidden": true, focusable: false as const,
});

export const IconWorkbench = ({ size = 16 }: P) => (
  <svg {...base(size)}><path d="M3 3v18h18" /><path d="m7 14 3-4 3 3 5-7" /></svg>
);
export const IconJournal = ({ size = 16 }: P) => (
  <svg {...base(size)}><path d="M4 4.5A2.5 2.5 0 0 1 6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5z" /><path d="M8 7h8M8 11h6" /></svg>
);
export const IconGlossary = ({ size = 16 }: P) => (
  <svg {...base(size)}><circle cx="12" cy="12" r="9" /><path d="M9.5 9a2.5 2.5 0 1 1 3.4 2.3c-.6.3-.9.9-.9 1.5v.4" /><path d="M12 17h.01" /></svg>
);
export const IconDescribe = ({ size = 16 }: P) => (
  <svg {...base(size)}><path d="M4 5h16M4 10h10M4 15h13M4 20h7" /></svg>
);
export const IconUpload = ({ size = 16 }: P) => (
  <svg {...base(size)}><path d="M12 15V3" /><path d="m7 8 5-5 5 5" /><path d="M3 15v4a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-4" /></svg>
);
export const IconRun = ({ size = 16 }: P) => (
  <svg {...base(size)}><path d="M6 4.5v15l12-7.5z" /></svg>
);
export const IconSettings = ({ size = 16 }: P) => (
  <svg {...base(size)}><circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.9l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-2.9 1.2v.1a2 2 0 1 1-4 0v-.2a1.7 1.7 0 0 0-3-1.1l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1A1.7 1.7 0 0 0 3 15a2 2 0 1 1 0-4h.2a1.7 1.7 0 0 0 1.1-3l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 2.9-1.2V4a2 2 0 1 1 4 0v.2a1.7 1.7 0 0 0 2.9 1.1l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0 1.1 2.9H21a2 2 0 1 1 0 4h-.2a1.7 1.7 0 0 0-1.4 1z" /></svg>
);
export const IconChart = ({ size = 16 }: P) => (
  <svg {...base(size)}><path d="M3 3v18h18" /><rect x="7" y="12" width="3" height="6" rx="1" /><rect x="12" y="8" width="3" height="10" rx="1" /><rect x="17" y="5" width="3" height="13" rx="1" /></svg>
);
export const IconWarning = ({ size = 16 }: P) => (
  <svg {...base(size)}><path d="M10.3 3.9 2.4 17.5A2 2 0 0 0 4.1 20.5h15.8a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z" /><path d="M12 9v4M12 17h.01" /></svg>
);
export const IconAudit = ({ size = 16 }: P) => (
  <svg {...base(size)}><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" /><path d="M14 2v6h6" /><path d="m9 14 2 2 4-4" /></svg>
);
export const IconCheck = ({ size = 16 }: P) => (
  <svg {...base(size)}><path d="m4 12.5 5 5L20 6.5" /></svg>
);
export const IconX = ({ size = 16 }: P) => (
  <svg {...base(size)}><path d="M18 6 6 18M6 6l12 12" /></svg>
);
export const IconInfo = ({ size = 16 }: P) => (
  <svg {...base(size)}><circle cx="12" cy="12" r="9" /><path d="M12 11v5M12 8h.01" /></svg>
);
export const IconDownload = ({ size = 16 }: P) => (
  <svg {...base(size)}><path d="M12 3v12" /><path d="m7 10 5 5 5-5" /><path d="M3 17v2a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-2" /></svg>
);
export const IconRefresh = ({ size = 16 }: P) => (
  <svg {...base(size)}><path d="M21 12a9 9 0 1 1-2.6-6.4" /><path d="M21 4v5h-5" /></svg>
);
export const IconSnapshot = ({ size = 16 }: P) => (
  <svg {...base(size)}><rect x="3" y="6" width="18" height="14" rx="2" /><circle cx="12" cy="13" r="3.2" /><path d="M8 6l1.5-2.5h5L16 6" /></svg>
);
export const IconRisk = ({ size = 16 }: P) => (
  <svg {...base(size)}><path d="M12 2 3 6v6c0 5 3.8 8.9 9 10 5.2-1.1 9-5 9-10V6z" /><path d="M12 8v4M12 16h.01" /></svg>
);
export const IconTimeline = ({ size = 16 }: P) => (
  <svg {...base(size)}><path d="M3 12h18" /><circle cx="7" cy="12" r="2" /><circle cx="16" cy="12" r="2" /><path d="M7 7v3M16 14v3" /></svg>
);
export const IconCompare = ({ size = 16 }: P) => (
  <svg {...base(size)}><path d="M12 3v18" /><path d="M6 8H3l3-4 3 4H6zM6 8v8" /><path d="M18 16h3l-3 4-3-4h3zm0 0V8" /></svg>
);
export const IconEmptyChart = ({ size = 32 }: P) => (
  <svg {...base(size)} strokeWidth={1.3}><rect x="3" y="4" width="18" height="16" rx="2" /><path d="M3 15l4-4 3 3 4-6 3 4" strokeDasharray="2 2.5" /></svg>
);
