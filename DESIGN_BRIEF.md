# Master design prompt — The Honest Backtester

Paste the block below to start a redesign pass. Fill in the two `[BRACKETS]`
first — they carry more weight than everything else combined.

---

## THE PROMPT

> Redesign `[SCREEN — e.g. the /landing page]` of The Honest Backtester so it
> looks like it was made by a senior product designer at a company like
> `[2–3 REFERENCES — e.g. Linear, Stripe, Mercury]`, not generated.
>
> **Before writing any code, propose 3 distinct visual directions.** For each:
> typeface pairing + exact sizes, background/surface/accent hex values, layout
> structure (in words), and one sentence on what feeling it creates. Show them
> as a short comparison. Wait for me to pick one. Do not build all three.
>
> Then implement only the direction I choose, obeying these constraints:
>
> **Typography — the highest-leverage lever**
> - Use one real self-hosted typeface via `next/font` (not the system stack).
> - The type scale must span at least 5× from smallest to largest. Hero
>   ≥ 64px with tight tracking (−0.03 to −0.04em). Body 16–17px, not 13px.
> - Collapse to at most 6 sizes total. If two sizes are within 2px of each
>   other, they are the same size — merge them.
> - Set line-height by role: 1.05–1.15 for display, 1.5–1.6 for body.
>
> **Layout — break the grid**
> - No 3-column grids of identical icon cards. That is the single strongest
>   "AI made this" signal. Vary card size, use asymmetric two-column splits,
>   let one element dominate.
> - Whitespace between sections should be 3–4× the space within them. Dramatic
>   vertical rhythm, not uniform 32px everywhere.
> - Each section needs exactly one focal point. If everything is equally
>   prominent, nothing is.
>
> **Surfaces — borders are the last resort**
> - Replace most 1px borders with background shifts, spacing, or a single
>   hairline rule. Aim to cut bordered-box rules by at least half.
> - Shadows: at most one soft elevation level, used sparingly.
> - Radii: pick ONE value and use it everywhere.
>
> **Color — mostly monochrome**
> - One accent color, covering under 5% of the visible surface. Everything
>   else is a neutral ramp.
> - Semantic colors (win/loss/warning) appear only in data, never in chrome.
> - Avoid the default "trustworthy SaaS" blue-on-slate — it is what every
>   generator outputs. Earn the palette from the product's character: this
>   tool is a skeptical auditor, not a cheerful dashboard.
>
> **Show the product**
> - The landing page must contain at least one real screenshot of an actual
>   Truth Report, in a subtle browser/device frame. Take it from the running
>   app — do not mock it up, and do not invent numbers.
> - Prefer showing a real interface over describing features in prose.
>
> **Motion — restrained**
> - 150–250ms, ease-out, on hover/focus/enter only.
> - No scroll-jacking, no parallax, no decorative animation.
> - Respect `prefers-reduced-motion`.
>
> **Non-negotiables that survive any redesign**
> - The audit is never hidden, minimized, or made to look optional.
> - The benchmark comparison appears on every result.
> - The educational disclaimer stays on every report and export.
> - Severity is always icon + text label, never color alone.
> - Every text pair ≥ 4.5:1 contrast; verify, don't assume.
> - Charts keep their table fallbacks.
>
> **Working method**
> - Load the `anthropic-skills:ui-ux-pro-max` skill for the design system, and
>   `design:design-critique` to review your own output before showing me.
> - After building, screenshot it in the browser at 1440px and 375px and show
>   me. Fix what looks wrong before telling me it's done.
> - Do not touch backend logic, tests, or the DSL.

---

## Why the current version reads as AI-generated

Measured, not vibes:

| Signal | Current state | Target |
|---|---|---|
| Font sizes in use | 17, with 11 crammed between 10.5–15px | ≤ 6, spanning 5× |
| Largest type | 44px | 64–80px |
| Bordered-box rules | 15 | ≤ 7 |
| Product screenshots on landing | 0 | ≥ 1 real one |
| Icon-card grids | 6 identical cards in a row | 0 |
| Typeface | system stack | one self-hosted family |

The app itself (workbench, Truth Report) is defensible — dense and functional
is correct for an analyst tool. **The landing page is where the slop is**, and
it is also the only page a stranger will ever judge you on.

---

## How to prompt for design work (the meta-skill)

**1. Give a reference, never an adjective.** "Minimal and elegant" produces the
same generic output every time, because it is the average of everything. Naming
*Linear, Stripe, Mercury* immediately constrains type, spacing, color, and
density. This one input changes the result more than any other.

**2. One screen per request.** "Redesign the app" produces uniform mush.
"Redesign the landing hero" produces craft. Do the landing page first — it is
the highest-visibility, lowest-risk surface.

**3. Demand options before code.** Always ask for 3 directions first. The
default output is the model's average; forcing a choice breaks the average.

**4. Iterate on specifics.** "Make it better" → nothing. "The hero is too
small, take it to 72px and tighten tracking to −0.035em; the story strip should
be one wide row not three boxes" → exactly what you want.

**5. Make me look at it.** Require a browser screenshot at the end of every
pass. Design that is never viewed drifts.

---

## Reference shortlist

Name whichever match your taste — this is the `[REFERENCES]` slot:

- **Linear** — dense, restrained, dark-capable, superb type
- **Stripe** — editorial, generous whitespace, technical credibility
- **Mercury** — calm fintech, serious about money without being cold
- **Vercel / Geist** — stark monochrome, near-zero decoration
- **Arc / Raycast** — playful precision, strong product shots
- **Bloomberg / FT** — data-dense editorial (fits this product's character)

For this product specifically, **Mercury × FT** is a stronger fit than the
generic Linear-clone look: financial seriousness plus editorial typography,
which suits a tool whose job is to disagree with you.

---

## Skills and tools worth using

Already available in this environment:

| Tool | Use it for |
|---|---|
| `anthropic-skills:ui-ux-pro-max` | Design system: palettes, type pairings, style categories, UX rules |
| `design:design-critique` | Structured review of a screen before you accept it |
| `design:ux-copy` | Microcopy, empty states, button labels, error wording |
| `design:design-system` | Documenting components once the look is settled |
| `design:accessibility-review` | WCAG audit before shipping |
| `dataviz` | The equity/drawdown charts specifically |
| Figma MCP | Real design files — **needs authorizing first** (claude.ai connector settings, or `/mcp` in an interactive terminal) |

Worth adding to the project:

- **A real typeface.** The single biggest upgrade. Self-host via `next/font`
  so there is no runtime network dependency. Good pairings for this product:
  *Söhne / Suisse / Inter Tight* for UI, *Tiempos / Newsreader / Fraunces* for
  editorial headlines, `ui-monospace` for numbers.
- **Tailwind + shadcn/ui** — only if you want to rebuild the component layer.
  It buys consistency and a large well-designed primitive set, at the cost of
  a migration. Not required for a good result; the current CSS is coherent.
- **Real screenshots** — run the app, capture a genuine Truth Report, and put
  it on the landing page.
