"use client";

// Motion primitives for the Broadsheet landing page (/landing).
//
// The contract, in priority order:
//   1. Server HTML ships fully visible and fully styled. Nothing in this
//      file is required to read the page — motion is an enhancement.
//   2. Hidden states are applied only after hydration, and only to elements
//      still below the viewport. Content the reader can already see is
//      never hidden — no flash, and no CLS (opacity/transform only).
//   3. prefers-reduced-motion: reduce disables everything here; the CSS
//      side independently neutralises the armed states as a second guard.
//   4. Every effect fires exactly once; observers are disconnected after.

import {
  useEffect,
  useRef,
  type CSSProperties,
  type ComponentPropsWithoutRef,
  type ElementType,
} from "react";

/* ------------------------------------------------------------------ */
/* Observer plumbing shared by <Reveal> and <FigureCount>              */
/* ------------------------------------------------------------------ */

// Fire once at ~18% visibility. The negative bottom rootMargin pulls the
// trigger line 10% of the viewport up from the fold, so an element starts
// moving just after it enters the reading zone — comfortably before it is
// fully in view — and the motion is actually witnessed, never wasted
// off-screen.
const IO_OPTIONS: IntersectionObserverInit = {
  threshold: 0.18,
  rootMargin: "0px 0px -10% 0px",
};

function reducedMotion(): boolean {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/** Is any part of the element inside the current viewport? */
function onScreen(el: Element): boolean {
  const r = el.getBoundingClientRect();
  const vh = window.innerHeight || document.documentElement.clientHeight;
  return r.top < vh && r.bottom > 0;
}

/**
 * May this element be armed (hidden, then revealed)? False means: leave
 * the server-rendered state exactly as painted.
 */
function mayArm(el: Element): boolean {
  return (
    typeof IntersectionObserver !== "undefined" &&
    !reducedMotion() &&
    !onScreen(el) // never hide what the reader is currently looking at
  );
}

/** Observe until first intersection, fire once, disconnect. Returns a disposer. */
function observeOnce(el: Element, fire: () => void): () => void {
  const io = new IntersectionObserver((entries) => {
    if (entries.some((e) => e.isIntersecting)) {
      io.disconnect();
      fire();
    }
  }, IO_OPTIONS);
  io.observe(el);
  return () => io.disconnect();
}

/* ------------------------------------------------------------------ */
/* <Reveal> — scroll-triggered fade-up, or a drawn-in section rule     */
/* ------------------------------------------------------------------ */

export type RevealProps<T extends ElementType = "div"> = {
  /** Rendered tag. Keep the semantic element: as="h2", as="article", ... */
  as?: T;
  /** Stagger in ms — becomes transition-delay once the element is on screen. */
  delay?: number;
  /**
   * "fade" (default): the element itself fades up 14px (.bs-reveal).
   * "rule": only the element's 2px top rule draws in from the left
   *         (.bs-rule); its children are left alone so they can carry
   *         their own reveals.
   */
  variant?: "fade" | "rule";
} & Omit<ComponentPropsWithoutRef<T>, "as" | "delay" | "variant">;

// Internal wide shape: TS cannot destructure named keys off a generic
// Omit<ComponentPropsWithoutRef<T>, ...>, so the implementation widens and
// the public generic signature above keeps call sites strictly checked.
type RevealLooseProps = {
  as?: ElementType;
  delay?: number;
  variant?: "fade" | "rule";
  className?: string;
  style?: CSSProperties;
} & Record<string, unknown>;

export function Reveal<T extends ElementType = "div">(props: RevealProps<T>) {
  const {
    as: Tag = "div",
    delay = 0,
    variant = "fade",
    className,
    style,
    ...rest
  } = props as unknown as RevealLooseProps;

  const ref = useRef<HTMLElement | null>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el || !mayArm(el)) return;
    el.setAttribute("data-motion", "pending");
    const dispose = observeOnce(el, () => {
      el.setAttribute("data-motion", "in");
    });
    return () => {
      dispose();
      el.removeAttribute("data-motion"); // strict-mode / unmount hygiene
    };
  }, []);

  const base = variant === "rule" ? "bs-rule" : "bs-reveal";
  return (
    <Tag
      ref={ref}
      className={className ? `${base} ${className}` : base}
      style={{ ...style, "--reveal-delay": `${delay}ms` } as CSSProperties}
      {...rest}
    />
  );
}

/* ------------------------------------------------------------------ */
/* <FigureCount> — the three .bs-fig numbers                           */
/* ------------------------------------------------------------------ */

const COUNT_MS = 900; // count-up duration
const STRIKE_LAG_MS = 300; // pause after the count lands, before the strike
const NUM_RE = /\d[\d.]*/; // first run of digits in the printed string

export type FigureCountProps = {
  /**
   * The exact final printed string: "+1650%", "+159%", "Lost".
   * Contains digits -> counts up and lands on this exact string.
   * Contains none  -> stamped in (slight scale + opacity), never counted.
   */
  value: string;
  /** ms after this number scrolls on screen before its effect starts. */
  delay?: number;
  /** Draw the oxblood line-through ~300ms after the count lands (fig a). */
  strike?: boolean;
  className?: string;
};

export function FigureCount({
  value,
  delay = 0,
  strike = false,
  className,
}: FigureCountProps) {
  const ref = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el || !mayArm(el)) return;

    let raf = 0;
    const timers: number[] = [];
    const later = (fn: () => void, ms: number): void => {
      timers.push(window.setTimeout(fn, ms));
    };

    const match = NUM_RE.exec(value);

    /* Stamp mode — no digits ("Lost"): pressed in like a rubber stamp. */
    if (!match) {
      el.dataset.stamp = "pending";
      const dispose = observeOnce(el, () => {
        later(() => {
          el.dataset.stamp = "in";
        }, delay);
      });
      return () => {
        dispose();
        timers.forEach((t) => clearTimeout(t));
        delete el.dataset.stamp;
      };
    }

    /* Count mode — "+1650%" -> prefix "+", target 1650, suffix "%". */
    const target = Number(match[0]);
    const decimals = (match[0].split(".")[1] ?? "").length;
    const prefix = value.slice(0, match.index);
    const suffix = value.slice(match.index + match[0].length);

    // Armed while safely off screen: rewind to zero. Setting [data-strike]
    // also suppresses the static line-through so it cannot show mid-count.
    el.textContent = `${prefix}${(0).toFixed(decimals)}${suffix}`;
    if (strike) el.dataset.strike = "pending";

    const dispose = observeOnce(el, () => {
      later(() => {
        const t0 = performance.now();
        const tick = (now: number) => {
          const t = Math.min((now - t0) / COUNT_MS, 1);
          const eased = 1 - Math.pow(1 - t, 3); // ease-out cubic = --bs-ease
          if (t < 1) {
            el.textContent = `${prefix}${(eased * target).toFixed(decimals)}${suffix}`;
            raf = requestAnimationFrame(tick);
          } else {
            el.textContent = value; // land on the exact printed string
            if (strike) {
              later(() => {
                el.dataset.strike = "drawn";
              }, STRIKE_LAG_MS);
            }
          }
        };
        raf = requestAnimationFrame(tick);
      }, delay);
    });

    return () => {
      dispose();
      timers.forEach((t) => clearTimeout(t));
      cancelAnimationFrame(raf);
      delete el.dataset.strike;
      el.textContent = value; // restore the exact printed string
    };
  }, [value, delay, strike]);

  return (
    <span ref={ref} className={className ? `bs-count ${className}` : "bs-count"}>
      {value}
    </span>
  );
}
