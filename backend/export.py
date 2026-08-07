"""Truth Report export as PNG/PDF (Gap #10).

Rendered server-side with matplotlib so the educational disclaimer is BAKED
into the artifact pixels (and embedded in file metadata) — a shared image
detached from the app still carries the legal footer. The same drawing code
produces both formats.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager as _fmgr  # noqa: E402

# ---------------------------------------------------------------------------
# Fonts: the export must look like the app, not like matplotlib. We ship the
# same Montserrat (display) + Inter (UI/body) the web uses as static TTFs in
# assets/fonts/, register them, and reference them BY FILE so the exact weight
# renders regardless of the family name the file reports. Bundled (not
# system-loaded) so the Linux backend renders identically to local.
# ---------------------------------------------------------------------------
_FONT_DIR = Path(__file__).parent / "assets" / "fonts"
_FONT_FILES = {
    ("display", 600): "Montserrat-600.ttf",
    ("display", 700): "Montserrat-700.ttf",
    ("display", 800): "Montserrat-800.ttf",
    ("body", 400): "Inter-400.ttf",
    ("body", 500): "Inter-500.ttf",
    ("body", 600): "Inter-600.ttf",
}
_FONTS_OK = all((_FONT_DIR / f).exists() for f in _FONT_FILES.values())
if _FONTS_OK:
    for _f in _FONT_FILES.values():
        _fmgr.fontManager.addfont(str(_FONT_DIR / _f))
    # Baseline for anything drawn without an explicit face (ticks, legend).
    matplotlib.rcParams["font.family"] = "Inter"


@lru_cache(maxsize=None)
def _fp(kind: str, weight: int, size: float):
    """A sized FontProperties for our bundled faces (falls back to the default
    sans if the files are somehow missing, so exports never crash)."""
    if not _FONTS_OK:
        return _fmgr.FontProperties(size=size,
                                    weight="bold" if weight >= 700 else "normal")
    fp = _fmgr.FontProperties(fname=str(_FONT_DIR / _FONT_FILES[(kind, weight)]))
    fp.set_size(size)
    return fp


# Warm paper-and-ink palette, matching the web app (globals.css) rather than
# the old slate scheme, so the exported card is the same design.
INK = "#1a1815"
INK_2 = "#4a453e"
INK_3 = "#6e675d"
RULE = "#ddd8ce"
VERDICT_COLORS = {"red": "#8c2f2f", "amber": "#7a5a18", "green": "#2f5d3a"}
VERDICT_BG = {"red": "#f7eded", "amber": "#f8f2e2", "green": "#edf3ee"}


def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v * 100:+.1f}%"


def _wrap_to(text_obj, width_px: float):
    """Make matplotlib wrap `text_obj` at `width_px` instead of the figure edge.

    matplotlib's wrap=True computes the wrap width from the FIGURE boundary, so
    text placed inside a drawn box happily runs through the box's right border
    before wrapping (the defect: the verdict sub-text kissed the banner frame).
    `_get_wrap_line_width` is a private hook, but it is the only override point
    and has been stable for years; test_exports pins the behaviour by measuring
    rendered extents, so a matplotlib upgrade that breaks this fails loudly.
    """
    text_obj.set_wrap(True)
    text_obj._get_wrap_line_width = lambda: width_px
    return text_obj


def render_report(report: dict, out_path: Path, fmt: str) -> Path:
    verdict = report["verdict"]
    metrics = report["metrics"]
    bench = report["benchmark"]
    disclaimer = report["disclaimer"]

    fig = plt.figure(figsize=(10, 12.5), dpi=150)
    fig.patch.set_facecolor("white")
    grid = fig.add_gridspec(
        5, 1, height_ratios=[1.35, 0.8, 1.3, 2.2, 1.6], hspace=0.55,
        left=0.09, right=0.96, top=0.95, bottom=0.075,
    )

    # ---- verdict block ----
    # Matches the web verdict scene exactly: NO filled box — a bordered stamp
    # pill (verdict colour), an ink headline, then a 2px ink rule above the
    # metrics. The old amber-filled banner is what made the export look unlike
    # the app.
    ax0 = fig.add_subplot(grid[0])
    ax0.axis("off")
    color = VERDICT_COLORS[verdict["color"]]
    label = {"red": "KILLED", "amber": "WOUNDED", "green": "SURVIVED"}[verdict["color"]]

    ax0.text(0.0, 1.02, f"The Honest Backtester — {report['spec'].get('name', '')}",
             fontproperties=_fp("body", 500, 9), color=INK_3,
             transform=ax0.transAxes, va="top")

    # Bordered stamp pill (transparent fill, coloured border) — .ld-stamp. The
    # label sits in Montserrat; the ▲/✓/✕ glyph is drawn in DejaVu (Montserrat
    # lacks it) into the leading-space gap so both live inside one border.
    ax0.text(0.0, 0.74, f"       {label} — {verdict['color'].upper()} VERDICT",
             fontproperties=_fp("display", 700, 9), color=color,
             va="center", ha="left",
             bbox=dict(boxstyle="square,pad=0.62", fc="none", ec=color, lw=1.3),
             transform=ax0.transAxes)
    ax0.text(0.011, 0.742, verdict["icon"],
             fontproperties=_fmgr.FontProperties(family="DejaVu Sans", size=8),
             color=color, va="center", ha="center", transform=ax0.transAxes)

    # Headline in ink, no box (.ld-verdict). Wrap at the full content width.
    content_px = ax0.get_position().width * fig.get_figwidth() * fig.dpi
    _wrap_to(ax0.text(0.0, 0.50, verdict["headline"],
                      fontproperties=_fp("display", 700, 16),
                      color=INK, transform=ax0.transAxes, va="top"), content_px)
    _wrap_to(ax0.text(0.0, 0.14, verdict["sub"][:180],
                      fontproperties=_fp("body", 400, 8.5),
                      color=INK_2, transform=ax0.transAxes, va="top"), content_px)

    # ---- stat row ----
    ax1 = fig.add_subplot(grid[1])
    ax1.axis("off")
    # 2px ink rule above the metrics — .ld-metrics border-top.
    ax1.plot([0, 1], [1.06, 1.06], color=INK, linewidth=1.6,
             transform=ax1.transAxes, clip_on=False)
    stats = [
        ("Total return (net 5 bps)", _fmt_pct(metrics.get("total_return")),
         _fmt_pct(bench.get("total_return"))),
        ("CAGR", _fmt_pct(metrics.get("cagr")), _fmt_pct(bench.get("cagr"))),
        ("Sharpe", f"{metrics.get('sharpe', 0):.2f}", f"{bench.get('sharpe', 0):.2f}"),
        ("Max drawdown", _fmt_pct(metrics.get("max_drawdown")),
         _fmt_pct(bench.get("max_drawdown"))),
    ]
    for i, (k, v, b) in enumerate(stats):
        x = i / 4 + 0.01
        ax1.text(x, 0.85, k.upper(), fontproperties=_fp("body", 600, 7),
                 color=INK_3, transform=ax1.transAxes)
        ax1.text(x, 0.45, v, fontproperties=_fp("display", 700, 14),
                 color=INK, transform=ax1.transAxes)
        ax1.text(x, 0.12, f"{bench.get('label', 'benchmark')}: {b}",
                 fontproperties=_fp("body", 400, 7.5), color=INK_3,
                 transform=ax1.transAxes)

    # ---- cost sweep table ----
    ax2 = fig.add_subplot(grid[2])
    ax2.axis("off")
    ax2.set_title("Cost sweep", loc="left", fontproperties=_fp("display", 700, 12),
                  color=INK)
    rows = []
    for row in report["cost_sweep"]:
        name = row.get("label") or f"{row['bps']:.0f} bps"
        beats = ("—" if row["beats_benchmark"] is None
                 else ("beats" if row["beats_benchmark"] else "LOSES"))
        rows.append([name, _fmt_pct(row["total_return"]),
                     f"{row['sharpe']:.2f}", _fmt_pct(row["max_drawdown"]), beats])
    table = ax2.table(
        cellText=rows,
        colLabels=["Costs", "Total return", "Sharpe", "Max DD", "vs benchmark"],
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.scale(1, 1.35)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(RULE)
        if r == 0:
            cell.set_text_props(fontproperties=_fp("body", 600, 8.5), color=INK_2)
        elif rows[r - 1][4] == "LOSES" and c in (1, 4):
            cell.set_text_props(fontproperties=_fp("body", 600, 8.5),
                                color=VERDICT_COLORS["red"])
        else:
            cell.set_text_props(fontproperties=_fp("body", 400, 8.5), color=INK)

    # ---- equity curve (log) ----
    ax3 = fig.add_subplot(grid[3])
    curve = report["equity_curve"]
    dates = [pd.Timestamp(p[0]) for p in curve]
    net = [p[1] for p in curve]
    gross = [p[2] for p in curve]
    bench_curve = [p[3] for p in curve]
    ax3.plot(dates, gross, color=INK_3, linewidth=0.8, linestyle="--",
             label="Gross (flattering)")
    ax3.plot(dates, bench_curve, color=INK_3, linewidth=1.2,
             label=bench.get("label", "benchmark"))
    ax3.plot(dates, net, color=INK, linewidth=1.8, label="Net of 5 bps")
    oos_start = pd.Timestamp(report["oos"]["out_of_sample"]["start"])
    ax3.axvspan(oos_start, dates[-1], color=VERDICT_BG["amber"], zorder=0)
    ax3.set_yscale("log")
    ax3.set_title("Growth of $1 (log scale — shaded = 30% holdout)",
                  loc="left", fontproperties=_fp("display", 700, 12), color=INK)
    ax3.legend(prop=_fp("body", 400, 7.5), loc="upper left")
    ax3.grid(True, alpha=0.25)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for lbl in (*ax3.get_xticklabels(), *ax3.get_yticklabels()):
        lbl.set_fontproperties(_fp("body", 400, 8))
        lbl.set_color(INK_3)

    # ---- findings summary ----
    ax4 = fig.add_subplot(grid[4])
    ax4.axis("off")
    ax4.set_title("Audit findings", loc="left", fontproperties=_fp("display", 700, 12),
                  color=INK)
    y = 0.88
    shown = [f for f in report["audit"] if f["severity"] != "pass"][:6]
    for f in shown:
        sev_color = {"fatal": VERDICT_COLORS["red"],
                     "warning": VERDICT_COLORS["amber"],
                     "note": INK_3}[f["severity"]]
        ax4.text(0.0, y, f"[{f['severity'].upper()}]", fontproperties=_fp("body", 600, 8),
                 color=sev_color, transform=ax4.transAxes)
        ax4.text(0.12, y, f["name"][:90], fontproperties=_fp("body", 400, 8.5),
                 color=INK, transform=ax4.transAxes)
        y -= 0.14
    passes = sum(1 for f in report["audit"] if f["severity"] == "pass")
    ax4.text(0.0, y, f"+ {passes} checks passed", fontproperties=_fp("body", 500, 8),
             color=VERDICT_COLORS["green"], transform=ax4.transAxes)

    # ---- the baked-in disclaimer footer ----
    # Same wrap fix: centered text wrapped at the raw figure edge produces
    # margin-to-margin lines; wrap at 88% so the footer keeps side margins.
    _wrap_to(fig.text(0.5, 0.022, disclaimer, fontproperties=_fp("body", 400, 6.8),
                      color=INK_3, ha="center", va="bottom",
                      bbox=dict(boxstyle="square,pad=0.6", facecolor="#faf9f6",
                                edgecolor=RULE)),
             fig.get_figwidth() * fig.dpi * 0.88)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "png":
        from PIL import PngImagePlugin

        tmp = out_path.with_suffix(".tmp.png")
        fig.savefig(tmp, format="png")
        # embed the disclaimer in PNG metadata as well as pixels
        from PIL import Image

        img = Image.open(tmp)
        meta = PngImagePlugin.PngInfo()
        meta.add_text("Disclaimer", disclaimer)
        meta.add_text("Software", "The Honest Backtester V0")
        img.save(out_path, pnginfo=meta)
        tmp.unlink()
    else:
        fig.savefig(out_path, format="pdf",
                    metadata={"Subject": disclaimer,
                              "Creator": "The Honest Backtester V0"})
    plt.close(fig)
    return out_path
