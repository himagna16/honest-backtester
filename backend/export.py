"""Truth Report export as PNG/PDF (Gap #10).

Rendered server-side with matplotlib so the educational disclaimer is BAKED
into the artifact pixels (and embedded in file metadata) — a shared image
detached from the app still carries the legal footer. The same drawing code
produces both formats.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

VERDICT_COLORS = {"red": "#b42318", "amber": "#92600a", "green": "#1c7a43"}
VERDICT_BG = {"red": "#fef3f2", "amber": "#fffaeb", "green": "#f0faf4"}


def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v * 100:+.1f}%"


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

    # ---- verdict banner ----
    ax0 = fig.add_subplot(grid[0])
    ax0.axis("off")
    color = VERDICT_COLORS[verdict["color"]]
    ax0.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax0.transAxes,
                                facecolor=VERDICT_BG[verdict["color"]],
                                edgecolor=color, linewidth=2))
    label = {"red": "KILLED", "amber": "WOUNDED", "green": "SURVIVED"}[verdict["color"]]
    ax0.text(0.03, 0.78, f"{verdict['icon']}  {label} — {verdict['color'].upper()} VERDICT",
             fontsize=10, fontweight="bold", color=color, transform=ax0.transAxes)
    ax0.text(0.03, 0.47, verdict["headline"], fontsize=15, fontweight="bold",
             color=color, transform=ax0.transAxes, wrap=True)
    ax0.text(0.03, 0.16, verdict["sub"][:180], fontsize=8.5, color="#4b5563",
             transform=ax0.transAxes, wrap=True)
    ax0.text(0.03, 1.06, f"The Honest Backtester — {report['spec'].get('name', '')}",
             fontsize=9, color="#8a919c", transform=ax0.transAxes)

    # ---- stat row ----
    ax1 = fig.add_subplot(grid[1])
    ax1.axis("off")
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
        ax1.text(x, 0.85, k.upper(), fontsize=7, color="#8a919c",
                 transform=ax1.transAxes)
        ax1.text(x, 0.45, v, fontsize=13, fontweight="bold", transform=ax1.transAxes)
        ax1.text(x, 0.12, f"{bench.get('label', 'benchmark')}: {b}", fontsize=7.5,
                 color="#4b5563", transform=ax1.transAxes)

    # ---- cost sweep table ----
    ax2 = fig.add_subplot(grid[2])
    ax2.axis("off")
    ax2.set_title("Cost sweep", loc="left", fontsize=11, fontweight="bold")
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
    table.set_fontsize(8.5)
    table.scale(1, 1.35)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#e4e2dc")
        if r == 0:
            cell.set_text_props(fontweight="bold", color="#4b5563")
        elif rows[r - 1][4] == "LOSES" and c in (1, 4):
            cell.set_text_props(color=VERDICT_COLORS["red"], fontweight="bold")

    # ---- equity curve (log) ----
    ax3 = fig.add_subplot(grid[3])
    curve = report["equity_curve"]
    dates = [pd.Timestamp(p[0]) for p in curve]
    net = [p[1] for p in curve]
    gross = [p[2] for p in curve]
    bench_curve = [p[3] for p in curve]
    ax3.plot(dates, gross, color="#8a919c", linewidth=0.8, linestyle="--",
             label="Gross (flattering)")
    ax3.plot(dates, bench_curve, color="#1f4e5f", linewidth=1.2,
             label=bench.get("label", "benchmark"))
    ax3.plot(dates, net, color="#1a1d21", linewidth=1.8, label="Net of 5 bps")
    oos_start = pd.Timestamp(report["oos"]["out_of_sample"]["start"])
    ax3.axvspan(oos_start, dates[-1], color="#fffaeb", zorder=0)
    ax3.set_yscale("log")
    ax3.set_title("Growth of $1 (log scale — shaded = 30% holdout)",
                  loc="left", fontsize=11, fontweight="bold")
    ax3.legend(fontsize=7.5, loc="upper left")
    ax3.grid(True, alpha=0.25)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # ---- findings summary ----
    ax4 = fig.add_subplot(grid[4])
    ax4.axis("off")
    ax4.set_title("Audit findings", loc="left", fontsize=11, fontweight="bold")
    y = 0.88
    shown = [f for f in report["audit"] if f["severity"] != "pass"][:6]
    for f in shown:
        sev_color = {"fatal": VERDICT_COLORS["red"],
                     "warning": VERDICT_COLORS["amber"],
                     "note": "#4b5563"}[f["severity"]]
        ax4.text(0.0, y, f"[{f['severity'].upper()}]", fontsize=8,
                 fontweight="bold", color=sev_color, transform=ax4.transAxes)
        ax4.text(0.12, y, f["name"][:90], fontsize=8.5, transform=ax4.transAxes)
        y -= 0.14
    passes = sum(1 for f in report["audit"] if f["severity"] == "pass")
    ax4.text(0.0, y, f"+ {passes} checks passed", fontsize=8, color="#1c7a43",
             transform=ax4.transAxes)

    # ---- the baked-in disclaimer footer ----
    fig.text(0.5, 0.022, disclaimer, fontsize=6.8, color="#4b5563",
             ha="center", va="bottom", wrap=True,
             bbox=dict(boxstyle="square,pad=0.6", facecolor="#faf9f6",
                       edgecolor="#e4e2dc"))

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
