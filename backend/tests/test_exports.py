"""M5 acceptance: exports carry the disclaimer inside the artifact; spec
JSON export/import round-trips; client telemetry is namespaced."""
import io
import json

import pytest
from fastapi.testclient import TestClient

from main import app
from tests.test_validator import EXAMPLE_MOMENTUM


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def finished_run(client):
    from tests.test_audit import war_tickers
    spec = {
        "name": "Export test strategy",
        "universe": {"preset": "user_watchlist", "resolved_tickers": war_tickers()},
        "selection": {"rank_by": {"type": "return", "lookback_days": 126},
                      "direction": "top", "n": 5},
        "weighting": "equal", "rebalance": "monthly", "benchmark": "SPY",
        "window": {"start": "2024-06-03", "end": "2026-05-29"},
    }
    resp = client.post("/api/run-backtest", json={"spec": spec})
    run_id = resp.json()["run_id"]
    status = client.get(f"/api/runs/{run_id}").json()
    assert status["status"] == "done", status.get("error")
    return {"strategy_id": resp.json()["strategy_id"],
            "report_id": status["report_id"]}


def test_png_export_bakes_disclaimer(client, finished_run):
    from PIL import Image

    resp = client.get(f"/api/report/{finished_run['report_id']}/export.png")
    assert resp.status_code == 200
    img = Image.open(io.BytesIO(resp.content))
    assert img.format == "PNG"
    assert "not investment advice" in img.info.get("Disclaimer", "").lower()
    assert img.width > 1000  # a real rendered report, not a stub


def test_pdf_export_bakes_disclaimer(client, finished_run):
    resp = client.get(f"/api/report/{finished_run['report_id']}/export.pdf")
    assert resp.status_code == 200
    assert resp.content[:5] == b"%PDF-"
    # matplotlib writes metadata uncompressed → the Subject text is findable,
    # and the drawn footer shares the same code path as the PNG pixels.
    assert b"not investment advice" in resp.content


def test_spec_export_import_roundtrip(client, finished_run):
    resp = client.get(f"/api/strategies/{finished_run['strategy_id']}/spec.json")
    assert resp.status_code == 200
    exported = resp.json()
    assert exported["spec_version"] == 1

    imported = client.post("/api/import-spec", json={"spec": exported}).json()
    assert imported["valid"], imported["errors"]
    assert imported["expanded_spec"] == exported  # byte-for-byte fixed point


def test_import_rejects_garbage(client):
    imported = client.post("/api/import-spec",
                           json={"spec": {**EXAMPLE_MOMENTUM, "leverage": 3}}).json()
    assert imported["valid"] is False


def test_client_events_namespaced(client):
    ok = client.post("/api/event", json={"type": "ui_spec_card_shown"})
    assert ok.status_code == 200
    bad = client.post("/api/event", json={"type": "run_completed"})
    assert bad.status_code == 400


def test_verdict_text_stays_inside_banner():
    """Regression: matplotlib wrap=True wraps at the FIGURE edge, so verdict
    text used to run through the banner's right border before wrapping.
    Probe the exact banner geometry render_report uses and assert the drawn
    extents of long headlines and a 180-char sub stay inside the box for all
    three verdict colors. (Full-pipeline rendering with real data is covered
    by test_png_export_bakes_disclaimer.)"""
    import matplotlib.pyplot as plt

    import export as ex

    long_sub = ("You chose these tickers knowing how they performed. A ranked "
                "strategy on a hindsight universe cannot earn a green verdict "
                "— re-run it on the full S&P 500 and compare like for like.")
    cases = [
        ("amber", "Beats the benchmark — but your universe is hand-picked.",
         long_sub),
        ("green", "Survived the gauntlet. Beats the benchmark net of 20 bps "
                  "across regimes.",
         "Surviving a backtest is necessary, not sufficient. This is not "
         "investment advice."),
        ("red", "Killed: the edge does not survive realistic trading costs on "
                "this window.",
         long_sub),
    ]
    # Mirror render_report's figure/grid so the probe measures the real box.
    fig = plt.figure(figsize=(10, 12.5), dpi=150)
    grid = fig.add_gridspec(5, 1, height_ratios=[1.35, 0.8, 1.3, 2.2, 1.6],
                            hspace=0.55, left=0.09, right=0.96, top=0.95,
                            bottom=0.075)
    ax0 = fig.add_subplot(grid[0])
    ax0.axis("off")
    banner_px = ax0.get_position().width * fig.get_figwidth() * fig.dpi
    inner_px = banner_px * (1 - 2 * 0.03)
    box_right_px = ax0.get_position().x1 * fig.get_figwidth() * fig.dpi
    box_bottom_px = ax0.get_position().y0 * fig.get_figheight() * fig.dpi
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for color, headline, sub in cases:
        texts = [
            ex._wrap_to(ax0.text(0.03, 0.71, headline, fontsize=15,
                                 fontweight="bold", transform=ax0.transAxes,
                                 va="top"), inner_px),
            ex._wrap_to(ax0.text(0.03, 0.32, sub[:180], fontsize=8.5,
                                 transform=ax0.transAxes, va="top"), inner_px),
        ]
        fig.canvas.draw()
        head_ext = texts[0].get_window_extent(renderer)
        sub_ext = texts[1].get_window_extent(renderer)
        for label, ext in (("headline", head_ext), ("sub", sub_ext)):
            assert ext.x1 <= box_right_px + 1, (
                f"{color} {label} crosses banner right border "
                f"({ext.x1:.0f}px > {box_right_px:.0f}px)")
            assert ext.y0 >= box_bottom_px - 1, (
                f"{color} {label} crosses banner bottom border")
        # a two-line headline must not collide with the sub below it
        assert head_ext.y0 >= sub_ext.y1 - 1, (
            f"{color}: headline overlaps the sub-text")
        for t in texts:
            t.remove()
    plt.close(fig)
