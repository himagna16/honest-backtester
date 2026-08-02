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
