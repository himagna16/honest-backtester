"""End-to-end run API: POST → background execution → SSE → report,
plus snapshot replay determinism (Gap #2 — also an M5 acceptance)."""
import pytest
from fastapi.testclient import TestClient

from main import app
from tests.test_audit import war_tickers


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


SPEC = {
    "name": "War momentum (api e2e)",
    "universe": {"preset": "user_watchlist"},
    "selection": {"rank_by": {"type": "return", "lookback_days": 126},
                  "direction": "top", "n": 5},
    "weighting": "equal", "rebalance": "monthly", "benchmark": "SPY",
    "window": {"start": "2024-06-03", "end": "2026-05-29"},
}


@pytest.fixture(scope="module")
def first_run(client):
    spec = {**SPEC, "universe": {"preset": "user_watchlist",
                                 "resolved_tickers": war_tickers()}}
    resp = client.post("/api/run-backtest", json={"spec": spec})
    assert resp.status_code == 202
    run_id = resp.json()["run_id"]
    # TestClient executes BackgroundTasks before returning → run is done
    status = client.get(f"/api/runs/{run_id}").json()
    assert status["status"] == "done", status.get("error")
    return status


def test_run_produces_report(client, first_run):
    report_id = first_run["report_id"]
    assert report_id
    rep = client.get(f"/api/report/{report_id}").json()
    assert rep["report"]["verdict"]["color"] in ("red", "amber", "green")
    assert rep["report"]["disclaimer"]
    assert rep["report"]["run"]["snapshot_hash"] == first_run["snapshot_hash"]


def test_invalid_spec_never_runs(client):
    resp = client.post("/api/run-backtest", json={"spec": {**SPEC, "leverage": 3}})
    assert resp.status_code == 422
    assert "nothing was run" in resp.json()["error"]["message"]


def test_sse_events_replay_after_completion(client, first_run):
    resp = client.get(f"/api/run-backtest/{first_run['id']}/events")
    assert resp.status_code == 200
    assert "Truth Report ready" in resp.text
    assert '"stage": "done"' in resp.text


def test_snapshot_replay_is_byte_identical(client, first_run):
    """Re-running from the snapshot reproduces the metrics byte-for-byte."""
    rerun = client.post("/api/run-backtest",
                        json={"rerun_of": first_run["id"]}).json()
    status = client.get(f"/api/runs/{rerun['run_id']}").json()
    assert status["status"] == "done", status.get("error")
    assert status["data_source"] == f"snapshot:{first_run['id']}"
    assert status["snapshot_hash"] == first_run["snapshot_hash"]

    rep_a = client.get(f"/api/report/{first_run['report_id']}").json()["report"]
    rep_b = client.get(f"/api/report/{status['report_id']}").json()["report"]
    for key in ("metrics", "gross_metrics", "net20_metrics", "benchmark",
                "cost_sweep", "equity_curve", "oos", "sensitivity"):
        assert rep_a[key] == rep_b[key], key
    # attempt counter moved on — disclosed, not hidden
    assert rep_b["multiple_testing"]["attempts"] > rep_a["multiple_testing"]["attempts"]


def test_strategy_and_versions_created(client, first_run):
    strategies = client.get("/api/strategies").json()
    match = [s for s in strategies if s["name"] == "War momentum (api e2e)"]
    assert match and match[0]["runs"] >= 2
