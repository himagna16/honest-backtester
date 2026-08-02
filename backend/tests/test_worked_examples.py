"""Definition of Done: the three worked-example strategies (Section 2) run
end-to-end through the API to complete Truth Reports, offline, on the
seeded fixture data. (The English → spec leg is covered by the eval harness;
here the specs enter at the confirm-and-run gate.)"""
import pytest
from fastapi.testclient import TestClient

from main import app
from tests.test_validator import EXAMPLE_6040, EXAMPLE_DIP, EXAMPLE_MOMENTUM


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def run_to_report(client, spec: dict) -> dict:
    resp = client.post("/api/run-backtest", json={"spec": spec})
    assert resp.status_code == 202, resp.text
    run_id = resp.json()["run_id"]
    status = client.get(f"/api/runs/{run_id}").json()
    assert status["status"] == "done", f"{spec['name']}: {status.get('error')}"
    report = client.get(f"/api/report/{status['report_id']}").json()["report"]
    # Every report: verdict with icon+text, benchmark, sweep, audit, OOS,
    # regimes, curves, disclaimer — the complete Truth Report shape.
    assert report["verdict"]["color"] in ("red", "amber", "green")
    assert report["verdict"]["icon"] and report["verdict"]["headline"]
    assert report["benchmark"]["total_return"] is not None
    assert len([r for r in report["cost_sweep"] if r["bps"] is not None]) >= 3
    assert report["audit"] and report["oos"]["split_date"]
    assert report["regimes"]["segments"]
    assert len(report["equity_curve"]) > 100
    assert "not investment advice" in report["disclaimer"].lower()
    return report


def test_worked_example_momentum(client):
    spec = {**EXAMPLE_MOMENTUM, "window": {"start": "2016-01-04", "end": "2026-01-02"}}
    report = run_to_report(client, spec)
    # current-members survivorship caveat always disclosed on presets
    assert any(f["check"] == "survivorship_universe" for f in report["audit"])


def test_worked_example_dip_buyer(client):
    spec = {**EXAMPLE_DIP, "window": {"start": "2019-01-02", "end": "2024-12-31"}}
    report = run_to_report(client, spec)
    assert report["spec"]["entry_exit"]["hold_days"] == 5
    assert report["metrics"]["annual_turnover"] is not None


def test_worked_example_sixty_forty(client):
    spec = {**EXAMPLE_6040, "window": {"start": "2016-01-04", "end": "2026-01-02"}}
    report = run_to_report(client, spec)
    assert report["benchmark"]["name"] == "sixty_forty"
    # a static allocation vs its own blend should track it closely
    assert abs(report["metrics"]["total_return"]
               - report["benchmark"]["total_return"]) < 0.15
