"""API contract tests (Section 4): validation endpoint is pure, errors use
{"error": {code, message}} shape, watchlist upload catches bad tickers."""
import io

import pytest
from fastapi.testclient import TestClient

from main import app
from tests.test_validator import EXAMPLE_DIP, EXAMPLE_MOMENTUM


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_validate_spec_endpoint(client):
    r = client.post("/api/validate-spec", json={"spec": EXAMPLE_MOMENTUM})
    body = r.json()
    assert r.status_code == 200
    assert body["valid"] is True
    assert "readback" in body and "126 trading days" in body["readback"]
    assert body["universe_size"] > 400

    r2 = client.post("/api/validate-spec", json={"spec": {**EXAMPLE_DIP, "leverage": 2}})
    assert r2.json()["valid"] is False
    assert any("closed vocabulary" in e for e in r2.json()["errors"])


def test_validate_is_pure(client):
    """Two identical calls, identical answers, no side effects."""
    a = client.post("/api/validate-spec", json={"spec": EXAMPLE_DIP}).json()
    b = client.post("/api/validate-spec", json={"spec": EXAMPLE_DIP}).json()
    assert a == b


def test_watchlist_upload_flags_problems(client):
    content = b"AAPL\nMSFT\nFAKECO\nFB\n"
    r = client.post(
        "/api/universe/watchlist",
        files={"file": ("mylist.txt", io.BytesIO(content), "text/plain")},
    )
    body = r.json()
    assert r.status_code == 200
    assert set(body["valid"]) == {"AAPL", "MSFT"}
    problems = {p["ticker"]: p.get("suggestion", "") for p in body["problems"]}
    assert "FAKECO" in problems
    assert "META" in problems.get("FB", "")


def test_conventions_endpoint(client):
    r = client.get("/api/conventions")
    rows = r.json()["conventions"]
    assert any("12-1" in row["rule"] for row in rows)
