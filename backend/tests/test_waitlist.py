"""Waitlist capture — the validation-sprint instrument."""
import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_signup_and_idempotent_resubmit(client):
    r1 = client.post("/api/waitlist", json={"email": "Sai@Example.com ",
                                            "note": "12-1 momentum lied to me"})
    assert r1.status_code == 201
    assert r1.json() == {"ok": True, "already": False}

    # same address, different casing/whitespace → not a duplicate row
    r2 = client.post("/api/waitlist", json={"email": "sai@example.com"})
    assert r2.json()["already"] is True


@pytest.mark.parametrize("email", ["", "nope", "a@b", "no-at-sign.com", "x@y@z.com"])
def test_rejects_bad_emails(client, email):
    r = client.post("/api/waitlist", json={"email": email})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_email"


def test_stats_reports_progress_not_addresses(client):
    client.post("/api/waitlist", json={"email": "another@example.com"})
    stats = client.get("/api/waitlist/stats").json()
    assert stats["total"] >= 2
    assert stats["target"] == 50          # the brief's go/no-go bar
    assert stats["met"] is (stats["total"] >= 50)
    # counts only — no addresses leak through this endpoint
    assert "emails" not in stats and "email" not in stats


def test_csv_export_is_local_and_complete(client):
    r = client.get("/api/waitlist/export.csv")
    assert r.status_code == 200
    assert r.text.splitlines()[0] == "email,source,note,created_at"
    assert "sai@example.com" in r.text
    # notes with quotes must not break the CSV
    client.post("/api/waitlist", json={"email": "quoted@example.com",
                                       "note": 'it said "+1650%"'})
    body = client.get("/api/waitlist/export.csv").text
    assert '""+1650%""' in body
