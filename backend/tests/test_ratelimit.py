"""The paid translation endpoint is rate-limited so a public deploy can't have
its OpenAI budget drained by abuse. The real money ceiling is OpenAI's billing
cap; these limits keep it from being hit."""
import types

import pytest
from fastapi.testclient import TestClient

from config import settings
from main import app
import llm.translate as tr
from ratelimit import limiter


@pytest.fixture()
def client():
    limiter.reset()
    with TestClient(app) as c:
        yield c
    limiter.reset()


@pytest.fixture()
def fake_translate(monkeypatch):
    """Translation itself is stubbed — we test the gate, not the model."""
    monkeypatch.setattr(tr, "translate",
                        lambda text, use_cache=True: {"status": "spec", "spec": {}})
    # route imported the name directly
    import routes.translate as rt
    monkeypatch.setattr(rt, "translate",
                        lambda text, use_cache=True: {"status": "spec", "spec": {}})


def test_per_minute_burst_is_capped(client, fake_translate, monkeypatch):
    monkeypatch.setattr(settings, "rl_per_min", 3)
    monkeypatch.setattr(settings, "rl_per_day", 1000)
    monkeypatch.setattr(settings, "rl_global_per_day", 100000)
    ok = 0
    for _ in range(10):
        r = client.post("/api/translate-strategy", json={"text": "buy top 5"})
        if r.status_code == 200:
            ok += 1
        else:
            assert r.status_code == 429
            assert r.json()["error"]["code"] == "rate_limited"
            assert r.headers.get("Retry-After") == "60"
            break
    assert ok == 3, "exactly the per-minute allowance should pass"


def test_global_daily_budget_guard(client, fake_translate, monkeypatch):
    monkeypatch.setattr(settings, "rl_per_min", 10000)
    monkeypatch.setattr(settings, "rl_per_day", 10000)
    monkeypatch.setattr(settings, "rl_global_per_day", 5)
    codes = [client.post("/api/translate-strategy", json={"text": "x"}).status_code
             for _ in range(7)]
    assert codes.count(200) == 5
    assert codes.count(429) == 2  # the budget guard stops the rest


def test_limit_is_per_ip(client, fake_translate, monkeypatch):
    monkeypatch.setattr(settings, "rl_per_min", 2)
    monkeypatch.setattr(settings, "rl_per_day", 1000)
    monkeypatch.setattr(settings, "rl_global_per_day", 100000)
    # exhaust one IP
    for _ in range(2):
        client.post("/api/translate-strategy", json={"text": "x"},
                    headers={"X-Forwarded-For": "1.1.1.1"})
    blocked = client.post("/api/translate-strategy", json={"text": "x"},
                          headers={"X-Forwarded-For": "1.1.1.1"})
    assert blocked.status_code == 429
    # a different IP is unaffected
    other = client.post("/api/translate-strategy", json={"text": "x"},
                        headers={"X-Forwarded-For": "2.2.2.2"})
    assert other.status_code == 200


def test_openai_quota_exhausted_is_a_clean_503(monkeypatch):
    """When the OpenAI budget runs out, the user sees a clear message, not 500."""
    import openai
    from llm import providers
    from llm.pricing import BY_KEY

    class Boom:
        class chat:
            class completions:
                @staticmethod
                def parse(**kw):
                    raise openai.RateLimitError(
                        "insufficient_quota", response=types.SimpleNamespace(
                            status_code=429, headers={}, request=None), body=None)

    class Client:
        def __init__(self, **kw):
            self.chat = Boom.chat()

    monkeypatch.setattr("openai.OpenAI", Client)
    with pytest.raises(providers.ProviderError) as exc:
        providers.call_model("gpt-5-nano", "key", "sys",
                             [{"role": "user", "content": "hi"}])
    assert exc.value.code == "quota_exhausted"


def test_loopback_is_exempt(client, fake_translate, monkeypatch):
    """Local dev must never be throttled — real users arrive via X-Forwarded-For."""
    monkeypatch.setattr(settings, "rl_per_min", 1)
    monkeypatch.setattr(settings, "rl_global_per_day", 100000)
    codes = [client.post("/api/translate-strategy", json={"text": "x"},
                         headers={"X-Forwarded-For": "127.0.0.1"}).status_code
             for _ in range(5)]
    assert codes == [200] * 5, "loopback IP should never be rate-limited"
