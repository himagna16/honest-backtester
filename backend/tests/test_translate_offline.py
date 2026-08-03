"""Offline translation-pipeline tests (no network, no API key).

The live model call is mocked; these verify the pipeline around it: caching,
validation + self-repair routing, readback source (template, never LLM),
event logging, and the not-configured error path.
"""
import json

import pytest

import llm.translate as tr
from llm.translate import (
    ClarifyingQuestion, TranslateError, TranslatorReply, translate,
)
from dsl.schema import StrategySpec


GOOD_SPEC = {
    "name": "6mo momentum top-5",
    "universe": {"preset": "sp500"},
    "selection": {"rank_by": {"type": "return", "lookback_days": 126},
                  "direction": "top", "n": 5},
    "weighting": "equal",
    "rebalance": "monthly",
    "benchmark": "SPY",
}


@pytest.fixture
def fake_key(monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "anthropic_api_key", "test-key-not-real")


def make_reply(**kw) -> TranslatorReply:
    defaults = dict(status="spec", spec=StrategySpec(**GOOD_SPEC),
                    assumptions=["Benchmark defaulted to SPY."])
    defaults.update(kw)
    return TranslatorReply(**defaults)


def test_requires_api_key(monkeypatch):
    """Blank EVERY provider key, not just Anthropic's.

    Clearing one provider used to be enough only because no developer had a
    key configured; once a real OPENAI_API_KEY existed in .env this test kept
    passing a key through and stopped testing anything. The outcome must not
    depend on which provider the machine happens to be set up for.
    """
    from config import settings
    for attr in ("anthropic_api_key", "openai_api_key", "gemini_api_key"):
        monkeypatch.setattr(settings, attr, "")
    with pytest.raises(TranslateError) as exc:
        translate("buy the top 5 momentum stocks")
    assert exc.value.code == "llm_not_configured"


def test_spec_path_uses_template_readback(fake_key, monkeypatch):
    calls = []

    def fake_call(messages):
        calls.append(messages)
        return make_reply()

    monkeypatch.setattr(tr, "_call_model", fake_call)
    result = translate("unique input alpha", use_cache=False)
    assert result["status"] == "spec"
    assert result["spec"]["spec_version"] == 1
    assert result["spec"]["window"]["start"]  # defaults expanded
    # readback is deterministic template output mentioning the conversion
    assert "126 trading days" in result["readback"]
    assert len(calls) == 1


def test_cache_roundtrip(fake_key, monkeypatch):
    calls = []
    monkeypatch.setattr(tr, "_call_model", lambda m: calls.append(m) or make_reply())

    first = translate("unique input beta", use_cache=True)
    second = translate("  Unique   input BETA ", use_cache=True)  # normalized hash
    assert first["cached"] is False
    assert second["cached"] is True
    assert second["spec"] == first["spec"]
    assert len(calls) == 1


def test_self_repair_round(fake_key, monkeypatch):
    bad = make_reply(spec=StrategySpec(**{
        **GOOD_SPEC,
        "selection": {"rank_by": {"type": "sma_cross", "fast": 200, "slow": 50},
                      "direction": "top", "n": 5},
    }))
    replies = [bad, make_reply()]
    seen = []

    def fake_call(messages):
        seen.append(messages)
        return replies.pop(0)

    monkeypatch.setattr(tr, "_call_model", fake_call)
    result = translate("unique input gamma", use_cache=False)
    assert result["status"] == "spec"
    assert len(seen) == 2
    # repair message carries the validator's plain-English error
    repair_text = seen[1][-1]["content"]
    assert "failed validation" in repair_text and "shorter" in repair_text


def test_second_failure_surfaces_error(fake_key, monkeypatch):
    bad_spec = StrategySpec(**{
        **GOOD_SPEC,
        "selection": {"rank_by": {"type": "sma_cross", "fast": 200, "slow": 50},
                      "direction": "top", "n": 5},
    })
    monkeypatch.setattr(tr, "_call_model", lambda m: make_reply(spec=bad_spec))
    result = translate("unique input delta", use_cache=False)
    assert result["status"] == "error"
    assert result["errors"]


def test_clarification_passthrough(fake_key, monkeypatch):
    reply = make_reply(
        status="needs_clarification", spec=None,
        questions=[ClarifyingQuestion(question="Over what lookback?",
                                      options=["63", "126", "252"])],
        assumptions=[],
    )
    monkeypatch.setattr(tr, "_call_model", lambda m: reply)
    result = translate("unique input epsilon", use_cache=False)
    assert result["status"] == "needs_clarification"
    assert len(result["questions"]) == 1


def test_unsupported_passthrough(fake_key, monkeypatch):
    reply = make_reply(status="unsupported", spec=None,
                       unsupported_reason="options aren't supported", assumptions=[])
    monkeypatch.setattr(tr, "_call_model", lambda m: reply)
    result = translate("unique input zeta", use_cache=False)
    assert result["status"] == "unsupported"
    assert "options" in result["unsupported_reason"]


def test_system_prompt_is_byte_stable():
    a, b = tr.system_prompt(), tr.system_prompt()
    assert a == b
    assert "126" in a and "{CONVENTIONS_TABLE}" not in a


@pytest.mark.parametrize("model,expected", [
    ("claude-sonnet-5", True),
    ("claude-opus-5", True),
    ("claude-sonnet-4-6", True),
    ("claude-haiku-4-5", False),      # cheaper tier: effort would 400
    ("claude-sonnet-4-5", False),
])
def test_effort_is_gated_by_model(model, expected):
    """Switching HONEST_LLM_MODEL to a cheaper tier must not 400 on the
    effort parameter, which those models reject."""
    assert tr.supports_effort(model) is expected
