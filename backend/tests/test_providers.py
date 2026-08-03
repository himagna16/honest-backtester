"""Provider-layer tests with mocked SDK clients.

No network, no API keys. These verify the parts that would otherwise only be
exercised by a paid live call: routing, usage accounting, cost math, the
JSON-mode fallback, and the effort gate.
"""
import types

import pytest

from llm import providers
from llm.pricing import BY_KEY, CATALOGUE, ModelSpec
from llm.reply import TranslatorReply

GOOD = {
    "status": "spec",
    "spec": {
        "name": "6mo momentum top-5",
        "universe": {"preset": "sp500"},
        "selection": {"rank_by": {"type": "return", "lookback_days": 126},
                      "direction": "top", "n": 5},
        "weighting": "equal", "rebalance": "monthly", "benchmark": "SPY",
    },
    "assumptions": ["Benchmark defaulted to SPY."],
    "questions": [], "unsupported_reason": None,
}


# ---------------------------------------------------------------- pricing

def test_catalogue_keys_unique_and_wellformed():
    keys = [m.key for m in CATALOGUE]
    assert len(keys) == len(set(keys))
    for m in CATALOGUE:
        assert m.provider in ("anthropic", "openai", "google")
        assert m.price_in > 0 and m.price_out > 0
        assert m.model_id


def test_cost_math_uncached():
    spec = ModelSpec("t", "openai", "t", price_in=1.0, price_out=10.0)
    # 2000 in @ $1/M + 500 out @ $10/M
    assert spec.cost(2000, 500) == pytest.approx(0.002 + 0.005)


def test_cost_math_applies_cache_discount():
    spec = ModelSpec("t", "openai", "t", price_in=1.0, price_out=10.0, cached_in=0.1)
    # 1500 cached @ $0.1/M + 500 fresh @ $1/M + 500 out @ $10/M
    got = spec.cost(2000, 500, cached_tokens=1500)
    assert got == pytest.approx(0.00015 + 0.0005 + 0.005)
    assert got < spec.cost(2000, 500)


def test_no_cache_rate_falls_back_to_full_price():
    spec = ModelSpec("t", "google", "t", price_in=1.0, price_out=10.0, cached_in=None)
    assert spec.cost(2000, 500, cached_tokens=2000) == spec.cost(2000, 500)


# ---------------------------------------------------------------- routing

def test_unknown_model_rejected():
    with pytest.raises(providers.ProviderError) as e:
        providers.call_model("not-a-model", "k", "sys", [])
    assert e.value.code == "unknown_model"


def test_missing_key_rejected():
    with pytest.raises(providers.ProviderError) as e:
        providers.call_model("gpt-5-nano", "", "sys", [])
    assert e.value.code == "no_api_key"


@pytest.mark.parametrize("model,expected", [
    ("claude-sonnet-5", True), ("claude-opus-5", True),
    ("claude-haiku-4-5", False), ("claude-sonnet-4-5", False),
])
def test_effort_gate(model, expected):
    assert providers.supports_effort(model) is expected


# ---------------------------------------------------------------- OpenAI

def _fake_openai(monkeypatch, *, parse_ok=True, cached=0, create_content=None,
                 finish_reason="stop"):
    calls = {"parse": 0, "create": 0, "parse_kw": None, "create_kw": None}
    content = ('{"status":"unsupported","unsupported_reason":"options"}'
               if create_content is None else create_content)

    class Msg:
        refusal = None
        parsed = TranslatorReply.model_validate(GOOD) if parse_ok else None

    class CreateMsg:
        refusal = None

    class Choice:
        def __init__(self, msg):
            self.message = msg
            self.finish_reason = finish_reason

    class Details:
        cached_tokens = cached

    class Usage:
        prompt_tokens = 2500
        completion_tokens = 400
        prompt_tokens_details = Details()

    def _resp(msg):
        return types.SimpleNamespace(choices=[Choice(msg)], usage=Usage())

    class Completions:
        def parse(self, **kw):
            calls["parse"] += 1
            calls["parse_kw"] = kw
            if not parse_ok:
                raise RuntimeError("schema rejected by strict mode")
            return _resp(Msg())

        def create(self, **kw):
            calls["create"] += 1
            calls["create_kw"] = kw
            m = CreateMsg()
            m.content = content
            return _resp(m)

    class Client:
        def __init__(self, **kw):
            self.chat = types.SimpleNamespace(completions=Completions())

    monkeypatch.setattr("openai.OpenAI", Client)
    return calls


def test_openai_native_path(monkeypatch):
    calls = _fake_openai(monkeypatch, cached=2000)
    reply, usage = providers.call_model(
        "gpt-5-nano", "key", "sys", [{"role": "user", "content": "hi"}])
    assert reply.status == "spec"
    assert usage.path == "native"
    assert usage.input_tokens == 2500 and usage.output_tokens == 400
    assert usage.cached_tokens == 2000
    assert calls["parse"] == 1 and calls["create"] == 0
    # cached tokens must make it cheaper than the all-fresh price
    spec = BY_KEY["gpt-5-nano"]
    assert spec.cost(2500, 400, 2000) < spec.cost(2500, 400)


def test_openai_falls_back_to_json_mode(monkeypatch):
    calls = _fake_openai(monkeypatch, parse_ok=False)
    reply, usage = providers.call_model(
        "gpt-5-nano", "key", "sys", [{"role": "user", "content": "hi"}])
    # measures the model, not the SDK's schema translation
    assert usage.path == "json_fallback"
    assert reply.status == "unsupported"
    assert calls["parse"] == 1 and calls["create"] == 1


def test_gpt5_asks_for_minimal_reasoning(monkeypatch):
    """gpt-5* reasoning models must be told to reason minimally, or reasoning
    tokens exhaust the completion budget and the reply comes back empty."""
    calls = _fake_openai(monkeypatch)
    providers.call_model("gpt-5-nano", "key", "sys",
                         [{"role": "user", "content": "hi"}])
    assert calls["parse_kw"].get("reasoning_effort") == "minimal"


def test_non_reasoning_model_omits_effort(monkeypatch):
    """gpt-4.1-nano rejects reasoning_effort — it must not be sent."""
    calls = _fake_openai(monkeypatch)
    providers.call_model("gpt-4.1-nano", "key", "sys",
                         [{"role": "user", "content": "hi"}])
    assert "reasoning_effort" not in calls["parse_kw"]


def test_empty_content_raises_clear_token_limit_error(monkeypatch):
    """A blank fallback reply (budget exhausted) must surface a clear error,
    not the cryptic 'model returned non-JSON: Expecting value line 1 col 1'."""
    _fake_openai(monkeypatch, parse_ok=False, create_content="",
                 finish_reason="length")
    with pytest.raises(providers.ProviderError) as exc:
        providers.call_model("gpt-5-nano", "key", "sys",
                             [{"role": "user", "content": "hi"}])
    assert exc.value.code == "token_limit"
    assert "non-JSON" not in str(exc.value)


# ---------------------------------------------------------------- Gemini

def _fake_gemini(monkeypatch, *, native_ok=True):
    calls = {"n": 0}

    class Meta:
        prompt_token_count = 2400
        candidates_token_count = 380
        cached_content_token_count = 0

    class Response:
        parsed = TranslatorReply.model_validate(GOOD) if native_ok else None
        text = '{"status":"needs_clarification","questions":[{"question":"n?","options":["5"]}]}'
        usage_metadata = Meta()

    class Models:
        def generate_content(self, **kw):
            calls["n"] += 1
            if not native_ok and calls["n"] == 1:
                raise RuntimeError("response_schema unsupported")
            return Response()

    class Client:
        def __init__(self, **kw):
            self.models = Models()

    monkeypatch.setattr("google.genai.Client", Client)
    return calls


def test_gemini_native_path(monkeypatch):
    _fake_gemini(monkeypatch)
    reply, usage = providers.call_model(
        "gemini-2.5-flash", "key", "sys", [{"role": "user", "content": "hi"}])
    assert reply.status == "spec"
    assert usage.input_tokens == 2400 and usage.output_tokens == 380
    assert usage.path == "native"


def test_gemini_falls_back(monkeypatch):
    calls = _fake_gemini(monkeypatch, native_ok=False)
    reply, usage = providers.call_model(
        "gemini-2.5-flash", "key", "sys", [{"role": "user", "content": "hi"}])
    assert usage.path == "json_fallback"
    assert reply.status == "needs_clarification"
    assert calls["n"] == 2


# ---------------------------------------------------------------- Anthropic

def test_anthropic_counts_cached_and_fresh_input(monkeypatch):
    class U:
        input_tokens = 500
        output_tokens = 400
        cache_read_input_tokens = 2000
        cache_creation_input_tokens = 0

    class Response:
        stop_reason = "end_turn"
        parsed_output = TranslatorReply.model_validate(GOOD)
        usage = U()

    class Messages:
        def parse(self, **kw):
            # the cheap Anthropic tier must not receive the effort param
            assert "output_config" not in kw
            return Response()

    class Client:
        def __init__(self, **kw):
            self.messages = Messages()

    monkeypatch.setattr("anthropic.Anthropic", Client)
    reply, usage = providers.call_model(
        "claude-haiku-4-5", "key", "sys", [{"role": "user", "content": "hi"}])
    assert reply.status == "spec"
    # total prompt tokens = fresh + cached (so cost math sees the whole prompt)
    assert usage.input_tokens == 2500
    assert usage.cached_tokens == 2000


def test_anthropic_refusal_surfaces(monkeypatch):
    class Response:
        stop_reason = "refusal"
        parsed_output = None
        usage = types.SimpleNamespace(input_tokens=10, output_tokens=0)

    class Client:
        def __init__(self, **kw):
            self.messages = types.SimpleNamespace(parse=lambda **kw: Response())

    monkeypatch.setattr("anthropic.Anthropic", Client)
    with pytest.raises(providers.ProviderError) as e:
        providers.call_model("claude-sonnet-5", "key", "sys",
                             [{"role": "user", "content": "hi"}])
    assert e.value.code == "model_refusal"


# ---------------------------------------------------------------- validation

def test_validate_rejects_non_json():
    with pytest.raises(providers.ProviderError) as e:
        providers._validate("not json at all")
    assert e.value.code == "parse_failed"


def test_validate_rejects_wrong_shape():
    with pytest.raises(providers.ProviderError) as e:
        providers._validate('{"status": "banana"}')
    assert e.value.code == "parse_failed"
