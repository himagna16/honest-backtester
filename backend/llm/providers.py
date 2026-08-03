"""Provider adapters: Anthropic, OpenAI, Google Gemini.

Every adapter takes (system prompt, messages) and returns a validated
TranslatorReply plus token usage, so the bench can compare accuracy AND cost
on identical inputs.

Two paths per provider:
  1. native structured output (the model is constrained to our schema), and
  2. a JSON-mode fallback that validates with Pydantic ourselves.
The fallback exists so a provider's schema-translation quirks don't get
scored as "this model can't translate strategies" — we want to measure the
model, not the SDK. Which path ran is recorded on the Usage.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from pydantic import ValidationError

from llm.pricing import BY_KEY, ModelSpec
from llm.reply import TranslatorReply


class ProviderError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    latency_s: float = 0.0
    path: str = "native"          # native | json_fallback
    raw: dict = field(default_factory=dict)


MAX_OUTPUT_TOKENS = 4000

# Effort is rejected (400) by the cheaper Anthropic tiers.
_NO_EFFORT_PREFIXES = ("claude-haiku", "claude-sonnet-4-5", "claude-3")


def supports_effort(model: str) -> bool:
    return not model.startswith(_NO_EFFORT_PREFIXES)


def _schema_hint(system: str) -> str:
    """Appended in JSON-fallback mode so the model still knows the shape."""
    schema = json.dumps(TranslatorReply.model_json_schema())
    return (
        f"{system}\n\nReturn ONLY a JSON object conforming to this JSON Schema. "
        f"No prose, no markdown fences.\n{schema}"
    )


def _validate(payload: str | dict) -> TranslatorReply:
    try:
        data = json.loads(payload) if isinstance(payload, str) else payload
    except json.JSONDecodeError as exc:
        raise ProviderError("parse_failed", f"model returned non-JSON: {exc}") from exc
    try:
        return TranslatorReply.model_validate(data)
    except ValidationError as exc:
        raise ProviderError(
            "parse_failed", f"reply did not match the translator schema: {exc}"
        ) from exc


# ----------------------------------------------------------------- Anthropic

def _call_anthropic(spec: ModelSpec, api_key: str, system: str,
                    messages: list[dict]) -> tuple[TranslatorReply, Usage]:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key, timeout=90.0, max_retries=2)
    kwargs: dict = {}
    if supports_effort(spec.model_id):
        kwargs["output_config"] = {"effort": "low"}

    t0 = time.perf_counter()
    response = client.messages.parse(
        model=spec.model_id,
        max_tokens=MAX_OUTPUT_TOKENS,
        system=[{"type": "text", "text": system,
                 "cache_control": {"type": "ephemeral"}}],
        messages=messages,
        output_format=TranslatorReply,
        **kwargs,
    )
    latency = time.perf_counter() - t0

    if response.stop_reason == "refusal":
        raise ProviderError("model_refusal", "the model declined this request")
    parsed = response.parsed_output
    if parsed is None:
        raise ProviderError("parse_failed", "no parsable reply")

    u = response.usage
    cached = getattr(u, "cache_read_input_tokens", 0) or 0
    return parsed, Usage(
        # Anthropic reports uncached and cached input separately; the bench
        # wants the total the prompt actually cost us.
        input_tokens=(u.input_tokens or 0) + cached
        + (getattr(u, "cache_creation_input_tokens", 0) or 0),
        output_tokens=u.output_tokens or 0,
        cached_tokens=cached,
        latency_s=latency,
    )


# -------------------------------------------------------------------- OpenAI

def _openai_reasoning_effort(model_id: str) -> str | None:
    """Reasoning models (gpt-5*) spend the completion budget on hidden reasoning
    tokens BEFORE writing any answer — gpt-5-nano burns ~2,000-2,600 on a single
    translation. With a 4k ceiling that can leave nothing for the actual reply:
    empty content, surfaced as "model returned non-JSON".

    Translation is extraction, not problem-solving, so ask for minimal reasoning.
    It keeps the whole budget for output, and is faster and ~10x cheaper. Older
    non-reasoning models (gpt-4.1-nano) reject the parameter, so gate on gpt-5*.
    """
    return "minimal" if model_id.startswith("gpt-5") else None


def _call_openai(spec: ModelSpec, api_key: str, system: str,
                 messages: list[dict]) -> tuple[TranslatorReply, Usage]:
    import openai
    from openai import OpenAI

    def _guard(exc: Exception) -> None:
        # Turn provider-budget / bad-key failures into clear, typed errors so
        # the app can say "translation unavailable" instead of a raw 500 when
        # the OpenAI spending cap is reached.
        if isinstance(exc, openai.RateLimitError):
            raise ProviderError(
                "quota_exhausted",
                "the OpenAI budget or rate limit is exhausted — translation is "
                "temporarily unavailable",
            ) from exc
        if isinstance(exc, openai.AuthenticationError):
            raise ProviderError(
                "llm_not_configured", "the OpenAI API key was rejected") from exc

    client = OpenAI(api_key=api_key, timeout=90.0, max_retries=2)
    chat = [{"role": "system", "content": system}, *messages]

    reason: dict = {}
    effort = _openai_reasoning_effort(spec.model_id)
    if effort:
        reason["reasoning_effort"] = effort

    t0 = time.perf_counter()
    path = "native"
    try:
        response = client.chat.completions.parse(
            model=spec.model_id,
            messages=chat,
            response_format=TranslatorReply,
            max_completion_tokens=MAX_OUTPUT_TOKENS,
            **reason,
        )
        message = response.choices[0].message
        if getattr(message, "refusal", None):
            raise ProviderError("model_refusal", message.refusal)
        # Empty content with finish_reason "length" means the token budget was
        # exhausted (usually by reasoning) — say so plainly instead of letting a
        # blank string fall through to a cryptic JSON error.
        if getattr(response.choices[0], "finish_reason", None) == "length" \
                and message.parsed is None:
            raise ProviderError(
                "token_limit",
                f"{spec.model_id} hit the {MAX_OUTPUT_TOKENS}-token limit before "
                "returning a spec — raise MAX_OUTPUT_TOKENS or lower reasoning effort",
            )
        parsed = message.parsed
        if parsed is None:
            raise ProviderError("parse_failed", "no parsable reply")
    except ProviderError:
        raise
    except Exception as exc:
        # A budget/auth failure is terminal — don't waste a second call on the
        # fallback; surface it as a clear typed error.
        _guard(exc)
        # otherwise: schema rejected by strict mode → measure the model, not the SDK
        path = "json_fallback"
        try:
            response = client.chat.completions.create(
                model=spec.model_id,
                messages=[{"role": "system", "content": _schema_hint(system)}, *messages],
                response_format={"type": "json_object"},
                max_completion_tokens=MAX_OUTPUT_TOKENS,
                **reason,
            )
        except Exception as exc2:
            _guard(exc2)
            raise
        content = response.choices[0].message.content or ""
        if not content.strip():
            raise ProviderError(
                "token_limit",
                f"{spec.model_id} returned no content (finish_reason="
                f"{response.choices[0].finish_reason}) — the token budget was "
                "exhausted before any JSON was written",
            )
        parsed = _validate(content)
    latency = time.perf_counter() - t0

    u = response.usage
    details = getattr(u, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", 0) or 0 if details else 0
    return parsed, Usage(
        input_tokens=u.prompt_tokens or 0,
        output_tokens=u.completion_tokens or 0,
        cached_tokens=cached,
        latency_s=latency,
        path=path,
    )


# -------------------------------------------------------------------- Google

def _call_gemini(spec: ModelSpec, api_key: str, system: str,
                 messages: list[dict]) -> tuple[TranslatorReply, Usage]:
    from google import genai

    client = genai.Client(api_key=api_key)
    contents = [
        {"role": "user" if m["role"] == "user" else "model",
         "parts": [{"text": m["content"]}]}
        for m in messages
    ]

    t0 = time.perf_counter()
    path = "native"
    try:
        response = client.models.generate_content(
            model=spec.model_id,
            contents=contents,
            config={
                "system_instruction": system,
                "response_mime_type": "application/json",
                "response_schema": TranslatorReply,
                "max_output_tokens": MAX_OUTPUT_TOKENS,
            },
        )
        parsed = getattr(response, "parsed", None)
        if parsed is None:
            parsed = _validate(response.text or "")
    except ProviderError:
        raise
    except Exception:
        path = "json_fallback"
        response = client.models.generate_content(
            model=spec.model_id,
            contents=contents,
            config={
                "system_instruction": _schema_hint(system),
                "response_mime_type": "application/json",
                "max_output_tokens": MAX_OUTPUT_TOKENS,
            },
        )
        parsed = _validate(response.text or "")
    latency = time.perf_counter() - t0

    meta = getattr(response, "usage_metadata", None)
    return parsed, Usage(
        input_tokens=getattr(meta, "prompt_token_count", 0) or 0,
        output_tokens=getattr(meta, "candidates_token_count", 0) or 0,
        cached_tokens=getattr(meta, "cached_content_token_count", 0) or 0,
        latency_s=latency,
        path=path,
    )


_DISPATCH = {
    "anthropic": _call_anthropic,
    "openai": _call_openai,
    "google": _call_gemini,
}


def call_model(model_key: str, api_key: str, system: str,
               messages: list[dict]) -> tuple[TranslatorReply, Usage]:
    spec = BY_KEY.get(model_key)
    if spec is None:
        raise ProviderError("unknown_model",
                            f"{model_key} is not in llm/pricing.py CATALOGUE")
    if not api_key:
        raise ProviderError("no_api_key", f"no API key for {spec.provider}")
    return _DISPATCH[spec.provider](spec, api_key, system, messages)
