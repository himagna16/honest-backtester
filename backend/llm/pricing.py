"""Model catalogue + cost math for the translator bench.

Prices are USD per 1M tokens, copied from the providers' own pricing pages on
the date below. THEY GO STALE — re-check before trusting a cost comparison,
and edit here rather than scattering numbers through the code.

`cached_in` is the discounted rate for a cached prompt prefix. Our system
prompt is ~2.4K tokens, so whether a provider caches it at all materially
changes the real cost (see notes on each row).
"""
from __future__ import annotations

from dataclasses import dataclass

PRICES_FETCHED = "2026-08-03"
PRICE_SOURCES = {
    "anthropic": "https://platform.claude.com/docs/en/pricing",
    "openai": "https://developers.openai.com/api/docs/pricing",
    "google": "https://ai.google.dev/gemini-api/docs/pricing",
}


@dataclass(frozen=True)
class ModelSpec:
    key: str            # short name used on the CLI
    provider: str       # anthropic | openai | google
    model_id: str       # exact API model id
    price_in: float     # $/1M input tokens
    price_out: float    # $/1M output tokens
    cached_in: float | None = None   # $/1M cached input, None = no prompt cache
    note: str = ""

    def cost(self, input_tokens: int, output_tokens: int,
             cached_tokens: int = 0) -> float:
        fresh = max(input_tokens - cached_tokens, 0)
        cached_rate = self.cached_in if self.cached_in is not None else self.price_in
        return (
            fresh * self.price_in / 1_000_000
            + cached_tokens * cached_rate / 1_000_000
            + output_tokens * self.price_out / 1_000_000
        )


# Ordered cheapest-first by list input price.
CATALOGUE: list[ModelSpec] = [
    ModelSpec("gpt-5-nano", "openai", "gpt-5-nano", 0.05, 0.40, 0.005,
              "cheapest listed OpenAI tier"),
    ModelSpec("gpt-4.1-nano", "openai", "gpt-4.1-nano", 0.10, 0.40, 0.025),
    ModelSpec("gemini-2.5-flash-lite", "google", "gemini-2.5-flash-lite",
              0.10, 0.40, None,
              "RETIRES 2026-10-16 — do not build on this"),
    ModelSpec("gpt-5-mini", "openai", "gpt-5-mini", 0.25, 2.00, 0.025),
    ModelSpec("gemini-3.1-flash-lite", "google", "gemini-3.1-flash-lite",
              0.25, 1.50, None, "successor to 2.5-flash-lite"),
    ModelSpec("gemini-2.5-flash", "google", "gemini-2.5-flash", 0.30, 2.50, None),
    ModelSpec("gemini-3.5-flash-lite", "google", "gemini-3.5-flash-lite",
              0.30, 2.50, None),
    ModelSpec("gpt-5.4-nano", "openai", "gpt-5.4-nano", 0.20, 1.25, 0.02),
    ModelSpec("gpt-5.6-luna", "openai", "gpt-5.6-luna", 0.20, 1.20, 0.02),
    ModelSpec("claude-haiku-4-5", "anthropic", "claude-haiku-4-5", 1.00, 5.00, 0.10,
              "prompt cache needs a >=4096-token prefix; ours is ~2.4K so it "
              "will NOT cache — treat cached_in as unreachable here"),
    ModelSpec("gemini-3.6-flash", "google", "gemini-3.6-flash", 1.50, 7.50, None),
    ModelSpec("claude-sonnet-5", "anthropic", "claude-sonnet-5", 3.00, 15.00, 0.30,
              "current default; caches at >=1024 tokens so the prefix does hit "
              "cache on repeat calls"),
]

BY_KEY = {m.key: m for m in CATALOGUE}

# Sensible default bench set: the cheap tiers from each provider plus the
# incumbent, so the table answers "can I go cheaper without losing accuracy".
DEFAULT_BENCH = [
    "gpt-5-nano", "gpt-5-mini", "gemini-3.1-flash-lite", "gemini-2.5-flash",
    "claude-haiku-4-5", "claude-sonnet-5",
]

PROVIDER_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GEMINI_API_KEY",
}


def estimate(spec: ModelSpec, input_tokens: int = 2500, output_tokens: int = 500,
             assume_cached: bool = False) -> float:
    """Back-of-envelope cost for one translation, before measuring."""
    cached = input_tokens if (assume_cached and spec.cached_in is not None) else 0
    return spec.cost(input_tokens, output_tokens, cached)
