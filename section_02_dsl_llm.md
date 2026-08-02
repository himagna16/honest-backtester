## Strategy DSL & LLM Layer

### Design philosophy

The DSL is a **closed vocabulary of building blocks**, not a programming language. Every strategy the engine can run is expressible as a single JSON document validated against a fixed schema. The LLM's only job is translation from English to that document — it never generates code, never touches the backtest engine, and never sees price data. This is what makes v0 safe to build alone: the worst a bad translation can do is produce a spec the validator rejects or a backtest that answers the wrong question (which the confirmation step catches).

### The strategy spec (JSON Schema sketch)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "additionalProperties": false,
  "required": ["name", "universe", "selection", "weighting", "rebalance", "costs", "benchmark"],
  "properties": {
    "name":        { "type": "string", "maxLength": 80 },
    "universe": {
      "type": "object", "additionalProperties": false,
      "properties": {
        "preset":  { "enum": ["sp500", "nasdaq100", "sector_etfs", "dow30", "user_watchlist"] },
        "watchlist_file": { "type": "string" }
      },
      "required": ["preset"]
    },
    "filters": {
      "type": "array", "maxItems": 5,
      "items": {
        "type": "object", "additionalProperties": false,
        "required": ["signal", "op", "value"],
        "properties": {
          "signal": { "$ref": "#/$defs/signal" },
          "op":     { "enum": ["gt", "lt", "gte", "lte", "between"] },
          "value":  { "type": ["number", "array"] }
        }
      }
    },
    "selection": {
      "type": "object", "additionalProperties": false,
      "properties": {
        "rank_by":   { "$ref": "#/$defs/signal" },
        "direction": { "enum": ["top", "bottom"] },
        "n":         { "type": "integer", "minimum": 1, "maximum": 50 },
        "static_allocation": {
          "type": "array",
          "items": { "type": "object", "properties": {
            "ticker": {"type": "string"}, "weight": {"type": "number"} } }
        }
      }
    },
    "weighting":  { "enum": ["equal", "rank", "inverse_vol"] },
    "rebalance":  { "enum": ["daily", "weekly", "monthly", "quarterly", "annual"] },
    "entry_exit": {
      "type": "object", "additionalProperties": false,
      "properties": {
        "hold_days":     { "type": "integer", "minimum": 1, "maximum": 252 },
        "stop_loss_pct": { "type": "number" },
        "take_profit_pct": { "type": "number" }
      }
    },
    "costs": {
      "type": "object", "additionalProperties": false,
      "properties": {
        "sweep_bps": { "type": "array", "default": [0, 5, 20] }
      }
    },
    "benchmark":  { "enum": ["SPY", "QQQ", "sixty_forty"] },
    "window":     { "type": "object", "properties": {
        "start": {"type": "string", "format": "date"},
        "end":   {"type": "string", "format": "date"} } }
  },
  "$defs": {
    "signal": {
      "type": "object", "additionalProperties": false,
      "required": ["type"],
      "properties": {
        "type": { "enum": [
          "return",            
          "sma_cross",         
          "drawdown_from_high",
          "volatility",        
          "rsi",               
          "daily_return",      
          "fundamental"        
        ]},
        "lookback_days": { "type": "integer", "minimum": 1, "maximum": 756 },
        "skip_days":     { "type": "integer", "minimum": 0, "maximum": 63 },
        "fast": { "type": "integer" }, "slow": { "type": "integer" },
        "field": { "enum": ["pe", "pb", "div_yield"] }
      }
    }
  }
}
```

Notes: `skip_days` exists specifically so "12-1 momentum" (12-month return skipping the last month) is first-class — the exact construct from today's honest backtest. `fundamental` is a **V1 placeholder**: the validator rejects it in v0 with a message ("fundamental data requires licensed sources — coming in V1"), so the vocabulary is stable but the capability is gated. `costs.sweep_bps` is not optional and not user-tunable below the default — every result ships with the 0/5/20bps sweep; that's a product rule, not a preference.

### Three worked examples

**1. "Buy the 5 best 6-month performers in the S&P 500, rebalanced monthly"**

```json
{
  "name": "6mo momentum top-5",
  "universe": { "preset": "sp500" },
  "selection": { "rank_by": { "type": "return", "lookback_days": 126, "skip_days": 0 },
                 "direction": "top", "n": 5 },
  "weighting": "equal",
  "rebalance": "monthly",
  "costs": { "sweep_bps": [0, 5, 20] },
  "benchmark": "SPY",
  "window": { "start": "2016-01-01", "end": "2026-01-01" }
}
```

**2. "Buy stocks that dropped 5%+ yesterday, hold 5 days"**

```json
{
  "name": "Dip buyer, 5-day hold",
  "universe": { "preset": "sp500" },
  "filters": [ { "signal": { "type": "daily_return", "lookback_days": 1 },
                 "op": "lte", "value": -0.05 } ],
  "selection": { "rank_by": { "type": "daily_return", "lookback_days": 1 },
                 "direction": "bottom", "n": 20 },
  "weighting": "equal",
  "rebalance": "daily",
  "entry_exit": { "hold_days": 5 },
  "costs": { "sweep_bps": [0, 5, 20] },
  "benchmark": "SPY"
}
```

**3. "60/40 stocks and bonds, rebalanced once a year"**

```json
{
  "name": "60/40 annual rebalance",
  "universe": { "preset": "user_watchlist", "watchlist_file": "sixty_forty.txt" },
  "selection": { "static_allocation": [
    { "ticker": "SPY", "weight": 0.60 },
    { "ticker": "AGG", "weight": 0.40 } ] },
  "weighting": "equal",
  "rebalance": "annual",
  "costs": { "sweep_bps": [0, 5, 20] },
  "benchmark": "sixty_forty"
}
```

(`static_allocation` bypasses ranking entirely — the validator enforces that a spec has *either* `rank_by` *or* `static_allocation`, never both, and that static weights sum to 1.0 ± 0.001.)

### LLM translation layer

**Flow:** text box → Claude API call with the schema → validation → **spec echo** → user confirms → engine runs. Two hard rules:

1. **The parsed spec is always shown back before running** — rendered as both the raw JSON and a plain-English readback ("Every month, rank S&P 500 stocks by 6-month return and hold the top 5, equally weighted"). The readback is generated in the same API call. The user clicks "Run" or edits — the LLM never triggers a backtest itself.
2. **Ambiguity → questions, never guesses.** If the input underdetermines a required field, the model returns `{"status": "needs_clarification", "questions": [...]}` instead of a spec. "Buy good momentum stocks" should yield "Over what lookback — 3, 6, or 12 months? How many holdings? Rebalance how often?" — at most 3 targeted questions per round.

**Validation layer (deterministic, not LLM):** the returned JSON is checked against the schema with `jsonschema` in Python, plus semantic rules the schema can't express (weights sum to 1, `fast < slow`, `window.start < window.end`, v0 capability gates like the `fundamental` rejection). Schema failures are fed back to the model once for self-repair; a second failure surfaces an error to the user. Use the API's structured outputs (`output_config.format` with this JSON Schema, or `client.messages.parse()` with a Pydantic model mirroring it) so malformed JSON is impossible at the transport level — validation then only catches semantic issues. Note the schema above uses `between`/array values and `additionalProperties: false` throughout, which is structured-outputs-compatible; keep numeric range checks (`minimum`/`maximum`) in the Python validator since the API's schema enforcement doesn't support them.

**Guardrails:** no code execution path exists (there is no field whose value is ever `eval`'d); asset classes outside the preset universes (options, crypto, futures, shorting, leverage) are out-of-vocabulary — the system prompt instructs the model to decline them explicitly with a "not supported in v0" message rather than approximating.

### Draft system prompt for the translator

```
You translate plain-English investment strategy descriptions into a JSON
strategy spec. You are a translator, not an advisor.

Rules:
- Output ONLY a spec matching the provided schema, or a needs_clarification
  object with at most 3 questions. Never both.
- Never guess a required parameter the user did not state or clearly imply.
  Lookback windows, holding counts, and rebalance frequency are the usual
  gaps — ask.
- Convention: "6-month" = 126 trading days, "12-1 momentum" = lookback 252,
  skip 21. State conversions in the readback.
- If the strategy needs anything outside the schema (options, shorting,
  leverage, intraday, fundamentals), return needs_clarification explaining
  it is unsupported in this version. Do not approximate it silently.
- Never comment on whether the strategy is good, likely to work, or
  suitable for the user. Never suggest trades.
- Alongside the spec, produce a one-paragraph plain-English readback of
  exactly what will be tested, including every defaulted value.
```

Prompt-engineering notes: put the full JSON Schema and 4–6 few-shot examples (including one `needs_clarification` example and one "unsupported asset class" refusal) in the system prompt, in that order, and keep them byte-stable; the volatile user text goes only in the user turn. Few-shots matter more than instructions for the trading-day conversions and for teaching the model *when* to ask vs. translate.

### Cost control

- **Model:** default to `claude-opus-5` ($5/$25 per MTok) — translation quality is the product's front door, and a single translation is ~4K cached input + ~500 output tokens, well under a cent per call at cache-read rates. If Sai wants to cut costs after measuring accuracy, `claude-haiku-4-5` ($1/$5) is the step-down to A/B against a fixed test set of ~30 descriptions — but that's his call, made on data, not a premature optimization.
- **Prompt caching:** mark the last block of the system prompt (schema + few-shots) with `cache_control: {"type": "ephemeral"}`. Everything stable lives before the breakpoint; the user's strategy text comes after. Repeat translations then bill the ~4K-token prefix at ~0.1× — effectively the whole cost becomes the 500 output tokens.
- **Token caps:** `max_tokens: 2000` per translation (a full spec + readback fits in ~800); clarification rounds are cheaper still. Cap clarification loops at 2 rounds before telling the user to rephrase.
- **No retry storms:** one self-repair attempt on validation failure, then surface the error. Log every (input, spec, validation result) triple locally — that corpus becomes the eval set for any future model-tier or prompt change.