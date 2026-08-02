"""The strategy spec DSL — Pydantic models (Section 2, with Gap #1 and #8 applied).

This is a closed vocabulary, not a programming language. Nothing in a spec is
ever executed as code; the engine interprets these fields and nothing else.

Gap #1: stop_loss_pct / take_profit_pct are CUT from V0 — stops on daily bars
need fill rules (gap-through handling) that close-only data cannot honestly
provide. The validator rejects them with that explanation.

Gap #8: every stored spec carries spec_version and is saved with all defaults
expanded, so old runs stay interpretable after schema changes.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SPEC_VERSION = 1

SignalType = Literal[
    "return", "sma_cross", "drawdown_from_high", "volatility", "rsi",
    "daily_return", "fundamental",
]
UniversePreset = Literal["sp500", "nasdaq100", "sector_etfs", "dow30", "user_watchlist"]
Weighting = Literal["equal", "rank", "inverse_vol"]
Rebalance = Literal["daily", "weekly", "monthly", "quarterly", "annual"]
Benchmark = Literal["SPY", "QQQ", "sixty_forty"]
FilterOp = Literal["gt", "lt", "gte", "lte", "between"]

# Signals that don't take lookback_days
_NO_LOOKBACK = {"sma_cross"}


class StrictModel(BaseModel):
    # additionalProperties: false — unknown keys are validation errors, so a
    # spec smuggling extra fields (e.g. stop_loss_pct) is rejected, never
    # silently ignored. The semantic validator turns the raw pydantic error
    # into a plain-English message.
    model_config = ConfigDict(extra="forbid")


class Signal(StrictModel):
    type: SignalType
    lookback_days: int | None = Field(default=None, ge=1, le=756)
    skip_days: int = Field(default=0, ge=0, le=63)
    fast: int | None = Field(default=None, ge=1, le=756)
    slow: int | None = Field(default=None, ge=2, le=756)
    field: Literal["pe", "pb", "div_yield"] | None = None


class Filter(StrictModel):
    signal: Signal
    op: FilterOp
    value: float | list[float]


class Universe(StrictModel):
    preset: UniversePreset
    watchlist_file: str | None = None
    # Filled at validation time for user_watchlist (and stored in the expanded
    # spec) so a run stays reproducible even if the watchlist file changes.
    resolved_tickers: list[str] | None = None


class StaticPosition(StrictModel):
    ticker: str
    weight: float = Field(gt=0, le=1)


class Selection(StrictModel):
    rank_by: Signal | None = None
    direction: Literal["top", "bottom"] | None = None
    n: int | None = Field(default=None, ge=1, le=50)
    static_allocation: list[StaticPosition] | None = None


class EntryExit(StrictModel):
    hold_days: int | None = Field(default=None, ge=1, le=252)


class Costs(StrictModel):
    # Product rule: the 0/5/20 bps sweep is not optional and not tunable below
    # the default. The validator enforces {0, 5, 20} ⊆ sweep_bps.
    sweep_bps: list[float] = Field(default=[0, 5, 20])


class Window(StrictModel):
    start: str | None = None  # ISO date; defaults expanded at validation
    end: str | None = None


class StrategySpec(StrictModel):
    spec_version: int = SPEC_VERSION
    name: str = Field(max_length=80)
    universe: Universe
    filters: list[Filter] = Field(default_factory=list, max_length=5)
    selection: Selection
    weighting: Weighting
    rebalance: Rebalance
    entry_exit: EntryExit = Field(default_factory=EntryExit)
    costs: Costs = Field(default_factory=Costs)
    benchmark: Benchmark = "SPY"
    window: Window = Field(default_factory=Window)

    @property
    def is_static(self) -> bool:
        return self.selection.static_allocation is not None

    def universe_key(self) -> str:
        """Stable key for the multiple-testing attempt counter."""
        if self.universe.preset == "user_watchlist":
            tickers = sorted(self.universe.resolved_tickers or [])
            return "watchlist:" + ",".join(tickers)
        return self.universe.preset
