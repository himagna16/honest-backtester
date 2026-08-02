"""Strategy library / journal (Section 1). List views show verdict badges,
never returns — the library rewards honesty, not big numbers."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from store import repo

router = APIRouter(prefix="/api")


@router.get("/strategies")
def strategies():
    return repo.list_strategies()


@router.get("/strategies/{strategy_id}")
def strategy(strategy_id: str):
    detail = repo.strategy_detail(strategy_id)
    if not detail:
        raise HTTPException(404, detail=f"strategy {strategy_id} not found")
    return detail
