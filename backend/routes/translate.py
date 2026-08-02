"""POST /api/translate-strategy — translation only, never runs anything."""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from llm.translate import TranslateError, translate

router = APIRouter(prefix="/api")

_STATUS_CODES = {
    "empty_input": 400,
    "input_too_long": 400,
    "llm_not_configured": 503,
    "model_refusal": 502,
    "parse_failed": 502,
}


@router.post("/translate-strategy", response_model=None)
def translate_strategy(body: dict):
    text = body.get("text", "")
    refresh = bool(body.get("refresh", False))
    try:
        return translate(text, use_cache=not refresh)
    except TranslateError as exc:
        return JSONResponse(
            status_code=_STATUS_CODES.get(exc.code, 500),
            content={"error": {"code": exc.code, "message": str(exc)}},
        )
