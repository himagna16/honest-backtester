"""POST /api/translate-strategy — translation only, never runs anything."""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from llm.translate import TranslateError, translate
from ratelimit import RateLimit, client_ip, is_local, limiter

router = APIRouter(prefix="/api")

_STATUS_CODES = {
    "empty_input": 400,
    "input_too_long": 400,
    "llm_not_configured": 503,
    "quota_exhausted": 503,
    "model_refusal": 502,
    "parse_failed": 502,
    "token_limit": 502,
}


@router.post("/translate-strategy", response_model=None)
def translate_strategy(body: dict, request: Request):
    # Rate limit the only endpoint that spends money before doing any work.
    # Same-machine (local dev) callers are exempt; see ratelimit.is_local.
    ip = client_ip(request)
    if not is_local(ip):
        try:
            limiter.check(ip)
        except RateLimit as exc:
            return JSONResponse(
                status_code=429,
                content={"error": {"code": "rate_limited", "message": exc.message}},
                headers={"Retry-After": str(exc.retry_after)},
            )

    text = body.get("text", "")
    refresh = bool(body.get("refresh", False))
    try:
        return translate(text, use_cache=not refresh)
    except TranslateError as exc:
        return JSONResponse(
            status_code=_STATUS_CODES.get(exc.code, 500),
            content={"error": {"code": exc.code, "message": str(exc)}},
        )
