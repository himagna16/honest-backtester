"""The Honest Backtester — FastAPI backend.

Routes are added per milestone; every error returns {"error": {code, message}}.
"""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import settings
from store.db import get_engine

app = FastAPI(title="The Honest Backtester", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"error": {"code": "internal_error", "message": str(exc)}},
    )


@app.on_event("startup")
def startup() -> None:
    settings.ensure_dirs()
    get_engine()  # applies migrations


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "llm_configured": bool(settings.anthropic_api_key),
        "db": str(settings.db_path),
    }
