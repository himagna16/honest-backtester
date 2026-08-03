"""The Honest Backtester — FastAPI backend.

Routes are added per milestone; every error returns {"error": {code, message}}.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import settings
from routes.exports import router as exports_router
from routes.journal import router as journal_router
from routes.runs import router as runs_router
from routes.specs import router as specs_router
from routes.translate import router as translate_router
from routes.waitlist import router as waitlist_router
from store.db import get_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_dirs()
    get_engine()  # applies migrations
    yield


app = FastAPI(title="The Honest Backtester", version="0.1.0", lifespan=lifespan)
app.include_router(specs_router)
app.include_router(translate_router)
app.include_router(runs_router)
app.include_router(journal_router)
app.include_router(exports_router)
app.include_router(waitlist_router)

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


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "llm_configured": bool(settings.anthropic_api_key),
        "db": str(settings.db_path),
    }
