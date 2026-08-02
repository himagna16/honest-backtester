"""Spec validation + watchlist upload + conventions. Pure functions over
HTTP — validate-spec never has side effects (Section 4 API contract)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, UploadFile

from config import settings
from dsl.conventions import CONVENTIONS_TABLE
from dsl.readback import readback
from dsl.universes import normalize_ticker
from dsl.validator import validate_spec
from dsl.symbols import check_tickers
from store.events import log_event

router = APIRouter(prefix="/api")


@router.post("/validate-spec")
def validate_spec_endpoint(body: dict) -> dict:
    spec_raw = body.get("spec")
    if spec_raw is None:
        raise HTTPException(400, detail="body must be {\"spec\": {...}}")
    result = validate_spec(spec_raw)
    out = {
        "valid": result.valid,
        "errors": result.errors,
        "warnings": result.warnings,
    }
    if result.valid and result.spec is not None:
        out["expanded_spec"] = result.expanded
        out["readback"] = readback(result.spec)
        out["universe_size"] = len(result.resolved_universe)
    return out


@router.post("/universe/watchlist")
async def upload_watchlist(file: UploadFile) -> dict:
    """Upload a plain-text ticker list (one per line). Tickers are checked
    against the symbol master immediately; problems come back with
    suggestions rather than failing silently later (Gap #4)."""
    content = (await file.read()).decode("utf-8", errors="replace")
    tickers = [normalize_ticker(line) for line in content.splitlines()
               if line.strip() and not line.strip().startswith("#")]
    if not tickers:
        raise HTTPException(400, detail="the file contains no tickers")
    if len(tickers) > 500:
        raise HTTPException(400, detail="watchlists are capped at 500 tickers")
    valid, problems = check_tickers(tickers)

    wl_dir = settings.db_path.parent / "watchlists"
    wl_dir.mkdir(parents=True, exist_ok=True)
    from pathlib import Path
    safe = Path(file.filename or "watchlist.txt").name
    (wl_dir / safe).write_text("\n".join(valid) + "\n")
    log_event("watchlist_uploaded", {"file": safe, "n_valid": len(valid),
                                     "n_problems": len(problems)})
    return {"file": safe, "valid": valid, "problems": problems}


@router.get("/conventions")
def conventions() -> dict:
    return {"conventions": CONVENTIONS_TABLE}
