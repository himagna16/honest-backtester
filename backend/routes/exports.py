"""Export endpoints (Gap #10): Truth Report as PNG/PDF with the disclaimer
baked into the artifact, and spec JSON export/import round-tripping."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from config import settings
from dsl.validator import validate_spec
from export import render_report
from store import repo
from store.events import log_event

router = APIRouter(prefix="/api")


@router.get("/report/{report_id}/export.{fmt}")
def export_report(report_id: str, fmt: str):
    if fmt not in ("png", "pdf"):
        raise HTTPException(400, detail="format must be png or pdf")
    envelope = repo.get_report(report_id)
    if not envelope:
        raise HTTPException(404, detail=f"report {report_id} not found")
    out = Path(settings.exports_dir) / f"{report_id}.{fmt}"
    render_report(envelope["report"], out, fmt)
    log_event("report_exported", {"report_id": report_id, "format": fmt})
    name = envelope["report"]["spec"].get("name", "truth-report")
    safe = "".join(c if c.isalnum() or c in "-_ " else "" for c in name)[:60].strip()
    return FileResponse(out, filename=f"truth-report {safe}.{fmt}")


@router.get("/strategies/{strategy_id}/spec.json")
def export_spec(strategy_id: str):
    detail = repo.strategy_detail(strategy_id)
    if not detail or not detail["versions"]:
        raise HTTPException(404, detail=f"strategy {strategy_id} not found")
    latest = detail["versions"][0]
    log_event("spec_exported", {"strategy_id": strategy_id})
    return JSONResponse(
        content=latest["spec"],
        headers={"Content-Disposition":
                 f'attachment; filename="{detail["name"][:60]}.spec.json"'},
    )


@router.post("/import-spec")
def import_spec(body: dict):
    """Validate an imported spec JSON. Pure — nothing runs, nothing saved;
    the client proceeds to the normal confirm-then-run flow."""
    spec_raw = body.get("spec", body)
    result = validate_spec(spec_raw)
    log_event("spec_imported", {"valid": result.valid})
    out = {"valid": result.valid, "errors": result.errors,
           "warnings": result.warnings}
    if result.valid:
        out["expanded_spec"] = result.expanded
    return out


@router.post("/event")
def client_event(body: dict):
    """Local telemetry from the frontend (Gap #11) — spec-card views,
    confirm abandonment. Local SQLite only; never third-party."""
    event_type = str(body.get("type", ""))[:60]
    if not event_type.startswith("ui_"):
        raise HTTPException(400, detail="client events must be namespaced ui_*")
    log_event(event_type, {k: v for k, v in body.items() if k != "type"})
    return {"ok": True}
