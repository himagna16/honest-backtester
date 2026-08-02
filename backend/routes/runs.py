"""Run orchestration: validate → snapshot → engine + full audit → report.

POST /api/run-backtest        → 202 {run_id}; work happens in a BackgroundTask
GET  /api/run-backtest/{id}/events → SSE progress (the audit-themed checklist)
GET  /api/report/{report_id}  → the Truth Report JSON

The ONLY code path that executes a backtest is audit.pipeline.build_report —
there is no unaudited variant and no toggle (non-negotiable #1). Re-runs of a
prior run replay its frozen snapshot by default; refresh_data=true is the
only way to fetch new data (Gap #2).
"""
from __future__ import annotations

import asyncio
import json
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from audit.extra import bump_attempts
from audit.pipeline import build_report
from config import settings
from dsl.validator import validate_spec
from engine.benchmark import BENCHMARK_TICKERS
from engine.data import get_panel, load_snapshot, save_snapshot
from store import repo
from store.events import log_event

router = APIRouter(prefix="/api")

# In-process progress buffers: run_id -> list of {stage, pct, ts}. SSE readers
# poll these; the terminal event carries report_id or error. Restart-safe:
# finished runs answer from the DB instead.
_progress: dict[str, list[dict]] = {}
_progress_lock = threading.Lock()

_STAGES_PCT = {
    "queued": 2, "fetching": 10, "snapshot": 20, "simulating": 35,
    "auditing": 55, "finalizing": 90, "done": 100, "error": 100,
}


def _push(run_id: str, stage: str, detail: str, **extra) -> None:
    event = {
        "stage": stage,
        "detail": detail,
        "pct": _STAGES_PCT.get(stage, 50),
        "ts": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    with _progress_lock:
        _progress.setdefault(run_id, []).append(event)


def _execute_run(run_id: str, spec_raw: dict, universe: list[str],
                 replay_of: str | None, refresh: bool) -> None:
    try:
        repo.update_run(run_id, status="running")
        result = validate_spec(spec_raw)   # re-validate: the security boundary
        if not result.valid or result.spec is None:
            raise RuntimeError("spec failed validation: " + "; ".join(result.errors))
        spec = result.spec

        # ---- data: replay snapshot by default, fetch otherwise ----
        run_dir = Path(settings.runs_dir) / run_id
        if replay_of and not refresh:
            _push(run_id, "fetching", "Replaying the frozen data snapshot…")
            prior = repo.get_run(replay_of)
            if not prior or not prior["snapshot_path"]:
                raise RuntimeError(f"run {replay_of} has no snapshot to replay")
            panel = load_snapshot(Path(prior["snapshot_path"]).parent)
            data_source = f"snapshot:{replay_of}"
        else:
            _push(run_id, "fetching", "Fetching prices…")
            needed = universe + BENCHMARK_TICKERS[spec.benchmark] + ["^IRX"]
            panel = get_panel(needed, start=spec.window.start,
                              end=spec.window.end, refresh=refresh)
            data_source = "network" if refresh else "cache"

        _push(run_id, "snapshot", "Freezing the exact price panel (content-hashed)…")
        snap_path, snap_hash = save_snapshot(panel, run_dir)
        repo.update_run(run_id, snapshot_path=snap_path, snapshot_hash=snap_hash,
                        data_as_of=panel.as_of, data_source=data_source)

        universe_in_panel = [t for t in universe if t in panel.prices.columns]
        if not universe_in_panel:
            raise RuntimeError(
                "none of the universe tickers have price data — run `make seed` "
                "or check the network"
            )

        _push(run_id, "simulating", "Simulating with costs (0 / 5 / 20 bps)…")
        attempts = bump_attempts(spec.universe_key())

        def progress(stage_detail: str) -> None:
            _push(run_id, "auditing", stage_detail)

        report = build_report(spec, panel, universe_in_panel, attempts,
                              progress=progress)
        report["run"] = {
            "id": run_id, "snapshot_hash": snap_hash, "data_source": data_source,
            "replay_of": replay_of,
        }

        _push(run_id, "finalizing", "Writing the Truth Report…")
        report_id = repo.save_report(run_id, report["verdict"], report)
        repo.update_run(run_id, status="done",
                        finished_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
        log_event("run_completed", {
            "run_id": run_id, "verdict": report["verdict"]["color"],
            "universe_key": spec.universe_key(), "attempts": attempts,
        })
        _push(run_id, "done", "Truth Report ready", report_id=report_id)
    except Exception as exc:  # noqa: BLE001 — surfaced honestly to the user
        repo.update_run(run_id, status="error", error=str(exc))
        log_event("run_failed", {"run_id": run_id, "error": str(exc)})
        traceback.print_exc()
        _push(run_id, "error", str(exc))


@router.post("/run-backtest", status_code=202)
def run_backtest(body: dict, background: BackgroundTasks):
    spec_raw = body.get("spec")
    replay_of = body.get("rerun_of")
    refresh = bool(body.get("refresh_data", False))

    if replay_of and not spec_raw:
        prior = repo.get_run(replay_of)
        if not prior:
            raise HTTPException(404, detail=f"run {replay_of} not found")
        spec_raw = prior["spec_json"]
    if not isinstance(spec_raw, dict):
        return JSONResponse(status_code=400, content={"error": {
            "code": "missing_spec", "message": "body must include a spec"}})

    result = validate_spec(spec_raw)
    if not result.valid or result.expanded is None:
        return JSONResponse(status_code=422, content={"error": {
            "code": "invalid_spec",
            "message": "the spec failed validation — nothing was run",
            "details": result.errors}})

    strategy_id = body.get("strategy_id")
    name = result.expanded["name"]
    if not strategy_id:
        strategy_id = repo.find_strategy_by_name(name) or repo.create_strategy(
            name, result.spec.universe_key())
    version_id, version, _ = repo.get_or_create_version(strategy_id, result.expanded)
    run_id = repo.create_run(version_id, result.expanded)

    _push(run_id, "queued", "Queued — the audit always runs, no toggle exists.")
    log_event("run_started", {"run_id": run_id, "strategy_id": strategy_id})
    background.add_task(_execute_run, run_id, result.expanded,
                        result.resolved_universe, replay_of, refresh)
    return {"run_id": run_id, "strategy_id": strategy_id, "version": version,
            "warnings": result.warnings}


@router.get("/run-backtest/{run_id}/events")
async def run_events(run_id: str):
    run = repo.get_run(run_id)
    if run is None and run_id not in _progress:
        raise HTTPException(404, detail=f"run {run_id} not found")

    async def stream():
        sent = 0
        # Finished before this process started (no buffer): answer from DB.
        if run_id not in _progress and run is not None:
            if run["status"] == "done":
                rep = repo.report_for_run(run_id)
                yield "data: " + json.dumps({
                    "stage": "done", "pct": 100, "detail": "Truth Report ready",
                    "report_id": rep}) + "\n\n"
            else:
                yield "data: " + json.dumps({
                    "stage": "error", "pct": 100,
                    "detail": run["error"] or "run did not finish"}) + "\n\n"
            return
        while True:
            with _progress_lock:
                events = list(_progress.get(run_id, []))
            for event in events[sent:]:
                yield "data: " + json.dumps(event) + "\n\n"
            sent = len(events)
            if events and events[-1]["stage"] in ("done", "error"):
                return
            await asyncio.sleep(0.25)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


@router.get("/runs/{run_id}")
def run_status(run_id: str):
    run = repo.get_run(run_id)
    if not run:
        raise HTTPException(404, detail=f"run {run_id} not found")
    run["report_id"] = repo.report_for_run(run_id)
    return run


@router.get("/report/{report_id}")
def get_report(report_id: str):
    rep = repo.get_report(report_id)
    if not rep:
        raise HTTPException(404, detail=f"report {report_id} not found")
    return rep
