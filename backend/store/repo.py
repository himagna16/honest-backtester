"""SQLite persistence for strategies, versions, runs, and reports.

Every Run stores the resolved spec verbatim (defaults expanded — Gap #8) and
its frozen panel snapshot path + content hash (Gap #2), so old Truth Reports
stay interpretable and reproducible after anything changes.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import text as sql

from store.db import get_engine


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def create_strategy(name: str, family_key: str) -> str:
    sid = new_id("st")
    with get_engine().begin() as conn:
        conn.execute(sql(
            "INSERT INTO strategies (id, name, family_key) VALUES (:i, :n, :f)"
        ), {"i": sid, "n": name[:80], "f": family_key})
    return sid


def get_or_create_version(strategy_id: str, spec_json: dict) -> tuple[str, int, bool]:
    """→ (version_id, version_number, created_new). A spec identical to the
    latest version reuses it; any edit becomes a new version."""
    payload = json.dumps(spec_json, sort_keys=True)
    with get_engine().begin() as conn:
        row = conn.execute(sql(
            "SELECT id, version, spec_json FROM strategy_versions "
            "WHERE strategy_id = :s ORDER BY version DESC LIMIT 1"
        ), {"s": strategy_id}).fetchone()
        if row and json.dumps(json.loads(row[2]), sort_keys=True) == payload:
            return row[0], row[1], False
        version = (row[1] + 1) if row else 1
        vid = new_id("sv")
        conn.execute(sql(
            "INSERT INTO strategy_versions "
            "(id, strategy_id, version, spec_version, spec_json) "
            "VALUES (:i, :s, :v, :sv, :j)"
        ), {"i": vid, "s": strategy_id, "v": version,
            "sv": spec_json.get("spec_version", 1), "j": payload})
        conn.execute(sql(
            "UPDATE strategies SET updated_at = :t WHERE id = :s"
        ), {"t": _now(), "s": strategy_id})
    return vid, version, True


def create_run(strategy_version_id: str, spec_json: dict) -> str:
    rid = new_id("r")
    with get_engine().begin() as conn:
        conn.execute(sql(
            "INSERT INTO runs (id, strategy_version_id, status, spec_json) "
            "VALUES (:i, :v, 'queued', :j)"
        ), {"i": rid, "v": strategy_version_id,
            "j": json.dumps(spec_json, sort_keys=True)})
    return rid


def update_run(run_id: str, **fields) -> None:
    if not fields:
        return
    sets = ", ".join(f"{k} = :{k}" for k in fields)
    with get_engine().begin() as conn:
        conn.execute(sql(f"UPDATE runs SET {sets} WHERE id = :run_id"),
                     {**fields, "run_id": run_id})


def get_run(run_id: str) -> dict | None:
    with get_engine().begin() as conn:
        row = conn.execute(sql(
            "SELECT id, strategy_version_id, status, spec_json, snapshot_path, "
            "snapshot_hash, data_as_of, data_source, error, created_at, finished_at "
            "FROM runs WHERE id = :i"), {"i": run_id}).fetchone()
    if not row:
        return None
    keys = ["id", "strategy_version_id", "status", "spec_json", "snapshot_path",
            "snapshot_hash", "data_as_of", "data_source", "error",
            "created_at", "finished_at"]
    out = dict(zip(keys, row))
    out["spec_json"] = json.loads(out["spec_json"])
    return out


def save_report(run_id: str, verdict: dict, report: dict) -> str:
    rep_id = new_id("rep")
    with get_engine().begin() as conn:
        conn.execute(sql(
            "INSERT INTO reports (id, run_id, verdict_color, verdict_headline, "
            "report_json) VALUES (:i, :r, :c, :h, :j)"
        ), {"i": rep_id, "r": run_id, "c": verdict["color"],
            "h": verdict["headline"], "j": json.dumps(report)})
    return rep_id


def get_report(report_id: str) -> dict | None:
    with get_engine().begin() as conn:
        row = conn.execute(sql(
            "SELECT r.id, r.run_id, r.verdict_color, r.verdict_headline, "
            "r.report_json, r.created_at, runs.strategy_version_id "
            "FROM reports r JOIN runs ON runs.id = r.run_id WHERE r.id = :i"
        ), {"i": report_id}).fetchone()
    if not row:
        return None
    return {
        "id": row[0], "run_id": row[1], "verdict_color": row[2],
        "verdict_headline": row[3], "report": json.loads(row[4]),
        "created_at": row[5], "strategy_version_id": row[6],
    }


def report_for_run(run_id: str) -> str | None:
    with get_engine().begin() as conn:
        row = conn.execute(sql(
            "SELECT id FROM reports WHERE run_id = :r ORDER BY created_at DESC LIMIT 1"
        ), {"r": run_id}).fetchone()
    return row[0] if row else None


def list_strategies() -> list[dict]:
    with get_engine().begin() as conn:
        rows = conn.execute(sql(
            """
            SELECT s.id, s.name, s.created_at, s.updated_at,
                   (SELECT COUNT(*) FROM strategy_versions v WHERE v.strategy_id = s.id),
                   (SELECT COUNT(*) FROM runs r JOIN strategy_versions v
                     ON r.strategy_version_id = v.id WHERE v.strategy_id = s.id),
                   (SELECT rep.verdict_color FROM reports rep
                     JOIN runs r ON r.id = rep.run_id
                     JOIN strategy_versions v ON v.id = r.strategy_version_id
                     WHERE v.strategy_id = s.id ORDER BY rep.created_at DESC LIMIT 1),
                   (SELECT rep.verdict_headline FROM reports rep
                     JOIN runs r ON r.id = rep.run_id
                     JOIN strategy_versions v ON v.id = r.strategy_version_id
                     WHERE v.strategy_id = s.id ORDER BY rep.created_at DESC LIMIT 1),
                   (SELECT rep.id FROM reports rep
                     JOIN runs r ON r.id = rep.run_id
                     JOIN strategy_versions v ON v.id = r.strategy_version_id
                     WHERE v.strategy_id = s.id ORDER BY rep.created_at DESC LIMIT 1)
            FROM strategies s ORDER BY s.updated_at DESC
            """
        )).fetchall()
    return [
        {"id": r[0], "name": r[1], "created_at": r[2], "updated_at": r[3],
         "versions": r[4], "runs": r[5], "last_verdict_color": r[6],
         "last_verdict_headline": r[7], "last_report_id": r[8]}
        for r in rows
    ]


def strategy_detail(strategy_id: str) -> dict | None:
    with get_engine().begin() as conn:
        srow = conn.execute(sql(
            "SELECT id, name, family_key, created_at FROM strategies WHERE id = :i"
        ), {"i": strategy_id}).fetchone()
        if not srow:
            return None
        versions = conn.execute(sql(
            "SELECT id, version, spec_json, created_at FROM strategy_versions "
            "WHERE strategy_id = :i ORDER BY version DESC"), {"i": strategy_id}
        ).fetchall()
        runs = conn.execute(sql(
            """
            SELECT r.id, v.version, r.status, r.created_at, r.finished_at,
                   r.data_as_of, r.data_source, r.error,
                   rep.id, rep.verdict_color, rep.verdict_headline
            FROM runs r
            JOIN strategy_versions v ON v.id = r.strategy_version_id
            LEFT JOIN reports rep ON rep.run_id = r.id
            WHERE v.strategy_id = :i ORDER BY r.created_at DESC
            """), {"i": strategy_id}).fetchall()
    return {
        "id": srow[0], "name": srow[1], "family_key": srow[2], "created_at": srow[3],
        "versions": [
            {"id": v[0], "version": v[1], "spec": json.loads(v[2]), "created_at": v[3]}
            for v in versions
        ],
        "runs": [
            {"id": r[0], "version": r[1], "status": r[2], "created_at": r[3],
             "finished_at": r[4], "data_as_of": r[5], "data_source": r[6],
             "error": r[7], "report_id": r[8], "verdict_color": r[9],
             "verdict_headline": r[10]}
            for r in runs
        ],
    }


def find_strategy_by_name(name: str) -> str | None:
    with get_engine().begin() as conn:
        row = conn.execute(sql(
            "SELECT id FROM strategies WHERE name = :n LIMIT 1"), {"n": name[:80]}
        ).fetchone()
    return row[0] if row else None
