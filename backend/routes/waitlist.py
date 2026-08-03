"""Waitlist capture for the validation sprint (Section 5, step 2).

Deliberately minimal and LOCAL: emails land in the same SQLite file as
everything else and never leave the machine. Strategy text is confidential
user IP — no third-party analytics, here or anywhere else.

This is a validation instrument, not a growth funnel: the brief's success
threshold is 50+ signups or 10+ substantive replies in 3 weeks, and the
kill criterion is real.
"""
from __future__ import annotations

import re

from fastapi import APIRouter
from fastapi.responses import JSONResponse, PlainTextResponse
from sqlalchemy import text as sql

from store.db import get_engine
from store.events import log_event

router = APIRouter(prefix="/api")

# Deliberately permissive — bouncing a valid-but-unusual address costs a real
# signal; a junk row costs nothing.
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s.]+\.[^@\s]+$")


@router.post("/waitlist", status_code=201)
def join_waitlist(body: dict):
    email = str(body.get("email", "")).strip().lower()
    note = str(body.get("note", "")).strip()[:500] or None
    source = str(body.get("source", "landing")).strip()[:40] or "landing"

    if not EMAIL_RE.match(email) or len(email) > 254:
        return JSONResponse(status_code=400, content={"error": {
            "code": "invalid_email", "message": "that doesn't look like an email address"}})

    with get_engine().begin() as conn:
        existing = conn.execute(
            sql("SELECT id FROM waitlist WHERE email = :e"), {"e": email}
        ).fetchone()
        if existing:
            # Idempotent: re-submitting is not an error, and we don't leak
            # whether an address is already on the list beyond this response.
            return {"ok": True, "already": True}
        conn.execute(
            sql("INSERT INTO waitlist (email, source, note) VALUES (:e, :s, :n)"),
            {"e": email, "s": source, "n": note},
        )
    log_event("waitlist_signup", {"source": source, "has_note": bool(note)})
    return {"ok": True, "already": False}


@router.get("/waitlist/stats")
def waitlist_stats():
    """Counts only — never the addresses. Used by the local dashboard."""
    with get_engine().begin() as conn:
        total = conn.execute(sql("SELECT COUNT(*) FROM waitlist")).scalar() or 0
        with_note = conn.execute(
            sql("SELECT COUNT(*) FROM waitlist WHERE note IS NOT NULL")
        ).scalar() or 0
    return {
        "total": total,
        "with_note": with_note,
        # The brief's go/no-go bar for the validation sprint.
        "target": 50,
        "met": total >= 50,
    }


@router.get("/waitlist/export.csv")
def export_waitlist():
    """Local export so the list is yours, not locked in a vendor."""
    with get_engine().begin() as conn:
        rows = conn.execute(sql(
            "SELECT email, source, note, created_at FROM waitlist "
            "ORDER BY created_at"
        )).fetchall()
    lines = ["email,source,note,created_at"]
    for email, source, note, created in rows:
        clean = (note or "").replace('"', '""')
        lines.append(f'{email},{source},"{clean}",{created}')
    return PlainTextResponse(
        "\n".join(lines) + "\n",
        headers={"Content-Disposition": 'attachment; filename="waitlist.csv"'},
    )
