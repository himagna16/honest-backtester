"""Local event log (Gap #11): translation outcomes, spec confirms/abandons,
run verdicts, exports. Doubles as the future LLM eval corpus. Local only —
strategy text is confidential and never goes to third-party analytics."""
from __future__ import annotations

import json

from sqlalchemy import text

from store.db import get_engine


def log_event(event_type: str, payload: dict | None = None) -> None:
    with get_engine().begin() as conn:
        conn.execute(
            text("INSERT INTO events (type, payload) VALUES (:t, :p)"),
            {"t": event_type, "p": json.dumps(payload or {})},
        )


def event_stats() -> dict:
    with get_engine().begin() as conn:
        rows = conn.execute(
            text("SELECT type, COUNT(*) FROM events GROUP BY type")
        ).fetchall()
    return {r[0]: r[1] for r in rows}
