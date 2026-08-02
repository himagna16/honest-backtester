"""SQLite access via SQLAlchemy Core + a tiny plain-SQL migration runner.

Migrations are numbered .sql files in backend/migrations/, applied in order and
recorded in schema_migrations. Transparent and adequate for a single-user
SQLite file; SQLAlchemy keeps the V1 Postgres swap mechanical.
"""
from __future__ import annotations

import re
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from config import settings

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"

_engine: Engine | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        settings.ensure_dirs()
        _engine = create_engine(
            f"sqlite:///{settings.db_path}",
            connect_args={"check_same_thread": False},
        )
        run_migrations(_engine)
    return _engine


def run_migrations(engine: Engine) -> list[str]:
    """Apply pending .sql migrations in filename order. Returns those applied."""
    applied: list[str] = []
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            " version INTEGER PRIMARY KEY,"
            " name TEXT NOT NULL,"
            " applied_at TEXT NOT NULL DEFAULT (datetime('now')))"
        ))
        done = {row[0] for row in conn.execute(text("SELECT version FROM schema_migrations"))}
        for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            m = re.match(r"(\d+)_", path.name)
            if not m:
                continue
            version = int(m.group(1))
            if version in done:
                continue
            for statement in path.read_text().split(";\n\n"):
                if statement.strip():
                    conn.execute(text(statement))
            conn.execute(
                text("INSERT INTO schema_migrations (version, name) VALUES (:v, :n)"),
                {"v": version, "n": path.name},
            )
            applied.append(path.name)
    return applied


def reset_engine_for_tests() -> None:
    global _engine
    if _engine is not None:
        _engine.dispose()
    _engine = None
