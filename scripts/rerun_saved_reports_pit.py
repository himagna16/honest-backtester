"""Re-run every saved report that relied on today's S&P 500 list.

Two kinds of saved run did:
  * S&P 500 preset runs — the universe itself was today's members.
  * Ranked hand-picked (user_watchlist) runs — their selection-bias check
    compared against today's S&P 500 list.

Each is re-run through the one audit pipeline (audit.pipeline.build_report) on
its own FROZEN price snapshot, and saved as a NEW run of the same strategy
version with `replay_of` pointing at the original. Nothing stored is edited:
the journal keeps the old report as history and shows the new one as latest.

For S&P 500 runs the snapshot lacks the closes of names that were members then
but are not today, so those are added from the local price cache on the
snapshot's own dates (never touching a snapshot column) and the augmented
panel is frozen as the new run's snapshot. Hand-picked runs replay their
snapshot unchanged; only the neutral comparison is new.

Writes docs/pit_rerun_log.md (the database is not in git) — every figure in it
is read back from the new reports.

Usage: HONEST_OFFLINE=1 .venv/bin/python scripts/rerun_saved_reports_pit.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backend"))

import pandas as pd  # noqa: E402
from sqlalchemy import text  # noqa: E402

from audit.extra import bump_attempts, record_trial_sharpe, trial_sharpes  # noqa: E402
from audit.pipeline import build_report  # noqa: E402
from config import settings  # noqa: E402
from dsl.validator import validate_spec  # noqa: E402
from engine.data import PriceCache, PricePanel, load_snapshot, save_snapshot  # noqa: E402
from store import repo  # noqa: E402
from store.db import get_engine  # noqa: E402
from store.events import log_event  # noqa: E402

REASON = "point-in-time S&P 500 membership"


def affected_runs() -> list[dict]:
    with get_engine().begin() as conn:
        rows = conn.execute(text(
            "SELECT r.id, r.strategy_version_id, r.spec_json, rep.id, rep.report_json, "
            "v.strategy_id FROM runs r JOIN reports rep ON rep.run_id = r.id "
            "JOIN strategy_versions v ON v.id = r.strategy_version_id "
            "WHERE r.status = 'done' AND json_extract(rep.report_json, '$.run.replay_of') IS NULL "
            "ORDER BY r.created_at")).fetchall()
    out = []
    for run_id, version_id, spec_json, rep_id, report_json, strategy_id in rows:
        spec = json.loads(spec_json)
        report = json.loads(report_json)
        if report.get("data", {}).get("universe_membership"):
            continue                         # already made with membership applied
        preset = spec["universe"]["preset"]
        ranked = spec["selection"].get("static_allocation") is None
        if preset == "sp500" or (preset == "user_watchlist" and ranked):
            out.append({"run_id": run_id, "report_id": rep_id, "strategy_id": strategy_id,
                        "spec": spec, "old": report})
    return out


def augmented(panel: PricePanel, symbols: list[str]) -> tuple[PricePanel, list[str]]:
    cache = PriceCache()
    prices = panel.prices
    extra, added = {}, []
    for sym in symbols:
        if sym in prices.columns:
            continue
        s = cache.get_series(sym)
        if s is None:
            continue
        s = s.reindex(prices.index)
        if s.notna().any():
            extra[sym] = s
            added.append(sym)
    if not extra:
        return panel, added
    new_prices = pd.concat([prices, pd.DataFrame(extra, index=prices.index)], axis=1)
    first_dates = {**panel.first_dates,
                   **{k: str(v.dropna().index[0].date()) for k, v in extra.items()}}
    notes = panel.notes + [
        f"{len(added)} former S&P 500 member(s) added from the local price cache for "
        f"{REASON}; the original snapshot's columns are unchanged"
    ]
    return PricePanel(prices=new_prices, as_of=panel.as_of,
                      sources={**panel.sources, **{k: "cache" for k in added}},
                      missing=panel.missing, first_dates=first_dates, stale=panel.stale,
                      notes=notes), added


def rerun(item: dict) -> dict:
    result = validate_spec(item["spec"])
    if not result.valid:
        raise RuntimeError(f"{item['run_id']}: spec no longer validates: {result.errors}")
    spec = result.spec
    panel = load_snapshot(Path(settings.runs_dir) / item["run_id"])
    added: list[str] = []
    if spec.universe.preset == "sp500":
        panel, added = augmented(panel, result.resolved_universe)
    universe = [t for t in result.resolved_universe if t in panel.prices.columns]

    version_id, _, _ = repo.get_or_create_version(item["strategy_id"], result.expanded)
    run_id = repo.create_run(version_id, result.expanded)
    run_dir = Path(settings.runs_dir) / run_id
    snap_path, snap_hash = save_snapshot(panel, run_dir)
    source = f"snapshot:{item['run_id']}" + ("+pit-members" if added else "")
    repo.update_run(run_id, status="running", snapshot_path=snap_path,
                    snapshot_hash=snap_hash, data_as_of=panel.as_of, data_source=source)

    attempts = bump_attempts(spec.universe_key())
    prior = trial_sharpes(spec.universe_key())
    report = build_report(spec, panel, universe, attempts, trial_sharpes_annual=prior)
    report["run"] = {"id": run_id, "snapshot_hash": snap_hash, "data_source": source,
                     "replay_of": item["run_id"], "rerun_reason": REASON}
    record_trial_sharpe(run_id, spec.universe_key(), report["metrics"].get("sharpe"))
    report_id = repo.save_report(run_id, report["verdict"], report)
    repo.update_run(run_id, status="done",
                    finished_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
    log_event("run_completed", {"run_id": run_id, "verdict": report["verdict"]["color"],
                                "universe_key": spec.universe_key(), "attempts": attempts,
                                "replay_of": item["run_id"], "reason": REASON})
    return {"old_run": item["run_id"], "old_report": item["report_id"], "new_run": run_id,
            "new_report": report_id, "name": spec.name, "preset": spec.universe.preset,
            "added_members": len(added), "old": item["old"], "new": report}


def pct(x) -> str:
    return "—" if x is None else f"{x:+,.1%}"


def bias_line(report: dict) -> str:
    f = next((f for f in report["audit"] if f["check"] in ("selection_bias", "survivorship_universe")), None)
    return f"{f['severity']}: {f['name']}" if f else "—"


def render(rows: list[dict]) -> str:
    lines = [
        "# Saved reports re-run with point-in-time S&P 500 membership",
        "",
        f"Generated {datetime.now(timezone.utc).date()} by `scripts/rerun_saved_reports_pit.py`. "
        "Every figure is read back from the stored reports; the originals are kept in the "
        "journal as history. Both the old and new numbers are net of 5 bps. New reports use "
        "the current engine, which also changed after most originals were made (see "
        "`docs/pit_membership_study.md`).",
        "",
        "| Strategy | Old report | Old: return / verdict | New report | New: return / verdict | SPY (new) | Universe check (new) |",
        "|---|---|---|---|---|---:|---|",
    ]
    for r in rows:
        o, n = r["old"], r["new"]
        lines.append(
            f"| {r['name']} | `{r['old_report']}` | {pct(o['metrics']['total_return'])} / "
            f"{o['verdict']['color']} | `{r['new_report']}` | {pct(n['metrics']['total_return'])} / "
            f"{n['verdict']['color']} | {pct(n['benchmark']['total_return'])} | {bias_line(n)} |")
    lines += ["", "## Hand-picked runs: the neutral comparison, before and after", ""]
    for r in rows:
        o, n = r["old"].get("neutral_rerun"), r["new"].get("neutral_rerun")
        if not n:
            continue
        lines.append(
            f"- **{r['name']}** (`{r['new_report']}`): your list {pct(n['user_net5_total_return'])} "
            f"vs S&P 500 as it stood {pct(n['net5_total_return'])} → {n['wealth_ratio']:.2f}× "
            f"(point-in-time: {n['point_in_time']}). Before: today's list "
            f"{pct(o['net5_total_return']) if o else '—'}, {o['wealth_ratio']:.2f}× "
            f"on the old report." if o else "")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    items = affected_runs()
    rows = []
    for item in items:
        print(f"re-running {item['run_id']} ({item['spec']['name']})…", flush=True)
        row = rerun(item)
        rows.append(row)
        print(f"  → {row['new_report']}: {pct(row['new']['metrics']['total_return'])} "
              f"{row['new']['verdict']['color']} (was {pct(row['old']['metrics']['total_return'])} "
              f"{row['old']['verdict']['color']}); +{row['added_members']} former members", flush=True)
    out = REPO / "docs" / "pit_rerun_log.md"
    out.write_text(render(rows))
    print(f"wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
