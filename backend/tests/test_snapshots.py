"""Snapshot storage: panels are content-addressed and shared across runs,
without weakening byte-for-byte reproducibility, and older run layouts still
replay. See SAAS_PLAN.md — this is the fix that lets snapshots survive
multi-user traffic."""


def test_panels_are_content_addressed_and_shared(tmp_path, monkeypatch):
    """Two runs that freeze an identical panel must share ONE stored blob.

    This is the multi-user storage fix from SAAS_PLAN: a 500-ticker x 10-year
    panel is megabytes, and everyone backtests the same S&P 500 window. Storing
    one copy per run does not survive real traffic.
    """
    import pandas as pd

    from config import settings
    from engine.data import PricePanel, load_snapshot, save_snapshot

    panels = tmp_path / "panels"
    monkeypatch.setattr(settings, "panels_dir", panels)

    idx = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=5, freq="D"))
    prices = pd.DataFrame({"AAA": [1.0, 1.1, 1.2, 1.3, 1.4],
                           "BBB": [9.0, 9.1, 9.2, 9.3, 9.4]}, index=idx)
    panel = PricePanel(prices=prices, as_of="2024-01-05", sources={},
                       missing=[], first_dates={}, stale=False, notes=[])

    p1, h1 = save_snapshot(panel, tmp_path / "runs" / "r_one")
    p2, h2 = save_snapshot(panel, tmp_path / "runs" / "r_two")

    assert h1 == h2, "identical panels must hash identically"
    assert p1 == p2, "identical panels must resolve to the same stored blob"
    assert len(list(panels.glob("*.parquet"))) == 1, "the blob was duplicated"

    # Both runs still replay their own panel exactly.
    for run in ("r_one", "r_two"):
        back = load_snapshot(tmp_path / "runs" / run)
        pd.testing.assert_frame_equal(back.prices, prices, check_freq=False)

    # A genuinely different panel gets its own blob.
    other = prices * 2
    save_snapshot(PricePanel(prices=other, as_of="2024-01-05", sources={},
                             missing=[], first_dates={}, stale=False, notes=[]),
                  tmp_path / "runs" / "r_three")
    assert len(list(panels.glob("*.parquet"))) == 2


def test_legacy_in_run_snapshots_still_replay(tmp_path, monkeypatch):
    """Journals written before content-addressing must keep working."""
    import json

    import pandas as pd

    from config import settings
    from engine.data import load_snapshot, panel_content_hash

    monkeypatch.setattr(settings, "panels_dir", tmp_path / "panels")
    run_dir = tmp_path / "runs" / "r_legacy"
    run_dir.mkdir(parents=True)

    idx = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=3, freq="D"))
    prices = pd.DataFrame({"AAA": [1.0, 1.1, 1.2]}, index=idx)
    # the OLD layout: parquet inside the run dir, meta with no panel_blob
    prices.to_parquet(run_dir / "panel.parquet")
    (run_dir / "panel_meta.json").write_text(json.dumps({
        "as_of": "2024-01-03", "sources": {}, "missing": [], "first_dates": {},
        "stale": False, "notes": [], "content_hash": panel_content_hash(prices),
    }))

    back = load_snapshot(run_dir)
    pd.testing.assert_frame_equal(back.prices, prices, check_freq=False)
