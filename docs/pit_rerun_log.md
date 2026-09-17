# Saved reports re-run with point-in-time S&P 500 membership

Generated 2026-09-17 by `scripts/rerun_saved_reports_pit.py`. Every figure is read back from the stored reports; the originals are kept in the journal as history. Both the old and new numbers are net of 5 bps. New reports use the current engine, which also changed after most originals were made (see `docs/pit_membership_study.md`).

| Strategy | Old report | Old: return / verdict | New report | New: return / verdict | SPY (new) | Universe check (new) |
|---|---|---|---|---|---:|---|
| War list 12mo momentum top-5 | `rep_a19cab866c` | +1,974.2% / amber | `rep_5ba385ec93` | +1,971.2% / red | +93.7% | fatal: Selection bias detected |
| War list 12mo momentum top-5 (S&P 500 re-run) | `rep_433acc9eb9` | +3,553.1% / green | `rep_cfada2fc3f` | +286.4% / green | +93.7% | note: Point-in-time index membership |
| DoD verify strategy | `rep_2c6332f417` | +1,816.5% / amber | `rep_9534f1bae0` | +1,646.6% / red | +41.9% | fatal: Selection bias detected |
| CAS check | `rep_d465bba2f9` | +949.8% / amber | `rep_ab16e74a5e` | +359.1% / green | +41.6% | note: Point-in-time index membership |
| CAS check 2 | `rep_efba12d48a` | +949.8% / amber | `rep_3aa8a59292` | +359.1% / green | +41.6% | note: Point-in-time index membership |
| Tech momentum fast-growers (SP500 tech) | `rep_e9204053db` | +11,920.4% / amber | `rep_06e1fd7bcb` | +889.3% / amber | +305.4% | warning: Point-in-time index, with gaps |
| Least volatile SPY stocks (quarterly) | `rep_43fc882769` | +4,126.6% / amber | `rep_094683bcae` | +357.5% / green | +305.4% | warning: Point-in-time index, with gaps |
| 6mo momentum top-5 | `rep_96338a692d` | +18,693.0% / amber | `rep_5cad8e2df2` | +211.0% / amber | +316.1% | warning: Point-in-time index, with gaps |

## Hand-picked runs: the neutral comparison, before and after

- **War list 12mo momentum top-5** (`rep_5ba385ec93`): your list +1,971.2% vs S&P 500 as it stood +286.3% → 5.36× (point-in-time: True). Before: today's list +3,553.1%, 0.57× on the old report.
- **DoD verify strategy** (`rep_9534f1bae0`): your list +1,646.6% vs S&P 500 as it stood +363.5% → 3.77× (point-in-time: True). Before: today's list +960.2%, 1.81× on the old report.
