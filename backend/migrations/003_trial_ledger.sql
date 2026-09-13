-- Trial ledger: the Sharpe ratios of every attempt against a universe.
--
-- attempt_counters already records HOW MANY strategies were tried against a
-- universe, which is what the multiple-testing finding quotes. The Deflated
-- Sharpe Ratio (Bailey & Lopez de Prado 2014) needs more than the count: its
-- luck threshold
--     SR0 = sqrt(V[{SR_n}]) * ( (1-g)*Z^-1[1 - 1/N] + g*Z^-1[1 - 1/(N*e)] )
-- is scaled by the VARIANCE of the trial Sharpes. With only a count we cannot
-- compute SR0, so the audit had to report the DSR as not computable and say
-- why. This table is that missing spread.
--
-- One row per completed run, keyed by the run so a replay of the same run can
-- never double-count itself into its own haircut.
CREATE TABLE IF NOT EXISTS trial_sharpes (
    run_id        TEXT PRIMARY KEY,
    universe_key  TEXT NOT NULL,
    sharpe_annual REAL NOT NULL,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_trial_sharpes_universe
    ON trial_sharpes (universe_key);
