CREATE TABLE strategies (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  family_key TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE strategy_versions (
  id TEXT PRIMARY KEY,
  strategy_id TEXT NOT NULL REFERENCES strategies(id),
  version INTEGER NOT NULL,
  spec_version INTEGER NOT NULL,
  spec_json TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE (strategy_id, version)
);

CREATE TABLE runs (
  id TEXT PRIMARY KEY,
  strategy_version_id TEXT NOT NULL REFERENCES strategy_versions(id),
  status TEXT NOT NULL DEFAULT 'queued',
  spec_json TEXT NOT NULL,
  snapshot_path TEXT,
  snapshot_hash TEXT,
  data_as_of TEXT,
  data_source TEXT,
  error TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  finished_at TEXT
);

CREATE TABLE reports (
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES runs(id),
  verdict_color TEXT NOT NULL,
  verdict_headline TEXT NOT NULL,
  report_json TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE attempt_counters (
  universe_key TEXT PRIMARY KEY,
  attempts INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE translations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  prompt_version TEXT NOT NULL,
  input_hash TEXT NOT NULL,
  input_text TEXT NOT NULL,
  status TEXT NOT NULL,
  response_json TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE (prompt_version, input_hash)
);

CREATE TABLE events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL DEFAULT (datetime('now')),
  type TEXT NOT NULL,
  payload TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX idx_versions_strategy ON strategy_versions(strategy_id);

CREATE INDEX idx_runs_version ON runs(strategy_version_id);

CREATE INDEX idx_reports_run ON reports(run_id);

CREATE INDEX idx_events_type ON events(type);
