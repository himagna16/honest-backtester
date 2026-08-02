import os
import sys
from pathlib import Path

# Tests import backend modules the same way uvicorn --app-dir does.
BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

# Tests never touch the network; the seeded parquet cache and the checked-in
# fixture CSVs are the only data sources.
os.environ.setdefault("HONEST_OFFLINE", "1")

# Isolate the DB and run artifacts from dev data (config reads env at import).
import tempfile  # noqa: E402

_TMP = Path(tempfile.mkdtemp(prefix="honest_test_"))
os.environ.setdefault("HONEST_DB_PATH", str(_TMP / "honest_test.db"))
os.environ.setdefault("HONEST_RUNS_DIR", str(_TMP / "runs"))
os.environ.setdefault("HONEST_EXPORTS_DIR", str(_TMP / "exports"))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

GOLDEN = Path(__file__).parent / "golden"
FIXTURES = BACKEND.parent / "fixtures"


def load_csv_panel(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True).astype(float)
    df.index.name = "Date"
    return df


@pytest.fixture(scope="session")
def three_ticker_panel() -> pd.DataFrame:
    return load_csv_panel(GOLDEN / "three_ticker.csv")


@pytest.fixture(scope="session")
def cohort_panel() -> pd.DataFrame:
    return load_csv_panel(GOLDEN / "cohort.csv")


@pytest.fixture(scope="session")
def war_panel() -> pd.DataFrame:
    return load_csv_panel(FIXTURES / "war_list_prices.csv")


@pytest.fixture(scope="session")
def sp500_panel() -> pd.DataFrame:
    return load_csv_panel(FIXTURES / "sp500_prices.csv")
