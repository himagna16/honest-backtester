"""Central configuration. Secrets come only from .env / environment.

The single secret is ANTHROPIC_API_KEY, and only the /api/translate-strategy
endpoint needs it — the engine, audits, journal, and exports work offline.
"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HONEST_",
        env_file=(REPO_ROOT / ".env", REPO_ROOT / "backend" / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # The API key deliberately has no HONEST_ prefix — it's the standard name.
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-5"

    db_path: Path = REPO_ROOT / "data" / "honest.db"
    cache_dir: Path = REPO_ROOT / "data" / "price_cache"
    runs_dir: Path = REPO_ROOT / "data" / "runs"
    exports_dir: Path = REPO_ROOT / "data" / "exports"
    fixtures_dir: Path = REPO_ROOT / "fixtures"

    account_size: float = 10_000.0
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    def ensure_dirs(self) -> None:
        for d in (self.db_path.parent, self.cache_dir, self.runs_dir, self.exports_dir):
            d.mkdir(parents=True, exist_ok=True)


class _KeyFallback(BaseSettings):
    """Reads the unprefixed ANTHROPIC_API_KEY from env/.env."""

    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", REPO_ROOT / "backend" / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )
    anthropic_api_key: str = ""


settings = Settings()
if not settings.anthropic_api_key:
    settings.anthropic_api_key = _KeyFallback().anthropic_api_key
