"""In-memory rate limiter for the one endpoint that spends money.

A public LLM endpoint with our key behind it is a public wallet. $5 of
gpt-5-nano is ~27,000 translations, which a scripted abuser would drain in
under an hour. This limiter does NOT cap dollars directly — that hard ceiling
lives in OpenAI's own billing cap. What it does is keep that ceiling from being
hit by abuse, and keep legitimate users getting a fair share, via three limits:

  - per-IP per-minute  (burst control)
  - per-IP per-day     (one visitor can't hog the budget)
  - global per-day      (budget guard: bounds total daily spend)

Defaults are tuned so a busy month stays comfortably under $5 (global 800/day
≈ 24k/month ≈ $4.5 uncached, far less once OpenAI caches the system prompt).
All three are overridable via HONEST_RL_* env vars.

In-memory is adequate for a single small instance (Fly auto-stop keeps one
machine warm). If you ever run multiple instances, swap the counters for a
shared store (Redis, or the SQLite DB) — the check() surface stays the same.
"""
from __future__ import annotations

import threading
from collections import defaultdict
from datetime import datetime, timezone

from config import settings


class RateLimit(Exception):
    """Raised when a request exceeds a limit. Carries a Retry-After hint."""

    def __init__(self, scope: str, retry_after: int, message: str):
        super().__init__(message)
        self.scope = scope
        self.retry_after = retry_after
        self.message = message


class _Limiter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._minute: dict[str, int] = {}   # "ip@YYYY-MM-DDTHH:MM" -> count
        self._day: dict[str, int] = {}      # "ip@YYYY-MM-DD"       -> count
        self._global: dict[str, int] = {}   # "YYYY-MM-DD"          -> count

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    def check(self, ip: str) -> None:
        now = self._now()
        minute = now.strftime("%Y-%m-%dT%H:%M")
        day = now.strftime("%Y-%m-%d")
        with self._lock:
            # Fixed calendar windows; drop any bucket that isn't the current
            # one so the dicts can't grow without bound.
            self._minute = {k: v for k, v in self._minute.items() if k.endswith(minute)}
            self._day = {k: v for k, v in self._day.items() if k.endswith(day)}
            self._global = {k: v for k, v in self._global.items() if k == day}

            gkey, dkey, mkey = day, f"{ip}@{day}", f"{ip}@{minute}"

            if self._global.get(gkey, 0) >= settings.rl_global_per_day:
                raise RateLimit(
                    "global_daily", 3600,
                    "The daily translation budget is used up — try again tomorrow.")
            if self._day.get(dkey, 0) >= settings.rl_per_day:
                raise RateLimit(
                    "ip_daily", 3600,
                    "You've hit today's translation limit. Try again tomorrow.")
            if self._minute.get(mkey, 0) >= settings.rl_per_min:
                raise RateLimit(
                    "ip_minute", 60,
                    "Too many translations in a row — give it a minute.")

            self._minute[mkey] = self._minute.get(mkey, 0) + 1
            self._day[dkey] = self._day.get(dkey, 0) + 1
            self._global[gkey] = self._global.get(gkey, 0) + 1

    def reset(self) -> None:
        """For tests."""
        with self._lock:
            self._minute.clear()
            self._day.clear()
            self._global.clear()


limiter = _Limiter()


def client_ip(request) -> str:
    """The caller's IP. Behind Vercel/Fly the real IP is the first hop of
    X-Forwarded-For; the socket peer is the proxy. Only the FIRST entry is
    trustworthy (later ones are attacker-supplied)."""
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


_LOCAL = {"127.0.0.1", "::1", "localhost"}


def is_local(ip: str) -> bool:
    """Same-machine callers are exempt: the limiter guards a PUBLIC endpoint
    against abuse, and it shouldn't nag the developer running the app locally.
    In production the real user IP arrives via X-Forwarded-For, so this never
    exempts remote traffic."""
    return ip in _LOCAL
