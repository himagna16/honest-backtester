"use client";

import { useEffect, useState } from "react";
import { api, type Health } from "@/lib/api";

export default function Home() {
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api<Health>("/health")
      .then(setHealth)
      .catch((e) => setError(String(e)));
  }, []);

  return (
    <div>
      <h1>The Honest Backtester</h1>
      <p className="muted">
        Describe a trading strategy in plain English. Get a backtest that tries
        to kill it.
      </p>
      <div className="card">
        {health ? (
          <p>
            Backend connected. LLM translation:{" "}
            {health.llm_configured ? "configured" : "not configured (set ANTHROPIC_API_KEY in .env)"}
          </p>
        ) : error ? (
          <p>
            Backend not reachable — start it with <code>make dev</code>.
          </p>
        ) : (
          <p className="faint">Checking backend…</p>
        )}
      </div>
    </div>
  );
}
