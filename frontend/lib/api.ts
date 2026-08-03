// Typed fetch client for the FastAPI backend. The backend owns all secrets;
// the browser only ever talks to the local backend (port 8000, or the
// fallback port dev.sh chose and passed via NEXT_PUBLIC_API_BASE).

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export class ApiError extends Error {
  code: string;
  status: number;
  constructor(status: number, code: string, message: string) {
    super(message);
    this.code = code;
    this.status = status;
  }
}

export async function api<T>(
  path: string,
  init?: RequestInit & { json?: unknown },
): Promise<T> {
  const { json, ...rest } = init ?? {};
  const res = await fetch(`${API_BASE}${path}`, {
    ...rest,
    headers: {
      ...(json !== undefined ? { "Content-Type": "application/json" } : {}),
      ...rest.headers,
    },
    body: json !== undefined ? JSON.stringify(json) : rest.body,
    cache: "no-store",
  });
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    const err = body?.error ?? {};
    throw new ApiError(
      res.status,
      err.code ?? "unknown",
      err.message ?? `Request failed (${res.status})`,
    );
  }
  return body as T;
}

export interface Health {
  status: string;
  llm_configured: boolean;
  db: string;
}
