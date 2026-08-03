# Backend image: FastAPI + the engine. The frontend deploys separately (Vercel).
FROM python:3.11-slim

# libgomp is needed by numpy/pyarrow wheels; matplotlib renders headless (Agg).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend ./backend
COPY fixtures ./fixtures

# Writable state (SQLite, price cache, content-addressed panels, exports) lives
# on a mounted volume — NOT in the image, so a redeploy never wipes the journal.
ENV HONEST_DB_PATH=/data/honest.db \
    HONEST_CACHE_DIR=/data/price_cache \
    HONEST_RUNS_DIR=/data/runs \
    HONEST_PANELS_DIR=/data/panels \
    HONEST_EXPORTS_DIR=/data/exports \
    MPLCONFIGDIR=/tmp/mpl \
    PYTHONUNBUFFERED=1

EXPOSE 8080
CMD ["python", "-m", "uvicorn", "main:app", "--app-dir", "backend", \
     "--host", "0.0.0.0", "--port", "8080"]
