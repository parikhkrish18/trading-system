FROM python:3.11-slim

# LightGBM's wheel dlopen()s libgomp.so.1 at import time; python:slim
# doesn't ship it, and the failure is at runtime, not at build.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
# Shell form so the platform-injected $PORT is honoured (Railway routes the
# public domain to it); falls back to 8501 when nothing is injected.
CMD uvicorn monitoring.dashboard.server:app --host 0.0.0.0 --port ${PORT:-8501}
