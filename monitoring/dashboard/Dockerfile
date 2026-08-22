# The image for every hosted process: the dashboard web service and the
# scheduled cycle/monitor services all run this same build, differing only
# in their start command. One image means the thing that placed an order is
# byte-for-byte the thing the dashboard reports on.
#
# Python 3.12, because that is what requirements.txt was frozen against
# (a 3.12.5 venv) and several of those pins REQUIRE it — numpy 2.5.1 ships
# no wheel for 3.11 at all, so the previous 3.11 pin could not install this
# file. Bump this and the pins together, never one alone.
FROM python:3.12-slim

# libgomp1: LightGBM's wheel is a compiled extension that dlopen()s
# libgomp.so.1 (OpenMP) at import time. It is NOT in python:slim, and its
# absence fails at `import lightgbm` — i.e. at runtime in the middle of a
# screening run, not at build time. psycopg2-binary bundles its own libpq
# and needs nothing here.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Documentation only (the actual port is $PORT, see below); matches the
# local docker-compose publish.
EXPOSE 8501
# Explicit, deliberate non-loopback bind: a container has to listen on
# 0.0.0.0 for its published port to work. The token check in server.py
# reads this same variable, so a networked dashboard without
# DASHBOARD_API_TOKEN set refuses its mutating endpoints.
ENV DASHBOARD_HOST=0.0.0.0
# Unbuffered stdout, so a crash loop's traceback reaches the platform log
# instead of dying in the buffer with the process.
ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "monitoring.dashboard.server"]
