"""
Shared HTTP helper for Polygon API calls. Every Polygon-backed ingestion
script (fundamentals.py, news.py) hits the same free-tier rate limit
(~5 requests/minute, observed empirically — a plain per-symbol loop over a
few hundred universe symbols will hit this), so the 429 backoff logic lives
here once instead of being duplicated per script.
"""
from __future__ import annotations

import logging
import time

import requests

logger = logging.getLogger(__name__)

DEFAULT_SLEEP_SECONDS = 13.0  # paced for a ~5 req/min tier with margin


def polygon_get(url: str, params: dict, timeout: int = 30, max_retries: int = 5) -> requests.Response:
    """
    GET with 429-aware backoff. Without this, a multi-symbol universe run
    just crashes partway through the first time the per-minute quota is hit.
    """
    for attempt in range(1, max_retries + 1):
        resp = requests.get(url, params=params, timeout=timeout)
        if resp.status_code != 429:
            resp.raise_for_status()
            return resp
        retry_after = float(resp.headers.get("Retry-After", DEFAULT_SLEEP_SECONDS))
        logger.warning("Polygon rate limit hit (attempt %s/%s) — sleeping %ss", attempt, max_retries, retry_after)
        time.sleep(retry_after)
    raise RuntimeError(f"Polygon rate limit exceeded after {max_retries} retries: {url}")
