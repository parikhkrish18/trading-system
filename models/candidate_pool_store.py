"""
Persists this cycle's actual Claude-advised picks from a concentrated-mode
screen — so a later flat-book reactivation
(execution/full_book_rebalance.py) can redeploy against this week's
already-computed analysis instead of paying for a fresh ensemble retrain +
rescreen (several minutes) every single time it finds the book empty.

Picks only, not every symbol Claude was merely consulted on: the
reactivation path treats every persisted row as confident=True and ranks
purely by llm_confidence (see full_book_rebalance.py's
_candidates_from_recent_pool), with no other way to tell "Claude picked
this" apart from "Claude analyzed this and recommended passing". A symbol
Claude wrote up as a pass would otherwise still be eligible to resurface
days later once it was one of the few candidates left in a thinning pool.

Deliberately separate from models/screener.py's own DB reads (training
data) for the same reason monitoring/breaker_state.py stays separate from
risk/circuit_breakers.py: this is pure persistence, not part of the
screening math. Uses to_sql(if_exists="append"), which auto-creates the
table on first write if data/schema/019_llm_candidate_pool.sql was never
explicitly migrated — this module never requires a separate migration step
to start working.
"""
from __future__ import annotations

import datetime as dt
import json
import logging

import pandas as pd
from sqlalchemy import text

from data.ingest.db import get_engine

logger = logging.getLogger(__name__)

_TABLE = "llm_candidate_pool"

# "This week's analysis" per the design intent — a pool older than this is
# treated as stale rather than reused, and the caller falls back to a fresh
# screen instead of redeploying against a plan the market may have moved
# well past.
MAX_POOL_AGE = dt.timedelta(days=7)


def save_candidate_pool(feature_set_id: str, pool: list[dict]) -> None:
    """
    `pool`: one dict per symbol Claude actually picked this cycle (see this
    module's docstring for why not every consulted symbol) — symbol, side,
    predicted_return, direction_agreement, conviction_score, confidence,
    take_profit_pct, stop_loss_pct, reasoning (the phase 2-4 list). Call
    once per concentrated, LLM-advised screen.
    Best-effort: a write failure here must never break the screen that
    produced this pool, only cost the next flat-book event its fast path.
    """
    if not pool:
        return
    ts = dt.datetime.now(tz=dt.UTC)
    rows = [
        {
            "ts": ts,
            "feature_set_id": feature_set_id,
            "symbol": c["symbol"],
            "side": c["side"],
            "predicted_return": c.get("predicted_return"),
            "direction_agreement": c.get("direction_agreement"),
            "conviction_score": c.get("conviction_score"),
            "confidence": c["confidence"],
            "take_profit_pct": c.get("take_profit_pct"),
            "stop_loss_pct": c.get("stop_loss_pct"),
            "reasoning": json.dumps(c.get("reasoning") or []),
        }
        for c in pool
    ]
    try:
        pd.DataFrame(rows).to_sql(_TABLE, get_engine(), if_exists="append", index=False)
    except Exception:
        logger.exception("Could not persist this cycle's candidate pool — reactivation will fall back to a fresh screen next time it is empty.")


def load_recent_pool(feature_set_id: str, max_age: dt.timedelta = MAX_POOL_AGE) -> list[dict]:
    """
    The most recently saved batch (every row sharing that save call's exact
    ts) for `feature_set_id`, provided it is within `max_age` — [] if none
    exists, the latest batch has aged out, or the read fails for any reason
    (e.g. the table hasn't been created by a first save yet).
    """
    try:
        # The latest batch picked out with a subquery, in one round trip --
        # binding a fetched timestamp back into a second query's WHERE
        # clause is exactly the kind of cross-driver round-trip that can
        # silently stop matching (string vs. native timestamp serialization
        # differs by backend), so the "which batch is latest" question is
        # answered entirely in SQL instead.
        df = pd.read_sql(
            text(
                "SELECT ts, symbol, side, predicted_return, direction_agreement, conviction_score, "
                "confidence, take_profit_pct, stop_loss_pct, reasoning FROM llm_candidate_pool "
                "WHERE feature_set_id = :fsid AND ts = ("
                "    SELECT MAX(ts) FROM llm_candidate_pool WHERE feature_set_id = :fsid"
                ") ORDER BY confidence DESC"
            ),
            get_engine(),
            params={"fsid": feature_set_id},
        )
    except Exception:
        logger.exception("Could not read the persisted candidate pool — falling back to a fresh screen.")
        return []

    if df.empty:
        return []

    latest_ts = pd.Timestamp(df["ts"].iloc[0])
    if latest_ts.tzinfo is None:
        latest_ts = latest_ts.tz_localize("UTC")
    if dt.datetime.now(tz=dt.UTC) - latest_ts.to_pydatetime() > max_age:
        return []

    return [
        {
            "symbol": row["symbol"],
            "side": row["side"],
            "predicted_return": row["predicted_return"],
            "direction_agreement": row["direction_agreement"],
            "conviction_score": row["conviction_score"],
            "confidence": row["confidence"],
            "take_profit_pct": row["take_profit_pct"],
            "stop_loss_pct": row["stop_loss_pct"],
            "reasoning": _decode_reasoning(row["reasoning"]),
        }
        for _, row in df.iterrows()
    ]


def _decode_reasoning(value: object) -> list:
    """
    `reasoning` is a JSONB column (data/schema/019_llm_candidate_pool.sql) --
    the driver hands back an already-deserialized list, never a string, so
    json.loads on it raises ("...not list"). Only decode when it actually is
    a string, which covers the one case that isn't JSONB: the table getting
    auto-created by to_sql (see this module's docstring) before the JSONB
    migration ever ran, where reasoning lands as plain TEXT instead.
    """
    if not value:
        return []
    if isinstance(value, str):
        return json.loads(value)
    return list(value)
