"""
Replays the screener over past Mondays so the dashboard's "Was the model
right?" panel has matured predictions to score.

The panel compares each logged decision against what the price did next.
Every real decision in the table was logged in the last few days, so none of
them have played out yet and the panel is empty. This script fills that gap
by asking, for each of the last N Mondays: "if the screener had run that
morning, knowing only what was knowable then, what would it have picked?"

The whole point is the honesty of that question, so the cutoff rules are
strict and are the thing the tests actually pin down:

  * Training rows must be fully in the past. Not just `ts < as_of` — the
    target is a forward return, so a row whose forward window is still open
    on `as_of` was scored using prices from after `as_of`. Those rows are
    dropped too (see training_frame_as_of).
  * The scored snapshot is the latest feature row at `ts <= as_of`, i.e.
    that Monday's close-based features, never a later one.
  * A fresh ensemble is trained per Monday. Reusing one model across all of
    them would leak the whole period's data into every pick.

Nothing here re-implements scoring or sizing: it slices frames by date and
hands them to models.train / models.screener unchanged.

Rows are written with mode='backfill' so they can never be mistaken for
paper or live decisions, and so re-running is a clean delete-then-insert of
exactly those rows. Real decisions are never touched.

Usage:
    python -m scripts.backfill_decisions --feature-set-id v3 --weeks 26 --top-k 5
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd
from sqlalchemy import text

from config.settings import settings
from data.ingest.db import get_engine
from models.screener import (
    TradeCandidate,
    build_correlation_matrix,
    log_candidates,
    per_symbol_regimes,
    score_universe,
    select_trades,
)
from models.forecast.ensemble import EnsembleForecastModel
from models.regime.trend_chop_classifier import TREND
from models.train import load_feature_frame, load_training_frame
from monitoring.forecast_accuracy import BACKFILL_MODE

logger = logging.getLogger(__name__)

# Columns that are not model inputs, mirroring models.screener.run_screen.
_NON_FEATURE_COLS = ("symbol", "ts", "close", "fwd_return", "target_ts")


def symbols_in_feature_set(feature_set_id: str) -> list[str]:
    """
    The symbols that actually have features for this set — the backfill can
    only replay history that exists, so this is a better default than the
    live universe table (which may list names with no feature history).
    """
    engine = get_engine()
    rows = pd.read_sql(
        text("SELECT DISTINCT symbol FROM features WHERE feature_set_id = :fsid ORDER BY symbol"),
        engine,
        params={"fsid": feature_set_id},
    )
    return rows["symbol"].tolist()


def recent_mondays(end: pd.Timestamp, weeks: int) -> list[pd.Timestamp]:
    """
    The `weeks` most recent Mondays at or before `end`, oldest first, at
    midnight UTC. Monday is just a stable weekly cadence — it carries no
    meaning beyond "one pick batch per week, always the same weekday".
    """
    end = pd.Timestamp(end)
    if end.tz is None:
        end = end.tz_localize("UTC")
    else:
        end = end.tz_convert("UTC")
    end = end.normalize()

    last_monday = end - pd.Timedelta(int(end.dayofweek), unit="D")
    return sorted(last_monday - pd.Timedelta(i, unit="W") for i in range(weeks))


def attach_target_ts(feature_frame: pd.DataFrame, train_frame: pd.DataFrame, target_horizon_days: int) -> pd.DataFrame:
    """
    Adds `target_ts` — the date the row's forward-return target is read from.

    models.train.load_training_frame builds its target by shifting close
    `target_horizon_days` rows back within each symbol, so the matching date
    is the same positional shift on the *unfiltered* feature frame. It has to
    be computed there rather than on train_frame, because load_training_frame
    has already dropped the tail rows whose target date would be needed here.
    """
    dated = feature_frame.sort_values(["symbol", "ts"]).copy()
    dated["target_ts"] = dated.groupby("symbol")["ts"].shift(-target_horizon_days)
    return train_frame.merge(dated[["symbol", "ts", "target_ts"]], on=["symbol", "ts"], how="left")


def training_frame_as_of(train_frame: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """
    The training rows a model run on `as_of` was allowed to see.

    Both conditions matter and neither implies the other in an obvious way:
    `ts < as_of` is the feature-side cutoff (the row itself must be history),
    and `target_ts < as_of` is the label-side cutoff (its forward return must
    have already finished playing out). Dropping only the first would train
    on answers that had not happened yet.
    """
    if train_frame.empty:
        return train_frame
    return train_frame.loc[(train_frame["ts"] < as_of) & (train_frame["target_ts"] < as_of)].copy()


def snapshot_as_of(
    feature_frame: pd.DataFrame, as_of: pd.Timestamp, max_staleness_days: int = 10
) -> pd.DataFrame:
    """
    The most recent feature row per symbol at `ts <= as_of` — the counterpart
    of models.screener.load_latest_features, frozen to a past date.

    `as_of` itself is included: those are that day's close-based features,
    which is what a screener running after the close would have had. Symbols
    whose newest row is more than `max_staleness_days` old are dropped rather
    than scored on stale inputs (a symbol that stopped reporting shouldn't
    keep generating picks off its last known state).
    """
    available = feature_frame.loc[feature_frame["ts"] <= as_of]
    if available.empty:
        return available
    latest = available.sort_values("ts").groupby("symbol", as_index=False).tail(1)
    fresh = latest.loc[latest["ts"] >= as_of - pd.Timedelta(max_staleness_days, unit="D")]
    return fresh.reset_index(drop=True)


def picks_for_date(
    train_frame: pd.DataFrame,
    feature_frame: pd.DataFrame,
    as_of: pd.Timestamp,
    n_ensemble_models: int = 5,
    min_direction_agreement: float = 0.8,
    min_abs_return: float = 0.0,
    top_k: int = 5,
    min_train_rows: int = 250,
) -> list[TradeCandidate]:
    """
    One Monday's shortlist: train on history, score that day's snapshot, size.

    Mirrors models.screener.run_screen step for step, with every input first
    cut down to what existed on `as_of`. Returns [] when there isn't enough
    history to train on, or when nothing clears the confidence bar.

    `is_shortable_fn` is deliberately left off: this is offline replay with no
    broker to ask, and inventing a shortability answer for a past date would
    be a guess dressed up as a fact.
    """
    train_as_of = training_frame_as_of(train_frame, as_of)
    if len(train_as_of) < min_train_rows:
        logger.info(
            "%s: only %d training row(s) available before this date (need %d) — skipped.",
            as_of.date(), len(train_as_of), min_train_rows,
        )
        return []

    snapshot = snapshot_as_of(feature_frame, as_of)
    if snapshot.empty:
        logger.info("%s: no feature snapshot available on or before this date — skipped.", as_of.date())
        return []

    feature_cols = [c for c in train_as_of.columns if c not in _NON_FEATURE_COLS]
    forecast_scale = float(train_as_of["fwd_return"].std())

    ensemble = EnsembleForecastModel(n_models=n_ensemble_models)
    ensemble.fit(train_as_of[feature_cols], train_as_of["fwd_return"])

    scored = score_universe(ensemble, snapshot, feature_cols, min_direction_agreement, min_abs_return)

    return select_trades(
        scored,
        regime=TREND,
        regime_by_symbol=per_symbol_regimes(snapshot),
        forecast_scale=forecast_scale,
        max_position_pct=settings.max_single_position_pct,
        max_short_position_pct=settings.max_short_position_pct,
        max_correlated_exposure_pct=settings.max_correlated_exposure_pct,
        correlation_matrix=build_correlation_matrix(train_as_of[["symbol", "ts", "close"]]),
        top_k=top_k,
    )


def clear_backfill(mode: str = BACKFILL_MODE) -> int:
    """
    Deletes previously backfilled rows so a re-run replaces rather than
    duplicates them. Scoped to `mode` alone, which is what keeps real
    paper/live decisions out of reach of this script.
    """
    engine = get_engine()
    with engine.begin() as conn:
        result = conn.execute(text("DELETE FROM decisions WHERE mode = :mode"), {"mode": mode})
    return int(result.rowcount or 0)


def run_backfill(
    feature_set_id: str,
    symbols: list[str],
    weeks: int = 26,
    target_horizon_days: int = 5,
    n_ensemble_models: int = 5,
    min_direction_agreement: float = 0.8,
    min_abs_return: float = 0.0,
    top_k: int = 5,
    min_train_rows: int = 250,
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    Replays the screener across the last `weeks` Mondays and logs the picks.

    Both frames are loaded once and sliced per date in memory: the cutoffs are
    what make a run honest, and doing them in one place is easier to verify
    than a per-date query that could quietly forget its WHERE clause.

    Returns one summary row per Monday (including skipped ones, at 0 picks).
    """
    feature_frame = load_feature_frame(feature_set_id, symbols)
    train_frame = attach_target_ts(
        feature_frame, load_training_frame(feature_set_id, symbols, target_horizon_days), target_horizon_days
    )

    mondays = recent_mondays(feature_frame["ts"].max(), weeks)
    logger.info(
        "Replaying %d Monday(s) from %s to %s over %d symbol(s).",
        len(mondays), mondays[0].date(), mondays[-1].date(), len(symbols),
    )

    if not dry_run:
        deleted = clear_backfill()
        logger.info("Cleared %d existing mode='%s' row(s) before inserting.", deleted, BACKFILL_MODE)

    summary = []
    for as_of in mondays:
        candidates = picks_for_date(
            train_frame,
            feature_frame,
            as_of,
            n_ensemble_models=n_ensemble_models,
            min_direction_agreement=min_direction_agreement,
            min_abs_return=min_abs_return,
            top_k=top_k,
            min_train_rows=min_train_rows,
        )
        n_written = 0
        if candidates and not dry_run:
            # The historical timestamp is the whole point — it is what lets the
            # accuracy panel find prices from after the decision.
            n_written = log_candidates(
                candidates, feature_set_id, mode=BACKFILL_MODE, ts=as_of.to_pydatetime()
            )

        summary.append(
            {
                "as_of": as_of.date(),
                "n_picks": len(candidates),
                "n_long": sum(1 for c in candidates if c.side == "long"),
                "n_short": sum(1 for c in candidates if c.side == "short"),
                "n_written": n_written,
            }
        )
        logger.info("%s: %d pick(s).", as_of.date(), len(candidates))

    return pd.DataFrame(summary)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill historical screener picks into the decisions table (mode='backfill')."
    )
    parser.add_argument("--feature-set-id", required=True)
    parser.add_argument("--symbols", default=None, help="Comma-separated; defaults to every symbol in the feature set.")
    parser.add_argument("--weeks", type=int, default=26, help="How many past Mondays to replay.")
    parser.add_argument("--target-horizon-days", type=int, default=5)
    parser.add_argument("--n-ensemble-models", type=int, default=5)
    parser.add_argument("--min-direction-agreement", type=float, default=0.8)
    parser.add_argument("--min-abs-return", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-train-rows", type=int, default=250)
    parser.add_argument("--dry-run", action="store_true", help="Report what would be logged without writing.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    symbols = (
        [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        if args.symbols
        else symbols_in_feature_set(args.feature_set_id)
    )
    if not symbols:
        raise SystemExit(f"No symbols found for feature set '{args.feature_set_id}'.")

    summary = run_backfill(
        args.feature_set_id,
        symbols,
        weeks=args.weeks,
        target_horizon_days=args.target_horizon_days,
        n_ensemble_models=args.n_ensemble_models,
        min_direction_agreement=args.min_direction_agreement,
        min_abs_return=args.min_abs_return,
        top_k=args.top_k,
        min_train_rows=args.min_train_rows,
        dry_run=args.dry_run,
    )

    print(summary.to_string(index=False))
    total = int(summary["n_written"].sum())
    skipped = int((summary["n_picks"] == 0).sum())
    if args.dry_run:
        print(f"\nDry run — nothing written. {int(summary['n_picks'].sum())} pick(s) would have been logged.")
    else:
        print(f"\nWrote {total} row(s) to decisions with mode='{BACKFILL_MODE}'.")
    print(f"{skipped} of {len(summary)} date(s) produced no picks (insufficient history, or nothing confident enough).")


if __name__ == "__main__":
    main()
