"""
TARGET_MODE — what the model is actually trained to predict.

"absolute" predicts each stock's raw forward return, which is dominated by
whatever the whole market did; the model spent its capacity learning market
drift and was then credited with the drift as skill. "relative" predicts
the cross-sectional excess over the same-day universe mean, with features
z-scored per date to match.

The invariant that matters most and is tested hardest: whichever label the
model trains on, `fwd_return` stays the untouched absolute return, because
every money metric — trade returns, the buy-and-hold benchmark, excess —
has to stay measured in real money to be comparable across modes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.settings import Settings
from models import screener as scr
from models import train

# --------------------------------------------------------------------------
# the setting
# --------------------------------------------------------------------------


def test_target_mode_defaults_to_relative():
    assert Settings(_env_file=None).target_mode == "relative"


def test_target_mode_reads_from_the_environment(monkeypatch):
    monkeypatch.setenv("TARGET_MODE", "absolute")
    assert Settings(_env_file=None).target_mode == "absolute"


def test_an_unknown_target_mode_fails_loudly(monkeypatch):
    """
    A typo must not silently fall back to absolute — that is precisely how a
    run gets reported under the wrong label.
    """
    monkeypatch.setattr(train, "load_feature_frame", lambda *a, **k: pd.DataFrame())
    with pytest.raises(ValueError, match="target_mode"):
        train.load_training_frame("v4", ["AAPL"], 5, target_mode="market_neutral")


# --------------------------------------------------------------------------
# load_training_frame
# --------------------------------------------------------------------------


def _feature_frame():
    """
    Three symbols x six days. Every symbol's price path is the market's path
    times a symbol-specific drift, so the cross-sectional excess is a clean,
    hand-checkable quantity.
    """
    dates = pd.date_range("2026-01-05", periods=6, freq="B")
    rows = []
    for symbol, step in (("AAA", 1.02), ("BBB", 1.01), ("CCC", 1.00)):
        rows.append(
            pd.DataFrame(
                {
                    "symbol": symbol,
                    "ts": dates,
                    "close": 100.0 * step ** np.arange(len(dates)),
                    "rsi": np.linspace(30, 70, len(dates)),
                    "vol": np.linspace(1.0, 2.0, len(dates)),
                }
            )
        )
    return pd.concat(rows, ignore_index=True).sort_values(["symbol", "ts"])


def _load(monkeypatch, mode, horizon=2):
    monkeypatch.setattr(train, "load_feature_frame", lambda *a, **k: _feature_frame())
    return train.load_training_frame("v4", ["AAA", "BBB", "CCC"], horizon, target_mode=mode)


def test_absolute_mode_targets_the_raw_forward_return(monkeypatch):
    df = _load(monkeypatch, "absolute")
    assert list(df["target"]) == pytest.approx(list(df["fwd_return"]))


def test_relative_mode_targets_the_excess_over_the_same_day_universe(monkeypatch):
    df = _load(monkeypatch, "relative")
    expected = df["fwd_return"] - df.groupby("ts")["fwd_return"].transform("mean")
    assert list(df["target"]) == pytest.approx(list(expected))


def test_relative_targets_sum_to_zero_on_every_date(monkeypatch):
    """
    That is the whole point: the market term is gone, so there is no drift
    left for the model to be accidentally rewarded for predicting.
    """
    df = _load(monkeypatch, "relative")
    per_date = df.groupby("ts")["target"].mean()
    assert per_date.abs().max() == pytest.approx(0.0, abs=1e-12)


def test_fwd_return_stays_absolute_in_both_modes(monkeypatch):
    """
    The invariant everything downstream depends on. If relative mode
    overwrote fwd_return, the benchmark would become zero by construction
    and every excess figure would be meaningless.
    """
    absolute = _load(monkeypatch, "absolute")
    relative = _load(monkeypatch, "relative")
    assert list(absolute["fwd_return"]) == pytest.approx(list(relative["fwd_return"]))
    assert relative["fwd_return"].mean() > 0  # a rising market is still rising


def test_relative_mode_z_scores_features_within_each_date(monkeypatch):
    df = _load(monkeypatch, "relative")
    for col in ("rsi", "vol"):
        per_date = df.groupby("ts")[col]
        assert per_date.mean().abs().max() == pytest.approx(0.0, abs=1e-12)


def test_absolute_mode_leaves_features_on_their_raw_scale(monkeypatch):
    df = _load(monkeypatch, "absolute")
    assert df["rsi"].min() >= 30 and df["rsi"].max() <= 70


def test_close_is_never_treated_as_a_feature(monkeypatch):
    """
    close is the input the label is built from; z-scoring or feeding it to
    the model would be a look-ahead disaster dressed as a feature.
    """
    df = _load(monkeypatch, "relative")
    assert "close" not in train.feature_columns(df)
    assert df["close"].max() > 100  # untouched, still raw prices


def test_feature_columns_excludes_both_label_columns():
    df = pd.DataFrame(
        columns=["symbol", "ts", "close", "fwd_return", "target", "rsi", "macd"]
    )
    assert train.feature_columns(df) == ["rsi", "macd"]


def test_cross_sectional_feature_columns_excludes_broadcast_macro_features():
    """
    macro_mkt_sentiment is still a real, active model input -- feature_columns()
    must keep it. It's only excluded from the cross-sectional z-scoring pass
    (see NON_CROSS_SECTIONAL_FEATURES's own docstring for why).
    """
    df = pd.DataFrame(
        columns=["symbol", "ts", "close", "fwd_return", "target", "rsi", "macro_mkt_sentiment"]
    )
    assert train.cross_sectional_feature_columns(df) == ["rsi"]
    assert "macro_mkt_sentiment" in train.feature_columns(df)


def test_relative_mode_leaves_a_broadcast_macro_feature_raw_not_zeroed(monkeypatch):
    """
    Regression: macro_mkt_sentiment is the SAME value for every symbol on a
    given date (see features/qualitative/macro_sentiment.py) -- a column
    with zero cross-sectional variation has its per-date std pinned at
    exactly 0.0 every day, so blindly cross-sectionally z-scoring it (like
    every other feature) would collapse it to a permanent 0.0 regardless of
    the real value (see cross_sectional_zscore's own zero-variance guard).
    NON_CROSS_SECTIONAL_FEATURES exists specifically to keep this one raw.
    """
    df = _feature_frame()
    dates = df["ts"].unique()
    macro_by_date = dict(zip(dates, np.linspace(-0.4, 0.4, len(dates)), strict=False))
    df = df.assign(macro_mkt_sentiment=df["ts"].map(macro_by_date))
    monkeypatch.setattr(train, "load_feature_frame", lambda *a, **k: df)

    result = train.load_training_frame("v4", ["AAA", "BBB", "CCC"], 2, target_mode="relative")

    expected = result["ts"].map(macro_by_date)
    assert list(result["macro_mkt_sentiment"]) == pytest.approx(list(expected))
    assert not (result["macro_mkt_sentiment"] == 0.0).all()
    # A genuinely cross-sectional feature in the same frame is still z-scored normally.
    per_date = result.groupby("ts")["rsi"]
    assert per_date.mean().abs().max() == pytest.approx(0.0, abs=1e-12)


def test_relative_labels_are_smaller_than_absolute_ones_in_a_trending_market(monkeypatch):
    """
    Sanity check on the premise: the absolute label carries the market's
    drift, the relative label does not, so the relative one is centred on
    zero while the absolute one is not.
    """
    absolute = _load(monkeypatch, "absolute")
    relative = _load(monkeypatch, "relative")
    assert absolute["target"].mean() > 0.01
    assert relative["target"].mean() == pytest.approx(0.0, abs=1e-12)


def test_the_cross_sectional_transforms_cannot_see_the_future(monkeypatch):
    """
    load_training_frame applies the label and feature transforms to the whole
    frame BEFORE the walk-forward splits it, which looks like leakage and is
    not: both group by ts, so a row's value depends only on other stocks on
    its own date.

    Proven directly rather than argued — transform the full history, then
    transform a truncated copy that has never seen anything after the cutoff,
    and require every overlapping row to come out bit-for-bit identical. If
    any future date could influence an earlier row, these would differ.
    """
    rng = np.random.default_rng(0)
    dates = pd.date_range("2026-01-01", periods=40)
    frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "symbol": f"S{s}",
                    "ts": dates,
                    "close": 100 * np.cumprod(1 + rng.normal(scale=0.01, size=40)),
                    "f1": rng.normal(size=40),
                }
            )
            for s in range(12)
        ],
        ignore_index=True,
    )
    cutoff = dates[25]

    monkeypatch.setattr(train, "load_feature_frame", lambda *a, **k: frame)
    full = train.load_training_frame("v4", [], 2, target_mode="relative")
    monkeypatch.setattr(train, "load_feature_frame", lambda *a, **k: frame[frame["ts"] < cutoff])
    past_only = train.load_training_frame("v4", [], 2, target_mode="relative")

    merged = full.merge(past_only, on=["symbol", "ts"], suffixes=("_full", "_past"))
    assert len(merged) > 100  # the comparison is actually exercising rows
    for col in ("f1", "target"):
        assert list(merged[f"{col}_full"]) == pytest.approx(list(merged[f"{col}_past"]))


# --------------------------------------------------------------------------
# wiring: the mode has to reach the code that uses it
# --------------------------------------------------------------------------


def test_run_walk_forward_defaults_to_the_configured_mode(monkeypatch):
    monkeypatch.setattr(train.settings, "target_mode", "relative")
    captured = {}

    def _capture(feature_set_id, symbols, horizon, mode=None):
        captured["mode"] = mode
        raise RuntimeError("stop — only the mode resolution is under test")

    monkeypatch.setattr(train, "load_training_frame", _capture)
    with pytest.raises(RuntimeError):
        train.run_walk_forward("v4", ["AAPL"])

    assert captured["mode"] == "relative"


def test_an_explicit_mode_beats_the_config(monkeypatch):
    monkeypatch.setattr(train.settings, "target_mode", "relative")
    captured = {}

    def _capture(feature_set_id, symbols, horizon, mode=None):
        captured["mode"] = mode
        raise RuntimeError("stop")

    monkeypatch.setattr(train, "load_training_frame", _capture)
    with pytest.raises(RuntimeError):
        train.run_walk_forward("v4", ["AAPL"], target_mode="absolute")

    assert captured["mode"] == "absolute"


def test_the_screener_scores_on_the_same_scale_it_trained_on(monkeypatch):
    """
    A model trained on per-date z-scores and then scored on raw feature
    levels would be reading a different scale from the one it learned, and
    would produce confident nonsense. load_latest_features must normalise
    in relative mode.
    """
    monkeypatch.setattr(scr.settings, "target_mode", "relative")
    monkeypatch.setattr(
        scr,
        "load_feature_frame",
        lambda *a, **k: pd.DataFrame(
            {
                "symbol": ["AAA", "BBB", "CCC"],
                "ts": pd.to_datetime(["2026-02-02"] * 3),
                "close": [100.0, 200.0, 300.0],
                "rsi": [30.0, 50.0, 70.0],
            }
        ),
    )

    latest = scr.load_latest_features("v4", ["AAA", "BBB", "CCC"])

    assert latest["rsi"].mean() == pytest.approx(0.0, abs=1e-12)
    assert latest["rsi"].iloc[0] < 0 < latest["rsi"].iloc[2]  # ranking preserved
    assert "_as_of" not in latest.columns
    assert list(latest["close"]) == [100.0, 200.0, 300.0]  # not a feature, untouched


def test_the_screener_leaves_a_broadcast_macro_feature_raw_in_relative_mode(monkeypatch):
    """Same regression as train.load_training_frame's, for the live-scoring path."""
    monkeypatch.setattr(scr.settings, "target_mode", "relative")
    monkeypatch.setattr(
        scr,
        "load_feature_frame",
        lambda *a, **k: pd.DataFrame(
            {
                "symbol": ["AAA", "BBB", "CCC"],
                "ts": pd.to_datetime(["2026-02-02"] * 3),
                "close": [100.0, 200.0, 300.0],
                "rsi": [30.0, 50.0, 70.0],
                "macro_mkt_sentiment": [-0.41, -0.41, -0.41],
            }
        ),
    )

    latest = scr.load_latest_features("v4", ["AAA", "BBB", "CCC"])

    assert list(latest["macro_mkt_sentiment"]) == [-0.41, -0.41, -0.41]
    assert latest["rsi"].mean() == pytest.approx(0.0, abs=1e-12)  # rsi still z-scored normally


def test_the_screener_leaves_features_raw_in_absolute_mode(monkeypatch):
    monkeypatch.setattr(scr.settings, "target_mode", "absolute")
    monkeypatch.setattr(
        scr,
        "load_feature_frame",
        lambda *a, **k: pd.DataFrame(
            {
                "symbol": ["AAA", "BBB"],
                "ts": pd.to_datetime(["2026-02-02"] * 2),
                "close": [100.0, 200.0],
                "rsi": [30.0, 70.0],
            }
        ),
    )

    latest = scr.load_latest_features("v4", ["AAA", "BBB"])

    assert list(latest["rsi"]) == [30.0, 70.0]


def test_the_screener_sizes_against_the_spread_of_what_it_predicts(monkeypatch):
    """
    forecast_scale must come from `target`, not `fwd_return`. In relative
    mode the two differ by roughly the market's own volatility, and the
    wrong one would mis-size every position in the book.
    """
    monkeypatch.setattr(scr.settings, "strategy_mode", "diversified")
    monkeypatch.setattr(scr.settings, "allow_shorts", True)
    monkeypatch.setattr(scr.settings, "full_deployment", False)
    monkeypatch.setattr(scr.settings, "screener_top_k", 5)

    train_df = pd.DataFrame(
        {
            "symbol": ["A"] * 4,
            "ts": pd.bdate_range("2026-01-01", periods=4, tz="UTC"),
            "close": [1.0, 1.1, 1.2, 1.3],
            # Deliberately different spreads, not just different levels: a
            # constant offset between the two would leave their standard
            # deviations equal and the assertion below would pass whichever
            # column the code read.
            "fwd_return": [0.02, 0.09, 0.14, 0.23],  # wide, drift-dominated
            "target": [-0.01, 0.0, 0.01, 0.02],  # narrow, market-relative
            "f1": [1, 2, 3, 4],
        }
    )
    monkeypatch.setattr(scr, "load_training_frame", lambda *a, **k: train_df)

    class _NoopEnsemble:
        def __init__(self, n_models=5): ...
        def fit(self, X, y): ...

    monkeypatch.setattr(scr, "EnsembleForecastModel", _NoopEnsemble)
    monkeypatch.setattr(scr, "load_latest_features", lambda *a, **k: pd.DataFrame({"symbol": ["A"], "f1": [3]}))
    monkeypatch.setattr(
        scr, "score_universe",
        lambda *a, **k: pd.DataFrame(
            {
                "symbol": ["A"],
                "predicted_return": [0.02],
                "direction_agreement": [1.0],
                "conviction_score": [0.02],
                "confident": [True],
            }
        ),
    )
    monkeypatch.setattr(scr, "build_correlation_matrix", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(scr, "_attach_reasoning", lambda *a, **k: None)
    monkeypatch.setattr(scr, "_load_fundamentals_context", lambda *a, **k: {})

    captured = {}

    def fake_select_trades(scored, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(scr, "select_trades", fake_select_trades)
    scr.run_screen("v4", ["A"])

    assert captured["forecast_scale"] == pytest.approx(float(train_df["target"].std()))
    assert captured["forecast_scale"] != pytest.approx(float(train_df["fwd_return"].std()))


def test_the_ensemble_is_fitted_on_target_not_on_the_raw_return(monkeypatch):
    """The single line that decides what the whole system is learning."""
    monkeypatch.setattr(scr.settings, "strategy_mode", "concentrated")
    monkeypatch.setattr(scr.settings, "allow_shorts", True)

    train_df = pd.DataFrame(
        {
            "symbol": ["A"] * 3,
            "ts": pd.bdate_range("2026-01-01", periods=3, tz="UTC"),
            "close": [1.0, 1.1, 1.2],
            "fwd_return": [0.10, 0.11, 0.12],
            "target": [-0.01, 0.0, 0.01],
            "f1": [1, 2, 3],
        }
    )
    monkeypatch.setattr(scr, "load_training_frame", lambda *a, **k: train_df)

    fitted = {}

    class _CapturingEnsemble:
        def __init__(self, n_models=5): ...
        def fit(self, X, y):
            fitted["y"] = list(y)

    monkeypatch.setattr(scr, "EnsembleForecastModel", _CapturingEnsemble)
    monkeypatch.setattr(scr, "load_latest_features", lambda *a, **k: pd.DataFrame({"symbol": ["A"], "f1": [3]}))
    monkeypatch.setattr(scr, "score_universe", lambda *a, **k: pd.DataFrame(
        columns=["symbol", "predicted_return", "direction_agreement", "conviction_score", "confident"]
    ))
    monkeypatch.setattr(scr, "_attach_reasoning", lambda *a, **k: None)
    monkeypatch.setattr(scr, "_load_fundamentals_context", lambda *a, **k: {})

    scr.run_screen("v4", ["A"])

    assert fitted["y"] == pytest.approx([-0.01, 0.0, 0.01])
