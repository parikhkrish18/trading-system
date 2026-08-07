import pandas as pd
import pytest

from monitoring.forecast_accuracy import BACKFILL_MODE, accuracy_by_mode, compute_forecast_accuracy
from scripts.backfill_decisions import (
    attach_target_ts,
    clear_backfill,
    picks_for_date,
    recent_mondays,
    snapshot_as_of,
    training_frame_as_of,
)

HORIZON = 5


def _feature_frame(symbols=("AAA", "BBB", "CCC"), n_days=80, start="2026-01-01"):
    """A dense daily feature frame shaped like models.train.load_feature_frame's output."""
    dates = pd.date_range(start, periods=n_days, freq="B", tz="UTC")
    rows = []
    for s_i, symbol in enumerate(symbols):
        for d_i, ts in enumerate(dates):
            rows.append(
                {
                    "symbol": symbol,
                    "ts": ts,
                    "f1": (d_i % 7) / 7 + s_i,
                    "adx_14": 20.0 + (d_i % 20),
                    "close": 100.0 + s_i * 10 + d_i,
                }
            )
    return pd.DataFrame(rows).sort_values(["symbol", "ts"]).reset_index(drop=True)


def _training_frame(feature_frame, horizon=HORIZON):
    """Reproduces load_training_frame's target, then attaches target_ts like the script does."""
    df = feature_frame.copy()
    df["fwd_return"] = df.groupby("symbol")["close"].transform(lambda s: s.shift(-horizon) / s - 1)
    df = df.dropna(subset=["fwd_return"])
    return attach_target_ts(feature_frame, df, horizon)


# --- Date selection ----------------------------------------------------------
def test_recent_mondays_returns_mondays_oldest_first_at_or_before_end():
    mondays = recent_mondays(pd.Timestamp("2026-07-30", tz="UTC"), weeks=4)

    assert len(mondays) == 4
    assert all(m.dayofweek == 0 for m in mondays)
    assert mondays == sorted(mondays)
    assert mondays[-1] == pd.Timestamp("2026-07-27", tz="UTC")
    assert all(m <= pd.Timestamp("2026-07-30", tz="UTC") for m in mondays)
    # Consecutive weeks, no gaps or repeats.
    assert all((b - a) == pd.Timedelta(1, unit="W") for a, b in zip(mondays, mondays[1:]))


def test_recent_mondays_on_a_monday_includes_that_monday():
    mondays = recent_mondays(pd.Timestamp("2026-07-27", tz="UTC"), weeks=2)
    assert mondays[-1] == pd.Timestamp("2026-07-27", tz="UTC")


def test_recent_mondays_accepts_naive_timestamps_as_utc():
    assert recent_mondays(pd.Timestamp("2026-07-30"), weeks=1)[0] == pd.Timestamp("2026-07-27", tz="UTC")


# --- No look-ahead: the core guarantee ---------------------------------------
def test_training_frame_max_ts_is_strictly_before_the_as_of_date():
    """The headline rule: a pick may only be trained on rows from before its own date."""
    train = _training_frame(_feature_frame())
    as_of = pd.Timestamp("2026-03-02", tz="UTC")

    sliced = training_frame_as_of(train, as_of)

    assert not sliced.empty
    assert sliced["ts"].max() < as_of
    assert (sliced["ts"] < as_of).all()


def test_training_frame_excludes_rows_whose_target_has_not_matured():
    """
    A row dated before as_of can still be look-ahead: its forward return is
    read from a price after as_of. Those rows must be dropped too.
    """
    train = _training_frame(_feature_frame())
    as_of = pd.Timestamp("2026-03-02", tz="UTC")

    sliced = training_frame_as_of(train, as_of)

    assert (sliced["target_ts"] < as_of).all()
    # And the filter is doing real work: rows exist that pass the ts test but
    # fail the target test, so this is not vacuously true.
    naive_cut = train.loc[train["ts"] < as_of]
    assert len(naive_cut) > len(sliced)


def test_attach_target_ts_points_at_the_date_the_target_is_read_from():
    features = _feature_frame(symbols=("AAA",))
    train = _training_frame(features)

    row = train.iloc[0]
    expected = features.loc[features["symbol"] == "AAA", "ts"].iloc[HORIZON]
    assert row["target_ts"] == expected
    # Every retained training row knows its target date.
    assert train["target_ts"].notna().all()
    assert (train["target_ts"] > train["ts"]).all()


def test_snapshot_includes_the_as_of_date_but_nothing_after_it():
    features = _feature_frame()
    as_of = pd.Timestamp("2026-03-02", tz="UTC")  # a Monday inside the range

    snapshot = snapshot_as_of(features, as_of)

    assert (snapshot["ts"] <= as_of).all()
    assert len(snapshot) == features["symbol"].nunique()
    # It is that day's row, not an older one, when the day has data.
    assert (snapshot["ts"] == as_of).all()


def test_snapshot_drops_symbols_whose_features_are_stale():
    features = _feature_frame(symbols=("AAA", "BBB"))
    features = features.loc[~((features["symbol"] == "BBB") & (features["ts"] > pd.Timestamp("2026-02-02", tz="UTC")))]

    snapshot = snapshot_as_of(features, pd.Timestamp("2026-03-02", tz="UTC"), max_staleness_days=10)

    assert set(snapshot["symbol"]) == {"AAA"}


# --- The real code path only ever sees the past ------------------------------
class _RecordingEnsemble:
    """Stands in for EnsembleForecastModel and records the rows it was trained on."""

    fitted_index = None

    def __init__(self, n_models=5):
        self.n_models = n_models

    def fit(self, X, y):
        type(self).fitted_index = X.index

    def predict(self, X):
        return pd.DataFrame(
            {
                "mean_prediction": [0.04] * len(X),
                "std_prediction": [0.001] * len(X),
                "direction_agreement": [1.0] * len(X),
            },
            index=X.index,
        )


def test_picks_for_date_trains_only_on_matured_history(monkeypatch):
    """End-to-end guard: what actually reaches .fit() is all strictly in the past."""
    monkeypatch.setattr("scripts.backfill_decisions.EnsembleForecastModel", _RecordingEnsemble)
    _RecordingEnsemble.fitted_index = None

    features = _feature_frame()
    train = _training_frame(features)
    as_of = pd.Timestamp("2026-03-02", tz="UTC")

    candidates = picks_for_date(train, features, as_of, top_k=3, min_train_rows=10)

    trained_on = train.loc[_RecordingEnsemble.fitted_index]
    assert not trained_on.empty
    assert trained_on["ts"].max() < as_of
    assert trained_on["target_ts"].max() < as_of
    assert candidates  # the fake ensemble is confident, so picks come out
    assert len(candidates) <= 3


def test_picks_for_date_skips_dates_without_enough_history(monkeypatch):
    monkeypatch.setattr("scripts.backfill_decisions.EnsembleForecastModel", _RecordingEnsemble)
    _RecordingEnsemble.fitted_index = None

    features = _feature_frame()
    train = _training_frame(features)

    # Second day of the sample: almost nothing has matured yet.
    assert picks_for_date(train, features, pd.Timestamp("2026-01-02", tz="UTC"), min_train_rows=250) == []
    assert _RecordingEnsemble.fitted_index is None  # never even trained


def test_picks_for_date_skips_when_no_snapshot_exists(monkeypatch):
    monkeypatch.setattr("scripts.backfill_decisions.EnsembleForecastModel", _RecordingEnsemble)
    features = _feature_frame()
    train = _training_frame(features)

    # Far past the sample, so every symbol's newest row is long stale.
    assert picks_for_date(train, features, pd.Timestamp("2027-01-04", tz="UTC"), min_train_rows=10) == []


# --- Idempotency and blast radius --------------------------------------------
class _FakeConn:
    def __init__(self, log):
        self.log = log

    def execute(self, statement, params=None):
        self.log.append((str(statement), params))

        class _Result:
            rowcount = 7

        return _Result()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeEngine:
    def __init__(self, log):
        self.log = log

    def begin(self):
        return _FakeConn(self.log)


def test_clear_backfill_deletes_only_backfill_rows(monkeypatch):
    """Re-running must replace its own rows and never touch paper or live ones."""
    log = []
    monkeypatch.setattr("scripts.backfill_decisions.get_engine", lambda: _FakeEngine(log))

    deleted = clear_backfill()

    assert deleted == 7
    statement, params = log[0]
    assert "DELETE FROM decisions" in statement
    assert "mode = :mode" in statement
    assert params == {"mode": BACKFILL_MODE}
    # No unscoped delete, and nothing keyed on anything but mode.
    assert "WHERE" in statement.upper()


# --- The accuracy panel can see, and label, backfilled rows ------------------
def test_compute_forecast_accuracy_carries_mode_through():
    decisions = pd.DataFrame(
        {
            "symbol": ["SPY", "SPY"],
            "ts": pd.to_datetime(["2026-01-02", "2026-01-03"], utc=True),
            "forecast": [0.5, -0.3],
            "mode": [BACKFILL_MODE, "paper"],
        }
    )
    prices = pd.DataFrame(
        {
            "symbol": ["SPY"] * 4,
            "ts": pd.to_datetime(["2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"], utc=True),
            "close": [500.0, 505.0, 500.0, 495.0],
        }
    )

    result = compute_forecast_accuracy(decisions, prices, horizon_bars=1)

    assert "mode" in result.columns
    assert set(result["mode"]) == {BACKFILL_MODE, "paper"}


def test_accuracy_by_mode_splits_and_labels_backfill():
    accuracy = pd.DataFrame(
        {
            "symbol": ["A"] * 5,
            "ts": pd.to_datetime(["2026-01-0{}".format(i) for i in range(1, 6)], utc=True),
            "forecast": [0.1] * 5,
            "realized_return": [0.1] * 5,
            "hit": [True, True, True, False, True],
            "mode": [BACKFILL_MODE, BACKFILL_MODE, BACKFILL_MODE, BACKFILL_MODE, "paper"],
        }
    )

    by_mode = accuracy_by_mode(accuracy)

    backfill = by_mode.loc[by_mode["mode"] == BACKFILL_MODE].iloc[0]
    assert backfill["n"] == 4
    assert backfill["hit_rate"] == pytest.approx(0.75)
    assert "Backfill" in backfill["label"]
    assert by_mode.loc[by_mode["mode"] == "paper"].iloc[0]["n"] == 1


def test_accuracy_by_mode_without_a_mode_column_is_empty_not_an_error():
    accuracy = pd.DataFrame({"symbol": ["A"], "ts": [pd.Timestamp("2026-01-01", tz="UTC")], "hit": [True]})
    assert accuracy_by_mode(accuracy).empty
