import contextlib

import pandas as pd
import pytest

from features.qualitative import sentiment_worker as sw


@contextlib.contextmanager
def _free_lock(*a, **k):
    yield True


@contextlib.contextmanager
def _busy_lock(*a, **k):
    yield False


@pytest.fixture(autouse=True)
def _lock_free_by_default(monkeypatch):
    monkeypatch.setattr(sw, "advisory_lock", _free_lock)


def _pending_df(count: int, oldest_age_s: float | None):
    return pd.DataFrame({"n": [count], "oldest_age_s": [oldest_age_s]})


def test_run_once_is_a_noop_when_nothing_pending(monkeypatch):
    monkeypatch.setattr(sw.pd, "read_sql", lambda *a, **k: _pending_df(0, None))
    calls = []
    monkeypatch.setattr(sw, "backfill_unscored_news", lambda *a, **k: calls.append(1))

    scored = sw.run_once(engine=object())

    assert scored == 0
    assert calls == []


def test_run_once_is_a_noop_for_a_fresh_partial_batch(monkeypatch):
    """Below _BATCH_SIZE and well within _MAX_WAIT_SECONDS -- nothing is overdue, so this must not spend a Claude call."""
    monkeypatch.setattr(sw.pd, "read_sql", lambda *a, **k: _pending_df(5, oldest_age_s=10.0))
    calls = []
    monkeypatch.setattr(sw, "backfill_unscored_news", lambda *a, **k: calls.append(1))

    scored = sw.run_once(engine=object())

    assert scored == 0
    assert calls == []


def test_run_once_scores_immediately_once_batch_size_is_reached(monkeypatch):
    monkeypatch.setattr(sw.pd, "read_sql", lambda *a, **k: _pending_df(sw._BATCH_SIZE, oldest_age_s=1.0))
    calls = []
    monkeypatch.setattr(sw, "backfill_unscored_news", lambda **k: (calls.append(k), 20)[1])

    scored = sw.run_once(engine=object())

    assert scored == 20
    assert calls == [{"batch_size": sw._BATCH_SIZE}]


def test_run_once_flushes_a_partial_batch_once_the_oldest_row_is_overdue(monkeypatch):
    """
    Regression target for the user's "a lone headline during a quiet
    stretch must not wait an hour" ask: once the oldest pending row has
    been waiting past _MAX_WAIT_SECONDS, flush whatever's pending even if
    it's well under a full batch.
    """
    monkeypatch.setattr(sw.pd, "read_sql", lambda *a, **k: _pending_df(3, oldest_age_s=sw._MAX_WAIT_SECONDS + 1))
    calls = []
    monkeypatch.setattr(sw, "backfill_unscored_news", lambda **k: (calls.append(k), 3)[1])

    scored = sw.run_once(engine=object())

    assert scored == 3
    assert calls == [{"batch_size": sw._BATCH_SIZE}]


def test_run_forever_does_not_sleep_right_after_a_full_batch(monkeypatch):
    """A full batch just went out -- more may be queued right behind it, so the next tick must not wait out poll_interval first."""
    run_once_results = iter([sw._BATCH_SIZE, 0])
    monkeypatch.setattr(sw, "run_once", lambda *a, **k: next(run_once_results))

    sleep_calls = []

    def _sleep(seconds):
        sleep_calls.append(seconds)
        raise KeyboardInterrupt  # stop the loop once we've observed the first real sleep

    monkeypatch.setattr(sw.time, "sleep", _sleep)

    sw.run_forever(poll_interval=99)

    # First tick scored a full batch and skipped the sleep; second tick scored
    # nothing and slept -- then the fake sleep raises to end the loop.
    assert sleep_calls == [99]


def test_run_forever_sleeps_after_a_partial_batch(monkeypatch):
    monkeypatch.setattr(sw, "run_once", lambda *a, **k: 3)

    sleep_calls = []

    def _sleep(seconds):
        sleep_calls.append(seconds)
        raise KeyboardInterrupt

    monkeypatch.setattr(sw.time, "sleep", _sleep)

    sw.run_forever(poll_interval=42)

    assert sleep_calls == [42]


def test_run_forever_skips_a_tick_when_the_lock_is_contended(monkeypatch):
    monkeypatch.setattr(sw, "advisory_lock", _busy_lock)
    calls = []
    monkeypatch.setattr(sw, "run_once", lambda *a, **k: calls.append(1))

    sleep_calls = []

    def _sleep(seconds):
        sleep_calls.append(seconds)
        raise KeyboardInterrupt

    monkeypatch.setattr(sw.time, "sleep", _sleep)

    sw.run_forever(poll_interval=5)

    assert calls == []
    assert sleep_calls == [5]


def test_run_forever_retries_after_a_tick_raises(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("db down")

    monkeypatch.setattr(sw, "run_once", _boom)

    sleep_calls = []

    def _sleep(seconds):
        sleep_calls.append(seconds)
        raise KeyboardInterrupt

    monkeypatch.setattr(sw.time, "sleep", _sleep)

    sw.run_forever(poll_interval=7)  # must not raise -- a bad tick is logged and retried, not fatal

    assert sleep_calls == [7]
