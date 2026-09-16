"""The daily ingest entrypoint's symbol handling and job wiring."""
from __future__ import annotations

import pytest

from scripts import run_daily_ingest as rdi
from scripts.run_daily_ingest import _REGIME_PROXY, with_regime_proxy


def test_regime_proxy_is_appended_to_the_ingest_list():
    assert with_regime_proxy(["AAPL", "MSFT"]) == ["AAPL", "MSFT", _REGIME_PROXY]


def test_regime_proxy_is_not_duplicated():
    assert with_regime_proxy(["SPY", "AAPL"]) == ["SPY", "AAPL"]


@pytest.fixture(autouse=True)
def _no_file_logging(monkeypatch):
    monkeypatch.setattr(rdi, "configure_file_logging", lambda: None)


@pytest.fixture
def _calls():
    return []


def _recorder(calls, name, return_value=None, raises: Exception | None = None):
    def _fn(*args, **kwargs):
        calls.append((name, args, kwargs))
        if raises is not None:
            raise raises
        return return_value

    return _fn


def _wire_happy_path(monkeypatch, calls, symbols=("AAPL", "MSFT")):
    monkeypatch.setattr(rdi, "resolve_symbols", lambda *a, **k: list(symbols))
    monkeypatch.setattr(rdi, "ingest_prices", _recorder(calls, "price_ingest"))
    monkeypatch.setattr(rdi, "ingest_fundamentals", _recorder(calls, "fundamentals_ingest"))
    monkeypatch.setattr(rdi, "refresh_macro_calendar", _recorder(calls, "macro_calendar_refresh"))
    monkeypatch.setattr(rdi, "build_and_store", _recorder(calls, "build_features"))
    monkeypatch.setattr(rdi, "alert_pipeline_failure", _recorder(calls, "alert"))


def _run_main(monkeypatch, argv_extra=()):
    monkeypatch.setattr("sys.argv", ["run_daily_ingest.py", "--universe", *argv_extra])
    rdi.main()


def test_daily_cycle_ingests_fundamentals_and_macro_calendar_every_run(monkeypatch, _calls):
    """Financial figures and event-risk countdowns must refresh daily, not just on the weekly cycle."""
    _wire_happy_path(monkeypatch, _calls)

    _run_main(monkeypatch)

    job_order = [name for name, _args, _kwargs in _calls]
    assert job_order == ["price_ingest", "fundamentals_ingest", "macro_calendar_refresh"]


def test_feature_set_id_flag_triggers_a_daily_feature_rebuild(monkeypatch, _calls):
    _wire_happy_path(monkeypatch, _calls, symbols=("AAPL", "MSFT"))

    _run_main(monkeypatch, argv_extra=["--feature-set-id", "v4"])

    job_order = [name for name, _args, _kwargs in _calls]
    assert job_order == ["price_ingest", "fundamentals_ingest", "macro_calendar_refresh", "build_features"]
    build_features_call = next(c for c in _calls if c[0] == "build_features")
    assert build_features_call[1] == (["AAPL", "MSFT"], "v4")


def test_without_feature_set_id_flag_features_are_not_rebuilt(monkeypatch, _calls):
    _wire_happy_path(monkeypatch, _calls)

    _run_main(monkeypatch)

    assert "build_features" not in [name for name, _args, _kwargs in _calls]


def test_a_failed_fundamentals_job_still_lets_macro_calendar_and_features_proceed(monkeypatch, _calls):
    _wire_happy_path(monkeypatch, _calls)
    monkeypatch.setattr(
        rdi, "ingest_fundamentals", _recorder(_calls, "fundamentals_ingest", raises=RuntimeError("rate limited"))
    )

    _run_main(monkeypatch, argv_extra=["--feature-set-id", "v4"])

    job_order = [name for name, _args, _kwargs in _calls]
    assert "macro_calendar_refresh" in job_order
    assert "build_features" in job_order
    alert_calls = [c for c in _calls if c[0] == "alert"]
    assert any(c[1][0] == "fundamentals_ingest" for c in alert_calls)
