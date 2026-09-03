"""
data/ingest/db.py's startup warning for DB_PASSWORD still holding its
literal docker-compose default on what looks like a non-local deployment.

Deliberately its own file, not tests/test_db.py: that file's
`_widgets_table` fixture is applied to every test via a module-level
pytestmark and needs a real, working DB connection built from the real
(default-in-this-environment) settings — mixing that with monkeypatching
settings.db_password/dashboard_host per test here would fight it.
"""
from __future__ import annotations

import logging

from data.ingest import db


def test_warns_when_default_password_and_non_loopback_host(monkeypatch, caplog):
    monkeypatch.setattr(db, "_engine", None)
    monkeypatch.setattr(db.settings, "db_password", db._DEFAULT_DB_PASSWORD)
    monkeypatch.setattr(db.settings, "dashboard_host", "0.0.0.0")

    with caplog.at_level(logging.WARNING):
        db.get_engine()

    assert any("DB_PASSWORD" in r.message for r in caplog.records)


def test_no_warning_when_password_has_been_changed(monkeypatch, caplog):
    monkeypatch.setattr(db, "_engine", None)
    monkeypatch.setattr(db.settings, "db_password", "a-real-generated-password")
    monkeypatch.setattr(db.settings, "dashboard_host", "0.0.0.0")

    with caplog.at_level(logging.WARNING):
        db.get_engine()

    assert not any("DB_PASSWORD" in r.message for r in caplog.records)


def test_no_warning_when_default_password_but_loopback_host(monkeypatch, caplog):
    monkeypatch.setattr(db, "_engine", None)
    monkeypatch.setattr(db.settings, "db_password", db._DEFAULT_DB_PASSWORD)
    monkeypatch.setattr(db.settings, "dashboard_host", "127.0.0.1")

    with caplog.at_level(logging.WARNING):
        db.get_engine()

    assert not any("DB_PASSWORD" in r.message for r in caplog.records)


def test_check_only_runs_once_per_cached_engine(monkeypatch, caplog):
    """The warning is a first-connection check, not a per-call one — once
    _engine is cached, get_engine() must not re-run (or re-log) it."""
    monkeypatch.setattr(db, "_engine", None)
    monkeypatch.setattr(db.settings, "db_password", db._DEFAULT_DB_PASSWORD)
    monkeypatch.setattr(db.settings, "dashboard_host", "0.0.0.0")

    with caplog.at_level(logging.WARNING):
        db.get_engine()
        db.get_engine()
        db.get_engine()

    assert sum("DB_PASSWORD" in r.message for r in caplog.records) == 1
