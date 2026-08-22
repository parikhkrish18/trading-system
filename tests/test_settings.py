"""
Config that only breaks in production is the expensive kind. These cover
the two settings that differ between a laptop and a hosted container: the
database URL built from platform-generated credentials, and the port the
platform tells us to bind.

Every Settings here is constructed with `_env_file=None` so a developer's
own .env can never decide whether the suite passes.
"""
from __future__ import annotations

from sqlalchemy.engine import make_url

from config.settings import Settings


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


# --------------------------------------------------------------------------
# db_url
# --------------------------------------------------------------------------


def test_db_url_round_trips_a_password_full_of_url_syntax():
    """
    A managed Postgres generates the password, and Railway's generator can
    emit '@', '/', ':' and '#'. Interpolated raw, the '@' would end the
    userinfo section early and the driver would go looking for a host that
    doesn't exist — a wrong-password bug that reports itself as a network
    error. The parsed-back password must equal the original, exactly.
    """
    password = "p@ss/w:rd#1?x&y"
    settings = _settings(DB_PASSWORD=password, DB_HOST="db.internal", DB_NAME="railway")

    url = make_url(settings.db_url)

    assert url.password == password
    assert url.host == "db.internal"
    assert url.database == "railway"


def test_db_url_round_trips_a_username_with_reserved_characters():
    url = make_url(_settings(DB_USER="user@tenant").db_url)

    assert url.username == "user@tenant"


def test_db_url_keeps_the_ordinary_case_readable():
    """Encoding must not mangle credentials that needed no encoding."""
    url = _settings(DB_USER="trading", DB_PASSWORD="simplepass", DB_HOST="localhost", DB_PORT=5432, DB_NAME="trading").db_url

    assert url == "postgresql+psycopg2://trading:simplepass@localhost:5432/trading"


# --------------------------------------------------------------------------
# Hosted bind
# --------------------------------------------------------------------------


def test_port_is_read_from_the_platform_injected_variable(monkeypatch):
    """Railway and friends choose the port, inject $PORT, and route to it."""
    monkeypatch.setenv("PORT", "8080")

    assert _settings().dashboard_port == 8080


def test_bind_defaults_are_loopback_8501(monkeypatch):
    for var in ("PORT", "DASHBOARD_HOST"):
        monkeypatch.delenv(var, raising=False)

    fresh = _settings()

    assert fresh.dashboard_host == "127.0.0.1"
    assert fresh.dashboard_port == 8501


# --------------------------------------------------------------------------
# Paper-only invariants that hosting must not weaken
# --------------------------------------------------------------------------


def test_trading_mode_defaults_to_paper_and_is_not_live(monkeypatch):
    monkeypatch.delenv("TRADING_MODE", raising=False)

    assert _settings().trading_mode == "paper"
    assert _settings().is_live is False


def test_approval_mode_defaults_to_the_human_gate(monkeypatch):
    """
    Hosting must not create an unattended trading path. 'auto' has to be an
    explicit, deliberate act — never something a blank env var falls into.
    """
    monkeypatch.delenv("APPROVAL_MODE", raising=False)

    assert _settings().approval_mode == "telegram"
