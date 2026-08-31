import pandas as pd

from scripts import backfill_headline_entities as backfill


class _FakeConn:
    def __init__(self, log):
        self._log = log

    def execute(self, stmt, params=None):
        self._log.append(params)
        return None


class _FakeBeginCtx:
    def __init__(self, log):
        self._log = log

    def __enter__(self):
        return _FakeConn(self._log)

    def __exit__(self, *exc):
        return False


class _FakeEngine:
    def __init__(self):
        self.executed: list = []

    def begin(self):
        return _FakeBeginCtx(self.executed)


_ROWS = pd.DataFrame(
    [
        {"id": 1, "ts": "2026-08-30T12:00:00Z", "headline": "Designates ChatGPT As &#39;Very Large Online Search Engine&#39;"},
        {"id": 2, "ts": "2026-08-30T13:00:00Z", "headline": "Tesla &amp; SolarCity announce merger"},
        # No entities at all — should never be touched.
        {"id": 3, "ts": "2026-08-30T14:00:00Z", "headline": "Plain headline, nothing to fix"},
    ]
)


def test_fixes_only_the_rows_that_actually_decode_differently(monkeypatch):
    monkeypatch.setattr(backfill.pd, "read_sql", lambda *a, **k: _ROWS.copy())
    engine = _FakeEngine()
    monkeypatch.setattr(backfill, "get_engine", lambda: engine)

    fixed_count = backfill.find_and_fix()

    assert fixed_count == 2
    # One batch, containing exactly the two rows whose headline changed.
    assert len(engine.executed) == 1
    batch = engine.executed[0]
    assert {row["id"] for row in batch} == {1, 2}
    fixed_by_id = {row["id"]: row["fixed"] for row in batch}
    assert fixed_by_id[1] == "Designates ChatGPT As 'Very Large Online Search Engine'"
    assert fixed_by_id[2] == "Tesla & SolarCity announce merger"


def test_dry_run_reports_without_writing(monkeypatch):
    monkeypatch.setattr(backfill.pd, "read_sql", lambda *a, **k: _ROWS.copy())
    engine = _FakeEngine()
    monkeypatch.setattr(backfill, "get_engine", lambda: engine)

    fixed_count = backfill.find_and_fix(dry_run=True)

    assert fixed_count == 2
    assert engine.executed == []  # nothing written


def test_no_matching_rows_is_a_clean_noop(monkeypatch):
    monkeypatch.setattr(backfill.pd, "read_sql", lambda *a, **k: pd.DataFrame(columns=["id", "ts", "headline"]))
    engine = _FakeEngine()
    monkeypatch.setattr(backfill, "get_engine", lambda: engine)

    assert backfill.find_and_fix() == 0
    assert engine.executed == []


def test_a_headline_with_entity_shaped_text_that_unescapes_to_itself_is_left_alone(monkeypatch):
    """A pre-filter false positive (matches the '&...;' shape but isn't a
    real entity) must not get written back — html.unescape() is the actual
    correctness check, the regex is only a cheap prefilter."""
    rows = pd.DataFrame([{"id": 9, "ts": "2026-08-30T15:00:00Z", "headline": "Q&A;session recap"}])
    monkeypatch.setattr(backfill.pd, "read_sql", lambda *a, **k: rows)
    engine = _FakeEngine()
    monkeypatch.setattr(backfill, "get_engine", lambda: engine)

    assert backfill.find_and_fix() == 0
    assert engine.executed == []
