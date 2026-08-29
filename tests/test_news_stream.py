"""
data/ingest/news_stream.py's testable half: article_to_rows (pure) and
NewsStreamBuffer (injectable writer/clock, no websocket needed). run_stream
itself is a thin, hard-to-unit-test shell around NewsStreamBuffer and isn't
covered here — the lazy alpaca-py import and outer reconnect loop are the
kind of thing you'd want a live/paper key to exercise, not a unit test.
"""
from __future__ import annotations

import asyncio
import types

import pandas as pd
import pytest

from data.ingest.news_stream import (
    DEFAULT_FLUSH_INTERVAL_SECONDS,
    DEFAULT_FLUSH_MAX_BATCH,
    NewsStreamBuffer,
    _stream_stable_id,
    article_to_rows,
)


def _article(**overrides):
    defaults = dict(
        id="alpaca-article-1",
        headline="Some Company beats on earnings",
        created_at="2026-08-20T14:30:00Z",
        symbols=["AAPL", "MSFT"],
    )
    return types.SimpleNamespace(**{**defaults, **overrides})


class TestStreamStableId:
    def test_deterministic_and_fits_bigint(self):
        a = _stream_stable_id("art-1", "AAPL")
        b = _stream_stable_id("art-1", "AAPL")
        assert a == b
        assert 0 <= a < 2**63

    def test_differs_by_symbol_for_the_same_article(self):
        """Same fan-out safety as data/ingest/news.py's _stable_id: one story
        tagged to two symbols must not collide on (id, ts) in one upsert."""
        aapl_id = _stream_stable_id("shared-article", "AAPL")
        msft_id = _stream_stable_id("shared-article", "MSFT")
        assert aapl_id != msft_id

    def test_differs_from_a_different_article(self):
        a = _stream_stable_id("art-1", "AAPL")
        b = _stream_stable_id("art-2", "AAPL")
        assert a != b


class TestArticleToRows:
    def test_object_style_article_fans_out_one_row_per_symbol(self):
        rows = article_to_rows(_article())

        assert len(rows) == 2
        assert {r["symbol"] for r in rows} == {"AAPL", "MSFT"}
        assert all(r["source"] == "alpaca_stream" for r in rows)
        assert all(pd.isna(r["sentiment"]) and pd.isna(r["surprise"]) for r in rows)
        assert all(r["headline"] == "Some Company beats on earnings" for r in rows)

    def test_dict_style_article_is_also_accepted(self):
        """alpaca-py's stream can deliver raw_data=True dicts instead of News objects."""
        article = {
            "id": "alpaca-article-2",
            "headline": "Dict-shaped headline",
            "created_at": "2026-08-20T15:00:00Z",
            "symbols": ["TSLA"],
        }

        rows = article_to_rows(article)

        assert len(rows) == 1
        assert rows[0]["symbol"] == "TSLA"
        assert rows[0]["headline"] == "Dict-shaped headline"

    def test_naive_timestamp_is_localized_to_utc(self):
        rows = article_to_rows(_article(created_at="2026-08-20T14:30:00"))
        assert rows[0]["ts"].tzinfo is not None

    def test_aware_timestamp_is_converted_to_utc(self):
        rows = article_to_rows(_article(created_at="2026-08-20T10:30:00-04:00"))
        assert str(rows[0]["ts"].tz) == "UTC"
        assert rows[0]["ts"].hour == 14

    def test_same_article_and_symbol_produce_the_same_id_across_calls(self):
        """Idempotency: a redelivered message must upsert, not duplicate."""
        first = article_to_rows(_article())
        second = article_to_rows(_article())
        assert [r["id"] for r in first] == [r["id"] for r in second]

    @pytest.mark.parametrize(
        "overrides",
        [
            {"symbols": []},
            {"symbols": None},
            {"id": None},
            {"created_at": None},
        ],
    )
    def test_malformed_article_yields_no_rows(self, overrides):
        assert article_to_rows(_article(**overrides)) == []


class TestNewsStreamBuffer:
    def _buffer(self, **kwargs):
        writes: list[pd.DataFrame] = []

        def writer(df, table, conflict_cols):
            assert table == "news_events"
            assert conflict_cols == ["id", "ts"]
            writes.append(df)
            return len(df)

        clock = {"t": 0.0}
        buf = NewsStreamBuffer(writer=writer, clock=lambda: clock["t"], **kwargs)
        return buf, writes, clock

    def test_add_article_buffers_without_flushing_by_default(self):
        buf, writes, _ = self._buffer()

        added = buf.add_article(_article())

        assert added == 2
        assert writes == []
        assert len(buf._buffer) == 2

    def test_flush_writes_the_buffer_and_clears_it(self):
        buf, writes, _ = self._buffer()
        buf.add_article(_article())

        n = buf.flush()

        assert n == 2
        assert buf.total_ingested == 2
        assert len(writes) == 1
        assert list(writes[0]["symbol"]) == ["AAPL", "MSFT"]
        assert buf._buffer == []

    def test_flush_on_an_empty_buffer_is_a_noop(self):
        buf, writes, _ = self._buffer()

        assert buf.flush() == 0
        assert writes == []

    def test_malformed_article_is_dropped_not_buffered(self):
        buf, writes, _ = self._buffer()

        added = buf.add_article(_article(symbols=[]))

        assert added == 0
        assert buf._buffer == []

    def test_auto_flushes_once_max_batch_is_reached(self):
        buf, writes, _ = self._buffer(max_batch=2)

        buf.add_article(_article(id="a1", symbols=["AAPL"]))
        assert writes == []  # 1 row buffered, under the limit
        buf.add_article(_article(id="a2", symbols=["AAPL"]))  # 2nd row hits max_batch=2

        assert len(writes) == 1
        assert buf._buffer == []

    def test_auto_flushes_once_the_interval_elapses(self):
        buf, writes, clock = self._buffer(flush_interval=10.0, max_batch=1000)

        buf.add_article(_article(id="a1", symbols=["AAPL"]))
        assert writes == []

        clock["t"] = 11.0
        buf.add_article(_article(id="a2", symbols=["AAPL"]))

        assert len(writes) == 1  # the second add_article's own check trips the flush

    def test_handle_article_delegates_to_add_article(self):
        """handle_article is the coroutine handed to NewsDataStream.subscribe_news —
        run without a real event loop (no pytest-asyncio dependency needed)."""
        buf, writes, _ = self._buffer()

        asyncio.run(buf.handle_article(_article()))

        assert len(buf._buffer) == 2

    def test_defaults_match_module_constants(self):
        buf, _, _ = self._buffer()
        assert buf.flush_interval == DEFAULT_FLUSH_INTERVAL_SECONDS
        assert buf.max_batch == DEFAULT_FLUSH_MAX_BATCH
