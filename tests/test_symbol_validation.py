"""
data/ingest/db.py::validate_symbols / symbol_in_clause — the one sanctioned
way a symbol list enters a SQL string. Pure string logic, no DB needed.
"""
from __future__ import annotations

import pytest

from data.ingest.db import symbol_in_clause, validate_symbols


def test_real_tickers_pass():
    symbols = ["AAPL", "MSFT", "BRK.B", "BF-B", "MMM", "A", "GOOGL"]
    assert validate_symbols(symbols) == symbols


def test_in_clause_quotes_each_symbol():
    assert symbol_in_clause(["AAPL", "BRK.B"]) == "'AAPL', 'BRK.B'"


def test_empty_list_matches_nothing_instead_of_invalid_sql():
    assert symbol_in_clause([]) == "''"


@pytest.mark.parametrize(
    "bad",
    [
        "aapl",                          # lowercase — tickers are uppercase
        "1AAPL",                         # must start with a letter
        "AAPL'; DROP TABLE prices;--",   # the reason this module exists
        "AAPL OR 1=1",                   # whitespace can't pass
        "A" * 11,                        # longer than any real ticker
        "",                              # empty string is not a ticker
        "AAPL\n",                        # trailing newline
    ],
)
def test_non_tickers_are_rejected(bad):
    with pytest.raises(ValueError, match="don't look like tickers"):
        validate_symbols(["MSFT", bad])
    with pytest.raises(ValueError):
        symbol_in_clause([bad])


def test_error_message_counts_and_previews_offenders():
    with pytest.raises(ValueError, match="2 symbol"):
        validate_symbols(["ok not really", "MSFT", "also bad"])


def test_non_string_input_is_stringified_then_checked():
    with pytest.raises(ValueError):
        validate_symbols([None])
    assert validate_symbols([("AAPL")]) == ["AAPL"]
