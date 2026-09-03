"""
XSS-escaping regression guard for monitoring/dashboard/static/app.js.

There is no JS test harness anywhere in this repo (no package.json, no
jest/mocha config) to actually execute newsFeedHTML/positionCardHTML and
assert on their output, so this instead pins the exact bug down at the
source-text level: every place that interpolates external-origin text
(news headlines/sources, symbols) into an innerHTML template literal must
route it through escapeHTML() first, matching how newsCardHTML (the News
tab's card renderer) already does it. A future edit that drops one of
these back to a raw `${n.headline}`-style interpolation fails this test
immediately, in CI, in Python — no browser or node runtime required.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

APP_JS = (Path(__file__).resolve().parent.parent / "monitoring" / "dashboard" / "static" / "app.js").read_text()


def _function_body(name: str) -> str:
    """
    Slices out one top-level `function <name>(...) { ... }` body by brace
    counting -- these functions are simple enough (no template-literal
    braces of their own beyond `${...}`, which this doesn't need to look
    inside) that a full JS parser would be overkill here.
    """
    match = re.search(rf"function {re.escape(name)}\([^)]*\)\s*\{{", APP_JS)
    assert match, f"could not find function {name}() in app.js -- has it been renamed?"
    start = match.end()
    depth = 1
    i = start
    while depth > 0:
        if APP_JS[i] == "{":
            depth += 1
        elif APP_JS[i] == "}":
            depth -= 1
        i += 1
    return APP_JS[start : i - 1]


@pytest.mark.parametrize("fn_name,fields", [
    ("newsFeedHTML", ["n.headline", 'n.source || ""']),
    ("newsCardHTML", ["item.headline", 's.symbol', 'item.source || ""']),
])
def test_news_rendering_escapes_every_external_origin_field(fn_name, fields):
    body = _function_body(fn_name)
    for field in fields:
        assert f"escapeHTML({field})" in body, (
            f"{fn_name}() interpolates {field} without escapeHTML() -- this is the "
            f"stored-XSS pattern (see the module docstring)."
        )


def test_news_feed_html_never_interpolates_headline_or_source_unescaped():
    """
    Belt-and-suspenders on top of the escapeHTML() presence check above:
    a raw `${n.headline}` or `${n.source` (without escapeHTML(...) wrapping
    it) must never reappear in newsFeedHTML(), even if someone adds a
    *second*, still-unescaped interpolation of the same field alongside an
    escaped one.
    """
    body = _function_body("newsFeedHTML")
    assert not re.search(r"\$\{n\.headline\}", body)
    assert not re.search(r"\$\{n\.source\b", body)


@pytest.mark.parametrize("fn_name,field", [
    ("positionCardHTML", "p.symbol"),
    ("loadClosedTrades", "t.symbol"),
])
def test_symbol_rendering_is_escaped(fn_name, field):
    body = _function_body(fn_name)
    assert f"escapeHTML({field})" in body, (
        f"{fn_name}() interpolates {field} without escapeHTML() — symbols get the "
        f"same treatment as headline/source elsewhere in this file (see newsCardHTML)."
    )
