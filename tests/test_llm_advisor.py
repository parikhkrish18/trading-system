import json

import pytest

from models import llm_advisor


class _FakeTextBlock:
    def __init__(self, text):
        self.text = text


class _FakeMessage:
    def __init__(self, text):
        self.content = [_FakeTextBlock(text)]


class _FakeMessages:
    def __init__(self, response_fn):
        self._response_fn = response_fn

    def create(self, model, max_tokens, system, messages):
        return _FakeMessage(self._response_fn(messages))


class _FakeAnthropic:
    def __init__(self, response_fn, api_key=None):
        self.messages = _FakeMessages(response_fn)


def _candidate(symbol="AAPL", **overrides):
    base = {
        "symbol": symbol,
        "side": "long",
        "predicted_return": 0.03,
        "conviction_score": 0.03,
        "macro_sector_sentiment": 0.1,
        "current_price": 200.0,
        "daily_volatility_pct": 0.015,
        "top_features": [{"feature_name": "mom_ret_20d", "value": 0.08, "contribution": 0.01}],
        "recent_headlines": [],
        "quant_take_profit_pct": 0.07,
        "quant_stop_loss_pct": 0.05,
    }
    base.update(overrides)
    return base


def _valid_response(symbols, picks=None):
    return json.dumps(
        {
            "candidates": [
                {
                    "symbol": s,
                    "confidence": 0.7,
                    "take_profit_pct": 0.08,
                    "stop_loss_pct": 0.04,
                    "signals_summary": "sig", "signals_lines": ["a"],
                    "forecast_summary": "fc", "forecast_lines": ["b"],
                    "selection_summary": "sel", "selection_lines": ["c"],
                }
                for s in symbols
            ],
            "picks": picks if picks is not None else symbols,
        }
    )


def test_returns_none_when_api_key_is_unset(monkeypatch):
    monkeypatch.setattr(llm_advisor.settings, "anthropic_api_key", "")
    called = []
    monkeypatch.setattr(llm_advisor, "Anthropic", lambda api_key: called.append(1) or _FakeAnthropic(lambda m: "{}"))

    result = llm_advisor.get_llm_trade_advice([_candidate()], {}, max_picks=2)

    assert result is None
    assert called == []  # never even constructs a client


def test_returns_none_for_an_empty_candidate_pool(monkeypatch):
    monkeypatch.setattr(llm_advisor.settings, "anthropic_api_key", "test-key")
    assert llm_advisor.get_llm_trade_advice([], {}, max_picks=2) is None


def test_happy_path_parses_confidence_reasoning_and_picks(monkeypatch):
    monkeypatch.setattr(llm_advisor.settings, "anthropic_api_key", "test-key")
    monkeypatch.setattr(llm_advisor, "Anthropic", lambda api_key: _FakeAnthropic(lambda m: _valid_response(["AAPL", "MSFT"])))

    result = llm_advisor.get_llm_trade_advice([_candidate("AAPL"), _candidate("MSFT")], {}, max_picks=2)

    assert result["picks"] == ["AAPL", "MSFT"]
    aapl = result["by_symbol"]["AAPL"]
    assert aapl["confidence"] == pytest.approx(0.7)
    assert aapl["take_profit_pct"] == pytest.approx(0.08)
    assert aapl["stop_loss_pct"] == pytest.approx(0.04)
    assert [p["phase"] for p in aapl["reasoning"]] == [2, 3, 4]
    assert aapl["reasoning"][0]["summary"] == "sig"
    assert aapl["reasoning"][0]["lines"] == ["a"]


def test_confidence_is_clamped_to_zero_one(monkeypatch):
    monkeypatch.setattr(llm_advisor.settings, "anthropic_api_key", "test-key")

    def respond(messages):
        resp = json.loads(_valid_response(["AAPL"]))
        resp["candidates"][0]["confidence"] = 5.0
        return json.dumps(resp)

    monkeypatch.setattr(llm_advisor, "Anthropic", lambda api_key: _FakeAnthropic(respond))

    result = llm_advisor.get_llm_trade_advice([_candidate("AAPL")], {}, max_picks=1)
    assert result["by_symbol"]["AAPL"]["confidence"] == 1.0


def test_a_hallucinated_symbol_not_in_the_pool_is_dropped(monkeypatch):
    """Claude can never advise on -- or pick -- a symbol outside the gated pool it was given."""
    monkeypatch.setattr(llm_advisor.settings, "anthropic_api_key", "test-key")
    monkeypatch.setattr(
        llm_advisor, "Anthropic",
        lambda api_key: _FakeAnthropic(lambda m: _valid_response(["AAPL", "NOTREAL"], picks=["AAPL", "NOTREAL"])),
    )

    result = llm_advisor.get_llm_trade_advice([_candidate("AAPL")], {}, max_picks=2)

    assert "NOTREAL" not in result["by_symbol"]
    assert result["picks"] == ["AAPL"]


def test_code_fenced_json_is_stripped_before_parsing(monkeypatch):
    monkeypatch.setattr(llm_advisor.settings, "anthropic_api_key", "test-key")
    monkeypatch.setattr(
        llm_advisor, "Anthropic",
        lambda api_key: _FakeAnthropic(lambda m: f"```json\n{_valid_response(['AAPL'])}\n```"),
    )

    result = llm_advisor.get_llm_trade_advice([_candidate("AAPL")], {}, max_picks=1)
    assert result["picks"] == ["AAPL"]


def test_malformed_json_retries_then_returns_none(monkeypatch):
    monkeypatch.setattr(llm_advisor.settings, "anthropic_api_key", "test-key")
    monkeypatch.setattr(llm_advisor, "_MAX_ATTEMPTS", 3)
    attempts = []

    def respond(messages):
        attempts.append(1)
        return "not json at all"

    monkeypatch.setattr(llm_advisor, "Anthropic", lambda api_key: _FakeAnthropic(respond))

    result = llm_advisor.get_llm_trade_advice([_candidate("AAPL")], {}, max_picks=1)

    assert result is None
    assert len(attempts) == 3


def test_empty_picks_is_treated_as_a_failure_worth_retrying(monkeypatch):
    monkeypatch.setattr(llm_advisor.settings, "anthropic_api_key", "test-key")
    monkeypatch.setattr(llm_advisor, "_MAX_ATTEMPTS", 2)
    attempts = []

    def respond(messages):
        attempts.append(1)
        return _valid_response(["AAPL"], picks=[])

    monkeypatch.setattr(llm_advisor, "Anthropic", lambda api_key: _FakeAnthropic(respond))

    result = llm_advisor.get_llm_trade_advice([_candidate("AAPL")], {}, max_picks=1)

    assert result is None
    assert len(attempts) == 2


def test_a_transient_api_error_returns_none_immediately_without_retrying(monkeypatch):
    """
    Rate-limit/network errors are already retried inside the Anthropic
    client itself -- anything that still raises here is a harder failure
    (e.g. auth) that re-asking the identical request won't fix.
    """
    monkeypatch.setattr(llm_advisor.settings, "anthropic_api_key", "test-key")
    attempts = []

    class _BoomMessages:
        def create(self, **kwargs):
            attempts.append(1)
            raise RuntimeError("boom")

    class _BoomAnthropic:
        def __init__(self, api_key=None):
            self.messages = _BoomMessages()

    monkeypatch.setattr(llm_advisor, "Anthropic", _BoomAnthropic)

    result = llm_advisor.get_llm_trade_advice([_candidate("AAPL")], {}, max_picks=1)

    assert result is None
    assert len(attempts) == 1
