import json

from execution import contradiction_advisor


class _FakeTextBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class _FakeThinkingBlock:
    """Sonnet 5 runs adaptive thinking by default -- no .text attribute at all."""

    type = "thinking"

    def __init__(self, thinking="reasoning about it..."):
        self.thinking = thinking


class _FakeMessage:
    def __init__(self, text, leading_blocks=()):
        self.content = [*leading_blocks, _FakeTextBlock(text)]


class _FakeMessages:
    def __init__(self, response_fn):
        self._response_fn = response_fn

    def create(self, model, max_tokens, system, messages):
        return _FakeMessage(self._response_fn(messages))


class _FakeAnthropic:
    def __init__(self, response_fn, api_key=None):
        self.messages = _FakeMessages(response_fn)


def _position(symbol="P", **overrides):
    base = {
        "symbol": symbol,
        "side": "short",
        "reasons": ["P's own sentiment contradicts short"],
        "pnl_pct": -0.02,
        "recent_headlines": [],
    }
    base.update(overrides)
    return base


def _valid_response(symbols, close=True, reasoning="Consistent with the rule-based signal."):
    return json.dumps({"positions": [{"symbol": s, "close": close, "reasoning": reasoning} for s in symbols]})


def test_returns_none_when_api_key_is_unset(monkeypatch):
    monkeypatch.setattr(contradiction_advisor.settings, "anthropic_api_key", "")
    called = []
    monkeypatch.setattr(
        contradiction_advisor, "Anthropic", lambda api_key: called.append(1) or _FakeAnthropic(lambda m: "{}")
    )

    result = contradiction_advisor.get_second_opinions([_position()])

    assert result is None
    assert called == []  # never even constructs a client


def test_returns_none_for_an_empty_position_list(monkeypatch):
    monkeypatch.setattr(contradiction_advisor.settings, "anthropic_api_key", "test-key")
    assert contradiction_advisor.get_second_opinions([]) is None


def test_happy_path_parses_close_and_reasoning(monkeypatch):
    monkeypatch.setattr(contradiction_advisor.settings, "anthropic_api_key", "test-key")
    monkeypatch.setattr(
        contradiction_advisor, "Anthropic",
        lambda api_key: _FakeAnthropic(lambda m: _valid_response(["P"], close=False, reasoning="Ride it out.")),
    )

    result = contradiction_advisor.get_second_opinions([_position("P")])

    assert result == {"P": {"close": False, "reasoning": "Ride it out."}}


def test_a_hallucinated_symbol_not_in_the_given_positions_is_dropped(monkeypatch):
    monkeypatch.setattr(contradiction_advisor.settings, "anthropic_api_key", "test-key")
    monkeypatch.setattr(
        contradiction_advisor, "Anthropic",
        lambda api_key: _FakeAnthropic(lambda m: _valid_response(["P", "NOTREAL"])),
    )

    result = contradiction_advisor.get_second_opinions([_position("P")])

    assert "NOTREAL" not in result
    assert "P" in result


def test_a_leading_thinking_block_does_not_break_response_parsing(monkeypatch):
    """Regression test: same ThinkingBlock-before-TextBlock fix as models/llm_advisor.py."""
    monkeypatch.setattr(contradiction_advisor.settings, "anthropic_api_key", "test-key")

    class _ThinkingFirstMessages:
        def create(self, model, max_tokens, system, messages):
            return _FakeMessage(_valid_response(["P"]), leading_blocks=[_FakeThinkingBlock()])

    class _ThinkingFirstAnthropic:
        def __init__(self, api_key=None):
            self.messages = _ThinkingFirstMessages()

    monkeypatch.setattr(contradiction_advisor, "Anthropic", _ThinkingFirstAnthropic)

    result = contradiction_advisor.get_second_opinions([_position("P")])
    assert result["P"]["close"] is True


def test_code_fenced_json_is_stripped_before_parsing(monkeypatch):
    monkeypatch.setattr(contradiction_advisor.settings, "anthropic_api_key", "test-key")
    monkeypatch.setattr(
        contradiction_advisor, "Anthropic",
        lambda api_key: _FakeAnthropic(lambda m: f"```json\n{_valid_response(['P'])}\n```"),
    )

    result = contradiction_advisor.get_second_opinions([_position("P")])
    assert "P" in result


def test_malformed_json_retries_then_returns_none(monkeypatch):
    monkeypatch.setattr(contradiction_advisor.settings, "anthropic_api_key", "test-key")
    monkeypatch.setattr(contradiction_advisor, "_MAX_ATTEMPTS", 3)
    attempts = []

    def respond(messages):
        attempts.append(1)
        return "not json at all"

    monkeypatch.setattr(contradiction_advisor, "Anthropic", lambda api_key: _FakeAnthropic(respond))

    result = contradiction_advisor.get_second_opinions([_position("P")])

    assert result is None
    assert len(attempts) == 3


def test_empty_positions_response_is_treated_as_a_failure_worth_retrying(monkeypatch):
    monkeypatch.setattr(contradiction_advisor.settings, "anthropic_api_key", "test-key")
    monkeypatch.setattr(contradiction_advisor, "_MAX_ATTEMPTS", 2)
    attempts = []

    def respond(messages):
        attempts.append(1)
        return json.dumps({"positions": []})

    monkeypatch.setattr(contradiction_advisor, "Anthropic", lambda api_key: _FakeAnthropic(respond))

    result = contradiction_advisor.get_second_opinions([_position("P")])

    assert result is None
    assert len(attempts) == 2


def test_a_transient_api_error_returns_none_immediately_without_retrying(monkeypatch):
    monkeypatch.setattr(contradiction_advisor.settings, "anthropic_api_key", "test-key")
    attempts = []

    class _BoomMessages:
        def create(self, **kwargs):
            attempts.append(1)
            raise RuntimeError("boom")

    class _BoomAnthropic:
        def __init__(self, api_key=None):
            self.messages = _BoomMessages()

    monkeypatch.setattr(contradiction_advisor, "Anthropic", _BoomAnthropic)

    result = contradiction_advisor.get_second_opinions([_position("P")])

    assert result is None
    assert len(attempts) == 1
