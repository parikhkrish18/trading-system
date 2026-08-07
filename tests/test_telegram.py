"""
Transport-level tests for execution/telegram.py.

Nothing here touches the network: every test either monkeypatches
telegram.requests.post or injects a fake call_fn. There are no real tokens in
this file — the strings that look like credentials are obvious fakes.
"""
import pytest
import requests

from execution import telegram

FAKE_TOKEN = "123456:fake-test-token"  # not a real credential
FAKE_CHAT = "-100999"


class _FakeResponse:
    def __init__(self, status_code=200, body=None, headers=None, raises=False):
        self.status_code = status_code
        self._body = body if body is not None else {"ok": True, "result": {"message_id": 7}}
        self.headers = headers or {}
        self._raises = raises

    def json(self):
        if self._raises:
            raise ValueError("not json")
        return self._body


def _posts(responses, recorder=None):
    """A stand-in for requests.post that hands back queued responses."""

    def fake_post(url, json=None, timeout=None):
        if recorder is not None:
            recorder.append({"url": url, "json": json, "timeout": timeout})
        return responses.pop(0)

    return fake_post


# --- credentials ----------------------------------------------------------


def test_credentials_prefers_explicit_arguments_over_env(monkeypatch):
    monkeypatch.setattr(telegram.settings, "telegram_bot_token", "from-env")
    monkeypatch.setattr(telegram.settings, "telegram_chat_id", "111")

    assert telegram.credentials(FAKE_TOKEN, FAKE_CHAT) == (FAKE_TOKEN, FAKE_CHAT)


def test_credentials_falls_back_to_settings(monkeypatch):
    monkeypatch.setattr(telegram.settings, "telegram_bot_token", FAKE_TOKEN)
    monkeypatch.setattr(telegram.settings, "telegram_chat_id", FAKE_CHAT)

    assert telegram.credentials() == (FAKE_TOKEN, FAKE_CHAT)


def test_credentials_strips_whitespace_pasted_in_with_the_token(monkeypatch):
    """A trailing newline out of BotFather otherwise becomes a baffling 404."""
    monkeypatch.setattr(telegram.settings, "telegram_bot_token", f"  {FAKE_TOKEN}\n")
    monkeypatch.setattr(telegram.settings, "telegram_chat_id", " 555 ")

    assert telegram.credentials() == (FAKE_TOKEN, "555")


def test_credentials_reports_blank_when_nothing_is_configured(monkeypatch):
    monkeypatch.setattr(telegram.settings, "telegram_bot_token", "")
    monkeypatch.setattr(telegram.settings, "telegram_chat_id", "")

    assert telegram.credentials() == ("", "")


# --- the HTTP call --------------------------------------------------------


def test_call_posts_to_the_bot_api_and_returns_the_result(monkeypatch):
    sent = []
    monkeypatch.setattr(telegram.requests, "post", _posts([_FakeResponse()], sent))

    result = telegram.call("sendMessage", FAKE_TOKEN, {"chat_id": FAKE_CHAT, "text": "hi"})

    assert result == {"message_id": 7}
    assert sent[0]["url"] == f"https://api.telegram.org/bot{FAKE_TOKEN}/sendMessage"
    assert sent[0]["json"] == {"chat_id": FAKE_CHAT, "text": "hi"}
    assert sent[0]["timeout"] == telegram.DEFAULT_TIMEOUT


def test_call_raises_with_telegrams_own_description_when_ok_is_false(monkeypatch):
    monkeypatch.setattr(
        telegram.requests,
        "post",
        _posts([_FakeResponse(400, {"ok": False, "description": "Bad Request: chat not found"})]),
    )

    with pytest.raises(telegram.TelegramError, match="chat not found"):
        telegram.call("sendMessage", FAKE_TOKEN, {})


def test_call_never_leaks_the_token_into_the_error(monkeypatch):
    """The token lives in the URL path; an exception must not carry it out."""
    monkeypatch.setattr(
        telegram.requests,
        "post",
        _posts([_FakeResponse(401, {"ok": False, "description": "Unauthorized"})]),
    )

    with pytest.raises(telegram.TelegramError) as excinfo:
        telegram.call("getUpdates", FAKE_TOKEN, {})

    assert FAKE_TOKEN not in str(excinfo.value)


def test_call_retries_a_429_using_telegrams_retry_after(monkeypatch):
    responses = [
        _FakeResponse(429, {"ok": False, "parameters": {"retry_after": 4}}),
        _FakeResponse(),
    ]
    slept = []
    monkeypatch.setattr(telegram.requests, "post", _posts(responses))
    monkeypatch.setattr(telegram.time, "sleep", lambda s: slept.append(s))

    assert telegram.call("sendMessage", FAKE_TOKEN, {}) == {"message_id": 7}
    assert slept == [4.0]


def test_call_falls_back_to_the_retry_after_header(monkeypatch):
    responses = [_FakeResponse(429, {"ok": False}, headers={"Retry-After": "9"}), _FakeResponse()]
    slept = []
    monkeypatch.setattr(telegram.requests, "post", _posts(responses))
    monkeypatch.setattr(telegram.time, "sleep", lambda s: slept.append(s))

    telegram.call("sendMessage", FAKE_TOKEN, {})

    assert slept == [9.0]


def test_call_gives_up_after_max_retries_of_429(monkeypatch):
    monkeypatch.setattr(
        telegram.requests, "post", lambda *a, **k: _FakeResponse(429, {"ok": False})
    )
    monkeypatch.setattr(telegram.time, "sleep", lambda s: None)

    with pytest.raises(telegram.TelegramError, match="still rate-limited"):
        telegram.call("sendMessage", FAKE_TOKEN, {}, max_retries=2)


def test_call_turns_a_non_json_body_into_a_telegram_error(monkeypatch):
    monkeypatch.setattr(telegram.requests, "post", _posts([_FakeResponse(502, raises=True)]))

    with pytest.raises(telegram.TelegramError, match="non-JSON"):
        telegram.call("sendMessage", FAKE_TOKEN, {})


def test_call_wraps_a_network_failure_instead_of_leaking_requests_exceptions(monkeypatch):
    def boom(*args, **kwargs):
        raise requests.ConnectionError("no route to host")

    monkeypatch.setattr(telegram.requests, "post", boom)

    with pytest.raises(telegram.TelegramError, match="could not reach"):
        telegram.call("sendMessage", FAKE_TOKEN, {})


# --- send_message ---------------------------------------------------------


def test_send_message_builds_a_plain_text_payload():
    captured = {}

    def fake_call(method, token, payload=None):
        captured.update(method=method, token=token, payload=payload)
        return {"message_id": 1}

    telegram.send_message("hello phone", token=FAKE_TOKEN, chat_id=FAKE_CHAT, call_fn=fake_call)

    assert captured["method"] == "sendMessage"
    assert captured["token"] == FAKE_TOKEN
    assert captured["payload"]["chat_id"] == FAKE_CHAT
    assert captured["payload"]["text"] == "hello phone"
    # No parse_mode: the proposal text is full of %, +, - and | that markdown
    # would mangle or reject outright.
    assert "parse_mode" not in captured["payload"]


def test_send_message_refuses_without_a_token(monkeypatch):
    monkeypatch.setattr(telegram.settings, "telegram_bot_token", "")

    with pytest.raises(telegram.TelegramError, match="no bot token"):
        telegram.send_message("hi", chat_id=FAKE_CHAT, call_fn=lambda *a, **k: None)


def test_send_message_refuses_without_a_chat_id(monkeypatch):
    monkeypatch.setattr(telegram.settings, "telegram_chat_id", "")

    with pytest.raises(telegram.TelegramError, match="no chat id"):
        telegram.send_message("hi", token=FAKE_TOKEN, call_fn=lambda *a, **k: None)


def test_send_message_trims_a_body_over_the_bot_api_limit():
    captured = {}
    telegram.send_message(
        "x" * (telegram.MAX_MESSAGE_CHARS + 500),
        token=FAKE_TOKEN,
        chat_id=FAKE_CHAT,
        call_fn=lambda m, t, p=None: captured.update(payload=p),
    )

    text = captured["payload"]["text"]
    assert len(text) <= telegram.MAX_MESSAGE_CHARS
    assert text.endswith(telegram.TRUNCATION_NOTE)


def test_send_message_leaves_a_message_under_the_limit_alone():
    captured = {}
    telegram.send_message(
        "short enough",
        token=FAKE_TOKEN,
        chat_id=FAKE_CHAT,
        call_fn=lambda m, t, p=None: captured.update(payload=p),
    )

    assert captured["payload"]["text"] == "short enough"


# --- getUpdates / chat discovery -----------------------------------------


def _update(chat_id, text="hi", first_name="Neeraj", kind="private", key="message"):
    return {key: {"chat": {"id": chat_id, "first_name": first_name, "type": kind}, "text": text}}


def _recording_call(captured, result=None):
    def fake_call(method, token, payload=None, timeout=None):
        captured.update(method=method, payload=payload, http_timeout=timeout)
        return result if result is not None else [_update(42)]

    return fake_call


def test_fetch_updates_does_not_consume_or_long_poll_by_default():
    captured = {}

    telegram.fetch_updates(token=FAKE_TOKEN, call_fn=_recording_call(captured))

    assert captured["method"] == "getUpdates"
    # no offset -> nothing is marked delivered; timeout 0 -> returns immediately
    assert "offset" not in captured["payload"]
    assert captured["payload"]["timeout"] == 0


def test_fetch_updates_passes_an_offset_when_asked_to_consume():
    captured = {}

    telegram.fetch_updates(token=FAKE_TOKEN, offset=101, call_fn=_recording_call(captured))

    assert captured["payload"]["offset"] == 101


def test_fetch_updates_stretches_the_http_timeout_past_the_long_poll():
    """Otherwise the client hangs up mid-poll and every wait looks like an outage."""
    captured = {}

    telegram.fetch_updates(token=FAKE_TOKEN, poll_timeout=20, call_fn=_recording_call(captured))

    assert captured["payload"]["timeout"] == 20
    assert captured["http_timeout"] > 20


# --- offset arithmetic ----------------------------------------------------


def test_next_offset_is_one_past_the_highest_update_id():
    updates = [{"update_id": 100}, {"update_id": 102}, {"update_id": 101}]

    assert telegram.next_offset(updates) == 103


def test_next_offset_of_nothing_is_none():
    """No updates means no acknowledgement — never reset the cursor to 0."""
    assert telegram.next_offset([]) is None


def test_next_offset_ignores_malformed_entries():
    assert telegram.next_offset([{"no_id": 1}, {"update_id": 7}]) == 8


# --- reply filtering ------------------------------------------------------


def _reply_update(chat_id, text="approve 3", update_id=1):
    return {
        "update_id": update_id,
        "message": {"chat": {"id": chat_id, "first_name": "Neeraj", "type": "private"}, "text": text},
    }


def test_replies_from_returns_messages_from_the_configured_chat():
    replies = telegram.replies_from([_reply_update(424242, "approve 3", 9)], "424242")

    assert replies == [{"update_id": 9, "text": "approve 3", "name": "Neeraj"}]


def test_replies_from_drops_every_other_chat():
    """A bot username is public; "someone replied" is not "the owner replied"."""
    updates = [_reply_update(424242, "reject 3"), _reply_update(999999, "approve all")]

    replies = telegram.replies_from(updates, "424242")

    assert [r["text"] for r in replies] == ["reject 3"]


def test_replies_from_ignores_messages_with_no_text():
    """A sticker is not an approval."""
    updates = [{"update_id": 1, "message": {"chat": {"id": 424242, "type": "private"}, "sticker": {}}}]

    assert telegram.replies_from(updates, "424242") == []


def test_replies_from_matches_a_chat_id_given_as_int_or_string():
    assert len(telegram.replies_from([_reply_update(-100123)], " -100123 ")) == 1


def test_fetch_updates_refuses_without_a_token(monkeypatch):
    monkeypatch.setattr(telegram.settings, "telegram_bot_token", "")

    with pytest.raises(telegram.TelegramError, match="no bot token"):
        telegram.fetch_updates(call_fn=lambda *a, **k: [])


def test_fetch_updates_handles_an_empty_result():
    assert telegram.fetch_updates(token=FAKE_TOKEN, call_fn=lambda *a, **k: None) == []


def test_chat_candidates_extracts_id_name_and_text():
    chats = telegram.chat_candidates([_update(12345, text="hello bot")])

    assert chats == [
        {"chat_id": "12345", "name": "Neeraj", "kind": "private", "text": "hello bot"}
    ]


def test_chat_candidates_returns_newest_first_and_deduplicates():
    """Five messages from one chat is one answer, and the newest chat leads."""
    updates = [_update(111, text="old"), _update(111, text="older still"), _update(222, text="new")]

    chats = telegram.chat_candidates(updates)

    assert [c["chat_id"] for c in chats] == ["222", "111"]
    assert chats[1]["text"] == "older still"  # the most recent message from 111


def test_chat_candidates_reads_group_titles_and_channel_posts():
    updates = [
        {"channel_post": {"chat": {"id": -100123, "title": "Pulse alerts", "type": "channel"}, "text": "x"}}
    ]

    assert telegram.chat_candidates(updates) == [
        {"chat_id": "-100123", "name": "Pulse alerts", "kind": "channel", "text": "x"}
    ]


def test_chat_candidates_skips_updates_with_no_chat():
    """Non-message updates (poll answers, reactions) must not crash setup."""
    updates = [{"poll": {"id": "1"}}, {"message": {"no_chat": True}}, _update(7)]

    assert [c["chat_id"] for c in telegram.chat_candidates(updates)] == ["7"]


def test_chat_candidates_on_nothing_is_empty():
    assert telegram.chat_candidates([]) == []
