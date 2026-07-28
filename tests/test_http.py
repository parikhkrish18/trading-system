import pytest

from data.ingest import http


class _FakeResponse:
    def __init__(self, status_code, headers=None):
        self.status_code = status_code
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return {"ok": True}


def test_polygon_get_returns_response_on_success(monkeypatch):
    monkeypatch.setattr(http.requests, "get", lambda *a, **k: _FakeResponse(200))
    resp = http.polygon_get("http://example.com", {})
    assert resp.status_code == 200


def test_polygon_get_retries_on_429_then_succeeds(monkeypatch):
    responses = [_FakeResponse(429), _FakeResponse(200)]
    calls = []

    def fake_get(*a, **k):
        calls.append(1)
        return responses.pop(0)

    monkeypatch.setattr(http.requests, "get", fake_get)
    monkeypatch.setattr(http.time, "sleep", lambda s: None)

    resp = http.polygon_get("http://example.com", {}, max_retries=3)

    assert resp.status_code == 200
    assert len(calls) == 2


def test_polygon_get_raises_after_max_retries_of_429(monkeypatch):
    monkeypatch.setattr(http.requests, "get", lambda *a, **k: _FakeResponse(429))
    monkeypatch.setattr(http.time, "sleep", lambda s: None)

    with pytest.raises(RuntimeError, match="rate limit exceeded"):
        http.polygon_get("http://example.com", {}, max_retries=2)


def test_polygon_get_respects_retry_after_header(monkeypatch):
    responses = [_FakeResponse(429, headers={"Retry-After": "7"}), _FakeResponse(200)]
    monkeypatch.setattr(http.requests, "get", lambda *a, **k: responses.pop(0))

    slept = []
    monkeypatch.setattr(http.time, "sleep", lambda s: slept.append(s))

    http.polygon_get("http://example.com", {}, max_retries=3)
    assert slept == [7.0]
