from types import SimpleNamespace
import pytest
from pydantic import BaseModel
import gemini_llm as backend

class Answer(BaseModel):
    ok: bool

class ApiError(Exception):
    def __init__(self, code):
        self.code = code

@pytest.fixture
def calls(monkeypatch):
    calls = []
    monkeypatch.setattr(backend.limiter, "acquire", lambda: None)
    monkeypatch.setattr(backend, "handle_quota_error", lambda error: False)
    monkeypatch.setattr(backend.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(backend, "GEMINI_MODEL", "primary")
    monkeypatch.setattr(backend, "GEMINI_FALLBACK_MODEL", "second")
    monkeypatch.setattr(backend, "GEMINI_THIRD_MODEL", "third")
    def install(fn):
        def generate(**kwargs):
            calls.append(kwargs)
            return fn(len(calls))
        monkeypatch.setattr(backend, "ensure_ready", lambda: SimpleNamespace(models=SimpleNamespace(generate_content=generate)))
    return calls, install

def test_fallback_and_images(calls, capsys):
    seen, install = calls
    def respond(n):
        if n < 3:
            raise ApiError(503 if n == 1 else 504)
        return SimpleNamespace(parsed=Answer(ok=True))
    install(respond)
    assert backend.structured_call("prompt", Answer, images=["aGVsbG8="]).ok
    assert [c["model"] for c in seen] == ["primary", "second", "third"]
    assert seen[0]["contents"][1].inline_data.data == b"hello"
    output = capsys.readouterr().out
    assert "[Answer #" in output and "(503)" in output and "(504)" in output

def test_auth_failure_does_not_retry(calls):
    seen, install = calls
    def respond(n): raise ApiError(403)
    install(respond)
    with pytest.raises(backend.GeminiModelError, match="API key"):
        backend.structured_call("prompt", Answer)
    assert len(seen) == 1

def test_exhausted_retries_are_controlled(calls, monkeypatch):
    seen, install = calls
    monkeypatch.setattr(backend, "GEMINI_FALLBACK_MODEL", None)
    monkeypatch.setattr(backend, "GEMINI_THIRD_MODEL", None)
    def respond(n): raise ApiError(504)
    install(respond)
    with pytest.raises(backend.GeminiModelError):
        backend.structured_call("prompt", Answer)
    assert [c["model"] for c in seen] == ["primary"] * 3
