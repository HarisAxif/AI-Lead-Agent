from types import SimpleNamespace

import pytest
import clarify
import gemini_llm as backend
from gemini_errors import _is_temporary_gemini_error
from schemas import IntentResult, SearchRequest


@pytest.mark.parametrize("message", [
    "504 DEADLINE_EXCEEDED. Deadline expired before operation could complete.",
    "DEADLINE_EXCEEDED", "deadline exceeded", "503 UNAVAILABLE",
])
def test_deadline_errors_are_temporary(message):
    assert _is_temporary_gemini_error(RuntimeError(message))


def test_sdk_status_code_is_recognized():
    error = RuntimeError("Operation expired")
    error.code = 504
    assert _is_temporary_gemini_error(error)
    assert not _is_temporary_gemini_error(RuntimeError("400 INVALID_ARGUMENT"))


def stub_calls(monkeypatch, fn):
    monkeypatch.setattr(backend, "ensure_ready", lambda: SimpleNamespace(
        models=SimpleNamespace(generate_content=fn)))
    monkeypatch.setattr(backend, "limiter", SimpleNamespace(acquire=lambda: None))
    monkeypatch.setattr(backend, "handle_quota_error", lambda error: False)


def test_timeout_retry_can_recover(monkeypatch, capsys):
    calls, sleeps = [], []
    expected = SearchRequest(location="Pakistan")
    def generate(**kwargs):
        calls.append(kwargs["model"])
        if len(calls) < 3:
            raise RuntimeError("504 DEADLINE_EXCEEDED")
        return SimpleNamespace(parsed=expected)
    stub_calls(monkeypatch, generate)
    monkeypatch.setattr(clarify.time, "sleep", sleeps.append)
    monkeypatch.setattr(backend, "GEMINI_FALLBACK_MODEL", "test-fallback")
    monkeypatch.setattr(backend, "GEMINI_THIRD_MODEL", None)
    assert clarify._structured_call("prompt", SearchRequest) is expected
    assert calls == [backend.GEMINI_MODEL, "test-fallback", "test-fallback"]
    assert sleeps == [2, 4]
    assert "attempt 3/3" in capsys.readouterr().out


def test_repeated_deadlines_stop_without_restarting_extraction(monkeypatch, capsys):
    calls = []
    def generate(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("504 DEADLINE_EXCEEDED")
    stub_calls(monkeypatch, generate)
    monkeypatch.setattr(clarify.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(clarify, "classify_intent", lambda *args:
                        IntentResult(intent="lead_request"))
    assert clarify.prepare_request("3 restaurants in Pakistan") is None
    assert len(calls) == 3
    output = capsys.readouterr().out
    assert "could not complete validation" in output
    assert "No lead research was started" in output


def test_bad_arguments_are_not_retried(monkeypatch):
    calls = []
    def generate(**kwargs):
        calls.append(kwargs)
        raise ValueError("400 INVALID_ARGUMENT")
    stub_calls(monkeypatch, generate)
    with pytest.raises(clarify.TemporaryValidationError):
        clarify._structured_call("prompt", SearchRequest)
    assert len(calls) == 1
