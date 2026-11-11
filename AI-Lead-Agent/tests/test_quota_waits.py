from types import SimpleNamespace

import pytest
from pydantic import BaseModel
import gemini_limits
import gemini_llm


class Answer(BaseModel):
    ok: bool


def test_capacity_wait_reports_progress_and_recovers(monkeypatch, capsys):
    now = [0.0]
    monkeypatch.setattr(gemini_limits.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(gemini_limits.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    monkeypatch.setenv("GEMINI_MAX_WAIT_SECONDS", "90")
    limiter = gemini_limits.RequestLimiter(1)
    limiter.acquire()
    limiter.acquire()
    assert now[0] == pytest.approx(60.0)
    assert "Waiting" in capsys.readouterr().out


def test_long_server_cooldown_is_rejected_without_sleeping(monkeypatch):
    monkeypatch.setattr(gemini_limits.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(gemini_limits.time, "sleep", lambda seconds: pytest.fail("Should not sleep"))
    monkeypatch.setenv("GEMINI_MAX_WAIT_SECONDS", "90")
    limiter = gemini_limits.RequestLimiter(15)
    limiter.defer(600)
    with pytest.raises(TimeoutError, match="90s limit"):
        limiter.acquire()


def test_continually_extended_cooldown_cannot_wait_forever(monkeypatch):
    now = [0.0]
    limiter = gemini_limits.RequestLimiter(15)
    monkeypatch.setenv("GEMINI_MAX_WAIT_SECONDS", "30")
    monkeypatch.setattr(gemini_limits.time, "monotonic", lambda: now[0])
    def sleep(seconds):
        now[0] += seconds
        limiter.defer(20)
    monkeypatch.setattr(gemini_limits.time, "sleep", sleep)
    limiter.defer(20)
    with pytest.raises(TimeoutError):
        limiter.acquire()
    assert now[0] <= 30


def test_wait_timeout_becomes_controlled_model_error(monkeypatch):
    monkeypatch.setattr(gemini_llm, "ensure_ready", lambda: object())
    def acquire():
        raise TimeoutError("quota wait exceeded")
    monkeypatch.setattr(gemini_llm.limiter, "acquire", acquire)
    with pytest.raises(gemini_llm.GeminiModelError, match="quota wait exceeded"):
        gemini_llm.structured_call("test", Answer)


def test_daily_quota_is_not_retried(monkeypatch):
    calls = []
    class DailyQuota(Exception):
        code = 429
    def generate(**kwargs):
        calls.append(kwargs)
        raise DailyQuota("RequestsPerDay quota exceeded")
    monkeypatch.setattr(gemini_llm, "ensure_ready", lambda: SimpleNamespace(
        models=SimpleNamespace(generate_content=generate)))
    monkeypatch.setattr(gemini_llm.limiter, "acquire", lambda: None)
    monkeypatch.setattr(gemini_llm.time, "sleep", lambda seconds: pytest.fail("No daily quota retries"))
    with pytest.raises(gemini_llm.GeminiModelError, match="quota"):
        gemini_llm.structured_call("test", Answer)
    assert len(calls) == 1
