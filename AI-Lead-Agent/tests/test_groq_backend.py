import json
import httpx
import pytest
from pydantic import BaseModel
import groq_llm as backend
from schemas import SearchRequest
class Answer(BaseModel):
    ok: bool

@pytest.fixture
def mock(monkeypatch):
    seen=[]
    responses=[]
    monkeypatch.setattr(backend, "ensure_ready", lambda: {"Authorization":"Bearer test"})
    monkeypatch.setattr(backend.time, "sleep", lambda seconds: None)
    original=httpx.Client
    def handle(request):
        seen.append(json.loads(request.content))
        status,body=responses.pop(0)
        return httpx.Response(status,json=body)
    monkeypatch.setattr(backend.httpx,"Client",lambda **kw: original(transport=httpx.MockTransport(handle)))
    return seen,responses

def success(): return (200,{"choices":[{"finish_reason":"stop","message":{"content":"{\"ok\":true}"}}]})

def test_text_fallback(mock):
    seen,responses=mock
    responses.extend([(503,{}),success()])
    assert backend.structured_call("prompt",Answer).ok
    assert [x["model"] for x in seen]==[backend.GROQ_MODEL,backend.GROQ_FALLBACK_MODEL]

def test_vision_is_sent_to_vision_model(mock):
    seen,responses=mock
    responses.append(success())
    assert backend.structured_call("prompt",Answer,images=["aGVsbG8="]).ok
    assert seen[0]["model"]==backend.GROQ_VISION_MODEL
    assert seen[0]["messages"][0]["content"][1]["image_url"]["url"]=="data:image/png;base64,aGVsbG8="

def test_auth_error_is_not_retried(mock):
    seen,responses=mock
    responses.append((401,{}))
    with pytest.raises(backend.GroqModelError,match="API key"):
        backend.structured_call("prompt",Answer)
    assert len(seen)==1

def test_strict_schema_retains_nullable_fields():
    shape=backend.strict_schema(SearchRequest)
    assert shape["additionalProperties"] is False
    assert set(shape["required"])==set(shape["properties"])
    assert {"type":"null"} in shape["properties"]["location"]["anyOf"]

def test_daily_quota_does_not_sleep_for_hours(mock):
    seen,responses=mock
    responses.append((429,{}))
    # Default short quota retry is bounded to three attempts.
    responses.extend([(429,{}),(429,{})])
    with pytest.raises(backend.GroqModelError,match="three attempts"):
        backend.structured_call("prompt",Answer)
    assert len(seen)==3


def test_request_reservations_fit_small_vision_review(mock):
    seen,responses=mock
    responses.append(success())
    backend.structured_call("visual check",Answer,images=["aGVsbG8=","aGVsbG8="])
    assert seen[0]["max_completion_tokens"] == 768

def test_token_reset_headers_are_used(monkeypatch):
    monkeypatch.setattr(backend.time,"monotonic",lambda: 100.0)
    backend._remember_budget("test-model",{"x-ratelimit-remaining-tokens":"100", "x-ratelimit-reset-tokens":"1m2.5s"})
    assert backend._TOKEN_WINDOWS.pop("test-model") == (100,162.5)
