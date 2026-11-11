from types import SimpleNamespace

import clarify
import gemini_llm as backend
from schemas import SearchRequest, IntentResult, FieldValidation


def stub_request(monkeypatch, request):
    monkeypatch.setattr(clarify, "classify_intent", lambda *args: IntentResult(intent="lead_request"))
    monkeypatch.setattr(clarify, "understand_request", lambda *args: request)
    monkeypatch.setattr(clarify, "resolve_criteria", lambda r: r)
    monkeypatch.setattr(clarify, "clarify_request", lambda r: r)


def test_mismatch_with_no_explicit_criteria_stops_research(monkeypatch, capsys):
    request = SearchRequest(industry="local bakeries", location="Faisalabad",
                            number_of_leads=10, service="quantum computing consulting")
    stub_request(monkeypatch, request)
    checked = []
    def validate(r):
        checked.append(r)
        return FieldValidation(valid=False, reason="No credible specialized quantum use case was provided.")
    monkeypatch.setattr(clarify, "validate_service_fit", validate)
    assert clarify.prepare_request("Find 10 local bakeries in Faisalabad that need quantum computing consulting") is None
    assert checked == [request]
    assert "No lead research was started" in capsys.readouterr().out


def test_valid_fit_preserves_request(monkeypatch):
    request = SearchRequest(industry="private schools", location="Pakistan",
                            number_of_leads=3, service="parent portal")
    stub_request(monkeypatch, request)
    monkeypatch.setattr(clarify, "validate_service_fit", lambda r: FieldValidation(valid=True))
    assert clarify.prepare_request("Find schools for a parent portal") is request


def test_fit_check_has_context_and_does_not_invent_a_use_case(monkeypatch):
    captured = {}
    def call(prompt, schema):
        captured.update(prompt=prompt, schema=schema)
        return FieldValidation(valid=True)
    monkeypatch.setattr(clarify, "_structured_call", call)
    request = SearchRequest(industry="food-manufacturing research organizations",
                            service="quantum computing consulting",
                            criteria=["quantum optimization feasibility study"])
    assert clarify.validate_service_fit(request).valid
    assert captured["schema"] is FieldValidation
    assert request.criteria[0] in captured["prompt"]
    assert "Do not invent hypothetical" in captured["prompt"]


def test_fit_timeout_stops_preparation(monkeypatch, capsys):
    stub_request(monkeypatch, SearchRequest(industry="bakeries", service="consulting"))
    def timeout(request):
        raise clarify.TemporaryValidationError("timeout")
    monkeypatch.setattr(clarify, "validate_service_fit", timeout)
    assert clarify.prepare_request("Find bakeries") is None
    assert "No lead research was started" in capsys.readouterr().out


def sdk_deadline():
    from google.genai.errors import ServerError
    return ServerError(504, {"error": {"code": 504,
        "message": "Deadline expired before operation could complete.",
        "status": "DEADLINE_EXCEEDED"}})


def test_actual_sdk_deadline_fails_over_and_rejects_mismatch(monkeypatch, capsys):
    request = SearchRequest(industry="local bakeries", location="Faisalabad",
                            number_of_leads=10, service="quantum computing consulting")
    stub_request(monkeypatch, request)
    calls, delays = [], []
    def generate(**kwargs):
        calls.append(kwargs["model"])
        if len(calls) < 3:
            raise sdk_deadline()
        return SimpleNamespace(parsed=FieldValidation(valid=False, reason="No credible practical use case."))
    monkeypatch.setattr(backend, "ensure_ready", lambda: SimpleNamespace(models=SimpleNamespace(generate_content=generate)))
    monkeypatch.setattr(backend, "GEMINI_MODEL", "primary")
    monkeypatch.setattr(backend, "GEMINI_FALLBACK_MODEL", "second")
    monkeypatch.setattr(backend, "GEMINI_THIRD_MODEL", "third")
    monkeypatch.setattr(backend, "limiter", SimpleNamespace(acquire=lambda: None))
    monkeypatch.setattr(backend, "handle_quota_error", lambda error: False)
    monkeypatch.setattr(clarify.time, "sleep", delays.append)
    assert clarify.prepare_request("Find 10 local bakeries needing quantum computing consulting") is None
    assert calls == ["primary", "second", "third"]
    assert delays == [2, 4]
    output = capsys.readouterr().out
    assert "No lead research was started" in output
    assert "retrying with third" in output


def test_actual_sdk_deadline_exhaustion_exits_cleanly(monkeypatch, capsys):
    stub_request(monkeypatch, SearchRequest(industry="local bakeries", service="quantum consulting"))
    calls = []
    def generate(**kwargs):
        calls.append(kwargs)
        raise sdk_deadline()
    monkeypatch.setattr(backend, "ensure_ready", lambda: SimpleNamespace(models=SimpleNamespace(generate_content=generate)))
    monkeypatch.setattr(backend, "limiter", SimpleNamespace(acquire=lambda: None))
    monkeypatch.setattr(backend, "handle_quota_error", lambda error: False)
    monkeypatch.setattr(clarify.time, "sleep", lambda seconds: None)
    assert clarify.prepare_request("Find bakeries for quantum computing") is None
    assert len(calls) == 3
    assert "No lead research was started" in capsys.readouterr().out
