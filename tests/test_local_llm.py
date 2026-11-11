import ast
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import local_llm
import clarify
import planner
import discovery
import qualify
from schemas import FieldValidation, SearchRequest, LeadAnalysis, EvidencePlan


HTTP_CLIENT = httpx.Client

def use_transport(monkeypatch, handler):
    original = HTTP_CLIENT
    monkeypatch.setattr(local_llm.httpx, "Client", lambda **kwargs: original(
        transport=httpx.MockTransport(handler), **kwargs))
    monkeypatch.setattr(local_llm.time, "sleep", lambda seconds: None)


def test_local_json_schema_and_image_payload(monkeypatch):
    monkeypatch.setattr(local_llm, "OLLAMA_MODEL", "gemma3:4b")
    captured = []
    def handle(request):
        captured.append(request)
        return httpx.Response(200, json={"message": {"content": '{"valid": true, "reason": "ok"}'}})
    use_transport(monkeypatch, handle)
    result = local_llm.structured_call("check", FieldValidation, images=["base64-image"])
    assert result.valid
    data = json.loads(captured[0].content)
    assert captured[0].url.host == "127.0.0.1"
    assert data["model"] == "gemma3:4b"
    assert data["messages"][0]["images"] == ["base64-image"]
    assert data["format"] == FieldValidation.model_json_schema()
    assert data["stream"] is False


def test_local_failure_never_switches_provider(monkeypatch):
    requests = []
    def handle(request):
        requests.append(request)
        raise httpx.ReadTimeout("local timeout", request=request)
    use_transport(monkeypatch, handle)
    with pytest.raises(local_llm.LocalModelError, match="after two attempts"):
        local_llm.structured_call("check", FieldValidation)
    assert len(requests) == 2
    assert all(r.url.host == "127.0.0.1" for r in requests)


def test_missing_model_and_text_only_model_fail_preflight(monkeypatch):
    use_transport(monkeypatch, lambda r: httpx.Response(404, json={"error": "missing"}))
    with pytest.raises(local_llm.LocalModelError, match="ollama pull"):
        local_llm.ensure_ready()
    use_transport(monkeypatch, lambda r: httpx.Response(200, json={"capabilities": ["completion"]}))
    with pytest.raises(local_llm.LocalModelError, match="cannot inspect screenshots"):
        local_llm.ensure_ready()


def test_cloud_endpoint_is_rejected(monkeypatch):
    monkeypatch.setattr(local_llm, "OLLAMA_BASE_URL", "https://ollama.com")
    with pytest.raises(local_llm.LocalModelError, match="local Ollama server"):
        local_llm.structured_call("check", FieldValidation)


def test_invalid_json_is_not_accepted(monkeypatch):
    use_transport(monkeypatch, lambda r: httpx.Response(200, json={"message": {"content": "not json"}}))
    with pytest.raises(local_llm.LocalModelError):
        local_llm.structured_call("check", FieldValidation)


def test_validation_and_planning_use_local_backend(monkeypatch):
    expected = FieldValidation(valid=False, reason="mismatch")
    monkeypatch.setattr(clarify, "structured_call", lambda p, s: expected)
    assert clarify.validate_service_fit(SearchRequest(industry="bakeries", service="quantum consulting")) is expected
    expected_plan = EvidencePlan(ideal_buyer="Bakeries")
    monkeypatch.setattr(planner, "structured_call", lambda p, s: expected_plan)
    assert planner._generate_research_plan(SearchRequest()) is expected_plan


def test_discovery_uses_local_backend(monkeypatch):
    expected = SimpleNamespace(result_type="directory")
    monkeypatch.setattr(discovery, "structured_call", lambda p, s: expected)
    assert discovery.classify_search_result({}, "schools", "Pakistan") == (expected, None)


def test_qualification_passes_screenshots_to_local_model(monkeypatch):
    captured = {}
    def call(prompt, schema, images=None):
        if schema is qualify.VisualReview:
            captured["visual_images"] = images
            return qualify.VisualReview(observations=["Visible layout"], summary="Visible layout")
        captured.update(schema=schema, images=images)
        return "analysis"
    monkeypatch.setattr(qualify, "structured_call", call)
    company = {"title": "School", "url": "https://example.org", "snippet": ""}
    site = {"text": "School", "footprint": "", "desktop_screenshot": "desktop", "mobile_screenshot": "mobile"}
    state = {"service": "parent portal", "industry": "schools", "location": "Pakistan",
             "plan": {"ideal_buyer": "school", "need_indicators": [], "disqualifiers": [], "visual_indicators": []}}
    assert qualify.judge_company(company, site, state) == "analysis"
    assert captured["visual_images"] == ["desktop", "mobile"]
    assert captured["images"] is None
    assert captured["schema"] is LeadAnalysis


def test_active_modules_do_not_import_gemini():
    root = Path(__file__).resolve().parent.parent
    for file in ["main.py", "clarify.py", "planner.py", "discovery.py", "qualify.py"]:
        source = (root / file).read_text(encoding="utf-8")
        ast.parse(source)
        assert "from google import genai" not in source
        assert "generate_content(" not in source
        assert "interactions.create(" not in source


def test_thinking_control_is_explicit_for_local_reasoning_model(monkeypatch):
    monkeypatch.setattr(local_llm, "OLLAMA_MODEL", "qwen3.5:9b-q4_K_M")
    monkeypatch.setattr(local_llm, "OLLAMA_THINK", "false")
    payloads = []
    def handle(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": '{"valid": true}'}})
    use_transport(monkeypatch, handle)
    local_llm.structured_call("check", FieldValidation)
    assert payloads[0]["think"] is False
    assert payloads[0]["model"] == "qwen3.5:9b-q4_K_M"
