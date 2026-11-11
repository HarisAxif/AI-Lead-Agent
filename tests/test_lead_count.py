import pytest
import clarify
from schemas import SearchRequest, IntentResult, FieldValidation

MESSAGE = "Find 10 local bakeries in Faisalabad that need quantum computing consulting."


@pytest.mark.parametrize("message,count", [
    (MESSAGE, 10), ("I need 10 law firms in London for CRM software", 10),
    ("Find me 3 private schools in Pakistan", 3),
    ("3 restaurants for an AR menu", 3),
    ("Find schools with 500 students founded in 2024", None),
    ("Find 5 or 10 bakeries", None),
])
def test_explicit_count(message, count):
    assert clarify.explicit_lead_count(message) == count


def test_model_omission_does_not_trigger_count_question(monkeypatch):
    monkeypatch.setattr(clarify, "_structured_call", lambda *args: SearchRequest(
        location="Faisalabad", industry="local bakeries", service="quantum computing consulting"))
    request = clarify.understand_request(MESSAGE)
    assert request.number_of_leads == 10
    assert clarify.get_missing_slot(request) == (None, None)


def test_count_survives_followup_and_explicit_change(monkeypatch):
    monkeypatch.setattr(clarify, "_structured_call", lambda *args: SearchRequest())
    current = SearchRequest(number_of_leads=10)
    assert clarify.understand_request("Faisalabad", current).number_of_leads == 10
    assert clarify.understand_request("5", current).number_of_leads == 5
    assert clarify.understand_request("Find 7 bakeries", current).number_of_leads == 7


def test_exact_query_reaches_fit_guard_without_asking_again(monkeypatch, capsys):
    monkeypatch.setattr(clarify, "classify_intent", lambda *args: IntentResult(intent="lead_request"))
    monkeypatch.setattr(clarify, "_structured_call", lambda *args: SearchRequest(
        location="Faisalabad", industry="local bakeries", service="quantum computing consulting"))
    monkeypatch.setattr(clarify, "resolve_criteria", lambda request: request)
    monkeypatch.setattr("builtins.input", lambda *args: pytest.fail("Must not ask for an already supplied count"))
    monkeypatch.setattr(clarify, "validate_service_fit", lambda request:
                        FieldValidation(valid=False, reason="Unrelated service."))
    assert clarify.prepare_request(MESSAGE) is None
    assert "No lead research was started" in capsys.readouterr().out


def test_mismatch_is_checked_before_criteria_and_missing_count(monkeypatch, capsys):
    request = SearchRequest(industry="local bakeries", service="quantum computing consulting",
                            criteria=["because they need quantum computing consulting"])
    monkeypatch.setattr(clarify, "classify_intent", lambda *args: IntentResult(intent="lead_request"))
    monkeypatch.setattr(clarify, "understand_request", lambda *args: request)
    monkeypatch.setattr(clarify, "validate_service_fit", lambda r: FieldValidation(valid=False, reason="Mismatch"))
    monkeypatch.setattr(clarify, "resolve_criteria", lambda r: pytest.fail("Must reject mismatch before criteria clarification"))
    monkeypatch.setattr("builtins.input", lambda *args: pytest.fail("Must not ask unnecessary follow-up questions"))
    assert clarify.prepare_request(MESSAGE) is None
    assert "No lead research was started" in capsys.readouterr().out
