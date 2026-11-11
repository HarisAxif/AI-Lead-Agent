from evidence_gate import apply_audit, EvidenceAudit, CriterionCheck
from qualify import semantic_evidence_is_valid, verify_findings
from schemas import IndicatorFinding

CRITERION="websites have broken navigation or unreadable mobile layouts"

def test_slow_loading_does_not_replace_requested_navigation_fault():
    evidence=[{"indicator":"slow loading","evidence":"http_load_seconds: 7.54"}]
    audit=EvidenceAudit(accepted_evidence_indices=[0],criteria_checks=[CriterionCheck(criterion_index=0,supported=False,evidence_indices=[],reason="Different issue")])
    filtered,ok,reason=apply_audit(evidence,[CRITERION],audit)
    assert filtered==evidence and not ok
    assert CRITERION in reason

def test_direct_mobile_fault_satisfies_or_criterion():
    evidence=[{"indicator":"unreadable mobile layout","evidence":"Header text overlaps navigation"}]
    audit=EvidenceAudit(accepted_evidence_indices=[0],criteria_checks=[CriterionCheck(criterion_index=0,supported=True,evidence_indices=[0],reason="Mobile text overlaps")])
    assert apply_audit(evidence,[CRITERION],audit)[1]

def test_missing_criterion_check_is_not_accepted():
    audit=EvidenceAudit(accepted_evidence_indices=[0],criteria_checks=[])
    assert not apply_audit([{"evidence":"fault"}],[CRITERION],audit)[1]

def test_rejected_or_invented_evidence_cannot_support_criterion():
    audit=EvidenceAudit(accepted_evidence_indices=[],criteria_checks=[CriterionCheck(criterion_index=0,supported=True,evidence_indices=[0],reason="claimed")])
    assert not apply_audit([{"evidence":"fault"}],[CRITERION],audit)[1]
    audit.accepted_evidence_indices=[99]
    assert not apply_audit([{"evidence":"fault"}],[CRITERION],audit)[1]

def test_duplicate_criteria_cannot_cover_another_condition():
    check=CriterionCheck(criterion_index=0,supported=True,evidence_indices=[0],reason="fault")
    audit=EvidenceAudit(accepted_evidence_indices=[0],criteria_checks=[check,check])
    assert not apply_audit([{"evidence":"fault"}],[CRITERION,"Another condition"],audit)[1]

def test_public_contact_facts_do_not_prove_no_internal_crm():
    finding=IndicatorFinding(indicator="Business lacks internal CRM",present=True,evidence_type="technical_fact",footprint_fact="rendered_email_links: ['mailto:test@example.com']")
    assert not semantic_evidence_is_valid(finding)

def test_food_description_does_not_prove_professional_photography():
    finding=IndicatorFinding(indicator="Professional food photography",present=True,evidence_type="text_quote",quote="Delicious layers of pasta")
    assert not semantic_evidence_is_valid(finding)

def test_viewport_does_not_prove_site_wide_portal_absence():
    finding=IndicatorFinding(indicator="Website lacks a client portal",present=True,evidence_type="visual_observation",visual_observation="No login button is visible")
    assert verify_findings([finding],{"text":"","footprint":"","desktop_screenshot":"image"})==[]
