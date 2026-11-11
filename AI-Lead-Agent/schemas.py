from typing import Literal, TypedDict

from pydantic import BaseModel, Field


# =========================================================
# REQUEST / INPUT VALIDATION SCHEMAS
# =========================================================

class SearchRequest(BaseModel):
    location: str | None = None
    industry: str | None = None
    number_of_leads: int | None = None
    service: str | None = None
    criteria: list[str] = Field(default_factory=list)
    # ISO 3166-1 alpha-2 country of the target location (e.g. "AE", "US", "GB").
    # Used to validate phone numbers. None when unknown or multi-country.
    country_code: str | None = None


class IntentResult(BaseModel):
    intent: Literal[
        "lead_request",
        "slot_answer",
        "follow_up",
        "out_of_scope",
        "mixed_request",
        "prompt_injection",
        "unsafe",
    ]
    reason: str = ""
    lead_part: str | None = None
    unsupported_part: str | None = None


class CriterionAssessment(BaseModel):
    claim: str
    classification: Literal[
        "verifiable_public",
        "unverifiable",
        "implausible_for_industry",
        "mismatched_with_service",
        "self_defeating",
    ]
    reason: str
    suggested_replacement: str | None = None


class CriteriaValidation(BaseModel):
    assessments: list[CriterionAssessment] = Field(default_factory=list)


class ReplyAction(BaseModel):
    action: Literal[
        "accept_alternatives",
        "drop_criteria",
        "revised_criteria",
        "insist_original",
        "cancel",
        "unrelated",
    ]
    revised_text: str | None = None


class PlaceList(BaseModel):
    is_single_city: bool
    places: list[str] = Field(default_factory=list)


class FieldValidation(BaseModel):
    valid: bool
    reason: str = ""


class ServiceSuggestions(BaseModel):
    suggestions: list[str] = Field(default_factory=list)


class IndustrySuggestions(BaseModel):
    suggestions: list[str] = Field(default_factory=list)


# =========================================================
# DISCOVERY CLASSIFICATION SCHEMA
# =========================================================

class SearchResultClassification(BaseModel):
    result_type: Literal[
        "single_business",
        "directory",
        "aggregator",
        "marketplace",
        "article",
        "search_engine",
        "association",
        "other",
    ]
    matches_target_industry: bool
    confidence: float
    reason: str = ""


class IndexedSearchClassification(BaseModel):
    candidate_index: int
    classification: SearchResultClassification


class SearchClassificationBatch(BaseModel):
    items: list[IndexedSearchClassification] = Field(default_factory=list)


# =========================================================
# RESEARCH PLAN SCHEMA
# =========================================================

class EvidencePlan(BaseModel):
    adaptive_research: bool = True
    ideal_buyer: str
    search_queries: list[str] = Field(default_factory=list)
    need_indicators: list[str] = Field(default_factory=list)
    disqualifiers: list[str] = Field(default_factory=list)
    complaint_search_queries: list[str] = Field(default_factory=list)
    external_search_queries: list[str] = Field(default_factory=list)
    external_criteria: list[str] = Field(default_factory=list)
    pages_to_check: list[str] = Field(default_factory=list)
    evidence_observable: bool = True
    visual_review_needed: bool = False
    visual_indicators: list[str] = Field(default_factory=list)


# =========================================================
# QUALIFICATION SCHEMAS
# =========================================================

class IndicatorFinding(BaseModel):
    indicator: str
    present: bool
    evidence_type: Literal[
        "text_quote",
        "external_quote",
        "technical_fact",
        "visual_observation",
        "none",
    ] = "none"
    quote: str | None = None
    source_url: str | None = None
    footprint_fact: str | None = None
    visual_observation: str | None = None


class LeadAnalysis(BaseModel):
    company_name: str
    is_single_business_site: bool
    office_locations: list[str] = Field(default_factory=list)
    in_target_location: bool
    matches_ideal_buyer: bool
    hit_disqualifier: bool
    findings: list[IndicatorFinding] = Field(default_factory=list)
    visual_summary: str = ""
    reason: str
    opportunity: str


# =========================================================
# LANGGRAPH STATE
# =========================================================

class AgentState(TypedDict):
    location: str
    industry: str
    number_of_leads: int
    service: str
    plan: dict
    adaptive_queries: list
    adaptive_review_queries: list
    discovery_trace: list
    discovery_outcomes: list
    research_feedback: list
    discovery_stop_reason: str
    classification_cache: dict
    companies: list
    qualified_leads: list
    rejected_leads: list
    seen_domains: list
    seen_queries: list
    geo_terms: list
    failed_queries: list
    query_pool_exhausted: bool
    deadline: float
    research_stats: dict
    round: int
