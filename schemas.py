from typing import TypedDict, Literal

from pydantic import BaseModel


class SearchRequest(BaseModel):
    location: str | None = None
    industry: str | None = None
    number_of_leads: int | None = None
    service: str | None = None


class FieldValidation(BaseModel):
    status: Literal[
        "VALID",
        "TOO_BROAD",
        "UNCLEAR",
        "LIKELY_TYPO"
    ]
    reason: str
    suggested_correction: str | None = None


class ServiceSuggestions(BaseModel):
    services: list[str]


class IndustrySuggestions(BaseModel):
    industries: list[str]


class EvidencePlan(BaseModel):
    ideal_buyer: str

    search_queries: list[str]

    need_indicators: list[str]

    disqualifiers: list[str]

    pages_to_check: list[str]

    evidence_observable: bool

    visual_review_needed: bool

    visual_indicators: list[str]


class IndicatorFinding(BaseModel):
    indicator: str

    present: bool

    evidence_type: Literal[
        "text_quote",
        "technical_fact",
        "visual_observation",
        "none"
    ]

    quote: str | None = None

    footprint_fact: str | None = None

    visual_observation: str | None = None


class LeadAnalysis(BaseModel):
    company_name: str

    is_single_business_site: bool

    office_locations: list[str]

    in_target_location: bool

    matches_ideal_buyer: bool

    hit_disqualifier: bool

    findings: list[IndicatorFinding]

    visual_summary: str

    reason: str

    opportunity: str


class AgentState(TypedDict):
    location: str
    industry: str
    number_of_leads: int
    service: str

    plan: dict

    companies: list

    qualified_leads: list
    rejected_leads: list

    seen_domains: list

    round: int