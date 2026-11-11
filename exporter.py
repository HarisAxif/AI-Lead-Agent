"""
Saves every run to disk so leads can go straight into a CRM and every
rejection can be audited.

runs/<timestamp>/leads.csv      qualified leads, stable columns
runs/<timestamp>/leads.json     same data, full detail
runs/<timestamp>/report.json    request, plan, counts, every rejection, timings
"""

import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

LEAD_COLUMNS = [
    "company",
    "website",
    "tier",
    "evidence_strength",
    "offices",
    "phones",
    "emails",
    "reason",
    "pitch",
    "evidence",
    # Google Maps: ids and links only. Google's terms do not allow storing
    # review text, ratings or other Places content (the place id is allowed).
    "google_place_id",
    "google_maps_url",
    "google_reviews_checked",
    "google_review_mentions",
    "google_review_links",
]


def _join(value) -> str:
    if isinstance(value, (list, tuple)):
        return "; ".join(str(item) for item in value)

    return "" if value is None else str(value)


def lead_row(lead: dict) -> dict:
    evidence = " | ".join(
        f"{item.get('indicator', '')}: {item.get('evidence', '')} [{item.get('source_url', '')}]"
        for item in lead.get("evidence", []) or []
    )

    reviews = lead.get("review_evidence") or {}

    return {
        "google_place_id": reviews.get("place_id", ""),
        "google_maps_url": reviews.get("place_maps_uri", ""),
        "google_reviews_checked": reviews.get("checked", ""),
        "google_review_mentions": len(reviews.get("mentions", []) or [])
        if reviews else "",
        "google_review_links": _join(
            [
                mention.get("maps_uri", "")
                for mention in reviews.get("mentions", []) or []
                if mention.get("maps_uri")
            ]
        ),
        "company": lead.get("company", ""),
        "website": lead.get("website", ""),
        "tier": lead.get("tier", ""),
        "evidence_strength": lead.get("evidence_strength", ""),
        "offices": _join(lead.get("offices")),
        "phones": _join(lead.get("phones")),
        "emails": _join(lead.get("emails")),
        "reason": lead.get("reason", ""),
        "pitch": lead.get("opportunity", ""),
        "evidence": evidence,
    }


def public_lead(lead: dict) -> dict:
    """Copy of a lead that is safe to save: no Google review content."""
    lead = dict(lead)
    lead["evidence"] = [
        {**item, "evidence": "Provider review text is not retained; see source URL."}
        if item.get("provider") == "trustpilot_api" else dict(item)
        for item in lead.get("evidence", [])
    ]
    reviews = lead.get("review_evidence")

    if reviews:
        lead["review_evidence"] = {
            "place_id": reviews.get("place_id", ""),
            "place_maps_uri": reviews.get("place_maps_uri", ""),
            "checked": reviews.get("checked", 0),
            "mentions": [
                {"maps_uri": mention.get("maps_uri", "")}
                for mention in reviews.get("mentions", []) or []
            ],
        }

    return lead


def export_run(
    request,
    plan: dict,
    final_state: dict,
    timings: dict,
    base_dir: str = "runs",
) -> Path:
    qualified = [
        public_lead(lead)
        for lead in final_state.get("qualified_leads", []) or []
    ]
    rejected = final_state.get("rejected_leads", []) or []

    run_dir = Path(base_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(
        run_dir / "leads.csv",
        "w",
        newline="",
        encoding="utf-8-sig",   # opens cleanly in Excel
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=LEAD_COLUMNS)
        writer.writeheader()

        for lead in qualified:
            writer.writerow(lead_row(lead))

    (run_dir / "leads.json").write_text(
        json.dumps(qualified, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    counts = Counter(
        item.get("rejection_type") or "unspecified"
        for item in rejected
    )

    report = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "request": {
            "location": request.location,
            "industry": request.industry,
            "service": request.service,
            "requested_leads": request.number_of_leads,
            "criteria": request.criteria,
            "country_code": request.country_code,
        },
        "plan": plan,
        "discovery_trace":final_state.get("discovery_trace", []),
        "discovery_outcomes":final_state.get("discovery_outcomes", []),
        "discovery_stop_reason":final_state.get("discovery_stop_reason", ""),
        "result": {
            "qualified": len(qualified),
            "rejected": len(rejected),
            "rejected_by_type": dict(counts),
            "search_stats": final_state.get("research_stats", {}),
            "queries_used": len(final_state.get("seen_queries", []) or []),
        },
        "rejections": [
            {
                "website": item.get("website", ""),
                "type": item.get("rejection_type", ""),
                "reason": item.get("rejection_reason", ""),
                "external_research": item.get("external_research", {}),
                "research_loop":item.get("research_loop", {}),
                "missing_facts":item.get("missing_facts", []),
                "criteria_audit":item.get("criteria_audit", []),
                "clause_audit":item.get("clause_audit", []),
                "website_access":item.get("website_access", {}),
            }
            for item in rejected
        ],
        "timings_seconds": timings,
    }

    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    return run_dir
