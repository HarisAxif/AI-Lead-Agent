from gemini_llm import GEMINI_MODEL
"""
Finished runs are saved so a repeated request is answered instantly (no
Gemini call, no search, no browser).

Two keys are stored for every run:
  message key  - the exact text typed, checked BEFORE any Gemini call
  request key  - the understood request (location, industry, service,
                 criteria), checked after parsing, so different wording of the
                 same request also hits

Settings (.env or environment):
  RESULT_CACHE_HOURS=168   how long results stay fresh (0 turns the cache off)
  REFRESH=1                ignore saved results and research again

Saved leads never contain Google review text (see exporter.public_lead).
"""

import hashlib
import json
import os
import time
from pathlib import Path

from exporter import public_lead

CACHE_DIR = Path(".cache") / "results"

# main.show_results needs these on every lead. A saved lead from an older
# version that lacks one is ignored instead of crashing the display.
REQUIRED_LEAD_KEYS = (
    "company", "website", "tier", "reason", "opportunity",
    "visual_summary", "evidence",
)


def ttl_seconds() -> float:
    try:
        return float(os.getenv("RESULT_CACHE_HOURS", "168")) * 3600
    except ValueError:
        return 168 * 3600


def enabled() -> bool:
    return os.getenv("REFRESH") != "1" and ttl_seconds() > 0


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def message_key(message: str, fast: bool) -> str:
    normalized = " ".join((message or "").lower().split())
    return _digest(f"gemini-v6|{GEMINI_MODEL}|m|{'fast' if fast else 'full'}|{normalized}")


def request_key(plan_key: str, fast: bool) -> str:
    return _digest(f"gemini-v6|{GEMINI_MODEL}|r|{'fast' if fast else 'full'}|{plan_key}")


def save(keys: list, requested: int, final_state: dict, fast: bool) -> bool:
    """Returns True when something was saved. Empty runs are never cached."""
    qualified = [
        public_lead(lead)
        for lead in final_state.get("qualified_leads", []) or []
    ]

    if not qualified or ttl_seconds() <= 0:
        return False

    stats = final_state.get("research_stats", {})
    if len(qualified) < requested and (
            stats.get("api_failures") or stats.get("external_access_incomplete")
            or stats.get("unreachable") or stats.get("search_failures")
            or final_state.get("deadline") and time.time() >= final_state["deadline"]
            or final_state.get("discovery_stop_reason", "")):
        # A transient infrastructure failure must not become the answer for a week.
        return False

    payload = {
        "saved_at": time.time(),
        "fast": fast,
        "requested": requested,
        "qualified_leads": qualified,
        "rejected_leads": [
            {
                "website": item.get("website", ""),
                "rejection_type": item.get("rejection_type", ""),
                "rejection_reason": item.get("rejection_reason", ""),
            }
            for item in final_state.get("rejected_leads", []) or []
        ],
        "round": final_state.get("round", 0),
        "research_stats": final_state.get("research_stats", {}),
    }

    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        for key in keys:
            (CACHE_DIR / f"{key}.json").write_text(
                json.dumps(payload, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
    except Exception:
        return False

    return True


def load(key: str, requested: int | None = None):
    """
    Returns (state for show_results, age in seconds) or None.
    With `requested`, the saved run must have enough leads, or must have
    searched for at least that many (so a shortfall is not repeated blindly).
    """
    if not enabled():
        return None

    path = CACHE_DIR / f"{key}.json"

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    age = time.time() - payload.get("saved_at", 0)

    if age > ttl_seconds():
        return None

    leads = payload.get("qualified_leads", [])

    if any(
        key not in lead
        for lead in leads
        for key in REQUIRED_LEAD_KEYS
    ):
        return None

    if requested is not None:
        if (
            len(leads) < requested
            and payload.get("requested", 0) < requested
        ):
            return None

        leads = leads[:requested]

    state = {
        "qualified_leads": leads,
        "rejected_leads": payload.get("rejected_leads", []),
        "number_of_leads": requested or payload.get("requested", len(leads)),
        "round": payload.get("round", 0),
        "research_stats": payload.get("research_stats", {}),
    }

    return state, age


def describe_age(age: float) -> str:
    if age < 90:
        return "less than a minute ago"

    if age < 5400:
        return f"{round(age / 60)} minutes ago"

    if age < 129600:
        return f"{round(age / 3600)} hours ago"

    return f"{round(age / 86400)} days ago"
