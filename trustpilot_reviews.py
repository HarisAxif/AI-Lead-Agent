"""Optional authorized Trustpilot Data Solutions review retrieval.
Reference: https://developers.trustpilot.com/data-solutions-api
Requires a key with Insights access; no scraping or access-control bypass.
"""
import os
import re
from threading import Lock
import httpx

BASE = "https://datasolutions.trustpilot.com/v1"
_lock = Lock()
_requests = 0
_disabled = None


def reset_run():
    global _requests, _disabled
    with _lock:
        _requests, _disabled = 0, None


def configured():
    return (os.getenv("ALLOW_PAID_PROVIDERS") == "1"
            and bool(os.getenv("TRUSTPILOT_API_KEY", "").strip()))


def _get(client, path, params=None):
    global _requests, _disabled
    with _lock:
        if _disabled:
            raise RuntimeError(_disabled)
        if _requests >= max(0, int(os.getenv("TRUSTPILOT_MAX_REQUESTS_PER_RUN", "20"))):
            raise RuntimeError("Trustpilot API request budget reached")
        _requests += 1
    response = client.get(BASE + path, params=params)
    if response.status_code in (401, 403, 429):
        with _lock:
            _disabled = (f"Trustpilot API HTTP {response.status_code}; check Data Solutions "
                         "Insights access and account quota")
        raise RuntimeError(_disabled)
    if response.status_code != 200:
        raise RuntimeError(f"Trustpilot API HTTP {response.status_code}")
    return response.json()


def collect_reviews(company):
    if not configured():
        return {"sources": [], "errors": []}
    domain = company.get("domain", "").lower().removeprefix("www.")
    sources = []
    try:
        with httpx.Client(timeout=8, follow_redirects=False,
                headers={"apikey": os.environ["TRUSTPILOT_API_KEY"].strip()}) as client:
            data = _get(client, "/business-units", {"domain": domain, "perPage": 5})
            unit = None
            for item in data.get("businessUnits", []):
                names = item.get("name", {})
                domains = [names.get("identifying", "")] + names.get("referring", [])
                if domain in [str(d).lower().removeprefix("www.") for d in domains]:
                    unit = item
                    break
            if not unit:
                return {"sources": [], "errors": [{"kind": "provider_identity",
                    "error": "Trustpilot API returned no exact matching business domain"}]}
            unit_id = unit.get("id", "")
            if not re.fullmatch(r"[a-fA-F0-9]{24}", unit_id):
                raise ValueError("Invalid business-unit identifier")
            token = None
            # Bounded sample: never claim exhaustive review coverage.
            for _ in range(2):
                data = _get(client, f"/business-units/{unit_id}/reviews",
                            {"nextToken": token} if token else None)
                for review in data.get("reviews", []):
                    body = review.get("text")
                    if not isinstance(body, str) or not body.strip():
                        continue
                    review_id = review.get("id", "")
                    if not re.fullmatch(r"[a-fA-F0-9]{24}", review_id):
                        continue
                    name = unit.get("name", {}).get("display", domain)
                    sources.append({"url": "https://www.trustpilot.com/reviews/" + review_id,
                        "title": str(review.get("title", "")),
                        "text": f"{name} ({domain}). {body}",
                        "query": "Trustpilot Data Solutions API", "provider": "trustpilot_api",
                        "created_at": review.get("createdAt", "")})
                    if len(sources) >= 40:
                        break
                token = data.get("nextToken")
                if not token or len(sources) >= 40:
                    break
    except Exception as error:
        # Never include API response bodies or request headers in diagnostics.
        message = str(error) if isinstance(error, RuntimeError) else "Trustpilot API transport or response error"
        return {"sources": sources, "errors": [{"kind": "provider_failed", "error": message}]}
    return {"sources": sources, "errors": []}
