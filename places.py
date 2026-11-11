"""
Google Places API (New) source for the lead agent.

What it adds
------------
1. Discovery: Text Search returns real businesses with their official website,
   so no directories, listicles or social pages.
2. Evidence: up to 5 Google Maps reviews per business, scanned for
   complaints about slow or missing replies.

Billing
-------
Asking for `places.reviews` puts the request in the "Text Search Enterprise +
Atmosphere" SKU. Billing is per REQUEST (not per result); one request returns
up to 20 businesses. Check Google's current price list before large runs. A
per-run request cap (PLACES_MAX_REQUESTS_PER_RUN, default 15) protects you.

Google's policies (read them)
-----------------------------
- Places content must not be cached or stored (the place_id is the exception).
  This module only keeps results in memory during a run; exporter.py writes
  place ids and links, never review text.
- Reviews shown to users need author attribution and a link to the review on
  Google Maps (googleMapsUri). The console output in main.py does this.
- Google returns at most 5 reviews per place, chosen by Google. "No complaint
  found" therefore means "none among the reviews Google returned".
"""

import os
import re
from urllib.parse import urlparse

import httpx


ENDPOINT = "https://places.googleapis.com/v1/places:searchText"

FIELD_MASK = ",".join([
    "places.id",
    "places.displayName",
    "places.googleMapsUri",
    "places.businessStatus",
    "places.websiteUri",
    "places.rating",
    "places.userRatingCount",
    "places.reviews",          # triggers the Enterprise + Atmosphere SKU
])

TIMEOUT_SECONDS = 20.0
PAGE_SIZE = 20
QUOTE_MAX_WORDS = 25

CLOSED_STATUSES = {"CLOSED_PERMANENTLY", "CLOSED_TEMPORARILY"}


class PlacesError(Exception):
    """Any problem talking to Places. Callers fall back to web search."""


# ---------------------------------------------------------------- run state

_requests_made = 0
_disabled_reason = None


def reset_run():
    global _requests_made, _disabled_reason
    _requests_made = 0
    _disabled_reason = None


def requests_made() -> int:
    return _requests_made


def max_requests() -> int:
    try:
        return max(int(os.getenv("PLACES_MAX_REQUESTS_PER_RUN", "15")), 0)
    except ValueError:
        return 15


def api_key() -> str:
    return os.getenv("GOOGLE_PLACES_API_KEY", "").strip()


def is_configured() -> bool:
    """A key is present and Places has not been switched off in .env."""
    return (os.getenv("ALLOW_PAID_PROVIDERS") == "1"
            and bool(api_key()) and os.getenv("PLACES_DISABLED") != "1")


def can_search() -> bool:
    return (
        is_configured()
        and _disabled_reason is None
        and _requests_made < max_requests()
    )


def disabled_reason():
    return _disabled_reason


# ------------------------------------------------------------- API request

def _error_message(response) -> str:
    try:
        return response.json()["error"]["message"]
    except Exception:
        return response.text[:200]


def search_text(
    query: str,
    region_code: str | None = None,
    client: httpx.Client | None = None,
) -> list:
    """
    One Text Search request. Returns normalized places.
    Raises PlacesError on any failure (and disables Places for the rest of the
    run on key/permission/billing errors).
    """
    global _requests_made, _disabled_reason

    if not can_search():
        raise PlacesError("Places is not available for this run.")

    body = {
        "textQuery": query,
        "pageSize": PAGE_SIZE,
        "languageCode": "en",
    }

    if region_code:
        body["regionCode"] = region_code.upper()

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key(),
        "X-Goog-FieldMask": FIELD_MASK,
    }

    owns_client = client is None
    client = client or httpx.Client(timeout=TIMEOUT_SECONDS)

    _requests_made += 1  # billed per request, even when it returns nothing

    try:
        response = client.post(ENDPOINT, json=body, headers=headers)
    except httpx.HTTPError as error:
        raise PlacesError(
            f"network error ({type(error).__name__})"
        ) from None
    finally:
        if owns_client:
            client.close()

    if response.status_code in (401, 403):
        _disabled_reason = (
            f"HTTP {response.status_code}: {_error_message(response)}"
        )
        raise PlacesError(_disabled_reason)

    if response.status_code == 429:
        raise PlacesError("rate limited (HTTP 429)")

    if response.status_code != 200:
        raise PlacesError(
            f"HTTP {response.status_code}: {_error_message(response)}"
        )

    try:
        data = response.json()
    except ValueError:
        raise PlacesError("invalid JSON in response") from None

    return [
        normalize_place(raw)
        for raw in data.get("places", []) or []
    ]


# ------------------------------------------------------------ normalization

def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:
        return ""


def normalize_review(raw: dict) -> dict:
    text = (
        (raw.get("text") or {}).get("text")
        or (raw.get("originalText") or {}).get("text")
        or ""
    )

    author = raw.get("authorAttribution") or {}

    return {
        "text": text,
        "rating": raw.get("rating"),
        "relative_time": raw.get("relativePublishTimeDescription", ""),
        "author": author.get("displayName", ""),
        "author_uri": author.get("uri", ""),
        "maps_uri": raw.get("googleMapsUri", ""),
    }


def normalize_place(raw: dict) -> dict:
    place_id = raw.get("id") or (raw.get("name") or "").removeprefix(
        "places/"
    )

    return {
        "place_id": place_id,
        "name": (raw.get("displayName") or {}).get("text", ""),
        "website": raw.get("websiteUri", ""),
        "maps_uri": raw.get("googleMapsUri", ""),
        "status": raw.get("businessStatus", ""),
        "rating": raw.get("rating"),
        "rating_count": raw.get("userRatingCount"),
        "reviews": [
            normalize_review(review)
            for review in raw.get("reviews", []) or []
        ],
    }


# ------------------------------------------------- slow-reply review analysis

# Sentences that clearly say the business is slow or does not answer.
_NEGATORS = (
    r"(?:no|never|didn'?t|did\s+not|don'?t|do\s+not|doesn'?t|does\s+not|"
    r"won'?t|wouldn'?t|couldn'?t|failed\s+to|without|hardly|barely)"
)

_REPLY_WORDS = (
    r"(?:repl(?:y|ied|ies|ying)|respon(?:d|ded|ding|se|ses)|"
    r"answer(?:ed|ing|s)?|call(?:ed|ing|s)?\s+(?:me\s+|us\s+)?back|"
    r"get(?:ting)?\s+back|got\s+back|"
    r"return(?:ed|ing)?\s+(?:my\s+|our\s+)?(?:calls?|messages?|e-?mails?)|"
    r"pick(?:ed|ing)?\s+up|follow(?:ed|ing)?[\s-]?up)"
)

SLOW_REPLY_PATTERNS = [
    re.compile(
        rf"\b{_NEGATORS}\b[^.!?]{{0,30}}\b{_REPLY_WORDS}\b",
        re.I,
    ),
    re.compile(
        r"\bnot\s+(?:respond\w*|repl\w*|answer\w*|picking\s+up|"
        r"return\w*|getting\s+back|calling\s+back)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:slow|poor|terrible|awful|horrible|bad|lack\s+of|lacking)\b"
        r"[^.!?]{0,25}\b(?:respon\w*|repl\w*|communicat\w*|follow[\s-]?up|"
        r"turnaround)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:unrespons\w+|non[\s-]?respons\w+|unanswered|unreachable)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:took|take|takes|taking)\b[^.!?]{0,30}\b(?:days?|weeks?|ages|"
        r"forever|too\s+long)\b[^.!?]{0,30}\b"
        r"(?:repl\w*|respon\w*|answer\w*|get\s+back|call\s+back)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:waited|waiting|wait)\b[^.!?]{0,40}\b(?:days?|weeks?|hours|"
        r"ages|forever)\b[^.!?]{0,40}\b"
        r"(?:repl\w*|respon\w*|answer\w*|call\w*\s+back)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:ignor\w+|ghost\w*)\b[^.!?]{0,30}\b(?:me|us|my|our|messages?|"
        r"calls?|e-?mails?|inquir\w+|enquir\w+|requests?)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:still|yet)\b[^.!?]{0,20}\b(?:waiting|no\s+(?:reply|response|"
        r"answer))\b",
        re.I,
    ),
]

# A sentence that praises speed is not a complaint ("never had to chase,
# they replied quickly") unless it also contains a hard negative.
_PRAISE = re.compile(
    r"\b(?:quick(?:ly)?|fast|prompt(?:ly)?|immediate(?:ly)?|responsive|"
    r"right\s+away|same\s+day|within\s+(?:minutes|an\s+hour|\d+\s+hours?)|"
    r"hesitat\w*|always\s+(?:replied|responded|answered))\b",
    re.I,
)

_HARD_NEGATIVE = re.compile(
    r"\b(?:unrespons\w+|slow|unanswered|ignor\w+|ghost\w*|never\s+"
    r"(?:replied|responded|answered))\b",
    re.I,
)

_SENTENCES = re.compile(r"(?<=[.!?])\s+|\n+")


def _trim_quote(sentence: str) -> str:
    words = sentence.split()

    if len(words) <= QUOTE_MAX_WORDS:
        return sentence.strip()

    return " ".join(words[:QUOTE_MAX_WORDS]) + chr(0x2026)


def find_slow_reply_sentence(text: str):
    """Returns the first complaint sentence in a review, or None."""
    for sentence in _SENTENCES.split(text or ""):
        sentence = sentence.strip()

        if not sentence:
            continue

        if not any(
            pattern.search(sentence)
            for pattern in SLOW_REPLY_PATTERNS
        ):
            continue

        if _PRAISE.search(sentence) and not _HARD_NEGATIVE.search(sentence):
            continue

        return sentence

    return None


def review_summary(place: dict) -> dict:
    """
    {"checked": n, "mentions": [ {quote, rating, relative_time, author,
                                  author_uri, maps_uri}, ... ]}
    """
    reviews = (place or {}).get("reviews") or []
    mentions = []

    for review in reviews:
        sentence = find_slow_reply_sentence(review.get("text", ""))

        if sentence is None:
            continue

        mentions.append({
            "quote": _trim_quote(sentence),
            "rating": review.get("rating"),
            "relative_time": review.get("relative_time", ""),
            "author": review.get("author", ""),
            "author_uri": review.get("author_uri", ""),
            "maps_uri": review.get("maps_uri", ""),
        })

    return {
        "checked": len(reviews),
        "mentions": mentions,
    }


# ------------------------------------------ does the user's request need reviews?

REVIEW_CRITERION_RE = re.compile(
    r"\b(?:reviews?|complain\w*|unanswered|unrespons\w+|response\s+times?)\b"
    r"|slow\s+(?:repl|respon)"
    r"|(?:no|never|not)\s+(?:repl|respon|answer)"
    r"|ignor\w+\s+(?:inquir|enquir|message|call)",
    re.I,
)


def review_criteria(criteria: list) -> list:
    """The criteria that can only be checked from customer reviews."""
    return [
        criterion
        for criterion in (criteria or [])
        if REVIEW_CRITERION_RE.search(criterion or "")
    ]


def capability_text() -> str:
    return (
        "It searches DDGS for public customer reviews, complaints, forums, news "
        "and job pages and fetches readable pages. Only fetched passages tied "
        "to the correct business count; snippets and inaccessible pages do not. "
        "Coverage is not guaranteed. Google Places is optional and can also "
        "supply up to 5 Google Maps reviews per business when configured."
    )
