"""Discover official businesses from review search hits; snippets are hints only."""
import re
from urllib.parse import urlparse, unquote
import httpx
from bs4 import BeautifulSoup
from external_research import public_url, source_family
from research_budget import request_timeout, check, ResearchBudgetExceeded

REVIEW_HOSTS = ("trustpilot.com", "yell.com", "checkatrade.com", "reviews.io")


def is_review_url(url):
    if not public_url(url):
        return False
    host = (urlparse(url).hostname or "").lower()
    return any(host == domain or host.endswith("." + domain) for domain in REVIEW_HOSTS)


def official_from_review_url(url):
    if not public_url(url):
        return None
    parsed = urlparse(url)
    if source_family(url) == "trustpilot.com":
        match = re.fullmatch(r"/review/([^/]+)/?", parsed.path)
        if match:
            domain = unquote(match.group(1)).lower().removeprefix("www.")
            official = "https://" + domain + "/"
            if (re.fullmatch(r"[a-z0-9.-]+\.[a-z]{2,}", domain)
                    and public_url(official) and not is_review_url(official)):
                return official
    return None


def official_from_page(url):
    # Directory links are hints to official businesses, never qualification.
    if not public_url(url):
        return None
    try:
        with httpx.Client(timeout=8, follow_redirects=False) as client:
            response = client.get(url, timeout=request_timeout(8))
            if response.status_code != 200:
                return None
            soup = BeautifulSoup(response.text[:500000], "html.parser")
            for link in soup.find_all("a", href=True):
                target = link["href"]
                label = " ".join([link.get_text(" ", strip=True), link.get("aria-label", "")]).lower()
                if ("website" in label and public_url(target) and not is_review_url(target)
                        and (urlparse(target).hostname or "") != (urlparse(url).hostname or "")):
                    return target
    except (httpx.HTTPError, ResearchBudgetExceeded, ValueError):
        pass
    return None


def review_candidates(results, seen_domains, limit):
    candidates = []
    seen = set(seen_domains)
    # Bound directory lookups per discovery query.
    for result in results[:8]:
        try:
            check()
        except ResearchBudgetExceeded:
            break
        source = result.get("href") or result.get("url") or ""
        if not is_review_url(source):
            continue
        official = official_from_review_url(source) or official_from_page(source)
        if not official:
            continue
        domain = (urlparse(official).hostname or "").lower().removeprefix("www.")
        if domain in seen:
            continue
        seen.add(domain)
        candidates.append({"url": official, "domain": domain,
            "title": (result.get("title") or domain).split(" | ")[0],
            "snippet": result.get("body", ""), "seed_external_urls": [source],
            "discovery_source": "review_search"})
        if len(candidates) >= limit:
            break
    return candidates
