"""Fetch bounded public external evidence; search snippets never qualify leads."""
import json
import ipaddress
import os
import re
import time
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse, urljoin

import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS
from business_identity import search_name
import trustpilot_reviews
from network import tls_context
from research_budget import bounded, request_timeout, ResearchBudgetExceeded, run_with_deadline


MAX_EXTERNAL_PAGES = 4
MAX_EXTERNAL_ATTEMPTS = 8
KNOWN_SOURCE_FAMILIES = ("trustpilot.com", "reviews.io", "yell.com", "checkatrade.com", "facebook.com",
                        "reddit.com", "glassdoor.com", "google.com", "yelp.com")


def normalize(value):
    return " ".join((value or "").lower().split())


def public_url(url):
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
        if parsed.scheme not in ("https", "http") or not host or parsed.username:
            return False
        if "." not in host or host.endswith((".local", ".localhost", ".internal")):
            return False
        try:
            return ipaddress.ip_address(host).is_global
        except ValueError:
            return True
    except (ValueError, TypeError):
        return False


def source_family(url_or_host):
    try:
        value = url_or_host.strip("()\"'")
        host = urlparse(value if "://" in value else "//"+value).hostname or ""
    except ValueError:
        return ""
    host = host.lower().removeprefix("www.").rstrip(".")
    for family in KNOWN_SOURCE_FAMILIES:
        if host == family or host.endswith("." + family):
            return family
    return host


def exclude_sources(query, families):
    if not families:
        return query
    query = re.sub(r"(?<!-)\bsite:([^\s]+)",
        lambda match:"" if source_family(match.group(1)) in families else match.group(0), query)
    query = " ".join(query.split())
    return query + " " + " ".join("-site:"+family for family in sorted(families)[:6]
                                    if "-site:"+family not in query)


def public_links(soup, base_url, limit=60):
    links = []
    for node in soup.find_all("a", href=True):
        try:
            target = urljoin(base_url, node["href"])
            if public_url(target) and target not in links:
                links.append(target)
        except (ValueError, TypeError):
            continue
        if len(links) >= limit:
            break
    return links


def owned_review_profiles(soup, base_url):
    profiles = []
    for node in soup.find_all("a", href=True):
        try:
            target = urljoin(base_url, node["href"])
            label = " ".join([node.get_text(" ", strip=True), node.get("aria-label", ""), node.get("title", "")]).lower()
            if (public_url(target) and source_family(target) == "trustpilot.com"
                    and urlparse(target).path.startswith("/review/")
                    and any(term in label for term in ("review", "trustpilot")) and target not in profiles):
                profiles.append(target)
        except (ValueError, TypeError):
            continue
    return profiles


def business_search_name(company, location=""):
    return search_name(company, location)


def same_business_source(text, company, source_url=""):
    """Require an identity anchor in fetched text, independently of snippets."""
    value = normalize(text)
    domain = normalize(company.get("domain")).removeprefix("www.")
    parsed = urlparse(source_url)
    if source_family(source_url) == "trustpilot.com" and parsed.path.startswith("/review/"):
        profile_domain = parsed.path.split("/")[2].lower().removeprefix("www.")
        verified_profiles = {urlparse(url).path.split("/")[2].lower().removeprefix("www.")
                             for url in company.get("verified_review_profiles", [])
                             if "/review/" in urlparse(url).path and len(urlparse(url).path.split("/")) > 2}
        if profile_domain != domain and profile_domain not in verified_profiles:
            return False
    if domain and re.search(r"(?<![a-z0-9.-])(?:www\.)?" + re.escape(domain) + r"(?![a-z0-9.-])", value):
        return True
    # Preserve short parts of names (St, Paul's); dropping them broke identity
    # checks even when an exact business name was present in fetched content.
    page_tokens = " ".join(re.findall(r"[a-z0-9]+", value))
    for name in [business_search_name(company)] + company.get("verified_aliases", []):
        tokens = re.findall(r"[a-z0-9]+", name.lower())
        tokens = [t for t in tokens if t not in {"official", "website", "home", "welcome"}]
        if len(tokens) >= 2 and " " + " ".join(tokens) + " " in " " + page_tokens + " ":
            return True
    return False


def external_queries(company, location, plan):
    name = business_search_name(company, location).replace('"', '')
    anchor = f'"{name}"'
    templates = plan.get("external_search_queries", [])[:3]
    queries = []
    for template in templates:
        # Do not spend the entire research budget on Yelp/Maps restrictions.
        template = re.sub(r"(?<!-)\bsite:(?:www\.)?(?:yelp\.com|google\.com/maps)\S*", "", template, flags=re.I)
        query = template.replace('"{company}"', anchor).replace("{company}", anchor)
        query = query.replace("{domain}", company.get("domain", ""))
        query = query.replace("{location}", location)
        if normalize(name) not in normalize(query.replace('"', '')):
            query = anchor + " " + query
        if location and location.lower() not in query.lower():
            query += " " + location
        query = " ".join(query.split())
        if query not in queries:
            queries.append(query)
    return queries


def readable_external_text(soup):
    reviews = []
    def walk(value):
        if isinstance(value, dict):
            if isinstance(value.get("reviewBody"), str):
                reviews.append(value["reviewBody"])
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            walk(json.loads(script.string or script.get_text()))
        except (ValueError, TypeError, RecursionError):
            continue
    for node in soup(["script", "style", "nav", "footer"]):
        node.decompose()
    visible = soup.get_text(" ", strip=True)
    # Keep identity context and structured reviews even on long directory pages.
    return " ".join((visible[:3000] + " " + " ".join(reviews) + " " + visible[3000:]).split())[:24000]


def fetch_external_page(client, url, company, plan):
    """Apply source exclusions and the remaining deadline on every redirect hop."""
    current = url
    own_host = company.get("domain", "").lower().removeprefix("www.")
    with bounded(plan.get("_deadline", 0)):
        for _ in range(4):
            if not public_url(current):
                raise ValueError("Non-public redirect target")
            host = (urlparse(current).hostname or "").lower().removeprefix("www.")
            if source_family(host) in plan.get("_excluded_host_families", []):
                raise ValueError("Redirect target belongs to an excluded source")
            if own_host and (host == own_host or host.endswith("."+own_host)):
                raise ValueError("External source redirected to the business's own website")
            response = client.get(current, timeout=request_timeout(12))
            if not response.is_redirect:
                return response
            current = str(response.url.join(response.headers.get("location", "")))
    raise ValueError("Redirect limit exceeded")


def collect_external_sources(company, location, plan):
    templates = external_queries(company, location, plan)
    seed_urls = company.get("seed_external_urls", [])[:3]
    if seed_urls:
        templates = [None] + templates
    if not templates:
        return {"sources": [], "errors": []}
    deadline = plan.get("_deadline", 0)
    if deadline and time.time() >= deadline:
        return {"sources":[], "errors":[{"kind":"budget_exhausted","error":"Research deadline reached"}], "fetch_attempts":0}
    provider_result = ({"sources":[], "errors":[]} if plan.get("_skip_provider")
                       else trustpilot_reviews.collect_reviews(company))
    sources, errors = provider_result["sources"], provider_result["errors"]
    seen = set()
    if sources:
        # Authorized API reviews replace HTML scraping for this business.
        return {"sources": sources, "errors": errors,
                "diagnostics": [{"query": "Trustpilot Data Solutions API", "results": len(sources)}]}
    diagnostics, search_results, attempted_urls = [], [], []
    attempt_limit = min(MAX_EXTERNAL_ATTEMPTS, max(0, plan.get("_max_external_attempts", MAX_EXTERNAL_ATTEMPTS)))
    page_limit = min(MAX_EXTERNAL_PAGES, max(0, plan.get("_max_external_pages", MAX_EXTERNAL_PAGES)))
    excluded_urls = set(plan.get("_excluded_urls", []))
    refused_families = set(plan.get("_excluded_host_families", []))
    blocked_hosts = plan.get("_blocked_hosts", set())
    blocked_urls = plan.get("_blocked_urls", set())
    host_lock = plan.get("_host_lock", Lock())
    blocked_hosts.update(plan.get("_excluded_hosts", []))
    own_host = company.get("domain", "").lower().removeprefix("www.")
    with httpx.Client(timeout=12, follow_redirects=False, verify=tls_context()) as client:
        for template in templates:
            if deadline and time.time() >= deadline:
                errors.append({"kind":"budget_exhausted", "error":"Research deadline reached"})
                break
            query = exclude_sources(template, refused_families) if template else "review-led discovery source"
            search_details = {}
            try:
                if template is None:
                    results = [{"href": url, "title": "Discovered review source"} for url in seed_urls]
                else:
                    from web_search import search_public
                    results = search_public(query, DDGS, 8, search_details)
                diagnostics.append({"query": query, "results": len(results), **search_details})
                errors.extend({"query": query, "kind": "search_failed", "error": message}
                              for message in search_details.get("backend_errors", []))
            except Exception as error:
                errors.append({"query": query, "kind": "search_failed", "error": str(error)})
                continue
            search_results.extend({"url":r.get("href") or r.get("url") or "", "title":r.get("title", ""),
                                   "snippet":r.get("body", "")[:1000]} for r in results)
            # Identity metadata only prioritizes fetches; it never qualifies a
            # lead. Verification still requires the actual fetched page text.
            results.sort(key=lambda result: not same_business_source(
                " ".join(str(result.get(key, "")) for key in ("title", "body", "href")), company))
            query_attempts = 0
            query_budget = max(1, attempt_limit // len(templates))
            for result in results:
                url = result.get("href") or result.get("url") or ""
                if not public_url(url):
                    continue
                host = (urlparse(url).hostname or "").lower().removeprefix("www.")
                if (source_family(host) in refused_families
                        or url in seen or url in excluded_urls or not public_url(url) or host == own_host
                        or host.endswith("." + own_host)):
                    continue
                path = urlparse(url).path.rstrip("/")
                if (host.endswith("yelp.com") and path == "/search") or (
                    host in ("maps.google.com", "google.com") and path in ("", "/maps")):
                    errors.append({"url": url, "kind": "generic_listing",
                                   "error": "Generic search/maps page is not a business review source"})
                    continue
                if (len(attempted_urls) >= attempt_limit or len(sources) >= page_limit
                        or query_attempts >= query_budget):
                    break
                seen.add(url)
                with host_lock:
                    blocked = host in blocked_hosts or url in blocked_urls
                if blocked:
                    errors.append({"url": url, "kind": "access_denied", "error": "Source already refused access in this batch"})
                    continue
                if deadline and time.time() >= deadline:
                    break
                query_attempts += 1
                attempted_urls.append(url)
                try:
                    response = fetch_external_page(client, url, company,
                        {**plan,"_excluded_host_families":list(refused_families)})
                    final_url = str(response.url)
                    if response.status_code in (401, 403, 429):
                        with host_lock:
                            blocked_urls.add(url)
                            blocked_urls.add(final_url)
                            if response.status_code == 429:
                                blocked_hosts.add(host)
                        family = source_family(final_url)
                        if family in KNOWN_SOURCE_FAMILIES and response.status_code in (401, 403):
                            refused_families.add(family)
                        errors.append({"url": final_url, "requested_url":url, "kind": "access_denied",
                            "status": response.status_code,
                            "error": f"HTTP {response.status_code}: source refused automated access; trying other sources"})
                        continue
                    response.raise_for_status()
                    if "html" not in response.headers.get("content-type", ""):
                        errors.append({"url": url, "error": "Not an HTML page"})
                        continue
                    soup = BeautifulSoup(response.text[:1000000], "html.parser")
                    title = soup.title.get_text(" ", strip=True).lower() if soup.title else ""
                    if title in ("just a moment...", "attention required! | cloudflare", "access denied", "robot or human?"):
                        refused_families.add(source_family(final_url))
                        errors.append({"url":final_url,"kind":"access_denied","status":200,
                            "error":"Source returned an access challenge instead of content"})
                        continue
                    links = public_links(soup, str(response.url), 25)
                    text = readable_external_text(soup)
                    if len(text) < 100 or not same_business_source(text, company, final_url):
                        errors.append({"url": url, "kind": "identity_or_text", "error": "Insufficient text or business identity not established"})
                        continue
                    if final_url in {source["url"] for source in sources}:
                        continue
                    seen.add(final_url)
                    sources.append({"url": final_url, "requested_url":url, "title": result.get("title", ""),
                                    "text": text, "query": query, "links":links})
                except Exception as error:
                    errors.append({"url": url, "kind": "budget_exhausted" if isinstance(error, ResearchBudgetExceeded)
                        else "fetch_failed", "error": str(error).split("\n")[0]})
            if len(attempted_urls) >= attempt_limit or len(sources) >= page_limit:
                break
    return {"sources": sources, "errors": errors, "diagnostics": diagnostics,
            "search_results":search_results, "attempted_urls":attempted_urls,
            "fetch_attempts":len(attempted_urls), "rate_limited_hosts":list(blocked_hosts)}


def inspect_external_all(companies, sites, location, plan):
    if not plan.get("external_search_queries"):
        return
    # Share refusals across concurrently researched businesses without putting
    # transient locks/sets into the saved research plan.
    batch_plan = {**plan, "_blocked_hosts": set(), "_blocked_urls": set(), "_host_lock": Lock()}
    def inspect(pair):
        company, site = pair
        if not site.get("reachable"):
            return
        identity = {**company, "business_name": site.get("business_name") or company.get("business_name"),
                    "verified_aliases":site.get("verified_aliases", []),
                    "verified_review_profiles":site.get("verified_review_profiles", [])}
        result = collect_external_sources(identity, location, batch_plan)
        site["external_sources"] = result["sources"]
        site["external_errors"] = result["errors"]
        site["external_diagnostics"] = result.get("diagnostics", [])
        site["external_search_results"] = result.get("search_results", [])
        site["attempted_external_urls"] = result.get("attempted_urls", [])
        site["rate_limited_hosts"] = result.get("rate_limited_hosts", [])
        print(f"    External research: {company['domain']} "
              f"({len(result['sources'])} readable sources; {len(result['errors'])} errors/rejections)")
        for error in result["errors"][:2]:
            print(f"      {error.get('url') or error.get('query')}: {error['error']}")
    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(lambda pair:run_with_deadline(plan.get("_deadline", 0), inspect, pair), zip(companies, sites)))
