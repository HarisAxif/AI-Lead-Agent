from business_identity import extract_identity
from external_research import public_links, owned_review_profiles
import research_budget
import ast
import base64
import os
import re
import time

from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin, urlparse

import httpx
from network import tls_context
import phonenumbers

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

from timing_utils import (
    start_timer,
    elapsed,
)


MAX_INTERNAL_PAGES = 3
MAX_WORKERS = 8


def render_workers() -> int:
    """RENDER_WORKERS=3 renders three sites at once (default 1)."""
    try:
        return max(1, int(os.getenv("RENDER_WORKERS", "1")))
    except ValueError:
        return 1


# Scripts of well-known live-chat / chatbot vendors.
CHAT_VENDOR_MARKERS = (
    "tawk.to", "crisp.chat", "intercom", "tidio", "livechatinc",
    "zopim", "zdassets", "zendesk", "drift.com", "jivosite",
    "smartsupp", "botpress", "chatbase", "freshchat", "olark",
    "manychat", "chatra", "purechat",
)


def chat_widget_scan(
    html: str,
):
    """
    Measured replacement for "I see no chat bubble in the screenshot".
    A tag manager can load a chat widget after the page loads, so "none
    found" is only trusted when no tag manager is present.
    """
    lowered = (html or "").lower()

    found = [
        marker
        for marker in CHAT_VENDOR_MARKERS
        if marker in lowered
    ]

    if found:
        return "found " + ", ".join(found)

    if "googletagmanager.com" in lowered or "gtm.js" in lowered:
        return (
            "none found in page HTML "
            "(tag manager present, widget may load dynamically)"
        )

    return "none found in page HTML (no tag manager)"


# =========================================================
# BASIC URL HELPERS
# =========================================================

def get_domain(
    url: str
):

    try:

        domain = urlparse(
            url
        ).netloc.lower()

        return domain.removeprefix(
            "www."
        )

    except Exception:

        return ""


# =========================================================
# SIMPLE HTTP FETCH
# Used for text/content/internal pages
# =========================================================

# Why a URL could not be fetched, for the log (domain -> short message).
FETCH_ERRORS = {}

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def fetch_page(url: str, client=None):
    """Fetch public HTML with bounded retries and inspected redirect targets."""
    from external_research import public_url
    candidates = [url]
    if url.startswith("https://"):
        candidates.append("http://" + url[len("https://"):])
    elif url.startswith("http://"):
        candidates.append("https://" + url[len("http://"):])
    last_error = "unknown error"
    for candidate in candidates:
        for attempt in range(2):
            current = candidate
            try:
                for _ in range(5):
                    if not public_url(current):
                        raise ValueError("Blocked non-public URL or redirect target")
                    available = research_budget.check()
                    kwargs = {"timeout": min(8, available) if available is not None else 8, "follow_redirects": False, "headers": BROWSER_HEADERS}
                    response = client.get(current, **kwargs) if client is not None else httpx.get(current, verify=tls_context(), **kwargs)
                    if not response.is_redirect:
                        break
                    current = urljoin(current, response.headers.get("location", ""))
                else:
                    raise ValueError("Redirect limit exceeded")
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").lower()
                if "html" not in content_type:
                    last_error = f"not an HTML page ({content_type or 'no type'})"
                    break
                FETCH_ERRORS.pop(url, None)
                return response
            except httpx.HTTPStatusError as error:
                last_error = f"HTTP {error.response.status_code}: {error.response.url}"
                if error.response.status_code in (401, 403, 429):
                    FETCH_ERRORS[url] = last_error
                    return None
                if error.response.status_code < 500:
                    break
            except Exception as error:
                last_error = f"{type(error).__name__}: {str(error)[:160]}"
                if isinstance(error, (ValueError, research_budget.ResearchBudgetExceeded)) or "CERTIFICATE" in str(error).upper():
                    FETCH_ERRORS[url] = last_error
                    return None
                # Retrying a DNS failure on the same host/scheme cannot help.
                if "getaddrinfo" in str(error).lower():
                    FETCH_ERRORS[url] = last_error
                    return None
    FETCH_ERRORS[url] = last_error
    return None


# =========================================================
# CLEAN HTML TEXT
# =========================================================

def clean_page_text(
    html: str
):

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    for tag in soup(
        [
            "script",
            "style",
            "noscript",
            "svg"
        ]
    ):

        tag.decompose()

    text = soup.get_text(
        " ",
        strip=True
    )

    return re.sub(
        r"\s+",
        " ",
        text
    )


# =========================================================
# FIND INTERNAL PAGES RELEVANT TO RESEARCH PLAN
# =========================================================

# Plans often say "Contact Us page" or "Services page". Those full phrases
# never appear in a link label, so no subpage was ever read. Reduce each
# phrase to its keywords and always include the usual suspects.
_PAGE_STOPWORDS = {
    "page", "pages", "section", "sections", "the", "our", "home",
    "homepage", "main", "site", "website", "and", "for", "your",
}

PRIORITY_PAGE_KEYWORDS = ("contact", "about")

DEFAULT_PAGE_KEYWORDS = (
    "service", "location", "office", "faq", "support", "help",
)


def normalize_page_keywords(
    page_keywords: list[str],
):
    keywords = list(PRIORITY_PAGE_KEYWORDS)

    for phrase in page_keywords or []:
        for token in re.split(r"[^a-zA-Z]+", str(phrase).lower()):
            if (
                len(token) > 2
                and token not in _PAGE_STOPWORDS
                and token not in keywords
            ):
                keywords.append(token)

    for token in DEFAULT_PAGE_KEYWORDS:
        if token not in keywords:
            keywords.append(token)

    return keywords


def find_relevant_internal_links(
    soup,
    base_url: str,
    page_keywords: list[str]
):
    """
    Returns up to MAX_INTERNAL_PAGES same-site links. Contact and About
    pages are ranked first so a long services menu can't crowd them out.
    """
    base_domain = get_domain(
        base_url
    )

    keywords = normalize_page_keywords(
        page_keywords
    )

    base_clean = base_url.split("#")[0].rstrip("/")

    scored = []
    seen = set()

    for anchor in soup.find_all(
        "a",
        href=True
    ):
        href = anchor.get(
            "href",
            ""
        ).strip()

        if href.lower().startswith(
            ("mailto:", "tel:", "javascript:", "#")
        ):
            continue

        label = (
            anchor.get_text(
                " ",
                strip=True
            )
            + " "
            + href
        ).lower()

        rank = next(
            (
                index
                for index, keyword in enumerate(keywords)
                if keyword in label
            ),
            None,
        )

        if rank is None:
            continue

        full_url = urljoin(
            base_url,
            href
        ).split("#")[0]

        if get_domain(
            full_url
        ) != base_domain:
            continue

        path = urlparse(full_url).path

        if (
            full_url.rstrip("/") == base_clean
            or full_url in seen
            or len(path.rstrip("/").rsplit("/", 1)[-1]) > 40
        ):
            continue

        seen.add(
            full_url
        )

        scored.append(
            (rank, len(path), full_url)
        )

        if len(scored) >= 60:
            break

    scored.sort(
        key=lambda item: (item[0], item[1])
    )

    return [
        url
        for _, _, url in scored[:MAX_INTERNAL_PAGES]
    ]


# =========================================================
# SCRIPT HOSTS
# =========================================================

def extract_script_hosts(
    soup
):

    hosts = set()

    for script in soup.find_all(
        "script",
        src=True
    ):

        src = script.get(
            "src",
            ""
        )

        if src.startswith(
            "//"
        ):

            src = "https:" + src

        if not src.startswith(
            "http"
        ):

            continue

        host = get_domain(
            src
        )

        if host:

            hosts.add(
                host
            )

    return sorted(
        hosts
    )[:15]



# =========================================================
# CONTACT EXTRACTION
# =========================================================

def extract_phone_numbers(
    text: str
):
    # Start at +, (, or a digit so "(307) 742-2479" keeps its opening bracket.
    raw_matches = re.findall(
        r"(?<!\w)(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)|\d{2,4})"
        r"(?:[\s.-]?\d{2,4}){2,4}(?!\w)",
        text or "",
    )

    cleaned = []
    seen = set()

    for value in raw_matches:

        value = re.sub(
            r"\s+",
            " ",
            value,
        ).strip(
            " ,.;:"
        )

        digits = re.sub(
            r"\D",
            "",
            value,
        )

        if not (
            7 <= len(digits) <= 15
        ):
            continue

        # Reject obvious dates.
        if re.fullmatch(
            r"\d{1,2}[-/.]\d{1,2}[-/.]\d{4}",
            value,
        ):
            continue

        if value in seen:
            continue

        seen.add(
            value
        )
        cleaned.append(
            value
        )

    return cleaned[:5]


# =========================================================
# CONTACT EXTRACTION + VALIDATION
# =========================================================

EMAIL_RE = re.compile(
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+"
)

FREE_EMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com",
    "icloud.com", "live.com", "proton.me", "protonmail.com",
}

NOT_AN_EMAIL_SUFFIXES = (
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp",
    ".css", ".js", ".woff", ".woff2",
)

CREDIT_RE = re.compile(
    r"\b(designed|developed|built|powered|created|hosted|website|"
    r"site|web design|seo)\s*(?:&\s*\w+\s*)?by\b",
    re.I,
)


def _in_credit_block(
    tag,
):
    """
    True if the link sits next to a "designed by / powered by" credit.
    Only the DIRECT text of the link's own container (and its parent) is
    checked, so a short page body can't flag every link on the page.
    """
    node = tag.parent

    for _ in range(2):
        if node is None:
            break

        direct_text = " ".join(
            node.find_all(
                string=True,
                recursive=False,
            )
        )

        if CREDIT_RE.search(direct_text):
            return True

        node = node.parent

    return False


GENERIC_MAILBOXES = (
    "info", "contact", "support", "sales", "hello", "enquiry", "enquire",
    "inquiry", "office", "admin", "mail", "team", "help", "accounts",
    "service", "services", "reception", "booking", "bookings",
)


def prefer_generic_emails(
    emails: list,
):
    """
    Role mailboxes (info@, contact@) are the right place to reach a
    business. Named staff addresses are kept only when nothing generic exists.
    """
    generic = [
        email for email in emails
        if email.split("@")[0].split("+")[0].lower() in GENERIC_MAILBOXES
    ]

    return generic or emails


def email_belongs_to_site(
    email: str,
    site_domain: str,
):
    domain = email.split("@")[-1].lower()

    return (
        domain == site_domain
        or domain.endswith("." + site_domain)
        or domain in FREE_EMAIL_DOMAINS
    )


def normalize_phone(
    raw: str,
    country_code: str | None,
):
    """
    Returns (dedupe_key, display) or None when the number is invalid
    or belongs to a different country than the search location.
    """
    raw = (raw or "").strip()

    if not raw:
        return None

    digits = re.sub(r"\D", "", raw)

    if not 7 <= len(digits) <= 15:
        return None

    try:
        if raw.startswith("00"):
            number = phonenumbers.parse("+" + raw[2:], None)
        elif raw.startswith("+"):
            number = phonenumbers.parse(raw, None)
        elif country_code:
            number = phonenumbers.parse(raw, country_code)
        else:
            # No country to check against: keep as written.
            return digits, raw

    except phonenumbers.NumberParseException:
        return None

    if not phonenumbers.is_valid_number(number):
        return None

    if (
        country_code
        and phonenumbers.region_code_for_number(number) != country_code
    ):
        return None

    return (
        phonenumbers.format_number(
            number,
            phonenumbers.PhoneNumberFormat.E164,
        ),
        phonenumbers.format_number(
            number,
            phonenumbers.PhoneNumberFormat.INTERNATIONAL,
        ),
    )


def extract_contacts(
    soups: list,
    text: str,
    site_domain: str,
    country_code: str | None,
):
    """
    Prefer tel:/mailto: links (reliable), then page text. Drop developer
    credits, emails from unrelated domains, invalid numbers, numbers from
    another country, and duplicates.
    """
    raw_emails = []
    raw_phones = []

    for soup in soups:
        for link in soup.find_all("a", href=True):
            href = link["href"].strip()
            lowered = href.lower()

            whatsapp = re.search(
                r"(?:wa\.me/|whatsapp\.com/send\?[^\"']*phone=)\+?(\d{7,15})",
                lowered,
            )

            if whatsapp:
                if not _in_credit_block(link):
                    raw_phones.append(
                        "+" + whatsapp.group(1)
                    )
                continue

            if not (
                lowered.startswith("tel:")
                or lowered.startswith("mailto:")
            ):
                continue

            if _in_credit_block(link):
                continue

            if lowered.startswith("tel:"):
                raw_phones.append(
                    href[4:].split("?")[0]
                )
            else:
                raw_emails.append(
                    href[7:].split("?")[0]
                )

    raw_emails.extend(
        EMAIL_RE.findall(
            text or ""
        )
    )

    raw_phones.extend(
        extract_phone_numbers(
            text
        )
    )

    emails = []
    seen_emails = set()

    for email in raw_emails:
        email = email.strip().strip(".,;:").lower()

        if (
            not EMAIL_RE.fullmatch(email)
            or email.endswith(NOT_AN_EMAIL_SUFFIXES)
            or email in seen_emails
            or not email_belongs_to_site(email, site_domain)
        ):
            continue

        seen_emails.add(email)
        emails.append(email)

    phones = []
    seen_phones = set()

    for raw in raw_phones:
        normalized = normalize_phone(
            raw,
            country_code,
        )

        if normalized is None:
            continue

        key, display = normalized

        if key in seen_phones:
            continue

        seen_phones.add(key)
        phones.append(display)

    return prefer_generic_emails(emails)[:5], phones[:5]


def _facts_list(
    rendered_facts: str,
    name: str,
):
    match = re.search(
        rf"{name}: (\[.*?\])",
        rendered_facts or "",
    )

    if not match:
        return []

    try:
        value = ast.literal_eval(
            match.group(1)
        )
    except Exception:
        return []

    if not isinstance(value, (list, tuple)):
        return []

    return [str(item) for item in value]


def merge_rendered_contacts(
    site: dict,
    rendered_facts: str,
    site_domain: str,
    country_code: str | None,
):
    """
    Contacts that only exist after JavaScript runs (tel:, mailto:,
    WhatsApp links) are added to the lists found in the static HTML.
    """
    emails = list(site.get("emails") or [])
    phones = list(site.get("phones") or [])

    seen_phones = set()

    for phone in phones:
        normalized = normalize_phone(phone, country_code)

        if normalized:
            seen_phones.add(normalized[0])

    candidates = []

    for item in _facts_list(rendered_facts, "rendered_phone_links"):
        candidates.append(
            re.sub(r"^tel:", "", item.strip(), flags=re.I)
            .split("?")[0]
        )

    for item in _facts_list(rendered_facts, "rendered_whatsapp_links"):
        match = re.search(
            r"(?:wa\.me/|phone=)\+?(\d{7,15})",
            item,
        )

        if match:
            candidates.append("+" + match.group(1))

    for raw in candidates:
        normalized = normalize_phone(raw, country_code)

        if normalized and normalized[0] not in seen_phones:
            seen_phones.add(normalized[0])
            phones.append(normalized[1])

    for item in _facts_list(rendered_facts, "rendered_email_links"):
        email = (
            re.sub(r"^mailto:", "", item.strip(), flags=re.I)
            .split("?")[0]
            .strip()
            .lower()
        )

        if (
            EMAIL_RE.fullmatch(email)
            and email not in emails
            and email_belongs_to_site(email, site_domain)
        ):
            emails.append(email)

    site["emails"] = prefer_generic_emails(emails)[:5]
    site["phones"] = phones[:5]


# =========================================================
# HTML / TEXT INSPECTION
# =========================================================

def inspect_html(
    company: dict,
    plan: dict
):

    homepage_response = fetch_page(
        company[
            "url"
        ]
    )

    if homepage_response is None:

        print(
            f"    Could not fetch {company['domain']}: "
            f"{FETCH_ERRORS.get(company['url'], 'unknown error')}"
        )

        return {
            "reachable": False,
            "fetch_error": FETCH_ERRORS.get(company["url"], "unknown error"),
            "text": "",
            "footprint": "",
            "emails": [],
            "phones": [],
            "pages_read": [],
            "desktop_screenshot": None,
            "mobile_screenshot": None,
            "rendered_facts": ""
        }

    homepage_html = (
        homepage_response.text
    )

    homepage_soup = BeautifulSoup(
        homepage_html,
        "html.parser"
    )

    homepage_text = clean_page_text(
        homepage_html
    )

    pages_text = [
        homepage_text
    ]

    page_soups = [
        homepage_soup
    ]

    pages_read = [
        str(
            homepage_response.url
        )
    ]

    internal_links = (
        find_relevant_internal_links(
            homepage_soup,
            str(
                homepage_response.url
            ),
            plan[
                "pages_to_check"
            ]
        )
    )

    for link in internal_links:

        page_response = fetch_page(
            link
        )

        if page_response is None:

            continue

        final_url = str(
            page_response.url
        ).split("#")[0].rstrip("/")

        if final_url in {
            url.split("#")[0].rstrip("/")
            for url in pages_read
        }:
            continue

        page_text = clean_page_text(
            page_response.text
        )

        if page_text:

            page_soups.append(
                BeautifulSoup(
                    page_response.text,
                    "html.parser"
                )
            )

            pages_text.append(
                page_text
            )

            pages_read.append(
                str(
                    page_response.url
                )
            )

    combined_text = " ".join(
        pages_text
    )

    emails, phones = extract_contacts(
        soups=page_soups,
        text=combined_text,
        site_domain=get_domain(
            company["url"]
        ),
        country_code=plan.get(
            "country_code"
        ),
    )

    copyright_years = re.findall(
        r"(?:©|copyright)\D{0,10}(\d{4})",
        combined_text,
        re.I
    )[:5]

    script_hosts = extract_script_hosts(
        homepage_soup
    )

    footprint_lines = [
        (
            "https: "
            f"{str(homepage_response.url).startswith('https')}"
        ),
        (
            "http_load_seconds: "
            f"{round(homepage_response.elapsed.total_seconds(), 2)}"
        ),
        (
            "html_mobile_viewport_tag: "
            f"{bool(homepage_soup.find('meta', attrs={'name': 'viewport'}))}"
        ),
        (
            "html_forms_on_homepage: "
            f"{len(homepage_soup.find_all('form'))}"
        ),
        (
            "third_party_script_hosts: "
            f"{script_hosts}"
        ),
        (
            "chat_widget_scan: "
            f"{chat_widget_scan(homepage_html)}"
        ),
        (
            "copyright_years: "
            f"{copyright_years}"
        ),
        (
            "website_text_word_count: "
            f"{len(combined_text.split())}"
        ),
        (
            "pages_read: "
            f"{len(pages_read)}"
        ),
        (
            "emails_found: "
            f"{len(emails)}"
        ),
        (
            "phones_found: "
            f"{len(phones)}"
        )
    ]

    return {
        **extract_identity(homepage_soup, company["domain"], str(homepage_response.url)),
        "verified_review_profiles": owned_review_profiles(homepage_soup, str(homepage_response.url)),
        "discovered_links": public_links(homepage_soup, str(homepage_response.url)),
        "reachable": True,

        "text":
            combined_text[:10000],

        "footprint":
            "\n".join(
                footprint_lines
            ),

        "emails":
            emails,

        "phones":
            phones,

        "pages_read":
            pages_read,

        "desktop_screenshot":
            None,

        "mobile_screenshot":
            None,

        "rendered_facts":
            ""
    }


# =========================================================
# RENDERED DOM FACTS
# =========================================================

def get_rendered_dom_facts(
    page
):

    data = page.evaluate(
        """
        () => {

            const visible = (el) => {

                const style =
                    window.getComputedStyle(el);

                const rect =
                    el.getBoundingClientRect();

                return (
                    style.display !== "none" &&
                    style.visibility !== "hidden" &&
                    style.opacity !== "0" &&
                    rect.width > 0 &&
                    rect.height > 0
                );
            };


            const clickable =
                Array.from(
                    document.querySelectorAll(
                        'a, button, [role="button"], input[type="submit"]'
                    )
                )
                .filter(visible)
                .map(el => ({
                    text: (
                        el.innerText ||
                        el.value ||
                        el.textContent ||
                        ""
                    )
                    .trim()
                    .replace(/\\s+/g, " "),

                    href:
                        el.href || "",

                    tag:
                        el.tagName
                }))
                .filter(
                    x => x.text || x.href
                )
                .slice(
                    0,
                    60
                );


            const navigation =
                Array.from(
                    document.querySelectorAll(
                        'nav a, header a'
                    )
                )
                .filter(visible)
                .map(el => (
                    el.innerText ||
                    el.textContent ||
                    ""
                )
                .trim()
                .replace(/\\s+/g, " "))
                .filter(Boolean)
                .slice(
                    0,
                    30
                );


            const telLinks =
                Array.from(
                    document.querySelectorAll(
                        'a[href^="tel:"]'
                    )
                )
                .map(
                    a => a.href
                );


            const mailLinks =
                Array.from(
                    document.querySelectorAll(
                        'a[href^="mailto:"]'
                    )
                )
                .map(
                    a => a.href
                );


            const whatsappLinks =
                Array.from(
                    document.querySelectorAll(
                        'a'
                    )
                )
                .map(
                    a => a.href || ""
                )
                .filter(
                    href =>
                        href.includes("wa.me") ||
                        href.toLowerCase().includes("whatsapp")
                );


            const visibleForms =
                Array.from(
                    document.querySelectorAll(
                        "form"
                    )
                )
                .filter(
                    visible
                );


            const bodyText =
                (
                    document.body.innerText ||
                    ""
                )
                .trim()
                .replace(
                    /\\s+/g,
                    " "
                );


            return {

                title:
                    document.title,

                body_text_length:
                    bodyText.length,

                visible_forms:
                    visibleForms.length,

                clickable_count:
                    clickable.length,

                clickable:
                    clickable,

                navigation:
                    navigation,

                tel_links:
                    telLinks,

                mail_links:
                    mailLinks,

                whatsapp_links:
                    whatsappLinks
            };
        }
        """
    )

    lines = [

        (
            "rendered_page_title: "
            f"{data.get('title', '')}"
        ),

        (
            "rendered_body_text_length: "
            f"{data.get('body_text_length', 0)}"
        ),

        (
            "rendered_clickable_count: "
            f"{data.get('clickable_count', 0)}"
        ),

        (
            "rendered_visible_forms: "
            f"{data.get('visible_forms', 0)}"
        ),

        (
            "rendered_phone_links: "
            f"{data.get('tel_links', [])}"
        ),

        (
            "rendered_email_links: "
            f"{data.get('mail_links', [])}"
        ),

        (
            "rendered_whatsapp_links: "
            f"{data.get('whatsapp_links', [])}"
        ),

        (
            "rendered_navigation: "
            f"{data.get('navigation', [])}"
        ),

        "rendered_clickable_elements:"
    ]

    for element in data.get(
        "clickable",
        []
    ):

        lines.append(
            (
                f"- text="
                f"{element.get('text', '')!r}; "
                f"href="
                f"{element.get('href', '')!r}"
            )
        )

    return "\n".join(
        lines
    )


# =========================================================
# PAGE READINESS / LOADING SCREEN CHECK
# =========================================================

def get_page_health(
    page
):

    try:

        return page.evaluate(
            """
            () => {

                const text =
                    (
                        document.body.innerText ||
                        ""
                    )
                    .trim();


                const links =
                    document.querySelectorAll(
                        'a'
                    ).length;


                const buttons =
                    document.querySelectorAll(
                        'button, [role="button"]'
                    ).length;


                const images =
                    document.querySelectorAll(
                        'img'
                    ).length;


                return {

                    text_length:
                        text.length,

                    links:
                        links,

                    buttons:
                        buttons,

                    images:
                        images
                };
            }
            """
        )

    except Exception:

        return {
            "text_length": 0,
            "links": 0,
            "buttons": 0,
            "images": 0
        }


def page_looks_incomplete(
    health: dict
):

    health = health or {}
    text_length = health.get(
        "text_length",
        0
    )

    links = health.get(
        "links",
        0
    )

    buttons = health.get(
        "buttons",
        0
    )

    # Very little visible content usually indicates
    # loader, splash page, failed JS render or blocked content.
    if (
        text_length < 250
        and links < 4
        and buttons < 3
    ):

        return True

    return False


# =========================================================
# WAIT FOR REAL PAGE CONTENT
# =========================================================

def wait_for_page_to_settle(page):
    """Wait for useful visible content instead of unconditional long sleeps."""
    try:
        page.wait_for_function(
            "() => document.body && document.body.innerText.trim().length > 250",
            timeout=int(research_budget.request_timeout(9)*1000),
        )
    except Exception:
        # Short hydration grace; subsequent health checks describe incomplete pages.
        page.wait_for_timeout(int(research_budget.request_timeout(1.1)*1000))


def route_public_requests(route):
    from external_research import public_url
    remaining = research_budget.remaining()
    if not public_url(route.request.url) or remaining is not None and remaining <= 0:
        route.abort()
    else:
        route.continue_()


def render_site(
    browser,
    company: dict,
    screenshots=True,
):
    remaining = research_budget.remaining()
    if remaining is not None and remaining <= 0:
        return None

    context = browser.new_context(
        viewport={
            "width": 1440,
            "height": 900
        }
    )

    page = context.new_page()
    context.route("**/*", route_public_requests)

    try:
        page.set_default_timeout(int(research_budget.request_timeout(15)*1000))

        response = page.goto(
            company[
                "url"
            ],
            wait_until="domcontentloaded",
            timeout=int(research_budget.request_timeout(20)*1000)
        )

        if (
            response is not None
            and response.status >= 400
        ):

            print(
                f"    {company['domain']} answered the browser with "
                f"HTTP {response.status}"
            )

            context.close()

            return None

        # =================================================
        # WAIT UNTIL SPLASH / LOADER / JS CONTENT SETTLES
        # =================================================

        wait_for_page_to_settle(
            page
        )

        # =================================================
        # FINAL HEALTH CHECK
        # =================================================

        health = get_page_health(
            page
        )
        final_url = page.url
        if get_domain(final_url) != company["domain"].removeprefix("www."):
            raise ValueError("Official website redirected to another business domain")
        if page_looks_incomplete(health):
            raise ValueError("Browser did not return enough readable business content")
        soup = BeautifulSoup(page.content(), "html.parser")
        title = soup.title.get_text(" ", strip=True).lower() if soup.title else ""
        if title in ("just a moment...", "attention required! | cloudflare", "access denied", "robot or human?"):
            raise ValueError("Website returned an access challenge")

        print(
            "    Rendered content: "
            f"{health.get('text_length', 0)} chars, "
            f"{health.get('links', 0)} links, "
            f"{health.get('buttons', 0)} buttons"
        )

        # =================================================
        # EXTRACT DOM ONLY AFTER PAGE HAS SETTLED
        # =================================================

        rendered_facts = (
            get_rendered_dom_facts(
                page
            )
        )

        try:

            rendered_text = page.locator(
                "body"
            ).inner_text(
                timeout=5000
            )

            rendered_text = re.sub(
                r"\s+",
                " ",
                rendered_text
                or ""
            ).strip()

        except Exception:

            rendered_text = ""

        metadata = {
            **extract_identity(soup, company["domain"], final_url),
            "discovered_links": public_links(soup, final_url),
            "verified_review_profiles": owned_review_profiles(soup, final_url),
            "rendered_url": final_url,
        }
        if not screenshots:
            context.close()
            return {
                **metadata, "desktop_screenshot": None, "mobile_screenshot": None,
                "rendered_facts": rendered_facts, "rendered_text": rendered_text,
                "render_health": health,
            }

        # =================================================
        # DESKTOP SCREENSHOT
        # =================================================

        desktop_bytes = page.screenshot(
            type="jpeg",
            quality=70,
            full_page=False
        )

        # =================================================
        # MOBILE VIEW
        # =================================================

        page.set_viewport_size(
            {
                "width": 390,
                "height": 844
            }
        )

        # Changing viewport can trigger responsive
        # JS/layout changes. Give it time to settle.
        page.wait_for_timeout(
            int(research_budget.request_timeout(1.8)*1000)
        )

        mobile_health = get_page_health(
            page
        )

        # If responsive version itself loads slowly,
        # give it another chance.
        if page_looks_incomplete(
            mobile_health
        ):

            page.wait_for_timeout(
                int(research_budget.request_timeout(3)*1000)
            )

        mobile_bytes = page.screenshot(
            type="jpeg",
            quality=70,
            full_page=False
        )

        context.close()

        return {
            **metadata,

            "desktop_screenshot":
                base64.b64encode(
                    desktop_bytes
                ).decode(
                    "utf-8"
                ),

            "mobile_screenshot":
                base64.b64encode(
                    mobile_bytes
                ).decode(
                    "utf-8"
                ),

            "rendered_facts":
                rendered_facts,

            "rendered_text":
                rendered_text,

            "render_health":
                health
        }

    except Exception as error:

        print(
            f"    Browser render failed: "
            f"{error}"
        )

        context.close()

        return None


# =========================================================
# INSPECT ALL CANDIDATES
# =========================================================

def _render_one(
    browser,
    index,
    company,
    site,
    plan,
    total,
):

    browser_fallback = (
        not site[
            "reachable"
        ]
    )

    print(
        f"\nRendering "
        f"[{index}/{total}]: "
        f"{company['domain']}"
    )

    render_start = start_timer()

    visual = render_site(
        browser,
        company,
        screenshots=bool(plan.get("visual_review_needed")),
    )

    render_duration = elapsed(
        render_start
    )

    print(
        f"    [TIMER] Chromium render ({company['domain']}): "
        f"{render_duration:.2f}s"
    )

    if visual is None:
        site["browser_error"] = "Browser could not read a valid official business page"

        print(
            "    Visual render unavailable."
        )

        return

    # Add screenshots + DOM facts
    site.update(
        visual
    )

    if browser_fallback:

        rendered_text = (
            visual.get(
                "rendered_text",
                ""
            )
        )

        if rendered_text:

            site[
                "reachable"
            ] = True

            site[
                "text"
            ] = (
                rendered_text[:10000]
            )

            site[
                "pages_read"
            ] = [
                visual.get("rendered_url", company["url"])
            ]

            site[
                "fallback_source"
            ] = "browser_render"

        else:

            site[
                "fallback_source"
            ] = "search_snippet"

            site[
                "fallback_text"
            ] = company.get(
                "snippet",
                ""
            )

    # Add rendered information into footprint
    # Gemini can now compare HTML facts against
    # what Chromium actually found.
    site["footprint"] = site.get("footprint", "") + (
        "\n"
        + visual[
            "rendered_facts"
        ]
    )

    merge_rendered_contacts(
        site,
        visual["rendered_facts"],
        get_domain(
            company["url"]
        ),
        plan.get(
            "country_code"
        ),
    )


def _render_partition(items, plan, total=None):
    if plan.get("_deadline") and time.time() >= plan["_deadline"]:
        return
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                for index, company, site in items:
                    if plan.get("_deadline") and time.time() >= plan["_deadline"]:
                        break
                    research_budget.run_with_deadline(plan.get("_deadline", 0),
                        _render_one, browser, index, company, site, plan, total or len(items))
            finally:
                browser.close()
    except Exception as error:
        print(f"    Browser recovery unavailable: {str(error).splitlines()[0]}", flush=True)
        for _, _, site in items:
            site["browser_error"] = str(error).splitlines()[0]


def inspect_all(
    companies: list,
    plan: dict
):

    inspection_start = start_timer()

    # =====================================================
    # STEP 1:
    # Fetch normal HTML/text in parallel
    # =====================================================

    html_start = start_timer()

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as pool:

        html_results = list(
            pool.map(
                lambda company:
                    research_budget.run_with_deadline(plan.get("_deadline", 0), inspect_html, company, plan),
                companies
            )
        )

    html_duration = elapsed(
        html_start
    )

    reachable_count = sum(
        1
        for site in html_results
        if site[
            "reachable"
        ]
    )

    print(
        f"\n[TIMER] HTML/text inspection batch: "
        f"{html_duration:.2f}s "
        f"({reachable_count}/{len(companies)} reachable)"
    )

    # =====================================================
    # STEP 2:
    # Recover unreadable pages independently of the visual-evidence decision.
    # =====================================================

    jobs = [(index, company, site) for index, (company, site) in
            enumerate(zip(companies, html_results), start=1)
            if plan.get("visual_review_needed") or not site.get("reachable")]

    if not jobs:

        inspection_duration = elapsed(
            inspection_start
        )

        print(
            f"[TIMER] ALL website inspection: "
            f"{inspection_duration:.2f}s"
        )

        return html_results

    print(
        "\nInspecting websites in Chromium (text recovery or requested visual review)..."
    )

    # =====================================================
    # STEP 3:
    # Open actual Chromium
    # Partition work across the configured number of browser instances.
    # =====================================================

    browser_batch_start = start_timer()

    workers = min(render_workers(), len(jobs))
    chunks = [jobs[position::workers] for position in range(workers)]
    if workers == 1:
        _render_partition(chunks[0], plan, len(companies))
    elif workers:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda chunk: _render_partition(chunk, plan, len(companies)), chunks))

    browser_batch_duration = elapsed(
        browser_batch_start
    )

    inspection_duration = elapsed(
        inspection_start
    )

    print(
        "\n" + "-" * 70
    )

    print(
        f"[TIMER] Chromium batch: "
        f"{browser_batch_duration:.2f}s"
    )

    print(
        f"[TIMER] ALL website inspection: "
        f"{inspection_duration:.2f}s"
    )

    print(
        "-" * 70
    )

    return html_results
