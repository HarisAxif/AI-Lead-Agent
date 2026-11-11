"""Extract business identity from owned website metadata, not SEO slogans."""
import html
import json
import re
from urllib.parse import urlparse

GENERIC = {"home", "welcome", "official", "website", "the", "and", "in", "of", "to",
    "best", "trusted", "top", "professional", "local", "services", "service", "experts",
    "plumbing", "plumber", "plumbers", "heating", "engineer", "engineers", "gas", "boiler",
    "boilers", "repair", "repairs", "replacement", "installation", "new", "contact", "us"}


def tokens(text):
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def useful_name(name, location=""):
    words = tokens(name)
    return bool(words) and bool(set(words) - GENERIC - set(tokens(location)))


def search_name(company, location=""):
    domain = company.get("domain", "").lower().removeprefix("www.")
    name = html.unescape(company.get("business_name") or "")
    if name and useful_name(name, location):
        parts = re.split(r"\s*[|\u2013\u2014]\s*|\s+-\s+", name)
        if len(parts) > 1:
            return search_name({"title": html.unescape(name), "domain": domain}, location)
        return name[:120]
    parts = re.split(r"\s*[|\u2013\u2014]\s*|\s+-\s+", company.get("title", ""))
    stem = domain.split(".")[0]
    for part in parts:
        distinctive = [t for t in tokens(part) if t not in GENERIC and t not in tokens(location)]
        if distinctive and any(len(t) > 2 and t in stem for t in distinctive):
            return part.strip()[:120]
    return domain


def extract_identity(soup, domain, page_url=None):
    host = domain.lower().removeprefix("www.")
    page_path = urlparse(page_url or "https://"+host+"/").path.rstrip("/")
    identities = []
    def nodes(value):
        if isinstance(value, list):
            for item in value:
                yield from nodes(item)
        elif isinstance(value, dict):
            yield value
            yield from nodes(value.get("@graph", []))
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            values = json.loads(script.string or script.get_text())
        except (ValueError, TypeError):
            continue
        for node in nodes(values):
            types = node.get("@type", [])
            types = [types] if isinstance(types, str) else types
            if not isinstance(types, list):
                continue
            if not any(isinstance(t, str) and (t.rsplit("/", 1)[-1] in {
                    "Organization", "LocalBusiness", "Plumber", "HVACBusiness", "Physician", "Dentist",
                    "MedicalClinic", "School", "Restaurant", "AutoRepair", "HairSalon", "ProfessionalService"}
                    or t.endswith(("Organization", "Business", "Clinic"))) for t in types):
                continue
            url = node.get("url")
            if url:
                try:
                    if not isinstance(url, str) or (urlparse(url).hostname or "").removeprefix("www.") != host:
                        continue
                except ValueError:
                    continue
            name = node.get("name")
            if isinstance(name, str) and useful_name(name):
                aliases = node.get("alternateName", [])
                aliases = [aliases] if isinstance(aliases, str) else aliases
                aliases = [value for value in aliases if isinstance(value, str)] if isinstance(aliases, list) else []
                if isinstance(node.get("legalName"), str):
                    aliases.append(node["legalName"])
                address = node.get("address", {})
                if isinstance(address, dict):
                    address = ", ".join(str(address.get(key, "")) for key in
                                        ("streetAddress", "addressLocality", "addressRegion", "addressCountry") if address.get(key))
                identity = {"business_name": name.strip(), "identity_source": "structured_data",
                        "identity_url":url if isinstance(url, str) else page_url,
                        "verified_aliases":list(dict.fromkeys([name.strip()] + aliases)),
                        "business_address":str(address)}
                node_path = urlparse(url).path.rstrip("/") if isinstance(url, str) else None
                # A clinic/location page identifies its branch ahead of a
                # network's generic Organization entry elsewhere in the graph.
                score = (10 if node_path == page_path else 1 if node_path == "" else 0)
                identities.append((score, identity))
    if identities:
        return max(identities, key=lambda item:item[0])[1]
    tag = soup.find("meta", attrs={"property": "og:site_name"})
    if tag and useful_name(tag.get("content", "")):
        return {"business_name": tag["content"].strip(), "identity_source": "og:site_name"}
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    return {"business_name": search_name({"title": title, "domain": domain}),
            "identity_source": "title_or_domain"}
