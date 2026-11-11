"""Run a live, reproducible local-model evaluation without discovery/network research."""
import base64
import json
import struct
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import local_llm
from schemas import SearchRequest, FieldValidation
from pydantic import BaseModel

MODEL = sys.argv[1]
local_llm.OLLAMA_MODEL = MODEL
REPORT = Path(".cache/model-benchmarks") / (MODEL.replace(":", "_") + ".json")
REPORT.parent.mkdir(parents=True, exist_ok=True)
results = []

CASES = [
    ("conversational_gyms", "Hey, I’m a web developer. Can you get me around 10 gyms in Lahore who’d actually need a new website?", 10, "lahore", "gym", ("web",), False),
    ("law_firms_crm", "I need 10 law firms in London, UK that would benefit from CRM software.", 10, "london", "law", ("crm", "customer relationship"), False),
    ("restaurant_ar", "3 restaurants that will be a good match for the virtual menu from all over pakistan.", 3, "pakistan", "restaurant", ("virtual", "augmented", "ar menu"), False),
    ("schools_complaints", "Find 3 private schools in Pakistan that could benefit from a parent portal because parents publicly report missed announcements.", 3, "pakistan", "school", ("parent",), True),
    ("quantum_bakeries", "Find 10 local bakeries in Faisalabad that need quantum computing consulting.", 10, "faisalabad", "baker", ("quantum",), False),
]

if len(sys.argv) > 2 and sys.argv[2] == "--vision":
    CASES.append(("unseen_phrasing", "Could you track down a dozen dental clinics in Karachi? I would like to sell them an AI phone receptionist.", 12, "karachi", "dental", ("receptionist",), False))

original_client = local_llm.httpx.Client
# Disable optional hidden reasoning on Qwen to test practical CPU latency.
if MODEL.startswith("qwen3"):
    class NonThinkingClient(original_client):
        def post(self, url, **kwargs):
            if url.endswith("/api/chat"):
                kwargs["json"]["think"] = False
            return super().post(url, **kwargs)
    local_llm.httpx.Client = NonThinkingClient


def save():
    REPORT.write_text(json.dumps({"model": MODEL, "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")


def measure(name, prompt, schema, check, images=None):
    started = time.perf_counter()
    try:
        output = local_llm.structured_call(prompt, schema, images=images)
        checks = check(output)
        item = {"name": name, "seconds": round(time.perf_counter() - started, 2),
                "passed": all(checks.values()), "checks": checks, "output": output.model_dump()}
    except Exception as error:
        item = {"name": name, "seconds": round(time.perf_counter() - started, 2), "passed": False, "error": str(error)}
    results.append(item)
    save()
    print(json.dumps(item, ensure_ascii=False), flush=True)


print("Benchmark:", MODEL, local_llm.ensure_ready().get("capabilities"), flush=True)
for name, message, count, location, industry, service_terms, has_criteria in CASES:
    prompt = """Extract the user's B2B lead-search request into SearchRequest.
Preserve the city/region as location, not its ISO country code. Interpret ordinary
conversational language. number_of_leads is the requested business count; around
N means N. service is what the user offers or asks businesses to adopt.
Criteria contain ONLY extra explicitly stated conditions. Do not invent website
problems or buying intent. 'Need [service]' merely identifies the service.
Return null for unknown fields. country_code is an ISO two-letter code.
USER REQUEST:
""" + message
    def check(output, count=count, location=location, industry=industry, service_terms=service_terms, has_criteria=has_criteria):
        data = output.model_dump()
        return {"country_code": data["country_code"] == ("GB" if location == "london" else "PK"),
                "count": data["number_of_leads"] == count,
                "location": location in (data["location"] or "").lower(),
                "industry": industry in (data["industry"] or "").lower(),
                "service": any(term in (data["service"] or "").lower() for term in service_terms),
                "criteria": (any("announcement" in c.lower() for c in data["criteria"]) if has_criteria else not data["criteria"])}
    measure(name, prompt, SearchRequest, check)

measure("service_mismatch", """Assess service/buyer fit. Ordinary local bakeries in Faisalabad
are being targeted for quantum computing consulting. No specialized research or
quantum use case is supplied. Is that a credible practical fit? Do not invent
advanced workloads. Return FieldValidation valid=false for a mismatch with a
concise reason; uncommon but credible use cases may be valid.""", FieldValidation,
        lambda output: {"mismatch_rejected": not output.valid})

class EvidenceCheck(BaseModel):
    only_phone_supported: bool
    crm_absence_supported: bool
    photography_supported: bool
    explanation: str

measure("evidence_meaning", """Check whether supplied public evidence establishes each claim.
Evidence: a law firm lists both a phone number and an email address on its contact
page. No CRM widget is visible in the provided website text. A restaurant page
says 'Delicious layers of pasta and melted cheese'; no images are supplied.
Claims: the law firm offers ONLY phone contact; the firm has NO internal CRM;
the restaurant has professional food photography. Evaluate strictly, without
assuming absence from things not shown.""", EvidenceCheck,
        lambda output: {"phone_exclusivity": not output.only_phone_supported,
                        "internal_crm": not output.crm_absence_supported,
                        "photo_quote": not output.photography_supported})

class VisionCheck(BaseModel):
    dominant_color: str

if len(sys.argv) > 2 and sys.argv[2] == "--vision":
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    width = height = 180
    pixels = b"".join(b"\x00" + bytes([255, 0, 0]) * width for _ in range(height))
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b"")
    measure("vision", "Name the dominant color of this image.", VisionCheck,
            lambda output: {"image_read": "red" in output.dominant_color.lower()},
            images=[base64.b64encode(png).decode()])
print("Complete:", sum(item["passed"] for item in results), "/", len(results), "Report:", REPORT, flush=True)
