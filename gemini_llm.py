"""Shared Gemini structured inference, with bounded retries and optional fallbacks."""
import base64
import os
import time
import itertools
import research_budget
import httpx
from dotenv import load_dotenv
from pydantic import ValidationError
from google import genai
from google.genai import types
from service_definitions import MENU_SERVICE_RULES

load_dotenv()
from gemini_models import GEMINI_MODEL, GEMINI_FALLBACK_MODEL, GEMINI_THIRD_MODEL
from gemini_errors import _is_temporary_gemini_error
from gemini_limits import limiter, handle_quota_error

class GeminiModelError(RuntimeError):
    pass


class GeminiBudgetError(GeminiModelError, research_budget.ResearchBudgetExceeded):
    pass

_client = None
_call_ids = itertools.count(1)
_unavailable_reason = ""


def reset_run():
    global _unavailable_reason
    _unavailable_reason = ""


def unavailable_reason():
    return _unavailable_reason

def ensure_ready():
    global _client
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key or key.startswith("gsk_"):
        raise GeminiModelError("Set GEMINI_API_KEY in .env to your Google AI Studio API key. A Groq key cannot authenticate with Gemini.")
    if _client is None:
        _client = genai.Client(api_key=key, vertexai=False, http_options=types.HttpOptions(
            timeout=60000, retry_options=types.HttpRetryOptions(attempts=1)))
    return _client

def structured_call(prompt, schema, images=None):
    global _unavailable_reason
    if _unavailable_reason:
        raise GeminiModelError(_unavailable_reason)
    client = ensure_ready()
    label = f"{schema.__name__} #{next(_call_ids)}"
    models = list(dict.fromkeys(m.strip() for m in
        (GEMINI_MODEL, GEMINI_FALLBACK_MODEL, GEMINI_THIRD_MODEL) if m and m.strip()))
    contents = [MENU_SERVICE_RULES + "\n" + prompt]
    for image in images or []:
        mime = "image/png"
        if image.startswith("data:"):
            header, image = image.split(",", 1)
            mime = header[5:].split(";")[0]
        contents.append(types.Part.from_bytes(data=base64.b64decode(image), mime_type=mime))
    last_error = None
    for attempt in range(3):
        model = models[min(attempt, len(models) - 1)]
        try:
            research_budget.check()
            limiter.acquire()
            left = research_budget.check()
        except TimeoutError as error:
            error_type = GeminiBudgetError if isinstance(error, research_budget.ResearchBudgetExceeded) else GeminiModelError
            raise error_type(f"Gemini [{label}] stopped: {error}") from error
        try:
            config = {"temperature":0,"response_mime_type":"application/json","response_schema":schema}
            if left is not None:
                config["http_options"] = {"timeout":max(1, int(min(60, left)*1000)),
                                          "retry_options":{"attempts":1}}
            response = client.models.generate_content(model=model, contents=contents, config=config)
            if isinstance(response.parsed, schema):
                return response.parsed
            return schema.model_validate_json(response.text or "")
        except Exception as error:
            last_error = error
            retryable = (_is_temporary_gemini_error(error) or
                         isinstance(error, (httpx.TransportError, ValidationError)))
            if getattr(error, "code", None) == 404 and attempt + 1 < len(models):
                retryable = True
            if not retryable:
                break
            # Daily/zero-allocation quotas do not recover through short waits.
            quota_text = str(error).lower()
            if getattr(error, "code", None) == 429 and any(term in quota_text for term in (
                    "perday", "per_day", "per day", "daily", "limit: 0", "limit:0",
                    "quotavalue: 0", '"quotavalue": "0"')):
                _unavailable_reason = "Gemini daily quota or free-tier allocation exhausted. Research stopped; retry after the quota resets."
                break
            handle_quota_error(error)
            if attempt < 2:
                next_model = models[min(attempt + 1, len(models) - 1)]
                status = getattr(error, "code", type(error).__name__)
                delay = 2 ** (attempt+1)
                remaining = research_budget.remaining()
                if remaining is not None and remaining <= delay:
                    raise GeminiBudgetError(f"Gemini [{label}] stopped: research time budget exhausted before retry") from error
                print(f"\n[{label}] {model} failed ({status}); retrying with {next_model} (attempt {attempt + 2}/3)...", flush=True)
                time.sleep(delay)
    code = getattr(last_error, "code", None)
    if code in (401, 403):
        detail = "Check the API key and model access in Google AI Studio."
        _unavailable_reason = detail
    elif code == 429:
        detail = "Your API quota is exhausted. Wait for it to reset or check AI Studio quotas."
    elif code == 404:
        detail = "The configured model is unavailable. Check GEMINI_MODEL in .env."
    else:
        detail = "The model could not complete this request. Please try again shortly."
    raise GeminiModelError(f"Gemini [{label}] failed{f' (HTTP {code})' if code else ''}. {detail}") from last_error
