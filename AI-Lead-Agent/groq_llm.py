"""Groq inference for every stage; GOOGLE_API_KEY is retained for compatibility."""
import copy
import itertools
import json
import os
import threading
import time
import re
import httpx
from dotenv import load_dotenv
from pydantic import ValidationError
from service_definitions import MENU_SERVICE_RULES

load_dotenv()
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "qwen/qwen3.8-27b")
GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b")
BASE_URL = "https://api.groq.com/openai/v1"
_GATE = threading.Semaphore(1)
_CALLS = itertools.count(1)
_TOKEN_WINDOWS = {}

def _reset_seconds(value):
    return sum(float(n) * (60 if unit == "m" else 1)
               for n, unit in re.findall(r"(\d+(?:\.\d+)?)(m|s)", value or ""))

def _pace(model, estimated_tokens):
    remaining, reset_at = _TOKEN_WINDOWS.get(model, (float("inf"), 0))
    delay = reset_at - time.monotonic()
    if remaining < estimated_tokens and delay > 0:
        if delay > 65:
            raise GroqModelError("Groq token quota requires a longer reset; wait before restarting research.")
        print(f"\nGroq token budget: waiting {delay:.0f}s for {model}.")
        while delay > 0:
            time.sleep(min(delay, 30))
            delay = reset_at - time.monotonic()

def _remember_budget(model, headers):
    try:
        remaining = int(headers["x-ratelimit-remaining-tokens"])
        delay = _reset_seconds(headers.get("x-ratelimit-reset-tokens", ""))
        _TOKEN_WINDOWS[model] = (remaining, time.monotonic() + delay)
    except (KeyError, ValueError):
        pass

class GroqModelError(RuntimeError):
    pass

def ensure_ready():
    key = os.getenv("GROQ_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise GroqModelError("Set your Groq key in GOOGLE_API_KEY (or GROQ_API_KEY) in .env.")
    return {"Authorization": "Bearer " + key}

def strict_schema(schema):
    result = copy.deepcopy(schema.model_json_schema())
    def visit(node):
        if isinstance(node, dict):
            node.pop("default", None)
            if node.get("type") == "object":
                node["additionalProperties"] = False
                node["required"] = list(node.get("properties", {}))
            for value in node.values(): visit(value)
        elif isinstance(node, list):
            for value in node: visit(value)
    visit(result)
    return result

def structured_call(prompt, schema, images=None):
    headers = ensure_ready()
    label = f"{schema.__name__} #{next(_CALLS)}"
    models = [GROQ_VISION_MODEL] if images else list(dict.fromkeys(
        m for m in (GROQ_MODEL, GROQ_FALLBACK_MODEL) if m))
    content = [{"type": "text", "text": MENU_SERVICE_RULES + "\n" + prompt}]
    for image in images or []:
        url = image if image.startswith("data:") else "data:image/png;base64," + image
        content.append({"type": "image_url", "image_url": {"url": url}})
    shape = strict_schema(schema)
    last_status = None
    with _GATE:
        for attempt in range(3):
            model = models[min(attempt, len(models)-1)]
            payload = {"model": model, "messages": [{"role": "user", "content": content}],
                       "temperature": 0, "max_completion_tokens": 768 if images else 2048,
                       "response_format": {"type": "json_schema", "json_schema": {
                           "name": schema.__name__, "strict": True, "schema": shape}}}
            if model.startswith("openai/gpt-oss"):
                payload["reasoning_effort"] = "low"
            elif model.startswith("qwen/"):
                payload["reasoning_effort"] = "none"
            estimated = (len(prompt) + len(MENU_SERVICE_RULES) + len(json.dumps(shape))) // 3 + (2048 * len(images or [])) + payload["max_completion_tokens"]
            _pace(model, estimated)
            delay = 2 ** (attempt+1)
            try:
                with httpx.Client(timeout=60) as client:
                    response = client.post(BASE_URL + "/chat/completions", headers=headers, json=payload)
                last_status = response.status_code
                _remember_budget(model, response.headers)
                if response.is_success:
                    data = response.json()
                    choice = data["choices"][0]
                    if choice.get("finish_reason") == "length":
                        raise GroqModelError("Groq output was truncated. Simplify the request or reduce evidence size.")
                    return schema.model_validate_json(choice["message"]["content"])
                if last_status in (401, 403):
                    raise GroqModelError("Groq rejected the API key or model access. Check your key and Groq console permissions.")
                if last_status in (413, 429):
                    message = response.json().get("error", {}).get("message", "Quota exceeded")
                    if last_status == 413 or "too large" in message.lower():
                        raise GroqModelError("Groq request exceeds the model token allowance: " + message[:300])
                    try: delay = max(delay, float(response.headers.get("retry-after", "10")))
                    except ValueError: delay = 10
                    if delay > 60:
                        raise GroqModelError("Groq free-tier quota reached. Wait for the reset shown in your Groq console.")
                elif last_status not in (408, 404, 500, 502, 503, 504):
                    # Do not include request headers, key, or full error response in logs.
                    message = response.json().get("error", {}).get("message", "Request rejected")
                    raise GroqModelError(f"Groq rejected {schema.__name__} (HTTP {last_status}): {message[:300]}")
            except (httpx.TransportError, ValidationError, KeyError, ValueError):
                pass
            if attempt < 2:
                next_model = models[min(attempt+1, len(models)-1)]
                print(f"\n[{label}] Groq {model} failed ({last_status or 'network/JSON'}); trying {next_model} in {delay:g}s (attempt {attempt+2}/3)...")
                time.sleep(delay)
    raise GroqModelError(f"Groq [{label}] failed after three attempts (HTTP {last_status or 'network/JSON'}). Try again shortly or check Groq quotas.")
