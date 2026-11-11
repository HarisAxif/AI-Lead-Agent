"""Local-only Ollama inference shared by every research stage."""
import json
import os
import threading
import time
from urllib.parse import urlparse

import httpx
from service_definitions import MENU_SERVICE_RULES
from dotenv import load_dotenv
from pydantic import ValidationError

load_dotenv()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "900"))
OLLAMA_THINK = os.getenv("OLLAMA_THINK")
OLLAMA_CONTEXT_LENGTH = int(os.getenv("OLLAMA_CONTEXT_LENGTH", "16384"))
_GATE = threading.Semaphore(1)


class LocalModelError(RuntimeError):
    """An actionable local model error with no cloud fallback."""


def _check_local():
    host = urlparse(OLLAMA_BASE_URL).hostname
    if host not in ("localhost", "127.0.0.1", "::1"):
        raise LocalModelError("OLLAMA_BASE_URL must point to the local Ollama server.")
    if OLLAMA_MODEL.endswith(":cloud") or "cloud" in OLLAMA_MODEL.split(":")[-1]:
        raise LocalModelError("Use a downloaded local model, not an Ollama cloud model.")


def ensure_ready():
    _check_local()
    try:
        with httpx.Client(timeout=15, trust_env=False) as client:
            response = client.post(OLLAMA_BASE_URL + "/api/show", json={"model": OLLAMA_MODEL})
            response.raise_for_status()
            info = response.json()
    except httpx.ConnectError as error:
        raise LocalModelError("Ollama is not running. Start Ollama or run 'ollama serve'.") from error
    except httpx.HTTPStatusError as error:
        raise LocalModelError(f"Local model unavailable. Run: ollama pull {OLLAMA_MODEL}") from error
    except (httpx.HTTPError, ValueError) as error:
        raise LocalModelError(f"Could not check Ollama: {error}") from error
    if "vision" not in info.get("capabilities", []):
        raise LocalModelError(f"{OLLAMA_MODEL} cannot inspect screenshots. Choose a downloaded local vision-capable model.")
    return info


def structured_call(prompt, schema, images=None):
    _check_local()
    message = {"role": "user", "content": MENU_SERVICE_RULES + "\n" + prompt + "\nReturn only JSON matching this schema:\n" + json.dumps(schema.model_json_schema())}
    if images:
        message["images"] = images
    payload = {"model": OLLAMA_MODEL, "messages": [message],
               "format": schema.model_json_schema(), "stream": False,
               "keep_alive": "10m",
               "options": {"temperature": 0, "seed": 7,
                           "num_ctx": OLLAMA_CONTEXT_LENGTH}}
    if OLLAMA_THINK is not None:
        payload["think"] = OLLAMA_THINK.lower() in ("1", "true", "yes")
    with _GATE:
        for attempt in range(1, 3):
            try:
                with httpx.Client(timeout=OLLAMA_TIMEOUT_SECONDS, trust_env=False) as client:
                    response = client.post(OLLAMA_BASE_URL + "/api/chat", json=payload)
                    response.raise_for_status()
                    data = response.json()
                if data.get("error"):
                    raise LocalModelError(data["error"])
                if data.get("done_reason") == "length":
                    raise LocalModelError("Local model output was truncated; simplify the request.")
                return schema.model_validate_json(data["message"]["content"])
            except httpx.HTTPStatusError as error:
                if error.response.status_code < 500:
                    raise LocalModelError(f"Ollama rejected the request: {error.response.text[:400]}") from error
                last_error = error
            except (httpx.HTTPError, ValidationError, ValueError, KeyError) as error:
                last_error = error
            if attempt < 2:
                print("\nLocal model request failed. Retrying once; no cloud fallback.")
                time.sleep(1)
        raise LocalModelError(f"Local model failed after two attempts: {last_error}") from last_error
