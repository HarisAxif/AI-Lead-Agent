"""Ordered model selection shared by every Gemini stage."""
import os

GEMINI_MODEL = os.getenv("GEMINI_MODEL") or "gemini-3.5-flash-lite"
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL") or None
GEMINI_THIRD_MODEL = os.getenv("GEMINI_THIRD_MODEL") or None


def model_for_attempt(attempt, primary, secondary=None, third=None):
    models = []
    for model in (primary, secondary, third):
        if model and model.strip() and model.strip() not in models:
            models.append(model.strip())
    return models[min(attempt - 1, len(models) - 1)]
