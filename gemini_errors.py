"""Classification of retryable Gemini API failures."""

def _is_temporary_gemini_error(
    error: Exception,
):
    message = str(
        error
    ).lower()

    # SDK errors expose a numeric code; status text can contain underscores.
    if getattr(error, "code", None) in (408, 429, 500, 502, 503, 504):
        return True

    temporary_markers = (
        "504",
        "deadline_exceeded",
        "503",
        "unavailable",
        "service_unavailable",
        "high demand",
        "temporarily",
        "timeout",
        "timed out",
        "deadline exceeded",
        "429",
        "resource_exhausted",
        "rate limit",
    )

    return any(
        marker in message
        for marker in temporary_markers
    )

