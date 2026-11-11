import os
import sys
from pathlib import Path

# Dummy key so modules that build a Gemini client at import time can load.
os.environ.setdefault("GEMINI_API_KEY", "test-key")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Provider failures are run-scoped. Tests must not poison later test cases.
import pytest


@pytest.fixture(autouse=True)
def reset_gemini_run():
    import gemini_llm
    gemini_llm.reset_run()
    yield
    gemini_llm.reset_run()
