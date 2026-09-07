"""Frontend runtime configuration."""

from __future__ import annotations

import os


API_BASE_URL = os.getenv("MATHLLM_API_URL", "http://localhost:8080/api").rstrip("/")
REQUEST_TIMEOUT = float(os.getenv("MATHLLM_REQUEST_TIMEOUT", "180"))
MEMORY_TRIGGER_TOKENS = int(os.getenv("MATHLLM_MEMORY_TRIGGER_TOKENS", "1200"))
RECENT_HISTORY_MESSAGES = int(os.getenv("MATHLLM_RECENT_HISTORY_MESSAGES", "6"))
MAX_CONTEXT_TOKENS = int(os.getenv("MATHLLM_MAX_CONTEXT_TOKENS", "2048"))
MAX_OUTPUT_TOKENS = int(os.getenv("MATHLLM_MAX_OUTPUT_TOKENS", "512"))
