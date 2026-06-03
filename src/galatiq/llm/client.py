"""Provider-agnostic chat-model factory. Default Grok-3 (xAI), with OpenAI fallback."""
from __future__ import annotations

import os
from typing import Any

_DEFAULT_PROVIDER = "grok"

# Per-request timeout (seconds) and max retries applied to every LLM call.
# A slow provider night then degrades gracefully: the call raises on timeout,
# the agents' fallback machinery catches it, and the stage drops to its
# deterministic stub instead of hanging the whole run. Override via env.
_DEFAULT_TIMEOUT = 45.0
_DEFAULT_MAX_RETRIES = 1


def _timeout() -> float:
    try:
        return float(os.environ.get("GALATIQ_LLM_TIMEOUT", _DEFAULT_TIMEOUT))
    except (TypeError, ValueError):
        return _DEFAULT_TIMEOUT


def _max_retries() -> int:
    try:
        return int(os.environ.get("GALATIQ_LLM_MAX_RETRIES", _DEFAULT_MAX_RETRIES))
    except (TypeError, ValueError):
        return _DEFAULT_MAX_RETRIES


class LLMUnavailable(RuntimeError):
    """Raised when the requested provider has no API key configured."""


def get_chat_model(*, temperature: float = 0.0, **kwargs: Any):
    provider = os.environ.get("LLM_PROVIDER", _DEFAULT_PROVIDER).lower()

    # Default timeout/retries unless the caller overrode them explicitly.
    kwargs.setdefault("timeout", _timeout())
    kwargs.setdefault("max_retries", _max_retries())

    if provider == "grok":
        from langchain_xai import ChatXAI

        if not os.environ.get("XAI_API_KEY"):
            raise LLMUnavailable("XAI_API_KEY is not set; export it or set LLM_PROVIDER=openai")
        model = os.environ.get("GROK_MODEL", "grok-3")
        return ChatXAI(model=model, temperature=temperature, **kwargs)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        if not os.environ.get("OPENAI_API_KEY"):
            raise LLMUnavailable("OPENAI_API_KEY is not set")
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model, temperature=temperature, **kwargs)

    raise LLMUnavailable(f"unknown LLM_PROVIDER={provider!r}")
