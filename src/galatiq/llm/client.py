"""Provider-agnostic chat-model factory. Default Grok-3 (xAI), with OpenAI fallback."""
from __future__ import annotations

import os
from typing import Any

_DEFAULT_PROVIDER = "grok"


class LLMUnavailable(RuntimeError):
    """Raised when the requested provider has no API key configured."""


def get_chat_model(*, temperature: float = 0.0, **kwargs: Any):
    provider = os.environ.get("LLM_PROVIDER", _DEFAULT_PROVIDER).lower()

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
