"""Shared helpers for LLM-driven specialist agents.

Each specialist (fraud_screener, vendor_onboarding, investigator, justifier) calls
``run_llm_agent`` to get a typed structured-output result with a deterministic
fallback on any failure (no API key, network error, schema-validation retry
exhaustion). The pipeline never breaks because of an LLM agent.
"""
from __future__ import annotations

import os
from typing import Callable, TypeVar

from pydantic import BaseModel

from galatiq.llm.client import LLMUnavailable
from galatiq.llm.structured import extract_structured

T = TypeVar("T", bound=BaseModel)


def llm_agents_enabled() -> bool:
    """Are LLM-driven specialist agents active?

    Resolution order:
    1. ``GALATIQ_LLM_AGENTS`` env var (``1/true/yes`` enables, anything else disables).
    2. Default ON if any provider key is configured, else OFF.
    """
    flag = os.environ.get("GALATIQ_LLM_AGENTS")
    if flag is not None:
        return flag.strip().lower() in {"1", "true", "yes", "on"}
    return bool(os.environ.get("XAI_API_KEY") or os.environ.get("OPENAI_API_KEY"))


def run_llm_agent(
    schema: type[T],
    *,
    system: str,
    user: str,
    fallback: Callable[[], T],
) -> tuple[T, str | None]:
    """Run a structured LLM call with a guaranteed fallback.

    Returns ``(result, error_message)``. ``error_message`` is None on success or when
    LLM agents are disabled (in which case the fallback ran intentionally).
    """
    if not llm_agents_enabled():
        return fallback(), None
    try:
        result, _retries = extract_structured(schema, system=system, user=user)
        return result, None
    except LLMUnavailable as e:
        return fallback(), f"llm unavailable: {e}"
    except Exception as e:  # noqa: BLE001 — fall back on *any* LLM-side error
        return fallback(), f"{type(e).__name__}: {e}"
