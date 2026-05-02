"""Shared helpers for LLM-driven specialist agents.

Each specialist (fraud_screener, vendor_onboarding, investigator, justifier,
critic) calls ``run_llm_agent`` (or ``run_tool_using_agent``) to get a typed
structured-output result with a deterministic fallback on any failure (no API
key, network error, schema-validation retry exhaustion). The pipeline never
breaks because of an LLM agent.
"""
from __future__ import annotations

import os
from typing import Any, Callable, TypeVar

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel

from galatiq.llm.client import LLMUnavailable, get_chat_model
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


def run_tool_using_agent(
    schema: type[T],
    *,
    system: str,
    user: str,
    tools: list[Any],
    fallback: Callable[[], T],
    max_tool_loops: int = 4,
) -> tuple[T, str | None, list[str]]:
    """ReAct-style tool loop, then a structured-output call to summarize.

    Phase 1: bind ``tools`` to the LLM and let it call them iteratively (up to
    ``max_tool_loops`` rounds). Phase 2: feed the accumulated message history to
    a fresh structured-output call to produce the final ``schema`` instance.

    Returns ``(result, error_message, tool_trace)`` where ``tool_trace`` lists
    the tool calls in order (e.g. ``["lookup_vendor(name='Acme')", ...]``) so
    the orchestrator/UI can show what the agent investigated.
    """
    if not llm_agents_enabled():
        return fallback(), None, []
    try:
        tools_by_name = {t.name: t for t in tools}
        llm = get_chat_model().bind_tools(tools)
        messages: list[Any] = [SystemMessage(content=system), HumanMessage(content=user)]
        trace: list[str] = []

        for _ in range(max_tool_loops):
            ai: AIMessage = llm.invoke(messages)
            messages.append(ai)
            calls = getattr(ai, "tool_calls", None) or []
            if not calls:
                break
            for call in calls:
                name = call.get("name")
                args = call.get("args", {})
                trace.append(f"{name}({_short_args(args)})")
                tool = tools_by_name.get(name)
                if tool is None:
                    messages.append(
                        ToolMessage(content=f"unknown tool {name!r}", tool_call_id=call["id"])
                    )
                    continue
                try:
                    result = tool.invoke(args)
                except Exception as e:  # noqa: BLE001
                    result = f"tool error: {type(e).__name__}: {e}"
                messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

        # Phase 2: structured summary.
        structured = get_chat_model().with_structured_output(schema)
        messages.append(
            HumanMessage(
                content=(
                    "Based on the information you gathered above, produce the final "
                    f"{schema.__name__} JSON. Do not call any more tools."
                )
            )
        )
        final = structured.invoke(messages)
        if not isinstance(final, schema):
            final = schema.model_validate(final)
        return final, None, trace
    except LLMUnavailable as e:
        return fallback(), f"llm unavailable: {e}", []
    except Exception as e:  # noqa: BLE001
        return fallback(), f"{type(e).__name__}: {e}", []


def _short_args(args: dict[str, Any]) -> str:
    parts = []
    for k, v in args.items():
        v_short = repr(v) if not isinstance(v, str) or len(v) <= 40 else repr(v[:37] + "...")
        parts.append(f"{k}={v_short}")
    return ", ".join(parts)
