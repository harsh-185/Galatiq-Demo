"""Per-stage event log for CLI walkthroughs.

Each pipeline node wraps its work with ``stage_event(state, name)`` which:
  • times the stage
  • resets and reads the LLM-call counter
  • appends a structured event to ``state["walkthrough"]``

The CLI's ``pay`` command formats the resulting list into a human-readable
play-by-play of the run with timing, LLM cost, and one-line per-stage
summaries. Same data is consumable by the dashboard if/when it wants to
render the walkthrough too.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, Literal

from galatiq.agents._llm_helpers import (
    get_llm_call_count,
    reset_llm_call_counter,
)


StageStatus = Literal["completed", "skipped", "failed"]


@dataclass
class StageEvent:
    name: str
    status: StageStatus = "completed"
    summary: str = ""
    details: list[str] = field(default_factory=list)
    duration_ms: float = 0.0
    llm_calls: int = 0
    tools_used: list[str] = field(default_factory=list)


@contextmanager
def stage_event(name: str) -> Iterator[StageEvent]:
    """Time a pipeline-node block and yield a StageEvent.

    The node fills in ``status`` / ``summary`` / ``details`` / ``tools_used``
    on the yielded object. ``duration_ms`` and ``llm_calls`` are populated
    automatically. The CALLER is responsible for including the event in its
    return dict (typically as ``{"walkthrough": [ev], ...}``) so LangGraph's
    ``operator.add`` reducer can append it across nodes.
    """
    event = StageEvent(name=name)
    reset_llm_call_counter()
    started = time.perf_counter()
    try:
        yield event
    except Exception as e:  # noqa: BLE001
        event.status = "failed"
        event.summary = f"{type(e).__name__}: {e}"
        raise
    finally:
        event.duration_ms = (time.perf_counter() - started) * 1000
        event.llm_calls = get_llm_call_count()


def render_walkthrough(events: list[StageEvent]) -> str:
    """Pretty CLI rendering of the per-stage walkthrough."""
    lines: list[str] = []
    width = 60
    lines.append("━" * width)
    lines.append("Pipeline walkthrough")
    lines.append("━" * width)

    for i, ev in enumerate(events, 1):
        marker = {
            "completed": "✓",
            "skipped":   "⊘",
            "failed":    "✗",
        }[ev.status]
        head = f"[{i}] {ev.name:<22} [{marker}] {ev.duration_ms:>7.1f} ms"
        if ev.llm_calls:
            head += f"  ·  {ev.llm_calls} LLM call{'s' if ev.llm_calls != 1 else ''}"
        if ev.tools_used:
            head += f"  ·  {len(ev.tools_used)} tool{'s' if len(ev.tools_used) != 1 else ''}"
        lines.append(head)
        if ev.summary:
            lines.append(f"     {ev.summary}")
        for detail in ev.details:
            lines.append(f"     {detail}")
        if ev.tools_used:
            for t in ev.tools_used[:5]:
                lines.append(f"     tool → {t}")
            if len(ev.tools_used) > 5:
                lines.append(f"     tool → … and {len(ev.tools_used) - 5} more")

    return "\n".join(lines)


def summary_stats(events: list[StageEvent]) -> dict[str, float | int]:
    """Aggregate per-run metrics across all stage events."""
    return {
        "total_ms": sum(e.duration_ms for e in events),
        "total_llm_calls": sum(e.llm_calls for e in events),
        "total_tool_calls": sum(len(e.tools_used) for e in events),
        "stages_completed": sum(1 for e in events if e.status == "completed"),
        "stages_skipped": sum(1 for e in events if e.status == "skipped"),
        "stages_failed": sum(1 for e in events if e.status == "failed"),
    }
