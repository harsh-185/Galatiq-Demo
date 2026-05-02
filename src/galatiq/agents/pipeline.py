"""LangGraph orchestrator: ingest -> validate -> approve.

The graph is a thin wrapper over the per-phase functions in
``galatiq.agents.{ingestion, validation, approval}``; each node remains usable
on its own. State is a TypedDict so LangGraph can merge partial updates.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, StateGraph

from galatiq.agents.approval import ApprovalDecision, approve
from galatiq.agents.ingestion import IngestionResult, ingest
from galatiq.agents.validation import ValidationReport, validate
from galatiq.models.invoice import Invoice


class PipelineState(TypedDict, total=False):
    # Inputs / config
    path: str
    allow_llm: bool
    db_path: str

    # Per-phase artifacts
    ingestion: IngestionResult
    invoice: Invoice
    validation: ValidationReport
    approval: ApprovalDecision

    # Terminal error info if a node bailed.
    error: str
    error_phase: str


def _ingest_node(state: PipelineState) -> PipelineState:
    try:
        result = ingest(state["path"], allow_llm=state.get("allow_llm", True))
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}", "error_phase": "ingestion"}
    return {"ingestion": result, "invoice": result.invoice}


def _validate_node(state: PipelineState) -> PipelineState:
    invoice = state.get("invoice")
    if invoice is None:
        return {}
    with sqlite3.connect(state["db_path"]) as conn:
        conn.row_factory = sqlite3.Row
        report = validate(invoice, conn=conn)
    return {"validation": report}


def _approve_node(state: PipelineState) -> PipelineState:
    invoice = state.get("invoice")
    report = state.get("validation")
    if invoice is None or report is None:
        return {}
    with sqlite3.connect(state["db_path"]) as conn:
        conn.row_factory = sqlite3.Row
        decision = approve(invoice, report, conn=conn, allow_llm=state.get("allow_llm", True))
    return {"approval": decision}


def _route_after_ingest(state: PipelineState) -> str:
    return "end" if state.get("error") else "validate"


def build_graph():
    g = StateGraph(PipelineState)
    g.add_node("ingest", _ingest_node)
    g.add_node("validate", _validate_node)
    g.add_node("approve", _approve_node)
    g.set_entry_point("ingest")
    g.add_conditional_edges("ingest", _route_after_ingest, {"validate": "validate", "end": END})
    g.add_edge("validate", "approve")
    g.add_edge("approve", END)
    return g.compile()


_GRAPH = None


def _graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def run_pipeline(
    path: str | Path,
    *,
    db_path: str | Path,
    allow_llm: bool = True,
) -> PipelineState:
    initial: PipelineState = {
        "path": str(path),
        "db_path": str(db_path),
        "allow_llm": allow_llm,
    }
    final: PipelineState = _graph().invoke(initial)  # type: ignore[assignment]
    return final
