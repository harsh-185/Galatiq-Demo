"""End-to-end LangGraph pipeline: ingest → validate → approve → pay.

This module is the *only* place that writes to the reference DB or filesystem.
Each node calls a pure agent and updates state; side-effects happen at the
edges (between nodes) inside this file.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from galatiq.agents.approval import ApprovalDecision, approve
from galatiq.agents.ingestion import IngestionResult, ingest
from galatiq.agents.payment import PaymentRecord, pay
from galatiq.agents.validation import ValidationReport, validate
from galatiq.db import (
    DEFAULT_DB_PATH,
    connect,
    has_invoice,
    lookup_vendor,
    record_approval,
    record_invoice,
    record_payment,
)
from galatiq.payments.receipt import DEFAULT_RECEIPT_DIR, render_receipt, write_receipt


class PipelineState(TypedDict, total=False):
    path: Path
    db_path: Path
    receipt_dir: Path
    ingestion: IngestionResult | None
    report: ValidationReport | None
    decision: ApprovalDecision | None
    payment: PaymentRecord | None
    receipt_body: str | None
    errors: list[str]


# --- nodes --------------------------------------------------------------------


def _ingest_node(state: PipelineState) -> PipelineState:
    try:
        result = ingest(state["path"], allow_llm=True)
    except Exception as e:  # noqa: BLE001
        return {"errors": state.get("errors", []) + [f"ingestion: {type(e).__name__}: {e}"]}
    return {"ingestion": result}


def _validate_node(state: PipelineState) -> PipelineState:
    if state.get("ingestion") is None:
        return {}
    invoice = state["ingestion"].invoice
    db_path = state["db_path"]
    with connect(db_path) as conn:
        report = validate(invoice, conn=conn)
        # Side-effect: record to ledger only when the invoice cleanly passes
        # AND it isn't already there (record_invoice uses INSERT OR REPLACE,
        # so guard with has_invoice to avoid silently overwriting).
        if report.verdict == "pass" and not has_invoice(
            conn, invoice.invoice_number, invoice.vendor
        ):
            record_invoice(
                conn,
                invoice_number=invoice.invoice_number,
                vendor=invoice.vendor,
                total=invoice.total,
                source_path=str(state["path"]),
            )
    return {"report": report}


def _approve_node(state: PipelineState) -> PipelineState:
    if state.get("ingestion") is None or state.get("report") is None:
        return {}
    invoice = state["ingestion"].invoice
    report = state["report"]
    db_path = state["db_path"]
    with connect(db_path) as conn:
        decision = approve(invoice, report, conn=conn)
        record_approval(
            conn,
            invoice_number=invoice.invoice_number,
            vendor=invoice.vendor,
            status=decision.status,
            approver_role=decision.approver_role,
            policy_id=decision.policy_id,
            total_usd=decision.total_usd,
        )
    return {"decision": decision}


def _pay_node(state: PipelineState) -> PipelineState:
    if (
        state.get("ingestion") is None
        or state.get("decision") is None
    ):
        return {}
    invoice = state["ingestion"].invoice
    decision = state["decision"]
    db_path = state["db_path"]
    receipt_dir = state.get("receipt_dir") or DEFAULT_RECEIPT_DIR

    with connect(db_path) as conn:
        vendor = lookup_vendor(conn, invoice.vendor)
        record = pay(invoice, decision, vendor=vendor)

        receipt_body: str | None = None
        receipt_path: str | None = None
        if record.status == "scheduled":
            receipt_body = render_receipt(invoice, decision, record)
            written = write_receipt(record, receipt_body, directory=Path(receipt_dir))
            receipt_path = str(written)
            record = replace(record, receipt_path=receipt_path)

        record_payment(
            conn,
            reference=record.reference,
            invoice_number=invoice.invoice_number,
            vendor=invoice.vendor,
            rail=record.rail,
            status=record.status,
            amount_usd=record.amount_usd,
            currency_paid=record.currency_paid,
            amount_paid=record.amount_paid,
            scheduled_for=record.scheduled_for.isoformat() if record.scheduled_for else None,
            receipt_path=receipt_path,
        )

    return {"payment": record, "receipt_body": receipt_body}


# --- graph --------------------------------------------------------------------


def _build_graph():
    g = StateGraph(PipelineState)
    g.add_node("ingest", _ingest_node)
    g.add_node("validate", _validate_node)
    g.add_node("approve", _approve_node)
    g.add_node("pay", _pay_node)
    g.add_edge(START, "ingest")
    g.add_edge("ingest", "validate")
    g.add_edge("validate", "approve")
    g.add_edge("approve", "pay")
    g.add_edge("pay", END)
    return g.compile()


_GRAPH = _build_graph()


def run_pipeline(
    path: str | Path,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    receipt_dir: str | Path = DEFAULT_RECEIPT_DIR,
) -> PipelineState:
    initial: PipelineState = {
        "path": Path(path),
        "db_path": Path(db_path),
        "receipt_dir": Path(receipt_dir),
        "errors": [],
    }
    final: PipelineState = _GRAPH.invoke(initial)
    return final
