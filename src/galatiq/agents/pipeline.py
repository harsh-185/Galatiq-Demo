"""End-to-end LangGraph pipeline.

Topology (linear with conditional branches):

    START → ingest → fraud_screener → validate → [supervisor]
                                                    ↙   ↓   ↘
                                          vendor_onb invest direct
                                                    ↘   ↓   ↙
                                                     approve → justifier → pay → END

The deterministic agents (ingest, validate, approve, pay) own routing decisions.
The LLM specialists (fraud_screener, vendor_onboarding, investigator, justifier)
add advisory context: extra findings, risk assessments, vendor profiles, and
human-readable justifications. Every LLM agent has a deterministic fallback so
the pipeline never breaks.

This module is the *only* place that writes to the reference DB or filesystem.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from galatiq.agents.approval import ApprovalDecision, approve
from galatiq.agents.fraud_screener import screen as fraud_screen
from galatiq.agents.ingestion import IngestionResult, ingest
from galatiq.agents.investigator import RiskAssessment, assess as investigate
from galatiq.agents.justifier import Justification, write_justification
from galatiq.agents.payment import PaymentRecord, pay
from galatiq.agents.validation import (
    Finding,
    ValidationReport,
    derive_verdict,
    validate,
)
from galatiq.agents.vendor_onboarding import VendorProfile, draft_profile
from galatiq.db import (
    DEFAULT_DB_PATH,
    STATUS_NEW,
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
    fraud_findings: list[Finding]
    report: ValidationReport | None
    risk_assessment: RiskAssessment | None
    vendor_profile: VendorProfile | None
    decision: ApprovalDecision | None
    llm_justification: Justification | None
    payment: PaymentRecord | None
    receipt_body: str | None
    errors: list[str]
    llm_agent_errors: list[str]


# --- nodes --------------------------------------------------------------------


def _ingest_node(state: PipelineState) -> PipelineState:
    try:
        result = ingest(state["path"], allow_llm=True)
    except Exception as e:  # noqa: BLE001
        return {"errors": state.get("errors", []) + [f"ingestion: {type(e).__name__}: {e}"]}
    return {"ingestion": result}


def _fraud_screen_node(state: PipelineState) -> PipelineState:
    if state.get("ingestion") is None:
        return {}
    invoice = state["ingestion"].invoice
    db_path = state["db_path"]
    with connect(db_path) as conn:
        findings, err = fraud_screen(invoice, conn=conn)
    update: PipelineState = {"fraud_findings": findings}
    if err:
        update["llm_agent_errors"] = state.get("llm_agent_errors", []) + [f"fraud_screener: {err}"]
    return update


def _validate_node(state: PipelineState) -> PipelineState:
    if state.get("ingestion") is None:
        return {}
    invoice = state["ingestion"].invoice
    db_path = state["db_path"]
    fraud_findings = state.get("fraud_findings") or []

    with connect(db_path) as conn:
        report = validate(invoice, conn=conn)
        for f in fraud_findings:
            report.add(f)
        if fraud_findings:
            report.verdict = derive_verdict(report.findings)

        # Side-effect: record to ledger only when the (post-merge) verdict is pass
        # AND the invoice isn't already there (record_invoice uses INSERT OR REPLACE,
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


def _vendor_onboarding_node(state: PipelineState) -> PipelineState:
    if state.get("ingestion") is None:
        return {}
    invoice = state["ingestion"].invoice
    db_path = state["db_path"]
    with connect(db_path) as conn:
        vendor = lookup_vendor(conn, invoice.vendor)
    if vendor is None:
        return {}
    profile, err = draft_profile(invoice, vendor)
    update: PipelineState = {"vendor_profile": profile}
    if err:
        update["llm_agent_errors"] = state.get("llm_agent_errors", []) + [f"vendor_onboarding: {err}"]
    return update


def _investigator_node(state: PipelineState) -> PipelineState:
    if state.get("ingestion") is None or state.get("report") is None:
        return {}
    invoice = state["ingestion"].invoice
    report = state["report"]
    db_path = state["db_path"]
    with connect(db_path) as conn:
        assessment, err = investigate(invoice, report, conn=conn)
    update: PipelineState = {"risk_assessment": assessment}
    if err:
        update["llm_agent_errors"] = state.get("llm_agent_errors", []) + [f"investigator: {err}"]
    return update


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


def _justifier_node(state: PipelineState) -> PipelineState:
    if (
        state.get("ingestion") is None
        or state.get("report") is None
        or state.get("decision") is None
    ):
        return {}
    invoice = state["ingestion"].invoice
    report = state["report"]
    decision = state["decision"]
    justification, err = write_justification(invoice, report, decision)
    update: PipelineState = {"llm_justification": justification}
    if err:
        update["llm_agent_errors"] = state.get("llm_agent_errors", []) + [f"justifier: {err}"]
    return update


def _pay_node(state: PipelineState) -> PipelineState:
    if state.get("ingestion") is None or state.get("decision") is None:
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


# --- supervisor (conditional edge function) -----------------------------------


def _supervisor_route(state: PipelineState) -> str:
    """Decide whether to invoke a specialist agent before approval.

    Deterministic routing — the 'supervisor' role is structural, not LLM-driven.
    The intelligence lives in the specialists themselves.
    """
    if state.get("ingestion") is None or state.get("report") is None:
        return "approve"
    invoice = state["ingestion"].invoice
    report = state["report"]
    if report.verdict == "reject":
        return "approve"  # rejected — let the approval agent record the denial
    with connect(state["db_path"]) as conn:
        vendor = lookup_vendor(conn, invoice.vendor)
    if vendor is not None and vendor.status == STATUS_NEW:
        return "vendor_onboarding"
    if report.verdict == "needs_review":
        return "investigator"
    return "approve"


# --- graph --------------------------------------------------------------------


def _build_graph():
    g = StateGraph(PipelineState)
    g.add_node("ingest", _ingest_node)
    g.add_node("fraud_screener", _fraud_screen_node)
    g.add_node("validate", _validate_node)
    g.add_node("vendor_onboarding", _vendor_onboarding_node)
    g.add_node("investigator", _investigator_node)
    g.add_node("approve", _approve_node)
    g.add_node("justifier", _justifier_node)
    g.add_node("pay", _pay_node)

    g.add_edge(START, "ingest")
    g.add_edge("ingest", "fraud_screener")
    g.add_edge("fraud_screener", "validate")
    g.add_conditional_edges(
        "validate",
        _supervisor_route,
        {
            "approve": "approve",
            "vendor_onboarding": "vendor_onboarding",
            "investigator": "investigator",
        },
    )
    g.add_edge("vendor_onboarding", "approve")
    g.add_edge("investigator", "approve")
    g.add_edge("approve", "justifier")
    g.add_edge("justifier", "pay")
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
        "llm_agent_errors": [],
    }
    final: PipelineState = _GRAPH.invoke(initial)
    return final
