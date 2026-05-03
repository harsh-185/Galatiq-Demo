"""End-to-end LangGraph pipeline (post-consolidation).

Topology (linear):

    START
      │
      ▼
    ingest                        ⚙ pure (LLM fallback only on parse failure)
      │
      ▼
    pre_approval_screener         🛠 ONE tool-using LLM (replaces fraud_screener
      │                             + investigator + vendor_onboarding)
      ▼
    validate                      ⚙ pure; merges fraud_findings; can flip pass→needs_review
      │
      ▼
    approve                       ⚙ pure (rule engine)
      │
      ▼
    council_or_skip               ◇ deterministic gate; skips council on
      │                             clean+TIER-AUTO+known-active vendor
    ├── (skip) ────────────────┐
    │                          │
    ▼                          ▼
    council                    │   🛠×N reviewers (tier-scaled; lite/std/deep/deepest)
      │                        │
      ▼                        ▼
    aggregator                 aggregator    🤖 LLM synth → final decision + narrative
      │                        │             ⛁ writes approval_log
      └────────────┬───────────┘
                   ▼
    hitl_queue                    ⚙ pure; ⛁ writes human_review_queue if pending_human
                   │
                   ▼
    payment_guards                ⚙ banking validator (det.) + 🛠 payment_review
                   │              (merged near-dup + critic; pre-filtered when
                   │               no ledger history exists)
                   ▼
    pay                           ⚙ pure; honors guard blockers; ⛁ writes payment_log + receipt
                   │
                   ▼
                  END

This module is the *only* place that writes to the reference DB or filesystem.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from galatiq.agents.approval import ApprovalDecision, approve
from galatiq.agents.ingestion import IngestionResult, ingest
from galatiq.agents.payment import PaymentRecord, pay
from galatiq.agents.payment_guards import PaymentGuardReport, run_payment_guards
from galatiq.agents.pre_approval_screener import PreApprovalSummary, screen as pre_approval_screen
from galatiq.agents.reviewers import ReviewerOpinion
from galatiq.agents.reviewers.aggregator import aggregate
from galatiq.agents.reviewers.compliance import review as compliance_review
from galatiq.agents.reviewers.fraud import review as fraud_review
from galatiq.agents.reviewers.policy import review as policy_review
from galatiq.agents.reviewers.profile import CouncilProfile, select_profile
from galatiq.agents.validation import (
    Finding,
    ValidationReport,
    derive_verdict,
    validate,
)
from galatiq.db import (
    DEFAULT_DB_PATH,
    STATUS_ACTIVE,
    connect,
    has_invoice,
    lookup_vendor,
    queue_for_human_review,
    record_approval,
    record_invoice,
    record_payment,
)
from galatiq.payments.receipt import DEFAULT_RECEIPT_DIR, render_receipt, write_receipt

_REVIEWER_FUNCS = {
    "compliance": compliance_review,
    "fraud": fraud_review,
    "policy": policy_review,
}


class PipelineState(TypedDict, total=False):
    path: Path
    db_path: Path
    receipt_dir: Path
    ingestion: IngestionResult | None
    pre_approval_summary: PreApprovalSummary | None
    pre_approval_tool_trace: list[str]
    fraud_findings: list[Finding]
    report: ValidationReport | None
    pre_council_decision: ApprovalDecision | None
    decision: ApprovalDecision | None
    council_profile: CouncilProfile | None
    council_skipped: bool
    reviewer_opinions: list[ReviewerOpinion]
    reviewer_traces: dict[str, list[str]]
    audit_narrative: str | None
    payment_guard_report: PaymentGuardReport | None
    human_review_id: int | None
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


def _pre_approval_node(state: PipelineState) -> PipelineState:
    """Single merged screener: fraud findings + items_to_verify + vendor profile."""
    if state.get("ingestion") is None:
        return {}
    invoice = state["ingestion"].invoice
    db_path = state["db_path"]
    with connect(db_path) as conn:
        summary, findings, err, trace = pre_approval_screen(invoice, conn=conn)
    update: PipelineState = {
        "pre_approval_summary": summary,
        "fraud_findings": findings,
        "pre_approval_tool_trace": trace,
    }
    if err:
        update["llm_agent_errors"] = state.get("llm_agent_errors", []) + [f"pre_approval_screener: {err}"]
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
    return {"pre_council_decision": decision, "decision": decision}


def _can_skip_council(state: PipelineState) -> bool:
    """Deterministic gate: skip the council only when EVERYTHING is clean.

    All of:
      • verdict == pass
      • engine matched TIER-AUTO
      • zero findings (rule + fraud)
      • vendor in table with status=active
    """
    decision = state.get("decision")
    report = state.get("report")
    summary = state.get("pre_approval_summary")
    ingestion = state.get("ingestion")
    if not (decision and report and ingestion):
        return False
    if decision.status != "auto_approved":
        return False
    if decision.policy_id != "TIER-AUTO":
        return False
    if report.findings:
        return False
    if summary and (summary.fraud_findings or summary.risk_severity not in ("none", "low") or summary.vendor_profile is not None):
        return False
    invoice = ingestion.invoice
    with connect(state["db_path"]) as conn:
        vendor = lookup_vendor(conn, invoice.vendor)
    return vendor is not None and vendor.status == STATUS_ACTIVE


def _council_node(state: PipelineState) -> PipelineState:
    """Run the tier-scaled council, OR skip when deterministic gate allows."""
    if state.get("ingestion") is None or state.get("decision") is None or state.get("report") is None:
        return {}
    decision = state["decision"]
    if decision.status == "rejected":
        # Hard rejects skip the council to save LLM compute.
        return {"council_profile": None, "reviewer_opinions": [], "reviewer_traces": {}, "council_skipped": True}

    if _can_skip_council(state):
        return {"council_profile": None, "reviewer_opinions": [], "reviewer_traces": {}, "council_skipped": True}

    invoice = state["ingestion"].invoice
    report = state["report"]
    profile = select_profile(decision.policy_id)
    summary = state.get("pre_approval_summary")
    db_path = state["db_path"]
    opinions: list[ReviewerOpinion] = []
    traces: dict[str, list[str]] = {}
    errors: list[str] = []

    with connect(db_path) as conn:
        for reviewer_name in profile.reviewers:
            review_fn = _REVIEWER_FUNCS[reviewer_name]
            opinion, err, trace = review_fn(
                invoice,
                report,
                decision,
                conn=conn,
                pre_approval_summary=summary,
                max_tool_loops=profile.max_tool_loops_per_reviewer,
            )
            opinions.append(opinion)
            traces[reviewer_name] = trace
            if err:
                errors.append(f"{reviewer_name}_reviewer: {err}")

    update: PipelineState = {
        "council_profile": profile,
        "reviewer_opinions": opinions,
        "reviewer_traces": traces,
        "council_skipped": False,
    }
    if errors:
        update["llm_agent_errors"] = state.get("llm_agent_errors", []) + errors
    return update


def _aggregator_node(state: PipelineState) -> PipelineState:
    """Synthesize council opinions → final decision + audit narrative.

    Records the FINAL decision to ``approval_log``. When the council was
    skipped (clean small invoice or hard reject), falls into the deterministic
    aggregation path which preserves the rule decision.
    """
    if state.get("ingestion") is None or state.get("decision") is None or state.get("report") is None:
        return {}
    invoice = state["ingestion"].invoice
    report = state["report"]
    pre = state.get("pre_council_decision") or state["decision"]
    opinions = state.get("reviewer_opinions") or []

    final, narrative, err = aggregate(invoice, report, pre, opinions)

    db_path = state["db_path"]
    with connect(db_path) as conn:
        record_approval(
            conn,
            invoice_number=invoice.invoice_number,
            vendor=invoice.vendor,
            status=final.status,
            approver_role=final.approver_role,
            policy_id=final.policy_id,
            total_usd=final.total_usd,
        )

    update: PipelineState = {"decision": final, "audit_narrative": narrative}
    if err:
        update["llm_agent_errors"] = state.get("llm_agent_errors", []) + [f"aggregator: {err}"]
    return update


def _hitl_queue_node(state: PipelineState) -> PipelineState:
    """Write a human-review-queue entry for any pending_human decision."""
    if state.get("decision") is None or state.get("ingestion") is None:
        return {}
    decision = state["decision"]
    if decision.status != "pending_human":
        return {}
    invoice = state["ingestion"].invoice
    with connect(state["db_path"]) as conn:
        review_id = queue_for_human_review(
            conn,
            invoice_number=invoice.invoice_number,
            vendor=invoice.vendor,
            decision_status=decision.status,
            approver_role=decision.approver_role,
            policy_id=decision.policy_id,
            total_usd=decision.total_usd,
            source_path=str(state["path"]),
            narrative=state.get("audit_narrative"),
        )
    return {"human_review_id": review_id}


def _payment_guards_node(state: PipelineState) -> PipelineState:
    if state.get("ingestion") is None or state.get("decision") is None:
        return {}
    decision = state["decision"]
    if decision.status != "auto_approved":
        return {}
    invoice = state["ingestion"].invoice
    with connect(state["db_path"]) as conn:
        vendor = lookup_vendor(conn, invoice.vendor)
        report, err = run_payment_guards(invoice, decision, vendor=vendor, conn=conn)
    update: PipelineState = {"payment_guard_report": report}
    if err:
        update["llm_agent_errors"] = state.get("llm_agent_errors", []) + [f"payment_guards: {err}"]
    return update


def _pay_node(state: PipelineState) -> PipelineState:
    if state.get("ingestion") is None or state.get("decision") is None:
        return {}
    invoice = state["ingestion"].invoice
    decision = state["decision"]
    db_path = state["db_path"]
    receipt_dir = state.get("receipt_dir") or DEFAULT_RECEIPT_DIR
    guard_report = state.get("payment_guard_report")

    with connect(db_path) as conn:
        vendor = lookup_vendor(conn, invoice.vendor)
        record = pay(invoice, decision, vendor=vendor)

        if record.status == "scheduled" and guard_report is not None:
            if not guard_report.approved:
                record = replace(
                    record,
                    status="failed",
                    rail="none",
                    notes=record.notes + [f"guard blocker: {b}" for b in guard_report.blockers],
                )
            elif guard_report.suggested_rail and guard_report.suggested_rail != record.rail:
                new_rail = guard_report.suggested_rail
                record = replace(
                    record,
                    rail=new_rail,
                    reference=record.reference.rsplit("-", 1)[0] + "-" + new_rail.upper(),
                    notes=record.notes + [f"guard rail switch → {new_rail}"],
                )

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
    g.add_node("pre_approval", _pre_approval_node)
    g.add_node("validate", _validate_node)
    g.add_node("approve", _approve_node)
    g.add_node("council", _council_node)
    g.add_node("aggregator", _aggregator_node)
    g.add_node("hitl_queue", _hitl_queue_node)
    g.add_node("payment_guards", _payment_guards_node)
    g.add_node("pay", _pay_node)

    g.add_edge(START, "ingest")
    g.add_edge("ingest", "pre_approval")
    g.add_edge("pre_approval", "validate")
    g.add_edge("validate", "approve")
    g.add_edge("approve", "council")
    g.add_edge("council", "aggregator")
    g.add_edge("aggregator", "hitl_queue")
    g.add_edge("hitl_queue", "payment_guards")
    g.add_edge("payment_guards", "pay")
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
