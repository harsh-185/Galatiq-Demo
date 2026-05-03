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

import operator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from galatiq.agents._walkthrough import StageEvent, stage_event
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


def _emit(ev: StageEvent, **updates) -> dict:
    """Bundle a node's state update with its walkthrough event so LangGraph's
    operator.add reducer accumulates events across nodes."""
    return {"walkthrough": [ev], **updates}


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
    walkthrough: Annotated[list[StageEvent], operator.add]


# --- nodes --------------------------------------------------------------------


def _ingest_node(state: PipelineState) -> PipelineState:
    with stage_event("ingest") as ev:
        try:
            result = ingest(state["path"], allow_llm=True)
        except Exception as e:  # noqa: BLE001
            ev.status = "failed"
            ev.summary = f"{type(e).__name__}: {e}"
            return _emit(ev, errors=state.get("errors", []) + [f"ingestion: {type(e).__name__}: {e}"])
        inv = result.invoice
        ev.summary = (
            f"path={result.path_taken} (retries={result.llm_retries})  "
            f"vendor={inv.vendor!r}  total={inv.total} {inv.currency}  "
            f"line_items={len(inv.line_items)}"
        )
        if inv.ingestion_warnings:
            ev.details.append(f"{len(inv.ingestion_warnings)} ingestion warning(s)")
            for w in inv.ingestion_warnings[:3]:
                ev.details.append(f"  - {w.code}: {w.message}")
        return _emit(ev, ingestion=result)


def _pre_approval_node(state: PipelineState) -> PipelineState:
    """Always-on merged screener: an LLM examines every invoice for fraud
    findings, items to verify, and vendor onboarding signals. No deterministic
    skip — the LLM is part of the approval loop."""
    with stage_event("pre_approval_screener") as ev:
        if state.get("ingestion") is None:
            ev.status = "skipped"
            ev.summary = "no invoice (ingest failed upstream)"
            return _emit(ev)
        invoice = state["ingestion"].invoice
        db_path = state["db_path"]
        with connect(db_path) as conn:
            summary, findings, err, trace = pre_approval_screen(invoice, conn=conn)
        ev.tools_used = list(trace)
        ev.summary = (
            f"risk={summary.risk_severity}  fraud_findings={len(findings)}  "
            f"items_to_verify={len(summary.items_to_verify)}  "
            f"vendor_profile={'yes' if summary.vendor_profile else 'no'}"
        )
        if summary.risk_hypothesis:
            ev.details.append(f"hypothesis: {summary.risk_hypothesis}")
        for f in findings:
            ev.details.append(f"finding: [{f.severity}] {f.code} — {f.message}")
        if err:
            ev.details.append(f"⚠ fallback: {err}")
        update: PipelineState = {
            "pre_approval_summary": summary,
            "fraud_findings": findings,
            "pre_approval_tool_trace": trace,
        }
        if err:
            update["llm_agent_errors"] = state.get("llm_agent_errors", []) + [f"pre_approval_screener: {err}"]
        return _emit(ev, **update)


def _validate_node(state: PipelineState) -> PipelineState:
    with stage_event("validate") as ev:
        if state.get("ingestion") is None:
            ev.status = "skipped"
            return _emit(ev)
        invoice = state["ingestion"].invoice
        db_path = state["db_path"]
        fraud_findings = state.get("fraud_findings") or []

        with connect(db_path) as conn:
            report = validate(invoice, conn=conn)
            for f in fraud_findings:
                report.add(f)
            if fraud_findings:
                report.verdict = derive_verdict(report.findings)

            recorded_to_ledger = False
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
                recorded_to_ledger = True

        sev_count = report.by_severity()
        ev.summary = (
            f"verdict={report.verdict.upper()}  "
            f"errors={len(sev_count['error'])}  warns={len(sev_count['warn'])}  "
            f"info={len(sev_count['info'])}"
        )
        for f in report.findings[:5]:
            ev.details.append(f"[{f.severity}] {f.code} — {f.message}")
        if len(report.findings) > 5:
            ev.details.append(f"… and {len(report.findings) - 5} more findings")
        if recorded_to_ledger:
            ev.details.append("⛁ recorded to invoice_ledger")
        return _emit(ev, report=report)


def _approve_node(state: PipelineState) -> PipelineState:
    with stage_event("approve") as ev:
        if state.get("ingestion") is None or state.get("report") is None:
            ev.status = "skipped"
            return _emit(ev)
        invoice = state["ingestion"].invoice
        report = state["report"]
        db_path = state["db_path"]
        with connect(db_path) as conn:
            decision = approve(invoice, report, conn=conn)
        ev.summary = (
            f"{decision.status.upper()}  approver={decision.approver_role}  "
            f"policy={decision.policy_id or '—'}  total_usd=${decision.total_usd}"
        )
        if decision.escalations:
            ev.details.append(f"escalations: {', '.join(decision.escalations)}")
        return _emit(ev, pre_council_decision=decision, decision=decision)


def _council_node(state: PipelineState) -> PipelineState:
    """Always-on tier-scaled council. Reviewers run on every approval (except
    hard rule-engine rejects, where there's nothing to review)."""
    with stage_event("council") as ev:
        if state.get("ingestion") is None or state.get("decision") is None or state.get("report") is None:
            ev.status = "skipped"
            return _emit(ev)
        decision = state["decision"]
        if decision.status == "rejected":
            ev.status = "skipped"
            ev.summary = "skipped (rule engine rejected — rules own hard rejects)"
            return _emit(ev, council_profile=None, reviewer_opinions=[], reviewer_traces={}, council_skipped=True)

        invoice = state["ingestion"].invoice
        report = state["report"]
        profile = select_profile(decision.policy_id)
        summary = state.get("pre_approval_summary")
        db_path = state["db_path"]
        traces: dict[str, list[str]] = {}
        errors: list[str] = []

        # Run reviewers concurrently. Each opens its own short-lived sqlite
        # connection — sqlite is thread-safe for independent connections.
        def _run_one(reviewer_name: str):
            with connect(db_path) as conn:
                return reviewer_name, _REVIEWER_FUNCS[reviewer_name](
                    invoice,
                    report,
                    decision,
                    conn=conn,
                    pre_approval_summary=summary,
                    max_tool_loops=profile.max_tool_loops_per_reviewer,
                )

        results: dict[str, tuple] = {}
        if len(profile.reviewers) <= 1:
            # Sequential fast path for the lite profile (one reviewer).
            for name in profile.reviewers:
                _, r = _run_one(name)
                results[name] = r
        else:
            with ThreadPoolExecutor(max_workers=len(profile.reviewers)) as ex:
                for name, r in ex.map(_run_one, profile.reviewers):
                    results[name] = r

        # Re-assemble in the canonical reviewer order so output is deterministic.
        opinions: list[ReviewerOpinion] = []
        for reviewer_name in profile.reviewers:
            opinion, err, trace = results[reviewer_name]
            opinions.append(opinion)
            traces[reviewer_name] = trace
            if err:
                errors.append(f"{reviewer_name}_reviewer: {err}")

        ev.summary = (
            f"profile={profile.name}  reviewers={len(profile.reviewers)}  "
            f"max_tool_loops/reviewer={profile.max_tool_loops_per_reviewer}"
        )
        for op in opinions:
            ev.details.append(f"[{op.reviewer}/{op.severity}] {op.verdict}: {op.rationale}")
        for reviewer_name, trace in traces.items():
            for t in trace:
                ev.tools_used.append(f"{reviewer_name}.{t}")

        update: PipelineState = {
            "council_profile": profile,
            "reviewer_opinions": opinions,
            "reviewer_traces": traces,
            "council_skipped": False,
        }
        if errors:
            update["llm_agent_errors"] = state.get("llm_agent_errors", []) + errors
        return _emit(ev, **update)


def _aggregator_node(state: PipelineState) -> PipelineState:
    """Synthesize council opinions → final decision + audit narrative.

    Records the FINAL decision to ``approval_log``. When the council was
    skipped (clean small invoice or hard reject), falls into the deterministic
    aggregation path which preserves the rule decision.
    """
    with stage_event("aggregator") as ev:
        if state.get("ingestion") is None or state.get("decision") is None or state.get("report") is None:
            ev.status = "skipped"
            return _emit(ev)
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

        if (pre.status, pre.policy_id) != (final.status, final.policy_id):
            ev.summary = f"OVERRIDE  {pre.status} → {final.status}  ({final.approver_role}, {final.policy_id or '—'})"
        else:
            ev.summary = f"{final.status.upper()}  ({final.approver_role}, {final.policy_id or '—'})  ⛁ recorded"
        if narrative:
            ev.details.append(f"narrative: {narrative}")
        if err:
            ev.details.append(f"⚠ fallback: {err}")

        update: PipelineState = {"decision": final, "audit_narrative": narrative}
        if err:
            update["llm_agent_errors"] = state.get("llm_agent_errors", []) + [f"aggregator: {err}"]
        return _emit(ev, **update)


def _hitl_queue_node(state: PipelineState) -> PipelineState:
    """Write a human-review-queue entry for any pending_human decision."""
    with stage_event("hitl_queue") as ev:
        if state.get("decision") is None or state.get("ingestion") is None:
            ev.status = "skipped"
            return _emit(ev)
        decision = state["decision"]
        if decision.status != "pending_human":
            ev.status = "skipped"
            ev.summary = f"skipped (decision={decision.status}, no human review needed)"
            return _emit(ev)
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
        ev.summary = f"⛁ queued review id={review_id} for {decision.approver_role}"
        ev.details.append(f"resolve with: human resolve --id {review_id} --action approve|reject")
        return _emit(ev, human_review_id=review_id)


def _payment_guards_node(state: PipelineState) -> PipelineState:
    with stage_event("payment_guards") as ev:
        if state.get("ingestion") is None or state.get("decision") is None:
            ev.status = "skipped"
            return _emit(ev)
        decision = state["decision"]
        if decision.status != "auto_approved":
            ev.status = "skipped"
            ev.summary = f"skipped (decision={decision.status}, no payment to guard)"
            return _emit(ev)
        invoice = state["ingestion"].invoice
        with connect(state["db_path"]) as conn:
            vendor = lookup_vendor(conn, invoice.vendor)
            report, err = run_payment_guards(invoice, decision, vendor=vendor, conn=conn)
        ev.summary = (
            f"approved={report.approved}  banking={report.payment_method_status}  "
            f"review={report.review_action}"
        )
        ev.tools_used = list(report.review_trace)
        for b in report.blockers:
            ev.details.append(f"BLOCKER: {b}")
        for w in report.warnings:
            ev.details.append(f"warn: {w}")
        if report.suggested_rail:
            ev.details.append(f"suggested rail: {report.suggested_rail}")
        if report.near_dup_matches:
            ev.details.append(f"near-dup matches: {', '.join(report.near_dup_matches)}")
        update: PipelineState = {"payment_guard_report": report}
        if err:
            update["llm_agent_errors"] = state.get("llm_agent_errors", []) + [f"payment_guards: {err}"]
        return _emit(ev, **update)


def _pay_node(state: PipelineState) -> PipelineState:
    with stage_event("pay") as ev:
        if state.get("ingestion") is None or state.get("decision") is None:
            ev.status = "skipped"
            return _emit(ev)
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

        ev.summary = (
            f"{record.status.upper()}  rail={record.rail}  "
            f"ref={record.reference}  amount={record.amount_paid} {record.currency_paid}"
        )
        if record.scheduled_for:
            ev.details.append(f"scheduled_for: {record.scheduled_for}")
        if receipt_path:
            ev.details.append(f"⛁ receipt: {receipt_path}")
        for n in record.notes:
            ev.details.append(f"note: {n}")
        ev.details.append(f"⛁ payment_log row keyed on {record.reference}")

        return _emit(ev, payment=record, receipt_body=receipt_body)


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
        "walkthrough": [],
    }
    final: PipelineState = _GRAPH.invoke(initial)
    return final
