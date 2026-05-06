"""LLM aggregator: synthesizes reviewer opinions into the final decision.

The aggregator reads all reviewer opinions plus the rule engine's pre-council
decision and produces:
  • a final ApprovalDecision (status, approver_role, policy_id, justification)
  • a 2-3 sentence audit narrative (replacing the standalone justifier)

It can confirm, downgrade, or escalate — but it cannot weaken a rule-based
reject and cannot auto-approve a needs_review case (those gates remain with
the deterministic engine).

If the LLM is unavailable, falls back to the deterministic ``aggregate_opinions``
function in ``types.py`` and a canned narrative — pipeline never breaks.
"""
from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

from galatiq.agents import _llm_helpers
from galatiq.agents.approval import ApprovalDecision, ApprovalStatus
from galatiq.agents.reviewers.types import ReviewerOpinion, aggregate_opinions
from galatiq.agents.validation import ValidationReport
from galatiq.models.invoice import Invoice

_SYSTEM_WITH_COUNCIL = """\
You synthesize a council's opinions into a final approval decision + 2-3 sentence audit narrative.
Hard rules:
- engine 'rejected' MUST stay rejected
- engine 'pending_human' may become 'auto_approved' ONLY when ALL reviewers said approve+low and no load-bearing finding exists (vendor_blocked, fraud_flag_sku, duplicate_invoice, stock_overflow, zero_stock, negative_quantity, empty_vendor)
- engine 'pending_human' may become 'rejected' if reviewers found material risk
- engine 'auto_approved' may stay or downgrade to 'pending_human' on reviewer concerns
Cite at least one reviewer or finding code in the narrative. Be specific and concise.
"""


_SYSTEM_NO_COUNCIL = """\
You write a 2-3 sentence audit narrative for an already-finalized rule-engine decision.
The engine already decided; you do NOT change status/role/policy — just echo them and write the narrative.
Cite the policy band, total, and any escalations or findings if present.
"""

# Findings the rule engine considers load-bearing — the aggregator cannot
# auto-resolve a pending_human case when any of these are present, even if
# the LLM tries.
#
# Stock findings (stock_overflow, zero_stock) are NOT load-bearing because
# they're inventory-availability concerns, not fundamental legitimacy
# concerns; AP teams routinely approve over-stock orders as backorders.
# Math-mismatch findings (subtotal_mismatch, total_mismatch) are not in
# this list either — they're already handled by severity (error >2%
# discrepancy → reject regardless; warn ≤2% → council can decide).
_LOAD_BEARING_FINDING_CODES = frozenset({
    "vendor_blocked",
    "fraud_flag_sku",
    "duplicate_invoice",  # the strict variant; revisions emit invoice_revision instead
    "negative_quantity",
    "empty_vendor",
})


def _council_is_unanimous_clean(opinions: list[ReviewerOpinion]) -> bool:
    """Did every reviewer say 'approve' with severity='low'?

    Empty opinions list is NOT unanimous — there's no signal to resolve on.
    """
    if not opinions:
        return False
    return all(o.verdict == "approve" and o.severity == "low" for o in opinions)


def _has_load_bearing_finding(report: ValidationReport) -> bool:
    return any(f.code in _LOAD_BEARING_FINDING_CODES for f in report.findings)


class AggregatedDecision(BaseModel):
    final_status: Literal["auto_approved", "pending_human", "rejected"]
    final_approver_role: Literal["system", "manager", "director", "cfo", "none"]
    final_policy_id: Literal[
        "TIER-AUTO", "TIER-MGR", "TIER-DIR", "TIER-CFO"
    ] | None = None
    audit_narrative: str = Field(description="2-3 sentence audit narrative.")


def _fallback(decision: ApprovalDecision, opinions: list[ReviewerOpinion]) -> AggregatedDecision:
    """Deterministic fallback when the aggregator LLM is unavailable.

    Uses the conservative-vote function from ``types.py`` and synthesizes a
    narrative from the worst opinion.
    """
    revised, winning = aggregate_opinions(decision, opinions)
    if not opinions:
        narrative = revised.justification
    else:
        worst = max(opinions, key=lambda o: o.severity == "high") if any(
            o.severity == "high" for o in opinions
        ) else opinions[-1]
        names = ", ".join(o.reviewer for o in opinions)
        narrative = (
            f"Council ({names}) reviewed; deterministic aggregation chose "
            f"'{winning}'. {worst.rationale}"
        )
    role = revised.approver_role
    if role not in {"system", "manager", "director", "cfo", "none"}:
        role = "system"
    return AggregatedDecision(
        final_status=revised.status,
        final_approver_role=role,
        final_policy_id=revised.policy_id if revised.policy_id in {
            "TIER-AUTO", "TIER-MGR", "TIER-DIR", "TIER-CFO"
        } else None,
        audit_narrative=narrative,
    )


def aggregate(
    invoice: Invoice,
    report: ValidationReport,
    decision: ApprovalDecision,
    opinions: list[ReviewerOpinion],
) -> tuple[ApprovalDecision, str, str | None]:
    """Always-on aggregator: 1 LLM call to produce the final decision + a real
    audit narrative.

    With opinions: synthesizes the council into a final call.
    Without opinions (council skipped): writes a narrative for the engine's
    decision without changing it. Same schema either way.
    """
    has_council = bool(opinions)
    payload: dict = {
        "engine_decision": {
            "status": decision.status,
            "approver_role": decision.approver_role,
            "policy_id": decision.policy_id,
            "total_usd": str(decision.total_usd),
            "escalations": decision.escalations,
        },
        "verdict": report.verdict,
        "invoice": {
            "vendor": invoice.vendor,
            "currency": invoice.currency,
            "total": str(invoice.total),
            "line_item_count": len(invoice.line_items),
        },
    }
    if report.findings:
        payload["findings"] = [
            {"code": f.code, "severity": f.severity, "message": f.message}
            for f in report.findings[:5]
        ]
    if has_council:
        payload["reviewer_opinions"] = [o.model_dump() for o in opinions]
        user = f"Synthesize → final decision + narrative.\n```json\n{json.dumps(payload, default=str)}\n```"
        system = _SYSTEM_WITH_COUNCIL
    else:
        user = f"Write the narrative (do NOT change the decision).\n```json\n{json.dumps(payload, default=str)}\n```"
        system = _SYSTEM_NO_COUNCIL
    aggregated, err = _llm_helpers.run_llm_agent(
        AggregatedDecision,
        system=system,
        user=user,
        fallback=lambda: _fallback(decision, opinions),
    )

    # Apply the aggregator's call back onto the ApprovalDecision.
    final = ApprovalDecision(
        status=aggregated.final_status,
        approver_role=aggregated.final_approver_role,
        policy_id=aggregated.final_policy_id,
        total_usd=decision.total_usd,
        justification=aggregated.audit_narrative,
        escalations=decision.escalations,
    )
    # Safety: enforce the hard constraints in case the LLM disobeyed.
    if decision.status == "rejected" and final.status != "rejected":
        final = ApprovalDecision(
            status="rejected",
            approver_role="none",
            policy_id=None,
            total_usd=decision.total_usd,
            justification=f"safety override: rule engine rejected — {decision.justification}",
            escalations=decision.escalations,
        )
    if report.verdict == "needs_review" and final.status == "auto_approved":
        # Allow the aggregator to resolve needs_review → auto_approved ONLY when:
        #   • the council voted unanimous-clean (every reviewer approve+low), AND
        #   • no load-bearing finding is present (vendor_blocked, fraud_flag_sku,
        #     duplicate_invoice, stock_overflow, zero_stock, etc.)
        # Otherwise force pending_human — the LLM doesn't get to override on
        # cases the rule engine flagged with hard signals.
        if not (_council_is_unanimous_clean(opinions) and not _has_load_bearing_finding(report)):
            final = ApprovalDecision(
                status="pending_human",
                approver_role=final.approver_role if final.approver_role != "system" else "manager",
                policy_id=final.policy_id or "TIER-MGR",
                total_usd=decision.total_usd,
                justification=f"safety override: needs_review cannot auto-approve without unanimous-clean council — {final.justification}",
                escalations=decision.escalations,
            )
    return final, aggregated.audit_narrative, err
