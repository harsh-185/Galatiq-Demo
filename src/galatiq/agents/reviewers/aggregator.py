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

_SYSTEM = """\
You are the APPROVAL COUNCIL AGGREGATOR. You see the rule engine's tier-based
decision plus opinions from up to three specialist reviewers (compliance,
fraud, policy). Your job is to:

1. Synthesize the reviewers' views into a single final decision.
2. Write a 2-3 sentence audit narrative explaining the outcome.

Output schema fields:
- final_status: "auto_approved" | "pending_human" | "rejected"
- final_approver_role: "system" | "manager" | "director" | "cfo" | "none"
- final_policy_id: "TIER-AUTO" | "TIER-MGR" | "TIER-DIR" | "TIER-CFO" | null
- audit_narrative: 2-3 sentence plain prose, citing specific reviewer
  opinions or finding codes.

Hard constraints:
- If the rule engine produced ``rejected``, you MUST keep it rejected.
  (Rules own hard rejections.)
- If the engine produced ``pending_human``, you may keep it pending or escalate
  the tier (manager → director → cfo) — but you MUST NOT auto-approve.
- If the engine produced ``auto_approved``, you may keep it OR downgrade to
  ``pending_human`` based on reviewer concerns.
- Pick the most conservative reviewer verdict by default; only relax if
  reviewers all say "approve".

Be specific. Cite at least one reviewer (e.g. "the fraud reviewer noted X")
or a finding code in the narrative.
"""


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
    """Run the LLM aggregator. Returns ``(final_decision, narrative, error)``."""
    payload = {
        "engine_decision": {
            "status": decision.status,
            "approver_role": decision.approver_role,
            "policy_id": decision.policy_id,
            "total_usd": str(decision.total_usd),
            "engine_justification": decision.justification,
            "escalations": decision.escalations,
        },
        "verdict": report.verdict,
        "findings": [
            {"code": f.code, "severity": f.severity, "message": f.message}
            for f in report.findings
        ],
        "reviewer_opinions": [o.model_dump() for o in opinions],
        "invoice_excerpt": {
            "invoice_number": invoice.invoice_number,
            "vendor": invoice.vendor,
            "currency": invoice.currency,
            "total": str(invoice.total),
            "line_item_count": len(invoice.line_items),
        },
    }
    user = (
        "Synthesize the reviewers' opinions into a final decision and audit "
        "narrative.\n\n"
        f"```json\n{json.dumps(payload, indent=2, default=str)}\n```"
    )
    aggregated, err = _llm_helpers.run_llm_agent(
        AggregatedDecision,
        system=_SYSTEM,
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
        # The aggregator tried to auto-approve a needs_review; force human review.
        final = ApprovalDecision(
            status="pending_human",
            approver_role=final.approver_role if final.approver_role != "system" else "manager",
            policy_id=final.policy_id or "TIER-MGR",
            total_usd=decision.total_usd,
            justification=f"safety override: needs_review cannot auto-approve — {final.justification}",
            escalations=decision.escalations,
        )
    return final, aggregated.audit_narrative, err
