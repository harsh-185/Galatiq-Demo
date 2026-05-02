"""Critic: LLM agent that reflects on the rule-based approval decision.

Single-pass critique loop. The critic receives (invoice, report, decision) and
either confirms the rule-based decision or recommends a *more conservative*
override — escalate one tier, or downgrade auto_approved → pending_human. The
critic is NEVER allowed to weaken a rule-based reject (rules own hard rejects)
or auto-approve a needs_review case (only the deterministic engine can).

If the critic disagrees, the pipeline applies the override exactly once. The
revised decision is what gets recorded in approval_log. This satisfies the
spec's "reflection or critique loop" requirement without risking unbounded
iteration.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Literal

from pydantic import BaseModel, Field

from galatiq.agents._llm_helpers import run_llm_agent
from galatiq.agents.approval import ApprovalDecision, ApprovalStatus
from galatiq.agents.validation import ValidationReport
from galatiq.models.invoice import Invoice

CritiqueAction = Literal[
    "confirm",
    "escalate_to_manager",
    "escalate_to_director",
    "escalate_to_cfo",
    "downgrade_to_pending_human",
]

_SYSTEM = """\
You are an approval critic. A deterministic policy engine has produced a tier-
based decision on an invoice. Your job is to *reflect* on that decision, not
re-decide. Confirm the engine when it looks right; escalate or downgrade only
when there is a specific concern the engine could not see.

You may produce ONE of these actions:
- "confirm": the decision is appropriate. (Default when in doubt.)
- "escalate_to_manager" / "escalate_to_director" / "escalate_to_cfo": the
  decision is too lenient — bump approver one or more tiers. Only use this
  when there is a vendor-relationship, fraud-pattern, or aggregate-exposure
  concern the rule engine could not have known about.
- "downgrade_to_pending_human": the engine auto-approved, but you spotted a
  reason a human should still review (subtle anomaly, multiple soft warnings
  in concert, vendor history concern).

Hard constraints:
- NEVER recommend auto-approval. The rule engine owns that path.
- NEVER override a "rejected" decision. Rules own hard rejects.
- Cite at least one specific finding code or invoice fact in your rationale.
- Default to "confirm" unless you have a concrete escalation reason.
"""


class Critique(BaseModel):
    action: CritiqueAction = Field(default="confirm")
    rationale: str = Field(default="No override needed.")


def _fallback() -> Critique:
    return Critique(action="confirm", rationale="critic agent unavailable; deferring to rule engine")


_TIER_BY_ACTION: dict[CritiqueAction, tuple[str, str]] = {
    "escalate_to_manager":  ("manager",  "TIER-MGR"),
    "escalate_to_director": ("director", "TIER-DIR"),
    "escalate_to_cfo":      ("cfo",      "TIER-CFO"),
}


def critique(
    invoice: Invoice,
    report: ValidationReport,
    decision: ApprovalDecision,
) -> tuple[Critique, str | None]:
    if decision.status == "rejected":
        # Rules own rejects; skip the critique entirely.
        return Critique(action="confirm", rationale="rejection is owned by rule engine"), None

    payload = {
        "decision": {
            "status": decision.status,
            "approver_role": decision.approver_role,
            "policy_id": decision.policy_id,
            "total_usd": str(decision.total_usd),
            "deterministic_justification": decision.justification,
            "escalations": decision.escalations,
        },
        "verdict": report.verdict,
        "findings": [
            {"code": f.code, "severity": f.severity, "message": f.message}
            for f in report.findings
        ],
        "invoice": {
            "invoice_number": invoice.invoice_number,
            "vendor": invoice.vendor,
            "vendor_address": invoice.vendor_address,
            "currency": invoice.currency,
            "total": str(invoice.total),
            "line_item_count": len(invoice.line_items),
            "line_items": [
                {"item": li.item, "quantity": li.quantity, "unit_price": str(li.unit_price)}
                for li in invoice.line_items
            ],
        },
    }
    user = (
        "Reflect on the engine's decision. Confirm or escalate.\n\n"
        f"```json\n{__import__('json').dumps(payload, indent=2, default=str)}\n```"
    )
    return run_llm_agent(
        Critique,
        system=_SYSTEM,
        user=user,
        fallback=_fallback,
    )


def apply_override(decision: ApprovalDecision, crit: Critique) -> ApprovalDecision:
    """Return a revised decision per the critic's action. No-op on 'confirm'."""
    if crit.action == "confirm":
        return decision
    if decision.status == "rejected":
        return decision  # safety: rules own rejects
    if crit.action == "downgrade_to_pending_human":
        if decision.status != "auto_approved":
            return decision
        revised_role = decision.approver_role
        revised_policy = decision.policy_id
        # Pull the role from policy_id when downgrading from system → human in same tier.
        # If the engine matched TIER-AUTO (system), the next-up tier is manager.
        if decision.approver_role == "system":
            revised_role = "manager"
            revised_policy = "TIER-MGR"
        new_just = f"critic override: {crit.rationale}"
        return replace(
            decision,
            status="pending_human",
            approver_role=revised_role,
            policy_id=revised_policy,
            justification=new_just,
        )
    if crit.action in _TIER_BY_ACTION:
        role, policy = _TIER_BY_ACTION[crit.action]
        new_just = f"critic override: {crit.rationale}"
        return replace(
            decision,
            status="pending_human",
            approver_role=role,
            policy_id=policy,
            justification=new_just,
        )
    return decision
