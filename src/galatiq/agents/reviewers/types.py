"""Shared types and aggregation logic for the approval council.

Each reviewer produces a ``ReviewerOpinion``. The aggregator combines:
  • the rule-based ``ApprovalDecision`` (pre-council)
  • the rule engine's verdict
  • each reviewer's opinion

…into a final ``ApprovalDecision`` using a *conservative-vote* policy: the
final outcome is the most conservative recommendation across all reviewers.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Literal

from pydantic import BaseModel, Field

from galatiq.agents.approval import ApprovalDecision

# Verdicts the reviewers can express. Conservative-ranked from least to most.
ReviewerVerdict = Literal[
    "approve",                 # the engine's decision is fine as-is
    "approve_with_notes",      # fine, but flag concerns in the audit trail
    "downgrade_to_human",      # too risky to auto-approve; bump to manager
    "escalate_one_tier",       # bump approver one tier (manager → director, etc.)
    "escalate_to_cfo",         # serious concern; jump straight to CFO
    "reject",                  # this should not be approved at all
]

_VERDICT_RANK: dict[ReviewerVerdict, int] = {
    "approve": 0,
    "approve_with_notes": 1,
    "downgrade_to_human": 2,
    "escalate_one_tier": 3,
    "escalate_to_cfo": 4,
    "reject": 5,
}

_TIER_ORDER = ["TIER-AUTO", "TIER-MGR", "TIER-DIR", "TIER-CFO"]
_ROLE_BY_TIER = {
    "TIER-AUTO": "system",
    "TIER-MGR": "manager",
    "TIER-DIR": "director",
    "TIER-CFO": "cfo",
}


class ReviewerOpinion(BaseModel):
    """Output of a single specialist reviewer."""
    reviewer: str = Field(description="Reviewer name (e.g. 'compliance', 'fraud').")
    verdict: ReviewerVerdict
    severity: Literal["low", "medium", "high"] = "low"
    rationale: str = Field(description="One- to two-sentence explanation citing specific findings.")
    concerns: list[str] = Field(
        default_factory=list,
        description="Short bullet list of concrete concerns to surface in the audit trail.",
    )


def _bump_tier(policy_id: str | None, by: int) -> tuple[str, str]:
    """Bump the approver tier ``by`` steps. Returns (new_policy_id, new_role)."""
    if policy_id not in _TIER_ORDER:
        return "TIER-CFO", _ROLE_BY_TIER["TIER-CFO"]
    idx = min(len(_TIER_ORDER) - 1, _TIER_ORDER.index(policy_id) + by)
    new_policy = _TIER_ORDER[idx]
    return new_policy, _ROLE_BY_TIER[new_policy]


def aggregate_opinions(
    decision: ApprovalDecision,
    opinions: list[ReviewerOpinion],
) -> tuple[ApprovalDecision, ReviewerVerdict]:
    """Produce the council's final decision via conservative-vote.

    Rules:
      • If any reviewer says ``reject``, the council rejects (rules still own
        actual hard-rule rejections, but the council can also reject for
        soft reasons).
      • Otherwise pick the most conservative verdict and apply it on top of
        the rule-based decision.
      • The council CANNOT auto-approve a needs_review case (rules own that
        gate) and CANNOT override a rule-based reject.

    Returns ``(final_decision, winning_verdict)``.
    """
    if decision.status == "rejected":
        return decision, "reject"

    if not opinions:
        return decision, "approve"

    worst = max(opinions, key=lambda o: _VERDICT_RANK[o.verdict])
    verdict = worst.verdict

    if verdict == "approve":
        return decision, verdict

    rationale = (
        f"council [{worst.reviewer}/{worst.severity}]: {worst.rationale}"
    )

    if verdict == "approve_with_notes":
        # Decision unchanged, just record the concerns in justification.
        return replace(decision, justification=rationale), verdict

    if verdict == "reject":
        return (
            replace(
                decision,
                status="rejected",
                approver_role="none",
                policy_id=None,
                justification=f"council reject: {worst.rationale}",
            ),
            verdict,
        )

    # Escalation paths.
    if decision.status == "auto_approved":
        # Any escalation downgrades auto → pending_human.
        if verdict == "downgrade_to_human":
            new_policy, new_role = _bump_tier(decision.policy_id, 1)
        elif verdict == "escalate_one_tier":
            new_policy, new_role = _bump_tier(decision.policy_id, 2)
        else:  # escalate_to_cfo
            new_policy, new_role = "TIER-CFO", "cfo"
        return (
            replace(
                decision,
                status="pending_human",
                approver_role=new_role,
                policy_id=new_policy,
                justification=rationale,
            ),
            verdict,
        )

    # Already pending_human — escalate the tier.
    bump = {"escalate_one_tier": 1, "escalate_to_cfo": 99, "downgrade_to_human": 0}[verdict]
    if bump == 0:
        return replace(decision, justification=rationale), verdict
    new_policy, new_role = _bump_tier(decision.policy_id, bump)
    return (
        replace(
            decision,
            approver_role=new_role,
            policy_id=new_policy,
            justification=rationale,
        ),
        verdict,
    )
