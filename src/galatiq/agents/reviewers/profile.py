"""Council profile: tier-scaled compute selector.

Maps the rule engine's policy band (TIER-AUTO / MGR / DIR / CFO) to the
*shape* of the council that reviews the case. Smaller invoices get a lite
single-reviewer pass; larger ones get the full council with deeper tool
budgets per reviewer.

This is the deterministic "meta-orchestrator" — it decides which agents and
how much compute to spend before any LLM call is made. Easy to elevate to an
LLM router later by swapping ``select_profile`` for an LLM call.
"""
from __future__ import annotations

from dataclasses import dataclass, field

ReviewerName = str  # one of "compliance", "fraud", "policy"


@dataclass(frozen=True)
class CouncilProfile:
    name: str
    reviewers: list[ReviewerName] = field(default_factory=list)
    max_tool_loops_per_reviewer: int = 3
    rationale: str = ""


_LITE = CouncilProfile(
    name="lite",
    reviewers=["fraud"],
    max_tool_loops_per_reviewer=2,
    rationale="small invoice; single-reviewer fraud check is sufficient",
)
_STANDARD = CouncilProfile(
    name="standard",
    reviewers=["compliance", "fraud", "policy"],
    max_tool_loops_per_reviewer=3,
    rationale="manager-tier invoice; full 3-reviewer council with standard tool budget",
)
_DEEP = CouncilProfile(
    name="deep",
    reviewers=["compliance", "fraud", "policy"],
    max_tool_loops_per_reviewer=4,
    rationale="director-tier invoice; deeper tool budget per reviewer",
)
_DEEPEST = CouncilProfile(
    name="deepest",
    reviewers=["compliance", "fraud", "policy"],
    max_tool_loops_per_reviewer=6,
    rationale="CFO-tier invoice; maximum scrutiny — extended tool budget per reviewer",
)


def select_profile(policy_id: str | None) -> CouncilProfile:
    """Pick a council profile based on the rule engine's tier match.

    Conservative defaults: an unrecognized policy_id maps to STANDARD so we
    don't accidentally under-review.
    """
    return {
        "TIER-AUTO": _LITE,
        "TIER-MGR":  _STANDARD,
        "TIER-DIR":  _DEEP,
        "TIER-CFO":  _DEEPEST,
    }.get(policy_id or "", _STANDARD)
