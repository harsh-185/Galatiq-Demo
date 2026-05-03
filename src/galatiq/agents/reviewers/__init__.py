"""Approval-council reviewer agents.

Each reviewer is a tool-using LLM agent that examines an invoice through a
specific lens (compliance, fraud, policy). The aggregator combines their
opinions with the rule-based decision to produce the final ApprovalDecision.

Pattern: more compute = more reviewers / deeper investigation per reviewer,
NOT recursive critique (which has diminishing returns and risks deadlock).
"""
from galatiq.agents.reviewers.types import (
    ReviewerOpinion,
    ReviewerVerdict,
    aggregate_opinions,
)

__all__ = ["ReviewerOpinion", "ReviewerVerdict", "aggregate_opinions"]
