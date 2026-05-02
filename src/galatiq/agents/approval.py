"""Approval agent: deterministic tier routing over (Invoice, ValidationReport).

Pure function: never mutates the DB. The orchestrator decides whether to record an
approval_log entry. Uses ``galatiq.fx.to_usd`` to normalize amount before tier match.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal

from galatiq.agents.validation import ValidationReport
from galatiq.db import lookup_policy
from galatiq.fx import to_usd
from galatiq.models.invoice import Invoice

ApprovalStatus = Literal["auto_approved", "pending_human", "rejected"]


@dataclass(frozen=True)
class ApprovalDecision:
    status: ApprovalStatus
    approver_role: str
    policy_id: str | None
    total_usd: Decimal
    justification: str
    escalations: list[str] = field(default_factory=list)


def approve(
    invoice: Invoice,
    report: ValidationReport,
    *,
    conn: sqlite3.Connection,
) -> ApprovalDecision:
    total_usd = to_usd(invoice.total, invoice.currency)
    escalations = sorted({f.code for f in report.findings if f.severity in ("warn", "error")})

    if report.verdict == "reject":
        first_error = next((f for f in report.findings if f.severity == "error"), None)
        justification = first_error.message if first_error else "validation rejected the invoice"
        return ApprovalDecision(
            status="rejected",
            approver_role="none",
            policy_id=None,
            total_usd=total_usd,
            justification=justification,
            escalations=escalations,
        )

    policy = lookup_policy(conn, total_usd)
    if policy is None:
        # Should not happen if seed covers $0..∞, but be defensive.
        return ApprovalDecision(
            status="pending_human",
            approver_role="cfo",
            policy_id=None,
            total_usd=total_usd,
            justification=f"no policy band matched total_usd={total_usd}; escalating to CFO",
            escalations=escalations,
        )

    if policy.approver_role == "system" and report.verdict == "pass":
        return ApprovalDecision(
            status="auto_approved",
            approver_role="system",
            policy_id=policy.policy_id,
            total_usd=total_usd,
            justification=f"auto-approved under {policy.policy_id} (total_usd={total_usd} < {policy.max_usd})",
            escalations=escalations,
        )

    # needs_review verdict OR amount above auto threshold → human approval.
    if report.verdict == "needs_review":
        reason = f"validation flagged {len(escalations)} concern(s); requires {policy.approver_role}"
    else:
        reason = f"total_usd={total_usd} routes to {policy.approver_role} under {policy.policy_id}"
    return ApprovalDecision(
        status="pending_human",
        approver_role=policy.approver_role,
        policy_id=policy.policy_id,
        total_usd=total_usd,
        justification=reason,
        escalations=escalations,
    )
