"""Compliance reviewer: examines vendor/regulatory concerns.

Lens: vendor sanctions / status, currency drift vs vendor default, duplicate-
invoice patterns, jurisdiction (via vendor address). Calls vendor lookup and
ledger history tools.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from galatiq.agents import _llm_helpers
from galatiq.agents.approval import ApprovalDecision
from galatiq.agents.reviewers.types import ReviewerOpinion
from galatiq.agents.tools import build_investigator_tools
from galatiq.agents.validation import ValidationReport
from galatiq.models.invoice import Invoice

_SYSTEM = """\
You are the COMPLIANCE reviewer in a council of approval reviewers. Your lens
is regulatory and counterparty risk: blocked vendors, sanctions concerns,
currency mismatches that imply jurisdiction issues, suspicious duplicate-
invoice patterns, and unverifiable vendor identity.

You can call:
- lookup_vendor(name) — confirm vendor status / aliases
- lookup_catalog_item(name) — only if needed
- recent_invoices_for_vendor(vendor, limit) — check submission patterns

Verdict scale (least to most conservative):
- "approve": no compliance concerns
- "approve_with_notes": minor concern worth flagging in the audit trail
- "downgrade_to_human": auto-approval is too risky given compliance risk
- "escalate_one_tier": bump approver one tier
- "escalate_to_cfo": serious compliance risk (sanctions, jurisdiction, etc.)
- "reject": vendor or transaction is non-compliant; do not approve

Be specific. Cite finding codes or DB lookups in your rationale.
"""


def _fallback() -> ReviewerOpinion:
    return ReviewerOpinion(
        reviewer="compliance",
        verdict="approve",
        severity="low",
        rationale="LLM unavailable; deferring to rule engine.",
    )


def review(
    invoice: Invoice,
    report: ValidationReport,
    decision: ApprovalDecision,
    *,
    conn: sqlite3.Connection,
) -> tuple[ReviewerOpinion, str | None, list[str]]:
    db_path = Path(conn.execute("PRAGMA database_list").fetchone()["file"])
    payload = {
        "invoice": invoice.model_dump(mode="json"),
        "engine_decision": {
            "status": decision.status,
            "approver_role": decision.approver_role,
            "policy_id": decision.policy_id,
            "total_usd": str(decision.total_usd),
        },
        "verdict": report.verdict,
        "findings": [
            {"code": f.code, "severity": f.severity, "message": f.message}
            for f in report.findings
        ],
    }
    user = (
        "Review this invoice from a COMPLIANCE perspective. Use tools as needed.\n\n"
        f"```json\n{json.dumps(payload, indent=2, default=str)}\n```"
    )
    opinion, err, trace = _llm_helpers.run_tool_using_agent(
        ReviewerOpinion,
        system=_SYSTEM,
        user=user,
        tools=build_investigator_tools(db_path),
        fallback=_fallback,
    )
    # Force the reviewer name to match this module (LLM might emit something else).
    if opinion.reviewer != "compliance":
        opinion = opinion.model_copy(update={"reviewer": "compliance"})
    return opinion, err, trace
