"""Policy reviewer: examines tier appropriateness and business-policy fit.

Lens: is the rule engine routing this invoice to the right tier? Are there
soft business-policy concerns (vendor concentration, frequent-vendor velocity,
unusual payment terms) that warrant escalation independent of fraud or
compliance signals?
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
You are the POLICY reviewer in a council of approval reviewers. Your lens is
business-policy fit: tier appropriateness, vendor-relationship concentration,
unusual payment terms, and whether the rule-engine's tier (TIER-AUTO/MGR/DIR/
CFO) is appropriate for this case.

You can call:
- lookup_vendor(name) — vendor record
- lookup_catalog_item(name) — SKU details
- recent_invoices_for_vendor(vendor, limit) — frequency / concentration

Verdict scale (least to most conservative):
- "approve": tier is appropriate, no policy concerns
- "approve_with_notes": minor concern (e.g. unusual payment terms) — flag it
- "downgrade_to_human": the rule engine auto-approved but business policy
  suggests a human should at least see this
- "escalate_one_tier": tier looks too low given vendor concentration / cadence
- "escalate_to_cfo": material policy concern; CFO sign-off warranted
- "reject": violates a business policy outright

Default to "approve" unless you see something specific. Cite facts.
"""


def _fallback() -> ReviewerOpinion:
    return ReviewerOpinion(
        reviewer="policy",
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
        "Review this invoice from a POLICY perspective. Use tools as needed.\n\n"
        f"```json\n{json.dumps(payload, indent=2, default=str)}\n```"
    )
    opinion, err, trace = _llm_helpers.run_tool_using_agent(
        ReviewerOpinion,
        system=_SYSTEM,
        user=user,
        tools=build_investigator_tools(db_path),
        fallback=_fallback,
    )
    if opinion.reviewer != "policy":
        opinion = opinion.model_copy(update={"reviewer": "policy"})
    return opinion, err, trace
