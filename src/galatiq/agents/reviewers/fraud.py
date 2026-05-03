"""Fraud reviewer: examines fraud-pattern signals beyond hard rules.

Lens: typosquats vs known vendors, round-number padding, anomalous amounts
relative to vendor history, line-item categories that don't match the catalog.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from galatiq.agents import _llm_helpers
from galatiq.agents.approval import ApprovalDecision
from galatiq.agents.reviewers.types import ReviewerOpinion
from galatiq.agents.tools import build_screener_tools, build_investigator_tools
from galatiq.agents.validation import ValidationReport
from galatiq.models.invoice import Invoice

_SYSTEM = """\
You are the FRAUD reviewer in a council of approval reviewers. Your lens is
deception patterns: vendor-name typosquats, round-number invoice padding,
amounts that look outsized vs the vendor's history, line items that don't fit
the vendor's catalog category.

You can call:
- lookup_vendor(name) — confirm canonical names + aliases
- list_known_vendors() — scan for typosquat candidates
- lookup_catalog_item(name) — verify SKU details
- recent_invoices_for_vendor(vendor, limit) — compare to history

Verdict scale (least to most conservative):
- "approve": no fraud signals
- "approve_with_notes": low-confidence pattern; flag in audit trail
- "downgrade_to_human": pattern warrants manual review before payment
- "escalate_one_tier": multiple correlated fraud signals
- "escalate_to_cfo": strong fraud indicators
- "reject": clear fraud — vendor impersonation, ghost invoices, etc.

Default to lower severity unless you have specific signals. Do not invent
patterns. Cite specific facts in your rationale.
"""


def _fallback() -> ReviewerOpinion:
    return ReviewerOpinion(
        reviewer="fraud",
        verdict="approve",
        severity="low",
        rationale="LLM unavailable; deferring to rule engine.",
    )


def _tools(db_path: Path):
    # Combine screener + investigator tools — fraud reviewer wants both vendor
    # listings and ledger history.
    seen = set()
    out = []
    for t in [*build_screener_tools(db_path), *build_investigator_tools(db_path)]:
        if t.name in seen:
            continue
        seen.add(t.name)
        out.append(t)
    return out


def review(
    invoice: Invoice,
    report: ValidationReport,
    decision: ApprovalDecision,
    *,
    conn: sqlite3.Connection,
    pre_approval_summary=None,
    max_tool_loops: int = 4,
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
        "pre_approval_summary": pre_approval_summary.model_dump() if pre_approval_summary else None,
    }
    user = (
        "Review this invoice from a FRAUD perspective. Use tools to verify.\n\n"
        f"```json\n{json.dumps(payload, indent=2, default=str)}\n```"
    )
    opinion, err, trace = _llm_helpers.run_tool_using_agent(
        ReviewerOpinion,
        system=_SYSTEM,
        user=user,
        tools=_tools(db_path),
        fallback=_fallback,
        max_tool_loops=max_tool_loops,
    )
    if opinion.reviewer != "fraud":
        opinion = opinion.model_copy(update={"reviewer": "fraud"})
    return opinion, err, trace
