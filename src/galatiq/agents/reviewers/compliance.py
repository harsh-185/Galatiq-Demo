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
COMPLIANCE reviewer. Lens: regulatory + counterparty risk (blocked vendors, sanctions, currency drift, dup-invoice patterns).
Tools: lookup_vendor, lookup_catalog_item, recent_invoices_for_vendor.
Verdict (least→most conservative): approve | approve_with_notes | downgrade_to_human | escalate_one_tier | escalate_to_cfo | reject.
Cite specific finding codes or DB facts. Concise.
"""


def _fallback() -> ReviewerOpinion:
    # severity="medium" so the unanimous-clean relaxation in the aggregator
    # does NOT fire on deterministic-mode runs — relaxation requires real
    # LLM agreement (severity="low" from the model).
    return ReviewerOpinion(
        reviewer="compliance",
        verdict="approve",
        severity="medium",
        rationale="LLM unavailable; deferring to rule engine.",
    )


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
        "Review this invoice from a COMPLIANCE perspective. Use tools as needed.\n\n"
        f"```json\n{json.dumps(payload, indent=2, default=str)}\n```"
    )
    opinion, err, trace = _llm_helpers.run_tool_using_agent(
        ReviewerOpinion,
        system=_SYSTEM,
        user=user,
        tools=build_investigator_tools(db_path),
        fallback=_fallback,
        max_tool_loops=max_tool_loops,
    )
    # Force the reviewer name to match this module (LLM might emit something else).
    if opinion.reviewer != "compliance":
        opinion = opinion.model_copy(update={"reviewer": "compliance"})
    return opinion, err, trace
