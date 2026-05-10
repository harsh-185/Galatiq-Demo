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
FRAUD reviewer. Lens: deception patterns (typosquats, anomalous amounts vs vendor history, category mismatches).
Tools: lookup_vendor, list_known_vendors, lookup_catalog_item, recent_invoices_for_vendor.
Verdict: approve | approve_with_notes | downgrade_to_human | escalate_one_tier | escalate_to_cfo | reject.

Operating definitions (use these — do not invent your own):
- typosquat: invoice vendor name differs from a known vendor by ≤2 character edits AND shares a brand keyword. Different legal-entity suffixes ("X Corp" vs "X Services LLC", "X Inc." vs "X LLC") are NOT typosquats — they are routine business shapes.
- anomalous amount: total is >2× the vendor's recent ledger average AND >$10k. If the vendor has no ledger history, you cannot make this claim — say "no history, cannot assess" instead.
- category_mismatch: a line item exists in the catalog but the invoice description requires a different category (e.g. catalog 'hardware' but line says 'consulting hours').

NOT fraud signals (do NOT cite these as concerns; if asked about them, set verdict='approve'):
- Round-number totals ($1k/$5k/$10k). Routine in finance.
- Amounts that 'feel high' without a catalog or history reference.
- Different legal-entity suffixes between similar names.

If you have nothing concrete: verdict='approve', severity='low', rationale='no concrete fraud signals'. Don't reach for findings.
Cite specific facts (vendor names, SKU codes, invoice numbers, dates) in rationale.
"""


def _fallback() -> ReviewerOpinion:
    return ReviewerOpinion(
        reviewer="fraud",
        verdict="approve",
        severity="low",
        rationale="LLM unavailable; deferring to rule engine (no concrete fraud signals).",
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
