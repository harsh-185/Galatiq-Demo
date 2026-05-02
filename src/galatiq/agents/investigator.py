"""Investigator: tool-using LLM agent that researches needs_review cases.

The agent has tools to look up vendor records, catalog items, and recent ledger
history. It investigates the invoice and findings, calls tools as needed, and
produces a structured RiskAssessment for the human approver.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from galatiq.agents import _llm_helpers
from galatiq.agents.tools import build_investigator_tools
from galatiq.agents.validation import ValidationReport
from galatiq.models.invoice import Invoice

_SYSTEM = """\
You are an invoice investigator. A deterministic rule engine has flagged this
invoice as needs_review. Investigate the case using the tools available, then
produce a structured assessment for the human who will decide.

Tools you can call:
- lookup_vendor(name) — pull the vendor row (status, aliases, default_currency)
- lookup_catalog_item(name) — pull a SKU row (stock, unit_price, status)
- recent_invoices_for_vendor(vendor, limit) — recent ledger entries for context

Be specific in your output:
- severity_summary: 1 sentence on overall severity (low/medium/high) with reason.
- root_cause_hypothesis: your best guess at what is happening (data entry typo,
  fraud, legitimate price change, etc.). Pick the most likely story and commit.
- recommended_action: one of "approve_with_notes", "request_clarification",
  "reject_and_escalate", "ask_vendor_to_resubmit".
- items_to_verify: 2-5 concrete checks the human should perform. Each one is
  one short imperative sentence (e.g. "Confirm WidgetA list price with procurement").

Do not opine on policy bands or approver tiers — that is the rule engine's job.
"""


class RiskAssessment(BaseModel):
    severity_summary: str
    root_cause_hypothesis: str
    recommended_action: Literal[
        "approve_with_notes",
        "request_clarification",
        "reject_and_escalate",
        "ask_vendor_to_resubmit",
    ]
    items_to_verify: list[str] = Field(default_factory=list)


def _fallback(report: ValidationReport) -> RiskAssessment:
    finding_codes = sorted({f.code for f in report.findings})
    summary = (
        f"medium — {len(report.findings)} finding(s): {', '.join(finding_codes) or 'none'}"
        if report.findings
        else "low — no findings"
    )
    return RiskAssessment(
        severity_summary=summary,
        root_cause_hypothesis="LLM investigator unavailable; defer to validation findings.",
        recommended_action="request_clarification",
        items_to_verify=[f"Reconfirm finding: {code}" for code in finding_codes[:5]],
    )


def assess(
    invoice: Invoice,
    report: ValidationReport,
    *,
    conn: sqlite3.Connection,
) -> tuple[RiskAssessment, str | None, list[str]]:
    db_path = _conn_path(conn)
    payload = {
        "invoice": invoice.model_dump(mode="json"),
        "verdict": report.verdict,
        "findings": [
            {"code": f.code, "severity": f.severity, "field": f.field, "message": f.message}
            for f in report.findings
        ],
    }
    user = (
        "Investigate this needs_review invoice. Use the tools to verify "
        "vendor and catalog facts before assessing.\n\n"
        f"```json\n{json.dumps(payload, indent=2, default=str)}\n```"
    )
    return _llm_helpers.run_tool_using_agent(
        RiskAssessment,
        system=_SYSTEM,
        user=user,
        tools=build_investigator_tools(db_path),
        fallback=lambda: _fallback(report),
    )


def _conn_path(conn: sqlite3.Connection) -> Path:
    row = conn.execute("PRAGMA database_list").fetchone()
    return Path(row["file"]) if row and row["file"] else Path("inventory.db")
