"""Investigator: LLM agent that researches needs_review cases.

Reads the invoice, validation findings, and vendor history. Produces a structured
RiskAssessment for the human approver: severity summary, root-cause hypothesis,
recommended action, and a short list of items to verify. Advisory only.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Literal

from pydantic import BaseModel, Field

from galatiq.agents._llm_helpers import run_llm_agent
from galatiq.agents.validation import ValidationReport
from galatiq.db import lookup_vendor
from galatiq.models.invoice import Invoice

_SYSTEM = """\
You are an invoice investigator. A deterministic rule engine has flagged this
invoice as needs_review. Read the invoice, the findings, and any vendor history
provided, then produce a structured assessment for the human who will decide.

Be specific:
- severity_summary: 1 sentence on the overall severity (low/medium/high) and why.
- root_cause_hypothesis: your best guess at what is actually going on (data
  entry typo, real fraud, legitimate price change, etc.). Pick the most likely
  story and commit to it.
- recommended_action: one of "approve_with_notes", "request_clarification",
  "reject_and_escalate", "ask_vendor_to_resubmit".
- items_to_verify: 2-5 concrete checks the human should perform before deciding
  (e.g. "confirm WidgetA list price with procurement"). Each item is one short
  imperative sentence.

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
) -> tuple[RiskAssessment, str | None]:
    vendor = lookup_vendor(conn, invoice.vendor)
    payload = {
        "invoice": invoice.model_dump(mode="json"),
        "verdict": report.verdict,
        "findings": [
            {"code": f.code, "severity": f.severity, "field": f.field, "message": f.message}
            for f in report.findings
        ],
        "vendor_record": (
            {
                "vendor_id": vendor.vendor_id,
                "name": vendor.name,
                "status": vendor.status,
                "default_currency": vendor.default_currency,
            }
            if vendor
            else None
        ),
    }
    user = (
        "Investigate this needs_review invoice.\n\n"
        f"```json\n{json.dumps(payload, indent=2, default=str)}\n```"
    )
    return run_llm_agent(
        RiskAssessment,
        system=_SYSTEM,
        user=user,
        fallback=lambda: _fallback(report),
    )
