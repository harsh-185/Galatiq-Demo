"""Pre-approval screener: a single tool-using LLM agent that subsumes the
fraud_screener, investigator, and vendor_onboarding agents.

Runs once between ingest and validate. Output:
  • fraud_findings   — merged into ValidationReport (can promote pass→needs_review)
  • items_to_verify  — concrete checks for the human (formerly investigator)
  • risk_severity    — overall lens of the invoice
  • risk_hypothesis  — best guess at root cause if there are concerns
  • vendor_profile   — drafted iff vendor looks new/unknown (formerly vendor_onboarding)

One LLM call (with up to 3 tool loops) replaces what used to be two-or-three
separate calls. The agent has all 5 DB tools available and uses them
selectively. Deterministic fallback returns an empty summary so the pipeline
never breaks.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from galatiq.agents import _llm_helpers
from galatiq.agents.tools import build_screener_tools, build_investigator_tools
from galatiq.agents.validation import Finding
from galatiq.models.invoice import Invoice


class VendorProfile(BaseModel):
    """Drafted onboarding profile for first-time / unknown vendors.

    Populated only when the screener decides the vendor warrants onboarding
    review (status_new, vendor_unknown, etc.). The HITL queue can attach this
    to the human-review entry for manual confirmation.
    """
    suggested_aliases: list[str] = Field(default_factory=list)
    normalized_address: str | None = None
    default_currency_guess: str | None = None
    recommendation: str = "needs_more_info"
    rationale: str = ""

_SYSTEM = """\
You are the PRE-APPROVAL SCREENER — a single agent that examines an invoice
before any rule-based validation runs. You have tools to query the vendor
table, the catalog, and the recent ledger. Use them selectively.

Your output has four parts:

1. fraud_findings — list of advisory findings about non-rule anomalies that
   warrant human attention. Codes you can emit:
     • vendor_typosquat        — vendor name resembles a known vendor
     • category_mismatch        — line items don't fit catalog category
     • round_number_padding     — suspicious all-round amounts
     • suspicious_invoice_number — pattern that suggests forgery
   Severity: "warn" for credible suspicion, "info" for benign heuristics.
   Never use "error" — you are advisory.

2. items_to_verify — 0 to 5 concrete verification steps the human should
   perform if anything is off. One short imperative sentence each.
   ("Confirm WidgetA list price with procurement.")

3. risk_severity — overall: "low", "medium", "high", or "none" (no concerns).
   risk_hypothesis — 1 short sentence explaining the severity. Empty if "none".

4. vendor_profile — populate ONLY when the vendor appears to be unknown OR
   new (use lookup_vendor to confirm). Suggest aliases, normalized address,
   currency guess, and a recommendation ("approve_onboarding",
   "needs_more_info", "reject"). Leave null when the vendor is well-known.

Rules:
  • Default to empty / low severity. Do not invent concerns.
  • Cite specific facts (vendor name, SKU, dollar amount) in any rationale.
  • Use tools for verification — don't guess about the vendor or catalog.
"""


_Severity = Literal["info", "warn"]


class _ScreenedFinding(BaseModel):
    code: str
    severity: _Severity = "warn"
    message: str
    field: str | None = None


class PreApprovalSummary(BaseModel):
    fraud_findings: list[_ScreenedFinding] = Field(default_factory=list)
    items_to_verify: list[str] = Field(default_factory=list)
    risk_severity: Literal["none", "low", "medium", "high"] = "none"
    risk_hypothesis: str = ""
    vendor_profile: VendorProfile | None = None


def _fallback() -> PreApprovalSummary:
    return PreApprovalSummary()


def _merged_tools(db_path: Path):
    """Both screener and investigator tools, deduped by name."""
    seen = set()
    out = []
    for t in [*build_screener_tools(db_path), *build_investigator_tools(db_path)]:
        if t.name in seen:
            continue
        seen.add(t.name)
        out.append(t)
    return out


def screen(
    invoice: Invoice, *, conn: sqlite3.Connection, max_tool_loops: int = 2
) -> tuple[PreApprovalSummary, list[Finding], str | None, list[str]]:
    """Run the pre-approval screener.

    Returns ``(summary, fraud_findings_for_validate, error, tool_trace)``.
    The ``fraud_findings_for_validate`` list is what the validate step merges
    into its report; it's the same content as ``summary.fraud_findings`` but
    wrapped in the validation ``Finding`` shape.
    """
    db_path = _conn_path(conn)
    user = (
        "Screen this invoice. Use tools to verify the vendor and catalog as "
        "needed, then produce the structured summary.\n\n"
        f"```json\n{json.dumps(invoice.model_dump(mode='json'), indent=2, default=str)}\n```"
    )
    summary, err, trace = _llm_helpers.run_tool_using_agent(
        PreApprovalSummary,
        system=_SYSTEM,
        user=user,
        tools=_merged_tools(db_path),
        fallback=_fallback,
        max_tool_loops=max_tool_loops,
    )
    findings = [
        Finding(code=f.code, severity=f.severity, message=f.message, field=f.field)
        for f in summary.fraud_findings
    ]
    return summary, findings, err, trace


def _conn_path(conn: sqlite3.Connection) -> Path:
    row = conn.execute("PRAGMA database_list").fetchone()
    return Path(row["file"]) if row and row["file"] else Path("inventory.db")
