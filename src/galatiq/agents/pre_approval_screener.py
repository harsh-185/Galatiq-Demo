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
Pre-approval screener. Tools: lookup_vendor, lookup_catalog_item, list_known_vendors, list_catalog.
Produce: fraud_findings (codes: vendor_typosquat | category_mismatch | suspicious_invoice_number; severity warn|info, never error), items_to_verify (0-5 imperative checks), risk_severity (none|low|medium|high), risk_hypothesis (1 sentence), vendor_profile (only when vendor looks new/unknown).

What COUNTS as a finding (must be cited with concrete evidence):
- vendor_typosquat: the invoice vendor differs from a *known* vendor by ≤2 character edits AND shares a brand keyword (e.g. "Acme Corp" vs "Acrne Corp"). Different legal-entity suffixes ("X Corp" vs "X Services LLC") are NOT typosquats.
- category_mismatch: a line item exists in the catalog with a category that is incompatible with the invoice description (e.g. catalog 'hardware' but invoice line says 'consulting hours'). Identical names with different SKUs are NOT mismatches.
- suspicious_invoice_number: invoice_number contains obvious tells (placeholder text, "TEST", impossibly low/high sequential numbers, control characters). Numeric IDs that just look unusual to you are NOT findings.

What is NOT a fraud signal (do NOT emit findings for these):
- Round-number totals like $1000, $5000, $10000 — these are routine in finance (consulting retainers, SaaS, bulk orders, contract milestones). Round numbers alone are never a finding.
- Vendor legal-entity differences (LLC vs Corp vs Inc.).
- Invoices that are "just on the high side" without a concrete catalog or history reference.

Default empty/low. Don't invent concerns. Cite specific facts (concrete vendor names, SKU codes, exact differences). Concise.
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
    # Belt-and-suspenders: even if the LLM ignored the prompt and emitted a
    # disallowed code (e.g. round_number_padding), drop it before it can
    # influence validation or aggregator decisions.
    summary.fraud_findings = [
        f for f in summary.fraud_findings if f.code in _ALLOWED_FRAUD_CODES
    ]
    findings = [
        Finding(code=f.code, severity=f.severity, message=f.message, field=f.field)
        for f in summary.fraud_findings
    ]
    return summary, findings, err, trace


_ALLOWED_FRAUD_CODES = frozenset({
    "vendor_typosquat",
    "category_mismatch",
    "suspicious_invoice_number",
})


def _conn_path(conn: sqlite3.Connection) -> Path:
    row = conn.execute("PRAGMA database_list").fetchone()
    return Path(row["file"]) if row and row["file"] else Path("inventory.db")
