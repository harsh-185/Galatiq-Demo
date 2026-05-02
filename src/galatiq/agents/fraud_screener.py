"""Fraud screener: tool-using LLM agent that catches anomalies the rule engine misses.

The agent has tools to look up vendors and catalog items selectively, plus the
full vendor/catalog listings as fallback context. It investigates the invoice,
calls the tools it needs, and returns advisory ``Finding``s. Findings merge
into the ValidationReport and can promote ``pass → needs_review``.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from galatiq.agents import _llm_helpers
from galatiq.agents.tools import build_screener_tools
from galatiq.agents.validation import Finding
from galatiq.models.invoice import Invoice

_SYSTEM = """\
You are a fraud-screening agent for an invoice automation pipeline. You receive
one invoice and have tools to query the vendor table and catalog. Investigate
*selectively* — call tools only when they help — then return up to 5 advisory
findings about non-rule-based anomalies that warrant human attention.

Tools you can call:
- lookup_vendor(name) — confirm a vendor / detect typosquats vs canonical names
- lookup_catalog_item(name) — pull the catalog row for a SKU
- list_known_vendors() — full vendor list (use to scan for typosquat candidates)
- list_catalog() — full catalog (use to cross-check categories)

Anomaly codes you may emit:
- vendor_typosquat: vendor name resembles a known vendor with edits
- category_mismatch: line-item description doesn't fit the catalog category
- round_number_padding: amounts suspiciously round / repeated zeros
- suspicious_invoice_number: format pattern that suggests forgery

Severity rules:
- "warn" for credible suspicion.
- "info" for a heuristic worth noting but probably benign.
- Never use "error" — you are advisory only.

If nothing looks wrong, return an empty list. Do not invent findings.
"""


_Severity = Literal["info", "warn"]


class _ScreenedFinding(BaseModel):
    code: str
    severity: _Severity = "warn"
    message: str
    field: str | None = None


class FraudScreenResult(BaseModel):
    findings: list[_ScreenedFinding] = Field(default_factory=list)
    tool_trace: list[str] = Field(default_factory=list)


def screen(invoice: Invoice, *, conn: sqlite3.Connection) -> tuple[list[Finding], str | None, list[str]]:
    """Run the fraud-screener tool-using agent. Returns (findings, error, tool_trace)."""
    db_path = _conn_path(conn)
    user = (
        "Screen this invoice for advisory anomalies. Investigate via the tools "
        "before deciding.\n\n"
        f"```json\n{json.dumps(invoice.model_dump(mode='json'), indent=2, default=str)}\n```"
    )
    result, err, trace = _llm_helpers.run_tool_using_agent(
        FraudScreenResult,
        system=_SYSTEM,
        user=user,
        tools=build_screener_tools(db_path),
        fallback=lambda: FraudScreenResult(findings=[]),
    )
    findings = [
        Finding(code=f.code, severity=f.severity, message=f.message, field=f.field)
        for f in result.findings
    ]
    # Persist the trace on the result for the orchestrator (some Pydantic versions
    # don't propagate it through the LLM output, so prefer the helper's return).
    return findings, err, trace


def _conn_path(conn: sqlite3.Connection) -> Path:
    """Return the file path the connection is bound to (best-effort)."""
    row = conn.execute("PRAGMA database_list").fetchone()
    return Path(row["file"]) if row and row["file"] else Path("inventory.db")
