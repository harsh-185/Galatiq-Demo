"""Fraud screener: LLM agent that catches anomalies the rule engine cannot.

Looks for impersonation typos, round-number padding, line items mismatched to
catalog category, and reused invoice-number patterns. Adds advisory ``Finding``s
that the rule engine merges into its report. Never alters routing on its own.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Literal

from pydantic import BaseModel, Field

from galatiq.agents._llm_helpers import run_llm_agent
from galatiq.agents.validation import Finding
from galatiq.db import list_inventory, list_vendors
from galatiq.models.invoice import Invoice

_SYSTEM = """\
You are a fraud-screening agent for an invoice automation pipeline. You receive
one invoice and reference data (known vendors, known SKUs). You return up to 5
*advisory* findings about non-rule-based anomalies that warrant human attention:

- Vendor-name typos that suggest impersonation of a known vendor (e.g. "Acrne Corp"
  vs "Acme Corp"). Flag `code=vendor_typosquat`.
- Line-item descriptions that don't fit the catalog category for the SKU (e.g.
  GadgetX is electronics but description says "office furniture"). Flag
  `code=category_mismatch`.
- Round-number padding (every line item suspiciously a round multiple, or amounts
  ending in repeated zeros that don't match catalog prices). Flag
  `code=round_number_padding`.
- Invoice-number patterns suggesting forgery (sequential bursts, format mismatch
  with vendor's prior invoices). Flag `code=suspicious_invoice_number`.

Severity rules:
- "warn" for credible suspicion.
- "info" for a heuristic worth noting but probably benign.
- Never use "error" — you are advisory only; the rule engine owns hard rejects.

If nothing looks wrong, return an empty list. Do not invent findings to fill space.
"""


_Severity = Literal["info", "warn"]


class _ScreenedFinding(BaseModel):
    code: str
    severity: _Severity = "warn"
    message: str
    field: str | None = None


class FraudScreenResult(BaseModel):
    findings: list[_ScreenedFinding] = Field(default_factory=list)


def screen(invoice: Invoice, *, conn: sqlite3.Connection) -> tuple[list[Finding], str | None]:
    """Run the fraud-screener LLM agent. Returns (findings, error)."""
    user = _build_prompt(invoice, conn)
    result, err = run_llm_agent(
        FraudScreenResult,
        system=_SYSTEM,
        user=user,
        fallback=lambda: FraudScreenResult(findings=[]),
    )
    findings = [
        Finding(code=f.code, severity=f.severity, message=f.message, field=f.field)
        for f in result.findings
    ]
    return findings, err


def _build_prompt(invoice: Invoice, conn: sqlite3.Connection) -> str:
    vendors = [
        {"name": v.name, "aliases": v.aliases, "status": v.status}
        for v in list_vendors(conn)
    ]
    catalog = [
        {
            "item": i.item,
            "category": i.category,
            "unit_price": str(i.unit_price) if i.unit_price else None,
            "status": i.status,
        }
        for i in list_inventory(conn)
    ]
    payload = {
        "invoice": invoice.model_dump(mode="json"),
        "known_vendors": vendors,
        "catalog": catalog,
    }
    return (
        "Screen the following invoice for advisory anomalies.\n\n"
        f"```json\n{json.dumps(payload, indent=2, default=str)}\n```"
    )
