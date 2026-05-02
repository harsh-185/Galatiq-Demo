"""Vendor onboarding: LLM agent that drafts a profile for STATUS_NEW vendors.

Triggered when a first-time vendor appears. Produces a structured suggestion
(aliases, normalized address, currency guess, recommendation) for a human to
review before the vendor record is finalized. The agent never writes to the
vendor table itself.
"""
from __future__ import annotations

import json

from pydantic import BaseModel, Field

from galatiq.agents._llm_helpers import run_llm_agent
from galatiq.db import Vendor
from galatiq.models.invoice import Invoice

_SYSTEM = """\
You are a vendor onboarding agent. You receive an invoice from a first-time
vendor (status=new) and the existing vendor row. Draft a suggested profile for
human review. Be conservative — when unsure, leave a field blank rather than
guessing.

Return:
- suggested_aliases: short list of plausible alternate spellings/abbreviations
  that the alias matcher should accept (e.g. "Acme" for "Acme Corp"). Do NOT
  invent aliases unrelated to the actual vendor name.
- normalized_address: a single-line cleaned version of the vendor address from
  the invoice; null if the invoice has no address.
- default_currency_guess: the currency this vendor most likely bills in,
  inferred from the invoice. Use ISO 4217 codes (USD, EUR, GBP, JPY, CAD, AUD).
- recommendation: "approve_onboarding", "needs_more_info", or "reject".
- rationale: one sentence explaining the recommendation.
"""


class VendorProfile(BaseModel):
    suggested_aliases: list[str] = Field(default_factory=list)
    normalized_address: str | None = None
    default_currency_guess: str | None = None
    recommendation: str = "needs_more_info"
    rationale: str = ""


def _fallback(invoice: Invoice, vendor: Vendor) -> VendorProfile:
    return VendorProfile(
        suggested_aliases=[],
        normalized_address=invoice.vendor_address,
        default_currency_guess=invoice.currency,
        recommendation="needs_more_info",
        rationale="LLM agent unavailable; manual review required.",
    )


def draft_profile(invoice: Invoice, vendor: Vendor) -> tuple[VendorProfile, str | None]:
    payload = {
        "invoice_excerpt": {
            "vendor": invoice.vendor,
            "vendor_address": invoice.vendor_address,
            "currency": invoice.currency,
            "total": str(invoice.total),
            "line_items": [
                {"item": li.item, "quantity": li.quantity, "unit_price": str(li.unit_price)}
                for li in invoice.line_items
            ],
        },
        "current_vendor_row": {
            "vendor_id": vendor.vendor_id,
            "name": vendor.name,
            "aliases": vendor.aliases,
            "address": vendor.address,
            "status": vendor.status,
            "default_currency": vendor.default_currency,
        },
    }
    user = (
        "Draft an onboarding profile for this first-time vendor.\n\n"
        f"```json\n{json.dumps(payload, indent=2, default=str)}\n```"
    )
    return run_llm_agent(
        VendorProfile,
        system=_SYSTEM,
        user=user,
        fallback=lambda: _fallback(invoice, vendor),
    )
