"""Payment guards: independent checks before money moves.

Three sub-agents run between the council's final decision and the payment
execution. Each can block the payment with a structured reason:

  • near_duplicate_check   — LLM tool agent: scans recent ledger for the vendor,
                              flags near-duplicates by amount + date pattern
                              (catches dupes the deterministic invoice_number
                              ledger guard misses).
  • banking_validator      — deterministic: confirms the vendor has a registered
                              account on the proposed rail and that it's active.
  • payment_critic         — LLM reflection: final synthesis. Can recommend
                              "approve_payment", "switch_rail", "block".

The output is a ``PaymentGuardReport`` consumed by the pay node, which respects
``approved`` and refuses to schedule when ``approved=False``.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from galatiq.agents import _llm_helpers
from galatiq.agents.approval import ApprovalDecision
from galatiq.agents.tools import build_investigator_tools
from galatiq.db import (
    Vendor,
    list_vendor_payment_methods,
    lookup_payment_method,
)
from galatiq.models.invoice import Invoice


# --- structured outputs ------------------------------------------------------


class _NearDupResult(BaseModel):
    has_near_duplicate: bool = False
    rationale: str = ""
    matched_invoice_numbers: list[str] = Field(default_factory=list)


class _PaymentCriticResult(BaseModel):
    action: Literal["approve_payment", "switch_rail", "block"]
    suggested_rail: Literal["ach", "wire", "check"] | None = None
    rationale: str


@dataclass(frozen=True)
class PaymentGuardReport:
    approved: bool
    blockers: list[str]
    warnings: list[str]
    suggested_rail: str | None
    near_dup_trace: list[str]
    near_dup_matches: list[str]
    payment_method_status: str  # "active" | "disabled" | "pending_verification" | "missing"
    critic_action: str
    critic_rationale: str


# --- near-duplicate check (LLM tool-using) ----------------------------------


_NEAR_DUP_SYSTEM = """\
You are the NEAR-DUPLICATE checker for invoice payments. You look at recent
ledger entries for this vendor and decide whether the current invoice looks
like a near-duplicate of one already paid (e.g. same amount + similar date,
same line items reframed, etc.).

You have ``recent_invoices_for_vendor(vendor, limit)`` available. Call it
first to see history. If history is empty, default to has_near_duplicate=False.

Be conservative: only flag near-duplicates when the evidence is concrete
(e.g. "INV-1042 paid 5 days ago for the same $2400 — possible double-bill").
Do not invent matches.
"""


def _near_dup_fallback() -> _NearDupResult:
    return _NearDupResult(
        has_near_duplicate=False,
        rationale="LLM unavailable; near-duplicate check skipped.",
    )


def _near_duplicate_check(
    invoice: Invoice, db_path: Path
) -> tuple[_NearDupResult, str | None, list[str]]:
    user = (
        "Check whether this invoice is a near-duplicate of a recent payment to "
        "the same vendor.\n\n"
        f"```json\n{json.dumps(invoice.model_dump(mode='json'), indent=2, default=str)}\n```"
    )
    return _llm_helpers.run_tool_using_agent(
        _NearDupResult,
        system=_NEAR_DUP_SYSTEM,
        user=user,
        tools=build_investigator_tools(db_path),
        fallback=_near_dup_fallback,
    )


# --- payment critic (final reflection on the proposed payment) --------------


_CRITIC_SYSTEM = """\
You are the PAYMENT CRITIC. You see:
  - the approved invoice
  - the proposed rail and amount
  - the registered vendor payment methods (and their status)
  - the near-duplicate check result

Decide one of:
  - "approve_payment": proceed as proposed
  - "switch_rail": same vendor, different rail (e.g. proposed ach but vendor
    only has wire active — suggest switching to wire)
  - "block": do not pay (near-duplicate found, all rails disabled, etc.)

Cite specific facts. Default to approve_payment if everything looks clean.
"""


def _critic_fallback() -> _PaymentCriticResult:
    return _PaymentCriticResult(
        action="approve_payment",
        suggested_rail=None,
        rationale="LLM unavailable; deferring to deterministic checks only.",
    )


def _payment_critic(
    invoice: Invoice,
    decision: ApprovalDecision,
    proposed_rail: str,
    methods: list,
    near_dup: _NearDupResult,
) -> tuple[_PaymentCriticResult, str | None]:
    payload = {
        "invoice_number": invoice.invoice_number,
        "vendor": invoice.vendor,
        "currency": invoice.currency,
        "total": str(invoice.total),
        "amount_usd": str(decision.total_usd),
        "approver_role": decision.approver_role,
        "proposed_rail": proposed_rail,
        "registered_payment_methods": [
            {"rail": m.rail, "status": m.status, "account_ref_present": m.account_ref is not None}
            for m in methods
        ],
        "near_duplicate_check": {
            "has_near_duplicate": near_dup.has_near_duplicate,
            "matched_invoice_numbers": near_dup.matched_invoice_numbers,
            "rationale": near_dup.rationale,
        },
    }
    user = (
        "Reflect on the proposed payment and decide whether to approve, switch "
        "rails, or block.\n\n"
        f"```json\n{json.dumps(payload, indent=2, default=str)}\n```"
    )
    return _llm_helpers.run_llm_agent(
        _PaymentCriticResult,
        system=_CRITIC_SYSTEM,
        user=user,
        fallback=_critic_fallback,
    )


# --- public entry point ------------------------------------------------------


_RAIL_BY_USD = lambda usd: (  # noqa: E731 — terse local lookup
    "ach" if usd < Decimal("5000") else ("wire" if usd < Decimal("50000") else "check")
)


def run_payment_guards(
    invoice: Invoice,
    decision: ApprovalDecision,
    *,
    vendor: Vendor | None,
    conn: sqlite3.Connection,
) -> tuple[PaymentGuardReport, str | None]:
    """Run all three guards. Returns ``(report, error)``.

    The returned ``error`` is None on success or set to the first LLM-side error
    encountered (the deterministic fallback still runs in that case).
    """
    db_path = Path(conn.execute("PRAGMA database_list").fetchone()["file"])
    blockers: list[str] = []
    warnings: list[str] = []
    err_summary: list[str] = []

    proposed_rail = _RAIL_BY_USD(decision.total_usd)

    # 1. Banking validator (deterministic).
    methods: list = []
    payment_method_status = "missing"
    suggested_rail: str | None = None
    if vendor is not None:
        methods = list_vendor_payment_methods(conn, vendor.vendor_id)
        active_rails = [m.rail for m in methods if m.status == "active"]
        if not methods:
            blockers.append(f"vendor {vendor.vendor_id} has no registered payment methods")
        elif all(m.status == "disabled" for m in methods):
            blockers.append(f"vendor {vendor.vendor_id} has all payment methods disabled")
            payment_method_status = "disabled"
        elif any(m.status == "pending_verification" for m in methods) and not active_rails:
            blockers.append(f"vendor {vendor.vendor_id} payment methods are pending verification")
            payment_method_status = "pending_verification"
        else:
            method = lookup_payment_method(conn, vendor.vendor_id, proposed_rail)
            if method is None or method.status != "active":
                # The proposed rail isn't supported. Suggest a fallback if any active rail exists.
                if active_rails:
                    suggested_rail = active_rails[0]
                    warnings.append(
                        f"proposed rail '{proposed_rail}' not registered for vendor; "
                        f"suggest switching to '{suggested_rail}'"
                    )
                    payment_method_status = "active"
                else:
                    blockers.append(
                        f"no active rail for vendor {vendor.vendor_id} on proposed '{proposed_rail}'"
                    )
            else:
                payment_method_status = "active"
    else:
        warnings.append("vendor record not found; banking verification skipped")

    # 2. Near-duplicate check (LLM tool agent).
    near_dup, near_err, trace = _near_duplicate_check(invoice, db_path)
    if near_err:
        err_summary.append(f"near_dup_check: {near_err}")
    if near_dup.has_near_duplicate:
        blockers.append(f"near-duplicate detected: {near_dup.rationale}")

    # 3. Payment critic (LLM reflection).
    critic, critic_err = _payment_critic(invoice, decision, proposed_rail, methods, near_dup)
    if critic_err:
        err_summary.append(f"payment_critic: {critic_err}")
    if critic.action == "block":
        blockers.append(f"payment critic blocks: {critic.rationale}")
    elif critic.action == "switch_rail" and critic.suggested_rail:
        warnings.append(
            f"payment critic recommends rail switch to '{critic.suggested_rail}': {critic.rationale}"
        )
        if suggested_rail is None:
            suggested_rail = critic.suggested_rail

    approved = not blockers
    err = "; ".join(err_summary) if err_summary else None
    return (
        PaymentGuardReport(
            approved=approved,
            blockers=blockers,
            warnings=warnings,
            suggested_rail=suggested_rail,
            near_dup_trace=trace,
            near_dup_matches=near_dup.matched_invoice_numbers,
            payment_method_status=payment_method_status,
            critic_action=critic.action,
            critic_rationale=critic.rationale,
        ),
        err,
    )
