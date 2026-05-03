"""Payment guards: deterministic banking validator + a single merged LLM
``payment_review`` agent that subsumes the old near_duplicate_check and
payment_critic.

Pipeline order:
  1. Deterministic banking validator (no LLM): registered methods + status
  2. Deterministic pre-filter: skip LLM payment_review entirely when there's
     no recent ledger history for the vendor (nothing to compare against).
  3. LLM payment_review (one call, tool-using): scans recent ledger for
     near-duplicates AND reflects on whether the proposed payment should
     proceed, switch rails, or be blocked.

Banking blockers are load-bearing. The LLM payment_review is advisory but can
itself produce a blocker if it sees a near-duplicate or other concrete issue.
"""
from __future__ import annotations

import json
import os
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


# --- public output ----------------------------------------------------------


@dataclass(frozen=True)
class PaymentGuardReport:
    approved: bool
    blockers: list[str]
    warnings: list[str]
    suggested_rail: str | None
    review_trace: list[str]
    near_dup_matches: list[str]
    payment_method_status: str  # active | disabled | pending_verification | missing | unknown_vendor
    review_action: str          # approve_payment | switch_rail | block | skipped
    review_rationale: str


# --- merged LLM payment_review ---------------------------------------------


_REVIEW_SYSTEM_FULL = """\
You review a proposed payment. Banking already validated. Tools:
- recent_invoices_for_vendor(vendor, limit) — scan for near-duplicates
- lookup_vendor / lookup_catalog_item — verify facts
Decide one: approve_payment | switch_rail | block.
Default approve_payment. Cite specific invoice numbers/amounts/dates in rationale. Concise.
"""


_REVIEW_SYSTEM_NARRATIVE = """\
Write a 1-sentence rationale for this payment. Do not call tools.
Set action='approve_payment' unless banking is broken (then set to a sensible action with rationale).
Output the structured PaymentReview.
"""


class _PaymentReview(BaseModel):
    has_near_duplicate: bool = False
    near_dup_invoice_numbers: list[str] = Field(default_factory=list)
    action: Literal["approve_payment", "switch_rail", "block"] = "approve_payment"
    suggested_rail: Literal["ach", "wire", "check"] | None = None
    rationale: str = "no concerns"


def _review_fallback() -> _PaymentReview:
    return _PaymentReview(
        action="approve_payment",
        rationale="LLM unavailable; deferring to deterministic checks only.",
    )


# --- rail selector (matches pay agent's logic) -----------------------------


def _proposed_rail_for(usd: Decimal) -> str:
    if usd < Decimal("5000"):
        return "ach"
    if usd < Decimal("50000"):
        return "wire"
    return "check"


# --- public entry point -----------------------------------------------------


def run_payment_guards(
    invoice: Invoice,
    decision: ApprovalDecision,
    *,
    vendor: Vendor | None,
    conn: sqlite3.Connection,
) -> tuple[PaymentGuardReport, str | None]:
    """Run banking validation + payment_review. Returns ``(report, error)``."""
    db_path = _conn_path(conn)
    blockers: list[str] = []
    warnings: list[str] = []
    err_summary: list[str] = []

    proposed_rail = _proposed_rail_for(decision.total_usd)

    # 1. Banking validator (deterministic) ---------------------------------
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
        payment_method_status = "unknown_vendor"

    # 2. Deterministic pre-filter for the LLM payment_review ---------------
    # Look for prior invoices from this vendor — but EXCLUDE the current one,
    # since validate may have just written it to the ledger upstream.
    has_history = False
    if vendor is not None:
        row = conn.execute(
            "SELECT 1 FROM invoice_ledger WHERE vendor = ? AND invoice_number != ? LIMIT 1",
            (vendor.name, invoice.invoice_number),
        ).fetchone()
        has_history = row is not None

    # 3. ALWAYS run a payment_review LLM call so the audit trail has an
    #    LLM-written rationale for every payment decision. Mode depends on
    #    whether there's actually something to investigate:
    #      - "full":      tool-using investigation (when ledger history exists)
    #      - "narrative": single-shot, no tools, ~1-sentence rationale (otherwise)
    # Demo / showcase mode (GALATIQ_SHOW_AGENTS) forces full mode so users
    # can see the tool-using payment investigation regardless of history.
    show_agents = os.environ.get("GALATIQ_SHOW_AGENTS", "").strip().lower() in {"1", "true", "yes", "on"}
    review_mode = "full" if (has_history or show_agents) else "narrative"
    review, err, review_trace = _payment_review(
        invoice, decision, proposed_rail, methods, db_path, mode=review_mode,
    )
    review_action = review.action
    review_rationale = review.rationale
    near_dup_matches = list(review.near_dup_invoice_numbers)
    if err:
        err_summary.append(f"payment_review: {err}")
    if review.has_near_duplicate:
        blockers.append(
            f"near-duplicate detected (matches: {', '.join(review.near_dup_invoice_numbers) or 'unspecified'}): {review.rationale}"
        )
    elif review.action == "block":
        blockers.append(f"payment_review blocks: {review.rationale}")
    elif review.action == "switch_rail" and review.suggested_rail:
        warnings.append(
            f"payment_review recommends rail switch to '{review.suggested_rail}': {review.rationale}"
        )
        if suggested_rail is None:
            suggested_rail = review.suggested_rail

    approved = not blockers
    err = "; ".join(err_summary) if err_summary else None
    return (
        PaymentGuardReport(
            approved=approved,
            blockers=blockers,
            warnings=warnings,
            suggested_rail=suggested_rail,
            review_trace=review_trace,
            near_dup_matches=near_dup_matches,
            payment_method_status=payment_method_status,
            review_action=review_action,
            review_rationale=review_rationale,
        ),
        err,
    )


def _payment_review(
    invoice: Invoice,
    decision: ApprovalDecision,
    proposed_rail: str,
    methods: list,
    db_path: Path,
    mode: Literal["full", "narrative"] = "full",
) -> tuple[_PaymentReview, str | None, list[str]]:
    payload = {
        "invoice_number": invoice.invoice_number,
        "vendor": invoice.vendor,
        "currency": invoice.currency,
        "total_usd": str(decision.total_usd),
        "proposed_rail": proposed_rail,
        "registered_methods": [
            {"rail": m.rail, "status": m.status} for m in methods
        ],
    }
    if mode == "narrative":
        # 1 LLM call, no tools. Just produces a rationale + action.
        user = (
            f"Write a payment-review rationale (no tools needed; no ledger history).\n"
            f"```json\n{json.dumps(payload, default=str)}\n```"
        )
        review, err = _llm_helpers.run_llm_agent(
            _PaymentReview,
            system=_REVIEW_SYSTEM_NARRATIVE,
            user=user,
            fallback=_review_fallback,
        )
        return review, err, []  # no tool trace in narrative mode

    # Full mode: tool-using ReAct loop for actual near-dup investigation.
    user = (
        f"Investigate near-duplicates and decide approve_payment | switch_rail | block.\n"
        f"```json\n{json.dumps(payload, default=str)}\n```"
    )
    return _llm_helpers.run_tool_using_agent(
        _PaymentReview,
        system=_REVIEW_SYSTEM_FULL,
        user=user,
        tools=build_investigator_tools(db_path),
        fallback=_review_fallback,
    )


def _conn_path(conn: sqlite3.Connection) -> Path:
    row = conn.execute("PRAGMA database_list").fetchone()
    return Path(row["file"]) if row and row["file"] else Path("inventory.db")
