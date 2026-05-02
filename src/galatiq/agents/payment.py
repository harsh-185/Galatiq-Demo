"""Payment agent: deterministic mock-rail routing over (Invoice, ApprovalDecision, Vendor).

Pure function: never writes the DB or filesystem. The orchestrator persists the
PaymentRecord and writes the receipt artifact.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from typing import Literal

from galatiq.agents.approval import ApprovalDecision
from galatiq.db import STATUS_BLOCKED, Vendor
from galatiq.models.invoice import Invoice

PaymentStatus = Literal["scheduled", "skipped", "failed"]
PaymentRail = Literal["ach", "wire", "check", "none"]

_WIRE_THRESHOLD_USD = Decimal("5000")
_CHECK_THRESHOLD_USD = Decimal("50000")


@dataclass(frozen=True)
class PaymentRecord:
    status: PaymentStatus
    rail: PaymentRail
    reference: str
    amount_usd: Decimal
    currency_paid: str
    amount_paid: Decimal
    vendor_id: str | None
    scheduled_for: date | None
    receipt_path: str | None = None
    notes: list[str] = field(default_factory=list)


def pay(
    invoice: Invoice,
    decision: ApprovalDecision,
    *,
    vendor: Vendor | None,
    today: date | None = None,
) -> PaymentRecord:
    today = today or date.today()
    vendor_id = vendor.vendor_id if vendor else None

    base = dict(
        amount_usd=decision.total_usd,
        currency_paid=invoice.currency,
        amount_paid=Decimal(invoice.total),
        vendor_id=vendor_id,
    )

    if decision.status == "rejected":
        return PaymentRecord(
            status="skipped",
            rail="none",
            reference=_reference(vendor_id, invoice.invoice_number, "none"),
            scheduled_for=None,
            notes=[f"approval rejected: {decision.justification}"],
            **base,
        )

    if decision.status == "pending_human":
        return PaymentRecord(
            status="skipped",
            rail="none",
            reference=_reference(vendor_id, invoice.invoice_number, "none"),
            scheduled_for=None,
            notes=[f"awaiting human approval ({decision.approver_role})"],
            **base,
        )

    # auto_approved from here on.
    if vendor is not None and vendor.status == STATUS_BLOCKED:
        return PaymentRecord(
            status="failed",
            rail="none",
            reference=_reference(vendor_id, invoice.invoice_number, "none"),
            scheduled_for=None,
            notes=["vendor is blocked; refusing to schedule payment"],
            **base,
        )

    rail = _select_rail(decision.total_usd)
    scheduled_for = _schedule_date(invoice.due_date, today)
    return PaymentRecord(
        status="scheduled",
        rail=rail,
        reference=_reference(vendor_id, invoice.invoice_number, rail),
        scheduled_for=scheduled_for,
        notes=[f"scheduled via {rail} rail"],
        **base,
    )


def _select_rail(total_usd: Decimal) -> PaymentRail:
    if total_usd < _WIRE_THRESHOLD_USD:
        return "ach"
    if total_usd < _CHECK_THRESHOLD_USD:
        return "wire"
    return "check"


def _schedule_date(due_date: date | None, today: date) -> date:
    if due_date is not None and due_date >= today:
        return due_date
    return today + timedelta(days=1)


def _reference(vendor_id: str | None, invoice_number: str, rail: PaymentRail) -> str:
    vid = vendor_id or "UNKNOWN"
    return f"PAY-{vid}-{invoice_number}-{rail.upper()}"
