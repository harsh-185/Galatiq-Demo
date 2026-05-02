"""Plain-text payment receipts.

``render_receipt`` is a pure function — same inputs always produce the same body.
``write_receipt`` is the only side-effect: writes to ``data/receipts/{reference}.txt``,
short-circuiting if the file already contains identical content (idempotent reruns).
"""
from __future__ import annotations

from pathlib import Path

from galatiq.agents.approval import ApprovalDecision
from galatiq.agents.payment import PaymentRecord
from galatiq.models.invoice import Invoice

DEFAULT_RECEIPT_DIR = Path("data/receipts")


def render_receipt(
    invoice: Invoice, decision: ApprovalDecision, record: PaymentRecord
) -> str:
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("GALATIQ PAYMENT RECEIPT")
    lines.append("=" * 60)
    lines.append(f"Reference     : {record.reference}")
    lines.append(f"Status        : {record.status.upper()}")
    lines.append(f"Rail          : {record.rail.upper()}")
    lines.append(f"Scheduled for : {record.scheduled_for or '—'}")
    lines.append("")
    lines.append("-- Vendor --")
    lines.append(f"Name        : {invoice.vendor}")
    lines.append(f"Vendor ID   : {record.vendor_id or '—'}")
    if invoice.vendor_address:
        lines.append(f"Address     : {invoice.vendor_address}")
    lines.append("")
    lines.append("-- Invoice --")
    lines.append(f"Number      : {invoice.invoice_number}")
    lines.append(f"Date        : {invoice.date}")
    lines.append(f"Due date    : {invoice.due_date or '—'}")
    lines.append(f"Currency    : {invoice.currency}")
    lines.append(f"Terms       : {invoice.payment_terms or '—'}")
    lines.append("")
    lines.append("-- Line items --")
    if invoice.line_items:
        lines.append(f"{'Item':<24} {'Qty':>6} {'Unit':>14} {'Total':>14}")
        for li in invoice.line_items:
            lines.append(
                f"{li.item[:24]:<24} {li.quantity:>6} {str(li.unit_price):>14} "
                f"{str(li.line_total):>14}"
            )
    else:
        lines.append("(none)")
    lines.append("")
    lines.append("-- Totals --")
    lines.append(f"Subtotal    : {invoice.subtotal} {invoice.currency}")
    lines.append(f"Tax         : {invoice.tax} {invoice.currency}")
    lines.append(f"Total       : {invoice.total} {invoice.currency}")
    lines.append(f"Total (USD) : {record.amount_usd}")
    lines.append("")
    lines.append("-- Approval --")
    lines.append(f"Status      : {decision.status}")
    lines.append(f"Approver    : {decision.approver_role}")
    lines.append(f"Policy      : {decision.policy_id or '—'}")
    lines.append(f"Justification: {decision.justification}")
    if decision.escalations:
        lines.append(f"Escalations : {', '.join(decision.escalations)}")
    lines.append("")
    if record.notes:
        lines.append("-- Notes --")
        for n in record.notes:
            lines.append(f"- {n}")
        lines.append("")
    lines.append("=" * 60)
    lines.append("END OF RECEIPT")
    lines.append("=" * 60)
    return "\n".join(lines) + "\n"


def write_receipt(
    record: PaymentRecord,
    body: str,
    *,
    directory: Path = DEFAULT_RECEIPT_DIR,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{record.reference}.txt"
    if target.exists() and target.read_text() == body:
        return target
    target.write_text(body)
    return target
