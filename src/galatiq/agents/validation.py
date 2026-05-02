"""Validation agent: deterministic rule engine over (Invoice, reference DB).

Pure-function: never mutates the DB. The orchestrator decides whether to record a
ledger entry (only on verdict == 'pass').
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Iterable, Literal

from galatiq.db import (
    STATUS_BLOCKED,
    STATUS_DISCONTINUED,
    STATUS_FRAUD,
    STATUS_NEW,
    has_invoice,
    lookup_item,
    lookup_vendor,
)
from galatiq.models.invoice import Invoice

Severity = Literal["info", "warn", "error"]
Verdict = Literal["pass", "needs_review", "reject"]

_SEVERITY_RANK: dict[Severity, int] = {"info": 0, "warn": 1, "error": 2}

# Tunables — kept as constants so they can be lifted into config later.
PRICE_DRIFT_WARN_PCT = Decimal("25")  # >|25%| from catalog → warn


@dataclass(frozen=True)
class Finding:
    code: str
    severity: Severity
    message: str
    field: str | None = None


@dataclass
class ValidationReport:
    findings: list[Finding] = field(default_factory=list)
    verdict: Verdict = "pass"

    def add(self, finding: Finding) -> None:
        self.findings.append(finding)

    def by_severity(self) -> dict[Severity, list[Finding]]:
        out: dict[Severity, list[Finding]] = {"info": [], "warn": [], "error": []}
        for f in self.findings:
            out[f.severity].append(f)
        return out


def validate(invoice: Invoice, *, conn: sqlite3.Connection) -> ValidationReport:
    report = ValidationReport()
    _check_ingestion_warnings(invoice, report)
    _check_line_items(invoice, conn, report)
    _check_vendor(invoice, conn, report)
    _check_duplicate(invoice, conn, report)
    report.verdict = _derive_verdict(report.findings)
    return report


# --- rules --------------------------------------------------------------------

# Ingestion-level codes promoted up to validation severity.
_INGESTION_PROMOTIONS: dict[str, Severity] = {
    "subtotal_mismatch": "error",
    "total_mismatch": "error",
    "negative_tax": "error",
    "excessive_tax": "warn",
    "zero_unit_price": "warn",
    "zero_quantity": "warn",
    "implausible_date": "warn",
    "due_before_invoice_date": "warn",
    "empty_vendor": "error",
}


def _check_ingestion_warnings(invoice: Invoice, report: ValidationReport) -> None:
    for w in invoice.ingestion_warnings:
        severity: Severity = _INGESTION_PROMOTIONS.get(w.code, "info")
        report.add(Finding(code=w.code, severity=severity, message=w.message))


def _check_line_items(invoice: Invoice, conn: sqlite3.Connection, report: ValidationReport) -> None:
    for li in invoice.line_items:
        if li.quantity < 0:
            report.add(
                Finding(
                    code="negative_quantity",
                    severity="error",
                    message=f"line item {li.item!r} has negative quantity {li.quantity}",
                    field=li.item,
                )
            )

        item = lookup_item(conn, li.item)
        if item is None:
            report.add(
                Finding(
                    code="unknown_sku",
                    severity="warn",
                    message=f"item {li.item!r} not in inventory catalog",
                    field=li.item,
                )
            )
            continue

        if item.status == STATUS_FRAUD:
            report.add(
                Finding(
                    code="fraud_flag_sku",
                    severity="error",
                    message=f"item {li.item!r} is flagged as fraud_flag in catalog",
                    field=li.item,
                )
            )
        elif item.status == STATUS_DISCONTINUED:
            report.add(
                Finding(
                    code="discontinued_sku",
                    severity="warn",
                    message=f"item {li.item!r} is discontinued",
                    field=li.item,
                )
            )

        if li.quantity > 0:
            if item.stock == 0 and item.status != STATUS_FRAUD:
                report.add(
                    Finding(
                        code="zero_stock",
                        severity="error",
                        message=f"item {li.item!r} has zero stock; requested {li.quantity}",
                        field=li.item,
                    )
                )
            elif li.quantity > item.stock:
                report.add(
                    Finding(
                        code="stock_overflow",
                        severity="error",
                        message=(
                            f"requested {li.quantity} of {li.item!r} exceeds stock {item.stock}"
                        ),
                        field=li.item,
                    )
                )

        # Price drift vs catalog.
        if item.unit_price is not None and item.unit_price > 0:
            drift_pct = (li.unit_price - item.unit_price) / item.unit_price * Decimal("100")
            if abs(drift_pct) > PRICE_DRIFT_WARN_PCT:
                direction = "over" if drift_pct > 0 else "under"
                report.add(
                    Finding(
                        code="price_drift_high",
                        severity="warn",
                        message=(
                            f"unit_price {li.unit_price} is {abs(drift_pct):.1f}% {direction} "
                            f"catalog price {item.unit_price} for {li.item!r}"
                        ),
                        field=li.item,
                    )
                )


def _check_vendor(invoice: Invoice, conn: sqlite3.Connection, report: ValidationReport) -> None:
    if not invoice.vendor.strip():
        # already promoted by ingestion warning to error; don't double-log
        return
    vendor = lookup_vendor(conn, invoice.vendor)
    if vendor is None:
        report.add(
            Finding(
                code="vendor_unknown",
                severity="warn",
                message=f"vendor {invoice.vendor!r} not in vendor table",
                field="vendor",
            )
        )
        return
    if vendor.status == STATUS_BLOCKED:
        report.add(
            Finding(
                code="vendor_blocked",
                severity="error",
                message=f"vendor {vendor.name!r} ({vendor.vendor_id}) is blocked",
                field="vendor",
            )
        )
    elif vendor.status == STATUS_NEW:
        report.add(
            Finding(
                code="vendor_new",
                severity="warn",
                message=f"vendor {vendor.name!r} is first-time; escalate before payment",
                field="vendor",
            )
        )
    if vendor.default_currency and vendor.default_currency != invoice.currency:
        report.add(
            Finding(
                code="currency_drift",
                severity="warn",
                message=(
                    f"invoice currency {invoice.currency} differs from vendor default "
                    f"{vendor.default_currency}"
                ),
                field="currency",
            )
        )


def _check_duplicate(invoice: Invoice, conn: sqlite3.Connection, report: ValidationReport) -> None:
    if not (invoice.invoice_number and invoice.vendor):
        return
    if has_invoice(conn, invoice.invoice_number, invoice.vendor):
        report.add(
            Finding(
                code="duplicate_invoice",
                severity="error",
                message=(
                    f"invoice {invoice.invoice_number!r} from {invoice.vendor!r} already in ledger"
                ),
                field="invoice_number",
            )
        )


def _derive_verdict(findings: Iterable[Finding]) -> Verdict:
    worst = max((_SEVERITY_RANK[f.severity] for f in findings), default=-1)
    if worst >= _SEVERITY_RANK["error"]:
        return "reject"
    if worst >= _SEVERITY_RANK["warn"]:
        return "needs_review"
    return "pass"
