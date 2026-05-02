from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field, model_validator

Currency = Literal["USD", "EUR", "GBP", "JPY", "CAD", "AUD"]

_DATE_LOWER_BOUND = date(2000, 1, 1)
_DATE_UPPER_BOUND_DELTA_DAYS = 365 * 5  # ~5 years past today


class LineItem(BaseModel):
    item: str = Field(description="Item name / SKU as it appears on the invoice")
    quantity: int = Field(description="Quantity ordered (may be negative for credits)")
    unit_price: Decimal = Field(description="Per-unit price in the invoice currency")

    @property
    def line_total(self) -> Decimal:
        return self.unit_price * self.quantity


class IngestionWarning(BaseModel):
    code: str
    message: str


class Invoice(BaseModel):
    invoice_number: str
    vendor: str
    vendor_address: str | None = None
    date: date
    due_date: date | None = None
    currency: Currency = "USD"
    line_items: list[LineItem]
    subtotal: Decimal
    tax: Decimal = Decimal("0")
    total: Decimal
    payment_terms: str | None = None
    source_path: str | None = None
    ingestion_warnings: list[IngestionWarning] = Field(default_factory=list)

    @model_validator(mode="after")
    def _attach_ingestion_warnings(self) -> "Invoice":
        # Cross-check totals.
        computed_subtotal = sum((li.line_total for li in self.line_items), Decimal("0"))
        if abs(computed_subtotal - self.subtotal) > Decimal("0.01"):
            self._warn(
                "subtotal_mismatch",
                f"line items sum to {computed_subtotal} but invoice says {self.subtotal}",
            )
        computed_total = self.subtotal + self.tax
        if abs(computed_total - self.total) > Decimal("0.01"):
            self._warn(
                "total_mismatch",
                f"subtotal+tax = {computed_total} but invoice says {self.total}",
            )

        # Per-line anomalies — preserve the values, just flag.
        for li in self.line_items:
            if li.quantity == 0:
                self._warn("zero_quantity", f"line item {li.item!r} has quantity 0")
            if li.unit_price == 0:
                self._warn("zero_unit_price", f"line item {li.item!r} has unit_price 0")

        # Tax plausibility.
        if self.tax < 0:
            self._warn("negative_tax", f"tax is negative ({self.tax})")
        elif self.subtotal > 0 and self.tax > self.subtotal / 2:
            self._warn(
                "excessive_tax",
                f"tax {self.tax} exceeds 50% of subtotal {self.subtotal}",
            )

        # Date sanity.
        today = datetime.now(timezone.utc).date()
        upper = date.fromordinal(today.toordinal() + _DATE_UPPER_BOUND_DELTA_DAYS)
        if self.date < _DATE_LOWER_BOUND or self.date > upper:
            self._warn("implausible_date", f"invoice date {self.date} is outside plausible range")
        if self.due_date is not None and self.due_date < self.date:
            self._warn(
                "due_before_invoice_date",
                f"due_date {self.due_date} is before invoice date {self.date}",
            )

        # Vendor sanity (file-level only — DB lookup happens in validation phase).
        if not self.vendor.strip():
            self._warn("empty_vendor", "vendor field is empty")

        return self

    def _warn(self, code: str, message: str) -> None:
        self.ingestion_warnings.append(IngestionWarning(code=code, message=message))
