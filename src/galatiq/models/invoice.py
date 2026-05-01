from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field, model_validator

Currency = Literal["USD", "EUR", "GBP", "JPY", "CAD", "AUD"]


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
    def _attach_math_warnings(self) -> "Invoice":
        # Cross-check totals; do not fail — surface as warnings the validation phase weighs.
        computed_subtotal = sum((li.line_total for li in self.line_items), Decimal("0"))
        if abs(computed_subtotal - self.subtotal) > Decimal("0.01"):
            self.ingestion_warnings.append(
                IngestionWarning(
                    code="subtotal_mismatch",
                    message=f"line items sum to {computed_subtotal} but invoice says {self.subtotal}",
                )
            )
        computed_total = self.subtotal + self.tax
        if abs(computed_total - self.total) > Decimal("0.01"):
            self.ingestion_warnings.append(
                IngestionWarning(
                    code="total_mismatch",
                    message=f"subtotal+tax = {computed_total} but invoice says {self.total}",
                )
            )
        return self
