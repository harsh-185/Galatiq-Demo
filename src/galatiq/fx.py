"""Fixed-rate FX conversions for the demo.

A real system would call a live rate feed (or read a daily snapshot from your
treasury system) and stamp the rate used onto the audit record. This module is
deliberately small: a constant table and one pure function.
"""
from __future__ import annotations

from decimal import Decimal

# Illustrative mid-market rates, USD per 1 unit of the currency.
# Update these manually if the demo dataset gains a new currency.
USD_PER_UNIT: dict[str, Decimal] = {
    "USD": Decimal("1.0000"),
    "EUR": Decimal("1.0800"),
    "GBP": Decimal("1.2700"),
    "JPY": Decimal("0.0067"),
    "CAD": Decimal("0.7400"),
    "AUD": Decimal("0.6600"),
}


def to_usd(amount: Decimal | float | int | str, currency: str) -> Decimal:
    """Convert ``amount`` in ``currency`` into a USD-equivalent Decimal."""
    rate = USD_PER_UNIT.get(currency.upper())
    if rate is None:
        raise ValueError(f"no FX rate configured for currency {currency!r}")
    return (Decimal(str(amount)) * rate).quantize(Decimal("0.01"))
