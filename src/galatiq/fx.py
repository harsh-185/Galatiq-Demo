"""FX conversion to USD.

Static rate table — adequate for a deterministic demo. A production system would
fetch rates per invoice date from a feed; this stays offline and reproducible.
"""
from __future__ import annotations

from decimal import Decimal

USD_RATES: dict[str, Decimal] = {
    "USD": Decimal("1"),
    "EUR": Decimal("1.08"),
    "GBP": Decimal("1.27"),
    "JPY": Decimal("0.0067"),
    "CAD": Decimal("0.74"),
    "AUD": Decimal("0.66"),
}


def to_usd(amount: Decimal, currency: str) -> Decimal:
    """Convert ``amount`` in ``currency`` to USD using the static rate table.

    Raises ``ValueError`` for unknown currencies.
    """
    rate = USD_RATES.get(currency.upper())
    if rate is None:
        raise ValueError(f"unknown currency {currency!r}; expected one of {sorted(USD_RATES)}")
    return (Decimal(amount) * rate).quantize(Decimal("0.01"))
