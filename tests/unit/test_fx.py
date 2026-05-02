from __future__ import annotations

from decimal import Decimal

import pytest

from galatiq.fx import USD_RATES, to_usd


def test_usd_passthrough():
    assert to_usd(Decimal("100.00"), "USD") == Decimal("100.00")


def test_eur_to_usd_uses_table_rate():
    expected = (Decimal("100") * USD_RATES["EUR"]).quantize(Decimal("0.01"))
    assert to_usd(Decimal("100"), "EUR") == expected


@pytest.mark.parametrize("currency", ["GBP", "JPY", "CAD", "AUD"])
def test_supported_currencies_round_trip(currency):
    out = to_usd(Decimal("1000"), currency)
    assert out > 0
    assert out.as_tuple().exponent == -2  # quantized to 2 decimal places


def test_currency_case_insensitive():
    assert to_usd(Decimal("1"), "usd") == Decimal("1.00")


def test_unknown_currency_raises():
    with pytest.raises(ValueError):
        to_usd(Decimal("1"), "BTC")
