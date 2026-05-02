from decimal import Decimal

import pytest

from galatiq.fx import to_usd


def test_usd_is_identity():
    assert to_usd("100.00", "USD") == Decimal("100.00")


def test_eur_converts():
    # 100 EUR * 1.08 = 108.00
    assert to_usd("100.00", "EUR") == Decimal("108.00")


def test_jpy_converts_with_small_rate():
    assert to_usd("10000", "JPY") == Decimal("67.00")


def test_lowercase_currency_is_accepted():
    assert to_usd("50", "eur") == Decimal("54.00")


def test_unknown_currency_raises():
    with pytest.raises(ValueError):
        to_usd("10", "ZZZ")
