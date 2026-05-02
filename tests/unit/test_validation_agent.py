from __future__ import annotations

from decimal import Decimal

import pytest

from galatiq.agents.validation import validate
from galatiq.db import connect, init_db, record_invoice
from galatiq.models.invoice import Invoice


def _invoice(**overrides):
    base = {
        "invoice_number": "INV-T1",
        "vendor": "Widgets Inc.",
        "date": "2026-01-01",
        "due_date": "2026-02-01",
        "currency": "USD",
        "line_items": [{"item": "WidgetA", "quantity": 2, "unit_price": "250.00"}],
        "subtotal": "500.00",
        "tax": "0.00",
        "total": "500.00",
    }
    base.update(overrides)
    return Invoice.model_validate(base)


@pytest.fixture
def db_conn(tmp_path):
    path = tmp_path / "inv.db"
    init_db(path)
    with connect(path) as conn:
        yield conn


def _codes(report):
    return [f.code for f in report.findings]


def test_clean_invoice_passes(db_conn):
    report = validate(_invoice(), conn=db_conn)
    assert report.verdict == "pass"
    assert report.findings == []


def test_stock_overflow_rejects(db_conn):
    inv = _invoice(
        line_items=[{"item": "GadgetX", "quantity": 20, "unit_price": "750.00"}],
        subtotal="15000.00",
        total="15000.00",
    )
    report = validate(inv, conn=db_conn)
    assert "stock_overflow" in _codes(report)
    assert report.verdict == "reject"


def test_fraud_flag_sku_rejects(db_conn):
    inv = _invoice(
        line_items=[{"item": "FakeItem", "quantity": 1, "unit_price": "9.99"}],
        subtotal="9.99",
        total="9.99",
    )
    report = validate(inv, conn=db_conn)
    assert "fraud_flag_sku" in _codes(report)
    assert report.verdict == "reject"


def test_unknown_sku_warns(db_conn):
    inv = _invoice(
        line_items=[{"item": "SuperGizmo", "quantity": 1, "unit_price": "10.00"}],
        subtotal="10.00",
        total="10.00",
    )
    report = validate(inv, conn=db_conn)
    assert "unknown_sku" in _codes(report)
    assert report.verdict == "needs_review"


def test_negative_quantity_rejects(db_conn):
    inv = _invoice(
        line_items=[{"item": "WidgetA", "quantity": -3, "unit_price": "250.00"}],
        subtotal="-750.00",
        total="-750.00",
    )
    report = validate(inv, conn=db_conn)
    assert "negative_quantity" in _codes(report)
    assert report.verdict == "reject"


def test_discontinued_sku_warns(db_conn):
    inv = _invoice(
        line_items=[{"item": "GizmoPro", "quantity": 1, "unit_price": "200.00"}],
        subtotal="200.00",
        total="200.00",
    )
    report = validate(inv, conn=db_conn)
    assert "discontinued_sku" in _codes(report)
    assert report.verdict == "needs_review"


def test_price_drift_warns(db_conn):
    inv = _invoice(
        line_items=[{"item": "WidgetA", "quantity": 1, "unit_price": "1000.00"}],  # catalog $250
        subtotal="1000.00",
        total="1000.00",
    )
    report = validate(inv, conn=db_conn)
    assert "price_drift_high" in _codes(report)
    assert report.verdict == "needs_review"


def test_blocked_vendor_rejects(db_conn):
    inv = _invoice(vendor="ShadyVendor LLC")
    report = validate(inv, conn=db_conn)
    assert "vendor_blocked" in _codes(report)
    assert report.verdict == "reject"


def test_unknown_vendor_warns(db_conn):
    inv = _invoice(vendor="Mystery Vendor")
    report = validate(inv, conn=db_conn)
    assert "vendor_unknown" in _codes(report)
    assert report.verdict == "needs_review"


def test_new_vendor_warns(db_conn):
    inv = _invoice(vendor="NewCo")
    report = validate(inv, conn=db_conn)
    assert "vendor_new" in _codes(report)
    assert report.verdict == "needs_review"


def test_alias_match_treats_vendor_as_known(db_conn):
    inv = _invoice(vendor="Widgets")  # alias of Widgets Inc.
    report = validate(inv, conn=db_conn)
    assert "vendor_unknown" not in _codes(report)
    assert "vendor_blocked" not in _codes(report)


def test_currency_drift_warns(db_conn):
    inv = _invoice(currency="EUR")
    report = validate(inv, conn=db_conn)
    assert "currency_drift" in _codes(report)
    assert report.verdict == "needs_review"


def test_duplicate_invoice_rejects(db_conn):
    record_invoice(
        db_conn,
        invoice_number="INV-T1",
        vendor="Widgets Inc.",
        total=Decimal("500.00"),
    )
    report = validate(_invoice(), conn=db_conn)
    assert "duplicate_invoice" in _codes(report)
    assert report.verdict == "reject"


def test_subtotal_mismatch_promoted_to_error(db_conn):
    # raw 1 × $250 = $250, but invoice claims subtotal=$500 → math warning from ingestion
    inv = _invoice(
        line_items=[{"item": "WidgetA", "quantity": 1, "unit_price": "250.00"}],
        subtotal="500.00",
        total="500.00",
    )
    report = validate(inv, conn=db_conn)
    assert "subtotal_mismatch" in _codes(report)
    assert report.verdict == "reject"
