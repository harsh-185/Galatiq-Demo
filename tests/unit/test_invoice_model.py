from decimal import Decimal

from galatiq.models.invoice import Invoice, LineItem


def _base(**overrides):
    return {
        "invoice_number": "INV-X",
        "vendor": "Test Vendor",
        "date": "2026-01-01",
        "due_date": "2026-02-01",
        "line_items": [{"item": "WidgetA", "quantity": 2, "unit_price": "100.00"}],
        "subtotal": "200.00",
        "tax": "0.00",
        "total": "200.00",
        **overrides,
    }


def test_clean_invoice_has_no_warnings():
    inv = Invoice.model_validate(_base())
    assert inv.ingestion_warnings == []
    assert inv.line_items[0].line_total == Decimal("200")


def test_subtotal_mismatch_raises_warning():
    inv = Invoice.model_validate(_base(subtotal="999.00", total="999.00", tax="0.00"))
    codes = [w.code for w in inv.ingestion_warnings]
    assert "subtotal_mismatch" in codes


def test_total_mismatch_raises_warning():
    inv = Invoice.model_validate(_base(total="500.00"))
    codes = [w.code for w in inv.ingestion_warnings]
    assert "total_mismatch" in codes


def test_negative_quantity_is_preserved():
    data = _base(line_items=[{"item": "WidgetA", "quantity": -5, "unit_price": "250.00"}],
                 subtotal="-1250.00", total="-1250.00")
    inv = Invoice.model_validate(data)
    assert inv.line_items[0].quantity == -5
    assert inv.total == Decimal("-1250.00")


def test_empty_vendor_emits_warning_but_does_not_fail():
    inv = Invoice.model_validate(_base(vendor=""))
    assert inv.vendor == ""
    codes = [w.code for w in inv.ingestion_warnings]
    assert "empty_vendor" in codes


def test_zero_quantity_warning():
    inv = Invoice.model_validate(
        _base(
            line_items=[{"item": "WidgetA", "quantity": 0, "unit_price": "100.00"}],
            subtotal="0.00",
            total="0.00",
        )
    )
    codes = [w.code for w in inv.ingestion_warnings]
    assert "zero_quantity" in codes


def test_zero_unit_price_warning():
    inv = Invoice.model_validate(
        _base(
            line_items=[{"item": "WidgetA", "quantity": 5, "unit_price": "0.00"}],
            subtotal="0.00",
            total="0.00",
        )
    )
    codes = [w.code for w in inv.ingestion_warnings]
    assert "zero_unit_price" in codes


def test_negative_tax_warning():
    inv = Invoice.model_validate(_base(tax="-10.00", total="190.00"))
    codes = [w.code for w in inv.ingestion_warnings]
    assert "negative_tax" in codes


def test_excessive_tax_warning():
    inv = Invoice.model_validate(_base(tax="150.00", total="350.00"))
    codes = [w.code for w in inv.ingestion_warnings]
    assert "excessive_tax" in codes


def test_implausible_date_warning():
    inv = Invoice.model_validate(_base(date="1995-06-01", due_date="1995-07-01"))
    codes = [w.code for w in inv.ingestion_warnings]
    assert "implausible_date" in codes


def test_due_before_invoice_date_warning():
    inv = Invoice.model_validate(_base(date="2026-03-01", due_date="2026-02-01"))
    codes = [w.code for w in inv.ingestion_warnings]
    assert "due_before_invoice_date" in codes
