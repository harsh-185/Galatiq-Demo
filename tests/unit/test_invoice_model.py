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


def test_empty_vendor_passes_ingestion():
    inv = Invoice.model_validate(_base(vendor=""))
    assert inv.vendor == ""  # validation phase will flag this
