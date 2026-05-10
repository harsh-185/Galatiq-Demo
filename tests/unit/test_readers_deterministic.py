"""Deterministic-only ingestion tests across the cleanly-structured formats."""
from pathlib import Path

import pytest

from galatiq.agents.ingestion import ingest

DATA = Path(__file__).resolve().parents[2] / "data" / "invoices"


@pytest.mark.parametrize("filename, expected_inv, expected_li", [
    ("invoice_1004.json", "INV-1004", 2),
    ("invoice_1004_revised.json", "INV-1004", 3),
    ("invoice_1005.json", "INV-1005", 3),
    ("invoice_1006.csv", "INV-1006", 2),
    ("invoice_1007.csv", "INV-1007", 3),  # MM/DD/YYYY dates → normalized to ISO, no LLM needed
    ("invoice_1009.json", "INV-1009", 2),
    ("invoice_1013.json", "INV-1013", 8),
    ("invoice_1014.xml", "INV-1014", 2),
    ("invoice_1015.csv", "INV-1015", 3),
    ("invoice_1016.json", "INV-1016", 3),
])
def test_deterministic_path(filename, expected_inv, expected_li):
    result = ingest(DATA / filename, allow_llm=False)
    assert result.path_taken == "deterministic"
    assert result.invoice.invoice_number == expected_inv
    assert len(result.invoice.line_items) == expected_li


def test_negative_quantity_preserved():
    result = ingest(DATA / "invoice_1009.json", allow_llm=False)
    qtys = [li.quantity for li in result.invoice.line_items]
    assert -5 in qtys


def test_eur_currency_preserved():
    result = ingest(DATA / "invoice_1014.xml", allow_llm=False)
    assert result.invoice.currency == "EUR"


def test_empty_vendor_is_passed_through():
    result = ingest(DATA / "invoice_1009.json", allow_llm=False)
    assert result.invoice.vendor == ""


def test_us_date_format_is_normalized():
    """Regression: CSV/JSON/XML files using MM/DD/YYYY now ingest
    deterministically instead of falling through to the LLM path."""
    from datetime import date
    result = ingest(DATA / "invoice_1007.csv", allow_llm=False)
    assert result.path_taken == "deterministic"
    assert result.invoice.date == date(2026, 1, 28)
    assert result.invoice.due_date == date(2026, 2, 28)


def test_normalize_date_helper_accepts_both_formats():
    from galatiq.io.readers import _normalize_date
    assert _normalize_date("01/28/2026") == "2026-01-28"
    assert _normalize_date("1/2/2026") == "2026-01-02"
    assert _normalize_date("2026-01-28") == "2026-01-28"
    assert _normalize_date("2026/01/28") == "2026-01-28"
    assert _normalize_date("01-28-2026") == "2026-01-28"
    assert _normalize_date(None) is None
    assert _normalize_date("") is None
    # Garbage stays garbage; Pydantic will surface a clean error downstream.
    assert _normalize_date("yesterday") == "yesterday"
