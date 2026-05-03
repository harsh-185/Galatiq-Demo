from __future__ import annotations

import json
from pathlib import Path

import pytest

from galatiq.agents.pipeline import run_pipeline
from galatiq.db import connect, init_db, list_approvals, list_payments


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload))
    return path


@pytest.fixture
def db_path(tmp_path):
    p = tmp_path / "pipeline.db"
    init_db(p)
    return p


@pytest.fixture
def receipt_dir(tmp_path):
    return tmp_path / "receipts"


def test_clean_invoice_full_pipeline(tmp_path, db_path, receipt_dir):
    inv_path = _write(
        tmp_path / "clean.json",
        {
            "invoice_number": "INV-CLEAN",
            "vendor": "Acme Corp",
            "date": "2026-01-01",
            "due_date": "2099-01-01",
            "currency": "USD",
            "line_items": [{"item": "WidgetA", "quantity": 2, "unit_price": "10.00"}],
            "subtotal": "20.00",
            "tax": "0.00",
            "total": "20.00",
        },
    )
    state = run_pipeline(inv_path, db_path=db_path, receipt_dir=receipt_dir)
    assert not state.get("errors")
    assert state["report"].verdict == "pass"
    assert state["decision"].status == "auto_approved"
    assert state["payment"].status == "scheduled"
    assert state["payment"].rail == "ach"
    assert state["payment"].receipt_path is not None
    assert Path(state["payment"].receipt_path).exists()

    with connect(db_path) as conn:
        approvals = list_approvals(conn)
        payments = list_payments(conn)
    assert len(approvals) == 1
    assert approvals[0]["status"] == "auto_approved"
    assert len(payments) == 1
    assert payments[0]["rail"] == "ach"


def test_rejected_invoice_skips_payment(tmp_path, db_path, receipt_dir):
    inv_path = _write(
        tmp_path / "reject.json",
        {
            "invoice_number": "INV-REJ",
            "vendor": "Acme Corp",
            "date": "2026-01-01",
            "currency": "USD",
            "line_items": [{"item": "PhantomSKU", "quantity": 1, "unit_price": "9.99"}],
            "subtotal": "9.99",
            "tax": "0.00",
            "total": "9.99",
        },
    )
    state = run_pipeline(inv_path, db_path=db_path, receipt_dir=receipt_dir)
    assert state["report"].verdict == "reject"
    assert state["decision"].status == "rejected"
    assert state["payment"].status == "skipped"
    assert state["payment"].rail == "none"
    assert state["payment"].receipt_path is None
    assert not receipt_dir.exists() or not list(receipt_dir.iterdir())


def test_eur_invoice_routed_by_usd_total(tmp_path, db_path, receipt_dir):
    # 9300 EUR × 1.08 ≈ 10044 USD → above the $10k auto threshold → manager
    inv_path = _write(
        tmp_path / "eur.json",
        {
            "invoice_number": "INV-EUR",
            "vendor": "Beta Industries",
            "date": "2026-01-01",
            "currency": "EUR",
            "line_items": [{"item": "WidgetA", "quantity": 15, "unit_price": "620.00"}],
            "subtotal": "9300.00",
            "tax": "0.00",
            "total": "9300.00",
        },
    )
    state = run_pipeline(inv_path, db_path=db_path, receipt_dir=receipt_dir)
    assert state["decision"].approver_role == "manager"
    assert state["payment"].status == "skipped"


def test_second_run_blocks_duplicate_payment(tmp_path, db_path, receipt_dir):
    inv_path = _write(
        tmp_path / "idem.json",
        {
            "invoice_number": "INV-IDEM",
            "vendor": "Acme Corp",
            "date": "2026-01-01",
            "due_date": "2099-01-01",
            "currency": "USD",
            "line_items": [{"item": "WidgetA", "quantity": 1, "unit_price": "10.00"}],
            "subtotal": "10.00",
            "tax": "0.00",
            "total": "10.00",
        },
    )
    first = run_pipeline(inv_path, db_path=db_path, receipt_dir=receipt_dir)
    second = run_pipeline(inv_path, db_path=db_path, receipt_dir=receipt_dir)
    # First run schedules; second sees the ledger duplicate and rejects → skipped.
    assert first["payment"].status == "scheduled"
    assert first["payment"].rail == "ach"
    assert second["payment"].status == "skipped"
    assert second["decision"].status == "rejected"
    assert "duplicate_invoice" in second["decision"].escalations
