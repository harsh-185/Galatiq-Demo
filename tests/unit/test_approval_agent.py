from __future__ import annotations

from decimal import Decimal

import pytest

from galatiq.agents.approval import approve
from galatiq.agents.validation import validate
from galatiq.db import connect, init_db, list_recent_decisions
from galatiq.models.invoice import Invoice


@pytest.fixture
def db_conn(tmp_path):
    path = tmp_path / "inv.db"
    init_db(path)
    with connect(path) as conn:
        yield conn


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


def _approve(inv, conn):
    report = validate(inv, conn=conn)
    return validate_then_approve(inv, report, conn)


def validate_then_approve(inv, report, conn):
    return approve(inv, report, conn=conn, allow_llm=False)


def test_clean_small_invoice_auto_approves(db_conn):
    decision = _approve(_invoice(), db_conn)
    assert decision.status == "auto_approved"
    assert decision.approver_role is None
    assert decision.policy_id is not None
    rows = list_recent_decisions(db_conn)
    assert rows[0]["status"] == "auto_approved"


def test_clean_invoice_in_manager_band_routes_to_manager(db_conn):
    inv = _invoice(
        line_items=[{"item": "WidgetA", "quantity": 10, "unit_price": "250.00"}],
        subtotal="2500.00",
        total="2500.00",
    )
    decision = _approve(inv, db_conn)
    assert decision.status == "pending_human"
    assert decision.approver_role == "manager"


def test_clean_invoice_in_director_band_routes_to_director(db_conn):
    inv = _invoice(
        invoice_number="INV-DIR",
        line_items=[
            {"item": "WidgetA", "quantity": 15, "unit_price": "250.00"},
            {"item": "WidgetB", "quantity": 10, "unit_price": "500.00"},
            {"item": "GadgetX", "quantity": 5, "unit_price": "750.00"},
        ],
        subtotal="12500.00",
        total="12500.00",
    )
    decision = _approve(inv, db_conn)
    assert decision.status == "pending_human"
    assert decision.approver_role == "director"


def test_clean_invoice_in_cfo_band_routes_to_cfo(db_conn):
    inv = _invoice(
        invoice_number="INV-CFO",
        line_items=[{"item": "LaserCutterPro", "quantity": 3, "unit_price": "25000.00"}],
        subtotal="75000.00",
        total="75000.00",
    )
    decision = _approve(inv, db_conn)
    assert decision.status == "pending_human"
    assert decision.approver_role == "cfo"


def test_reject_short_circuits_with_no_approver(db_conn):
    inv = _invoice(
        invoice_number="INV-REJ",
        line_items=[{"item": "WidgetA", "quantity": -1, "unit_price": "250.00"}],
        subtotal="-250.00",
        total="-250.00",
    )
    report = validate(inv, conn=db_conn)
    assert report.verdict == "reject"
    decision = approve(inv, report, conn=db_conn, allow_llm=False)
    assert decision.status == "rejected"
    assert decision.approver_role is None
    rows = list_recent_decisions(db_conn)
    assert rows[0]["status"] == "rejected"


def test_needs_review_routes_with_escalations(db_conn):
    inv = _invoice(
        invoice_number="INV-NR",
        vendor="Widgets Inc.",
        line_items=[{"item": "SuperGizmo", "quantity": 1, "unit_price": "10.00"}],
        subtotal="10.00",
        total="10.00",
    )
    report = validate(inv, conn=db_conn)
    assert report.verdict == "needs_review"
    decision = approve(inv, report, conn=db_conn, allow_llm=False)
    assert decision.status == "pending_human"
    assert "unknown_sku" in decision.escalations


def test_eur_is_normalized_for_band_lookup(db_conn):
    # 1000 EUR * 1.08 = 1080 USD -> manager band
    inv = _invoice(
        invoice_number="INV-EUR",
        vendor="TechParts International",
        currency="EUR",
        line_items=[{"item": "WidgetA", "quantity": 4, "unit_price": "250.00"}],
        subtotal="1000.00",
        total="1000.00",
    )
    report = validate(inv, conn=db_conn)
    decision = approve(inv, report, conn=db_conn, allow_llm=False)
    assert decision.total_usd == Decimal("1080.00")
    assert decision.approver_role == "manager"


def test_audit_log_records_every_decision(db_conn):
    # Auto-approve
    approve(_invoice(invoice_number="A1"), validate(_invoice(invoice_number="A1"), conn=db_conn), conn=db_conn, allow_llm=False)
    # Reject
    bad = _invoice(
        invoice_number="A2",
        line_items=[{"item": "WidgetA", "quantity": -1, "unit_price": "250.00"}],
        subtotal="-250.00",
        total="-250.00",
    )
    approve(bad, validate(bad, conn=db_conn), conn=db_conn, allow_llm=False)
    rows = list_recent_decisions(db_conn)
    assert len(rows) == 2
    assert {r["status"] for r in rows} == {"auto_approved", "rejected"}
