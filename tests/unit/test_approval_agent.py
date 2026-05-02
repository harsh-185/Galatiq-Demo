from __future__ import annotations

from decimal import Decimal

import pytest

from galatiq.agents.approval import approve
from galatiq.agents.validation import validate
from galatiq.db import connect, init_db
from galatiq.fx import USD_RATES
from galatiq.models.invoice import Invoice


def _invoice(**overrides):
    base = {
        "invoice_number": "INV-A1",
        "vendor": "Acme Corp",
        "date": "2026-01-01",
        "due_date": "2026-02-01",
        "currency": "USD",
        "line_items": [{"item": "WidgetA", "quantity": 2, "unit_price": "10.00"}],
        "subtotal": "20.00",
        "tax": "0.00",
        "total": "20.00",
    }
    base.update(overrides)
    return Invoice.model_validate(base)


@pytest.fixture
def db_conn(tmp_path):
    path = tmp_path / "inv.db"
    init_db(path)
    with connect(path) as conn:
        yield conn


def test_clean_small_invoice_auto_approved(db_conn):
    inv = _invoice()
    report = validate(inv, conn=db_conn)
    decision = approve(inv, report, conn=db_conn)
    assert decision.status == "auto_approved"
    assert decision.approver_role == "system"
    assert decision.policy_id == "TIER-AUTO"
    assert decision.total_usd == Decimal("20.00")


def test_manager_band_pending_human(db_conn):
    inv = _invoice(
        line_items=[{"item": "BoltPack", "quantity": 400, "unit_price": "5.00"}],
        subtotal="2000.00",
        total="2000.00",
    )
    report = validate(inv, conn=db_conn)
    decision = approve(inv, report, conn=db_conn)
    assert decision.status == "pending_human"
    assert decision.approver_role == "manager"
    assert decision.policy_id == "TIER-MGR"


def test_director_band_pending_human(db_conn):
    inv = _invoice(
        line_items=[{"item": "LaserCutterPro", "quantity": 1, "unit_price": "25000.00"}],
        subtotal="25000.00",
        total="25000.00",
    )
    report = validate(inv, conn=db_conn)
    decision = approve(inv, report, conn=db_conn)
    assert decision.approver_role == "director"
    assert decision.policy_id == "TIER-DIR"


def test_cfo_band_for_huge_invoice(db_conn):
    inv = _invoice(
        line_items=[{"item": "LaserCutterPro", "quantity": 3, "unit_price": "25000.00"}],
        subtotal="75000.00",
        total="75000.00",
    )
    report = validate(inv, conn=db_conn)
    decision = approve(inv, report, conn=db_conn)
    assert decision.approver_role == "cfo"
    assert decision.policy_id == "TIER-CFO"


def test_reject_short_circuits(db_conn):
    inv = _invoice(
        line_items=[{"item": "PhantomSKU", "quantity": 1, "unit_price": "9.99"}],
        subtotal="9.99",
        total="9.99",
    )
    report = validate(inv, conn=db_conn)
    decision = approve(inv, report, conn=db_conn)
    assert decision.status == "rejected"
    assert decision.approver_role == "none"
    assert decision.policy_id is None
    assert "fraud_flag_sku" in decision.escalations


def test_eur_invoice_normalizes_to_usd_for_tier_match(db_conn):
    # 950 EUR ~= 1026 USD at rate 1.08 → above the $1k auto threshold → manager
    inv = _invoice(
        currency="EUR",
        line_items=[{"item": "BoltPack", "quantity": 190, "unit_price": "5.00"}],
        subtotal="950.00",
        total="950.00",
    )
    report = validate(inv, conn=db_conn)
    decision = approve(inv, report, conn=db_conn)
    expected_usd = (Decimal("950") * USD_RATES["EUR"]).quantize(Decimal("0.01"))
    assert decision.total_usd == expected_usd
    assert decision.approver_role == "manager"


def test_needs_review_routes_to_human_even_below_auto(db_conn):
    # BoltPack catalog price $5; invoiced at $50 → +900% drift → warn → needs_review.
    # Total stays under the $1k auto threshold.
    inv = _invoice(
        line_items=[{"item": "BoltPack", "quantity": 1, "unit_price": "50.00"}],
        subtotal="50.00",
        total="50.00",
    )
    report = validate(inv, conn=db_conn)
    decision = approve(inv, report, conn=db_conn)
    assert decision.status == "pending_human"
    assert "price_drift_high" in decision.escalations
