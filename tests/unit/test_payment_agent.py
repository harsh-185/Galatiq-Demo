from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

import pytest

from galatiq.agents.approval import ApprovalDecision
from galatiq.agents.payment import pay
from galatiq.db import STATUS_ACTIVE, STATUS_BLOCKED, Vendor
from galatiq.models.invoice import Invoice


def _invoice(**overrides):
    base = {
        "invoice_number": "INV-P1",
        "vendor": "Acme Corp",
        "date": "2026-01-01",
        "due_date": "2026-03-01",
        "currency": "USD",
        "line_items": [{"item": "WidgetA", "quantity": 2, "unit_price": "10.00"}],
        "subtotal": "20.00",
        "tax": "0.00",
        "total": "20.00",
    }
    base.update(overrides)
    return Invoice.model_validate(base)


def _vendor(**overrides):
    base = {
        "vendor_id": "VEND-001",
        "name": "Acme Corp",
        "aliases": [],
        "address": None,
        "status": STATUS_ACTIVE,
        "default_currency": "USD",
    }
    base.update(overrides)
    return Vendor(**base)


def _decision(status="auto_approved", total_usd=Decimal("20.00"), **kw):
    return ApprovalDecision(
        status=status,
        approver_role=kw.get("approver_role", "system"),
        policy_id=kw.get("policy_id", "TIER-AUTO"),
        total_usd=total_usd,
        justification=kw.get("justification", "ok"),
        escalations=kw.get("escalations", []),
    )


def test_ach_for_small_amount():
    record = pay(_invoice(), _decision(total_usd=Decimal("100")), vendor=_vendor())
    assert record.status == "scheduled"
    assert record.rail == "ach"


def test_wire_for_mid_amount():
    record = pay(_invoice(), _decision(total_usd=Decimal("7500")), vendor=_vendor())
    assert record.rail == "wire"


def test_check_for_large_amount():
    record = pay(_invoice(), _decision(total_usd=Decimal("75000")), vendor=_vendor())
    assert record.rail == "check"


def test_blocked_vendor_fails_payment():
    record = pay(
        _invoice(),
        _decision(),
        vendor=_vendor(status=STATUS_BLOCKED, vendor_id="VEND-003", name="ShadyVendor LLC"),
    )
    assert record.status == "failed"
    assert record.rail == "none"


def test_rejected_decision_skips():
    record = pay(_invoice(), _decision(status="rejected"), vendor=_vendor())
    assert record.status == "skipped"
    assert record.rail == "none"


def test_pending_human_decision_skips():
    record = pay(_invoice(), _decision(status="pending_human"), vendor=_vendor())
    assert record.status == "skipped"
    assert record.rail == "none"


def test_reference_is_deterministic():
    inv = _invoice()
    dec = _decision()
    a = pay(inv, dec, vendor=_vendor())
    b = pay(inv, dec, vendor=_vendor())
    assert a.reference == b.reference
    assert a.reference.startswith("PAY-VEND-001-INV-P1-")


def test_scheduled_for_uses_due_date_when_future():
    inv = _invoice(due_date="2099-12-31")
    record = pay(inv, _decision(), vendor=_vendor(), today=date(2026, 1, 1))
    assert record.scheduled_for == date(2099, 12, 31)


def test_scheduled_for_falls_back_when_due_date_past():
    inv = _invoice(date="2020-01-01", due_date="2020-02-01")
    today = date(2026, 1, 1)
    record = pay(inv, _decision(), vendor=_vendor(), today=today)
    assert record.scheduled_for == today + timedelta(days=1)


def test_no_vendor_still_routes():
    record = pay(_invoice(), _decision(total_usd=Decimal("100")), vendor=None)
    assert record.status == "scheduled"
    assert record.vendor_id is None
    assert "UNKNOWN" in record.reference
