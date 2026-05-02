from __future__ import annotations

from decimal import Decimal

from galatiq.db import (
    SEED_INVENTORY,
    SEED_VENDORS,
    STATUS_ACTIVE,
    STATUS_BLOCKED,
    STATUS_DISCONTINUED,
    STATUS_FRAUD,
    connect,
    get_stock,
    has_invoice,
    init_db,
    init_schema,
    list_inventory,
    list_items,
    list_vendors,
    lookup_item,
    lookup_vendor,
    record_invoice,
    seed_defaults,
)


def test_init_db_creates_file_and_seeds(tmp_path):
    db_path = tmp_path / "inv.db"
    returned = init_db(db_path)
    assert returned == db_path
    assert db_path.exists()
    with connect(db_path) as conn:
        items = dict(list_items(conn))
        vendors = list_vendors(conn)
    assert len(items) == len(SEED_INVENTORY)
    assert items["WidgetA"] == 15
    assert items["FakeItem"] == 0
    assert {v.vendor_id for v in vendors} == {v["vendor_id"] for v in SEED_VENDORS}


def test_lookup_item_returns_full_row(tmp_path):
    db_path = tmp_path / "inv.db"
    init_db(db_path)
    with connect(db_path) as conn:
        widget = lookup_item(conn, "WidgetA")
        fake = lookup_item(conn, "FakeItem")
        gizmo = lookup_item(conn, "GizmoPro")
        unknown = lookup_item(conn, "DoesNotExist")
    assert widget is not None and widget.unit_price == Decimal("10.00") and widget.status == STATUS_ACTIVE
    assert fake is not None and fake.status == STATUS_FRAUD and fake.unit_price is None
    assert gizmo is not None and gizmo.status == STATUS_DISCONTINUED
    assert unknown is None


def test_list_inventory_includes_high_value_item(tmp_path):
    db_path = tmp_path / "inv.db"
    init_db(db_path)
    with connect(db_path) as conn:
        rows = {i.item: i for i in list_inventory(conn)}
    assert "LaserCutterPro" in rows
    assert rows["LaserCutterPro"].unit_price == Decimal("25000.00")


def test_lookup_vendor_matches_canonical_and_alias(tmp_path):
    db_path = tmp_path / "inv.db"
    init_db(db_path)
    with connect(db_path) as conn:
        canonical = lookup_vendor(conn, "Acme Corp")
        alias = lookup_vendor(conn, "ACME")
        case_insensitive = lookup_vendor(conn, "acme corporation")
        unknown = lookup_vendor(conn, "Mystery Vendor")
    assert canonical is not None and canonical.vendor_id == "VEND-001"
    assert alias is not None and alias.vendor_id == "VEND-001"
    assert case_insensitive is not None and case_insensitive.vendor_id == "VEND-001"
    assert unknown is None


def test_lookup_vendor_surfaces_blocked_status(tmp_path):
    db_path = tmp_path / "inv.db"
    init_db(db_path)
    with connect(db_path) as conn:
        v = lookup_vendor(conn, "ShadyVendor LLC")
    assert v is not None and v.status == STATUS_BLOCKED


def test_invoice_ledger_dedup(tmp_path):
    db_path = tmp_path / "inv.db"
    init_db(db_path)
    with connect(db_path) as conn:
        assert has_invoice(conn, "INV-1001", "Acme Corp") is False
        record_invoice(
            conn,
            invoice_number="INV-1001",
            vendor="Acme Corp",
            total=Decimal("250.00"),
            source_path="data/invoices/invoice_1001.txt",
        )
        assert has_invoice(conn, "INV-1001", "Acme Corp") is True
        assert has_invoice(conn, "INV-1001", "Beta Industries") is False


def test_seed_is_idempotent(tmp_path):
    db_path = tmp_path / "inv.db"
    init_db(db_path)
    with connect(db_path) as conn:
        conn.execute("UPDATE inventory SET stock = ? WHERE item = ?", (99, "WidgetA"))
        conn.commit()
        inserted = seed_defaults(conn)
        assert inserted == 0
        assert get_stock(conn, "WidgetA") == 99


def test_init_schema_is_safe_to_call_twice(tmp_path):
    db_path = tmp_path / "inv.db"
    with connect(db_path) as conn:
        init_schema(conn)
        init_schema(conn)
        seed_defaults(conn)
        assert get_stock(conn, "GadgetX") == 5


def test_init_db_fresh_resets_existing_file(tmp_path):
    db_path = tmp_path / "inv.db"
    init_db(db_path)
    with connect(db_path) as conn:
        conn.execute("UPDATE inventory SET stock = 0 WHERE item = 'WidgetA'")
        conn.commit()
    init_db(db_path, fresh=True)
    with connect(db_path) as conn:
        assert get_stock(conn, "WidgetA") == 15


def test_lookup_policy_bands(tmp_path):
    from galatiq.db import lookup_policy

    db_path = tmp_path / "inv.db"
    init_db(db_path)
    with connect(db_path) as conn:
        assert lookup_policy(conn, Decimal("0")).policy_id == "TIER-AUTO"
        assert lookup_policy(conn, Decimal("999.99")).policy_id == "TIER-AUTO"
        assert lookup_policy(conn, Decimal("1000")).policy_id == "TIER-MGR"
        assert lookup_policy(conn, Decimal("9999")).policy_id == "TIER-MGR"
        assert lookup_policy(conn, Decimal("10000")).policy_id == "TIER-DIR"
        assert lookup_policy(conn, Decimal("49999")).policy_id == "TIER-DIR"
        assert lookup_policy(conn, Decimal("50000")).policy_id == "TIER-CFO"
        assert lookup_policy(conn, Decimal("9999999")).policy_id == "TIER-CFO"


def test_record_approval_round_trips(tmp_path):
    from galatiq.db import list_approvals, record_approval

    db_path = tmp_path / "inv.db"
    init_db(db_path)
    with connect(db_path) as conn:
        record_approval(
            conn,
            invoice_number="INV-A",
            vendor="Acme Corp",
            status="auto_approved",
            approver_role="system",
            policy_id="TIER-AUTO",
            total_usd=Decimal("42.00"),
        )
        rows = list_approvals(conn)
    assert len(rows) == 1
    assert rows[0]["status"] == "auto_approved"
    assert rows[0]["policy_id"] == "TIER-AUTO"


def test_record_payment_is_idempotent_on_reference(tmp_path):
    from galatiq.db import list_payments, record_payment

    db_path = tmp_path / "inv.db"
    init_db(db_path)
    with connect(db_path) as conn:
        for _ in range(3):
            record_payment(
                conn,
                reference="PAY-VEND-001-INV-X-ACH",
                invoice_number="INV-X",
                vendor="Acme Corp",
                rail="ach",
                status="scheduled",
                amount_usd=Decimal("100"),
                currency_paid="USD",
                amount_paid=Decimal("100"),
                scheduled_for="2026-02-01",
                receipt_path="data/receipts/PAY-VEND-001-INV-X-ACH.txt",
            )
        rows = list_payments(conn)
    assert len(rows) == 1
    assert rows[0]["rail"] == "ach"
