from __future__ import annotations

from galatiq.db import (
    SEED_INVENTORY,
    connect,
    get_stock,
    init_db,
    init_schema,
    list_items,
    seed_defaults,
)


def test_init_db_creates_file_and_seeds(tmp_path):
    db_path = tmp_path / "inv.db"
    returned = init_db(db_path)
    assert returned == db_path
    assert db_path.exists()
    with connect(db_path) as conn:
        items = dict(list_items(conn))
    assert items == SEED_INVENTORY


def test_seed_is_idempotent(tmp_path):
    db_path = tmp_path / "inv.db"
    init_db(db_path)
    with connect(db_path) as conn:
        # Mutate one row, then re-seed: existing row must not be overwritten.
        conn.execute("UPDATE inventory SET stock = ? WHERE item = ?", (99, "WidgetA"))
        conn.commit()
        inserted = seed_defaults(conn)
        assert inserted == 0
        assert get_stock(conn, "WidgetA") == 99


def test_get_stock_handles_missing_item(tmp_path):
    db_path = tmp_path / "inv.db"
    init_db(db_path)
    with connect(db_path) as conn:
        assert get_stock(conn, "GadgetX") == 5
        assert get_stock(conn, "FakeItem") == 0
        assert get_stock(conn, "DoesNotExist") is None


def test_init_schema_is_safe_to_call_twice(tmp_path):
    db_path = tmp_path / "inv.db"
    with connect(db_path) as conn:
        init_schema(conn)
        init_schema(conn)
        seed_defaults(conn)
        assert dict(list_items(conn)) == SEED_INVENTORY
