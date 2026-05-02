"""Inventory store backing the validation phase.

Schema is the minimum specified in the case README:

    CREATE TABLE IF NOT EXISTS inventory (item TEXT PRIMARY KEY, stock INTEGER)

Seed data drives the documented validation scenarios (stock overflow on INV-1002,
zero-stock detection on INV-1003, unknown SKUs on INV-1008/1016, etc.).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

DEFAULT_DB_PATH = Path("inventory.db")

SEED_INVENTORY: dict[str, int] = {
    "WidgetA": 15,
    "WidgetB": 10,
    "GadgetX": 5,
    "FakeItem": 0,
}

_DDL = "CREATE TABLE IF NOT EXISTS inventory (item TEXT PRIMARY KEY, stock INTEGER)"


def connect(path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.execute(_DDL)
    conn.commit()


def seed_defaults(conn: sqlite3.Connection, items: dict[str, int] | None = None) -> int:
    """Insert seed inventory rows. Idempotent — existing rows are left untouched."""
    rows = (items or SEED_INVENTORY).items()
    cur = conn.executemany("INSERT OR IGNORE INTO inventory(item, stock) VALUES (?, ?)", rows)
    conn.commit()
    return cur.rowcount


def get_stock(conn: sqlite3.Connection, item: str) -> int | None:
    row = conn.execute("SELECT stock FROM inventory WHERE item = ?", (item,)).fetchone()
    return row["stock"] if row is not None else None


def list_items(conn: sqlite3.Connection) -> list[tuple[str, int]]:
    rows = conn.execute("SELECT item, stock FROM inventory ORDER BY item").fetchall()
    return [(r["item"], r["stock"]) for r in rows]


def init_db(path: str | Path = DEFAULT_DB_PATH) -> Path:
    """Create the file (if absent), apply schema, and seed defaults. Returns the path."""
    p = Path(path)
    with connect(p) as conn:
        init_schema(conn)
        seed_defaults(conn)
    return p
