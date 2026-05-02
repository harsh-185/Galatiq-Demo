"""SQLite-backed reference data for validation/approval phases.

Schema starts from the README's minimum (``inventory(item, stock)``) and extends
it with columns and tables needed for richer validation scenarios:

- inventory: + unit_price, category, status (active|discontinued|fraud_flag)
- vendors: known counterparties, alias matching, blocked/new status
- invoice_ledger: per-invoice dedup record written after validation passes

``init_db`` recreates the file from scratch — keeps demo state predictable.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

DEFAULT_DB_PATH = Path("inventory.db")

# Status enums — kept as plain strings to stay sqlite-friendly.
STATUS_ACTIVE = "active"
STATUS_DISCONTINUED = "discontinued"
STATUS_FRAUD = "fraud_flag"
STATUS_BLOCKED = "blocked"
STATUS_NEW = "new"

_DDL = [
    """
    CREATE TABLE IF NOT EXISTS inventory (
        item       TEXT PRIMARY KEY,
        stock      INTEGER NOT NULL,
        unit_price NUMERIC,
        category   TEXT,
        status     TEXT NOT NULL DEFAULT 'active'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS vendors (
        vendor_id        TEXT PRIMARY KEY,
        name             TEXT NOT NULL,
        aliases          TEXT,
        address          TEXT,
        status           TEXT NOT NULL DEFAULT 'active',
        default_currency TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS invoice_ledger (
        invoice_number TEXT NOT NULL,
        vendor         TEXT NOT NULL,
        total          NUMERIC,
        source_path    TEXT,
        ingested_at    TEXT NOT NULL,
        PRIMARY KEY (invoice_number, vendor)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS approval_policies (
        policy_id     TEXT PRIMARY KEY,
        min_usd       NUMERIC NOT NULL,
        max_usd       NUMERIC,
        approver_role TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS approval_log (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        invoice_number TEXT NOT NULL,
        vendor         TEXT NOT NULL,
        status         TEXT NOT NULL,
        approver_role  TEXT NOT NULL,
        policy_id      TEXT,
        total_usd      NUMERIC NOT NULL,
        decided_at     TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS payment_log (
        reference      TEXT PRIMARY KEY,
        invoice_number TEXT NOT NULL,
        vendor         TEXT NOT NULL,
        rail           TEXT NOT NULL,
        status         TEXT NOT NULL,
        amount_usd     NUMERIC NOT NULL,
        currency_paid  TEXT NOT NULL,
        amount_paid    NUMERIC NOT NULL,
        scheduled_for  TEXT,
        receipt_path   TEXT,
        recorded_at    TEXT NOT NULL
    )
    """,
]

SEED_INVENTORY: list[dict[str, Any]] = [
    {"item": "WidgetA", "stock": 15, "unit_price": "10.00", "category": "hardware", "status": STATUS_ACTIVE},
    {"item": "WidgetB", "stock": 10, "unit_price": "25.00", "category": "hardware", "status": STATUS_ACTIVE},
    {"item": "GadgetX", "stock": 5,  "unit_price": "50.00", "category": "electronics", "status": STATUS_ACTIVE},
    {"item": "FakeItem", "stock": 0, "unit_price": None,    "category": None,         "status": STATUS_FRAUD},
    {"item": "GizmoPro", "stock": 100, "unit_price": "200.00", "category": "electronics", "status": STATUS_DISCONTINUED},
    {"item": "BoltPack", "stock": 500, "unit_price": "5.00",   "category": "hardware",    "status": STATUS_ACTIVE},
    {"item": "LaserCutterPro", "stock": 3, "unit_price": "25000.00", "category": "equipment", "status": STATUS_ACTIVE},
]

SEED_APPROVAL_POLICIES: list[dict[str, Any]] = [
    {"policy_id": "TIER-AUTO", "min_usd": "0",      "max_usd": "1000",   "approver_role": "system"},
    {"policy_id": "TIER-MGR",  "min_usd": "1000",   "max_usd": "10000",  "approver_role": "manager"},
    {"policy_id": "TIER-DIR",  "min_usd": "10000",  "max_usd": "50000",  "approver_role": "director"},
    {"policy_id": "TIER-CFO",  "min_usd": "50000",  "max_usd": None,     "approver_role": "cfo"},
]


SEED_VENDORS: list[dict[str, Any]] = [
    {
        "vendor_id": "VEND-001",
        "name": "Acme Corp",
        "aliases": ["Acme", "Acme Co.", "ACME", "ACME Corp", "Acme Corporation"],
        "address": "123 Acme St, Springfield",
        "status": STATUS_ACTIVE,
        "default_currency": "USD",
    },
    {
        "vendor_id": "VEND-002",
        "name": "Beta Industries",
        "aliases": ["Beta Ind", "Beta", "Beta Industries Inc."],
        "address": "456 Beta Ave",
        "status": STATUS_ACTIVE,
        "default_currency": "USD",
    },
    {
        "vendor_id": "VEND-003",
        "name": "ShadyVendor LLC",
        "aliases": [],
        "address": None,
        "status": STATUS_BLOCKED,
        "default_currency": "USD",
    },
    {
        "vendor_id": "VEND-004",
        "name": "NewCo",
        "aliases": [],
        "address": "789 Newcomer Rd",
        "status": STATUS_NEW,
        "default_currency": "USD",
    },
]


@dataclass(frozen=True)
class InventoryItem:
    item: str
    stock: int
    unit_price: Decimal | None
    category: str | None
    status: str


@dataclass(frozen=True)
class Vendor:
    vendor_id: str
    name: str
    aliases: list[str]
    address: str | None
    status: str
    default_currency: str | None


@dataclass(frozen=True)
class ApprovalPolicy:
    policy_id: str
    min_usd: Decimal
    max_usd: Decimal | None  # None = no upper bound
    approver_role: str


def connect(path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    for stmt in _DDL:
        conn.execute(stmt)
    conn.commit()


def seed_inventory(
    conn: sqlite3.Connection,
    rows: Iterable[dict[str, Any]] | None = None,
) -> int:
    payload = [
        (
            r["item"],
            r["stock"],
            r.get("unit_price"),
            r.get("category"),
            r.get("status", STATUS_ACTIVE),
        )
        for r in (rows if rows is not None else SEED_INVENTORY)
    ]
    cur = conn.executemany(
        "INSERT OR IGNORE INTO inventory(item, stock, unit_price, category, status) VALUES (?,?,?,?,?)",
        payload,
    )
    conn.commit()
    return cur.rowcount


def seed_vendors(
    conn: sqlite3.Connection,
    rows: Iterable[dict[str, Any]] | None = None,
) -> int:
    payload = [
        (
            r["vendor_id"],
            r["name"],
            json.dumps(r.get("aliases") or []),
            r.get("address"),
            r.get("status", STATUS_ACTIVE),
            r.get("default_currency"),
        )
        for r in (rows if rows is not None else SEED_VENDORS)
    ]
    cur = conn.executemany(
        "INSERT OR IGNORE INTO vendors(vendor_id, name, aliases, address, status, default_currency) VALUES (?,?,?,?,?,?)",
        payload,
    )
    conn.commit()
    return cur.rowcount


def seed_approval_policies(
    conn: sqlite3.Connection,
    rows: Iterable[dict[str, Any]] | None = None,
) -> int:
    payload = [
        (
            r["policy_id"],
            r["min_usd"],
            r.get("max_usd"),
            r["approver_role"],
        )
        for r in (rows if rows is not None else SEED_APPROVAL_POLICIES)
    ]
    cur = conn.executemany(
        "INSERT OR IGNORE INTO approval_policies(policy_id, min_usd, max_usd, approver_role) "
        "VALUES (?,?,?,?)",
        payload,
    )
    conn.commit()
    return cur.rowcount


def seed_defaults(conn: sqlite3.Connection) -> int:
    """Seed inventory, vendor, and approval-policy reference data. Returns total rows inserted."""
    return seed_inventory(conn) + seed_vendors(conn) + seed_approval_policies(conn)


def lookup_item(conn: sqlite3.Connection, name: str) -> InventoryItem | None:
    row = conn.execute(
        "SELECT item, stock, unit_price, category, status FROM inventory WHERE item = ?",
        (name,),
    ).fetchone()
    if row is None:
        return None
    return InventoryItem(
        item=row["item"],
        stock=int(row["stock"]),
        unit_price=Decimal(str(row["unit_price"])) if row["unit_price"] is not None else None,
        category=row["category"],
        status=row["status"],
    )


def lookup_vendor(conn: sqlite3.Connection, name: str) -> Vendor | None:
    """Match by canonical name first, then by any alias (case-insensitive)."""
    target = name.strip().lower()
    if not target:
        return None
    rows = conn.execute(
        "SELECT vendor_id, name, aliases, address, status, default_currency FROM vendors"
    ).fetchall()
    for r in rows:
        if r["name"].strip().lower() == target:
            return _vendor_from_row(r)
    for r in rows:
        try:
            aliases = json.loads(r["aliases"] or "[]")
        except json.JSONDecodeError:
            aliases = []
        if any(a.strip().lower() == target for a in aliases):
            return _vendor_from_row(r)
    return None


def _vendor_from_row(row: sqlite3.Row) -> Vendor:
    try:
        aliases = json.loads(row["aliases"] or "[]")
    except json.JSONDecodeError:
        aliases = []
    return Vendor(
        vendor_id=row["vendor_id"],
        name=row["name"],
        aliases=list(aliases),
        address=row["address"],
        status=row["status"],
        default_currency=row["default_currency"],
    )


def has_invoice(conn: sqlite3.Connection, invoice_number: str, vendor: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM invoice_ledger WHERE invoice_number = ? AND vendor = ?",
        (invoice_number, vendor),
    ).fetchone()
    return row is not None


def record_invoice(
    conn: sqlite3.Connection,
    *,
    invoice_number: str,
    vendor: str,
    total: Decimal | float | str | None,
    source_path: str | None = None,
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO invoice_ledger(invoice_number, vendor, total, source_path, ingested_at) "
        "VALUES (?,?,?,?,?)",
        (
            invoice_number,
            vendor,
            str(total) if total is not None else None,
            source_path,
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ),
    )
    conn.commit()


def list_items(conn: sqlite3.Connection) -> list[tuple[str, int]]:
    rows = conn.execute("SELECT item, stock FROM inventory ORDER BY item").fetchall()
    return [(r["item"], int(r["stock"])) for r in rows]


def list_inventory(conn: sqlite3.Connection) -> list[InventoryItem]:
    rows = conn.execute(
        "SELECT item, stock, unit_price, category, status FROM inventory ORDER BY item"
    ).fetchall()
    return [
        InventoryItem(
            item=r["item"],
            stock=int(r["stock"]),
            unit_price=Decimal(str(r["unit_price"])) if r["unit_price"] is not None else None,
            category=r["category"],
            status=r["status"],
        )
        for r in rows
    ]


def list_vendors(conn: sqlite3.Connection) -> list[Vendor]:
    rows = conn.execute(
        "SELECT vendor_id, name, aliases, address, status, default_currency FROM vendors ORDER BY name"
    ).fetchall()
    return [_vendor_from_row(r) for r in rows]


def _policy_from_row(row: sqlite3.Row) -> ApprovalPolicy:
    return ApprovalPolicy(
        policy_id=row["policy_id"],
        min_usd=Decimal(str(row["min_usd"])),
        max_usd=Decimal(str(row["max_usd"])) if row["max_usd"] is not None else None,
        approver_role=row["approver_role"],
    )


def lookup_policy(conn: sqlite3.Connection, total_usd: Decimal) -> ApprovalPolicy | None:
    """Return the policy whose [min_usd, max_usd) band contains ``total_usd``."""
    amount = Decimal(total_usd)
    rows = conn.execute(
        "SELECT policy_id, min_usd, max_usd, approver_role FROM approval_policies "
        "ORDER BY min_usd"
    ).fetchall()
    for r in rows:
        policy = _policy_from_row(r)
        if amount < policy.min_usd:
            continue
        if policy.max_usd is None or amount < policy.max_usd:
            return policy
    return None


def list_policies(conn: sqlite3.Connection) -> list[ApprovalPolicy]:
    rows = conn.execute(
        "SELECT policy_id, min_usd, max_usd, approver_role FROM approval_policies ORDER BY min_usd"
    ).fetchall()
    return [_policy_from_row(r) for r in rows]


def record_approval(
    conn: sqlite3.Connection,
    *,
    invoice_number: str,
    vendor: str,
    status: str,
    approver_role: str,
    policy_id: str | None,
    total_usd: Decimal,
) -> None:
    conn.execute(
        "INSERT INTO approval_log(invoice_number, vendor, status, approver_role, policy_id, total_usd, decided_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (
            invoice_number,
            vendor,
            status,
            approver_role,
            policy_id,
            str(total_usd),
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ),
    )
    conn.commit()


def record_payment(
    conn: sqlite3.Connection,
    *,
    reference: str,
    invoice_number: str,
    vendor: str,
    rail: str,
    status: str,
    amount_usd: Decimal,
    currency_paid: str,
    amount_paid: Decimal,
    scheduled_for: str | None,
    receipt_path: str | None,
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO payment_log("
        "reference, invoice_number, vendor, rail, status, amount_usd, currency_paid, "
        "amount_paid, scheduled_for, receipt_path, recorded_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            reference,
            invoice_number,
            vendor,
            rail,
            status,
            str(amount_usd),
            currency_paid,
            str(amount_paid),
            scheduled_for,
            receipt_path,
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ),
    )
    conn.commit()


def list_approvals(conn: sqlite3.Connection, *, limit: int = 50) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT id, invoice_number, vendor, status, approver_role, policy_id, total_usd, decided_at "
        "FROM approval_log ORDER BY id DESC LIMIT ?",
        (int(limit),),
    ).fetchall()


def list_payments(conn: sqlite3.Connection, *, limit: int = 50) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT reference, invoice_number, vendor, rail, status, amount_usd, currency_paid, "
        "amount_paid, scheduled_for, receipt_path, recorded_at "
        "FROM payment_log ORDER BY recorded_at DESC LIMIT ?",
        (int(limit),),
    ).fetchall()


def init_db(path: str | Path = DEFAULT_DB_PATH, *, fresh: bool = True) -> Path:
    """Create the file (overwriting if ``fresh``), apply schema, and seed defaults."""
    p = Path(path)
    if fresh and p.exists():
        p.unlink()
    with connect(p) as conn:
        init_schema(conn)
        seed_defaults(conn)
    return p


# Backwards-compat alias for the v1 helper (was: get_stock).
def get_stock(conn: sqlite3.Connection, item: str) -> int | None:
    row = conn.execute("SELECT stock FROM inventory WHERE item = ?", (item,)).fetchone()
    return int(row["stock"]) if row is not None else None
