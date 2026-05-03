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
    """
    CREATE TABLE IF NOT EXISTS vendor_payment_methods (
        vendor_id      TEXT NOT NULL,
        rail           TEXT NOT NULL,
        account_ref    TEXT,
        status         TEXT NOT NULL DEFAULT 'active',
        PRIMARY KEY (vendor_id, rail)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS human_review_queue (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        invoice_number  TEXT NOT NULL,
        vendor          TEXT NOT NULL,
        decision_status TEXT NOT NULL,
        approver_role   TEXT NOT NULL,
        policy_id       TEXT,
        total_usd       NUMERIC NOT NULL,
        source_path     TEXT,
        narrative       TEXT,
        queued_at       TEXT NOT NULL,
        resolved_at     TEXT,
        resolved_by     TEXT,
        resolution      TEXT,
        resolution_note TEXT
    )
    """,
]

# Spec-mandated seed: matches the README exactly so the documented test
# scenarios (INV-1001/1002/1003/1008/1009/1016) reproduce as described.
# Other columns are NULL/default — the spec uses (item, stock) only.
SEED_INVENTORY: list[dict[str, Any]] = [
    {"item": "WidgetA",  "stock": 15},
    {"item": "WidgetB",  "stock": 10},
    {"item": "GadgetX",  "stock": 5},
    {"item": "FakeItem", "stock": 0},
]

# Optional extensions exercised by additional edge-case tests. The spec
# explicitly permits "additional items or columns to support richer validation".
# These items demonstrate price-drift, discontinued, and fraud-flag rules.
SEED_INVENTORY_EXTRA: list[dict[str, Any]] = [
    {"item": "GizmoPro",       "stock": 100, "unit_price": "200.00",  "category": "electronics", "status": STATUS_DISCONTINUED},
    {"item": "BoltPack",       "stock": 500, "unit_price": "5.00",    "category": "hardware",    "status": STATUS_ACTIVE},
    {"item": "LaserCutterPro", "stock": 3,   "unit_price": "25000.00","category": "equipment",   "status": STATUS_ACTIVE},
    {"item": "PhantomSKU",     "stock": 0,   "unit_price": None,      "category": None,          "status": STATUS_FRAUD},
]

SEED_VENDOR_PAYMENT_METHODS: list[dict[str, Any]] = [
    # Acme Corp — accepts ach (default) and wire for larger amounts.
    {"vendor_id": "VEND-001", "rail": "ach",   "account_ref": "ACME-ACH-001",  "status": "active"},
    {"vendor_id": "VEND-001", "rail": "wire",  "account_ref": "ACME-WIRE-001", "status": "active"},
    # Beta Industries — international wire only.
    {"vendor_id": "VEND-002", "rail": "wire",  "account_ref": "BETA-WIRE-001", "status": "active"},
    {"vendor_id": "VEND-002", "rail": "check", "account_ref": "BETA-CHK-001",  "status": "active"},
    # ShadyVendor — explicitly disabled across all rails (defense-in-depth).
    {"vendor_id": "VEND-003", "rail": "ach",   "account_ref": None,            "status": "disabled"},
    # NewCo — pending banking verification.
    {"vendor_id": "VEND-004", "rail": "ach",   "account_ref": None,            "status": "pending_verification"},
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


def seed_vendor_payment_methods(
    conn: sqlite3.Connection,
    rows: Iterable[dict[str, Any]] | None = None,
) -> int:
    payload = [
        (r["vendor_id"], r["rail"], r.get("account_ref"), r.get("status", "active"))
        for r in (rows if rows is not None else SEED_VENDOR_PAYMENT_METHODS)
    ]
    cur = conn.executemany(
        "INSERT OR IGNORE INTO vendor_payment_methods(vendor_id, rail, account_ref, status) "
        "VALUES (?,?,?,?)",
        payload,
    )
    conn.commit()
    return cur.rowcount


def seed_defaults(conn: sqlite3.Connection, *, include_extras: bool = True) -> int:
    """Seed reference data. Returns total rows inserted.

    The spec-mandated minimum is ``SEED_INVENTORY`` (4 items) and the vendors
    table. Extras (``SEED_INVENTORY_EXTRA``) cover discontinued, fraud-flag,
    and high-value items used by extended validation tests.
    """
    inserted = (
        seed_inventory(conn)
        + seed_vendors(conn)
        + seed_approval_policies(conn)
        + seed_vendor_payment_methods(conn)
    )
    if include_extras:
        inserted += seed_inventory(conn, SEED_INVENTORY_EXTRA)
    return inserted


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


@dataclass(frozen=True)
class VendorPaymentMethod:
    vendor_id: str
    rail: str
    account_ref: str | None
    status: str


def list_vendor_payment_methods(
    conn: sqlite3.Connection, vendor_id: str
) -> list[VendorPaymentMethod]:
    rows = conn.execute(
        "SELECT vendor_id, rail, account_ref, status FROM vendor_payment_methods "
        "WHERE vendor_id = ?",
        (vendor_id,),
    ).fetchall()
    return [
        VendorPaymentMethod(
            vendor_id=r["vendor_id"],
            rail=r["rail"],
            account_ref=r["account_ref"],
            status=r["status"],
        )
        for r in rows
    ]


def lookup_payment_method(
    conn: sqlite3.Connection, vendor_id: str, rail: str
) -> VendorPaymentMethod | None:
    row = conn.execute(
        "SELECT vendor_id, rail, account_ref, status FROM vendor_payment_methods "
        "WHERE vendor_id = ? AND rail = ?",
        (vendor_id, rail),
    ).fetchone()
    if row is None:
        return None
    return VendorPaymentMethod(
        vendor_id=row["vendor_id"],
        rail=row["rail"],
        account_ref=row["account_ref"],
        status=row["status"],
    )


# --- human review queue (HITL) -----------------------------------------------


def queue_for_human_review(
    conn: sqlite3.Connection,
    *,
    invoice_number: str,
    vendor: str,
    decision_status: str,
    approver_role: str,
    policy_id: str | None,
    total_usd: Decimal,
    source_path: str | None = None,
    narrative: str | None = None,
) -> int:
    cur = conn.execute(
        "INSERT INTO human_review_queue("
        "invoice_number, vendor, decision_status, approver_role, policy_id, "
        "total_usd, source_path, narrative, queued_at) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (
            invoice_number,
            vendor,
            decision_status,
            approver_role,
            policy_id,
            str(total_usd),
            source_path,
            narrative,
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def list_pending_reviews(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT id, invoice_number, vendor, decision_status, approver_role, "
        "policy_id, total_usd, source_path, narrative, queued_at "
        "FROM human_review_queue WHERE resolved_at IS NULL ORDER BY queued_at"
    ).fetchall()


def get_review_entry(conn: sqlite3.Connection, review_id: int) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT id, invoice_number, vendor, decision_status, approver_role, "
        "policy_id, total_usd, source_path, narrative, queued_at, resolved_at, "
        "resolved_by, resolution, resolution_note "
        "FROM human_review_queue WHERE id = ?",
        (int(review_id),),
    ).fetchone()


def resolve_review(
    conn: sqlite3.Connection,
    *,
    review_id: int,
    resolution: str,
    resolved_by: str = "cli",
    note: str | None = None,
) -> None:
    conn.execute(
        "UPDATE human_review_queue SET "
        "resolved_at = ?, resolved_by = ?, resolution = ?, resolution_note = ? "
        "WHERE id = ? AND resolved_at IS NULL",
        (
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            resolved_by,
            resolution,
            note,
            int(review_id),
        ),
    )
    conn.commit()


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
