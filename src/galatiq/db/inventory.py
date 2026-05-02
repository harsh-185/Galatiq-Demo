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
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        threshold_min NUMERIC NOT NULL,
        threshold_max NUMERIC,
        approver_role TEXT,
        description   TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS approval_log (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        invoice_number TEXT NOT NULL,
        vendor         TEXT NOT NULL,
        total          NUMERIC,
        currency       TEXT,
        total_usd      NUMERIC,
        verdict        TEXT,
        status         TEXT NOT NULL,
        approver_role  TEXT,
        policy_id      INTEGER,
        justification  TEXT,
        decided_at     TEXT NOT NULL
    )
    """,
]

SEED_INVENTORY: list[dict[str, Any]] = [
    # Catalog prices match the modal price observed in the case fixtures so the
    # documented "clean" invoices (INV-1001/1004/1006/...) pass without spurious
    # price-drift warnings. The fixtures occasionally use volume-discount prices
    # (e.g. WidgetA at $240) which sit comfortably under the 25% drift threshold.
    {"item": "WidgetA",        "stock": 15,  "unit_price": "250.00",   "category": "hardware",    "status": STATUS_ACTIVE},
    {"item": "WidgetB",        "stock": 10,  "unit_price": "500.00",   "category": "hardware",    "status": STATUS_ACTIVE},
    {"item": "GadgetX",        "stock": 5,   "unit_price": "750.00",   "category": "electronics", "status": STATUS_ACTIVE},
    {"item": "FakeItem",       "stock": 0,   "unit_price": None,        "category": None,          "status": STATUS_FRAUD},
    # Stress-test SKUs not referenced by any fixture; useful for manual probing.
    {"item": "GizmoPro",       "stock": 100, "unit_price": "200.00",   "category": "electronics", "status": STATUS_DISCONTINUED},
    {"item": "BoltPack",       "stock": 500, "unit_price": "5.00",     "category": "hardware",    "status": STATUS_ACTIVE},
    {"item": "LaserCutterPro", "stock": 3,   "unit_price": "25000.00", "category": "equipment",   "status": STATUS_ACTIVE},
]

SEED_VENDORS: list[dict[str, Any]] = [
    # Vendors that appear in the case fixtures — seeded as active so legitimate
    # invoices match by canonical name (or alias) and clear vendor_unknown.
    {"vendor_id": "VEND-100", "name": "Widgets Inc.",                "aliases": ["Widgets", "Widgets Incorporated"], "address": None, "status": STATUS_ACTIVE, "default_currency": "USD"},
    {"vendor_id": "VEND-101", "name": "Precision Parts Ltd.",        "aliases": ["Precision Parts"],                  "address": "742 Evergreen Terrace, Springfield, IL", "status": STATUS_ACTIVE, "default_currency": "USD"},
    {"vendor_id": "VEND-102", "name": "Global Supply Chain Partners","aliases": ["Global Supply Chain"],              "address": None, "status": STATUS_ACTIVE, "default_currency": "USD"},
    {"vendor_id": "VEND-103", "name": "Acme Industrial Supplies",    "aliases": ["Acme Industrial", "Acme Supplies"], "address": None, "status": STATUS_ACTIVE, "default_currency": "USD"},
    {"vendor_id": "VEND-104", "name": "MegaWidgets Corp",            "aliases": ["MegaWidgets"],                      "address": None, "status": STATUS_ACTIVE, "default_currency": "USD"},
    {"vendor_id": "VEND-105", "name": "Atlas Industrial Supply",     "aliases": ["Atlas Industrial"],                 "address": "500 Commerce Blvd, Detroit, MI", "status": STATUS_ACTIVE, "default_currency": "USD"},
    {"vendor_id": "VEND-106", "name": "TechParts International",     "aliases": ["TechParts"],                        "address": None, "status": STATUS_ACTIVE, "default_currency": "EUR"},
    {"vendor_id": "VEND-107", "name": "Reliable Components Inc.",    "aliases": ["Reliable Components"],              "address": None, "status": STATUS_ACTIVE, "default_currency": "USD"},
    {"vendor_id": "VEND-108", "name": "Consolidated Materials Group","aliases": ["Consolidated Materials"],           "address": None, "status": STATUS_ACTIVE, "default_currency": "USD"},
    {"vendor_id": "VEND-109", "name": "Summit Manufacturing Co.",    "aliases": ["Summit Manufacturing"],             "address": None, "status": STATUS_ACTIVE, "default_currency": "USD"},
    # Known-bad vendor that pairs with the FakeItem fraud SKU in INV-1003.
    {"vendor_id": "VEND-900", "name": "Fraudster LLC",               "aliases": [],                                   "address": None, "status": STATUS_BLOCKED, "default_currency": "USD"},
    # Stress-test vendors not present in any fixture.
    {"vendor_id": "VEND-901", "name": "ShadyVendor LLC",             "aliases": [],                                   "address": None, "status": STATUS_BLOCKED, "default_currency": "USD"},
    {"vendor_id": "VEND-902", "name": "NewCo",                       "aliases": [],                                   "address": "789 Newcomer Rd", "status": STATUS_NEW, "default_currency": "USD"},
]


# Approval policy tiers. All values are USD-equivalent (see galatiq.fx for FX).
# threshold_max == None means "open-ended on the upper side".
SEED_APPROVAL_POLICIES: list[dict[str, Any]] = [
    {"threshold_min": "0",     "threshold_max": "1000",  "approver_role": None,       "description": "Auto-approve small invoices"},
    {"threshold_min": "1000",  "threshold_max": "10000", "approver_role": "manager",  "description": "Manager approval"},
    {"threshold_min": "10000", "threshold_max": "50000", "approver_role": "director", "description": "Director approval"},
    {"threshold_min": "50000", "threshold_max": None,    "approver_role": "cfo",      "description": "CFO approval"},
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
    id: int
    threshold_min: Decimal
    threshold_max: Decimal | None
    approver_role: str | None
    description: str | None


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
    """Insert seed policy rows. Skipped entirely if the table already has any."""
    existing = conn.execute("SELECT COUNT(*) FROM approval_policies").fetchone()[0]
    if existing:
        return 0
    payload = [
        (
            r["threshold_min"],
            r.get("threshold_max"),
            r.get("approver_role"),
            r.get("description"),
        )
        for r in (rows if rows is not None else SEED_APPROVAL_POLICIES)
    ]
    cur = conn.executemany(
        "INSERT INTO approval_policies(threshold_min, threshold_max, approver_role, description) "
        "VALUES (?,?,?,?)",
        payload,
    )
    conn.commit()
    return cur.rowcount


def seed_defaults(conn: sqlite3.Connection) -> int:
    """Seed inventory + vendor + approval-policy reference data."""
    return seed_inventory(conn) + seed_vendors(conn) + seed_approval_policies(conn)


def find_policy_for_amount_usd(
    conn: sqlite3.Connection, amount_usd: Decimal | float | str
) -> ApprovalPolicy | None:
    """Return the policy whose [min, max) band contains amount_usd."""
    amt = Decimal(str(amount_usd))
    rows = conn.execute(
        "SELECT id, threshold_min, threshold_max, approver_role, description FROM approval_policies "
        "ORDER BY threshold_min"
    ).fetchall()
    for r in rows:
        lo = Decimal(str(r["threshold_min"]))
        hi = Decimal(str(r["threshold_max"])) if r["threshold_max"] is not None else None
        if amt >= lo and (hi is None or amt < hi):
            return ApprovalPolicy(
                id=int(r["id"]),
                threshold_min=lo,
                threshold_max=hi,
                approver_role=r["approver_role"],
                description=r["description"],
            )
    return None


def list_approval_policies(conn: sqlite3.Connection) -> list[ApprovalPolicy]:
    rows = conn.execute(
        "SELECT id, threshold_min, threshold_max, approver_role, description FROM approval_policies "
        "ORDER BY threshold_min"
    ).fetchall()
    return [
        ApprovalPolicy(
            id=int(r["id"]),
            threshold_min=Decimal(str(r["threshold_min"])),
            threshold_max=Decimal(str(r["threshold_max"])) if r["threshold_max"] is not None else None,
            approver_role=r["approver_role"],
            description=r["description"],
        )
        for r in rows
    ]


def record_approval_decision(
    conn: sqlite3.Connection,
    *,
    invoice_number: str,
    vendor: str,
    total: Decimal | float | str | None,
    currency: str | None,
    total_usd: Decimal | float | str | None,
    verdict: str | None,
    status: str,
    approver_role: str | None,
    policy_id: int | None,
    justification: str | None,
) -> int:
    """Append an audit-log row. Returns the new row id."""
    cur = conn.execute(
        """
        INSERT INTO approval_log
            (invoice_number, vendor, total, currency, total_usd, verdict, status,
             approver_role, policy_id, justification, decided_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            invoice_number,
            vendor,
            str(total) if total is not None else None,
            currency,
            str(total_usd) if total_usd is not None else None,
            verdict,
            status,
            approver_role,
            policy_id,
            justification,
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def list_recent_decisions(conn: sqlite3.Connection, limit: int = 50) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT invoice_number, vendor, total, currency, total_usd, verdict, status, "
        "approver_role, policy_id, justification, decided_at "
        "FROM approval_log ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


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
