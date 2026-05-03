"""LangChain tools that LLM specialist agents can call to query the reference DB.

These wrap the existing pure-function helpers in ``galatiq.db`` so an agent can
selectively look up vendors, items, and ledger history during reasoning instead
of being handed the entire context up front.

Tools are produced via ``build_tools(db_path)`` which closes over the SQLite
file path, opening short-lived connections per call. This keeps the tools
stateless from the LLM's perspective and reusable across agents.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from galatiq.db import (
    connect,
    list_inventory,
    list_vendors,
    lookup_item,
    lookup_vendor,
)


# --- argument schemas (Pydantic so the LLM gets validated arg shapes) --------


class _NameArgs(BaseModel):
    name: str = Field(description="Exact item or vendor name to look up.")


class _LedgerArgs(BaseModel):
    vendor: str = Field(description="Vendor name to fetch recent ledger entries for.")
    limit: int = Field(default=5, description="Max rows to return (1-20).")


class _NoArgs(BaseModel):
    """No arguments — for tools that fetch the whole table."""


# --- tool implementations ----------------------------------------------------


def _lookup_item_tool(db_path: Path):
    def _impl(name: str) -> str:
        with connect(db_path) as conn:
            item = lookup_item(conn, name)
        if item is None:
            return json.dumps({"found": False, "name": name})
        return json.dumps(
            {
                "found": True,
                "item": item.item,
                "stock": item.stock,
                "unit_price": str(item.unit_price) if item.unit_price else None,
                "category": item.category,
                "status": item.status,
            }
        )

    return StructuredTool.from_function(
        func=_impl,
        name="lookup_catalog_item",
        description=(
            "Look up an item in the inventory catalog by exact name. Returns "
            "stock, unit_price, category, and status. Use this to verify a "
            "line-item SKU exists or check its status."
        ),
        args_schema=_NameArgs,
    )


def _lookup_vendor_tool(db_path: Path):
    def _impl(name: str) -> str:
        with connect(db_path) as conn:
            vendor = lookup_vendor(conn, name)
        if vendor is None:
            return json.dumps({"found": False, "name": name})
        return json.dumps(
            {
                "found": True,
                "vendor_id": vendor.vendor_id,
                "name": vendor.name,
                "aliases": vendor.aliases,
                "address": vendor.address,
                "status": vendor.status,
                "default_currency": vendor.default_currency,
            }
        )

    return StructuredTool.from_function(
        func=_impl,
        name="lookup_vendor",
        description=(
            "Look up a vendor by canonical name or any registered alias "
            "(case-insensitive). Returns vendor_id, status (active/blocked/new), "
            "aliases, and default currency. Use this to confirm a vendor is known "
            "or to check for typosquats against canonical names."
        ),
        args_schema=_NameArgs,
    )


def _ledger_history_tool(db_path: Path):
    def _impl(vendor: str, limit: int = 5) -> str:
        limit = max(1, min(20, int(limit)))
        with connect(db_path) as conn:
            rows = conn.execute(
                "SELECT invoice_number, total, source_path, ingested_at "
                "FROM invoice_ledger WHERE vendor = ? "
                "ORDER BY ingested_at DESC LIMIT ?",
                (vendor, limit),
            ).fetchall()
        return json.dumps(
            [
                {
                    "invoice_number": r["invoice_number"],
                    "total": r["total"],
                    "source_path": r["source_path"],
                    "ingested_at": r["ingested_at"],
                }
                for r in rows
            ]
        )

    return StructuredTool.from_function(
        func=_impl,
        name="recent_invoices_for_vendor",
        description=(
            "Return up to ``limit`` most-recent ledger entries for a vendor, "
            "sorted newest-first. Useful for spotting unusual invoice frequency "
            "or repeated invoice-number patterns."
        ),
        args_schema=_LedgerArgs,
    )


def _list_catalog_tool(db_path: Path):
    def _impl() -> str:
        with connect(db_path) as conn:
            items = list_inventory(conn)
        return json.dumps(
            [
                {
                    "item": i.item,
                    "stock": i.stock,
                    "unit_price": str(i.unit_price) if i.unit_price else None,
                    "category": i.category,
                    "status": i.status,
                }
                for i in items
            ]
        )

    return StructuredTool.from_function(
        func=_impl,
        name="list_catalog",
        description=(
            "Return the full inventory catalog. Useful for cross-checking "
            "category mismatches against the universe of known items."
        ),
        args_schema=_NoArgs,
    )


def _list_known_vendors_tool(db_path: Path):
    def _impl() -> str:
        with connect(db_path) as conn:
            vendors = list_vendors(conn)
        return json.dumps(
            [{"name": v.name, "aliases": v.aliases, "status": v.status} for v in vendors]
        )

    return StructuredTool.from_function(
        func=_impl,
        name="list_known_vendors",
        description=(
            "Return the list of all known vendors with canonical names, "
            "aliases, and status. Use this to detect typosquats."
        ),
        args_schema=_NoArgs,
    )


# --- public bundles ----------------------------------------------------------


def build_screener_tools(db_path: str | Path) -> list[Any]:
    """Tools for the fraud-screener agent (vendor & catalog lookups)."""
    p = Path(db_path)
    return [
        _lookup_vendor_tool(p),
        _lookup_item_tool(p),
        _list_known_vendors_tool(p),
        _list_catalog_tool(p),
    ]


def build_investigator_tools(db_path: str | Path) -> list[Any]:
    """Tools for the investigator agent (catalog/vendor + ledger history)."""
    p = Path(db_path)
    return [
        _lookup_vendor_tool(p),
        _lookup_item_tool(p),
        _ledger_history_tool(p),
    ]
