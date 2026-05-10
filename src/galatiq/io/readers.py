"""Read raw invoice content and optionally produce a deterministic Invoice dict.

Each reader returns (raw_text, structured_hint).  `structured_hint` is a best-effort
mapping that already conforms to the Invoice schema; if non-None, the IngestionAgent
will try to validate it directly and skip the LLM call.
"""
from __future__ import annotations

import csv
import json
import re
import xml.etree.ElementTree as ET
from datetime import date as _date
from io import StringIO
from pathlib import Path
from typing import Any

from galatiq.io.pdf import read_pdf

ReadResult = tuple[str, dict[str, Any] | None]

# Year-first formats (ISO-ish): YYYY-MM-DD or YYYY/MM/DD.
_DATE_ISO_LIKE = re.compile(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})$")
# Month-first formats (US): M/D/YYYY or M-D-YYYY with 1-2 digit M/D.
_DATE_US = re.compile(r"^(\d{1,2})[-/](\d{1,2})[-/](\d{4})$")


def _normalize_date(value: str | None) -> str | None:
    """Coerce common date strings to ISO ``YYYY-MM-DD``. Returns the input
    unchanged if it doesn't match a recognised format — downstream Pydantic
    will then raise a clean ValidationError and the ingestion agent will fall
    back to the LLM path."""
    if value is None:
        return None
    v = value.strip()
    if not v:
        return None
    m = _DATE_ISO_LIKE.match(v)
    if m:
        y, mo, d = (int(x) for x in m.groups())
        try:
            return _date(y, mo, d).isoformat()
        except ValueError:
            return v
    m = _DATE_US.match(v)
    if m:
        mo, d, y = (int(x) for x in m.groups())
        try:
            return _date(y, mo, d).isoformat()
        except ValueError:
            return v
    return v


def read_invoice(path: str | Path) -> ReadResult:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        return read_pdf(p), None
    raw = p.read_text(encoding="utf-8", errors="replace")
    if suffix == ".json":
        return raw, _from_json(raw)
    if suffix == ".xml":
        return raw, _from_xml(raw)
    if suffix == ".csv":
        return raw, _from_csv(raw)
    return raw, None


def _from_json(raw: str) -> dict[str, Any] | None:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    vendor = data.get("vendor")
    vendor_name: str
    vendor_address: str | None = None
    if isinstance(vendor, dict):
        vendor_name = str(vendor.get("name") or "")
        addr = vendor.get("address")
        vendor_address = str(addr) if addr else None
    else:
        vendor_name = str(vendor or "")
    line_items = []
    for li in data.get("line_items") or []:
        if not isinstance(li, dict):
            continue
        name = li.get("item") or li.get("name") or li.get("description")
        qty = li.get("quantity") if li.get("quantity") is not None else li.get("qty")
        price = li.get("unit_price") if li.get("unit_price") is not None else li.get("price")
        if name is None or qty is None or price is None:
            continue
        line_items.append({"item": str(name), "quantity": int(qty), "unit_price": price})
    tax = data.get("tax_amount")
    if tax is None:
        tax = data.get("tax")
    return {
        "invoice_number": data.get("invoice_number"),
        "vendor": vendor_name,
        "vendor_address": vendor_address,
        "date": _normalize_date(data.get("date")),
        "due_date": _normalize_date(data.get("due_date")),
        "currency": data.get("currency", "USD"),
        "line_items": line_items,
        "subtotal": data.get("subtotal"),
        "tax": tax if tax is not None else 0,
        "total": data.get("total"),
        "payment_terms": data.get("payment_terms") or None,
    }


def _from_xml(raw: str) -> dict[str, Any] | None:
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return None

    def _t(elem: ET.Element | None) -> str | None:
        return elem.text.strip() if elem is not None and elem.text else None

    header = root.find("header") or root
    totals = root.find("totals") or root
    line_items_root = root.find("line_items")
    items: list[dict[str, Any]] = []
    if line_items_root is not None:
        for li in line_items_root.findall("item"):
            name = _t(li.find("name")) or _t(li.find("item"))
            qty = _t(li.find("quantity"))
            price = _t(li.find("unit_price"))
            if name and qty and price:
                items.append({"item": name, "quantity": int(qty), "unit_price": price})
    return {
        "invoice_number": _t(header.find("invoice_number")),
        "vendor": _t(header.find("vendor")) or "",
        "vendor_address": _t(header.find("vendor_address")),
        "date": _normalize_date(_t(header.find("date"))),
        "due_date": _normalize_date(_t(header.find("due_date"))),
        "currency": _t(header.find("currency")) or "USD",
        "line_items": items,
        "subtotal": _t(totals.find("subtotal")),
        "tax": _t(totals.find("tax_amount")) or _t(totals.find("tax")) or 0,
        "total": _t(totals.find("total")),
        "payment_terms": _t(root.find("payment_terms")),
    }


def _from_csv(raw: str) -> dict[str, Any] | None:
    rows = list(csv.reader(StringIO(raw)))
    if not rows:
        return None
    header = [c.strip().lower() for c in rows[0]]
    # Pivot format: header == ["field", "value"]
    if header[:2] == ["field", "value"]:
        return _csv_pivot(rows[1:])
    # Flat format: header includes invoice number / item / qty columns.
    if "invoice number" in header and ("item" in header or "description" in header):
        return _csv_flat(header, rows[1:])
    return None


def _csv_pivot(rows: list[list[str]]) -> dict[str, Any]:
    out: dict[str, Any] = {"line_items": [], "currency": "USD", "tax": 0}
    pending: dict[str, Any] = {}
    for row in rows:
        if len(row) < 2:
            continue
        key, value = row[0].strip().lower(), row[1].strip()
        if key == "item":
            if pending:
                out["line_items"].append(pending)
            pending = {"item": value}
        elif key == "quantity":
            pending["quantity"] = int(value) if value else 0
        elif key == "unit_price":
            pending["unit_price"] = value
        elif key in {"invoice_number", "vendor", "date", "due_date", "subtotal", "total", "tax", "payment_terms", "currency"}:
            out[key] = _normalize_date(value) if key in {"date", "due_date"} else value
    if pending:
        out["line_items"].append(pending)
    return out


def _csv_flat(header: list[str], rows: list[list[str]]) -> dict[str, Any]:
    idx = {name: i for i, name in enumerate(header)}

    def get(row: list[str], col: str) -> str | None:
        i = idx.get(col)
        if i is None or i >= len(row):
            return None
        return row[i].strip() or None

    out: dict[str, Any] = {"line_items": [], "currency": "USD", "tax": 0}
    for row in rows:
        if not any(c.strip() for c in row):
            continue
        inv_no = get(row, "invoice number")
        if inv_no:
            out.setdefault("invoice_number", inv_no)
            out.setdefault("vendor", get(row, "vendor") or "")
            out.setdefault("date", _normalize_date(get(row, "date")))
            out.setdefault("due_date", _normalize_date(get(row, "due date")))
            item = get(row, "item")
            qty = get(row, "qty") or get(row, "quantity")
            price = get(row, "unit price")
            if item and qty and price:
                out["line_items"].append({"item": item, "quantity": int(qty), "unit_price": price})
        else:
            # Totals row, e.g. Subtotal: / Tax: / Total: in trailing columns
            label = next((c.strip().rstrip(":").lower() for c in row if c.strip()), "")
            value = next((c for c in reversed(row) if c.strip()), "")
            if label.startswith("subtotal"):
                out["subtotal"] = value
            elif label.startswith("tax"):
                out["tax"] = value
            elif label.startswith("total"):
                out["total"] = value
    return out
