"""Local Streamlit dashboard to visualize each pipeline phase on a single invoice.

Run from the repo root:
    .venv/bin/python -m streamlit run dashboard/app.py

This file is gitignored — it's a developer tool, not part of the shipped package.
"""
from __future__ import annotations

import difflib
import json
import sys
import tempfile
from decimal import Decimal
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

from galatiq.agents.ingestion import ingest  # noqa: E402
from galatiq.agents.validation import validate  # noqa: E402
from galatiq.db import (  # noqa: E402
    DEFAULT_DB_PATH,
    STATUS_ACTIVE,
    STATUS_BLOCKED,
    STATUS_DISCONTINUED,
    STATUS_FRAUD,
    STATUS_NEW,
    connect,
    has_invoice,
    init_db,
    list_inventory,
    list_vendors,
    lookup_item,
    lookup_vendor,
    record_invoice,
)
from galatiq.io.readers import read_invoice  # noqa: E402
from galatiq.io.sanitize import strip_control_tags  # noqa: E402

INVOICE_DIR = REPO_ROOT / "data" / "invoices"
SUPPORTED_SUFFIXES = {".txt", ".json", ".csv", ".xml", ".pdf"}
DB_PATH = REPO_ROOT / DEFAULT_DB_PATH

st.set_page_config(page_title="Galatiq Pipeline Dashboard", layout="wide")
st.title("Galatiq Invoice Pipeline")
st.caption("Pick an invoice, see what each phase does to it.")


def _list_samples() -> list[Path]:
    if not INVOICE_DIR.exists():
        return []
    return sorted(p for p in INVOICE_DIR.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES)


def _resolve_selection() -> Path | None:
    samples = _list_samples()
    sample_labels = ["— none —"] + [p.name for p in samples]
    with st.sidebar:
        st.header("Invoice source")
        choice = st.selectbox("Sample invoice", sample_labels, index=1 if samples else 0)
        upload = st.file_uploader("…or upload one", type=[s.lstrip(".") for s in SUPPORTED_SUFFIXES])
        st.divider()
        st.markdown(
            "**Phases**\n\n"
            "1. Ingestion ✅\n"
            "2. Validation 🚧\n"
            "3. Approval 🚧\n"
            "4. Payment 🚧"
        )
        st.divider()
        st.markdown("**Reference DB**")
        if DB_PATH.exists():
            st.caption(f"`{DB_PATH.name}` present")
        else:
            st.caption(f"`{DB_PATH.name}` missing")
        if st.button("Initialize / refresh DB"):
            init_db(DB_PATH, fresh=True)
            st.success("Reference DB rebuilt from seed.")

    if upload is not None:
        suffix = Path(upload.name).suffix.lower()
        tmp = Path(tempfile.gettempdir()) / f"galatiq_upload_{upload.name}"
        tmp.write_bytes(upload.getvalue())
        return tmp
    if choice != "— none —":
        return INVOICE_DIR / choice
    return None


def _decimal_default(o):
    if isinstance(o, Decimal):
        return str(o)
    raise TypeError


def _render_raw(path: Path) -> str:
    raw, hint = read_invoice(path)
    cols = st.columns([2, 1])
    with cols[0]:
        st.subheader("Raw text (post-reader)")
        st.caption(
            f"PDFs are extracted to text by `read_pdf`. Other formats are read verbatim."
            if path.suffix.lower() == ".pdf"
            else "Read verbatim from disk."
        )
        st.code(raw or "<empty>", language="text")
    with cols[1]:
        st.subheader("Deterministic hint")
        if hint is None:
            st.info("No deterministic hint — the LLM path will be used.")
        else:
            st.success("Reader produced a structured hint (LLM may be skipped if it validates).")
            st.json(hint)
    return raw


def _render_sanitized(raw: str) -> None:
    sanitized = strip_control_tags(raw)
    st.subheader("After `strip_control_tags`")
    if sanitized == raw:
        st.success("No prompt-injection patterns detected — text is unchanged.")
        st.code(sanitized or "<empty>", language="text")
        return
    st.warning("Sanitizer removed suspicious patterns. Diff below (red = removed, green = kept).")
    diff = difflib.unified_diff(
        raw.splitlines(),
        sanitized.splitlines(),
        fromfile="raw",
        tofile="sanitized",
        lineterm="",
    )
    st.code("\n".join(diff), language="diff")
    with st.expander("Sanitized text"):
        st.code(sanitized or "<empty>", language="text")


def _render_ingestion(path: Path) -> None:
    st.subheader("Ingestion result")
    try:
        result = ingest(path, allow_llm=True)
    except Exception as e:  # noqa: BLE001
        st.error(f"Ingestion failed: {type(e).__name__}: {e}")
        return

    inv = result.invoice
    metric_cols = st.columns(4)
    metric_cols[0].metric("Path taken", result.path_taken)
    metric_cols[1].metric("LLM retries", result.llm_retries)
    metric_cols[2].metric("Line items", len(inv.line_items))
    metric_cols[3].metric("Warnings", len(inv.ingestion_warnings))

    st.markdown("**Header**")
    header_df = pd.DataFrame(
        {
            "field": [
                "invoice_number",
                "vendor",
                "vendor_address",
                "date",
                "due_date",
                "currency",
                "subtotal",
                "tax",
                "total",
                "payment_terms",
            ],
            "value": [
                inv.invoice_number,
                inv.vendor,
                inv.vendor_address,
                str(inv.date),
                str(inv.due_date) if inv.due_date else None,
                inv.currency,
                str(inv.subtotal),
                str(inv.tax),
                str(inv.total),
                inv.payment_terms,
            ],
        }
    )
    st.dataframe(header_df, hide_index=True, width="stretch")

    st.markdown("**Line items**")
    if inv.line_items:
        items_df = pd.DataFrame(
            [
                {
                    "item": li.item,
                    "quantity": li.quantity,
                    "unit_price": str(li.unit_price),
                    "line_total": str(li.line_total),
                }
                for li in inv.line_items
            ]
        )
        st.dataframe(items_df, hide_index=True, width="stretch")
    else:
        st.info("No line items.")

    if inv.ingestion_warnings:
        st.markdown("**Ingestion warnings**")
        for w in inv.ingestion_warnings:
            st.warning(f"`{w.code}` — {w.message}")
    if result.notes:
        st.markdown("**Notes**")
        for n in result.notes:
            st.info(n)

    _render_db_lookup_panel(inv)

    with st.expander("Full Invoice JSON"):
        st.code(
            json.dumps(inv.model_dump(mode="json"), indent=2, default=_decimal_default),
            language="json",
        )


_STATUS_BADGE = {
    STATUS_ACTIVE: "✅ active",
    STATUS_DISCONTINUED: "⏸ discontinued",
    STATUS_FRAUD: "🚫 fraud_flag",
    STATUS_BLOCKED: "🚫 blocked",
    STATUS_NEW: "🆕 new",
}


def _render_db_lookup_panel(inv) -> None:
    st.markdown("---")
    st.markdown("**Reference DB lookup** (preview of validation-phase signals)")
    if not DB_PATH.exists():
        st.info("Reference DB not initialized. Click **Initialize / refresh DB** in the sidebar.")
        return

    with connect(DB_PATH) as conn:
        # Vendor match
        vendor = lookup_vendor(conn, inv.vendor or "")
        if vendor is None:
            st.error(f"❓ Vendor `{inv.vendor}` not in vendor table — possible new or unrecognized counterparty.")
        else:
            badge = _STATUS_BADGE.get(vendor.status, vendor.status)
            line = f"{badge} matched **{vendor.name}** (`{vendor.vendor_id}`)"
            if vendor.aliases:
                line += f" via aliases {vendor.aliases}"
            if vendor.status == STATUS_BLOCKED:
                st.error(line + " — payment must not proceed.")
            elif vendor.status == STATUS_NEW:
                st.warning(line + " — first-time vendor, escalate.")
            else:
                st.success(line)
            if vendor.default_currency and inv.currency != vendor.default_currency:
                st.warning(
                    f"Currency drift: invoice in {inv.currency}, vendor default is {vendor.default_currency}."
                )

        # Dedup
        if inv.invoice_number and inv.vendor:
            already = has_invoice(conn, inv.invoice_number, inv.vendor)
            if already:
                st.warning(
                    f"Duplicate ledger entry: `{inv.invoice_number}` already recorded for `{inv.vendor}`."
                )
            else:
                st.caption(f"No ledger entry yet for `{inv.invoice_number}` × `{inv.vendor}`.")

        # Per-line item lookup
        rows = []
        for li in inv.line_items:
            db_item = lookup_item(conn, li.item)
            if db_item is None:
                rows.append(
                    {
                        "item": li.item,
                        "qty": li.quantity,
                        "qty vs stock": "❓ unknown SKU",
                        "status": "—",
                        "catalog price": "—",
                        "price drift": "—",
                    }
                )
                continue
            qty_marker = "✅"
            if li.quantity < 0:
                qty_marker = "❌ negative qty"
            elif li.quantity > db_item.stock and db_item.stock > 0:
                qty_marker = f"⚠️ {li.quantity} > stock {db_item.stock}"
            elif db_item.stock == 0:
                qty_marker = f"⚠️ stock 0"
            else:
                qty_marker = f"✅ {li.quantity} ≤ {db_item.stock}"

            drift = "—"
            if db_item.unit_price is not None:
                expected = db_item.unit_price
                actual = li.unit_price
                if expected == 0:
                    drift = "—"
                else:
                    pct = (actual - expected) / expected * 100
                    if abs(pct) < 1:
                        drift = "≈ catalog"
                    elif pct > 0:
                        drift = f"+{pct:.1f}% over catalog"
                    else:
                        drift = f"{pct:.1f}% under catalog"
            rows.append(
                {
                    "item": li.item,
                    "qty": li.quantity,
                    "qty vs stock": qty_marker,
                    "status": _STATUS_BADGE.get(db_item.status, db_item.status),
                    "catalog price": str(db_item.unit_price) if db_item.unit_price is not None else "—",
                    "price drift": drift,
                }
            )
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        else:
            st.info("No line items to look up.")


def _render_stub(name: str, description: str) -> None:
    st.subheader(f"{name} phase")
    st.info(f"🚧 Not implemented yet. {description}")


_VERDICT_BADGE = {
    "pass": ("✅ PASS", "success"),
    "needs_review": ("⚠️ NEEDS REVIEW", "warning"),
    "reject": ("❌ REJECT", "error"),
}


def _render_validation(path: Path) -> None:
    st.subheader("Validation report")
    if not DB_PATH.exists():
        st.warning("Reference DB not initialized. Use the sidebar button.")
        return
    try:
        ing = ingest(path, allow_llm=True)
    except Exception as e:  # noqa: BLE001
        st.error(f"Ingestion failed before validation could run: {type(e).__name__}: {e}")
        return
    invoice = ing.invoice
    with connect(DB_PATH) as conn:
        report = validate(invoice, conn=conn)

    label, kind = _VERDICT_BADGE[report.verdict]
    {"success": st.success, "warning": st.warning, "error": st.error}[kind](
        f"**Verdict:** {label} — {len(report.findings)} finding(s)"
    )
    by_sev = report.by_severity()
    cols = st.columns(3)
    cols[0].metric("Errors", len(by_sev["error"]))
    cols[1].metric("Warnings", len(by_sev["warn"]))
    cols[2].metric("Info", len(by_sev["info"]))

    if report.findings:
        st.markdown("**Findings**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "severity": f.severity,
                        "code": f.code,
                        "field": f.field or "—",
                        "message": f.message,
                    }
                    for f in report.findings
                ]
            ),
            hide_index=True,
            width="stretch",
        )

    if report.verdict == "pass":
        st.markdown("---")
        if st.button("Record this invoice in the ledger"):
            with connect(DB_PATH) as conn:
                record_invoice(
                    conn,
                    invoice_number=invoice.invoice_number,
                    vendor=invoice.vendor,
                    total=invoice.total,
                    source_path=str(path),
                )
            st.success(
                f"Recorded `{invoice.invoice_number}` × `{invoice.vendor}` in `invoice_ledger`."
            )


def main() -> None:
    path = _resolve_selection()
    if path is None:
        st.info("Pick a sample invoice or upload one from the sidebar to begin.")
        return

    st.caption(f"**File:** `{path.name}` ({path.suffix.lower().lstrip('.')})")
    (
        raw_tab,
        sanitized_tab,
        ingestion_tab,
        validation_tab,
        approval_tab,
        payment_tab,
        db_tab,
    ) = st.tabs(
        ["Raw", "Sanitized", "Ingestion", "Validation", "Approval", "Payment", "Reference DB"]
    )
    with raw_tab:
        raw = _render_raw(path)
    with sanitized_tab:
        _render_sanitized(raw)
    with ingestion_tab:
        _render_ingestion(path)
    with validation_tab:
        _render_validation(path)
    with approval_tab:
        _render_stub(
            "Approval",
            "Will route to the right approver based on amount thresholds and vendor.",
        )
    with payment_tab:
        _render_stub(
            "Payment",
            "Will schedule the payment via the configured rail and emit a receipt.",
        )
    with db_tab:
        _render_reference_db()


def _render_reference_db() -> None:
    st.subheader("Reference DB contents")
    if not DB_PATH.exists():
        st.info("DB not initialized yet — use the sidebar button.")
        return
    with connect(DB_PATH) as conn:
        inventory = list_inventory(conn)
        vendors = list_vendors(conn)
    st.markdown("**Inventory**")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "item": i.item,
                    "stock": i.stock,
                    "unit_price": str(i.unit_price) if i.unit_price is not None else "—",
                    "category": i.category or "—",
                    "status": _STATUS_BADGE.get(i.status, i.status),
                }
                for i in inventory
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    st.markdown("**Vendors**")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "vendor_id": v.vendor_id,
                    "name": v.name,
                    "status": _STATUS_BADGE.get(v.status, v.status),
                    "aliases": ", ".join(v.aliases) if v.aliases else "—",
                    "address": v.address or "—",
                    "default_currency": v.default_currency or "—",
                }
                for v in vendors
            ]
        ),
        hide_index=True,
        width="stretch",
    )


if __name__ == "__main__":
    main()
