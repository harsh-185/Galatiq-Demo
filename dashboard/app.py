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

from galatiq.agents.ingestion import ingest  # noqa: E402
from galatiq.io.readers import read_invoice  # noqa: E402
from galatiq.io.sanitize import strip_control_tags  # noqa: E402

INVOICE_DIR = REPO_ROOT / "data" / "invoices"
SUPPORTED_SUFFIXES = {".txt", ".json", ".csv", ".xml", ".pdf"}

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

    with st.expander("Full Invoice JSON"):
        st.code(
            json.dumps(inv.model_dump(mode="json"), indent=2, default=_decimal_default),
            language="json",
        )


def _render_stub(name: str, description: str) -> None:
    st.subheader(f"{name} phase")
    st.info(f"🚧 Not implemented yet. {description}")


def main() -> None:
    path = _resolve_selection()
    if path is None:
        st.info("Pick a sample invoice or upload one from the sidebar to begin.")
        return

    st.caption(f"**File:** `{path.name}` ({path.suffix.lower().lstrip('.')})")
    raw_tab, sanitized_tab, ingestion_tab, validation_tab, approval_tab, payment_tab = st.tabs(
        ["Raw", "Sanitized", "Ingestion", "Validation", "Approval", "Payment"]
    )
    with raw_tab:
        raw = _render_raw(path)
    with sanitized_tab:
        _render_sanitized(raw)
    with ingestion_tab:
        _render_ingestion(path)
    with validation_tab:
        _render_stub(
            "Validation",
            "Will cross-check totals, dates, vendor identity, and policy rules.",
        )
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


if __name__ == "__main__":
    main()
