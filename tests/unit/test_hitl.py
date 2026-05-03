"""End-to-end test for the human-in-the-loop queue.

Scenario:
  1. Run the pipeline on a needs_review invoice. The aggregator routes it to
     pending_human, the HITL queue node writes a row, the pay node skips.
  2. Use the CLI's `human resolve --action approve` to override the decision.
     The resolver re-runs the pay phase with an auto_approved decision.
  3. Verify the payment_log gained a scheduled entry.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from galatiq.agents.pipeline import run_pipeline
from galatiq.cli.app import app
from galatiq.db import (
    connect,
    get_review_entry,
    init_db,
    list_payments,
    list_pending_reviews,
)


@pytest.fixture(autouse=True)
def disable_llm(monkeypatch):
    monkeypatch.setenv("GALATIQ_LLM_AGENTS", "0")


def test_hitl_queue_then_resolve_approve(tmp_path):
    db = tmp_path / "hitl.db"
    init_db(db)
    receipts = tmp_path / "receipts"

    inv_path = tmp_path / "review.json"
    inv_path.write_text(json.dumps({
        "invoice_number": "INV-HITL-001",
        "vendor": "Mystery Vendor",
        "date": "2026-01-01",
        "due_date": "2099-01-01",
        "currency": "USD",
        "line_items": [{"item": "WidgetA", "quantity": 1, "unit_price": "10.00"}],
        "subtotal": "10.00",
        "tax": "0.00",
        "total": "10.00",
    }))

    state = run_pipeline(inv_path, db_path=db, receipt_dir=receipts)
    assert state["decision"].status == "pending_human"
    review_id = state.get("human_review_id")
    assert review_id is not None

    # Pre-resolve: queue has the entry, payment is skipped.
    with connect(db) as conn:
        pending = list_pending_reviews(conn)
        assert len(pending) == 1
        assert pending[0]["id"] == review_id
        pre_payments = list_payments(conn)
    skipped_payments = [p for p in pre_payments if p["status"] == "skipped"]
    assert len(skipped_payments) == 1

    # Use the CLI to resolve as approved.
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "human", "resolve",
            "--id", str(review_id),
            "--action", "approve",
            "--note", "Manually verified vendor offline",
            "--db", str(db),
            "--receipts", str(receipts),
        ],
        env={"GALATIQ_LLM_AGENTS": "0"},
    )
    assert result.exit_code == 0, result.output
    assert "scheduled" in result.output.lower()

    # Post-resolve: queue marked resolved, a scheduled payment exists.
    with connect(db) as conn:
        entry = get_review_entry(conn, review_id)
        post_payments = list_payments(conn)
    assert entry["resolved_at"] is not None
    assert entry["resolution"] == "approve"
    scheduled = [p for p in post_payments if p["status"] == "scheduled"]
    assert len(scheduled) == 1
    assert scheduled[0]["invoice_number"] == "INV-HITL-001"


def test_hitl_resolve_reject_does_not_schedule_payment(tmp_path):
    db = tmp_path / "hitl2.db"
    init_db(db)
    receipts = tmp_path / "receipts"

    inv_path = tmp_path / "review2.json"
    inv_path.write_text(json.dumps({
        "invoice_number": "INV-HITL-002",
        "vendor": "Mystery Vendor",
        "date": "2026-01-01",
        "currency": "USD",
        "line_items": [{"item": "WidgetA", "quantity": 1, "unit_price": "10.00"}],
        "subtotal": "10.00",
        "tax": "0.00",
        "total": "10.00",
    }))
    state = run_pipeline(inv_path, db_path=db, receipt_dir=receipts)
    review_id = state["human_review_id"]

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "human", "resolve",
            "--id", str(review_id),
            "--action", "reject",
            "--note", "Vendor confirmed unauthorized",
            "--db", str(db),
            "--receipts", str(receipts),
        ],
        env={"GALATIQ_LLM_AGENTS": "0"},
    )
    assert result.exit_code == 0
    assert "rejected" in result.output.lower()

    with connect(db) as conn:
        post_payments = list_payments(conn)
    scheduled = [p for p in post_payments if p["status"] == "scheduled"]
    assert len(scheduled) == 0


def test_hitl_resolve_unknown_id_errors(tmp_path):
    db = tmp_path / "hitl3.db"
    init_db(db)
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["human", "resolve", "--id", "9999", "--action", "approve", "--db", str(db)],
        env={"GALATIQ_LLM_AGENTS": "0"},
    )
    assert result.exit_code != 0
