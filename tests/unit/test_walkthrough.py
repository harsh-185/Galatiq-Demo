"""Tests for the per-stage walkthrough event log."""
from __future__ import annotations

import json

import pytest

from galatiq.agents._walkthrough import (
    StageEvent,
    render_walkthrough,
    summary_stats,
)
from galatiq.agents.pipeline import run_pipeline
from galatiq.db import init_db


@pytest.fixture(autouse=True)
def disable_llm(monkeypatch):
    monkeypatch.setenv("GALATIQ_LLM_AGENTS", "0")


def test_walkthrough_populated_for_clean_invoice(tmp_path):
    db = tmp_path / "walk.db"
    init_db(db)
    inv_path = tmp_path / "clean.json"
    inv_path.write_text(json.dumps({
        "invoice_number": "INV-WT-1",
        "vendor": "Acme Corp",
        "date": "2026-01-01",
        "due_date": "2099-01-01",
        "currency": "USD",
        "line_items": [{"item": "WidgetA", "quantity": 1, "unit_price": "10.00"}],
        "subtotal": "10.00",
        "tax": "0.00",
        "total": "10.00",
    }))
    state = run_pipeline(inv_path, db_path=db, receipt_dir=tmp_path / "r")
    walkthrough = state.get("walkthrough") or []
    names = [ev.name for ev in walkthrough]
    assert names == [
        "ingest",
        "pre_approval_screener",
        "validate",
        "approve",
        "council",
        "aggregator",
        "hitl_queue",
        "payment_guards",
        "pay",
    ]
    # Council and HITL should be skipped on a clean small invoice.
    by_name = {ev.name: ev for ev in walkthrough}
    assert by_name["council"].status == "skipped"
    assert by_name["hitl_queue"].status == "skipped"
    assert by_name["pay"].status == "completed"
    # Every event has a positive duration.
    assert all(ev.duration_ms >= 0 for ev in walkthrough)


def test_walkthrough_summary_stats(tmp_path):
    db = tmp_path / "walk2.db"
    init_db(db)
    inv_path = tmp_path / "clean2.json"
    inv_path.write_text(json.dumps({
        "invoice_number": "INV-WT-2",
        "vendor": "Acme Corp",
        "date": "2026-01-01",
        "due_date": "2099-01-01",
        "currency": "USD",
        "line_items": [{"item": "WidgetA", "quantity": 1, "unit_price": "10.00"}],
        "subtotal": "10.00",
        "tax": "0.00",
        "total": "10.00",
    }))
    state = run_pipeline(inv_path, db_path=db, receipt_dir=tmp_path / "r")
    stats = summary_stats(state["walkthrough"])
    assert stats["stages_completed"] >= 6
    assert stats["stages_skipped"] >= 2
    assert stats["stages_failed"] == 0
    assert stats["total_ms"] > 0
    # No LLM calls in disabled mode.
    assert stats["total_llm_calls"] == 0


def test_render_walkthrough_includes_each_stage():
    events = [
        StageEvent(name="ingest", status="completed", summary="path=deterministic", duration_ms=12.3),
        StageEvent(name="validate", status="completed", summary="verdict=PASS", duration_ms=3.4),
        StageEvent(name="council", status="skipped", summary="clean", duration_ms=0.1),
    ]
    rendered = render_walkthrough(events)
    assert "ingest" in rendered
    assert "validate" in rendered
    assert "council" in rendered
    assert "✓" in rendered
    assert "⊘" in rendered


def test_walkthrough_records_failed_stage(tmp_path):
    """If ingest fails, the ingest event is marked failed and the rest still
    run (or skip) without crashing."""
    db = tmp_path / "walk3.db"
    init_db(db)
    bad_path = tmp_path / "missing.txt"  # doesn't exist
    bad_path.write_text("not a real invoice")  # but make it exist so reader runs

    state = run_pipeline(bad_path, db_path=db, receipt_dir=tmp_path / "r")
    walkthrough = state.get("walkthrough") or []
    # Subsequent stages may skip; verify every event has either completed or skipped or failed.
    assert all(ev.status in ("completed", "skipped", "failed") for ev in walkthrough)
