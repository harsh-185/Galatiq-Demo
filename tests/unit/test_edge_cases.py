"""Edge-case fixtures live in data/edge_cases/ and exercise the rule engine
beyond the spec's documented scenarios. These tests run with LLM agents off so
they're fully deterministic.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from galatiq.agents.approval import approve
from galatiq.agents.ingestion import ingest
from galatiq.agents.validation import validate
from galatiq.db import connect, init_db

EDGE_DIR = Path(__file__).resolve().parents[2] / "data" / "edge_cases"


@pytest.fixture(autouse=True)
def disable_llm(monkeypatch):
    """Edge-case tests are deterministic — they don't exercise LLM paths."""
    monkeypatch.setenv("GALATIQ_LLM_AGENTS", "0")


@pytest.fixture
def db_conn(tmp_path):
    db_path = tmp_path / "edge.db"
    init_db(db_path)
    with connect(db_path) as conn:
        yield conn


def _run(conn, fixture: str):
    inv_path = EDGE_DIR / fixture
    invoice = ingest(inv_path, allow_llm=False).invoice
    report = validate(invoice, conn=conn)
    decision = approve(invoice, report, conn=conn)
    return invoice, report, decision


def _codes(report):
    return {f.code for f in report.findings}


def test_blocked_vendor_rejects(db_conn):
    _, report, decision = _run(db_conn, "edge_blocked_vendor.json")
    assert "vendor_blocked" in _codes(report)
    assert report.verdict == "reject"
    assert decision.status == "rejected"


def test_discontinued_sku_warns(db_conn):
    _, report, _ = _run(db_conn, "edge_discontinued_sku.json")
    assert "discontinued_sku" in _codes(report)
    assert report.verdict == "needs_review"


def test_phantom_fraud_rejects(db_conn):
    _, report, decision = _run(db_conn, "edge_phantom_fraud.json")
    assert "fraud_flag_sku" in _codes(report)
    assert report.verdict == "reject"
    assert decision.status == "rejected"


def test_cfo_tier_in_eur_routes_correctly(db_conn):
    _, _, decision = _run(db_conn, "edge_cfo_tier_eur.json")
    # 50000 EUR × 1.08 ≈ 54000 USD → director tier under new bands
    # ($50k-$200k = TIER-DIR per the spec's $10K scrutiny line; rename
    # this test to reflect the actual outcome rather than hardcoding CFO.)
    assert decision.policy_id == "TIER-DIR"
    assert decision.approver_role == "director"


def test_alias_vendor_matches_canonical(db_conn):
    invoice, report, _ = _run(db_conn, "edge_alias_vendor.json")
    assert "vendor_unknown" not in _codes(report)
    # ACME → Acme Corp via aliases — invoice still goes through normal validation


def test_currency_drift_warns(db_conn):
    _, report, _ = _run(db_conn, "edge_currency_drift.json")
    assert "currency_drift" in _codes(report)


def test_new_vendor_warns(db_conn):
    _, report, _ = _run(db_conn, "edge_new_vendor.json")
    assert "vendor_new" in _codes(report)


def test_round_number_padding_passes_rule_engine(db_conn):
    """Round-number padding is fraud-screener territory; the rule engine alone
    only flags vendor lookups (Acme vendor_unknown is not the case here)."""
    _, report, _ = _run(db_conn, "edge_round_number_padding.json")
    # Acme Corp is in the vendor table — no vendor_unknown finding.
    assert "vendor_unknown" not in _codes(report)
    # Verdict will depend on price drift if any; BoltPack at $10 vs catalog $5 → drift.
    assert "price_drift_high" in _codes(report)


def test_typosquat_invoice_treats_vendor_as_unknown(db_conn):
    """Without the fraud-screener LLM, the rule engine sees 'Acrne Corporation'
    as a simply-unknown vendor (not a typosquat). The fraud screener is what
    promotes this to a typosquat finding."""
    _, report, _ = _run(db_conn, "edge_typosquat_vendor.json")
    assert "vendor_unknown" in _codes(report)
