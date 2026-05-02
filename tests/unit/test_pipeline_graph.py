from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from galatiq.agents.pipeline import run_pipeline
from galatiq.db import init_db


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "data" / "invoices"


@pytest.fixture
def db_path(tmp_path):
    p = tmp_path / "inv.db"
    init_db(p)
    return p


def _run(name: str, db_path: Path):
    return run_pipeline(FIXTURES / name, db_path=db_path, allow_llm=False)


def test_pipeline_clean_invoice_passes_and_routes(db_path):
    state = _run("invoice_1004.json", db_path)
    assert "error" not in state
    assert state["validation"].verdict == "pass"
    # $1890 -> manager band
    assert state["approval"].status == "pending_human"
    assert state["approval"].approver_role == "manager"


def test_pipeline_stock_overflow_rejects(db_path):
    state = _run("invoice_1005.json", db_path)
    assert state["validation"].verdict == "reject"
    assert state["approval"].status == "rejected"


def test_pipeline_unknown_sku_routes_to_review(db_path):
    state = _run("invoice_1016.json", db_path)
    assert state["validation"].verdict == "needs_review"
    assert state["approval"].status == "pending_human"


def test_pipeline_eur_invoice_normalises_to_usd(db_path):
    state = _run("invoice_1014.xml", db_path)
    assert state["validation"].verdict == "pass"
    # €4125 * 1.08 = 4455 USD -> manager band
    assert state["approval"].approver_role == "manager"
    assert state["approval"].total_usd > 0


def test_pipeline_ingestion_failure_short_circuits(db_path):
    state = _run("invoice_1007.csv", db_path)  # bad date format, no LLM
    assert "error" in state
    assert state["error_phase"] == "ingestion"
    assert "validation" not in state
    assert "approval" not in state
