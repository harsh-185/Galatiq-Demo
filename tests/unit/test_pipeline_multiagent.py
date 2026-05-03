"""End-to-end pipeline tests after the agent consolidation.

The old `fraud_screener`, `investigator`, and `vendor_onboarding` are now one
`pre_approval_screener`. The supervisor's specialist routing is gone — the
pipeline is linear post-consolidation. The council still runs (tier-scaled),
its skip-gate now bypasses it on clean+small+known cases.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from galatiq.agents.pipeline import run_pipeline
from galatiq.agents.pre_approval_screener import (
    PreApprovalSummary,
    VendorProfile,
    _ScreenedFinding,
)
from galatiq.agents.reviewers import ReviewerOpinion
from galatiq.agents.reviewers.aggregator import AggregatedDecision
from galatiq.db import connect, init_db, list_pending_reviews


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload))
    return path


@pytest.fixture
def db_path(tmp_path):
    p = tmp_path / "multi.db"
    init_db(p)
    return p


@pytest.fixture
def receipt_dir(tmp_path):
    return tmp_path / "receipts"


@pytest.fixture(autouse=True)
def enable_llm_agents(monkeypatch):
    monkeypatch.setenv("GALATIQ_LLM_AGENTS", "1")


def _approve_reviewer_opinion(name="fraud"):
    return ReviewerOpinion(
        reviewer=name,
        verdict="approve",
        severity="low",
        rationale="no concerns",
    )


@pytest.fixture
def stub_llm(monkeypatch):
    """Default stubs: empty pre-approval summary + reviewer 'approve' opinion +
    aggregator-fallback (deterministic) so the engine decision is preserved."""
    canned = {
        PreApprovalSummary: PreApprovalSummary(),
        ReviewerOpinion: _approve_reviewer_opinion(),
    }

    def _fake_extract(schema, *, system, user, max_retries=2):
        if schema is AggregatedDecision:
            raise RuntimeError("force aggregator fallback in tests")
        if schema not in canned:
            raise RuntimeError(f"no stub for {schema!r}")
        return canned[schema], 0

    def _fake_tool_agent(schema, *, system, user, tools, fallback, max_tool_loops=4):
        if schema not in canned:
            return fallback(), f"no stub for {schema!r}", []
        return canned[schema], None, []

    import galatiq.agents._llm_helpers as helpers
    monkeypatch.setattr(helpers, "extract_structured", _fake_extract)
    monkeypatch.setattr(helpers, "run_tool_using_agent", _fake_tool_agent)
    return canned


# --- pass route -------------------------------------------------------------


def test_clean_small_invoice_skips_council(tmp_path, db_path, receipt_dir, stub_llm):
    """Clean Acme $10 invoice → TIER-AUTO + clean → council should be skipped."""
    inv = _write(
        tmp_path / "pass.json",
        {
            "invoice_number": "INV-PASS",
            "vendor": "Acme Corp",
            "date": "2026-01-01",
            "due_date": "2099-01-01",
            "currency": "USD",
            "line_items": [{"item": "WidgetA", "quantity": 1, "unit_price": "10.00"}],
            "subtotal": "10.00",
            "tax": "0.00",
            "total": "10.00",
        },
    )
    state = run_pipeline(inv, db_path=db_path, receipt_dir=receipt_dir)
    assert state["report"].verdict == "pass"
    assert state["decision"].status == "auto_approved"
    assert state["payment"].status == "scheduled"
    assert state.get("council_skipped") is True
    assert state.get("reviewer_opinions") == []
    assert state.get("audit_narrative") is not None


# --- needs_review route -----------------------------------------------------


def test_unknown_vendor_routes_to_pending_human(tmp_path, db_path, receipt_dir, stub_llm):
    """Mystery Vendor → vendor_unknown warn → needs_review → pending_human + HITL."""
    inv = _write(
        tmp_path / "review.json",
        {
            "invoice_number": "INV-REVIEW",
            "vendor": "Mystery Vendor",
            "date": "2026-01-01",
            "currency": "USD",
            "line_items": [{"item": "WidgetA", "quantity": 1, "unit_price": "10.00"}],
            "subtotal": "10.00",
            "tax": "0.00",
            "total": "10.00",
        },
    )
    state = run_pipeline(inv, db_path=db_path, receipt_dir=receipt_dir)
    assert state["report"].verdict == "needs_review"
    assert state["decision"].status == "pending_human"
    assert state["payment"].status == "skipped"
    # HITL queue gets the row.
    assert state.get("human_review_id") is not None
    with connect(db_path) as conn:
        pending = list_pending_reviews(conn)
    assert len(pending) == 1
    # Council was NOT skipped — needs_review is not clean.
    assert state.get("council_skipped") is False


# --- new vendor route -------------------------------------------------------


def test_new_vendor_runs_council_without_separate_onboarding_node(tmp_path, db_path, receipt_dir, monkeypatch):
    """The merged pre_approval_screener handles vendor onboarding inline; the
    pipeline no longer has a separate vendor_onboarding node."""
    monkeypatch.setenv("GALATIQ_LLM_AGENTS", "1")
    canned = {
        PreApprovalSummary: PreApprovalSummary(
            risk_severity="low",
            risk_hypothesis="First-time vendor; flag for onboarding review.",
            vendor_profile=VendorProfile(
                suggested_aliases=["NewCo Inc"],
                recommendation="approve_onboarding",
                rationale="Vendor address consistent with DB record.",
            ),
        ),
        ReviewerOpinion: _approve_reviewer_opinion(),
    }

    def _fake_extract(schema, *, system, user, max_retries=2):
        if schema is AggregatedDecision:
            raise RuntimeError("force fallback")
        return canned[schema], 0

    def _fake_tool_agent(schema, *, system, user, tools, fallback, max_tool_loops=4):
        return canned[schema], None, []

    import galatiq.agents._llm_helpers as helpers
    monkeypatch.setattr(helpers, "extract_structured", _fake_extract)
    monkeypatch.setattr(helpers, "run_tool_using_agent", _fake_tool_agent)

    inv = _write(
        tmp_path / "newvendor.json",
        {
            "invoice_number": "INV-NEW",
            "vendor": "NewCo",
            "date": "2026-01-01",
            "currency": "USD",
            "line_items": [{"item": "WidgetA", "quantity": 1, "unit_price": "10.00"}],
            "subtotal": "10.00",
            "tax": "0.00",
            "total": "10.00",
        },
    )
    state = run_pipeline(inv, db_path=db_path, receipt_dir=receipt_dir)
    assert state["pre_approval_summary"] is not None
    assert state["pre_approval_summary"].vendor_profile is not None
    assert state["pre_approval_summary"].vendor_profile.recommendation == "approve_onboarding"
    # NewCo is status=new → vendor_new warn → needs_review verdict.
    assert state["report"].verdict == "needs_review"


# --- reject route -----------------------------------------------------------


def test_engine_reject_skips_council_and_payment(tmp_path, db_path, receipt_dir, stub_llm):
    inv = _write(
        tmp_path / "reject.json",
        {
            "invoice_number": "INV-REJ-MA",
            "vendor": "Acme Corp",
            "date": "2026-01-01",
            "currency": "USD",
            "line_items": [{"item": "PhantomSKU", "quantity": 1, "unit_price": "9.99"}],
            "subtotal": "9.99",
            "tax": "0.00",
            "total": "9.99",
        },
    )
    state = run_pipeline(inv, db_path=db_path, receipt_dir=receipt_dir)
    assert state["report"].verdict == "reject"
    assert state["decision"].status == "rejected"
    # Council short-circuits on hard rejects.
    assert state.get("council_skipped") is True
    assert state.get("audit_narrative") is not None
    assert state["payment"].status == "skipped"


# --- pre-approval screener can flip pass to needs_review --------------------


def test_pre_approval_skip_gate_fires_on_clean_invoice(tmp_path, db_path, receipt_dir, monkeypatch):
    """Clean Acme + WidgetA invoice should trip the deterministic pre-approval
    skip gate — no LLM screening calls, no fraud findings emitted."""
    monkeypatch.setenv("GALATIQ_LLM_AGENTS", "1")
    # Stub everything to RAISE if called — proves no LLM call was made.
    def _fake_extract(schema, *, system, user, max_retries=2):
        raise AssertionError(f"unexpected LLM call for {schema!r}")

    def _fake_tool_agent(schema, *, system, user, tools, fallback, max_tool_loops=4):
        raise AssertionError(f"unexpected tool-using LLM call for {schema!r}")

    import galatiq.agents._llm_helpers as helpers
    monkeypatch.setattr(helpers, "extract_structured", _fake_extract)
    monkeypatch.setattr(helpers, "run_tool_using_agent", _fake_tool_agent)

    inv = _write(
        tmp_path / "clean.json",
        {
            "invoice_number": "INV-CLEAN-PRE",
            "vendor": "Acme Corp",
            "date": "2026-01-01",
            "due_date": "2099-01-01",
            "currency": "USD",
            "line_items": [{"item": "WidgetA", "quantity": 1, "unit_price": "10.00"}],
            "subtotal": "10.00",
            "tax": "0.00",
            "total": "10.00",
        },
    )
    state = run_pipeline(inv, db_path=db_path, receipt_dir=receipt_dir)
    # Pre-approval skipped → no fraud findings, screener summary is empty.
    summary = state.get("pre_approval_summary")
    assert summary is not None
    assert summary.fraud_findings == []
    assert summary.risk_severity == "none"
    # Whole pipeline completed without any LLM call.
    assert state["report"].verdict == "pass"
    assert state["decision"].status == "auto_approved"
    assert state["payment"].status == "scheduled"


def test_pre_approval_runs_when_vendor_is_unknown(tmp_path, db_path, receipt_dir, monkeypatch):
    """When the deterministic gate doesn't fire (e.g. unknown vendor), the
    LLM screener IS invoked and its findings flow into validate."""
    monkeypatch.setenv("GALATIQ_LLM_AGENTS", "1")
    canned = {
        PreApprovalSummary: PreApprovalSummary(
            fraud_findings=[
                _ScreenedFinding(
                    code="vendor_typosquat",
                    severity="warn",
                    message="vendor 'Mystery Vendor' resembles no known canonical name",
                )
            ],
            risk_severity="medium",
            risk_hypothesis="Unknown vendor with typosquat-feeling name.",
        ),
        ReviewerOpinion: _approve_reviewer_opinion(),
    }

    def _fake_extract(schema, *, system, user, max_retries=2):
        if schema is AggregatedDecision:
            raise RuntimeError("force fallback")
        return canned[schema], 0

    def _fake_tool_agent(schema, *, system, user, tools, fallback, max_tool_loops=4):
        return canned[schema], None, []

    import galatiq.agents._llm_helpers as helpers
    monkeypatch.setattr(helpers, "extract_structured", _fake_extract)
    monkeypatch.setattr(helpers, "run_tool_using_agent", _fake_tool_agent)

    inv = _write(
        tmp_path / "unknown.json",
        {
            "invoice_number": "INV-UNK",
            "vendor": "Mystery Vendor",
            "date": "2026-01-01",
            "due_date": "2099-01-01",
            "currency": "USD",
            "line_items": [{"item": "WidgetA", "quantity": 1, "unit_price": "10.00"}],
            "subtotal": "10.00",
            "tax": "0.00",
            "total": "10.00",
        },
    )
    state = run_pipeline(inv, db_path=db_path, receipt_dir=receipt_dir)
    # Skip-gate did NOT fire → screener ran → fraud finding present.
    assert any(f.code == "vendor_typosquat" for f in state["report"].findings)
    assert state["report"].verdict == "needs_review"
    assert state["decision"].status == "pending_human"


# --- LLM-disabled mode ------------------------------------------------------


def test_disabled_mode_skips_all_llm_agents(tmp_path, db_path, receipt_dir, monkeypatch):
    monkeypatch.setenv("GALATIQ_LLM_AGENTS", "0")
    inv = _write(
        tmp_path / "noLLM.json",
        {
            "invoice_number": "INV-NOLLM",
            "vendor": "Acme Corp",
            "date": "2026-01-01",
            "due_date": "2099-01-01",
            "currency": "USD",
            "line_items": [{"item": "WidgetA", "quantity": 1, "unit_price": "10.00"}],
            "subtotal": "10.00",
            "tax": "0.00",
            "total": "10.00",
        },
    )
    state = run_pipeline(inv, db_path=db_path, receipt_dir=receipt_dir)
    assert state.get("fraud_findings") == []
    assert state["payment"].status == "scheduled"
    assert state.get("audit_narrative") is not None
    assert state.get("llm_agent_errors") == []
