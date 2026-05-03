"""Tests for the approval council: profile selection, reviewer fallback,
deterministic aggregation, and end-to-end pipeline integration.
"""
from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from galatiq.agents.approval import ApprovalDecision
from galatiq.agents.pipeline import run_pipeline
from galatiq.agents.reviewers import ReviewerOpinion, aggregate_opinions
from galatiq.agents.reviewers.aggregator import AggregatedDecision, aggregate
from galatiq.agents.reviewers.profile import select_profile
from galatiq.agents.validation import ValidationReport
from galatiq.db import connect, init_db, list_approvals
from galatiq.models.invoice import Invoice


def _decision(status="auto_approved", role="system", policy="TIER-AUTO", total=Decimal("100")):
    return ApprovalDecision(
        status=status,
        approver_role=role,
        policy_id=policy,
        total_usd=total,
        justification="engine",
        escalations=[],
    )


def _invoice():
    return Invoice.model_validate({
        "invoice_number": "INV-COUNCIL",
        "vendor": "Acme Corp",
        "date": "2026-01-01",
        "due_date": "2099-01-01",
        "currency": "USD",
        "line_items": [{"item": "WidgetA", "quantity": 1, "unit_price": "10.00"}],
        "subtotal": "10.00",
        "tax": "0.00",
        "total": "10.00",
    })


# --- profile selector --------------------------------------------------------


def test_profile_lite_for_tier_auto():
    p = select_profile("TIER-AUTO")
    assert p.name == "lite"
    assert p.reviewers == ["fraud"]


def test_profile_standard_for_manager():
    p = select_profile("TIER-MGR")
    assert p.name == "standard"
    assert set(p.reviewers) == {"compliance", "fraud", "policy"}


def test_profile_deep_for_director_and_cfo():
    assert select_profile("TIER-DIR").name == "deep"
    cfo = select_profile("TIER-CFO")
    assert cfo.name == "deepest"
    assert cfo.max_tool_loops_per_reviewer >= 6


def test_profile_unknown_defaults_to_standard():
    p = select_profile(None)
    assert p.name == "standard"


# --- deterministic aggregate_opinions ---------------------------------------


def test_empty_opinions_keep_engine_decision():
    d = _decision()
    revised, verdict = aggregate_opinions(d, [])
    assert revised is d
    assert verdict == "approve"


def test_any_reject_overrides_to_council_reject():
    d = _decision()
    opinions = [
        ReviewerOpinion(reviewer="compliance", verdict="approve", severity="low", rationale="ok"),
        ReviewerOpinion(reviewer="fraud", verdict="reject", severity="high", rationale="impersonation pattern"),
    ]
    revised, verdict = aggregate_opinions(d, opinions)
    assert verdict == "reject"
    assert revised.status == "rejected"
    assert "impersonation" in revised.justification


def test_engine_reject_cannot_be_relaxed_by_council():
    d = _decision(status="rejected", role="none", policy=None)
    opinions = [
        ReviewerOpinion(reviewer="compliance", verdict="approve", severity="low", rationale="looks fine"),
    ]
    revised, _ = aggregate_opinions(d, opinions)
    assert revised.status == "rejected"


def test_downgrade_to_human_bumps_auto_to_manager():
    d = _decision()  # auto_approved/system/TIER-AUTO
    opinions = [
        ReviewerOpinion(reviewer="fraud", verdict="downgrade_to_human", severity="medium",
                        rationale="vendor has odd cadence"),
    ]
    revised, _ = aggregate_opinions(d, opinions)
    assert revised.status == "pending_human"
    assert revised.approver_role == "manager"
    assert revised.policy_id == "TIER-MGR"


def test_escalate_to_cfo_jumps_tiers():
    d = _decision(status="pending_human", role="manager", policy="TIER-MGR", total=Decimal("5000"))
    opinions = [
        ReviewerOpinion(reviewer="compliance", verdict="escalate_to_cfo", severity="high",
                        rationale="possible sanctions concern"),
    ]
    revised, _ = aggregate_opinions(d, opinions)
    assert revised.approver_role == "cfo"
    assert revised.policy_id == "TIER-CFO"


def test_most_conservative_opinion_wins():
    d = _decision()
    opinions = [
        ReviewerOpinion(reviewer="compliance", verdict="approve", severity="low", rationale="ok"),
        ReviewerOpinion(reviewer="fraud", verdict="approve_with_notes", severity="low", rationale="minor"),
        ReviewerOpinion(reviewer="policy", verdict="escalate_one_tier", severity="medium", rationale="cadence"),
    ]
    _, verdict = aggregate_opinions(d, opinions)
    assert verdict == "escalate_one_tier"


# --- LLM aggregator with stubbed helper -------------------------------------


@pytest.fixture(autouse=True)
def enable_llm(monkeypatch):
    monkeypatch.setenv("GALATIQ_LLM_AGENTS", "1")


def _stub_aggregator(monkeypatch, response: AggregatedDecision):
    def _fake(schema, *, system, user, max_retries=2):
        if schema is AggregatedDecision:
            return response, 0
        raise RuntimeError(f"unexpected schema {schema!r}")

    import galatiq.agents._llm_helpers as helpers
    monkeypatch.setattr(helpers, "extract_structured", _fake)


def test_aggregator_safety_override_keeps_engine_reject(monkeypatch):
    """If the LLM tries to override a rule-based reject, the safety net wins."""
    _stub_aggregator(
        monkeypatch,
        AggregatedDecision(
            final_status="auto_approved",
            final_approver_role="system",
            final_policy_id="TIER-AUTO",
            audit_narrative="trying to bypass",
        ),
    )
    rejected = _decision(status="rejected", role="none", policy=None)
    final, narrative, err = aggregate(_invoice(), ValidationReport(verdict="reject"), rejected, [])
    assert final.status == "rejected"
    assert "safety override" in final.justification


def test_aggregator_cannot_auto_approve_needs_review(monkeypatch):
    _stub_aggregator(
        monkeypatch,
        AggregatedDecision(
            final_status="auto_approved",
            final_approver_role="system",
            final_policy_id="TIER-AUTO",
            audit_narrative="overstepping",
        ),
    )
    pending = _decision(status="pending_human", role="manager", policy="TIER-MGR")
    final, _, _ = aggregate(_invoice(), ValidationReport(verdict="needs_review"), pending, [])
    assert final.status == "pending_human"


def test_aggregator_falls_back_when_llm_unavailable(monkeypatch):
    def _fake(schema, *, system, user, max_retries=2):
        raise RuntimeError("no api key")

    import galatiq.agents._llm_helpers as helpers
    monkeypatch.setattr(helpers, "extract_structured", _fake)
    final, narrative, err = aggregate(_invoice(), ValidationReport(verdict="pass"), _decision(), [])
    assert err is not None
    assert final.status == "auto_approved"  # deterministic aggregate keeps engine outcome
    assert narrative


# --- end-to-end with full council in fallback mode ---------------------------


def test_pipeline_runs_council_with_tier_lite_for_small_invoice(tmp_path, monkeypatch):
    """Small invoice ($10) → TIER-AUTO → 'lite' profile → just the fraud reviewer."""
    monkeypatch.setenv("GALATIQ_LLM_AGENTS", "0")  # use deterministic fallbacks throughout
    db = tmp_path / "council.db"
    init_db(db)
    inv_path = tmp_path / "small.json"
    inv_path.write_text(json.dumps({
        "invoice_number": "INV-SMALL",
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
    assert state["council_profile"] is not None
    assert state["council_profile"].name == "lite"
    assert [o.reviewer for o in state["reviewer_opinions"]] == ["fraud"]
    assert state["decision"].status == "auto_approved"
    # Audit log got the FINAL decision.
    with connect(db) as conn:
        rows = list_approvals(conn)
    assert len(rows) == 1
    assert rows[0]["status"] == "auto_approved"


def test_pipeline_runs_full_council_for_manager_tier(tmp_path, monkeypatch):
    """$2k invoice → TIER-MGR → 'standard' profile → all 3 reviewers run."""
    monkeypatch.setenv("GALATIQ_LLM_AGENTS", "0")
    db = tmp_path / "council2.db"
    init_db(db)
    inv_path = tmp_path / "midsize.json"
    inv_path.write_text(json.dumps({
        "invoice_number": "INV-MID",
        "vendor": "Acme Corp",
        "date": "2026-01-01",
        "due_date": "2099-01-01",
        "currency": "USD",
        "line_items": [{"item": "BoltPack", "quantity": 400, "unit_price": "5.00"}],
        "subtotal": "2000.00",
        "tax": "0.00",
        "total": "2000.00",
    }))
    state = run_pipeline(inv_path, db_path=db, receipt_dir=tmp_path / "r")
    assert state["council_profile"].name == "standard"
    reviewers = {o.reviewer for o in state["reviewer_opinions"]}
    assert reviewers == {"compliance", "fraud", "policy"}


def test_pipeline_skips_council_for_engine_reject(tmp_path, monkeypatch):
    """Rule engine rejected → council short-circuits to save LLM compute."""
    monkeypatch.setenv("GALATIQ_LLM_AGENTS", "0")
    db = tmp_path / "council3.db"
    init_db(db)
    inv_path = tmp_path / "reject.json"
    inv_path.write_text(json.dumps({
        "invoice_number": "INV-REJ-COUNCIL",
        "vendor": "Acme Corp",
        "date": "2026-01-01",
        "currency": "USD",
        "line_items": [{"item": "PhantomSKU", "quantity": 1, "unit_price": "9.99"}],
        "subtotal": "9.99",
        "tax": "0.00",
        "total": "9.99",
    }))
    state = run_pipeline(inv_path, db_path=db, receipt_dir=tmp_path / "r")
    assert state["decision"].status == "rejected"
    # Council didn't run on a hard reject.
    assert state.get("council_profile") is None
    assert state.get("reviewer_opinions") == []
