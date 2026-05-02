"""Tests for the approval critic / reflection loop."""
from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from galatiq.agents.approval import ApprovalDecision
from galatiq.agents.critic import Critique, apply_override, critique
from galatiq.agents.pipeline import run_pipeline
from galatiq.agents.validation import Finding, ValidationReport
from galatiq.db import connect, init_db, list_approvals
from galatiq.models.invoice import Invoice


def _invoice(**overrides):
    base = {
        "invoice_number": "INV-CRIT",
        "vendor": "Acme Corp",
        "date": "2026-01-01",
        "due_date": "2099-01-01",
        "currency": "USD",
        "line_items": [{"item": "WidgetA", "quantity": 1, "unit_price": "10.00"}],
        "subtotal": "10.00",
        "tax": "0.00",
        "total": "10.00",
    }
    base.update(overrides)
    return Invoice.model_validate(base)


def _decision(status="auto_approved", role="system", policy="TIER-AUTO", total=Decimal("10")):
    return ApprovalDecision(
        status=status,
        approver_role=role,
        policy_id=policy,
        total_usd=total,
        justification="auto",
        escalations=[],
    )


# --- apply_override unit cases (no LLM involved) -----------------------------


def test_confirm_is_a_noop():
    d = _decision()
    assert apply_override(d, Critique(action="confirm", rationale="ok")) is d


def test_downgrade_promotes_auto_to_manager():
    d = _decision()  # auto_approved by system
    revised = apply_override(d, Critique(action="downgrade_to_pending_human", rationale="vendor concern"))
    assert revised.status == "pending_human"
    assert revised.approver_role == "manager"
    assert revised.policy_id == "TIER-MGR"
    assert "vendor concern" in revised.justification


def test_escalate_to_director():
    d = _decision(status="pending_human", role="manager", policy="TIER-MGR", total=Decimal("5000"))
    revised = apply_override(d, Critique(action="escalate_to_director", rationale="aggregate exposure"))
    assert revised.approver_role == "director"
    assert revised.policy_id == "TIER-DIR"


def test_critic_cannot_override_reject():
    d = _decision(status="rejected", role="none", policy=None)
    revised = apply_override(d, Critique(action="downgrade_to_pending_human", rationale="want override"))
    # Rules own rejects — the override is silently dropped.
    assert revised.status == "rejected"


# --- critique() with stubbed LLM ---------------------------------------------


@pytest.fixture(autouse=True)
def enable_llm(monkeypatch):
    monkeypatch.setenv("GALATIQ_LLM_AGENTS", "1")


def _stub(monkeypatch, response: Critique):
    def _fake(schema, *, system, user, max_retries=2):
        return response, 0

    import galatiq.agents._llm_helpers as helpers
    monkeypatch.setattr(helpers, "extract_structured", _fake)


def test_critique_returns_confirm_on_clean_decision(monkeypatch):
    _stub(monkeypatch, Critique(action="confirm", rationale="all good"))
    crit, err = critique(_invoice(), ValidationReport(findings=[], verdict="pass"), _decision())
    assert err is None
    assert crit.action == "confirm"


def test_critique_skips_llm_for_rejected_decisions(monkeypatch):
    _stub(monkeypatch, Critique(action="escalate_to_director", rationale="ignored"))
    rejected = _decision(status="rejected", role="none", policy=None)
    crit, err = critique(_invoice(), ValidationReport(verdict="reject"), rejected)
    assert crit.action == "confirm"
    # LLM not called → no error to report
    assert err is None


def test_critique_falls_back_when_llm_unavailable(monkeypatch):
    def _fake(schema, *, system, user, max_retries=2):
        raise RuntimeError("no key")

    import galatiq.agents._llm_helpers as helpers
    monkeypatch.setattr(helpers, "extract_structured", _fake)
    crit, err = critique(_invoice(), ValidationReport(verdict="pass"), _decision())
    assert err is not None
    assert crit.action == "confirm"


# --- end-to-end pipeline integration -----------------------------------------


def test_pipeline_records_post_critique_decision(tmp_path, monkeypatch):
    monkeypatch.setenv("GALATIQ_LLM_AGENTS", "1")
    db_path = tmp_path / "critic.db"
    init_db(db_path)
    receipt_dir = tmp_path / "receipts"

    # Stub: critic downgrades the auto-approval, all other LLM agents return safe defaults.
    from galatiq.agents.fraud_screener import FraudScreenResult
    from galatiq.agents.investigator import RiskAssessment
    from galatiq.agents.justifier import Justification
    from galatiq.agents.vendor_onboarding import VendorProfile

    canned = {
        FraudScreenResult: FraudScreenResult(findings=[]),
        Critique: Critique(
            action="downgrade_to_pending_human",
            rationale="vendor recently flagged in another system",
        ),
        VendorProfile: VendorProfile(
            recommendation="needs_more_info", rationale="placeholder"
        ),
        RiskAssessment: RiskAssessment(
            severity_summary="low",
            root_cause_hypothesis="n/a",
            recommended_action="approve_with_notes",
            items_to_verify=[],
        ),
        Justification: Justification(text="Audit narrative."),
    }

    def _fake_extract(schema, *, system, user, max_retries=2):
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

    inv_path = tmp_path / "clean.json"
    inv_path.write_text(json.dumps({
        "invoice_number": "INV-DOWNGRADE",
        "vendor": "Acme Corp",
        "date": "2026-01-01",
        "due_date": "2099-01-01",
        "currency": "USD",
        "line_items": [{"item": "WidgetA", "quantity": 1, "unit_price": "10.00"}],
        "subtotal": "10.00",
        "tax": "0.00",
        "total": "10.00",
    }))

    state = run_pipeline(inv_path, db_path=db_path, receipt_dir=receipt_dir)
    # Engine would have auto-approved; critic downgraded.
    assert state["pre_critique_decision"].status == "auto_approved"
    assert state["decision"].status == "pending_human"
    assert state["decision"].approver_role == "manager"
    # Audit log records the FINAL revised decision.
    with connect(db_path) as conn:
        rows = list_approvals(conn)
    assert len(rows) == 1
    assert rows[0]["status"] == "pending_human"
    assert rows[0]["approver_role"] == "manager"
