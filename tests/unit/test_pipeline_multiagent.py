"""Integration tests for the multi-agent LangGraph pipeline.

We stub the LLM helper so each specialist returns a known structured result.
Then we verify the supervisor routes correctly:
- pass → direct to approve (no investigator, no onboarding)
- needs_review → investigator runs
- new vendor → vendor_onboarding runs
- reject → straight to approve (rejected decision recorded)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from galatiq.agents.critic import Critique
from galatiq.agents.fraud_screener import FraudScreenResult, _ScreenedFinding
from galatiq.agents.investigator import RiskAssessment
from galatiq.agents.justifier import Justification
from galatiq.agents.pipeline import run_pipeline
from galatiq.agents.vendor_onboarding import VendorProfile
from galatiq.db import connect, init_db


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


@pytest.fixture
def stub_llm(monkeypatch):
    """Route each schema to a canned response."""
    canned = {
        FraudScreenResult: FraudScreenResult(findings=[]),
        VendorProfile: VendorProfile(
            suggested_aliases=["NewCo Inc"],
            normalized_address="789 Newcomer Rd",
            default_currency_guess="USD",
            recommendation="approve_onboarding",
            rationale="Looks legitimate.",
        ),
        RiskAssessment: RiskAssessment(
            severity_summary="medium — vendor lookup failed",
            root_cause_hypothesis="Vendor name shorthand.",
            recommended_action="request_clarification",
            items_to_verify=["Confirm canonical vendor name"],
        ),
        Justification: Justification(text="Stub narrative."),
        Critique: Critique(action="confirm", rationale="rule engine looks correct"),
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
    return canned


def test_pass_route_skips_investigator_and_onboarding(tmp_path, db_path, receipt_dir, stub_llm):
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
    # Specialists that did NOT run produce no state entry.
    assert state.get("vendor_profile") is None
    assert state.get("risk_assessment") is None
    # fraud_screener and justifier always run.
    assert state.get("fraud_findings") == []
    assert state["llm_justification"].text == "Stub narrative."


def test_needs_review_route_invokes_investigator(tmp_path, db_path, receipt_dir, stub_llm):
    # Mystery Vendor → vendor_unknown warn → needs_review
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
    assert state["risk_assessment"] is not None
    assert state["risk_assessment"].recommended_action == "request_clarification"
    # Vendor onboarding does NOT run (no vendor row to onboard from).
    assert state.get("vendor_profile") is None
    assert state["payment"].status == "skipped"


def test_new_vendor_route_invokes_onboarding(tmp_path, db_path, receipt_dir, stub_llm):
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
    assert state["vendor_profile"] is not None
    assert state["vendor_profile"].recommendation == "approve_onboarding"
    # NewCo is status=new → vendor_new warn → needs_review verdict
    assert state["report"].verdict == "needs_review"
    # Investigator does NOT run for the new-vendor branch (router prefers onboarding).
    assert state.get("risk_assessment") is None


def test_reject_route_skips_specialists_and_runs_justifier(tmp_path, db_path, receipt_dir, stub_llm):
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
    # Investigator/onboarding don't run for rejects.
    assert state.get("risk_assessment") is None
    assert state.get("vendor_profile") is None
    # Justifier still runs to write the audit narrative for the rejection.
    assert state["llm_justification"].text == "Stub narrative."


def test_fraud_finding_can_promote_pass_to_needs_review(tmp_path, db_path, receipt_dir, monkeypatch):
    monkeypatch.setenv("GALATIQ_LLM_AGENTS", "1")

    canned = {
        FraudScreenResult: FraudScreenResult(
            findings=[
                _ScreenedFinding(
                    code="round_number_padding",
                    severity="warn",
                    message="all line items are exact multiples of 100",
                )
            ]
        ),
        RiskAssessment: RiskAssessment(
            severity_summary="medium — round-number anomaly",
            root_cause_hypothesis="Possible data entry shortcut.",
            recommended_action="request_clarification",
            items_to_verify=["Confirm pricing with vendor"],
        ),
        Justification: Justification(text="Routed to investigator due to fraud screen."),
        Critique: Critique(action="confirm", rationale="ok"),
    }

    def _fake_extract(schema, *, system, user, max_retries=2):
        return canned[schema], 0

    def _fake_tool_agent(schema, *, system, user, tools, fallback, max_tool_loops=4):
        return canned[schema], None, []

    import galatiq.agents._llm_helpers as helpers
    monkeypatch.setattr(helpers, "extract_structured", _fake_extract)
    monkeypatch.setattr(helpers, "run_tool_using_agent", _fake_tool_agent)

    # Otherwise-clean invoice. Without fraud findings → pass → no investigator.
    inv = _write(
        tmp_path / "fraud.json",
        {
            "invoice_number": "INV-FRAUD",
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
    assert any(f.code == "round_number_padding" for f in state["report"].findings)
    assert state["report"].verdict == "needs_review"
    assert state["risk_assessment"] is not None
    assert state["decision"].status == "pending_human"
    assert state["payment"].status == "skipped"


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
    assert state["fraud_findings"] == []
    assert state["payment"].status == "scheduled"
    # Justifier ran in fallback mode — its output is the canned decision.justification.
    assert state["llm_justification"].text == state["decision"].justification
    # No errors reported because the disabled branch is intentional, not a failure.
    assert state.get("llm_agent_errors") == []
