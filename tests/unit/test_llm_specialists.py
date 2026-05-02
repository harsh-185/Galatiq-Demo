"""Unit tests for the four LLM specialist agents.

Each test stubs ``extract_structured`` so the suite stays offline. We verify both
the success path (LLM returns a typed result) and the fallback path (LLM raises;
agent returns its deterministic fallback without crashing the pipeline).
"""
from __future__ import annotations

import os
from decimal import Decimal

import pytest

from galatiq.agents import fraud_screener, investigator, justifier, vendor_onboarding
from galatiq.agents.approval import ApprovalDecision
from galatiq.agents.fraud_screener import FraudScreenResult, _ScreenedFinding
from galatiq.agents.investigator import RiskAssessment
from galatiq.agents.justifier import Justification
from galatiq.agents.validation import Finding, ValidationReport, validate
from galatiq.agents.vendor_onboarding import VendorProfile
from galatiq.db import STATUS_NEW, Vendor, connect, init_db
from galatiq.models.invoice import Invoice


def _invoice(**overrides):
    base = {
        "invoice_number": "INV-LLM",
        "vendor": "Acme Corp",
        "date": "2026-01-01",
        "due_date": "2026-02-01",
        "currency": "USD",
        "line_items": [{"item": "WidgetA", "quantity": 2, "unit_price": "10.00"}],
        "subtotal": "20.00",
        "tax": "0.00",
        "total": "20.00",
    }
    base.update(overrides)
    return Invoice.model_validate(base)


@pytest.fixture
def db_conn(tmp_path):
    path = tmp_path / "llm.db"
    init_db(path)
    with connect(path) as conn:
        yield conn


@pytest.fixture(autouse=True)
def enable_llm_agents(monkeypatch):
    """Force the toggle ON so tests exercise the LLM path; the stub keeps it offline."""
    monkeypatch.setenv("GALATIQ_LLM_AGENTS", "1")
    yield
    monkeypatch.delenv("GALATIQ_LLM_AGENTS", raising=False)


def _stub_extract(monkeypatch, target_module, schema, instance):
    """Stub the simple structured-output path used by vendor_onboarding/justifier."""
    def _fake(schema_cls, *, system, user, max_retries=2):
        assert schema_cls is schema
        return instance, 0

    import galatiq.agents._llm_helpers as helpers
    monkeypatch.setattr(helpers, "extract_structured", _fake)


def _stub_extract_raises(monkeypatch, exc):
    def _fake(schema_cls, *, system, user, max_retries=2):
        raise exc

    import galatiq.agents._llm_helpers as helpers
    monkeypatch.setattr(helpers, "extract_structured", _fake)


def _stub_tool_agent(monkeypatch, instance, *, trace=None):
    """Stub the tool-using agent path used by fraud_screener/investigator."""
    def _fake(schema, *, system, user, tools, fallback, max_tool_loops=4):
        return instance, None, list(trace or [])

    import galatiq.agents._llm_helpers as helpers
    monkeypatch.setattr(helpers, "run_tool_using_agent", _fake)


def _stub_tool_agent_raises(monkeypatch, exc):
    def _fake(schema, *, system, user, tools, fallback, max_tool_loops=4):
        # Mirror the helper's failure mode: return fallback + error string.
        return fallback(), f"{type(exc).__name__}: {exc}", []

    import galatiq.agents._llm_helpers as helpers
    monkeypatch.setattr(helpers, "run_tool_using_agent", _fake)


# --- fraud_screener ----------------------------------------------------------


def test_fraud_screener_returns_findings(monkeypatch, db_conn):
    _stub_tool_agent(
        monkeypatch,
        FraudScreenResult(
            findings=[
                _ScreenedFinding(
                    code="vendor_typosquat",
                    severity="warn",
                    message="vendor name 'Acrne Corp' resembles 'Acme Corp'",
                    field="vendor",
                )
            ]
        ),
        trace=["lookup_vendor(name='Acrne Corp')", "list_known_vendors()"],
    )
    findings, err, trace = fraud_screener.screen(_invoice(), conn=db_conn)
    assert err is None
    assert len(findings) == 1
    assert findings[0].code == "vendor_typosquat"
    assert findings[0].severity == "warn"
    assert "lookup_vendor" in trace[0]


def test_fraud_screener_falls_back_on_llm_error(monkeypatch, db_conn):
    _stub_tool_agent_raises(monkeypatch, RuntimeError("boom"))
    findings, err, trace = fraud_screener.screen(_invoice(), conn=db_conn)
    assert findings == []
    assert err is not None
    assert "boom" in err
    assert trace == []


def test_fraud_screener_skipped_when_disabled(monkeypatch, db_conn):
    monkeypatch.setenv("GALATIQ_LLM_AGENTS", "0")
    findings, err, trace = fraud_screener.screen(_invoice(), conn=db_conn)
    assert findings == []
    assert err is None
    assert trace == []


# --- vendor_onboarding -------------------------------------------------------


def test_vendor_onboarding_returns_profile(monkeypatch):
    profile = VendorProfile(
        suggested_aliases=["NewCo Inc"],
        normalized_address="789 Newcomer Rd",
        default_currency_guess="USD",
        recommendation="approve_onboarding",
        rationale="Address matches DB record; invoice is consistent.",
    )
    _stub_extract(monkeypatch, vendor_onboarding, VendorProfile, profile)
    vendor = Vendor(
        vendor_id="VEND-004",
        name="NewCo",
        aliases=[],
        address="789 Newcomer Rd",
        status=STATUS_NEW,
        default_currency="USD",
    )
    out, err = vendor_onboarding.draft_profile(_invoice(vendor="NewCo"), vendor)
    assert err is None
    assert out.recommendation == "approve_onboarding"


def test_vendor_onboarding_fallback_preserves_invoice_data(monkeypatch):
    _stub_extract_raises(monkeypatch, RuntimeError("api down"))
    vendor = Vendor(
        vendor_id="VEND-004",
        name="NewCo",
        aliases=[],
        address="789 Newcomer Rd",
        status=STATUS_NEW,
        default_currency="USD",
    )
    out, err = vendor_onboarding.draft_profile(_invoice(vendor="NewCo"), vendor)
    assert err is not None
    assert out.recommendation == "needs_more_info"
    assert out.default_currency_guess == "USD"


# --- investigator ------------------------------------------------------------


def test_investigator_returns_assessment(monkeypatch, db_conn):
    assessment = RiskAssessment(
        severity_summary="medium — 2 findings centered on vendor mismatch",
        root_cause_hypothesis="Vendor name was entered in shorthand by AP team.",
        recommended_action="request_clarification",
        items_to_verify=["Confirm canonical vendor name", "Cross-check PO number"],
    )
    _stub_tool_agent(
        monkeypatch,
        assessment,
        trace=["lookup_vendor(name='Mystery Vendor')"],
    )
    inv = _invoice(vendor="Mystery Vendor")
    report = validate(inv, conn=db_conn)
    out, err, trace = investigator.assess(inv, report, conn=db_conn)
    assert err is None
    assert out.recommended_action == "request_clarification"
    assert any("lookup_vendor" in t for t in trace)


def test_investigator_fallback_summarizes_findings(monkeypatch, db_conn):
    _stub_tool_agent_raises(monkeypatch, RuntimeError("nope"))
    inv = _invoice(vendor="Mystery Vendor")
    report = validate(inv, conn=db_conn)
    out, err, trace = investigator.assess(inv, report, conn=db_conn)
    assert err is not None
    # Fallback echoes the validation finding codes back as items_to_verify.
    assert any("vendor_unknown" in s for s in out.items_to_verify)
    assert trace == []


# --- justifier ---------------------------------------------------------------


def test_justifier_returns_narrative(monkeypatch):
    _stub_extract(
        monkeypatch,
        justifier,
        Justification,
        Justification(text="Auto-approved under TIER-AUTO; no findings."),
    )
    decision = ApprovalDecision(
        status="auto_approved",
        approver_role="system",
        policy_id="TIER-AUTO",
        total_usd=Decimal("20.00"),
        justification="auto-approved under TIER-AUTO (total_usd=20.00 < 1000)",
        escalations=[],
    )
    report = ValidationReport(findings=[], verdict="pass")
    out, err = justifier.write_justification(_invoice(), report, decision)
    assert err is None
    assert "Auto-approved" in out.text


def test_justifier_falls_back_to_canned_text(monkeypatch):
    _stub_extract_raises(monkeypatch, RuntimeError("rate limit"))
    decision = ApprovalDecision(
        status="rejected",
        approver_role="none",
        policy_id=None,
        total_usd=Decimal("9.99"),
        justification="invoice 'INV-LLM' from 'Acme Corp' already in ledger",
        escalations=["duplicate_invoice"],
    )
    report = ValidationReport(
        findings=[Finding(code="duplicate_invoice", severity="error", message="dup")],
        verdict="reject",
    )
    out, err = justifier.write_justification(_invoice(), report, decision)
    assert err is not None
    assert out.text == decision.justification
