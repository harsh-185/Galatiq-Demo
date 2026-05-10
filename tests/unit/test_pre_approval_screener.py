"""Tests for the merged pre_approval_screener (replaces fraud_screener,
investigator, and vendor_onboarding).
"""
from __future__ import annotations

import pytest

from galatiq.agents import pre_approval_screener
from galatiq.agents.pre_approval_screener import (
    PreApprovalSummary,
    VendorProfile,
    _ScreenedFinding,
)
from galatiq.db import connect, init_db
from galatiq.models.invoice import Invoice


def _invoice(**overrides):
    base = {
        "invoice_number": "INV-PA",
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
    p = tmp_path / "pa.db"
    init_db(p)
    with connect(p) as conn:
        yield conn


@pytest.fixture(autouse=True)
def enable_llm(monkeypatch):
    monkeypatch.setenv("GALATIQ_LLM_AGENTS", "1")


def _stub(monkeypatch, summary: PreApprovalSummary, *, trace=None):
    def _fake(schema, *, system, user, tools, fallback, max_tool_loops=4):
        return summary, None, list(trace or [])

    import galatiq.agents._llm_helpers as helpers
    monkeypatch.setattr(helpers, "run_tool_using_agent", _fake)


def _stub_raises(monkeypatch, exc):
    def _fake(schema, *, system, user, tools, fallback, max_tool_loops=4):
        return fallback(), f"{type(exc).__name__}: {exc}", []

    import galatiq.agents._llm_helpers as helpers
    monkeypatch.setattr(helpers, "run_tool_using_agent", _fake)


def test_screener_emits_fraud_findings(monkeypatch, db_conn):
    _stub(
        monkeypatch,
        PreApprovalSummary(
            fraud_findings=[
                _ScreenedFinding(
                    code="vendor_typosquat",
                    severity="warn",
                    message="vendor 'Acrne Corp' resembles 'Acme Corp'",
                    field="vendor",
                )
            ],
            risk_severity="medium",
            risk_hypothesis="Likely impersonation attempt.",
        ),
        trace=["lookup_vendor(name='Acrne Corp')", "list_known_vendors()"],
    )
    summary, findings, err, trace = pre_approval_screener.screen(_invoice(), conn=db_conn)
    assert err is None
    assert len(findings) == 1
    assert findings[0].code == "vendor_typosquat"
    assert summary.risk_severity == "medium"
    assert "lookup_vendor" in trace[0]


def test_screener_emits_items_to_verify_for_unknown_vendor(monkeypatch, db_conn):
    _stub(
        monkeypatch,
        PreApprovalSummary(
            risk_severity="low",
            risk_hypothesis="Unknown vendor; needs verification.",
            items_to_verify=[
                "Confirm vendor identity with procurement",
                "Cross-check with PO records",
            ],
        ),
    )
    summary, findings, err, _ = pre_approval_screener.screen(
        _invoice(vendor="Mystery Vendor"), conn=db_conn
    )
    assert findings == []
    assert summary.items_to_verify == [
        "Confirm vendor identity with procurement",
        "Cross-check with PO records",
    ]


def test_screener_drafts_vendor_profile_for_new_vendor(monkeypatch, db_conn):
    _stub(
        monkeypatch,
        PreApprovalSummary(
            risk_severity="low",
            vendor_profile=VendorProfile(
                suggested_aliases=["NewCo Inc"],
                normalized_address="789 Newcomer Rd",
                default_currency_guess="USD",
                recommendation="approve_onboarding",
                rationale="Address matches DB record.",
            ),
        ),
    )
    summary, _, _, _ = pre_approval_screener.screen(_invoice(vendor="NewCo"), conn=db_conn)
    assert summary.vendor_profile is not None
    assert summary.vendor_profile.recommendation == "approve_onboarding"


def test_screener_falls_back_on_llm_error(monkeypatch, db_conn):
    _stub_raises(monkeypatch, RuntimeError("boom"))
    summary, findings, err, trace = pre_approval_screener.screen(_invoice(), conn=db_conn)
    assert findings == []
    assert summary.risk_severity == "none"
    assert summary.vendor_profile is None
    assert err is not None
    assert "boom" in err


def test_screener_skipped_when_disabled(monkeypatch, db_conn):
    monkeypatch.setenv("GALATIQ_LLM_AGENTS", "0")
    summary, findings, err, trace = pre_approval_screener.screen(_invoice(), conn=db_conn)
    assert findings == []
    assert summary.risk_severity == "none"
    assert err is None
    assert trace == []


def test_screener_drops_disallowed_finding_codes(monkeypatch, db_conn):
    """Regression: even if the LLM ignores the prompt and emits a disallowed
    code like 'round_number_padding', the post-filter must drop it before it
    can influence validation or aggregator decisions."""
    _stub(
        monkeypatch,
        PreApprovalSummary(
            fraud_findings=[
                _ScreenedFinding(
                    code="round_number_padding",
                    severity="info",
                    message="Total $5000 is round.",
                ),
                _ScreenedFinding(
                    code="vendor_typosquat",
                    severity="warn",
                    message="Acrne Corp vs Acme Corp (1-edit).",
                ),
            ],
            risk_severity="low",
        ),
    )
    summary, findings, _, _ = pre_approval_screener.screen(_invoice(), conn=db_conn)
    codes = {f.code for f in findings}
    assert "round_number_padding" not in codes, "disallowed code must be filtered out"
    assert "vendor_typosquat" in codes
    assert {f.code for f in summary.fraud_findings} == {"vendor_typosquat"}


def test_screener_drops_hedged_findings(monkeypatch, db_conn):
    """Regression: even if the LLM emits an allowed code like
    'category_mismatch', the post-filter must drop findings whose message
    admits non-verification (e.g. 'Unable to verify… missing catalog data').
    Those polluted the audit log on INV-1010 and re-introduced the
    false-positive-downgrade class for clean invoices."""
    _stub(
        monkeypatch,
        PreApprovalSummary(
            fraud_findings=[
                _ScreenedFinding(
                    code="category_mismatch",
                    severity="warn",
                    message="Unable to verify category for WidgetA due to missing catalog data.",
                ),
                _ScreenedFinding(
                    code="category_mismatch",
                    severity="warn",
                    message="Cannot assess amount; no history for vendor.",
                ),
                _ScreenedFinding(
                    code="vendor_typosquat",
                    severity="warn",
                    message="Acme Corp vs Acrne Corp differs by 1 edit, shares 'Acme' brand keyword.",
                ),
            ],
            risk_severity="low",
        ),
    )
    summary, findings, _, _ = pre_approval_screener.screen(_invoice(), conn=db_conn)
    codes = [f.code for f in findings]
    assert codes == ["vendor_typosquat"], "hedged findings must be filtered"
    assert {f.code for f in summary.fraud_findings} == {"vendor_typosquat"}


def test_is_hedged_phrase_matching():
    from galatiq.agents.pre_approval_screener import _is_hedged
    assert _is_hedged("Unable to verify the SKU.") is True
    assert _is_hedged("Missing catalog data for WidgetA.") is True
    assert _is_hedged("No history for vendor X to compare against.") is True
    assert _is_hedged("Vendor name 'Acrne Corp' is 1 edit away from 'Acme Corp'.") is False
    assert _is_hedged("Total $5000 exceeds threshold") is False
    assert _is_hedged("") is False
