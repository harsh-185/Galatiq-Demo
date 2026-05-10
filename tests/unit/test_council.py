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
from galatiq.agents.validation import Finding, ValidationReport
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
    # Tightened budget for speed (was 6, now 4); deepest still has the
    # highest tool budget in the system.
    assert cfo.max_tool_loops_per_reviewer >= 4
    assert cfo.max_tool_loops_per_reviewer >= select_profile("TIER-DIR").max_tool_loops_per_reviewer


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


def test_aggregator_can_auto_approve_tier_mgr_with_unanimous_clean_council(monkeypatch):
    """TIER-MGR ($10k-$50k) + clean verdict + unanimous-clean council can
    auto-approve. The AI council handles mid-tier decisions; humans only
    needed for genuinely ambiguous cases or TIER-CFO."""
    _stub_aggregator(
        monkeypatch,
        AggregatedDecision(
            final_status="auto_approved",
            final_approver_role="system",
            final_policy_id="TIER-MGR",
            audit_narrative="$25k clean Acme order; council unanimous; auto-approve.",
        ),
    )
    pending = _decision(status="pending_human", role="manager", policy="TIER-MGR", total=Decimal("25000"))
    report = ValidationReport(findings=[], verdict="pass")
    opinions = [
        ReviewerOpinion(reviewer="compliance", verdict="approve", severity="low", rationale="ok"),
        ReviewerOpinion(reviewer="fraud", verdict="approve", severity="low", rationale="ok"),
        ReviewerOpinion(reviewer="policy", verdict="approve", severity="low", rationale="ok"),
    ]
    final, _, _ = aggregate(_invoice(), report, pending, opinions)
    assert final.status == "auto_approved"
    assert final.policy_id == "TIER-MGR"


def test_aggregator_cannot_auto_approve_tier_cfo_even_when_clean(monkeypatch):
    """TIER-CFO ($200k+) ALWAYS requires human sign-off (fiduciary policy),
    even when the council is unanimous-clean."""
    _stub_aggregator(
        monkeypatch,
        AggregatedDecision(
            final_status="auto_approved",
            final_approver_role="system",
            final_policy_id="TIER-CFO",
            audit_narrative="trying to bypass CFO sign-off",
        ),
    )
    pending = _decision(status="pending_human", role="cfo", policy="TIER-CFO", total=Decimal("250000"))
    report = ValidationReport(findings=[], verdict="pass")
    opinions = [
        ReviewerOpinion(reviewer="compliance", verdict="approve", severity="low", rationale="ok"),
        ReviewerOpinion(reviewer="fraud", verdict="approve", severity="low", rationale="ok"),
        ReviewerOpinion(reviewer="policy", verdict="approve", severity="low", rationale="ok"),
    ]
    final, _, _ = aggregate(_invoice(), report, pending, opinions)
    # Safety net engages — CFO tier always requires human.
    assert final.status == "pending_human"
    assert final.approver_role == "cfo"
    assert "TIER-CFO" in final.justification


def test_aggregator_unanimous_clean_can_auto_resolve_needs_review(monkeypatch):
    """When all reviewers vote 'approve' with severity='low' and no load-
    bearing finding is present, the aggregator may flip needs_review →
    auto_approved. This is the relaxation that empowers the council to
    decide instead of always deferring to a human."""
    _stub_aggregator(
        monkeypatch,
        AggregatedDecision(
            final_status="auto_approved",
            final_approver_role="system",
            final_policy_id="TIER-AUTO",
            audit_narrative="Council unanimous-clean; warnings benign.",
        ),
    )
    pending = _decision(status="pending_human", role="manager", policy="TIER-MGR")
    # vendor_unknown is NOT in the load-bearing list, so unanimous-clean wins.
    report = ValidationReport(
        findings=[Finding(code="vendor_unknown", severity="warn", message="not in table")],
        verdict="needs_review",
    )
    opinions = [
        ReviewerOpinion(reviewer="compliance", verdict="approve", severity="low", rationale="benign"),
        ReviewerOpinion(reviewer="fraud", verdict="approve", severity="low", rationale="ok"),
        ReviewerOpinion(reviewer="policy", verdict="approve", severity="low", rationale="ok"),
    ]
    final, _, _ = aggregate(_invoice(), report, pending, opinions)
    assert final.status == "auto_approved"


def test_aggregator_load_bearing_finding_blocks_auto_resolve(monkeypatch):
    """Even with unanimous-clean reviewers, a load-bearing finding (e.g.
    duplicate_invoice) still forces pending_human."""
    _stub_aggregator(
        monkeypatch,
        AggregatedDecision(
            final_status="auto_approved",
            final_approver_role="system",
            final_policy_id="TIER-AUTO",
            audit_narrative="trying to bypass",
        ),
    )
    pending = _decision(status="pending_human", role="manager", policy="TIER-MGR")
    report = ValidationReport(
        findings=[Finding(code="duplicate_invoice", severity="error", message="dup")],
        verdict="needs_review",
    )
    opinions = [
        ReviewerOpinion(reviewer="compliance", verdict="approve", severity="low", rationale="ok"),
        ReviewerOpinion(reviewer="fraud", verdict="approve", severity="low", rationale="ok"),
        ReviewerOpinion(reviewer="policy", verdict="approve", severity="low", rationale="ok"),
    ]
    final, _, _ = aggregate(_invoice(), report, pending, opinions)
    # Safety net engaged because duplicate_invoice is load-bearing.
    assert final.status == "pending_human"


def test_aggregator_non_unanimous_blocks_auto_resolve(monkeypatch):
    """A single non-approve reviewer prevents the relaxation."""
    _stub_aggregator(
        monkeypatch,
        AggregatedDecision(
            final_status="auto_approved",
            final_approver_role="system",
            final_policy_id="TIER-AUTO",
            audit_narrative="trying to bypass",
        ),
    )
    pending = _decision(status="pending_human", role="manager", policy="TIER-MGR")
    report = ValidationReport(
        findings=[Finding(code="vendor_unknown", severity="warn", message="not in table")],
        verdict="needs_review",
    )
    opinions = [
        ReviewerOpinion(reviewer="compliance", verdict="approve", severity="low", rationale="ok"),
        ReviewerOpinion(reviewer="fraud", verdict="approve_with_notes", severity="medium", rationale="some concern"),
        ReviewerOpinion(reviewer="policy", verdict="approve", severity="low", rationale="ok"),
    ]
    final, _, _ = aggregate(_invoice(), report, pending, opinions)
    assert final.status == "pending_human"


def test_aggregator_safety_override_keeps_engine_reject(monkeypatch):
    """If the LLM tries to override a rule-based reject, the safety net wins.

    Uses non-empty opinions so the LLM path is actually exercised (the
    skip-LLM-on-empty-opinions short-circuit doesn't apply here).
    """
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
    opinions = [
        ReviewerOpinion(reviewer="fraud", verdict="approve", severity="low", rationale="ok"),
    ]
    final, narrative, err = aggregate(_invoice(), ValidationReport(verdict="reject"), rejected, opinions)
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
    """When the LLM is unavailable AND there are reviewer opinions to
    synthesize, the aggregator falls back to the deterministic path."""
    def _fake(schema, *, system, user, max_retries=2):
        raise RuntimeError("no api key")

    import galatiq.agents._llm_helpers as helpers
    monkeypatch.setattr(helpers, "extract_structured", _fake)
    opinions = [
        ReviewerOpinion(reviewer="fraud", verdict="approve", severity="low", rationale="ok"),
    ]
    final, narrative, err = aggregate(_invoice(), ValidationReport(verdict="pass"), _decision(), opinions)
    assert err is not None
    assert final.status == "auto_approved"  # deterministic aggregate keeps engine outcome
    assert narrative


def test_aggregator_runs_llm_for_narrative_even_when_no_opinions(monkeypatch):
    """No opinions = council was skipped, but the aggregator still produces an
    LLM-written audit narrative (with a concise no-council prompt). The audit
    trail always has an LLM-written justification.
    """
    _stub_aggregator(
        monkeypatch,
        AggregatedDecision(
            final_status="auto_approved",
            final_approver_role="system",
            final_policy_id="TIER-AUTO",
            audit_narrative="Engine approved under TIER-AUTO; clean invoice, no findings.",
        ),
    )
    final, narrative, err = aggregate(_invoice(), ValidationReport(verdict="pass"), _decision(), [])
    assert err is None
    assert final.status == "auto_approved"
    assert "TIER-AUTO" in narrative


# --- end-to-end with full council in fallback mode ---------------------------


def test_pipeline_runs_council_on_clean_small_invoice(tmp_path, monkeypatch):
    """LLM is always part of the approval loop. Even a clean small invoice
    runs the council (lite profile, 1 reviewer)."""
    monkeypatch.setenv("GALATIQ_LLM_AGENTS", "0")  # use deterministic fallbacks
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
    # Council ran — lite profile (1 reviewer: fraud).
    assert state.get("council_skipped") is False
    assert state["council_profile"].name == "lite"
    assert [o.reviewer for o in state["reviewer_opinions"]] == ["fraud"]
    assert state["decision"].status == "auto_approved"
    # Audit log got the FINAL decision via the aggregator.
    with connect(db) as conn:
        rows = list_approvals(conn)
    assert len(rows) == 1
    assert rows[0]["status"] == "auto_approved"


def test_pipeline_parallel_council_preserves_canonical_reviewer_order(tmp_path, monkeypatch):
    """The reviewers run concurrently; output ``reviewer_opinions`` must still
    be in the canonical profile order regardless of which thread finishes
    first."""
    monkeypatch.setenv("GALATIQ_LLM_AGENTS", "0")
    db = tmp_path / "parallel.db"
    init_db(db)
    inv_path = tmp_path / "midsize.json"
    inv_path.write_text(json.dumps({
        "invoice_number": "INV-PAR",
        "vendor": "Acme Corp",
        "date": "2026-01-01",
        "due_date": "2099-01-01",
        "currency": "USD",
        "line_items": [{"item": "LaserCutterPro", "quantity": 1, "unit_price": "25000.00"}],
        "subtotal": "25000.00",
        "tax": "0.00",
        "total": "25000.00",
    }))
    state = run_pipeline(inv_path, db_path=db, receipt_dir=tmp_path / "r")
    # Standard profile → 3 reviewers in canonical order.
    assert state["council_profile"].name == "standard"
    actual_order = [o.reviewer for o in state["reviewer_opinions"]]
    assert actual_order == ["compliance", "fraud", "policy"]


def test_pipeline_runs_full_council_for_manager_tier(tmp_path, monkeypatch):
    """$25k invoice → TIER-MGR → 'standard' profile → all 3 reviewers run."""
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
        "line_items": [{"item": "LaserCutterPro", "quantity": 1, "unit_price": "25000.00"}],
        "subtotal": "25000.00",
        "tax": "0.00",
        "total": "25000.00",
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
