"""Rubric-graded LLM judges.

These wrap `autoevals.LLMClassifier` with the rubrics from `evals/rubrics/`.
We deliberately pick `gpt-4o-mini` as the judge model — a different family
from Grok (which the system being evaluated uses) to dodge same-family bias.

Each judge:
- Uses chain-of-thought (`use_cot=True`) so its reasoning is logged for
  later inspection.
- Returns a Score with the verdict mapped to a 0–1 float.
- Falls back to a 0-scored Score with an `error` metadata field on any failure
  (network, schema, missing key) so the eval harness can keep running.

The `score_*` functions are the public surface used by the eval scripts.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
RUBRIC_DIR = REPO_ROOT / "rubrics"


_CHOICE_SCORES = {
    "excellent": 1.0,
    "good": 0.75,
    "fair": 0.5,
    "poor": 0.25,
    "wrong": 0.0,
}


def _read_rubric(name: str) -> str:
    return (RUBRIC_DIR / name).read_text()


def _build_classifier(rubric_name: str, judge_name: str, model: str | None = None):
    """Lazily build the judge so importing this module doesn't require an
    OpenAI key on every run."""
    try:
        from autoevals import LLMClassifier
    except ImportError as e:
        raise RuntimeError(
            "autoevals not installed; install via `pip install autoevals`"
        ) from e
    model = model or os.environ.get("JUDGE_MODEL", "gpt-4o-mini")
    return LLMClassifier(
        name=judge_name,
        prompt_template=_read_rubric(rubric_name),
        choice_scores=_CHOICE_SCORES,
        use_cot=True,
        model=model,
    )


def _safe_score(judge, name: str, **kwargs) -> dict:
    """Call a Braintrust/autoevals judge with the kwargs it expects, catching
    any error so the harness keeps running."""
    try:
        result = judge(**kwargs)
    except Exception as e:  # noqa: BLE001
        return {"name": name, "score": 0.0,
                "metadata": {"error": f"{type(e).__name__}: {e}"}}
    # autoevals returns a Score-like object with .score, .metadata
    score_val = getattr(result, "score", None)
    if score_val is None:
        return {"name": name, "score": 0.0,
                "metadata": {"error": "judge returned no score"}}
    return {
        "name": name,
        "score": float(score_val),
        "metadata": getattr(result, "metadata", None) or {},
    }


def _invoice_summary(invoice) -> dict:
    return {
        "invoice_number": invoice.invoice_number,
        "vendor": invoice.vendor,
        "total": str(invoice.total),
        "currency": invoice.currency,
        "line_items": [
            {"item": li.item, "quantity": li.quantity, "unit_price": str(li.unit_price)}
            for li in invoice.line_items
        ],
    }


def score_approval_narrative(state, golden) -> dict:
    """Score the audit narrative against the 4-criterion rubric."""
    judge = _build_classifier("approval_narrative.txt", "approval_narrative")
    invoice = state.get("ingestion").invoice
    decision = state["decision"]
    opinions = [asdict(o) for o in state.get("reviewer_opinions", [])]
    pre_decision = state.get("pre_council_decision")
    pre = asdict(pre_decision) if pre_decision else None
    return _safe_score(
        judge,
        "approval_narrative_judge",
        input={
            "invoice_summary": _invoice_summary(invoice),
            "engine_decision": pre,
            "opinions": opinions,
            "final_decision": {
                "status": decision.status,
                "approver_role": decision.approver_role,
                "policy_id": decision.policy_id,
            },
        },
        output={"narrative": state.get("audit_narrative") or ""},
        expected={"golden_status": golden.aggregator.expected_final_status},
    )


def score_validate_findings(state, golden) -> dict:
    judge = _build_classifier("validate_findings.txt", "validate_findings")
    invoice = state.get("ingestion").invoice
    report = state.get("report")
    findings = [
        {"code": f.code, "severity": f.severity, "message": f.message}
        for f in (report.findings if report else [])
    ]
    return _safe_score(
        judge,
        "validate_findings_judge",
        input={"invoice_summary": _invoice_summary(invoice)},
        output={"findings": findings},
        expected={"expected_verdict": golden.validate.expected_verdict},
    )


def score_trajectory(state, golden) -> dict:
    judge = _build_classifier("trajectory_path.txt", "trajectory_path")
    invoice = state.get("ingestion").invoice if state.get("ingestion") else None
    walkthrough = state.get("walkthrough", [])
    visited = [ev.name for ev in walkthrough if ev.status != "skipped"]
    skipped = [ev.name for ev in walkthrough if ev.status == "skipped"]
    decision = state.get("decision")
    payment = state.get("payment")
    return _safe_score(
        judge,
        "trajectory_judge",
        input={
            "invoice_summary": _invoice_summary(invoice) if invoice else {},
            "expected_outcome": {
                "status": golden.approve.expected_status,
                "pay_status": golden.pay.expected_status,
            },
            "stages_visited": visited,
            "stages_skipped": skipped,
        },
        output={
            "decision": {
                "status": getattr(decision, "status", None),
                "approver_role": getattr(decision, "approver_role", None),
                "policy_id": getattr(decision, "policy_id", None),
            } if decision else None,
            "payment": {
                "status": getattr(payment, "status", None),
                "rail": getattr(payment, "rail", None),
            } if payment else None,
            "narrative": state.get("audit_narrative") or "",
        },
        expected={"golden_id": golden.id},
    )
