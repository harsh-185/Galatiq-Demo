"""Justifier: LLM agent that drafts the human-readable approval narrative.

Replaces the canned ``decision.justification`` string with a 2-3 sentence
explanation that cites specific findings and the policy band. Advisory only —
the routing has already been decided by the deterministic approval agent.
"""
from __future__ import annotations

import json

from pydantic import BaseModel, Field

from galatiq.agents._llm_helpers import run_llm_agent
from galatiq.agents.approval import ApprovalDecision
from galatiq.agents.validation import ValidationReport
from galatiq.models.invoice import Invoice

_SYSTEM = """\
You write the audit trail for an approved/rejected invoice. The decision has
already been made by a deterministic policy engine — your job is to explain it
in 2-3 sentences for the audit log.

Inputs:
- decision: status (auto_approved | pending_human | rejected), approver_role,
  policy_id, total_usd
- validation findings (codes, severities, messages)
- the invoice itself

Rules:
- Cite at least one specific finding code or policy_id when relevant.
- Do not propose a different decision; you are documenting, not deciding.
- Plain prose. No bullet points. 2-3 sentences total.
- If the decision is auto_approved with no findings, write a single sentence.
"""


class Justification(BaseModel):
    text: str = Field(description="2-3 sentence audit narrative.")


def _fallback(decision: ApprovalDecision) -> Justification:
    return Justification(text=decision.justification)


def write_justification(
    invoice: Invoice,
    report: ValidationReport,
    decision: ApprovalDecision,
) -> tuple[Justification, str | None]:
    payload = {
        "decision": {
            "status": decision.status,
            "approver_role": decision.approver_role,
            "policy_id": decision.policy_id,
            "total_usd": str(decision.total_usd),
            "deterministic_justification": decision.justification,
            "escalations": decision.escalations,
        },
        "verdict": report.verdict,
        "findings": [
            {"code": f.code, "severity": f.severity, "message": f.message}
            for f in report.findings
        ],
        "invoice_excerpt": {
            "invoice_number": invoice.invoice_number,
            "vendor": invoice.vendor,
            "total": str(invoice.total),
            "currency": invoice.currency,
            "line_item_count": len(invoice.line_items),
        },
    }
    user = (
        "Write the audit narrative for this decision.\n\n"
        f"```json\n{json.dumps(payload, indent=2, default=str)}\n```"
    )
    return run_llm_agent(
        Justification,
        system=_SYSTEM,
        user=user,
        fallback=lambda: _fallback(decision),
    )
