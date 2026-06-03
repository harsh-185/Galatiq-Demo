"""Golden-trajectory dataclasses.

Each ``GoldenTrajectory`` captures the per-node expected behaviour AND the
end-to-end trajectory expectation for one invoice. The dataset (in
``golden_dataset.py``) is the source of truth that every per-node and
trajectory eval script slices into.

Shape rationale: each per-node nested dataclass is independent so scorers can
load just the slice they need, and unrelated changes in one node's golden
don't ripple. All numeric expectations are strings (Decimal-as-string) so
JSON snapshots round-trip without floating-point drift.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal

Verdict = Literal["pass", "needs_review", "reject"]
Status = Literal["auto_approved", "pending_human", "rejected"]
PayStatus = Literal["scheduled", "skipped", "failed"]
Rail = Literal["ach", "wire", "check", "none"]
RiskSeverity = Literal["none", "low", "medium", "high"]
CouncilProfileName = Literal["lite", "standard", "deep", "deepest"]
Category = Literal[
    "happy_path",
    "stock_issue",
    "math_error",
    "fraud",
    "unknown_sku",
    "fx",
    "data_integrity",
    "revision",
    "cfo_tier",
    "injection",
    "duplicate",
    "near_duplicate",
]


@dataclass
class IngestGold:
    expected_vendor: str
    expected_total: str  # Decimal-as-string
    expected_currency: str
    expected_line_item_count: int
    expected_path_taken: Literal["deterministic", "llm"]
    # phrases that, if found in the extracted Invoice JSON, indicate hallucination
    must_not_hallucinate_fields: list[str] = field(default_factory=list)
    # ingestion warnings (set by Invoice model_validator) that MUST appear
    expected_ingestion_warning_codes: list[str] = field(default_factory=list)


@dataclass
class ScreenerGold:
    expected_risk_severity: RiskSeverity
    # codes the screener MUST NOT emit (e.g. dropped round_number_padding)
    forbidden_finding_codes: list[str] = field(
        default_factory=lambda: ["round_number_padding"]
    )
    # phrases that, if present in any finding's message, fail the screener
    # (these correspond to the post-filter hedge-phrase guardrail)
    forbidden_message_phrases: list[str] = field(
        default_factory=lambda: [
            "unable to verify",
            "missing data",
            "cannot assess",
            "insufficient information",
        ]
    )
    max_findings: int = 4


@dataclass
class ValidateGold:
    expected_verdict: Verdict
    must_include_codes: list[str] = field(default_factory=list)
    must_not_include_codes: list[str] = field(default_factory=list)


@dataclass
class ApproveGold:
    expected_status: Status
    expected_approver_role: str | None
    expected_policy_id: str | None  # "TIER-AUTO" | "TIER-MGR" | "TIER-DIR" | "TIER-CFO" | None
    expected_total_usd: str
    usd_tolerance: str = "0.01"


@dataclass
class CouncilGold:
    # None ⇒ council was skipped (hard reject)
    expected_profile: CouncilProfileName | None
    expected_reviewer_verdicts_must_include: list[str] = field(default_factory=list)


@dataclass
class AggregatorGold:
    expected_final_status: Status
    narrative_must_cite_terms: list[str] = field(default_factory=list)
    # negative anchors: terms the narrative MUST NOT contain (e.g. dropped heuristic)
    narrative_must_not_cite_terms: list[str] = field(
        default_factory=lambda: ["round_number_padding"]
    )


@dataclass
class PaymentGoldGuard:
    expected_approved: bool
    max_blockers: int = 0


@dataclass
class PaymentGold:
    expected_status: PayStatus
    expected_rail: Rail
    expect_receipt_written: bool


@dataclass
class TrajectoryGold:
    must_visit_stages: list[str]
    must_skip_stages: list[str] = field(default_factory=list)
    # named gates the trajectory MUST satisfy (must-pass binary gates):
    #   no_double_pay, idempotency, cfo_human_gate, injection_resistance,
    #   near_dup_citation
    must_pass_gates: list[str] = field(default_factory=list)


@dataclass
class GoldenTrajectory:
    id: str
    file_path: str
    format: Literal["txt", "pdf", "json", "csv", "xml"]
    category: Category
    ingest: IngestGold
    screener: ScreenerGold
    validate: ValidateGold
    approve: ApproveGold
    council: CouncilGold
    aggregator: AggregatorGold
    payment_guards: PaymentGoldGuard
    pay: PaymentGold
    trajectory: TrajectoryGold

    def to_dict(self) -> dict:
        return asdict(self)


def load_goldens(only: list[str] | None = None) -> list[GoldenTrajectory]:
    """Load the canonical golden set. Optional ``only`` filters by id."""
    from evals.golden_dataset import GOLDENS

    if only:
        wanted = set(only)
        return [g for g in GOLDENS if g.id in wanted]
    return list(GOLDENS)
