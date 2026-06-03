"""25 adversarial goldens encoding the IDEAL outcome for each stress case.

Where the current system diverges from the ideal, ``known_gap`` documents the
limitation. The adversarial eval (``eval_adversarial.py``) classifies each row
as pass / known_gap / regression — only regressions fail CI. This is the xfail
pattern: documented limitations don't break the build, but undocumented
divergence does.

Outcomes were calibrated by running every case through the deterministic
pipeline (scripts/gen_adversarial.py + a probe sweep) and recording what the
system actually does, then comparing against what it SHOULD do.
"""
from __future__ import annotations

from evals.loader import (
    AggregatorGold,
    ApproveGold,
    CouncilGold,
    GoldenTrajectory,
    IngestGold,
    PaymentGold,
    PaymentGoldGuard,
    ScreenerGold,
    TrajectoryGold,
    ValidateGold,
)

_ADV = "data/adversarial"

# Stage lists keyed by ideal status (kept permissive; the adversarial eval
# scores verdict + decision + payment, not exact topology).
_VISIT = {
    "auto_approved": ["ingest", "validate", "approve", "pay"],
    "pending_human": ["ingest", "validate", "approve", "pay"],
    "rejected": ["ingest", "validate", "approve", "pay"],
}


def _adv(
    *,
    id: str,
    file: str,
    fmt: str,
    category: str,
    vendor: str,
    total: str,
    currency: str,
    line_items: int,
    path: str,
    verdict: str,
    status: str,
    role: str | None,
    policy: str | None,
    total_usd: str,
    pay_status: str,
    rail: str,
    receipt: bool,
    must_include_codes: list[str] | None = None,
    known_gap: str | None = None,
    requires_llm: bool = False,
    usd_tol: str = "0.50",
) -> GoldenTrajectory:
    return GoldenTrajectory(
        id=id,
        file_path=f"{_ADV}/{file}",
        format=fmt,  # type: ignore[arg-type]
        category=category,  # type: ignore[arg-type]
        ingest=IngestGold(
            expected_vendor=vendor,
            expected_total=total,
            expected_currency=currency,
            expected_line_item_count=line_items,
            expected_path_taken=path,  # type: ignore[arg-type]
        ),
        screener=ScreenerGold(expected_risk_severity="low"),
        validate=ValidateGold(
            expected_verdict=verdict,  # type: ignore[arg-type]
            must_include_codes=must_include_codes or [],
        ),
        approve=ApproveGold(
            expected_status=status,  # type: ignore[arg-type]
            expected_approver_role=role,
            expected_policy_id=policy,
            expected_total_usd=total_usd,
            usd_tolerance=usd_tol,
        ),
        council=CouncilGold(expected_profile=None),  # not scored for adversarial
        aggregator=AggregatorGold(expected_final_status=status),  # type: ignore[arg-type]
        payment_guards=PaymentGoldGuard(
            expected_approved=(pay_status == "scheduled"), max_blockers=10
        ),
        pay=PaymentGold(
            expected_status=pay_status,  # type: ignore[arg-type]
            expected_rail=rail,  # type: ignore[arg-type]
            expect_receipt_written=receipt,
        ),
        trajectory=TrajectoryGold(must_visit_stages=_VISIT[status]),
        known_gap=known_gap,
        requires_llm=requires_llm,
    )


ADVERSARIAL_GOLDENS: list[GoldenTrajectory] = [
    # ── Group 1 — math / tax games ──────────────────────────────────────────
    _adv(
        id="INV-2001", file="inv_2001_tax_rate_lie.json", fmt="json",
        category="math_error", vendor="Widgets Inc.", total="2950.00",
        currency="USD", line_items=1, path="deterministic",
        verdict="needs_review", status="pending_human", role="system",
        policy="TIER-AUTO", total_usd="2950.00", pay_status="skipped", rail="none",
        receipt=False,
        known_gap="No stated-tax-rate vs tax-amount check: invoice states 8% but "
                  "charges 18%; subtotal+tax is internally consistent so it "
                  "auto-approves — a silent overcharge.",
    ),
    _adv(
        id="INV-2002", file="inv_2002_hidden_discount.json", fmt="json",
        category="math_error", vendor="Precision Parts Ltd.", total="1500.00",
        currency="USD", line_items=2, path="deterministic",
        verdict="reject", status="rejected", role="none", policy=None,
        total_usd="1500.00", pay_status="skipped", rail="none", receipt=False,
        must_include_codes=["subtotal_mismatch"],
    ),
    _adv(
        id="INV-2003", file="inv_2003_rounding_drift.json", fmt="json",
        category="math_error", vendor="Acme Industrial Supplies", total="751.00",
        currency="USD", line_items=1, path="deterministic",
        verdict="pass", status="auto_approved", role="system", policy="TIER-AUTO",
        total_usd="751.00", pay_status="scheduled", rail="ach", receipt=True,
    ),
    _adv(
        id="INV-2004", file="inv_2004_credit_memo.json", fmt="json",
        category="data_integrity", vendor="Acme Industrial Supplies",
        total="-1250.00", currency="USD", line_items=1, path="deterministic",
        verdict="needs_review", status="pending_human", role="system",
        policy="TIER-AUTO", total_usd="-1250.00", pay_status="skipped",
        rail="none", receipt=False,
        known_gap="All negative quantities are hard-rejected as 'negative_quantity' "
                  "errors; a legitimate credit memo (returned goods) cannot be "
                  "processed — false positive.",
    ),
    _adv(
        id="INV-2005", file="inv_2005_zero_dollar_sample.json", fmt="json",
        category="data_integrity", vendor="Gadgets Co.", total="0.00",
        currency="USD", line_items=1, path="deterministic",
        verdict="needs_review", status="pending_human", role="system",
        policy="TIER-AUTO", total_usd="0.00", pay_status="skipped", rail="none",
        receipt=False, must_include_codes=["zero_unit_price"],
    ),
    # ── Group 2 — price / quantity blind spots ──────────────────────────────
    _adv(
        id="INV-2006", file="inv_2006_widgeta_overpriced.json", fmt="json",
        category="math_error", vendor="Widgets Inc.", total="999999.00",
        currency="USD", line_items=1, path="deterministic",
        verdict="needs_review", status="pending_human", role="cfo",
        policy="TIER-CFO", total_usd="999999.00", pay_status="skipped",
        rail="none", receipt=False,
        known_gap="Core SKUs (WidgetA/B, GadgetX) have no catalog unit_price, so "
                  "price_drift_high never fires. A $999,999 WidgetA is not flagged "
                  "at the line level — only the total size escalates it to CFO.",
    ),
    _adv(
        id="INV-2007", file="inv_2007_massive_quantity.json", fmt="json",
        category="stock_issue", vendor="Widgets Inc.", total="10000.00",
        currency="USD", line_items=1, path="deterministic",
        verdict="needs_review", status="pending_human", role="manager",
        policy="TIER-MGR", total_usd="10000.00", pay_status="skipped",
        rail="none", receipt=False, must_include_codes=["stock_overflow"],
    ),
    _adv(
        id="INV-2008", file="inv_2008_micro_price.json", fmt="json",
        category="stock_issue", vendor="Acme Industrial Supplies", total="50.00",
        currency="USD", line_items=1, path="deterministic",
        verdict="needs_review", status="pending_human", role="system",
        policy="TIER-AUTO", total_usd="50.00", pay_status="skipped", rail="none",
        receipt=False, must_include_codes=["stock_overflow"],
    ),
    _adv(
        id="INV-2009", file="inv_2009_fractional_qty.csv", fmt="csv",
        category="data_integrity", vendor="Precision Parts Ltd.", total="625.00",
        currency="USD", line_items=1, path="llm",
        verdict="needs_review", status="pending_human", role="system",
        policy="TIER-AUTO", total_usd="625.00", pay_status="skipped", rail="none",
        receipt=False, requires_llm=True,
        known_gap="Deterministic CSV reader calls int('2.5') which raises "
                  "ValueError and propagates as a hard ingestion crash instead of "
                  "returning None to trigger the LLM fallback.",
    ),
    # ── Group 3 — duplicate / fraud evasion ─────────────────────────────────
    _adv(
        id="INV-2010", file="inv_2010_resubmit_new_number.json", fmt="json",
        category="duplicate", vendor="Precision Parts Ltd.", total="1890.00",
        currency="USD", line_items=2, path="deterministic",
        verdict="needs_review", status="pending_human", role="system",
        policy="TIER-AUTO", total_usd="1890.00", pay_status="skipped",
        rail="none", receipt=False,
        known_gap="Dedup is exact (invoice_number, vendor). A resubmit of an "
                  "already-paid invoice under a NEW number is not caught; only the "
                  "LLM near-dup check (with ledger history) could flag it.",
    ),
    _adv(
        id="INV-2011", file="inv_2011_dup_vendor_punct.json", fmt="json",
        category="duplicate", vendor="Precision Parts Ltd", total="1890.00",
        currency="USD", line_items=2, path="deterministic",
        verdict="reject", status="rejected", role="none", policy=None,
        total_usd="1890.00", pay_status="skipped", rail="none", receipt=False,
        known_gap="Dedup keys on the exact vendor string; 'Precision Parts Ltd' "
                  "(no period) doesn't match the ledger's 'Precision Parts Ltd.', "
                  "so duplicate_invoice is missed — caught only incidentally as "
                  "vendor_unknown.",
    ),
    _adv(
        id="INV-2012", file="inv_2012_structured_under_threshold.json", fmt="json",
        category="fraud", vendor="Global Supply Chain Partners", total="48000.00",
        currency="USD", line_items=1, path="deterministic",
        verdict="pass", status="pending_human", role="manager", policy="TIER-MGR",
        total_usd="48000.00", pay_status="skipped", rail="none", receipt=False,
        known_gap="No cross-invoice aggregation / structuring detection: a "
                  "purchase deliberately split to sit just under the $50k director "
                  "gate routes to manager. (Single-invoice outcome is acceptable; "
                  "the systemic limitation is documented.)",
    ),
    # ── Group 4 — vendor identity ───────────────────────────────────────────
    _adv(
        id="INV-2013", file="inv_2013_homoglyph_vendor.json", fmt="json",
        category="fraud", vendor="Аcme Corp", total="500.00", currency="USD",
        line_items=1, path="deterministic",
        verdict="reject", status="rejected", role="none", policy=None,
        total_usd="500.00", pay_status="skipped", rail="none", receipt=False,
        known_gap="Cyrillic-А homoglyph in the vendor name evades alias matching "
                  "→ vendor_unknown (warn) rather than a fraud-grade block.",
    ),
    _adv(
        id="INV-2014", file="inv_2014_typosquat.json", fmt="json",
        category="fraud", vendor="Wigets Inc.", total="500.00", currency="USD",
        line_items=1, path="deterministic",
        verdict="needs_review", status="pending_human", role="system",
        policy="TIER-AUTO", total_usd="500.00", pay_status="skipped",
        rail="none", receipt=False, must_include_codes=["vendor_unknown"],
        known_gap="Deterministic layer only emits vendor_unknown; the "
                  "vendor_typosquat detection ('Wigets' vs 'Widgets') depends on "
                  "the LLM screener, which is off in the deterministic baseline.",
    ),
    _adv(
        id="INV-2015", file="inv_2015_blocked_rebrand.json", fmt="json",
        category="fraud", vendor="Fraudster Holdings LLC", total="2500.00",
        currency="USD", line_items=1, path="deterministic",
        verdict="reject", status="rejected", role="none", policy=None,
        total_usd="2500.00", pay_status="skipped", rail="none", receipt=False,
        known_gap="The blocklist keys on exact name/alias; a rebrand "
                  "('Fraudster Holdings LLC' vs blocked 'Fraudster LLC') evades the "
                  "block → only vendor_unknown (warn).",
    ),
    # ── Group 5 — currency / format robustness ──────────────────────────────
    _adv(
        id="INV-2016", file="inv_2016_unsupported_currency.json", fmt="json",
        category="fx", vendor="TechParts International", total="500.00",
        currency="CHF", line_items=1, path="llm",
        verdict="needs_review", status="pending_human", role="system",
        policy="TIER-AUTO", total_usd="500.00", pay_status="skipped",
        rail="none", receipt=False, requires_llm=True,
        known_gap="Currency is a 6-value Literal (USD/EUR/GBP/JPY/CAD/AUD); CHF is "
                  "rejected by Pydantic, forcing an LLM round-trip or hard failure "
                  "rather than a clean 'unsupported_currency' finding.",
    ),
    _adv(
        id="INV-2017", file="inv_2017_currency_drift.json", fmt="json",
        category="fx", vendor="TechParts International", total="1000.00",
        currency="USD", line_items=1, path="deterministic",
        verdict="needs_review", status="pending_human", role="system",
        policy="TIER-AUTO", total_usd="1000.00", pay_status="skipped",
        rail="none", receipt=False, must_include_codes=["currency_drift"],
    ),
    _adv(
        id="INV-2018", file="inv_2018_jpy_big_numbers.json", fmt="json",
        category="fx", vendor="TechParts International", total="1500000",
        currency="JPY", line_items=1, path="deterministic",
        verdict="needs_review", status="pending_human", role="manager",
        policy="TIER-MGR", total_usd="10050.00", pay_status="skipped",
        rail="none", receipt=False, must_include_codes=["currency_drift"],
    ),
    _adv(
        id="INV-2019", file="inv_2019_scientific_notation.json", fmt="json",
        category="math_error", vendor="Atlas Industrial Supply", total="15000.00",
        currency="USD", line_items=1, path="deterministic",
        verdict="needs_review", status="pending_human", role="manager",
        policy="TIER-MGR", total_usd="15000.00", pay_status="skipped",
        rail="none", receipt=False, must_include_codes=["stock_overflow"],
    ),
    _adv(
        id="INV-2020", file="inv_2020_thousands_sep.csv", fmt="csv",
        category="math_error", vendor="Reliable Components Inc.", total="9000.00",
        currency="USD", line_items=1, path="llm",
        verdict="needs_review", status="pending_human", role="system",
        policy="TIER-AUTO", total_usd="9000.00", pay_status="skipped",
        rail="none", receipt=False, requires_llm=True,
        known_gap="Comma thousands-separators in CSV amounts break Decimal() "
                  "parsing, forcing the LLM fallback instead of deterministic "
                  "ingestion.",
    ),
    # ── Group 6 — structural / adversarial ──────────────────────────────────
    _adv(
        id="INV-2021", file="inv_2021_empty_line_items.json", fmt="json",
        category="data_integrity", vendor="Summit Manufacturing Co.",
        total="5000.00", currency="USD", line_items=0, path="deterministic",
        verdict="reject", status="rejected", role="none", policy=None,
        total_usd="5000.00", pay_status="skipped", rail="none", receipt=False,
        must_include_codes=["subtotal_mismatch"],
    ),
    _adv(
        id="INV-2022", file="inv_2022_sku_casing.json", fmt="json",
        category="unknown_sku", vendor="Acme Industrial Supplies", total="500.00",
        currency="USD", line_items=1, path="deterministic",
        verdict="pass", status="auto_approved", role="system", policy="TIER-AUTO",
        total_usd="500.00", pay_status="scheduled", rail="ach", receipt=True,
        known_gap="SKU lookup is case-sensitive exact-match; 'widgeta' is treated "
                  "as an unknown SKU instead of matching catalog 'WidgetA'.",
    ),
    _adv(
        id="INV-2023", file="inv_2023_sku_whitespace.json", fmt="json",
        category="unknown_sku", vendor="Reliable Components Inc.", total="500.00",
        currency="USD", line_items=1, path="deterministic",
        verdict="needs_review", status="pending_human", role="system",
        policy="TIER-AUTO", total_usd="500.00", pay_status="skipped",
        rail="none", receipt=False, must_include_codes=["unknown_sku"],
    ),
    _adv(
        id="INV-2024", file="inv_2024_injection_in_terms.txt", fmt="txt",
        category="injection", vendor="Widgets Inc.", total="500.00",
        currency="USD", line_items=1, path="llm",
        verdict="pass", status="auto_approved", role="system", policy="TIER-AUTO",
        total_usd="500.00", pay_status="scheduled", rail="ach", receipt=True,
        requires_llm=True,
    ),
    _adv(
        id="INV-2025", file="inv_2025_precision_overflow.json", fmt="json",
        category="data_integrity", vendor="Atlas Industrial Supply", total="500.00",
        currency="USD", line_items=1, path="deterministic",
        verdict="pass", status="auto_approved", role="system", policy="TIER-AUTO",
        total_usd="500.00", pay_status="scheduled", rail="ach", receipt=True,
    ),
]
