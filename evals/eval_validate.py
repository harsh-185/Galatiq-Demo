"""Validate eval: scores screener filter compliance + rule-engine verdict +
finding-code accuracy. Code-only; LLM judge for finding-message quality is
deferred to eval_approval (where the audit narrative is more amenable).

The validate stage in the pipeline runs AFTER the screener (which can merge
fraud_findings into the report), so we run the pipeline through to the
validate step inclusive to get realistic findings.

Run:
    python -m evals.eval_validate --variant deterministic_baseline
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

from evals.braintrust_upload import upload  # noqa: E402
from evals.loader import GoldenTrajectory, load_goldens  # noqa: E402
from evals.scorers.code_scorers import (  # noqa: E402
    finding_codes_match,
    screener_filter_compliance,
    verdict_match,
)
from evals.variants import VARIANTS, apply_variant  # noqa: E402


def _run_pipeline(golden: GoldenTrajectory, db_path: Path, receipt_dir: Path):
    """Run the full pipeline; return PipelineState and any error string. We
    run the full pipeline (vs trying to stop after validate) because LangGraph
    nodes share state and the cleanest way is end-to-end."""
    from galatiq.agents.pipeline import run_pipeline

    path = REPO_ROOT / golden.file_path
    if not path.exists():
        return None, f"file not found: {path}"
    try:
        state = run_pipeline(
            str(path),
            db_path=str(db_path),
            receipt_dir=str(receipt_dir),
        )
        return state, None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=2)}"


def _score_one(golden: GoldenTrajectory, db_path: Path, receipt_dir: Path) -> dict:
    state, err = _run_pipeline(golden, db_path, receipt_dir)
    if err is not None or state is None or state.get("errors"):
        return {"id": golden.id, "ok": False,
                "error": err or "; ".join(state.get("errors", []))}

    report = state.get("report")
    if report is None:
        return {"id": golden.id, "ok": False, "error": "no validate report on state"}

    summary = state.get("pre_approval_summary")
    scores = {
        "screener_filter": screener_filter_compliance(summary, golden.screener),
        "verdict_match": verdict_match(report, golden.validate),
        "finding_codes_match": finding_codes_match(report, golden.validate),
    }
    overall = sum(s["score"] for s in scores.values()) / len(scores)
    return {"id": golden.id, "ok": overall >= 0.99, "overall": overall, "scores": scores}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="")
    parser.add_argument("--variant", default="deterministic_baseline")
    parser.add_argument("--db-path", default="evals.inventory.db",
                        help="Fresh DB created per run for dedup isolation")
    parser.add_argument("--threshold", type=float, default=0.99)
    parser.add_argument("--braintrust", action="store_true")
    args = parser.parse_args()

    only_ids = [s.strip() for s in args.only.split(",") if s.strip()] or None
    apply_variant(args.variant)
    goldens = load_goldens(only=only_ids)

    db_path = REPO_ROOT / args.db_path
    receipt_dir = REPO_ROOT / "evals_receipts"
    # Fresh DB so dedup state doesn't leak between rows. Re-create on every run.
    from galatiq.db import init_db
    if db_path.exists():
        db_path.unlink()
    init_db(db_path)
    receipt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nEVAL: validate  variant={args.variant}  N={len(goldens)}\n")
    print(f"{'id':<18} {'overall':>7}  {'screen':>6}  {'verdict':>7}  {'codes':>5}  notes")
    print("-" * 90)

    rows = []
    for g in goldens:
        row = _score_one(g, db_path, receipt_dir)
        rows.append(row)
        if "error" in row:
            print(f"{g.id:<18}  ERROR  {row['error'][:60]}")
            continue
        s = row["scores"]
        marker = "✓" if row["ok"] else "✗"
        notes = []
        if not row["ok"]:
            for k, sc in s.items():
                if sc["score"] < 1.0:
                    md = sc.get("metadata", {})
                    notes.append(f"{k}:{md}")
        print(
            f"{g.id:<18} {row['overall']:>7.2f}  "
            f"{s['screener_filter']['score']:>6.2f}  "
            f"{s['verdict_match']['score']:>7.2f}  "
            f"{s['finding_codes_match']['score']:>5.2f}  "
            f"{marker} {(' '.join(notes))[:80]}"
        )

    passed = sum(1 for r in rows if r.get("ok"))
    print(f"\n{passed}/{len(rows)} passed (threshold={args.threshold})")
    if args.braintrust:
        upload("validate", args.variant, rows, goldens)
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
