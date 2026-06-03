"""Trajectory eval: scores end-to-end pipeline runs.

This is the "did the agent reach the right answer via the right path" check.
Combines deterministic scorers (stages-visited check, final decision match,
payment-action match) and optionally the trajectory-rubric LLM judge.

Run:
    python -m evals.eval_trajectory --variant deterministic_baseline
    python -m evals.eval_trajectory --variant grok3_full --with-judge
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
    decision_match,
    payment_action_match,
    trajectory_stages_match,
)
from evals.variants import VARIANTS, apply_variant  # noqa: E402


def _run_pipeline(golden, db_path, receipt_dir):
    from galatiq.agents.pipeline import run_pipeline

    path = REPO_ROOT / golden.file_path
    if not path.exists():
        return None, f"file not found: {path}"
    try:
        return run_pipeline(str(path), db_path=str(db_path),
                            receipt_dir=str(receipt_dir)), None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=2)}"


def _score_one(golden, db_path, receipt_dir, use_judge: bool) -> dict:
    state, err = _run_pipeline(golden, db_path, receipt_dir)
    if err or state is None or state.get("errors"):
        return {"id": golden.id, "ok": False,
                "error": err or "; ".join(state.get("errors", []))}

    decision = state.get("decision")
    payment = state.get("payment")
    scores = {
        "decision_match": decision_match(decision, golden.approve),
        "payment_action_match": payment_action_match(payment, golden.pay),
        "trajectory_stages": trajectory_stages_match(state, golden.trajectory),
    }
    if use_judge:
        from evals.scorers.llm_judges import score_trajectory
        scores["trajectory_judge"] = score_trajectory(state, golden)

    overall = sum(s["score"] for s in scores.values()) / len(scores)
    return {"id": golden.id, "ok": overall >= 0.95, "overall": overall,
            "scores": scores}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="")
    parser.add_argument("--variant", default="deterministic_baseline")
    parser.add_argument("--db-path", default="evals.inventory.db")
    parser.add_argument("--with-judge", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--braintrust", action="store_true")
    args = parser.parse_args()

    only_ids = [s.strip() for s in args.only.split(",") if s.strip()] or None
    apply_variant(args.variant)
    goldens = load_goldens(only=only_ids)

    db_path = REPO_ROOT / args.db_path
    receipt_dir = REPO_ROOT / "evals_receipts"
    from galatiq.db import init_db
    if db_path.exists():
        db_path.unlink()
    init_db(db_path)
    receipt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nEVAL: trajectory  variant={args.variant}  judge={args.with_judge}  N={len(goldens)}\n")
    print(f"{'id':<18} {'overall':>7}  {'decis':>5}  {'pay':>4}  {'stages':>6}  notes")
    print("-" * 90)

    rows = []
    for g in goldens:
        row = _score_one(g, db_path, receipt_dir, args.with_judge)
        rows.append(row)
        if "error" in row:
            print(f"{g.id:<18}  ERROR  {row['error'][:60]}")
            continue
        s = row["scores"]
        marker = "✓" if row["ok"] else "✗"
        notes = []
        for k, sc in s.items():
            if sc["score"] < 1.0:
                md = sc.get("metadata", {})
                notes.append(f"{k}:{md}")
        print(
            f"{g.id:<18} {row['overall']:>7.2f}  "
            f"{s['decision_match']['score']:>5.2f}  "
            f"{s['payment_action_match']['score']:>4.2f}  "
            f"{s['trajectory_stages']['score']:>6.2f}  "
            f"{marker} {(' '.join(notes))[:80]}"
        )

    passed = sum(1 for r in rows if r.get("ok"))
    print(f"\n{passed}/{len(rows)} passed (threshold={args.threshold})")
    if args.braintrust:
        upload("trajectory", args.variant, rows, goldens)
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
