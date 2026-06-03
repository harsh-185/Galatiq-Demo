"""Payment eval: scores payment action correctness, payment_guards verdict,
idempotency (when --include-idempotency), and the near-dup-citation gate.

Run:
    python -m evals.eval_payment --variant deterministic_baseline --include-idempotency
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

from evals.loader import GoldenTrajectory, load_goldens  # noqa: E402
from evals.scorers.code_scorers import (  # noqa: E402
    idempotency_gate,
    near_dup_citation_gate,
    payment_action_match,
    payment_guards_match,
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


def _score_one(golden, db_path, receipt_dir, include_idempotency: bool) -> dict:
    # For INV-DUPLICATE we deliberately run twice and assert on the second run.
    if golden.id == "INV-DUPLICATE":
        # First run primes the ledger.
        first, err = _run_pipeline(golden, db_path, receipt_dir)
        if err:
            return {"id": golden.id, "ok": False, "error": f"first-run failed: {err}"}
        # Second run is what we score.
        state, err = _run_pipeline(golden, db_path, receipt_dir)
    else:
        state, err = _run_pipeline(golden, db_path, receipt_dir)

    if err or state is None or state.get("errors"):
        return {"id": golden.id, "ok": False,
                "error": err or "; ".join(state.get("errors", []))}

    payment = state.get("payment")
    if payment is None:
        return {"id": golden.id, "ok": False, "error": "no payment on state"}

    scores = {
        "payment_action_match": payment_action_match(payment, golden.pay),
        "payment_guards": payment_guards_match(state, golden.payment_guards),
    }

    # Must-pass gates
    if include_idempotency and "idempotency" in golden.trajectory.must_pass_gates:
        invoice = state.get("ingestion").invoice if state.get("ingestion") else None
        if invoice:
            scores["idempotency"] = idempotency_gate(db_path, invoice.invoice_number)

    if "near_dup_citation" in golden.trajectory.must_pass_gates:
        scores["near_dup_citation"] = near_dup_citation_gate(state, None)

    overall = sum(s["score"] for s in scores.values()) / len(scores)
    return {"id": golden.id, "ok": overall >= 0.99, "overall": overall,
            "scores": scores}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="")
    parser.add_argument("--variant", default="deterministic_baseline")
    parser.add_argument("--db-path", default="evals.inventory.db")
    parser.add_argument("--include-idempotency", action="store_true",
                        help="Also run the duplicate-payment idempotency gate")
    parser.add_argument("--threshold", type=float, default=0.99)
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

    print(f"\nEVAL: payment  variant={args.variant}  idempotency={args.include_idempotency}  N={len(goldens)}\n")
    print(f"{'id':<18} {'overall':>7}  {'action':>6}  {'guards':>6}  {'gates':>5}  notes")
    print("-" * 90)

    rows = []
    for g in goldens:
        row = _score_one(g, db_path, receipt_dir, args.include_idempotency)
        rows.append(row)
        if "error" in row:
            print(f"{g.id:<18}  ERROR  {row['error'][:60]}")
            continue
        s = row["scores"]
        gates = [k for k in s if k in {"idempotency", "near_dup_citation"}]
        gates_score = (sum(s[k]["score"] for k in gates) / len(gates)) if gates else 1.0
        marker = "✓" if row["ok"] else "✗"
        notes = []
        for k, sc in s.items():
            if sc["score"] < 1.0:
                md = sc.get("metadata", {})
                notes.append(f"{k}:{md}")
        print(
            f"{g.id:<18} {row['overall']:>7.2f}  "
            f"{s['payment_action_match']['score']:>6.2f}  "
            f"{s['payment_guards']['score']:>6.2f}  "
            f"{gates_score:>5.2f}  "
            f"{marker} {(' '.join(notes))[:80]}"
        )

    passed = sum(1 for r in rows if r.get("ok"))
    print(f"\n{passed}/{len(rows)} passed (threshold={args.threshold})")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
