"""Adversarial eval: runs the 25 stress invoices end-to-end and classifies each
row as pass / known_gap / regression.

- pass:       system met the ideal golden outcome
- known_gap:  system diverged AND the golden documents the limitation (xfail)
- regression: system diverged with no documented gap → a real, build-failing bug

Only regressions set a non-zero exit code. Known gaps are printed as ⚠ with the
documented reason so the report doubles as a living "known limitations" list.

Run:
    python -m evals.eval_adversarial --variant deterministic_baseline
    python -m evals.eval_adversarial --variant grok3_full        # exercises requires_llm cases
    python -m evals.eval_adversarial --braintrust
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
from evals.loader import GoldenTrajectory, classify_row, load_goldens  # noqa: E402
from evals.scorers.code_scorers import (  # noqa: E402
    decision_match,
    payment_action_match,
    verdict_match,
)
from evals.variants import apply_variant  # noqa: E402


def _run_pipeline(golden, db_path, receipt_dir):
    from galatiq.agents.pipeline import run_pipeline

    path = REPO_ROOT / golden.file_path
    if not path.exists():
        return None, f"file not found: {path}"
    try:
        return run_pipeline(str(path), db_path=str(db_path),
                            receipt_dir=str(receipt_dir)), None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=1)}"


def _score_one(golden: GoldenTrajectory, db_path: Path, receipt_dir: Path) -> dict:
    state, err = _run_pipeline(golden, db_path, receipt_dir)
    if err or state is None or state.get("errors"):
        # An ingestion crash / error IS the observed behaviour. If the golden
        # documents it as a known gap, that's a known_gap row, not a hard fail.
        return {"id": golden.id, "ok": False,
                "error": err or "; ".join(state.get("errors", [])),
                "scores": {}}

    report = state.get("report")
    decision = state.get("decision")
    payment = state.get("payment")
    scores = {
        "verdict_match": verdict_match(report, golden.validate),
        "decision_match": decision_match(decision, golden.approve),
        "payment_action_match": payment_action_match(payment, golden.pay),
    }
    overall = sum(s["score"] for s in scores.values()) / len(scores)
    return {"id": golden.id, "ok": overall >= 0.99, "overall": overall,
            "scores": scores}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="")
    parser.add_argument("--variant", default="deterministic_baseline")
    parser.add_argument("--db-path", default="evals.adversarial.db")
    parser.add_argument("--braintrust", action="store_true")
    args = parser.parse_args()

    only_ids = [s.strip() for s in args.only.split(",") if s.strip()] or None
    variant = apply_variant(args.variant)
    is_deterministic = variant["env"].get("GALATIQ_LLM_AGENTS") == "0"
    goldens = load_goldens(only=only_ids, dataset="adversarial")

    db_path = REPO_ROOT / args.db_path
    receipt_dir = REPO_ROOT / "evals_receipts"
    from galatiq.db import init_db
    if db_path.exists():
        db_path.unlink()
    init_db(db_path)
    receipt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nEVAL: adversarial  variant={args.variant}  N={len(goldens)}\n")
    print(f"{'id':<12} {'status':<11} {'overall':>7}  note")
    print("-" * 100)

    rows = []
    counts = {"pass": 0, "known_gap": 0, "regression": 0, "skipped": 0}
    for g in goldens:
        if g.requires_llm and is_deterministic:
            counts["skipped"] += 1
            rows.append({"id": g.id, "status": "skipped",
                         "note": "requires LLM; skipped in deterministic baseline"})
            print(f"{g.id:<12} {'skipped':<11} {'—':>7}  requires LLM (gap: {g.known_gap[:50] if g.known_gap else '—'})")
            continue

        row = _score_one(g, db_path, receipt_dir)
        status = classify_row(g, row.get("ok", False))
        row["status"] = status
        counts[status] += 1
        rows.append(row)

        overall = row.get("overall")
        overall_s = f"{overall:.2f}" if overall is not None else "ERR"
        marker = {"pass": "✓", "known_gap": "⚠", "regression": "✗"}[status]
        if status == "known_gap":
            note = f"KNOWN GAP: {g.known_gap[:70]}"
        elif status == "regression":
            note = f"REGRESSION: {row.get('error') or _fail_detail(row)}"
        else:
            note = ""
        print(f"{g.id:<12} {marker + ' ' + status:<11} {overall_s:>7}  {note[:78]}")
        # For regressions, dump the full (untruncated) per-scorer detail so the
        # divergence is diagnosable without a second run.
        if status == "regression":
            for k, sc in (row.get("scores") or {}).items():
                if sc["score"] < 1.0:
                    print(f"             └─ {k}: {sc.get('metadata', {})}")

    print(
        f"\n{counts['pass']} pass · {counts['known_gap']} known-gap (xfail) · "
        f"{counts['regression']} regression · {counts['skipped']} skipped"
    )
    if counts["regression"]:
        print("\n⚠ Regressions are undocumented divergences — investigate or add a known_gap.")

    if args.braintrust:
        upload("adversarial", args.variant, rows, goldens)

    return 1 if counts["regression"] else 0


def _fail_detail(row: dict) -> str:
    bits = []
    for k, sc in (row.get("scores") or {}).items():
        if sc["score"] < 1.0:
            bits.append(f"{k}={sc.get('metadata', {})}")
    return "; ".join(bits)


if __name__ == "__main__":
    raise SystemExit(main())
