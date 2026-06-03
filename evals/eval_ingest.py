"""Ingest eval: scores extraction quality against the golden Invoice fields.

Code-only — no LLM judge. Extraction correctness is deterministic enough that
a comparison + hallucination check is more trustworthy than an LLM judge would
be, and it's instant.

Run:
    python -m evals.eval_ingest                                # full set, default variant
    python -m evals.eval_ingest --only INV-1004,INV-1014       # subset
    python -m evals.eval_ingest --variant deterministic_baseline
    python -m evals.eval_ingest --braintrust                   # also upload to project

Exit code: 0 if all goldens score >= threshold, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import os
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
    hallucination_check,
    injection_resistance_gate,
    invoice_field_match,
)
from evals.variants import VARIANTS, apply_variant  # noqa: E402


def _ingest_one(golden: GoldenTrajectory) -> tuple[object | None, str | None]:
    """Run only the ingest stage. Returns (IngestionResult or None, error str or None).

    We bypass the orchestrator and call ingest() directly so we evaluate just
    the extraction step. allow_llm is True (the LLM fallback IS part of ingest)."""
    from galatiq.agents.ingestion import ingest

    path = REPO_ROOT / golden.file_path
    if not path.exists():
        return None, f"file not found: {path}"
    try:
        return ingest(path, allow_llm=True), None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=2)}"


def _score_one(golden: GoldenTrajectory) -> dict:
    result, err = _ingest_one(golden)
    if err is not None:
        return {
            "id": golden.id, "ok": False, "error": err,
            "scores": {},
        }
    scores = {
        "invoice_field_match": invoice_field_match(result, golden.ingest),
        "hallucination": hallucination_check(result, golden.ingest),
    }
    if golden.category == "injection":
        scores["injection_resistance"] = injection_resistance_gate(
            result, golden.ingest
        )
    overall = sum(s["score"] for s in scores.values()) / len(scores)
    return {
        "id": golden.id, "ok": overall >= 0.99, "overall": overall,
        "scores": scores,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="",
                        help="Comma-separated list of golden ids to run")
    parser.add_argument("--variant", default="deterministic_baseline",
                        help=f"Variant name (one of: {', '.join(v['name'] for v in VARIANTS)})")
    parser.add_argument("--braintrust", action="store_true",
                        help="Also upload results to Braintrust project 'galatiq'")
    parser.add_argument("--threshold", type=float, default=0.99,
                        help="Per-row pass threshold (default 0.99)")
    args = parser.parse_args()

    only_ids = [s.strip() for s in args.only.split(",") if s.strip()] or None
    apply_variant(args.variant)
    goldens = load_goldens(only=only_ids)

    print(f"\nEVAL: ingest  variant={args.variant}  N={len(goldens)}\n")
    print(f"{'id':<18} {'overall':>7}  {'field':>5}  {'halluc':>6}  notes")
    print("-" * 78)
    rows = []
    for g in goldens:
        row = _score_one(g)
        rows.append(row)
        if "error" in row:
            print(f"{g.id:<18}  ERROR  {row['error'][:60]}")
            continue
        field_s = row["scores"]["invoice_field_match"]["score"]
        halluc_s = row["scores"]["hallucination"]["score"]
        notes = []
        if not row["ok"]:
            md = row["scores"]["invoice_field_match"].get("metadata", {})
            bad = [k for k, v in md.items() if v is False]
            if bad:
                notes.append(f"failed={','.join(bad)}")
        marker = "✓" if row["ok"] else "✗"
        print(f"{g.id:<18} {row['overall']:>7.2f}  {field_s:>5.2f}  {halluc_s:>6.2f}  {marker} {' '.join(notes)}")

    passed = sum(1 for r in rows if r.get("ok"))
    print(f"\n{passed}/{len(rows)} passed (threshold={args.threshold})")

    if args.braintrust:
        _upload_to_braintrust("ingest", args.variant, rows, goldens)

    return 0 if passed == len(rows) else 1


def _upload_to_braintrust(eval_name, variant, rows, goldens) -> None:
    """Upload results as a Braintrust experiment under project 'galatiq'."""
    if not os.environ.get("BRAINTRUST_API_KEY"):
        print("(skipping braintrust upload: BRAINTRUST_API_KEY not set)")
        return
    try:
        from braintrust import init
    except ImportError:
        print("(skipping braintrust upload: braintrust package not installed)")
        return
    experiment = init(project="galatiq", experiment=f"{eval_name}_{variant}")
    by_id = {g.id: g for g in goldens}
    for row in rows:
        g = by_id[row["id"]]
        experiment.log(
            input={"id": g.id, "file_path": g.file_path, "category": g.category},
            output=row.get("scores"),
            expected={"ingest": json.loads(json.dumps(g.ingest, default=str))},
            scores={k: v["score"] for k, v in row.get("scores", {}).items()},
            metadata={
                "variant": variant,
                "ok": row.get("ok", False),
                "error": row.get("error"),
            },
        )
    experiment.flush()
    print(f"Uploaded {len(rows)} rows to Braintrust experiment '{eval_name}_{variant}'")


if __name__ == "__main__":
    raise SystemExit(main())
