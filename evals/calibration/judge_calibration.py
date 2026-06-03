"""Calibrate the approval_narrative LLM judge against hand labels.

Workflow:
1. You hand-label ~30 (state, expected verdict) pairs in human_labels.json.
2. This script runs the judge against the same 30 narratives and computes
   Cohen's kappa between judge labels and your labels.
3. Target κ ≥ 0.7. If lower, iterate the rubric (in rubrics/approval_narrative.txt)
   and re-run.

Run:
    python -m evals.calibration.judge_calibration

The human_labels.json file is committed; it's the judge's regression suite.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

LABELS_PATH = Path(__file__).parent / "human_labels.json"


def main() -> int:
    if not LABELS_PATH.exists():
        print(f"No labels at {LABELS_PATH}. Create it with shape:")
        print('  [{"id": "...", "narrative": "...", "human_label": "good"}]')
        return 2
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; cannot calibrate.")
        return 2

    from evals.scorers.llm_judges import _build_classifier  # private but ok here

    judge = _build_classifier("approval_narrative.txt", "approval_narrative")
    labels = json.loads(LABELS_PATH.read_text())

    human_labels = []
    judge_labels = []
    for item in labels:
        result = judge(
            input={"invoice_summary": item.get("invoice_summary", {}),
                   "engine_decision": item.get("engine_decision", {}),
                   "opinions": item.get("opinions", []),
                   "final_decision": item.get("final_decision", {})},
            output={"narrative": item["narrative"]},
            expected={"golden_status": item.get("golden_status")},
        )
        # autoevals returns scores in [0, 1]; map back to label via threshold
        score = float(result.score)
        bucket = {1.0: "excellent", 0.75: "good", 0.5: "fair",
                  0.25: "poor", 0.0: "wrong"}
        judge_label = min(bucket.keys(), key=lambda k: abs(score - k))
        judge_labels.append(bucket[judge_label])
        human_labels.append(item["human_label"])
        print(f"{item['id']:>14}  human={item['human_label']:<10}  judge={bucket[judge_label]:<10}  score={score:.2f}")

    try:
        from sklearn.metrics import cohen_kappa_score
    except ImportError:
        print("sklearn not installed; install via `pip install scikit-learn`")
        return 2

    k = cohen_kappa_score(human_labels, judge_labels)
    print(f"\nCohen's kappa: {k:.3f}  (target >= 0.7)")
    return 0 if k >= 0.7 else 1


if __name__ == "__main__":
    raise SystemExit(main())
