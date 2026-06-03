"""Shared Braintrust upload helper used by every eval script.

Keeping this in one place means all five suites log to the same project with
the same row shape, so the Braintrust UI can build a per-node leaderboard
across variants. Upload is a no-op (with a printed note) when the package or
API key is missing, so the eval scripts run fine offline.
"""
from __future__ import annotations

import json
import os
from typing import Any

PROJECT = "galatiq"


def _decimal_safe(o: Any) -> str:
    return str(o)


def upload(eval_name: str, variant: str, rows: list[dict], goldens: list) -> None:
    """Log one experiment per (eval_name, variant) to the 'galatiq' project.

    ``rows`` are the per-golden score dicts produced by an eval script:
        {"id", "ok", "overall"?, "scores"?, "error"?, "status"?}
    ``goldens`` is the matching GoldenTrajectory list (for input/expected).
    """
    if not os.environ.get("BRAINTRUST_API_KEY"):
        print("(skipping braintrust upload: BRAINTRUST_API_KEY not set)")
        return
    try:
        from braintrust import init
    except ImportError:
        print("(skipping braintrust upload: braintrust package not installed)")
        return

    experiment = init(project=PROJECT, experiment=f"{eval_name}_{variant}")
    by_id = {g.id: g for g in goldens}
    for row in rows:
        g = by_id.get(row["id"])
        if g is None:
            continue
        scores = {k: v["score"] for k, v in (row.get("scores") or {}).items()}
        experiment.log(
            input={
                "id": g.id,
                "file_path": g.file_path,
                "category": g.category,
                "known_gap": g.known_gap,
            },
            output=row.get("scores"),
            expected=json.loads(json.dumps(g.to_dict(), default=_decimal_safe)),
            scores=scores,
            metadata={
                "variant": variant,
                "eval": eval_name,
                "ok": row.get("ok", False),
                "status": row.get("status"),
                "error": row.get("error"),
                "overall": row.get("overall"),
            },
        )
    experiment.flush()
    print(f"Uploaded {len(rows)} rows to Braintrust '{eval_name}_{variant}'")
