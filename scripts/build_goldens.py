"""Dump the in-Python golden set to a JSON snapshot for human review / diff.

Run:
    python scripts/build_goldens.py            # writes evals/goldens.snapshot.json
    python scripts/build_goldens.py --check    # exits 1 if snapshot drifted from code

Useful in CI as a guard: any change to evals/golden_dataset.py must also
regenerate the snapshot, so PR reviewers see the expected-outcome diff.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from evals.adversarial_goldens import ADVERSARIAL_GOLDENS  # noqa: E402
from evals.golden_dataset import GOLDENS  # noqa: E402

SNAPSHOT_PATH = REPO_ROOT / "evals" / "goldens.snapshot.json"

_ALL = list(GOLDENS) + list(ADVERSARIAL_GOLDENS)


def _serialize() -> str:
    payload = {
        "core": [asdict(g) for g in GOLDENS],
        "adversarial": [asdict(g) for g in ADVERSARIAL_GOLDENS],
    }
    return json.dumps(payload, indent=2, default=str)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="Exit 1 if the snapshot is stale relative to the code")
    args = parser.parse_args()

    fresh = _serialize()
    if args.check:
        if not SNAPSHOT_PATH.exists():
            print(f"snapshot missing: {SNAPSHOT_PATH}")
            return 1
        if SNAPSHOT_PATH.read_text() != fresh:
            print(f"snapshot is stale relative to evals/golden_dataset.py")
            print(f"  run: python scripts/build_goldens.py")
            return 1
        print(f"snapshot up to date ({len(_ALL)} goldens)")
        return 0

    SNAPSHOT_PATH.write_text(fresh)
    print(f"wrote {SNAPSHOT_PATH} ({len(_ALL)} goldens)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
