"""Umbrella runner: runs all five eval suites against one or more variants.

Use this for the "full sweep" — typically nightly in CI or before a release.
Each suite is invoked as a subprocess so per-suite failures don't poison the
rest, and the exit code is the sum (per-suite count) of failures.

Run:
    python -m evals.run_all
    python -m evals.run_all --variant grok3_full --variant grok3_mini
    python -m evals.run_all --variant deterministic_baseline --only INV-1004,INV-1014
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

EVAL_MODULES = [
    ("ingest", "evals.eval_ingest", []),
    ("validate", "evals.eval_validate", []),
    ("approval", "evals.eval_approval", []),
    ("payment", "evals.eval_payment", ["--include-idempotency"]),
    ("trajectory", "evals.eval_trajectory", []),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", action="append", default=None,
                        help="May be passed multiple times")
    parser.add_argument("--only", default="")
    parser.add_argument("--with-judge", action="store_true",
                        help="Run LLM judges (validate + approval + trajectory)")
    parser.add_argument("--braintrust", action="store_true",
                        help="Upload all results to braintrust")
    args = parser.parse_args()

    variants = args.variant or ["deterministic_baseline"]
    total_failures = 0

    for variant in variants:
        print(f"\n{'=' * 70}\nVARIANT: {variant}\n{'=' * 70}")
        for label, module, extra_flags in EVAL_MODULES:
            print(f"\n--- {label} ---")
            cmd = [
                sys.executable, "-m", module,
                "--variant", variant,
                *extra_flags,
            ]
            if args.only:
                cmd += ["--only", args.only]
            if args.with_judge and label in {"approval", "trajectory"}:
                cmd += ["--with-judge"]
            if args.braintrust and label == "ingest":
                cmd += ["--braintrust"]
            try:
                result = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
                if result.returncode != 0:
                    total_failures += 1
                    print(f"  {label}: FAILED (exit {result.returncode})")
            except Exception as e:  # noqa: BLE001
                total_failures += 1
                print(f"  {label}: ERROR {e}")

    print(f"\n{'=' * 70}\nTOTAL: {total_failures} suite(s) failed across {len(variants)} variant(s)\n")
    return 1 if total_failures > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
