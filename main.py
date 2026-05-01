"""Spec-compliant entrypoint: `python main.py --invoice_path=...`.

For now, only the Ingestion phase is wired. Validation/Approval/Payment land in later PRs.
"""
from __future__ import annotations

import sys

from galatiq.cli.app import app


def _legacy_shim() -> None:
    """Map the bare `python main.py --invoice_path=foo` form to `ingest --invoice_path=foo`."""
    if len(sys.argv) > 1 and sys.argv[1].startswith("--invoice_path"):
        sys.argv.insert(1, "ingest")


if __name__ == "__main__":
    _legacy_shim()
    app()
