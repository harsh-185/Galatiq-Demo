"""Spec entry point: ``python main.py --invoice_path=data/invoices/invoice_1001.txt``.

The bare form runs the *full* multi-agent pipeline end-to-end (ingest →
pre-approval screener → validate → approve → council → aggregator → HITL queue
→ payment guards → pay) and prints a per-stage walkthrough with stats.

Subcommands (prefix with the command name):
  db-init       (re)create the SQLite reference DB and seed it
  audit-log     show recent approval_log + payment_log entries
  human review  list pending human-review queue entries
  human resolve resolve a queued review (approve/reject) and re-run pay

For all other use cases, the default form is what you want.
"""
from __future__ import annotations

import sys

from galatiq.cli.app import app


def _bare_form_shim() -> None:
    """Map ``python main.py --invoice_path=foo`` (no subcommand) to ``pay``.

    Keeps the spec's documented invocation form working while routing through
    the consolidated pipeline.
    """
    if len(sys.argv) > 1 and sys.argv[1].startswith("--invoice_path"):
        sys.argv.insert(1, "pay")


if __name__ == "__main__":
    _bare_form_shim()
    app()
