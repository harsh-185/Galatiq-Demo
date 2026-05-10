"""Sweep every invoice in data/invoices/ through the full pipeline.

Prints a per-file row (verdict / decision / approver / payment / rail / USD-eq
/ key findings) plus a roll-up at the end. Use the output as a one-look check
that nothing routes oddly after a code change.

Usage:
    python scripts/sweep_corpus.py                 # full corpus, LLM agents off
    python scripts/sweep_corpus.py --llm-agents    # specialist agents on (Grok)
    python scripts/sweep_corpus.py --only json,csv,xml   # restrict by suffix
    python scripts/sweep_corpus.py --output sweep.md     # also write markdown

The DB is recreated in a tmp directory each run so dedup state never leaks
across sweeps. Receipts are written to a tmp dir and discarded.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

from galatiq.agents.pipeline import run_pipeline  # noqa: E402
from galatiq.db import init_db  # noqa: E402


def _format_row(*, file: str, fmt: str, verdict: str, decision: str, role: str,
                pay: str, rail: str, usd: str, findings: str) -> str:
    return (
        f"{file:<32} {fmt:<4} {verdict:<13} {decision:<14} "
        f"{role:<10} {pay:<10} {rail:<6} {usd:>10}  {findings}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llm-agents", action="store_true",
        help="Enable specialist LLM agents (screener / council / aggregator / "
             "payment_review). Costs API calls."
    )
    parser.add_argument(
        "--only", default="",
        help="Comma-separated list of suffixes to include (e.g. 'json,csv,xml')."
    )
    parser.add_argument(
        "--output", default="",
        help="Also write a markdown report to this path."
    )
    args = parser.parse_args()

    import os
    os.environ["GALATIQ_LLM_AGENTS"] = "1" if args.llm_agents else "0"

    only = set(s.strip().lstrip(".").lower() for s in args.only.split(",") if s.strip())
    invoices = sorted(
        p for p in (REPO_ROOT / "data" / "invoices").iterdir()
        if p.suffix.lower().lstrip(".") in {"txt", "pdf", "json", "csv", "xml"}
    )
    if only:
        invoices = [p for p in invoices if p.suffix.lower().lstrip(".") in only]
    if not invoices:
        print("no invoices to sweep (check --only filter)", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="galatiq-sweep-") as tmp:
        tmp_path = Path(tmp)
        db_path = tmp_path / "sweep.db"
        receipt_dir = tmp_path / "receipts"
        init_db(db_path)

        rows: list[dict] = []
        header = _format_row(
            file="file", fmt="fmt", verdict="verdict", decision="decision",
            role="role", pay="pay", rail="rail", usd="usd", findings="findings",
        )
        print(header)
        print("-" * len(header))

        for p in invoices:
            fmt = p.suffix.lower().lstrip(".")
            try:
                state = run_pipeline(p, db_path=db_path, receipt_dir=receipt_dir)
            except Exception as e:  # noqa: BLE001
                row = {
                    "file": p.name, "fmt": fmt, "verdict": "EXC",
                    "decision": "—", "role": "—", "pay": "—", "rail": "—",
                    "usd": "—", "findings": f"{type(e).__name__}: {e}",
                }
                rows.append(row)
                print(_format_row(**row))
                continue
            if state.get("errors"):
                row = {
                    "file": p.name, "fmt": fmt, "verdict": "ERR",
                    "decision": "—", "role": "—", "pay": "—", "rail": "—",
                    "usd": "—", "findings": "; ".join(state["errors"])[:70],
                }
                rows.append(row)
                print(_format_row(**row))
                continue
            rep = state["report"]
            dec = state["decision"]
            pay = state["payment"]
            findings = ",".join(sorted({f.code for f in rep.findings}))
            row = {
                "file": p.name, "fmt": fmt, "verdict": rep.verdict,
                "decision": dec.status, "role": (dec.approver_role or "—"),
                "pay": pay.status, "rail": pay.rail,
                "usd": str(dec.total_usd), "findings": findings,
            }
            rows.append(row)
            print(_format_row(**row))

    # --- roll-up
    print()
    by_decision: dict[str, int] = {}
    by_verdict: dict[str, int] = {}
    errors = 0
    for r in rows:
        by_decision[r["decision"]] = by_decision.get(r["decision"], 0) + 1
        by_verdict[r["verdict"]] = by_verdict.get(r["verdict"], 0) + 1
        if r["verdict"] in {"ERR", "EXC"}:
            errors += 1
    print(f"Processed {len(rows)} invoices ({errors} errors)")
    print("  by verdict:  " + ", ".join(f"{k}={v}" for k, v in sorted(by_verdict.items())))
    print("  by decision: " + ", ".join(f"{k}={v}" for k, v in sorted(by_decision.items())))

    if args.output:
        out = Path(args.output)
        with out.open("w") as f:
            f.write("# Corpus sweep\n\n")
            f.write(f"LLM agents: {'on' if args.llm_agents else 'off'}\n\n")
            f.write("| file | fmt | verdict | decision | role | pay | rail | usd | findings |\n")
            f.write("|---|---|---|---|---|---|---|---:|---|\n")
            for r in rows:
                findings_cell = r["findings"].replace("|", "\\|") if r["findings"] else "—"
                f.write(
                    f"| {r['file']} | {r['fmt']} | {r['verdict']} | {r['decision']} "
                    f"| {r['role']} | {r['pay']} | {r['rail']} | {r['usd']} "
                    f"| {findings_cell} |\n"
                )
            f.write(f"\n**Totals**: {len(rows)} invoices, {errors} errors\n")
            f.write(f"- by verdict: {by_verdict}\n")
            f.write(f"- by decision: {by_decision}\n")
        print(f"wrote {out}")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
