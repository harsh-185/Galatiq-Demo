from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv

from galatiq.agents.approval import approve
from galatiq.agents.ingestion import ingest
from galatiq.agents.pipeline import run_pipeline
from galatiq.agents.validation import validate
from galatiq.db import (
    DEFAULT_DB_PATH,
    SEED_APPROVAL_POLICIES,
    SEED_INVENTORY,
    SEED_VENDORS,
    connect,
    init_db,
    list_approvals,
    list_inventory,
    list_payments,
    list_policies,
    list_vendors,
    record_approval,
    record_invoice,
)

app = typer.Typer(add_completion=False, help="Galatiq invoice automation CLI")


def _print_result(result, raw_json: bool) -> None:
    inv = result.invoice
    if raw_json:
        typer.echo(json.dumps(inv.model_dump(mode="json"), indent=2, default=str))
        return
    typer.echo(f"path        : {result.path_taken} (retries={result.llm_retries})")
    typer.echo(f"invoice     : {inv.invoice_number}")
    typer.echo(f"vendor      : {inv.vendor!r}")
    typer.echo(f"date / due  : {inv.date}  /  {inv.due_date}")
    typer.echo(f"currency    : {inv.currency}")
    typer.echo(f"line items  : {len(inv.line_items)}")
    for li in inv.line_items:
        typer.echo(f"  - {li.item:<20} qty={li.quantity:<5} @ {li.unit_price}")
    typer.echo(f"subtotal/tax/total : {inv.subtotal} / {inv.tax} / {inv.total}")
    typer.echo(f"terms       : {inv.payment_terms}")
    if inv.ingestion_warnings:
        typer.echo("warnings:")
        for w in inv.ingestion_warnings:
            typer.echo(f"  - [{w.code}] {w.message}")
    if result.notes:
        typer.echo("notes:")
        for n in result.notes:
            typer.echo(f"  - {n}")


@app.command("ingest")
def ingest_cmd(
    invoice_path: Path = typer.Option(..., "--invoice_path", exists=True, readable=True),
    allow_llm: bool = typer.Option(True, "--llm/--no-llm", help="Use LLM fallback if deterministic parse fails"),
    raw_json: bool = typer.Option(False, "--json", help="Print full JSON instead of summary"),
) -> None:
    """Run only the ingestion stage on a single invoice."""
    load_dotenv()
    try:
        result = ingest(invoice_path, allow_llm=allow_llm)
    except Exception as e:
        typer.echo(f"FAILED ({type(e).__name__}): {e}", err=True)
        raise typer.Exit(code=1)
    _print_result(result, raw_json)


@app.command("ingest-all")
def ingest_all_cmd(
    directory: Path = typer.Option(Path("data/invoices"), "--dir", exists=True),
    allow_llm: bool = typer.Option(True, "--llm/--no-llm"),
) -> None:
    """Run ingestion on every invoice in a directory and print a results table."""
    load_dotenv()
    files = sorted(p for p in directory.iterdir() if p.suffix.lower() in {".txt", ".json", ".csv", ".xml", ".pdf"})
    typer.echo(f"{'file':<32} {'path':<13} {'#li':>3} {'inv#':<10} {'vendor':<28} {'total':>10} {'warn':>4}")
    typer.echo("-" * 110)
    failures = 0
    for p in files:
        try:
            r = ingest(p, allow_llm=allow_llm)
            inv = r.invoice
            vendor = (inv.vendor or "").strip() or "<empty>"
            typer.echo(
                f"{p.name:<32} {r.path_taken+('('+str(r.llm_retries)+')' if r.llm_retries else ''):<13} "
                f"{len(inv.line_items):>3} {inv.invoice_number:<10} "
                f"{vendor[:27]:<28} {str(inv.total):>10} {len(inv.ingestion_warnings):>4}"
            )
        except Exception as e:  # noqa: BLE001
            failures += 1
            typer.echo(f"{p.name:<32} FAIL ({type(e).__name__}): {str(e)[:80]}")
    typer.echo("-" * 110)
    typer.echo(f"{len(files) - failures}/{len(files)} succeeded")
    if failures:
        sys.exit(1)


@app.command("validate")
def validate_cmd(
    invoice_path: Path = typer.Option(..., "--invoice_path", exists=True, readable=True),
    db_path: Path = typer.Option(DEFAULT_DB_PATH, "--db", help="Path to the SQLite file"),
    allow_llm: bool = typer.Option(True, "--llm/--no-llm"),
    record: bool = typer.Option(
        False,
        "--record/--no-record",
        help="If verdict == pass, write the invoice to the ledger",
    ),
) -> None:
    """Run ingestion + validation on a single invoice and print the report."""
    load_dotenv()
    if not db_path.exists():
        typer.echo(f"FAILED: reference DB not found at {db_path}; run `db-init` first", err=True)
        raise typer.Exit(code=2)
    try:
        ing = ingest(invoice_path, allow_llm=allow_llm)
    except Exception as e:
        typer.echo(f"INGESTION FAILED ({type(e).__name__}): {e}", err=True)
        raise typer.Exit(code=1)
    invoice = ing.invoice
    with connect(db_path) as conn:
        report = validate(invoice, conn=conn)
        recorded = False
        if record and report.verdict == "pass":
            record_invoice(
                conn,
                invoice_number=invoice.invoice_number,
                vendor=invoice.vendor,
                total=invoice.total,
                source_path=str(invoice_path),
            )
            recorded = True
    typer.echo(f"file        : {invoice_path.name}")
    typer.echo(f"ingestion   : {ing.path_taken} (retries={ing.llm_retries})")
    typer.echo(f"verdict     : {report.verdict.upper()}")
    if not report.findings:
        typer.echo("findings    : (none)")
    else:
        typer.echo("findings    :")
        for f in report.findings:
            field = f" [{f.field}]" if f.field else ""
            typer.echo(f"  - [{f.severity:<5}] {f.code}{field} — {f.message}")
    if recorded:
        typer.echo("ledger      : recorded")
    if report.verdict == "reject":
        raise typer.Exit(code=1)


@app.command("approve")
def approve_cmd(
    invoice_path: Path = typer.Option(..., "--invoice_path", exists=True, readable=True),
    db_path: Path = typer.Option(DEFAULT_DB_PATH, "--db", help="Path to the SQLite file"),
    allow_llm: bool = typer.Option(True, "--llm/--no-llm"),
) -> None:
    """Run ingest + validate + approve. Records an approval_log row."""
    load_dotenv()
    if not db_path.exists():
        typer.echo(f"FAILED: reference DB not found at {db_path}; run `db-init` first", err=True)
        raise typer.Exit(code=2)
    try:
        ing = ingest(invoice_path, allow_llm=allow_llm)
    except Exception as e:
        typer.echo(f"INGESTION FAILED ({type(e).__name__}): {e}", err=True)
        raise typer.Exit(code=1)
    invoice = ing.invoice
    with connect(db_path) as conn:
        report = validate(invoice, conn=conn)
        decision = approve(invoice, report, conn=conn)
        record_approval(
            conn,
            invoice_number=invoice.invoice_number,
            vendor=invoice.vendor,
            status=decision.status,
            approver_role=decision.approver_role,
            policy_id=decision.policy_id,
            total_usd=decision.total_usd,
        )
    typer.echo(f"file        : {invoice_path.name}")
    typer.echo(f"verdict     : {report.verdict.upper()}")
    typer.echo(f"decision    : {decision.status.upper()}  (approver={decision.approver_role}, policy={decision.policy_id or '—'})")
    typer.echo(f"total_usd   : {decision.total_usd}")
    typer.echo(f"justification: {decision.justification}")
    if decision.escalations:
        typer.echo(f"escalations : {', '.join(decision.escalations)}")
    if decision.status == "auto_approved":
        return
    if decision.status == "pending_human":
        raise typer.Exit(code=2)
    raise typer.Exit(code=1)


@app.command("pay")
def pay_cmd(
    invoice_path: Path = typer.Option(..., "--invoice_path", exists=True, readable=True),
    db_path: Path = typer.Option(DEFAULT_DB_PATH, "--db", help="Path to the SQLite file"),
    receipts: Path = typer.Option(Path("data/receipts"), "--receipts", help="Where to write receipt files"),
) -> None:
    """Run the full pipeline (ingest → validate → approve → pay) via LangGraph."""
    load_dotenv()
    if not db_path.exists():
        typer.echo(f"FAILED: reference DB not found at {db_path}; run `db-init` first", err=True)
        raise typer.Exit(code=2)
    state = run_pipeline(invoice_path, db_path=db_path, receipt_dir=receipts)
    if state.get("errors"):
        for err in state["errors"]:
            typer.echo(f"ERROR: {err}", err=True)
        raise typer.Exit(code=1)
    ing = state["ingestion"]
    report = state["report"]
    decision = state["decision"]
    payment = state["payment"]
    typer.echo(f"file        : {invoice_path.name}")
    typer.echo(f"ingestion   : {ing.path_taken} (retries={ing.llm_retries})")
    typer.echo(f"verdict     : {report.verdict.upper()}")
    typer.echo(f"decision    : {decision.status.upper()}  (approver={decision.approver_role}, policy={decision.policy_id or '—'})")
    typer.echo(f"payment     : {payment.status.upper()}  rail={payment.rail}  ref={payment.reference}")
    typer.echo(f"amount      : {payment.amount_paid} {payment.currency_paid}  ({payment.amount_usd} USD)")
    typer.echo(f"scheduled   : {payment.scheduled_for or '—'}")
    if payment.receipt_path:
        typer.echo(f"receipt     : {payment.receipt_path}")
    if payment.notes:
        for n in payment.notes:
            typer.echo(f"  note: {n}")
    if payment.status == "scheduled":
        return
    if payment.status == "skipped" and decision.status == "rejected":
        raise typer.Exit(code=1)
    if payment.status == "skipped":
        raise typer.Exit(code=2)
    raise typer.Exit(code=1)


@app.command("audit-log")
def audit_log_cmd(
    db_path: Path = typer.Option(DEFAULT_DB_PATH, "--db", help="Path to the SQLite file"),
    limit: int = typer.Option(20, "--limit", help="Max rows per section"),
) -> None:
    """Print recent approval and payment log rows."""
    if not db_path.exists():
        typer.echo(f"FAILED: reference DB not found at {db_path}", err=True)
        raise typer.Exit(code=2)
    with connect(db_path) as conn:
        approvals = list_approvals(conn, limit=limit)
        payments = list_payments(conn, limit=limit)
    typer.echo(f"approval_log  ({len(approvals)} most recent)")
    typer.echo("-" * 100)
    typer.echo(f"{'when':<26} {'invoice':<14} {'vendor':<22} {'status':<14} {'role':<10} {'policy':<10} {'usd':>10}")
    for r in approvals:
        typer.echo(
            f"{r['decided_at']:<26} {r['invoice_number']:<14} {r['vendor'][:21]:<22} "
            f"{r['status']:<14} {r['approver_role']:<10} {r['policy_id'] or '—':<10} "
            f"{r['total_usd']:>10}"
        )
    typer.echo("")
    typer.echo(f"payment_log  ({len(payments)} most recent)")
    typer.echo("-" * 100)
    typer.echo(f"{'when':<26} {'reference':<40} {'rail':<6} {'status':<10} {'usd':>10}")
    for r in payments:
        typer.echo(
            f"{r['recorded_at']:<26} {r['reference']:<40} {r['rail']:<6} "
            f"{r['status']:<10} {r['amount_usd']:>10}"
        )


@app.command("db-init")
def db_init_cmd(
    db_path: Path = typer.Option(DEFAULT_DB_PATH, "--db", help="Path to the SQLite file"),
    fresh: bool = typer.Option(True, "--fresh/--keep", help="Recreate the file from scratch"),
) -> None:
    """(Re)create the reference DB and apply seed inventory + vendor data."""
    path = init_db(db_path, fresh=fresh)
    with connect(path) as conn:
        inventory = list_inventory(conn)
        vendors = list_vendors(conn)
        policies = list_policies(conn)
    typer.echo(f"db          : {path}  ({'fresh' if fresh else 'kept'})")
    typer.echo(
        f"seed defined: {len(SEED_INVENTORY)} items, {len(SEED_VENDORS)} vendors, "
        f"{len(SEED_APPROVAL_POLICIES)} policies"
    )
    typer.echo("inventory   :")
    for inv_item in inventory:
        price = "—" if inv_item.unit_price is None else f"${inv_item.unit_price}"
        typer.echo(
            f"  - {inv_item.item:<16} stock={inv_item.stock:<5} price={price:<10} "
            f"category={inv_item.category or '—':<12} status={inv_item.status}"
        )
    typer.echo("vendors     :")
    for v in vendors:
        aliases = ", ".join(v.aliases) if v.aliases else "—"
        typer.echo(f"  - {v.vendor_id} {v.name!r:<24} status={v.status:<12} aliases=[{aliases}]")
    typer.echo("policies    :")
    for p in policies:
        upper = "∞" if p.max_usd is None else f"${p.max_usd}"
        typer.echo(
            f"  - {p.policy_id:<10} ${p.min_usd}–{upper:<10} → {p.approver_role}"
        )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
