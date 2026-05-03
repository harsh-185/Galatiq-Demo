from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv

from galatiq.agents._walkthrough import render_walkthrough, summary_stats
from galatiq.agents.approval import ApprovalDecision, approve
from galatiq.agents.ingestion import ingest
from galatiq.agents.payment import pay as pay_agent
from galatiq.agents.payment_guards import run_payment_guards
from galatiq.agents.pipeline import run_pipeline
from galatiq.agents.validation import validate
from galatiq.db import (
    DEFAULT_DB_PATH,
    SEED_APPROVAL_POLICIES,
    SEED_INVENTORY,
    SEED_VENDORS,
    connect,
    get_review_entry,
    init_db,
    list_approvals,
    list_inventory,
    list_payments,
    list_pending_reviews,
    list_policies,
    list_vendors,
    lookup_vendor,
    record_approval,
    record_invoice,
    record_payment,
    resolve_review,
)
from galatiq.payments.receipt import render_receipt, write_receipt

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
    llm_agents: bool | None = typer.Option(
        None,
        "--llm-agents/--no-llm-agents",
        help="Force-enable or disable the LLM specialist agents (default: env-based)",
    ),
) -> None:
    """Run the full pipeline (ingest → fraud-screen → validate → … → pay) via LangGraph."""
    load_dotenv()
    if llm_agents is not None:
        os.environ["GALATIQ_LLM_AGENTS"] = "1" if llm_agents else "0"
    if not db_path.exists():
        typer.echo(f"FAILED: reference DB not found at {db_path}; run `db-init` first", err=True)
        raise typer.Exit(code=2)
    state = run_pipeline(invoice_path, db_path=db_path, receipt_dir=receipts)
    if state.get("errors"):
        for err in state["errors"]:
            typer.echo(f"ERROR: {err}", err=True)
        raise typer.Exit(code=1)

    decision = state["decision"]
    payment = state["payment"]

    # ── Stage-by-stage walkthrough ──────────────────────────────────────────
    walkthrough = state.get("walkthrough") or []
    typer.echo("")
    typer.echo(f"📄 Invoice: {invoice_path.name}")
    typer.echo(render_walkthrough(walkthrough))

    # ── Final summary ───────────────────────────────────────────────────────
    stats = summary_stats(walkthrough)
    typer.echo("")
    typer.echo("━" * 60)
    typer.echo("Summary")
    typer.echo("━" * 60)
    outcome_marker = {
        "scheduled": "✅",
        "skipped":   "⏭ ",
        "failed":    "❌",
    }.get(payment.status, "•")
    typer.echo(
        f"{outcome_marker} payment {payment.status.upper()}  "
        f"rail={payment.rail}  ref={payment.reference}"
    )
    typer.echo(
        f"   decision: {decision.status} ({decision.approver_role}, "
        f"{decision.policy_id or '—'})  total_usd=${decision.total_usd}"
    )
    if payment.receipt_path:
        typer.echo(f"   receipt: {payment.receipt_path}")
    review_id = state.get("human_review_id")
    if review_id is not None:
        typer.echo(
            f"   human review queued as id={review_id}; resolve with "
            f"`human resolve --id {review_id} --action approve|reject`"
        )
    typer.echo(
        f"   stats: {stats['total_ms']:.1f} ms total  ·  "
        f"{stats['total_llm_calls']} LLM calls  ·  "
        f"{stats['total_tool_calls']} tool calls  ·  "
        f"{stats['stages_completed']}/{len(walkthrough)} stages run "
        f"({stats['stages_skipped']} skipped, {stats['stages_failed']} failed)"
    )

    llm_errs = state.get("llm_agent_errors") or []
    if llm_errs:
        typer.echo("")
        typer.echo("⚠ LLM-agent fallbacks:")
        for e in llm_errs:
            typer.echo(f"   - {e}")

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


human_app = typer.Typer(help="Human-in-the-loop review queue commands")
app.add_typer(human_app, name="human")


@human_app.command("review")
def human_review_cmd(
    db_path: Path = typer.Option(DEFAULT_DB_PATH, "--db", help="Path to the SQLite file"),
) -> None:
    """List unresolved human-review-queue entries."""
    if not db_path.exists():
        typer.echo(f"FAILED: reference DB not found at {db_path}", err=True)
        raise typer.Exit(code=2)
    with connect(db_path) as conn:
        rows = list_pending_reviews(conn)
    if not rows:
        typer.echo("No pending reviews.")
        return
    typer.echo(f"{len(rows)} pending review(s):")
    typer.echo("-" * 110)
    typer.echo(f"{'id':<5} {'queued_at':<26} {'invoice':<16} {'vendor':<22} {'role':<10} {'usd':>10}")
    for r in rows:
        typer.echo(
            f"{r['id']:<5} {r['queued_at']:<26} {r['invoice_number']:<16} "
            f"{r['vendor'][:21]:<22} {r['approver_role']:<10} {r['total_usd']:>10}"
        )
        if r["narrative"]:
            typer.echo(f"      narrative: {r['narrative']}")


@human_app.command("resolve")
def human_resolve_cmd(
    review_id: int = typer.Option(..., "--id", help="Queue entry id to resolve"),
    action: str = typer.Option(..., "--action", help="approve | reject"),
    note: str = typer.Option(None, "--note", help="Reason for the decision"),
    resolved_by: str = typer.Option("cli-human", "--by", help="Who resolved it"),
    db_path: Path = typer.Option(DEFAULT_DB_PATH, "--db"),
    receipts: Path = typer.Option(Path("data/receipts"), "--receipts"),
) -> None:
    """Resolve a queued review. ``approve`` re-runs the pay phase with an
    overridden auto_approved decision; ``reject`` records the rejection."""
    if action not in ("approve", "reject"):
        typer.echo("--action must be 'approve' or 'reject'", err=True)
        raise typer.Exit(code=2)
    if not db_path.exists():
        typer.echo(f"FAILED: reference DB not found at {db_path}", err=True)
        raise typer.Exit(code=2)

    with connect(db_path) as conn:
        entry = get_review_entry(conn, review_id)
    if entry is None:
        typer.echo(f"No review entry id={review_id}", err=True)
        raise typer.Exit(code=1)
    if entry["resolved_at"] is not None:
        typer.echo(f"Entry id={review_id} already resolved at {entry['resolved_at']} as {entry['resolution']}", err=True)
        raise typer.Exit(code=1)

    with connect(db_path) as conn:
        resolve_review(conn, review_id=review_id, resolution=action, resolved_by=resolved_by, note=note)
        # Re-record approval with the human's resolution as the FINAL state.
        from decimal import Decimal as _D
        if action == "approve":
            human_decision = ApprovalDecision(
                status="auto_approved",
                approver_role=entry["approver_role"],
                policy_id=entry["policy_id"],
                total_usd=_D(entry["total_usd"]),
                justification=f"human approval ({resolved_by}): {note or 'no note'}",
                escalations=[],
            )
        else:
            human_decision = ApprovalDecision(
                status="rejected",
                approver_role="none",
                policy_id=None,
                total_usd=_D(entry["total_usd"]),
                justification=f"human rejection ({resolved_by}): {note or 'no note'}",
                escalations=[],
            )
        record_approval(
            conn,
            invoice_number=entry["invoice_number"],
            vendor=entry["vendor"],
            status=human_decision.status,
            approver_role=human_decision.approver_role,
            policy_id=human_decision.policy_id,
            total_usd=human_decision.total_usd,
        )

    if action == "reject":
        typer.echo(f"Rejected review id={review_id} for invoice {entry['invoice_number']}.")
        return

    # Approve path: re-ingest + re-run pay with the overridden decision.
    source_path = Path(entry["source_path"]) if entry["source_path"] else None
    if source_path is None or not source_path.exists():
        typer.echo(
            f"Approval recorded but source invoice ({entry['source_path']}) is missing — "
            "cannot re-run pay phase. Ledger reflects approval; payment must be triggered manually.",
            err=True,
        )
        raise typer.Exit(code=1)

    load_dotenv()
    ing = ingest(source_path, allow_llm=True)
    invoice = ing.invoice
    with connect(db_path) as conn:
        vendor = lookup_vendor(conn, invoice.vendor)
        guard_report, _ = run_payment_guards(invoice, human_decision, vendor=vendor, conn=conn)
        record = pay_agent(invoice, human_decision, vendor=vendor)
        receipt_body = None
        receipt_path = None
        if record.status == "scheduled" and guard_report.approved:
            receipt_body = render_receipt(invoice, human_decision, record)
            written = write_receipt(record, receipt_body, directory=receipts)
            receipt_path = str(written)
            from dataclasses import replace as _replace
            record = _replace(record, receipt_path=receipt_path)
        elif not guard_report.approved:
            from dataclasses import replace as _replace
            record = _replace(
                record,
                status="failed",
                rail="none",
                notes=record.notes + [f"guard blocker: {b}" for b in guard_report.blockers],
            )
        record_payment(
            conn,
            reference=record.reference,
            invoice_number=invoice.invoice_number,
            vendor=invoice.vendor,
            rail=record.rail,
            status=record.status,
            amount_usd=record.amount_usd,
            currency_paid=record.currency_paid,
            amount_paid=record.amount_paid,
            scheduled_for=record.scheduled_for.isoformat() if record.scheduled_for else None,
            receipt_path=receipt_path,
        )
    typer.echo(f"Resolved review id={review_id}: payment {record.status}, rail={record.rail}, ref={record.reference}")
    if receipt_path:
        typer.echo(f"receipt: {receipt_path}")


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
