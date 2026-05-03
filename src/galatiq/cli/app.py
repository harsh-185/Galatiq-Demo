from __future__ import annotations

import os
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv

from galatiq.agents._walkthrough import (
    StageEvent,
    set_stream_callback,
    summary_stats,
)
from galatiq.agents.approval import ApprovalDecision
from galatiq.agents.ingestion import ingest
from galatiq.agents.payment import pay as pay_agent
from galatiq.agents.payment_guards import run_payment_guards
from galatiq.agents.pipeline import run_pipeline
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
    record_payment,
    resolve_review,
)
from galatiq.payments.receipt import render_receipt, write_receipt

app = typer.Typer(add_completion=False, help="Galatiq invoice automation CLI")


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
    show_agents: bool = typer.Option(
        False,
        "--show-agents/--no-show-agents",
        help="Demo mode: bypass the deterministic skip-gates so every LLM agent runs (pre-approval screener with tools, full council, payment_review in full tool mode). Slower but shows the complete multi-agent flow on every invoice.",
    ),
) -> None:
    """Run the full pipeline (ingest → fraud-screen → validate → … → pay) via LangGraph."""
    load_dotenv()
    if llm_agents is not None:
        os.environ["GALATIQ_LLM_AGENTS"] = "1" if llm_agents else "0"
    if show_agents:
        os.environ["GALATIQ_SHOW_AGENTS"] = "1"
        # Force LLM on too — show-agents without LLM is incoherent.
        os.environ["GALATIQ_LLM_AGENTS"] = "1"
    if not db_path.exists():
        typer.echo(f"FAILED: reference DB not found at {db_path}; run `db-init` first", err=True)
        raise typer.Exit(code=2)
    # ── Live streaming printer ──────────────────────────────────────────────
    typer.echo("")
    typer.echo(f"📄 Invoice: {invoice_path.name}")
    typer.echo("━" * 60)
    typer.echo("Pipeline walkthrough (streaming)")
    typer.echo("━" * 60)

    stage_idx = {"n": 0}

    def _stream(event: StageEvent, phase: str) -> None:
        if phase == "started":
            stage_idx["n"] += 1
            sys.stdout.write(f"  ▶ [{stage_idx['n']}] {event.name} …\n")
            sys.stdout.flush()
            return
        # phase == "completed"
        marker = {"completed": "✓", "skipped": "⊘", "failed": "✗"}[event.status]
        head = (
            f"    [{marker}] {event.name:<22} {event.duration_ms:>7.1f} ms"
        )
        if event.llm_calls:
            head += f"  ·  {event.llm_calls} LLM call{'s' if event.llm_calls != 1 else ''}"
        if event.tools_used:
            head += f"  ·  {len(event.tools_used)} tool{'s' if len(event.tools_used) != 1 else ''}"
        sys.stdout.write(head + "\n")
        if event.summary:
            sys.stdout.write(f"        {event.summary}\n")
        for d in event.details[:5]:
            sys.stdout.write(f"        {d}\n")
        if event.tools_used:
            for t in event.tools_used[:3]:
                sys.stdout.write(f"        tool → {t}\n")
            if len(event.tools_used) > 3:
                sys.stdout.write(f"        tool → … and {len(event.tools_used) - 3} more\n")
        sys.stdout.flush()

    set_stream_callback(_stream)
    try:
        state = run_pipeline(invoice_path, db_path=db_path, receipt_dir=receipts)
    finally:
        set_stream_callback(None)

    if state.get("errors"):
        for err in state["errors"]:
            typer.echo(f"ERROR: {err}", err=True)
        raise typer.Exit(code=1)

    decision = state["decision"]
    payment = state["payment"]
    walkthrough = state.get("walkthrough") or []

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
