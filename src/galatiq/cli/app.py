from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv

from galatiq.agents.ingestion import ingest
from galatiq.agents.validation import validate
from galatiq.db import (
    DEFAULT_DB_PATH,
    SEED_INVENTORY,
    SEED_VENDORS,
    connect,
    init_db,
    list_inventory,
    list_vendors,
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
    typer.echo(f"db          : {path}  ({'fresh' if fresh else 'kept'})")
    typer.echo(f"seed defined: {len(SEED_INVENTORY)} items, {len(SEED_VENDORS)} vendors")
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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
