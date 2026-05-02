from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv

from galatiq.agents.ingestion import ingest
from galatiq.db import DEFAULT_DB_PATH, SEED_INVENTORY, connect, init_db, list_items

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


@app.command("db-init")
def db_init_cmd(
    db_path: Path = typer.Option(DEFAULT_DB_PATH, "--db", help="Path to the SQLite file"),
) -> None:
    """Create the inventory DB (if absent), apply schema, and insert seed rows."""
    path = init_db(db_path)
    with connect(path) as conn:
        rows = list_items(conn)
    typer.echo(f"db          : {path}")
    typer.echo(f"seed defined: {len(SEED_INVENTORY)} items")
    typer.echo("inventory   :")
    for item, stock in rows:
        typer.echo(f"  - {item:<12} {stock}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
