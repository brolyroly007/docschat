"""CLI command: manage document collections."""

import asyncio

import typer
from rich.console import Console
from rich.table import Table

from core.ingestion import get_chroma_client
from database import get_db
from database.repositories import delete_documents_by_collection

console = Console()
app = typer.Typer()


@app.callback(invoke_without_command=True)
def list_cols(
    ctx: typer.Context,
):
    """List all collections."""
    if ctx.invoked_subcommand is not None:
        return

    client = get_chroma_client()
    collections = client.list_collections()

    if not collections:
        console.print("[yellow]No collections found.[/yellow]")
        console.print("Use [bold]docschat ingest[/bold] to add documents.")
        return

    table = Table(title="Collections")
    table.add_column("Name", style="cyan")
    table.add_column("Chunks", justify="right")

    for col in collections:
        table.add_row(col.name, str(col.count()))

    console.print(table)


@app.command()
def delete(
    name: str = typer.Argument(..., help="Collection name to delete."),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation."),
):
    """Delete a collection."""
    if not force:
        confirm = typer.confirm(f"Delete collection '{name}' and all its documents?")
        if not confirm:
            console.print("[dim]Cancelled.[/dim]")
            raise typer.Exit()

    client = get_chroma_client()
    try:
        client.delete_collection(name=name)
    except Exception as exc:
        console.print(f"[red]Collection '{name}' not found.[/red]")
        raise typer.Exit(1) from exc

    deleted = asyncio.run(_delete_docs(name))
    console.print(
        f"[green]Deleted[/green] collection [cyan]{name}[/cyan] "
        f"({deleted} document records removed)"
    )


async def _delete_docs(name: str) -> int:
    db = await get_db()
    try:
        return await delete_documents_by_collection(db, name)
    finally:
        await db.close()
