"""CLI command: ingest files or directories."""

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from core.ingestion import SUPPORTED_EXTENSIONS, ingest_directory, ingest_file

console = Console()
app = typer.Typer()


@app.callback(invoke_without_command=True)
def ingest(
    path: str = typer.Argument(..., help="File or directory to ingest."),
    collection: str = typer.Option("default", "-c", "--collection", help="Target collection name."),
):
    """Ingest documents into a collection."""
    target = Path(path)

    if not target.exists():
        console.print(f"[red]Error:[/red] Path not found: {path}")
        raise typer.Exit(1)

    with console.status("[bold green]Ingesting..."):
        if target.is_dir():
            results = asyncio.run(ingest_directory(target, collection_name=collection))
        else:
            try:
                result = asyncio.run(ingest_file(target, collection_name=collection))
                results = [result]
            except ValueError as e:
                console.print(f"[red]Error:[/red] {e}")
                raise typer.Exit(1) from e

    if not results:
        console.print("[yellow]No supported files found.[/yellow]")
        console.print(f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}")
        raise typer.Exit(1)

    # Display results
    table = Table(title=f"Ingestion Results — collection: {collection}")
    table.add_column("File", style="cyan")
    table.add_column("Chunks", justify="right")
    table.add_column("Size", justify="right")
    table.add_column("Status", style="green")

    for r in results:
        if "error" in r:
            table.add_row(r["filename"], "-", "-", f"[red]{r['error']}[/red]")
        else:
            size = f"{r['file_size'] / 1024:.1f} KB"
            table.add_row(r["filename"], str(r["chunks"]), size, "OK")

    console.print(table)
