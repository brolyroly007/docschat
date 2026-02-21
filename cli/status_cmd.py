"""CLI command: system status."""

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config import settings
from core.ingestion import get_chroma_client
from database import get_db
from database.connection import DB_PATH
from providers import list_providers

console = Console()
app = typer.Typer()


@app.callback(invoke_without_command=True)
def status():
    """Show system status: database, vector store, providers."""
    asyncio.run(_show_status())


async def _show_status():
    # Database
    db_exists = DB_PATH.exists()
    db_size = f"{DB_PATH.stat().st_size / 1024:.1f} KB" if db_exists else "N/A"

    doc_count = 0
    conv_count = 0
    if db_exists:
        db = await get_db()
        try:
            cursor = await db.execute("SELECT COUNT(*) FROM documents")
            row = await cursor.fetchone()
            doc_count = row[0]
            cursor = await db.execute("SELECT COUNT(*) FROM conversations")
            row = await cursor.fetchone()
            conv_count = row[0]
        except Exception:
            pass
        finally:
            await db.close()

    # Vector store
    chroma_dir = Path(settings.chroma_persist_dir)
    chroma_exists = chroma_dir.exists()
    collection_count = 0
    total_chunks = 0
    if chroma_exists:
        try:
            client = get_chroma_client()
            collections = client.list_collections()
            collection_count = len(collections)
            total_chunks = sum(c.count() for c in collections)
        except Exception:
            pass

    # Display
    console.print(Panel("[bold]DocsChat Status[/bold]", expand=False))

    # Database table
    db_table = Table(title="Database", show_header=False)
    db_table.add_column("Key", style="cyan")
    db_table.add_column("Value")
    db_table.add_row("Path", str(DB_PATH))
    db_table.add_row("Exists", "Yes" if db_exists else "No")
    db_table.add_row("Size", db_size)
    db_table.add_row("Documents", str(doc_count))
    db_table.add_row("Conversations", str(conv_count))
    console.print(db_table)

    # Vector store table
    vs_table = Table(title="Vector Store (ChromaDB)", show_header=False)
    vs_table.add_column("Key", style="cyan")
    vs_table.add_column("Value")
    vs_table.add_row("Path", str(chroma_dir))
    vs_table.add_row("Exists", "Yes" if chroma_exists else "No")
    vs_table.add_row("Collections", str(collection_count))
    vs_table.add_row("Total Chunks", str(total_chunks))
    console.print(vs_table)

    # Providers table
    providers = list_providers()
    prov_table = Table(title="LLM Providers")
    prov_table.add_column("Name", style="cyan")
    prov_table.add_column("Available")
    prov_table.add_column("Default")
    for p in providers:
        is_default = "Yes" if p["name"] == settings.default_provider else ""
        available = "[green]Yes[/green]" if p["available"] else "[red]No[/red]"
        prov_table.add_row(p["name"], available, is_default)
    console.print(prov_table)

    # Config
    cfg_table = Table(title="Configuration", show_header=False)
    cfg_table.add_column("Key", style="cyan")
    cfg_table.add_column("Value")
    cfg_table.add_row("Embedding Provider", settings.embedding_provider)
    cfg_table.add_row("Chunk Size", str(settings.chunk_size))
    cfg_table.add_row("Chunk Overlap", str(settings.chunk_overlap))
    cfg_table.add_row("Top K", str(settings.top_k))
    console.print(cfg_table)
