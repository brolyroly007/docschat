"""DocsChat CLI — root Typer application."""

import typer
from rich.console import Console

from cli.chat_cmd import app as chat_app
from cli.collections_cmd import app as collections_app
from cli.ingest_cmd import app as ingest_app
from cli.status_cmd import app as status_app

console = Console()

app = typer.Typer(
    name="docschat",
    help="DocsChat — Chat with your documents using RAG and any LLM.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


def version_callback(value: bool):
    if value:
        console.print("[bold]docschat[/bold] version [green]1.0.0[/green]")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
):
    """DocsChat — Chat with your documents using RAG and any LLM."""


app.add_typer(chat_app, name="chat", help="Chat with your documents.")
app.add_typer(ingest_app, name="ingest", help="Ingest documents into a collection.")
app.add_typer(collections_app, name="collections", help="Manage document collections.")
app.add_typer(status_app, name="status", help="Show system status.")


def run():
    """Entry point for pyproject.toml console_scripts."""
    app()


if __name__ == "__main__":
    run()
