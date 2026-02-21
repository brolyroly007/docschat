"""CLI command: interactive chat REPL and single-question mode."""

import asyncio

import typer
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from core.rag import RAGOrchestrator

console = Console()
app = typer.Typer()


async def _run_chat_stream(rag: RAGOrchestrator, question: str) -> str:
    """Stream a response with Rich Live rendering."""
    full_response = ""
    sources = []

    with Live("", console=console, refresh_per_second=10) as live:
        async for event in rag.stream(question):
            if event["type"] == "sources":
                sources = event["data"]
            elif event["type"] == "token":
                full_response += event["data"]
                live.update(Markdown(full_response))
            elif event["type"] == "done":
                pass

    # Show sources
    if sources:
        source_text = ", ".join(f"{s['source']} (chunk {s['chunk_index']})" for s in sources[:3])
        console.print(f"\n[dim]Sources: {source_text}[/dim]")

    return full_response


async def _single_question(question: str, collection: str, provider: str | None):
    """Answer a single question and exit."""
    rag = RAGOrchestrator(collection=collection, provider_name=provider)
    await _run_chat_stream(rag, question)


async def _interactive_repl(collection: str, provider: str | None):
    """Interactive chat REPL."""
    console.print(
        Panel(
            "[bold]DocsChat[/bold] — Interactive Mode\n"
            f"Collection: [cyan]{collection}[/cyan] | "
            f"Provider: [cyan]{provider or 'default'}[/cyan]\n"
            "Type [bold]quit[/bold] or [bold]exit[/bold] to leave.",
            title="DocsChat",
        )
    )

    rag = RAGOrchestrator(collection=collection, provider_name=provider)

    while True:
        try:
            question = Prompt.ask("\n[bold green]You[/bold green]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not question.strip():
            continue
        if question.strip().lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break

        console.print()
        try:
            await _run_chat_stream(rag, question)
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")


@app.callback(invoke_without_command=True)
def chat(
    question: str | None = typer.Option(
        None, "-q", "--question", help="Single question (non-interactive)."
    ),
    collection: str = typer.Option(
        "default", "-c", "--collection", help="Collection to chat with."
    ),
    provider: str | None = typer.Option(
        None, "-p", "--provider", help="LLM provider (openai, gemini, ollama)."
    ),
):
    """Chat with your documents."""
    if question:
        asyncio.run(_single_question(question, collection, provider))
    else:
        asyncio.run(_interactive_repl(collection, provider))
