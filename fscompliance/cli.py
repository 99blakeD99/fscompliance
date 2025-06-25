"""
FSCompliance CLI entry point.

This module provides the command-line interface for the FSCompliance
MCP service for financial regulatory compliance.
"""

import typer
from rich.console import Console
from rich.panel import Panel

app = typer.Typer(
    name="fscompliance",
    help="Open-source MCP service for financial regulatory compliance",
    add_completion=False,
)
console = Console()


@app.command()
def version():
    """Show FSCompliance version information."""
    console.print(Panel(
        "[bold blue]FSCompliance[/bold blue] v0.1.0\n"
        "Open-source MCP service for financial regulatory compliance\n\n"
        "[dim]Repository:[/dim] https://github.com/99blakeD99/fscompliance\n"
        "[dim]License:[/dim] MIT",
        title="Version Info"
    ))


@app.command()
def server(
    host: str = typer.Option("localhost", help="Host to bind to"),
    port: int = typer.Option(8000, help="Port to bind to"),
    reload: bool = typer.Option(False, help="Enable auto-reload"),
):
    """Start the FSCompliance MCP server."""
    console.print(f"Starting FSCompliance MCP server on {host}:{port}")
    
    if reload:
        console.print("[yellow]Auto-reload enabled for development[/yellow]")
    
    # TODO: Import and start the actual MCP server
    console.print("[red]MCP server not yet implemented[/red]")


@app.command()
def init():
    """Initialize FSCompliance configuration."""
    console.print("Initializing FSCompliance configuration...")
    # TODO: Create configuration files and setup
    console.print("[red]Configuration initialization not yet implemented[/red]")


def main():
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()