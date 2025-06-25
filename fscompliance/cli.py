"""
FSCompliance CLI entry point.

This module provides the command-line interface for the FSCompliance
MCP service for financial regulatory compliance.
"""

import asyncio
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .server import FSComplianceServer, get_mcp_server_config

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
    transport: str = typer.Option("stdio", help="Transport type (stdio, sse, http)"),
    host: str = typer.Option("localhost", help="Host to bind to"),
    port: int = typer.Option(8000, help="Port to bind to"),
    name: str = typer.Option("FSCompliance", help="Server name"),
):
    """Start the FSCompliance MCP server."""
    console.print(Panel(
        f"[bold blue]Starting FSCompliance MCP Server[/bold blue]\n\n"
        f"Transport: {transport}\n"
        f"Host: {host}\n"
        f"Port: {port}\n"
        f"Name: {name}",
        title="Server Configuration"
    ))
    
    async def run_server():
        server = FSComplianceServer(name=name)
        await server.run(transport=transport)
    
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Server error: {e}[/red]")


@app.command()
def status():
    """Show FSCompliance MCP server status and configuration."""
    config = get_mcp_server_config()
    
    # Create configuration table
    table = Table(title="FSCompliance MCP Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Server Name", config.name)
    table.add_row("Version", config.version)
    table.add_row("Transport", config.transport.value)
    table.add_row("Host:Port", f"{config.host}:{config.port}")
    table.add_row("Default Framework", config.default_regulatory_framework)
    table.add_row("Memory Enabled", str(config.enable_memory))
    table.add_row("Data Anonymization", str(config.anonymize_data))
    table.add_row("Max Requests", str(config.max_analysis_requests))
    
    console.print(table)
    
    # Show capabilities
    caps_table = Table(title="Server Capabilities")
    caps_table.add_column("Capability", style="cyan")
    caps_table.add_column("Enabled", style="green")
    
    caps_table.add_row("Tools", str(config.capabilities.tools))
    caps_table.add_row("Resources", str(config.capabilities.resources))
    caps_table.add_row("Prompts", str(config.capabilities.prompts))
    caps_table.add_row("Logging", str(config.capabilities.logging))
    
    console.print(caps_table)


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