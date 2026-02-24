import argparse
import logging
import sys

from function_app import get_config, perform_channel_refresh, stop_channel
from rich.console import Console
from rich.logging import RichHandler

console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, console=console)],
)
# Mute overly verbose python-oauth2 google client logging
logging.getLogger("googleapiclient.discovery_cache").setLevel(logging.ERROR)


def check_config(config):
    if (
        not config.get("target_dir")
        or not config.get("webhook_token")
        or not config.get("webhook_url")
    ):
        console.print(
            "[red]Missing required configuration for channel operations.[/red]"
        )
        console.print(
            "Please ensure [bold]target_dir[/bold], [bold]webhook_token[/bold], and [bold]webhook_url[/bold] are set in config.toml or Environment vars."
        )
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Manage Google Drive webhook channels for Azure Functions."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Refresh command
    refresh_parser = subparsers.add_parser(
        "refresh",
        help="Refresh the push notification channel (stops old, creates new).",
    )

    # Stop command
    stop_parser = subparsers.add_parser(
        "stop", help="Stop the currently active push notification channel."
    )

    args = parser.parse_args()
    config = get_config()
    check_config(config)

    if args.command == "refresh":
        console.print("[bold blue]Starting Channel Refresh...[/bold blue]")
        try:
            perform_channel_refresh(config)
            console.print("[bold green]Refresh completed successfully![/bold green]")
        except Exception as e:
            console.print(f"[bold red]Failed to refresh channel:[/bold red] {e}")
            sys.exit(1)

    elif args.command == "stop":
        console.print("[bold yellow]Stopping Active Channel...[/bold yellow]")
        try:
            stopped = stop_channel(config)
            if stopped:
                console.print("[bold green]Channel stopped successfully![/bold green]")
            else:
                console.print("[dim]No channel needed stopping.[/dim]")
        except Exception as e:
            console.print(f"[bold red]Failed to stop channel:[/bold red] {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
