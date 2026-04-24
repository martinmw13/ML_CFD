"""Command-line entry point for mlcfd."""

from __future__ import annotations

from pathlib import Path

import click

from mlcfd.logging_config import LoggingLevel, configure_logging


@click.group()
@click.version_option(version="0.1.0", prog_name="mlcfd")
def main() -> None:
    """ML CFD: dimensionality reduction and related pipelines."""


@main.command("run")
@click.argument(
    "config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--log-level",
    default="INFO",
    show_default=True,
    help="Root logging level (DEBUG, INFO, WARNING, ...).",
)
def run_command(config: Path, log_level: str) -> None:
    """Run a reduction pipeline from a validated YAML configuration file."""
    level: LoggingLevel | str
    try:
        level = LoggingLevel[log_level.upper()]
    except KeyError:
        level = log_level
    configure_logging(level)
    from mlcfd.pipeline.runner import run_from_yaml

    run_from_yaml(config)


if __name__ == "__main__":
    main()
