"""Command-line entry point for mlcfd (subcommands added in later steps)."""

from __future__ import annotations

import click


@click.group()
def main() -> None:
    """ML CFD: dimensionality reduction and related pipelines."""


if __name__ == "__main__":
    main()
