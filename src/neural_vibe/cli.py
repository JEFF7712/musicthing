"""Neural Vibe CLI — music discovery via brain response similarity."""

from __future__ import annotations

import logging
import sys

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

console = Console()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_time=False, show_path=False)],
    )


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def main(verbose: bool) -> None:
    """Neural Vibe — discover music by how it feels, not how it sounds."""
    _setup_logging(verbose)


@main.command()
@click.argument("music_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--data-dir",
    default="data",
    show_default=True,
    help="Where to store the index.",
)
@click.option(
    "--cache-dir",
    default=".cache/tribev2",
    show_default=True,
    help="Model cache directory.",
)
def index(music_dir: str, data_dir: str, cache_dir: str) -> None:
    """Index a music library for neural similarity search."""
    from .encoder import NeuralEncoder
    from .indexer import build_index, find_audio_files

    files = find_audio_files(music_dir)
    if not files:
        console.print(f"[red]No audio files found in {music_dir}[/red]")
        sys.exit(1)

    console.print(f"Found [bold]{len(files)}[/bold] audio files in {music_dir}")

    encoder = NeuralEncoder(cache_dir=cache_dir)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Indexing songs…", total=0)

        def on_progress(current: int, total: int, filename: str) -> None:
            progress.update(task, total=total, completed=current, description=filename)

        idx, meta = build_index(
            music_dir, data_dir=data_dir, encoder=encoder, on_progress=on_progress
        )
        progress.update(task, completed=progress.tasks[0].total)

    console.print(
        f"\n[green]Done![/green] Index contains [bold]{len(meta)}[/bold] songs."
    )


@main.command()
@click.argument("seeds", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("-n", "--num", default=10, show_default=True, help="Number of results.")
@click.option("--data-dir", default="data", show_default=True)
@click.option("--cache-dir", default=".cache/tribev2", show_default=True)
def query(seeds: tuple[str, ...], num: int, data_dir: str, cache_dir: str) -> None:
    """Find songs with a similar neural vibe to the seed song(s)."""
    from .encoder import NeuralEncoder
    from .query import query_similar

    encoder = NeuralEncoder(cache_dir=cache_dir)

    console.print(
        f"Finding songs similar to [bold]{len(seeds)}[/bold] seed(s)…\n"
    )

    matches = query_similar(
        list(seeds), data_dir=data_dir, n=num, encoder=encoder
    )

    if not matches:
        console.print("[yellow]No matches found.[/yellow]")
        return

    table = Table(title="Neural Vibe Matches")
    table.add_column("#", style="dim", width=3)
    table.add_column("Title", style="bold")
    table.add_column("Artist")
    table.add_column("Distance", justify="right")
    table.add_column("Path", style="dim")

    for m in matches:
        table.add_row(
            str(m.rank),
            m.title,
            m.artist,
            f"{m.distance:.2f}",
            m.path,
        )

    console.print(table)


@main.command()
@click.option("--data-dir", default="data", show_default=True)
def info(data_dir: str) -> None:
    """Show index statistics."""
    from .indexer import load_index

    index, metadata = load_index(data_dir)

    if index is None:
        console.print(f"[yellow]No index found in {data_dir}.[/yellow]")
        return

    artists = {m.get("artist", "Unknown") for m in metadata}

    console.print(f"[bold]Index:[/bold] {data_dir}")
    console.print(f"  Songs:      {index.ntotal}")
    console.print(f"  Artists:    {len(artists)}")
    console.print(f"  Dimensions: {index.d}")
