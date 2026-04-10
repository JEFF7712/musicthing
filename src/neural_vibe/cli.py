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


# --- Fine-tuning commands ---


@main.command()
@click.argument(
    "data_dir",
    default="data/nakai2021",
    type=click.Path(),
)
def download(data_dir: str) -> None:
    """Download the Nakai 2021 Music Genre fMRI dataset from OpenNeuro."""
    from pathlib import Path

    from .studies.nakai2021 import Nakai2021Bold

    data_path = Path(data_dir)
    console.print(f"Downloading Nakai 2021 dataset to [bold]{data_path}[/bold]…")

    study = Nakai2021Bold(path=data_path)
    study.download()

    console.print("[green]Download complete![/green]")
    console.print(
        "\nNext step: preprocess with fMRIPrep:\n"
        f"  neural-vibe preprocess {data_dir}"
    )


@main.command()
@click.argument("data_dir", default="data/nakai2021", type=click.Path(exists=True))
@click.option("--n-cpus", default=4, show_default=True, help="CPUs for fMRIPrep.")
@click.option("--fs-license", type=click.Path(exists=True), help="FreeSurfer license file.")
def preprocess(data_dir: str, n_cpus: int, fs_license: str | None) -> None:
    """Run fMRIPrep on the raw Nakai 2021 BIDS data."""
    import subprocess
    from pathlib import Path

    raw_dir = Path(data_dir) / "raw"
    deriv_dir = Path(data_dir) / "derivatives"
    deriv_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        console.print(f"[red]Raw data not found at {raw_dir}[/red]")
        console.print("Run 'neural-vibe download' first.")
        sys.exit(1)

    cmd = [
        "fmriprep",
        str(raw_dir),
        str(deriv_dir),
        "participant",
        "--nprocs", str(n_cpus),
        "--output-spaces", "MNI152NLin2009cAsym",
        "--bold2t1w-dof", "6",
    ]
    if fs_license:
        cmd.extend(["--fs-license-file", fs_license])
    else:
        # Skip FreeSurfer surface reconstruction (faster)
        cmd.append("--fs-no-reconall")

    console.print(f"Running fMRIPrep with {n_cpus} CPUs…")
    console.print(f"  [dim]{' '.join(cmd)}[/dim]\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        console.print("[red]fMRIPrep failed.[/red]")
        sys.exit(1)

    console.print("[green]Preprocessing complete![/green]")
    console.print(
        f"\nNext step: fine-tune TRIBE v2:\n"
        f"  neural-vibe finetune {data_dir}"
    )


@main.command()
@click.argument("data_dir", default="data/nakai2021", type=click.Path(exists=True))
@click.option("--epochs", default=10, show_default=True)
@click.option("--lr", default=1e-5, show_default=True, help="Learning rate.")
@click.option("--batch-size", default=4, show_default=True)
@click.option("--output-dir", default="checkpoints/music-finetuned", show_default=True)
@click.option("--cache-dir", default=".cache/tribev2", show_default=True)
def finetune(
    data_dir: str,
    epochs: int,
    lr: float,
    batch_size: int,
    output_dir: str,
    cache_dir: str,
) -> None:
    """Fine-tune TRIBE v2 on the Nakai 2021 music fMRI data."""
    from .finetune import FinetuneConfig, finetune as run_finetune

    config = FinetuneConfig(
        data_dir=data_dir,
        cache_dir=cache_dir,
        output_dir=output_dir,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
    )

    console.print("[bold]Fine-tuning TRIBE v2 on music fMRI data[/bold]")
    console.print(f"  Data:   {data_dir}")
    console.print(f"  Epochs: {epochs}")
    console.print(f"  LR:     {lr}")
    console.print(f"  Output: {output_dir}\n")

    best_ckpt = run_finetune(config)

    console.print(f"\n[green]Done![/green] Best checkpoint: [bold]{best_ckpt}[/bold]")
    console.print(
        "\nTo use the fine-tuned model for indexing:\n"
        f"  neural-vibe index ~/Music --checkpoint {best_ckpt}"
    )


@main.command()
@click.argument("data_dir", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output",
    default=None,
    help="Output archive path. Defaults to <data_dir>-upload.tar.gz",
)
@click.option(
    "--include-raw",
    is_flag=True,
    help="Include raw BIDS data (needed if cloud hasn't downloaded it).",
)
def package(data_dir: str, output: str | None, include_raw: bool) -> None:
    """Package preprocessed data for upload to a cloud GPU instance."""
    import tarfile
    from pathlib import Path

    data_path = Path(data_dir)
    deriv_dir = data_path / "derivatives"
    raw_dir = data_path / "raw"

    if not deriv_dir.exists():
        console.print("[red]No derivatives/ found — run preprocessing first.[/red]")
        sys.exit(1)

    if output is None:
        output = f"{data_path.name}-upload.tar.gz"

    console.print(f"Packaging [bold]{data_dir}[/bold] for cloud upload…\n")

    # Collect what to include
    include_dirs = [("derivatives", deriv_dir)]

    # Always include stimuli (audio files needed for fine-tuning)
    stimuli_dir = raw_dir / "stimuli"
    if stimuli_dir.exists():
        include_dirs.append(("raw/stimuli", stimuli_dir))

    # Include events.tsv files (small, needed for timeline loading)
    events_files = list(raw_dir.rglob("*_events.tsv")) if raw_dir.exists() else []

    if include_raw:
        include_dirs = [("raw", raw_dir), ("derivatives", deriv_dir)]
        events_files = []  # already included via raw/

    total_size = 0
    with tarfile.open(output, "w:gz") as tar:
        for arcname_prefix, dir_path in include_dirs:
            for f in sorted(dir_path.rglob("*")):
                if not f.is_file():
                    continue
                arcname = f"{data_path.name}/{arcname_prefix}/{f.relative_to(dir_path)}"
                tar.add(str(f), arcname=arcname)
                total_size += f.stat().st_size

        # Add individual events.tsv files
        for events_f in events_files:
            rel = events_f.relative_to(raw_dir)
            arcname = f"{data_path.name}/raw/{rel}"
            tar.add(str(events_f), arcname=arcname)
            total_size += events_f.stat().st_size

    archive_size = Path(output).stat().st_size
    console.print(f"  Source data:  {total_size / 1024**3:.1f} GB")
    console.print(f"  Archive size: {archive_size / 1024**3:.1f} GB")
    console.print(f"\n[green]Saved to {output}[/green]")
    console.print(
        "\nUpload to your cloud instance and extract:\n"
        f"  scp {output} gpu-instance:~/\n"
        f"  ssh gpu-instance 'tar xzf {output}'\n"
        f"  ssh gpu-instance 'neural-vibe finetune ~/{data_path.name}'"
    )
