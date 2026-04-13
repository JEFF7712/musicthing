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
@click.option(
    "--checkpoint",
    default=None,
    type=click.Path(exists=True),
    help="Path to fine-tuned model checkpoint (.pt).",
)
@click.option("--no-clap", is_flag=True, help="Disable CLAP music embedding (brain-only fingerprints).")
def index(music_dir: str, data_dir: str, cache_dir: str, checkpoint: str | None, no_clap: bool) -> None:
    """Index a music library for neural similarity search."""
    from .encoder import NeuralEncoder
    from .indexer import build_index, find_audio_files

    files = find_audio_files(music_dir)
    if not files:
        console.print(f"[red]No audio files found in {music_dir}[/red]")
        sys.exit(1)

    console.print(f"Found [bold]{len(files)}[/bold] audio files in {music_dir}")
    if checkpoint:
        console.print(f"Using fine-tuned model: [bold]{checkpoint}[/bold]")
    if not no_clap:
        console.print("Using hybrid fingerprints: [bold]brain + CLAP music[/bold]")

    encoder = NeuralEncoder(cache_dir=cache_dir, checkpoint=checkpoint, use_clap=not no_clap)

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
@click.option(
    "--mode",
    type=click.Choice(["default", "sound", "emotion", "thought", "vibe"]),
    default="default",
    show_default=True,
    help="Search mode: weight auditory (sound), limbic (emotion), or prefrontal (thought) regions.",
)
@click.option("--data-dir", default="data", show_default=True)
@click.option("--cache-dir", default=".cache/tribev2", show_default=True)
@click.option(
    "--checkpoint",
    default=None,
    type=click.Path(exists=True),
    help="Path to fine-tuned model checkpoint (.pt).",
)
@click.option("--no-clap", is_flag=True, help="Disable CLAP music embedding (brain-only fingerprints).")
def query(seeds: tuple[str, ...], num: int, mode: str, data_dir: str, cache_dir: str, checkpoint: str | None, no_clap: bool) -> None:
    """Find songs with a similar neural vibe to the seed song(s)."""
    from .encoder import NeuralEncoder
    from .regions import PRESETS
    from .query import query_similar

    region_weights = PRESETS.get(mode) if mode != "default" else None
    encoder = NeuralEncoder(cache_dir=cache_dir, checkpoint=checkpoint, use_clap=not no_clap)

    mode_label = f" [dim]({mode} mode)[/dim]" if mode != "default" else ""
    console.print(
        f"Finding songs similar to [bold]{len(seeds)}[/bold] seed(s)…{mode_label}\n"
    )

    matches = query_similar(
        list(seeds), data_dir=data_dir, n=num, encoder=encoder,
        region_weights=region_weights,
    )

    if not matches:
        console.print("[yellow]No matches found.[/yellow]")
        return

    table = Table(title="Neural Vibe Matches")
    table.add_column("#", style="dim", width=3)
    table.add_column("Title", style="bold")
    table.add_column("Artist")
    table.add_column("Similarity", justify="right")
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
@click.option(
    "--runner",
    type=click.Choice(["docker", "apptainer", "native"]),
    default="docker",
    show_default=True,
    help="How to run fMRIPrep (docker/apptainer recommended).",
)
def preprocess(
    data_dir: str, n_cpus: int, fs_license: str | None, runner: str
) -> None:
    """Run fMRIPrep on the raw Nakai 2021 BIDS data.

    fMRIPrep requires FreeSurfer, FSL, ANTs, etc. The easiest way to
    run it is via Docker or Apptainer (Singularity), which bundle all
    dependencies.
    """
    import subprocess
    from pathlib import Path

    raw_dir = Path(data_dir).resolve() / "raw"
    deriv_dir = Path(data_dir).resolve() / "derivatives"
    deriv_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        console.print(f"[red]Raw data not found at {raw_dir}[/red]")
        console.print("Run 'neural-vibe download' first.")
        sys.exit(1)

    # fMRIPrep arguments (common to all runners)
    fmriprep_args = [
        str(raw_dir),
        str(deriv_dir),
        "participant",
        "--nprocs", str(n_cpus),
        "--output-spaces", "MNI152NLin2009cAsym",
        "--bold2anat-dof", "6",
    ]
    if fs_license:
        fmriprep_args.extend(["--fs-license-file", str(Path(fs_license).resolve())])
    else:
        fmriprep_args.append("--fs-no-reconall")

    if runner == "docker":
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{raw_dir}:{raw_dir}:ro",
            "-v", f"{deriv_dir}:{deriv_dir}",
        ]
        if fs_license:
            lic = Path(fs_license).resolve()
            cmd.extend(["-v", f"{lic}:{lic}:ro"])
        cmd.extend(["nipreps/fmriprep:latest", *fmriprep_args])
    elif runner == "apptainer":
        cmd = [
            "apptainer", "run", "--cleanenv",
            "-B", f"{raw_dir}:{raw_dir}:ro",
            "-B", f"{deriv_dir}:{deriv_dir}",
        ]
        if fs_license:
            lic = Path(fs_license).resolve()
            cmd.extend(["-B", f"{lic}:{lic}:ro"])
        cmd.extend(["docker://nipreps/fmriprep:latest", *fmriprep_args])
    else:
        cmd = ["fmriprep", *fmriprep_args]

    console.print(f"Running fMRIPrep via [bold]{runner}[/bold] with {n_cpus} CPUs…")
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


@main.command("prepare-stimuli")
@click.argument("data_dir", default="data/nakai2021", type=click.Path(exists=True))
def prepare_stimuli(data_dir: str) -> None:
    """Cut GTZAN audio clips to match the Nakai 2021 experiment timings.

    Requires GTZAN dataset at {DATA_DIR}/stimuli/gtzan/genres/.
    """
    from pathlib import Path

    from .stimuli import prepare_clips

    console.print(
        f"Preparing audio clips from GTZAN for [bold]{data_dir}[/bold]…\n"
    )

    clips_dir = prepare_clips(data_dir=Path(data_dir))

    console.print(f"\n[green]Done![/green] Clips saved to [bold]{clips_dir}[/bold]")
    console.print(
        "\nNext step: fine-tune TRIBE v2:\n"
        f"  neural-vibe finetune {data_dir}"
    )


@main.command()
@click.argument("data_dir", default="data/nakai2021", type=click.Path(exists=True))
@click.option("--extra-data", multiple=True, type=click.Path(exists=True), help="Additional dataset directories to include.")
@click.option("--epochs", default=20, show_default=True)
@click.option("--lr", default=1e-5, show_default=True, help="Learning rate.")
@click.option("--batch-size", default=4, show_default=True)
@click.option("--patience", default=5, show_default=True, help="Early stopping patience (epochs).")
@click.option(
    "--freeze",
    type=click.Choice(["head-only", "default", "none"]),
    default="default",
    show_default=True,
    help="Freezing strategy: head-only (freeze encoders+transformer), default (freeze encoders), none.",
)
@click.option("--output-dir", default="checkpoints/music-finetuned", show_default=True)
@click.option("--cache-dir", default=".cache/tribev2", show_default=True)
def finetune(
    data_dir: str,
    extra_data: tuple[str, ...],
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
    freeze: str,
    output_dir: str,
    cache_dir: str,
) -> None:
    """Fine-tune TRIBE v2 on music fMRI data."""
    from .finetune import FinetuneConfig, finetune as run_finetune

    freeze_transformer = freeze == "head-only"
    freeze_encoders = freeze != "none"

    config = FinetuneConfig(
        data_dir=data_dir,
        extra_data_dirs=list(extra_data) if extra_data else None,
        cache_dir=cache_dir,
        output_dir=output_dir,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        patience=patience,
        freeze_encoders=freeze_encoders,
        freeze_transformer=freeze_transformer,
    )

    all_dirs = [data_dir] + list(extra_data)
    console.print("[bold]Fine-tuning TRIBE v2 on music fMRI data[/bold]")
    console.print(f"  Data:     {', '.join(all_dirs)}")
    console.print(f"  Epochs:   {epochs} (early stop patience={patience})")
    console.print(f"  LR:       {lr} (cosine annealing with warmup)")
    console.print(f"  Loss:     Pearson correlation")
    console.print(f"  Freeze:   {freeze}")
    console.print(f"  Output:   {output_dir}\n")

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
    "--subjects",
    default=None,
    help="Comma-separated subjects to include (e.g. 'sub-001,sub-002'). Default: all.",
)
@click.option(
    "--bold-only",
    is_flag=True,
    default=True,
    show_default=True,
    help="Only include preprocessed BOLD + events (skip masks/confounds/transforms).",
)
def package(
    data_dir: str, output: str | None, subjects: str | None, bold_only: bool
) -> None:
    """Package preprocessed data for upload to a cloud GPU instance."""
    import tarfile
    from pathlib import Path

    data_path = Path(data_dir)
    deriv_dir = data_path / "derivatives"
    raw_dir = data_path / "raw"

    if not deriv_dir.exists():
        console.print("[red]No derivatives/ found — run preprocessing first.[/red]")
        sys.exit(1)

    # Filter subjects
    if subjects:
        subject_list = [s.strip() for s in subjects.split(",")]
    else:
        subject_list = sorted(
            d.name for d in deriv_dir.iterdir()
            if d.is_dir() and d.name.startswith("sub-")
        )

    if output is None:
        output = f"{data_path.name}-upload.tar.gz"

    console.print(f"Packaging [bold]{len(subject_list)}[/bold] subject(s) for cloud upload…")
    console.print(f"  Subjects: {', '.join(subject_list)}\n")

    total_size = 0
    with tarfile.open(output, "w:gz") as tar:
        # Add preprocessed BOLD files per subject
        for subj in subject_list:
            subj_deriv = deriv_dir / subj / "func"
            if not subj_deriv.exists():
                console.print(f"  [yellow]Skipping {subj} — no derivatives[/yellow]")
                continue

            for f in sorted(subj_deriv.iterdir()):
                if not f.is_file():
                    continue
                # If bold-only, skip everything except preproc BOLD and its JSON
                if bold_only and "preproc_bold" not in f.name:
                    continue
                arcname = f"{data_path.name}/derivatives/{subj}/func/{f.name}"
                tar.add(str(f), arcname=arcname)
                total_size += f.stat().st_size
                console.print(f"  [dim]{f.name} ({f.stat().st_size / 1024**2:.0f} MB)[/dim]")

            # Add events.tsv from raw
            subj_raw = raw_dir / subj / "func"
            if subj_raw.exists():
                for ef in sorted(subj_raw.glob("*_events.tsv")):
                    arcname = f"{data_path.name}/raw/{subj}/func/{ef.name}"
                    tar.add(str(ef), arcname=arcname)
                    total_size += ef.stat().st_size

        # Add dataset_description.json if present
        desc = deriv_dir / "dataset_description.json"
        if desc.exists():
            tar.add(str(desc), arcname=f"{data_path.name}/derivatives/dataset_description.json")

    archive_size = Path(output).stat().st_size
    console.print(f"\n  Source data:  {total_size / 1024**3:.1f} GB")
    console.print(f"  Archive size: {archive_size / 1024**3:.1f} GB")
    console.print(f"\n[green]Saved to {output}[/green]")
    console.print(
        "\nUpload to your cloud instance and extract:\n"
        f"  scp {output} gpu-instance:~/\n"
        f"  ssh gpu-instance 'tar xzf {output}'\n"
        f"  ssh gpu-instance 'neural-vibe finetune ~/{data_path.name}'"
    )
