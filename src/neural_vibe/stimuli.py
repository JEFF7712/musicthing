"""Prepare GTZAN audio stimuli for Nakai 2021 fine-tuning.

The Nakai 2021 fMRI study used 15-second clips from the GTZAN Music Genre
Collection. The events.tsv files specify which portion of each track was
played. This module cuts the relevant segments so they can be used as
input to the TRIBE v2 audio encoder.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile

log = logging.getLogger(__name__)


def _clip_id(genre: str, track: int, start: float, end: float) -> str:
    """Unique filename-safe identifier for a clip."""
    genre = genre.strip("'\"").lower()
    return f"{genre}_{track:05d}_{start:.2f}_{end:.2f}"


def _find_gtzan_track(genres_dir: Path, genre: str, track: int) -> Path:
    """Locate a GTZAN track, trying .wav then .au extensions."""
    genre = genre.strip("'\"").lower()
    for ext in (".wav", ".au"):
        p = genres_dir / genre / f"{genre}.{track:05d}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(
        f"GTZAN track not found: {genres_dir / genre / f'{genre}.{track:05d}.wav'}"
    )


def prepare_clips(data_dir: Path) -> Path:
    """Cut GTZAN audio clips to match Nakai 2021 events.tsv timings.

    Expects GTZAN at ``{data_dir}/stimuli/gtzan/genres/``.
    Writes clips to ``{data_dir}/stimuli/clips/``.

    Returns the clips directory.
    """
    genres_dir = data_dir / "stimuli" / "gtzan" / "genres"
    clips_dir = data_dir / "stimuli" / "clips"
    raw_dir = data_dir / "raw"

    if not genres_dir.exists():
        raise FileNotFoundError(
            f"GTZAN genres not found at {genres_dir}.\n"
            "Download the GTZAN dataset and extract so that:\n"
            f"  {genres_dir}/blues/blues.00000.wav\n"
            f"  {genres_dir}/disco/disco.00000.wav\n"
            "  ... etc."
        )

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw BIDS data not found at {raw_dir}.\n"
            "Run 'neural-vibe download' first."
        )

    clips_dir.mkdir(parents=True, exist_ok=True)

    # Collect unique clips from all events.tsv files
    clips: set[tuple[str, int, float, float]] = set()
    for events_tsv in sorted(raw_dir.rglob("*_events.tsv")):
        df = pd.read_csv(events_tsv, sep="\t")
        for _, row in df.iterrows():
            genre = str(row["genre"]).strip("'\"").lower()
            track = int(row["track"])
            start = float(row["start"])
            end = float(row["end"])
            clips.add((genre, track, start, end))

    log.info("Found %d unique clips to prepare.", len(clips))

    n_existing = 0
    n_cut = 0
    n_failed = 0
    for genre, track, start, end in sorted(clips):
        clip_name = _clip_id(genre, track, start, end)
        clip_path = clips_dir / f"{clip_name}.wav"

        if clip_path.exists():
            n_existing += 1
            continue

        try:
            gtzan_path = _find_gtzan_track(genres_dir, genre, track)
        except FileNotFoundError:
            log.warning("Missing GTZAN track: %s %05d", genre, track)
            n_failed += 1
            continue

        sr, data = wavfile.read(str(gtzan_path))
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        # Clamp to valid range
        start_sample = max(0, min(start_sample, len(data)))
        end_sample = max(start_sample, min(end_sample, len(data)))
        clip_data = data[start_sample:end_sample]
        wavfile.write(str(clip_path), sr, clip_data)
        n_cut += 1

    log.info(
        "Prepared %d new clips (%d already existed, %d failed).",
        n_cut,
        n_existing,
        n_failed,
    )
    return clips_dir


def get_clip_path(
    clips_dir: Path, genre: str, track: int, start: float, end: float
) -> Path | None:
    """Get path to a pre-cut clip, or None if not available."""
    clip_name = _clip_id(genre, track, start, end)
    clip_path = clips_dir / f"{clip_name}.wav"
    return clip_path if clip_path.exists() else None
