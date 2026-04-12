"""TRIBE v2 study adapter for the Nakai 2021 Music Genre fMRI dataset.

Dataset: OpenNeuro ds003720
Paper: Nakai et al. (2021) "Music genre neuroimaging dataset" (Data in Brief)

5 subjects listened to 540 music clips (15s each) across 10 genres
(Blues, Classical, Country, Disco, Hip-hop, Jazz, Metal, Pop, Reggae, Rock)
while undergoing fMRI scanning (TR=1.5s, 2mm isotropic).

Raw BIDS data must be preprocessed with fMRIPrep before use.
"""

from __future__ import annotations

import logging
import typing as tp
from pathlib import Path

import pandas as pd

from neuralset.events.study import Study

log = logging.getLogger(__name__)

_SUBJECTS = ["sub-001", "sub-002", "sub-003", "sub-004", "sub-005"]

# Two BIDS tasks: Training (12 runs) and Test (6 runs)
_TRAIN_TASK = "Training"
_TEST_TASK = "Test"
_TRAIN_RUNS = [f"run-{i:02d}" for i in range(1, 13)]
_TEST_RUNS = [f"run-{i:02d}" for i in range(1, 7)]


def _get_events_tsv(bids_root: Path, subject: str, task: str, run: str) -> Path:
    """Get the BIDS events.tsv file for a run."""
    return (
        bids_root / subject / "func" / f"{subject}_task-{task}_{run}_events.tsv"
    )


def _get_fmri_path(derivatives_root: Path, subject: str, task: str, run: str) -> Path:
    """Get the preprocessed (fMRIPrep) BOLD file for a run."""
    return (
        derivatives_root
        / subject
        / "func"
        / f"{subject}_task-{task}_{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    )


class Nakai2021Bold(Study):
    """Nakai 2021 Music Genre fMRI dataset for TRIBE v2 fine-tuning.

    Expects data laid out as:
        {path}/
        ├── raw/           # BIDS raw data (from OpenNeuro ds003720)
        │   ├── sub-001/
        │   ├── ...
        │   └── sub-005/
        └── derivatives/   # fMRIPrep output
            ├── sub-001/
            └── ...
    """

    device = "Fmri"
    licence = "CC0"
    url = "https://openneuro.org/datasets/ds003720"
    description = "Music genre fMRI - 5 subjects, 10 genres, 540 clips"
    bibtex = ""
    requirements = ("nibabel", "openneuro-py")

    TR_FMRI_S: float = 1.5
    CLIP_DURATION_S: float = 15.0

    def _download(self) -> None:
        """Download raw BIDS data from OpenNeuro using openneuro-py.

        openneuro-py handles resumption automatically — partially
        downloaded files will be completed rather than re-downloaded.
        """
        raw_dir = self.path / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        try:
            import openneuro

            log.info("Downloading ds003720 from OpenNeuro to %s…", raw_dir)
            log.info("(Already-downloaded files will be skipped automatically.)")
            openneuro.download(dataset="ds003720", target_dir=str(raw_dir))
        except ImportError:
            raise RuntimeError(
                "openneuro-py is required to download the dataset. "
                "Install it with: uv pip install openneuro-py"
            )

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        """Yield one timeline per (subject, task, run) combination."""
        for subject in _SUBJECTS:
            for run in _TRAIN_RUNS:
                yield dict(subject=subject, task=_TRAIN_TASK, run=run)
            for run in _TEST_RUNS:
                yield dict(subject=subject, task=_TEST_TASK, run=run)

    def _load_timeline_events(
        self, timeline: dict[str, tp.Any]
    ) -> pd.DataFrame:
        """Load fMRI + audio events for a single run.

        Creates:
        - One Fmri event spanning the full run
        - One Audio event per 15s music clip (with genre metadata)
        """
        import nibabel

        subject = timeline["subject"]
        task = timeline["task"]
        run = timeline["run"]
        raw_dir = self.path / "raw"
        deriv_dir = self.path / "derivatives"

        # --- fMRI data ---
        fmri_path = _get_fmri_path(deriv_dir, subject, task, run)
        if not fmri_path.exists():
            raise FileNotFoundError(
                f"Preprocessed fMRI not found: {fmri_path}\n"
                "Run fMRIPrep first: neural-vibe preprocess <data-dir>"
            )

        nii: tp.Any = nibabel.load(fmri_path, mmap=True)
        freq = 1.0 / self.TR_FMRI_S
        n_trs = nii.shape[-1]
        duration = n_trs / freq

        all_events: list[dict[str, tp.Any]] = []

        # fMRI event
        all_events.append(
            dict(
                type="Fmri",
                start=0,
                filepath=str(fmri_path),
                frequency=freq,
                duration=duration,
            )
        )

        # --- Music clip events from events.tsv ---
        events_tsv = _get_events_tsv(raw_dir, subject, task, run)
        if events_tsv.exists():
            events_df = pd.read_csv(events_tsv, sep="\t")
            # Columns: onset, duration, genre, track, start, end
            for _, row in events_df.iterrows():
                onset = float(row["onset"])
                dur = float(row.get("duration", self.CLIP_DURATION_S))
                genre = str(row.get("genre", "")).strip("'\"")

                audio_event = dict(
                    type="Audio",
                    start=onset,
                    duration=dur,
                    stop=onset + dur,
                    genre=genre,
                    track=int(row["track"]) if "track" in row else -1,
                )
                all_events.append(audio_event)

        result = pd.DataFrame(all_events)
        result["split"] = "test" if task == _TEST_TASK else "train"

        return result
