"""TRIBE v2 study adapter for the StudyForrest music perception dataset.

Dataset: OpenNeuro ds000113 (7T_musicperception component)
Paper: Hanke et al. (2015) "High-resolution 7-Tesla fMRI data on the
       perception of musical genres" (F1000Research)

20 subjects listened to 25 music clips (6s each) from 5 genres
(ambient, country, metal, rocknroll, symphonic) in a slow event-related
design while undergoing 7T fMRI scanning (TR=2.0s, 1.4mm isotropic).

Each subject completed 8 runs. Each run presents all 25 clips once
in pseudo-random order with 4-8s inter-stimulus intervals.

The dataset includes scanner-side distortion-corrected ("dico") BOLD
data which can be used directly, or raw data for fMRIPrep preprocessing.
"""

from __future__ import annotations

import logging
import typing as tp
from pathlib import Path

import pandas as pd

from neuralset.events.study import Study

log = logging.getLogger(__name__)

_SUBJECTS = [f"sub-{i:02d}" for i in range(1, 21)]
_RUNS = [f"run-{i:02d}" for i in range(1, 9)]

# Use runs 1-6 for training, 7-8 for validation (same stimuli, different order)
_TRAIN_RUNS = [f"run-{i:02d}" for i in range(1, 7)]
_VAL_RUNS = [f"run-{i:02d}" for i in range(7, 9)]


def _get_events_tsv(bids_root: Path, subject: str, run: str) -> Path:
    return (
        bids_root
        / subject
        / "ses-auditoryperception"
        / "func"
        / f"{subject}_ses-auditoryperception_task-auditoryperception_{run}_events.tsv"
    )


def _get_fmri_path(
    data_root: Path, subject: str, run: str, use_fmriprep: bool = False
) -> Path:
    """Get the BOLD file for a run.

    Checks for fMRIPrep derivatives first, then falls back to scanner-side
    distortion-corrected ("dico") data from the raw BIDS directory.
    """
    if use_fmriprep:
        return (
            data_root
            / "derivatives"
            / subject
            / "ses-auditoryperception"
            / "func"
            / f"{subject}_ses-auditoryperception_task-auditoryperception_{run}"
            f"_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
        )
    # Scanner-side distortion-corrected BOLD (available in raw BIDS)
    return (
        data_root
        / "raw"
        / subject
        / "ses-auditoryperception"
        / "func"
        / f"{subject}_ses-auditoryperception_task-auditoryperception_{run}_bold.nii.gz"
    )


class StudyForrestMusic(Study):
    """StudyForrest 7T music perception dataset for TRIBE v2 fine-tuning.

    Expects data laid out as:
        {path}/
        ├── raw/           # BIDS raw data (from OpenNeuro ds000113)
        │   ├── sub-01/
        │   │   └── ses-auditoryperception/
        │   │       └── func/
        │   │           ├── *_task-auditoryperception_run-01_bold.nii.gz
        │   │           └── *_task-auditoryperception_run-01_events.tsv
        │   └── ...
        ├── derivatives/   # fMRIPrep output (optional)
        └── stimuli/       # 25 music clips as WAV files
            ├── ambient_000.wav
            ├── country_001.wav
            └── ...
    """

    device = "Fmri"
    licence = "PDDL"
    url = "https://openneuro.org/datasets/ds000113"
    description = "7T music genre perception - 20 subjects, 5 genres, 25 clips"
    bibtex = ""
    requirements = ("nibabel",)

    TR_FMRI_S: float = 2.0
    CLIP_DURATION_S: float = 6.0

    def _download(self) -> None:
        raw_dir = self.path / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        try:
            import openneuro

            log.info("Downloading ds000113 music data from OpenNeuro to %s…", raw_dir)
            openneuro.download(
                dataset="ds000113",
                target_dir=str(raw_dir),
                include=["*auditoryperception*", "*.json", "participants.tsv"],
            )
        except ImportError:
            raise RuntimeError(
                "openneuro-py is required to download the dataset. "
                "Install it with: uv pip install openneuro-py"
            )

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        """Yield one timeline per (subject, run) combination."""
        for subject in _SUBJECTS:
            for run in _RUNS:
                yield dict(subject=subject, run=run)

    def _load_timeline_events(
        self, timeline: dict[str, tp.Any]
    ) -> pd.DataFrame:
        """Load fMRI + audio events for a single run.

        Creates:
        - One Fmri event spanning the full run
        - One Audio event per 6s music clip
        """
        import nibabel

        subject = timeline["subject"]
        run = timeline["run"]
        raw_dir = self.path / "raw"

        # --- Find fMRI data (prefer fMRIPrep, fall back to raw) ---
        fmri_path = _get_fmri_path(self.path, subject, run, use_fmriprep=True)
        if not fmri_path.exists():
            fmri_path = _get_fmri_path(self.path, subject, run, use_fmriprep=False)
        if not fmri_path.exists():
            log.warning("No BOLD file found for %s %s — skipping", subject, run)
            return pd.DataFrame()

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
        stimuli_dir = self.path / "stimuli"
        events_tsv = _get_events_tsv(raw_dir, subject, run)
        if events_tsv.exists():
            events_df = pd.read_csv(events_tsv, sep="\t")
            for _, row in events_df.iterrows():
                onset = float(row["onset"])
                dur = float(row.get("duration", self.CLIP_DURATION_S))
                genre = str(row.get("genre", "")).strip()
                stim_file = str(row.get("stim", "")).strip()

                audio_event: dict[str, tp.Any] = dict(
                    type="Audio",
                    start=onset,
                    duration=dur,
                    stop=onset + dur,
                    genre=genre,
                )

                # Link to stimulus audio file
                if stim_file and stimuli_dir.exists():
                    clip_path = stimuli_dir / stim_file
                    if clip_path.exists():
                        audio_event["filepath"] = str(clip_path)

                all_events.append(audio_event)

        result = pd.DataFrame(all_events)
        result["split"] = "val" if run in _VAL_RUNS else "train"

        return result
