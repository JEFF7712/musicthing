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

_SUBJECTS = ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05"]

# 12 training runs + 6 test runs per subject
_TRAIN_RUNS = [f"run-{i:02d}" for i in range(1, 13)]
_TEST_RUNS = [f"run-{i:02d}" for i in range(13, 19)]


def _get_audio_path(bids_root: Path, subject: str, run: str) -> Path:
    """Get the stimulus audio file for a given run.

    Audio stimuli are stored as extracted clips in a stimuli/ directory,
    one per run (concatenated from the 10 x 15s clips in that run).
    """
    return bids_root / "stimuli" / f"{subject}_{run}_audio.wav"


def _get_events_tsv(bids_root: Path, subject: str, run: str) -> Path:
    """Get the BIDS events.tsv file for a run."""
    return (
        bids_root / subject / "func" / f"{subject}_task-music_{run}_events.tsv"
    )


def _get_fmri_path(
    derivatives_root: Path, subject: str, run: str
) -> Path:
    """Get the preprocessed (fMRIPrep) BOLD file for a run."""
    return (
        derivatives_root
        / subject
        / "func"
        / f"{subject}_task-music_{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    )


class Nakai2021Bold(Study):
    """Nakai 2021 Music Genre fMRI dataset for TRIBE v2 fine-tuning.

    Expects data laid out as:
        {path}/
        ├── raw/           # BIDS raw data (from OpenNeuro ds003720)
        │   ├── sub-01/
        │   ├── ...
        │   ├── sub-05/
        │   └── stimuli/   # extracted audio per run
        └── derivatives/   # fMRIPrep output
            ├── sub-01/
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
        """Yield one timeline per (subject, run) combination."""
        for subject in _SUBJECTS:
            all_runs = _TRAIN_RUNS + _TEST_RUNS
            for run in all_runs:
                yield dict(subject=subject, run=run)

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
        run = timeline["run"]
        raw_dir = self.path / "raw"
        deriv_dir = self.path / "derivatives"

        # --- fMRI data ---
        fmri_path = _get_fmri_path(deriv_dir, subject, run)
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

        # --- Audio stimulus events ---
        events_tsv = _get_events_tsv(raw_dir, subject, run)
        if events_tsv.exists():
            events_df = pd.read_csv(events_tsv, sep="\t")
            for _, row in events_df.iterrows():
                onset = float(row["onset"])
                dur = float(row.get("duration", self.CLIP_DURATION_S))

                # The audio file for this specific clip
                # BIDS stim_file column points to the stimulus
                stim_file = row.get("stim_file", "")
                if stim_file and (raw_dir / stim_file).exists():
                    audio_path = str(raw_dir / stim_file)
                else:
                    # Fall back to per-run concatenated audio
                    audio_path = str(_get_audio_path(raw_dir, subject, run))

                audio_event = dict(
                    type="Audio",
                    start=onset,
                    duration=dur,
                    stop=onset + dur,
                    filepath=audio_path,
                )
                all_events.append(audio_event)

                # If genre info is available, add as metadata
                genre = row.get("genre", row.get("trial_type", None))
                if genre is not None:
                    all_events[-1]["genre"] = str(genre)
        else:
            # No events.tsv — treat the full run audio as one event
            audio_path = _get_audio_path(raw_dir, subject, run)
            if audio_path.exists():
                all_events.append(
                    dict(
                        type="Audio",
                        start=0,
                        duration=duration,
                        filepath=str(audio_path),
                    )
                )

        result = pd.DataFrame(all_events)

        # Mark split: runs 13-18 are test
        is_test = run in _TEST_RUNS
        result["split"] = "test" if is_test else "train"

        return result
