"""Fine-tune TRIBE v2 on the Nakai 2021 Music Genre fMRI dataset.

Uses TRIBE v2's data pipeline to extract Wav2VecBert audio features and
project fMRI data to cortical surface, then fine-tunes the brain model's
temporal transformer and prediction head to better map music audio to
brain responses.

Prerequisites:
    1. Preprocessed fMRI data (neural-vibe preprocess)
    2. GTZAN audio stimuli (neural-vibe prepare-stimuli)

Usage:
    neural-vibe finetune <data-dir> [--epochs 10] [--lr 1e-5] [--batch-size 4]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch

log = logging.getLogger(__name__)


@dataclass
class FinetuneConfig:
    """Configuration for fine-tuning TRIBE v2 on music fMRI data."""

    data_dir: str = "data/nakai2021"
    cache_dir: str = ".cache/tribev2"
    output_dir: str = "checkpoints/music-finetuned"

    # Training hyperparameters
    epochs: int = 10
    lr: float = 1e-5
    batch_size: int = 4

    # What to freeze during fine-tuning
    freeze_encoders: bool = True
    freeze_projectors: bool = False
    freeze_transformer: bool = False

    # Hardware
    device: str = "auto"

    # Segment parameters (matching TRIBE v2 defaults)
    segment_duration_trs: int = 20  # ~30s at TR=1.5s


def finetune(config: FinetuneConfig | None = None) -> Path:
    """Run fine-tuning of TRIBE v2 on music fMRI data.

    Returns the path to the best checkpoint.
    """
    if config is None:
        config = FinetuneConfig()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Validate stimuli ---
    data_dir = Path(config.data_dir)
    clips_dir = data_dir / "stimuli" / "clips"
    if not clips_dir.exists() or not any(clips_dir.glob("*.wav")):
        raise FileNotFoundError(
            f"Audio stimuli not found at {clips_dir}.\n"
            "The fine-tuning pipeline needs the original music that subjects\n"
            "listened to during the fMRI scans (GTZAN dataset).\n\n"
            "Steps to set up:\n"
            "  1. Download GTZAN dataset (genres.tar.gz)\n"
            f"  2. Extract to: {data_dir}/stimuli/gtzan/genres/\n"
            f"  3. Run: neural-vibe prepare-stimuli {data_dir}\n"
            f"  4. Run: neural-vibe finetune {data_dir}"
        )

    device = config.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load pretrained model ---
    log.info("Loading pretrained TRIBE v2 model…")
    from tribev2 import TribeModel

    tribe = TribeModel.from_pretrained(
        "facebook/tribev2",
        cache_folder=config.cache_dir,
        device=device,
        config_update={
            "data.features_to_use": ["audio"],
            "data.audio_feature.device": device,
            "data.duration_trs": config.segment_duration_trs,
            "data.batch_size": config.batch_size,
            "data.num_workers": 4,
            "data.shuffle_train": True,
            "data.shuffle_val": False,
            "average_subjects": True,
            # Use nilearn's bundled fsaverage mesh instead of FreeSurfer
            "data.neuro.projection.extract_fsaverage_from_mni": False,
        },
    )

    brain_model = tribe._model

    # --- Load study data ---
    log.info("Loading Nakai 2021 study from %s…", config.data_dir)
    from .studies.nakai2021 import Nakai2021Bold

    study = Nakai2021Bold(path=data_dir)
    events = study.run()

    n_train = (events.split == "train").sum()
    n_val = (events.split == "val").sum()
    n_audio = (events.type == "Audio").sum()
    n_fmri = (events.type == "Fmri").sum()
    log.info(
        "Events: %d total (%d train, %d val, %d audio, %d fmri)",
        len(events), n_train, n_val, n_audio, n_fmri,
    )

    # Check that audio events have filepaths
    audio_events = events[events.type == "Audio"]
    if "filepath" not in audio_events.columns or audio_events.filepath.isna().all():
        raise RuntimeError(
            "Audio events have no filepath — stimuli clips not linked.\n"
            f"Run: neural-vibe prepare-stimuli {config.data_dir}"
        )

    has_filepath = audio_events.filepath.notna().sum()
    log.info(
        "Audio events with filepath: %d / %d", has_filepath, len(audio_events)
    )

    # --- Build data loaders via TRIBE v2's pipeline ---
    log.info("Extracting features and building data loaders…")
    log.info("(This may take a while on first run — features are cached.)")
    loaders = tribe.data.get_loaders(events=events)

    train_loader = loaders.get("train")
    val_loader = loaders.get("val")

    if train_loader is None:
        raise RuntimeError("No training data produced — check events and extractors.")

    log.info(
        "Data loaders: %d train batches, %d val batches",
        len(train_loader),
        len(val_loader) if val_loader else 0,
    )

    # --- Apply freezing strategy ---
    brain_model.train()
    n_frozen = 0
    n_total = 0
    for name, param in brain_model.named_parameters():
        n_total += 1
        freeze = False
        if config.freeze_encoders and _is_encoder_param(name):
            freeze = True
        if config.freeze_projectors and "projector" in name:
            freeze = True
        if config.freeze_transformer and "transformer" in name.lower():
            freeze = True
        if freeze:
            param.requires_grad = False
            n_frozen += 1

    log.info(
        "Frozen %d / %d parameters (%.1f%% trainable)",
        n_frozen, n_total,
        100 * (n_total - n_frozen) / max(n_total, 1),
    )

    # --- Training ---
    optimizer = torch.optim.AdamW(
        [p for p in brain_model.parameters() if p.requires_grad],
        lr=config.lr,
    )
    criterion = torch.nn.MSELoss()

    best_val_loss = float("inf")
    best_ckpt = output_dir / "best.pt"

    log.info(
        "Starting fine-tuning: epochs=%d, lr=%s, batch_size=%d, device=%s",
        config.epochs, config.lr, config.batch_size, device,
    )

    for epoch in range(config.epochs):
        # --- Train ---
        brain_model.train()
        train_loss = 0.0
        n_train_batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            y_true = batch.data["fmri"]  # (B, D, T)
            y_pred = brain_model(batch)  # (B, D, T)

            # Flatten time: (B*T, D)
            B, D, T = y_true.shape
            y_true_flat = y_true.permute(0, 2, 1).reshape(-1, D)
            y_pred_flat = y_pred.permute(0, 2, 1).reshape(-1, D)

            # Skip zero-padded timesteps
            valid = ~(y_true_flat == 0).all(dim=1)
            if valid.sum() == 0:
                continue
            y_true_flat = y_true_flat[valid]
            y_pred_flat = y_pred_flat[valid]

            loss = criterion(y_pred_flat, y_true_flat)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            n_train_batches += 1

        # --- Validate ---
        val_loss = 0.0
        n_val_batches = 0
        brain_model.eval()

        if val_loader is not None:
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    y_true = batch.data["fmri"]
                    y_pred = brain_model(batch)

                    B, D, T = y_true.shape
                    y_true_flat = y_true.permute(0, 2, 1).reshape(-1, D)
                    y_pred_flat = y_pred.permute(0, 2, 1).reshape(-1, D)

                    valid = ~(y_true_flat == 0).all(dim=1)
                    if valid.sum() == 0:
                        continue
                    y_true_flat = y_true_flat[valid]
                    y_pred_flat = y_pred_flat[valid]

                    loss = criterion(y_pred_flat, y_true_flat)
                    val_loss += loss.item()
                    n_val_batches += 1

        avg_train = train_loss / max(n_train_batches, 1)
        avg_val = val_loss / max(n_val_batches, 1)

        log.info(
            "Epoch %d/%d — train_loss: %.4f, val_loss: %.4f",
            epoch + 1, config.epochs, avg_train, avg_val,
        )

        if n_val_batches > 0 and avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(brain_model.state_dict(), best_ckpt)
            log.info("  → New best model saved (val_loss=%.4f)", avg_val)
        elif n_val_batches == 0:
            # No validation data — save every epoch
            torch.save(brain_model.state_dict(), best_ckpt)

    log.info("Fine-tuning complete. Best checkpoint: %s", best_ckpt)
    return best_ckpt


def _is_encoder_param(name: str) -> bool:
    """Check if a parameter belongs to a feature encoder."""
    encoder_prefixes = (
        "text_encoder", "audio_encoder", "video_encoder", "image_encoder",
    )
    return any(name.startswith(p) for p in encoder_prefixes)
