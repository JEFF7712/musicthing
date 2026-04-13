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


# Known dataset types and their Study adapter classes
DATASET_TYPES = {
    "nakai2021": "neural_vibe.studies.nakai2021:Nakai2021Bold",
    "studyforrest": "neural_vibe.studies.studyforrest:StudyForrestMusic",
}


@dataclass
class FinetuneConfig:
    """Configuration for fine-tuning TRIBE v2 on music fMRI data."""

    data_dir: str = "data/nakai2021"
    extra_data_dirs: list[str] | None = None  # Additional datasets
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

    # Early stopping
    patience: int = 5

    # LR schedule
    warmup_epochs: int = 1

    # Hardware
    device: str = "auto"

    # Segment duration must match model's n_output_timesteps (100)
    segment_duration_trs: int = 100


def _load_study(data_dir: Path):
    """Auto-detect and load the right Study adapter for a data directory.

    Detection heuristic:
    - If a `study_type.txt` file exists, use its contents
    - If raw/ contains ses-auditoryperception dirs → StudyForrest
    - If raw/ contains task-Training events → Nakai2021
    """
    # Explicit type file
    type_file = data_dir / "study_type.txt"
    if type_file.exists():
        study_type = type_file.read_text().strip()
    else:
        # Auto-detect from directory structure
        raw_dir = data_dir / "raw"
        if any(raw_dir.glob("*/ses-auditoryperception")):
            study_type = "studyforrest"
        else:
            study_type = "nakai2021"

    if study_type not in DATASET_TYPES:
        raise ValueError(
            f"Unknown study type '{study_type}'. "
            f"Known types: {', '.join(DATASET_TYPES)}"
        )

    module_path, class_name = DATASET_TYPES[study_type].rsplit(":", 1)
    import importlib
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(path=data_dir)


def _pearson_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Negative Pearson correlation loss across cortical vertices.

    Args:
        pred, target: (N, D) tensors where D is number of vertices.

    Returns:
        Scalar: 1 - mean_correlation. Lower = better pattern match.
    """
    pred_c = pred - pred.mean(dim=1, keepdim=True)
    tgt_c = target - target.mean(dim=1, keepdim=True)
    num = (pred_c * tgt_c).sum(dim=1)
    den = torch.sqrt(
        (pred_c ** 2).sum(dim=1) * (tgt_c ** 2).sum(dim=1)
    )
    corr = num / (den + 1e-8)
    return 1.0 - corr.mean()


def finetune(config: FinetuneConfig | None = None) -> Path:
    """Run fine-tuning of TRIBE v2 on music fMRI data.

    Returns the path to the best checkpoint.
    """
    if config is None:
        config = FinetuneConfig()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    # --- Load study data (support multiple datasets) ---
    all_data_dirs = [config.data_dir]
    if config.extra_data_dirs:
        all_data_dirs.extend(config.extra_data_dirs)

    all_events = []
    for data_dir_str in all_data_dirs:
        data_dir = Path(data_dir_str)
        study = _load_study(data_dir)
        log.info("Loading %s from %s…", type(study).__name__, data_dir)
        events = study.run()
        all_events.append(events)
        log.info(
            "  %s: %d events (%d train, %d val, %d audio, %d fmri)",
            data_dir.name,
            len(events),
            (events.split == "train").sum(),
            (events.split == "val").sum(),
            (events.type == "Audio").sum(),
            (events.type == "Fmri").sum(),
        )

    events = pd.concat(all_events, ignore_index=True)

    n_train = (events.split == "train").sum()
    n_val = (events.split == "val").sum()
    n_audio = (events.type == "Audio").sum()
    n_fmri = (events.type == "Fmri").sum()
    log.info(
        "Combined events: %d total (%d train, %d val, %d audio, %d fmri)",
        len(events), n_train, n_val, n_audio, n_fmri,
    )

    # Check that audio events have filepaths
    audio_events = events[events.type == "Audio"]
    if "filepath" not in audio_events.columns or audio_events.filepath.isna().all():
        raise RuntimeError(
            "Audio events have no filepath — stimuli clips not linked.\n"
            "Ensure audio stimuli are set up for each dataset."
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

    # Cosine annealing with linear warmup
    warmup_epochs = min(config.warmup_epochs, config.epochs)
    cosine_epochs = config.epochs - warmup_epochs

    warmup_scheduler = (
        torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_epochs
        )
        if warmup_epochs > 0
        else None
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cosine_epochs, 1)
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[s for s in [warmup_scheduler, cosine_scheduler] if s is not None],
        milestones=[warmup_epochs] if warmup_epochs > 0 else [],
    )

    best_val_loss = float("inf")
    best_ckpt = output_dir / "best.pt"
    epochs_without_improvement = 0

    log.info(
        "Starting fine-tuning: epochs=%d, lr=%s, batch_size=%d, device=%s, "
        "loss=pearson, warmup=%d, patience=%d",
        config.epochs, config.lr, config.batch_size, device,
        warmup_epochs, config.patience,
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

            # Match time dimensions if they differ
            if y_pred.shape[2] != y_true.shape[2]:
                y_pred = torch.nn.functional.adaptive_avg_pool1d(
                    y_pred, y_true.shape[2]
                )

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

            loss = _pearson_loss(y_pred_flat, y_true_flat)
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

                    if y_pred.shape[2] != y_true.shape[2]:
                        y_pred = torch.nn.functional.adaptive_avg_pool1d(
                            y_pred, y_true.shape[2]
                        )

                    B, D, T = y_true.shape
                    y_true_flat = y_true.permute(0, 2, 1).reshape(-1, D)
                    y_pred_flat = y_pred.permute(0, 2, 1).reshape(-1, D)

                    valid = ~(y_true_flat == 0).all(dim=1)
                    if valid.sum() == 0:
                        continue
                    y_true_flat = y_true_flat[valid]
                    y_pred_flat = y_pred_flat[valid]

                    loss = _pearson_loss(y_pred_flat, y_true_flat)
                    val_loss += loss.item()
                    n_val_batches += 1

        avg_train = train_loss / max(n_train_batches, 1)
        avg_val = val_loss / max(n_val_batches, 1)
        current_lr = optimizer.param_groups[0]["lr"]

        # Correlation = 1 - loss (higher is better)
        train_corr = 1.0 - avg_train
        val_corr = 1.0 - avg_val if n_val_batches > 0 else float("nan")

        log.info(
            "Epoch %d/%d — train_corr: %.4f, val_corr: %.4f, lr: %.2e",
            epoch + 1, config.epochs, train_corr, val_corr, current_lr,
        )

        scheduler.step()

        # --- Checkpointing & early stopping ---
        if n_val_batches > 0 and avg_val < best_val_loss:
            best_val_loss = avg_val
            epochs_without_improvement = 0
            torch.save(brain_model.state_dict(), best_ckpt)
            log.info("  → New best model saved (val_corr=%.4f)", val_corr)
        elif n_val_batches > 0:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                log.info(
                    "Early stopping: no improvement for %d epochs.",
                    config.patience,
                )
                break
        else:
            # No validation data — save every epoch
            torch.save(brain_model.state_dict(), best_ckpt)

    best_corr = 1.0 - best_val_loss if best_val_loss < float("inf") else float("nan")
    log.info(
        "Fine-tuning complete. Best val_corr=%.4f, checkpoint: %s",
        best_corr, best_ckpt,
    )
    return best_ckpt


def _is_encoder_param(name: str) -> bool:
    """Check if a parameter belongs to a feature encoder."""
    encoder_prefixes = (
        "text_encoder", "audio_encoder", "video_encoder", "image_encoder",
    )
    return any(name.startswith(p) for p in encoder_prefixes)
