"""TRIBE v2 wrapper for encoding songs into neural fingerprints."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger(__name__)


class NeuralEncoder:
    """Encodes audio files into neural fingerprints using TRIBE v2.

    A neural fingerprint is a fixed-size vector summarizing the predicted
    brain response to a song — mean and std of ~20k cortical vertex
    activations over time.
    """

    def __init__(self, cache_dir: str = ".cache/tribev2", device: str | None = None):
        self._cache_dir = cache_dir
        self._device = device or self._pick_device()
        self._model = None

    @staticmethod
    def _pick_device() -> str:
        if torch.cuda.is_available():
            vram_mb = torch.cuda.get_device_properties(0).total_mem / 1024**2
            # Wav2VecBert (~600MB) + FmriEncoder (~200MB) fit in 4GB
            # but Llama 3.2 3B (~6GB fp16) does not.
            # We'll load the model on CPU and let TRIBE v2 handle device
            # placement, or force CPU if VRAM is tight.
            if vram_mb < 6000:
                log.info(
                    "Only %.0f MB VRAM available — using CPU to avoid OOM "
                    "(Llama 3.2 3B needs ~6GB). Indexing will be slower but safe.",
                    vram_mb,
                )
                return "cpu"
            return "cuda"
        return "cpu"

    def load(self) -> None:
        """Load the TRIBE v2 model. Call this once before encoding."""
        if self._model is not None:
            return

        from tribev2 import TribeModel

        log.info("Loading TRIBE v2 model (device=%s)…", self._device)
        self._model = TribeModel.from_pretrained(
            "facebook/tribev2",
            cache_folder=self._cache_dir,
        )
        # Move to chosen device if the model exposes a .to() method
        if hasattr(self._model, "to"):
            self._model.to(self._device)
        log.info("TRIBE v2 model loaded.")

    def predict_brain_response(self, audio_path: str | Path) -> np.ndarray:
        """Run TRIBE v2 on an audio file.

        Returns:
            Array of shape (n_timesteps, n_vertices) with predicted
            cortical activation.
        """
        self.load()
        audio_path = str(audio_path)

        log.info("Extracting events from %s", audio_path)
        df = self._model.get_events_dataframe(audio_path=audio_path)

        log.info("Predicting brain response…")
        preds, _segments = self._model.predict(events=df)

        # preds shape: (n_timesteps, n_vertices) — typically (~T, ~20k)
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()

        return preds

    def encode(self, audio_path: str | Path) -> np.ndarray:
        """Encode a song into a neural fingerprint vector.

        The fingerprint concatenates the mean and standard deviation of
        predicted cortical activation across time, giving a fixed-size
        vector regardless of song length.

        Returns:
            1-D float32 array of shape (2 * n_vertices,).
        """
        preds = self.predict_brain_response(audio_path)
        mean = preds.mean(axis=0)
        std = preds.std(axis=0)
        fingerprint = np.concatenate([mean, std]).astype(np.float32)
        log.info(
            "Fingerprint for %s: dim=%d, norm=%.2f",
            Path(audio_path).name,
            fingerprint.shape[0],
            np.linalg.norm(fingerprint),
        )
        return fingerprint
