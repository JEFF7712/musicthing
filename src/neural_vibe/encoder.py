"""TRIBE v2 wrapper for encoding songs into neural fingerprints."""

from __future__ import annotations

import logging
import os
from pathlib import Path

# Force CPU mode before torch is imported anywhere.
# PyTorch 2.6 (CUDA 12.4 runtime) is incompatible with CUDA 13.2 driver
# on this system, causing "CUDA unknown error". Hiding the GPU prevents
# TRIBE v2 internals from attempting .to("cuda") and crashing.
if not os.environ.get("NEURAL_VIBE_ALLOW_CUDA"):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch

log = logging.getLogger(__name__)

# WhisperX model to use for transcription. "large-v3" is most accurate
# but needs ~6GB VRAM; "small" needs ~500MB and is much faster on limited GPUs.
WHISPER_MODEL = os.environ.get("NEURAL_VIBE_WHISPER_MODEL", "small")


def _patch_whisperx_model():
    """Monkey-patch TRIBE v2's WhisperX invocation to use a smaller/faster model
    and allow GPU transcription on low-VRAM cards."""
    import tribev2.eventstransforms as et

    original = et.ExtractWordsFromAudio._get_transcript_from_audio

    @staticmethod
    def _patched_transcribe(wav_filename, language):
        import json
        import subprocess
        import tempfile

        language_codes = dict(
            english="en", french="fr", spanish="es", dutch="nl", chinese="zh"
        )
        if language not in language_codes:
            raise ValueError(f"Language {language} not supported")

        device = "cpu"
        compute_type = "int8"

        log.info(
            "WhisperX: model=%s, device=%s, compute=%s",
            WHISPER_MODEL, device, compute_type,
        )

        with tempfile.TemporaryDirectory() as output_dir:
            cmd = [
                "uvx", "whisperx",
                str(wav_filename),
                "--model", WHISPER_MODEL,
                "--language", language_codes[language],
                "--device", device,
                "--compute_type", compute_type,
                "--batch_size", "16",
                "--output_dir", output_dir,
                "--output_format", "json",
            ]
            # Add alignment model for English
            if language == "english":
                cmd.extend(["--align_model", "WAV2VEC2_ASR_LARGE_LV60K_960H"])

            env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}
            # Ensure CUDA is visible to the subprocess
            env.pop("CUDA_VISIBLE_DEVICES", None)
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
            if result.returncode != 0:
                raise RuntimeError(f"whisperx failed:\n{result.stderr}")

            json_path = Path(output_dir) / f"{wav_filename.stem}.json"
            transcript = json.loads(json_path.read_text())

        # Reconstruct word list exactly as TRIBE v2 expects
        import pandas as pd

        words = []
        for i, segment in enumerate(transcript["segments"]):
            sentence = segment["text"].replace('"', "")
            for word in segment.get("words", []):
                if "start" not in word:
                    continue
                words.append({
                    "text": word["word"].replace('"', ""),
                    "start": word["start"],
                    "duration": word["end"] - word["start"],
                    "sequence_id": i,
                    "sentence": sentence,
                })

        return pd.DataFrame(words) if words else pd.DataFrame(
            columns=["text", "start", "duration", "sequence_id", "sentence"]
        )

    et.ExtractWordsFromAudio._get_transcript_from_audio = _patched_transcribe


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
            return "cuda"
        log.info("Running on CPU (Whisper model: %s).", WHISPER_MODEL)
        return "cpu"

    def load(self) -> None:
        """Load the TRIBE v2 model. Call this once before encoding."""
        if self._model is not None:
            return

        # Patch WhisperX before any TRIBE v2 imports trigger it
        _patch_whisperx_model()

        from tribev2 import TribeModel

        log.info("Loading TRIBE v2 model (device=%s)…", self._device)
        # Override device for all extractors — the shipped config hardcodes "cuda"
        # Also disable text features: Llama 3.2 3B is too slow on CPU (~hours
        # per song). Audio features (Wav2VecBert) capture the sound itself,
        # which is what matters for music "vibe" matching.
        config_update = {
            "data.audio_feature.device": self._device,
            "data.features_to_use": ["audio"],
        }
        self._model = TribeModel.from_pretrained(
            "facebook/tribev2",
            cache_folder=self._cache_dir,
            config_update=config_update,
        )
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

        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()

        return preds

    def encode(self, audio_path: str | Path) -> np.ndarray:
        """Encode a song into a neural fingerprint vector.

        Uses PCA to reduce the ~20k cortical vertices to principal
        components, then computes distributional statistics over time
        for each component. This captures how the song's brain response
        *evolves* — not just its average.

        Returns:
            1-D float32 array.
        """
        from scipy.stats import skew as _skew

        preds = self.predict_brain_response(audio_path)
        # preds shape: (T, ~20k vertices)

        # PCA: reduce vertices to top N_COMPONENTS principal components
        N_COMPONENTS = 50
        # Center the data
        mean_vec = preds.mean(axis=0)
        centered = preds - mean_vec
        # SVD on (T, V) — since T < V, this is efficient
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        # Project onto top components: (T, N_COMPONENTS)
        n = min(N_COMPONENTS, U.shape[1])
        projected = U[:, :n] * S[:n]  # (T, n)

        # Compute distributional stats per component
        parts = [
            projected.mean(axis=0),                         # mean (n,)
            projected.std(axis=0),                          # std (n,)
            _skew(projected, axis=0),                       # skewness (n,)
            np.percentile(projected, 10, axis=0),           # 10th pctile (n,)
            np.percentile(projected, 50, axis=0),           # median (n,)
            np.percentile(projected, 90, axis=0),           # 90th pctile (n,)
            np.diff(projected, axis=0).std(axis=0),         # volatility (n,)
            S[:n] / S[:n].sum(),                            # explained variance ratio (n,)
        ]
        fingerprint = np.concatenate(parts).astype(np.float32)

        log.info(
            "Fingerprint for %s: dim=%d (PCA %d components × 8 stats)",
            Path(audio_path).name,
            fingerprint.shape[0],
            n,
        )
        return fingerprint
