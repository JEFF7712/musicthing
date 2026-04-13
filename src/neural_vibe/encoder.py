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

# Text LLM for lyric/semantic features. Llama 3.2 3B is the TRIBE v2 default
# but needs ~6 hours per song on CPU. GPT-2 (124M) is ~100x faster on CPU
# while still capturing lyrical/semantic content for brain prediction.
# Set NEURAL_VIBE_TEXT_MODEL to override (e.g. "meta-llama/Llama-3.2-3B" with GPU).
TEXT_MODEL = os.environ.get("NEURAL_VIBE_TEXT_MODEL", "gpt2")


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


def _patch_text_model_fp16():
    """Monkey-patch TRIBE v2's text model loading to use fp16 on GPU.

    Only applies to CUDA — fp16 on CPU is actually *slower* because most
    CPUs lack native fp16 compute and PyTorch upcasts every operation.
    """
    from neuralset.extractors.text import HuggingFaceText

    _original_load = HuggingFaceText._load_model

    def _load_model_fp16(self, **kwargs):
        if self.device == "cuda" and "torch_dtype" not in kwargs:
            kwargs["torch_dtype"] = torch.float16
            log.info("Loading text model %s in fp16 to save VRAM", self.model_name)
        return _original_load(self, **kwargs)

    HuggingFaceText._load_model = _load_model_fp16


class NeuralEncoder:
    """Encodes audio files into neural fingerprints using TRIBE v2.

    A neural fingerprint is a fixed-size vector summarizing the predicted
    brain response to a song — mean and std of ~20k cortical vertex
    activations over time.
    """

    # CLAP embedding dimension (laion/larger_clap_music projection output)
    CLAP_DIM = 512

    def __init__(
        self,
        cache_dir: str = ".cache/tribev2",
        device: str | None = None,
        region_weights: dict[str, float] | None = None,
        checkpoint: str | None = None,
        use_clap: bool = True,
    ):
        self._cache_dir = cache_dir
        self._device = device or self._pick_device()
        self._model = None
        self._region_weights = region_weights
        self._weight_vector = None
        self._checkpoint = checkpoint
        self._use_clap = use_clap
        self._clap_model = None
        self._clap_processor = None

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

        # Patch WhisperX and text model loading before any TRIBE v2 imports
        _patch_whisperx_model()
        _patch_text_model_fp16()

        from tribev2 import TribeModel

        log.info("Loading TRIBE v2 model (device=%s, text_model=%s)…", self._device, TEXT_MODEL)
        config_update = {
            "data.audio_feature.device": self._device,
            "data.text_feature.device": "cpu",
            "data.text_feature.model_name": TEXT_MODEL,
            "data.text_feature.contextualized": False,
            "data.text_feature.batch_size": 32,
            "data.features_to_use": ["audio", "text"],
        }
        self._model = TribeModel.from_pretrained(
            "facebook/tribev2",
            cache_folder=self._cache_dir,
            config_update=config_update,
        )

        # Load fine-tuned weights if a checkpoint was provided (before
        # projector adaptation, since the checkpoint has original dimensions).
        if self._checkpoint:
            ckpt_path = Path(self._checkpoint)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            log.info("Loading fine-tuned weights from %s", ckpt_path)
            state_dict = torch.load(
                ckpt_path, map_location=self._device, weights_only=True
            )
            self._model._model.load_state_dict(state_dict, strict=True)
            log.info("Fine-tuned weights loaded.")

        # Adapt the text projector if the text model's hidden size differs
        # from Llama 3.2 3B (which the pretrained projector was trained for).
        # Must happen AFTER checkpoint loading since checkpoints have original dims.
        self._adapt_text_projector()

        log.info("TRIBE v2 model loaded.")

    def _adapt_text_projector(self) -> None:
        """Resize the text projector if the text model's hidden size doesn't
        match the pretrained Llama 3.2 3B dimensions.

        The pretrained text projector is Linear(6144 → 384). If using a
        smaller model (e.g. GPT-2 with 1536-dim output), we derive new
        weights by folding the original weight matrix.
        """
        model = self._model._model
        if "text" not in model.projectors:
            return
        proj = model.projectors["text"]

        expected_in = proj.in_features  # 6144 for pretrained
        # Probe the actual text feature dimension by checking the model's hidden size
        text_cfg = self._model.data.text_feature
        from transformers import AutoConfig
        hf_config = AutoConfig.from_pretrained(text_cfg.model_name)
        hidden = hf_config.hidden_size

        # Compute actual feature dim: hidden_size × number of layer groups
        layers = text_cfg.layers if isinstance(text_cfg.layers, list) else [text_cfg.layers]
        if text_cfg.layer_aggregation == "group_mean":
            n_groups = max(len(layers) - 1, 1)
        elif text_cfg.layer_aggregation in ("mean", "sum"):
            n_groups = 1
        else:
            n_groups = len(layers)
        actual_in = hidden * n_groups

        if actual_in == expected_in:
            return

        log.info(
            "Adapting text projector: %d → %d (was %d)",
            actual_in, proj.out_features, expected_in,
        )
        with torch.no_grad():
            old_w = proj.weight.data  # (out, expected_in)
            old_b = proj.bias.data if proj.bias is not None else None
            ratio = expected_in // actual_in
            if expected_in % actual_in == 0 and ratio > 1:
                # Fold: reshape (out, ratio*actual_in) → (out, ratio, actual_in), average
                new_w = old_w.reshape(proj.out_features, ratio, actual_in).mean(dim=1)
            else:
                # Truncate or interpolate
                new_w = old_w[:, :actual_in]

            new_proj = torch.nn.Linear(actual_in, proj.out_features, bias=old_b is not None)
            new_proj.weight.data = new_w
            if old_b is not None:
                new_proj.bias.data = old_b
            new_proj.to(proj.weight.device)

        model.projectors["text"] = new_proj

    def _load_clap(self) -> None:
        """Lazy-load the CLAP music audio encoder."""
        if self._clap_model is not None:
            return
        from transformers import ClapModel, AutoFeatureExtractor

        log.info("Loading CLAP music encoder (laion/larger_clap_music)…")
        clap = ClapModel.from_pretrained(
            "laion/larger_clap_music", cache_dir=self._cache_dir
        )
        self._clap_model = clap.audio_model.eval()
        self._clap_projection = clap.audio_projection.eval()
        self._clap_processor = AutoFeatureExtractor.from_pretrained(
            "laion/larger_clap_music", cache_dir=self._cache_dir
        )
        log.info("CLAP loaded.")

    def _encode_clap(self, audio_path: str | Path) -> np.ndarray:
        """Get CLAP audio embedding for a song.

        Splits long audio into 10-second chunks and averages embeddings
        to represent the full song.

        Returns:
            1-D float32 array of shape (512,).
        """
        self._load_clap()
        import torchaudio

        waveform, sr = torchaudio.load(str(audio_path))
        target_sr = 48000
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        audio = waveform.mean(dim=0).numpy()  # mono

        # Split into 10s chunks for full-song coverage
        chunk_samples = target_sr * 10
        chunks = []
        for start in range(0, len(audio), chunk_samples):
            chunk = audio[start : start + chunk_samples]
            if len(chunk) < target_sr:  # skip < 1 second
                continue
            chunks.append(chunk)
        if not chunks:
            chunks = [audio]

        embeddings = []
        for chunk in chunks:
            inputs = self._clap_processor(
                audios=chunk, sampling_rate=target_sr, return_tensors="pt"
            )
            with torch.no_grad():
                audio_features = self._clap_model(**inputs).pooler_output
                audio_embed = self._clap_projection(audio_features)
            embeddings.append(audio_embed.squeeze().cpu().numpy())

        return np.mean(embeddings, axis=0).astype(np.float32)

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

    @staticmethod
    def _pca_stats(data: np.ndarray, n_components: int = 15) -> np.ndarray:
        """PCA + distributional stats on a (T, V) activation matrix."""
        from scipy.stats import skew as _skew

        mean_vec = data.mean(axis=0)
        centered = data - mean_vec
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        n = min(n_components, U.shape[1])
        projected = U[:, :n] * S[:n]

        parts = [
            projected.mean(axis=0),
            projected.std(axis=0),
            _skew(projected, axis=0),
            np.percentile(projected, 10, axis=0),
            np.percentile(projected, 50, axis=0),
            np.percentile(projected, 90, axis=0),
            np.diff(projected, axis=0).std(axis=0),
            S[:n] / S[:n].sum(),
        ]
        return np.concatenate(parts)

    def encode(self, audio_path: str | Path) -> np.ndarray:
        """Encode a song into a neural fingerprint vector.

        Computes separate PCA fingerprints for each brain region group
        (auditory, limbic, prefrontal) plus the full cortex. Optionally
        appends a CLAP music embedding for hybrid similarity.

        The fingerprint layout is:
          [auditory(120) | limbic(120) | prefrontal(120) | global(160) | clap(512)?]

        Returns:
            1-D float32 array (520 dims without CLAP, 1032 with).
        """
        from .regions import REGION_GROUPS, load_vertex_labels

        preds = self.predict_brain_response(audio_path)
        # preds shape: (T, ~20k vertices)

        labels = load_vertex_labels()

        # Per-region PCA fingerprints (15 components × 8 stats = 120 dims each)
        region_parts = []
        for group_name in ["auditory", "limbic", "prefrontal"]:
            mask = np.isin(labels, REGION_GROUPS[group_name])
            region_data = preds[:, mask]
            region_parts.append(self._pca_stats(region_data, n_components=15))

        # Global PCA (20 components × 8 stats = 160 dims)
        global_part = self._pca_stats(preds, n_components=20)
        brain_fp = np.concatenate(region_parts + [global_part]).astype(np.float32)

        if self._use_clap:
            clap_emb = self._encode_clap(audio_path)
            # Normalize each part independently so they contribute equally
            brain_norm = brain_fp / (np.linalg.norm(brain_fp) + 1e-8)
            clap_norm = clap_emb / (np.linalg.norm(clap_emb) + 1e-8)
            fingerprint = np.concatenate([brain_norm, clap_norm]).astype(np.float32)
            log.info(
                "Fingerprint for %s: dim=%d (brain 520 + CLAP 512)",
                Path(audio_path).name,
                fingerprint.shape[0],
            )
        else:
            fingerprint = brain_fp
            log.info(
                "Fingerprint for %s: dim=%d (brain only)",
                Path(audio_path).name,
                fingerprint.shape[0],
            )

        return fingerprint
