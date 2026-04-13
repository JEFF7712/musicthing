"""Query the neural fingerprint index for similar songs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from .encoder import NeuralEncoder
from .indexer import load_index

log = logging.getLogger(__name__)

# Fingerprint layout: [auditory(120) | limbic(120) | prefrontal(120) | global(160) | clap(512)?]
BRAIN_DIMS = {"auditory": 120, "limbic": 120, "prefrontal": 120, "global": 160}
BRAIN_TOTAL = sum(BRAIN_DIMS.values())  # 520


def _build_weight_mask(
    weights: dict[str, float] | None, fingerprint_dim: int
) -> np.ndarray | None:
    """Build a per-dimension weight mask from region group weights.

    Scales the fingerprint segments so that cosine similarity
    emphasizes the weighted regions more. Automatically handles
    both brain-only (520-dim) and hybrid (1032-dim) fingerprints.
    """
    if weights is None:
        return None

    parts = []
    for name, dim in BRAIN_DIMS.items():
        w = weights.get(name, 1.0)
        parts.append(np.full(dim, w, dtype=np.float32))

    # Append CLAP weight if fingerprint includes it
    if fingerprint_dim > BRAIN_TOTAL:
        clap_dim = fingerprint_dim - BRAIN_TOTAL
        w = weights.get("clap", 1.0)
        parts.append(np.full(clap_dim, w, dtype=np.float32))

    return np.concatenate(parts)


@dataclass
class Match:
    """A single search result."""

    title: str
    artist: str
    path: str
    distance: float
    rank: int


def query_similar(
    seed_paths: list[str | Path],
    data_dir: str | Path = "data",
    n: int = 10,
    encoder: NeuralEncoder | None = None,
    region_weights: dict[str, float] | None = None,
) -> list[Match]:
    """Find songs in the index most similar to the seed songs.

    Args:
        seed_paths: One or more audio files to use as the query.
        data_dir: Directory containing the FAISS index and metadata.
        n: Number of results to return.
        encoder: NeuralEncoder instance (created if not provided).
        region_weights: Optional dict weighting region groups
            (auditory, limbic, prefrontal, global).

    Returns:
        List of Match objects sorted by similarity (closest first).
    """
    index, metadata = load_index(data_dir)
    if index is None:
        raise FileNotFoundError(
            f"No index found in {data_dir}. Run 'neural-vibe index' first."
        )

    if encoder is None:
        encoder = NeuralEncoder()

    weight_mask = _build_weight_mask(region_weights, index.d)

    # Encode seed songs and average their fingerprints
    fingerprints = []
    seed_resolved = set()
    for p in seed_paths:
        p = Path(p).resolve()
        seed_resolved.add(str(p))
        fp = encoder.encode(p)
        fingerprints.append(fp)

    target = np.mean(fingerprints, axis=0).astype(np.float32).reshape(1, -1)

    if weight_mask is not None:
        # Apply weights to query and reconstruct index vectors for re-ranking
        all_vectors = np.zeros((index.ntotal, index.d), dtype=np.float32)
        for i in range(index.ntotal):
            all_vectors[i] = index.reconstruct(i)

        # Weight both query and index vectors
        target_w = target * weight_mask
        vectors_w = all_vectors * weight_mask

        # Normalize and compute cosine similarity manually
        faiss.normalize_L2(target_w)
        faiss.normalize_L2(vectors_w)
        similarities = (vectors_w @ target_w.T).flatten()

        # Sort by descending similarity
        ranked = np.argsort(-similarities)

        results = []
        for idx in ranked:
            meta = metadata[int(idx)]
            if meta["path"] in seed_resolved:
                continue
            results.append(
                Match(
                    title=meta["title"],
                    artist=meta["artist"],
                    path=meta["path"],
                    distance=float(similarities[idx]),
                    rank=len(results) + 1,
                )
            )
            if len(results) >= n:
                break
    else:
        # No weighting — use FAISS directly
        faiss.normalize_L2(target)
        k = min(n + len(seed_paths), index.ntotal)
        distances, indices = index.search(target, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            meta = metadata[idx]
            if meta["path"] in seed_resolved:
                continue
            results.append(
                Match(
                    title=meta["title"],
                    artist=meta["artist"],
                    path=meta["path"],
                    distance=float(dist),
                    rank=len(results) + 1,
                )
            )
            if len(results) >= n:
                break

    return results
