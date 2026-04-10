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
) -> list[Match]:
    """Find songs in the index most similar to the seed songs.

    Args:
        seed_paths: One or more audio files to use as the query.
        data_dir: Directory containing the FAISS index and metadata.
        n: Number of results to return.
        encoder: NeuralEncoder instance (created if not provided).

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

    # Encode seed songs and average their fingerprints
    fingerprints = []
    seed_resolved = set()
    for p in seed_paths:
        p = Path(p).resolve()
        seed_resolved.add(str(p))
        fp = encoder.encode(p)
        fingerprints.append(fp)

    target = np.mean(fingerprints, axis=0).astype(np.float32).reshape(1, -1)

    # Search — request extra results to filter out seed songs
    k = min(n + len(seed_paths), index.ntotal)
    distances, indices = index.search(target, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue
        meta = metadata[idx]
        # Skip seed songs from results
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
