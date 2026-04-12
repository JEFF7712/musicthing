"""Index a music library into a FAISS vector database of neural fingerprints."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import faiss
import numpy as np
from mutagen import File as MutagenFile

from .encoder import NeuralEncoder

log = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".mp3", ".flac", ".wav", ".ogg", ".m4a", ".opus", ".wma", ".aac"}

INDEX_FILE = "index.faiss"
META_FILE = "metadata.json"


def _file_id(path: Path) -> str:
    """Stable identifier for a file based on path + mtime + size."""
    stat = path.stat()
    key = f"{path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _extract_metadata(path: Path) -> dict:
    """Extract title and artist from audio file tags."""
    meta = {"path": str(path.resolve()), "title": path.stem, "artist": "Unknown"}
    try:
        tags = MutagenFile(path, easy=True)
        if tags:
            meta["title"] = str(tags.get("title", [path.stem])[0])
            meta["artist"] = str(tags.get("artist", ["Unknown"])[0])
    except Exception:
        log.debug("Could not read tags from %s", path)
    return meta


def find_audio_files(music_dir: str | Path) -> list[Path]:
    """Recursively find all audio files in a directory."""
    music_dir = Path(music_dir)
    files = []
    for f in sorted(music_dir.rglob("*")):
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS:
            files.append(f)
    return files


def load_index(data_dir: str | Path) -> tuple[faiss.Index | None, list[dict]]:
    """Load existing FAISS index and metadata, or return None."""
    data_dir = Path(data_dir)
    index_path = data_dir / INDEX_FILE
    meta_path = data_dir / META_FILE

    if not index_path.exists() or not meta_path.exists():
        return None, []

    index = faiss.read_index(str(index_path))
    with open(meta_path) as f:
        metadata = json.load(f)
    return index, metadata


def save_index(
    index: faiss.Index, metadata: list[dict], data_dir: str | Path
) -> None:
    """Persist FAISS index and metadata to disk."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(data_dir / INDEX_FILE))
    with open(data_dir / META_FILE, "w") as f:
        json.dump(metadata, f, indent=2)


def build_index(
    music_dir: str | Path,
    data_dir: str | Path = "data",
    encoder: NeuralEncoder | None = None,
    on_progress: callable | None = None,
) -> tuple[faiss.Index, list[dict]]:
    """Index all audio files in music_dir.

    Supports incremental indexing — already-indexed files are skipped.

    Args:
        music_dir: Directory containing audio files.
        data_dir: Where to store the index and metadata.
        encoder: NeuralEncoder instance (created if not provided).
        on_progress: Called with (current, total, filename) for each file.

    Returns:
        (faiss_index, metadata_list)
    """
    if encoder is None:
        encoder = NeuralEncoder()

    files = find_audio_files(music_dir)
    if not files:
        raise FileNotFoundError(f"No audio files found in {music_dir}")

    # Load existing index for incremental updates
    index, metadata = load_index(data_dir)
    existing_ids = {m["file_id"] for m in metadata}

    # Determine which files are new
    new_files = []
    for f in files:
        fid = _file_id(f)
        if fid not in existing_ids:
            new_files.append((f, fid))

    if not new_files:
        log.info("All %d files already indexed.", len(files))
        return index, metadata

    log.info(
        "%d new files to index (%d already indexed).",
        len(new_files),
        len(files) - len(new_files),
    )

    encoder.load()

    # Encode new files
    new_fingerprints = []
    new_metadata = []
    for i, (path, fid) in enumerate(new_files):
        if on_progress:
            on_progress(i, len(new_files), path.name)
        try:
            fp = encoder.encode(path)
            meta = _extract_metadata(path)
            meta["file_id"] = fid
            new_fingerprints.append(fp)
            new_metadata.append(meta)
        except Exception:
            log.exception("Failed to encode %s — skipping.", path)

    if not new_fingerprints:
        log.warning("No new files were successfully encoded.")
        if index is not None:
            return index, metadata
        raise RuntimeError("Failed to encode any files.")

    new_matrix = np.stack(new_fingerprints)
    dim = new_matrix.shape[1]

    # Normalize for cosine similarity
    faiss.normalize_L2(new_matrix)

    # Create or extend FAISS index (inner product on L2-normalized vectors = cosine sim)
    if index is None:
        index = faiss.IndexFlatIP(dim)
        metadata = []

    index.add(new_matrix)
    metadata.extend(new_metadata)

    save_index(index, metadata, data_dir)
    log.info(
        "Index updated: %d total songs (%d new).",
        len(metadata),
        len(new_metadata),
    )
    return index, metadata
