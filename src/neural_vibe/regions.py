"""Brain region mapping for weighted similarity search.

Maps fsaverage5 cortical vertices to functional groups so users can
search by "similar sound" (auditory cortex), "similar feeling" (limbic),
or "similar cognitive engagement" (prefrontal).
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

# Destrieux atlas region indices grouped by function.
# Each group maps to the brain systems most relevant for music perception.
REGION_GROUPS: dict[str, list[int]] = {
    "auditory": [
        33,  # G_temp_sup-G_T_transv (Heschl's gyrus — primary auditory cortex)
        34,  # G_temp_sup-Lateral
        35,  # G_temp_sup-Plan_polar (planum polare)
        36,  # G_temp_sup-Plan_tempo (planum temporale)
        74,  # S_temporal_sup
        75,  # S_temporal_transverse
        38,  # G_temporal_middle
        37,  # G_temporal_inf
        44,  # Pole_temporal
    ],
    "limbic": [
        6,   # G_and_S_cingul-Ant (anterior cingulate)
        7,   # G_and_S_cingul-Mid-Ant
        8,   # G_and_S_cingul-Mid-Post
        9,   # G_cingul-Post-dorsal
        10,  # G_cingul-Post-ventral
        17,  # G_Ins_lg_and_S_cent_ins (insula)
        18,  # G_insular_short
        23,  # G_oc-temp_med-Parahip (parahippocampal)
        32,  # G_subcallosal
        48,  # S_circular_insula_ant
        49,  # S_circular_insula_inf
        50,  # S_circular_insula_sup
        67,  # S_pericallosal
    ],
    "prefrontal": [
        1,   # G_and_S_frontomargin
        5,   # G_and_S_transv_frontopol
        12,  # G_front_inf-Opercular
        13,  # G_front_inf-Orbital
        14,  # G_front_inf-Triangul
        15,  # G_front_middle
        16,  # G_front_sup
        24,  # G_orbital
        31,  # G_rectus
        53,  # S_front_inf
        54,  # S_front_middle
        55,  # S_front_sup
    ],
}

# Default weights when no filter is applied — equal across all regions.
DEFAULT_WEIGHTS = {"auditory": 1.0, "limbic": 1.0, "prefrontal": 1.0}

# Preset weight profiles for common search modes.
# "clap" weights the CLAP music embedding (acoustic similarity).
PRESETS: dict[str, dict[str, float]] = {
    "sound":   {"auditory": 3.0, "limbic": 0.5, "prefrontal": 0.5, "clap": 2.0},
    "emotion": {"auditory": 0.5, "limbic": 3.0, "prefrontal": 0.5, "clap": 0.5},
    "thought": {"auditory": 0.5, "limbic": 0.5, "prefrontal": 3.0, "clap": 0.3},
    "vibe":    {"auditory": 1.5, "limbic": 2.0, "prefrontal": 1.0, "clap": 1.0},
}


def load_vertex_labels() -> np.ndarray:
    """Load Destrieux atlas labels for all fsaverage5 vertices.

    Returns:
        int array of shape (20484,) with region index per vertex.
    """
    from nilearn import datasets

    destrieux = datasets.fetch_atlas_surf_destrieux()
    labels_lh = np.array(destrieux["map_left"])
    labels_rh = np.array(destrieux["map_right"])
    return np.concatenate([labels_lh, labels_rh])


def build_weight_vector(
    weights: dict[str, float] | None = None,
    n_vertices: int = 20484,
) -> np.ndarray:
    """Build a per-vertex weight vector from region group weights.

    Vertices in named groups get the specified weight; all others get 1.0.

    Args:
        weights: Dict mapping group name to weight multiplier.
        n_vertices: Expected number of vertices.

    Returns:
        float32 array of shape (n_vertices,).
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    labels = load_vertex_labels()
    assert len(labels) == n_vertices, f"Expected {n_vertices} vertices, got {len(labels)}"

    w = np.ones(n_vertices, dtype=np.float32)
    for group_name, multiplier in weights.items():
        if group_name not in REGION_GROUPS:
            log.warning("Unknown region group: %s", group_name)
            continue
        region_ids = REGION_GROUPS[group_name]
        mask = np.isin(labels, region_ids)
        w[mask] = multiplier

    log.info(
        "Region weights: %s → %.0f auditory, %.0f limbic, %.0f prefrontal vertices weighted",
        weights,
        np.isin(labels, REGION_GROUPS["auditory"]).sum(),
        np.isin(labels, REGION_GROUPS["limbic"]).sum(),
        np.isin(labels, REGION_GROUPS["prefrontal"]).sum(),
    )
    return w
