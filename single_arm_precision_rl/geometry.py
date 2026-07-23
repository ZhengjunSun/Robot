from __future__ import annotations

import math

import numpy as np


def clip_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm <= float(max_norm) or norm <= 1e-12:
        return value
    return value * (float(max_norm) / norm)


def weighted_precision_error(
    lateral_mm: float,
    depth_error_mm: float,
    axis_error_deg: float,
    target_standoff_mm: float,
) -> float:
    axis_equivalent = float(target_standoff_mm) * math.sin(math.radians(float(axis_error_deg)))
    return float(
        math.sqrt(
            float(lateral_mm) ** 2
            + float(depth_error_mm) ** 2
            + axis_equivalent**2
        )
    )
