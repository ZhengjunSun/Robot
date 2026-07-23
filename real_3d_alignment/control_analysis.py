from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def axis_angle_deg(axis: np.ndarray) -> float:
    axis = _unit(axis)
    return float(math.degrees(math.acos(float(np.clip(axis[2], -1.0, 1.0)))))


def error_metrics(p_mm: np.ndarray, axis: np.ndarray, target_distance_mm: float) -> dict[str, Any]:
    axis = _unit(axis)
    lateral = float(np.linalg.norm(p_mm[:2]))
    depth = float(p_mm[2] - target_distance_mm)
    axis_deg = axis_angle_deg(axis)
    axis_equivalent = float(target_distance_mm * math.sin(math.radians(axis_deg)))
    weighted = float(math.sqrt(lateral**2 + depth**2 + axis_equivalent**2))
    return {
        "camera_position_mm": [float(v) for v in p_mm],
        "trocar_axis_camera": [float(v) for v in axis],
        "lateral_error_mm": lateral,
        "depth_error_mm": depth,
        "axis_angle_error_deg": axis_deg,
        "axis_equivalent_mm": axis_equivalent,
        "weighted_3d_error_mm": weighted,
    }


def simulate_candidate(
    *,
    name: str,
    p_mm: np.ndarray,
    axis: np.ndarray,
    target_distance_mm: float,
    translation_step_mm: np.ndarray,
    rotvec_step_rad: np.ndarray,
) -> dict[str, Any]:
    R_delta = R.from_rotvec(rotvec_step_rad).as_matrix()
    next_p = R_delta.T @ (p_mm - translation_step_mm)
    next_axis = R_delta.T @ axis
    before = error_metrics(p_mm, axis, target_distance_mm)
    after = error_metrics(next_p, next_axis, target_distance_mm)
    return {
        "name": name,
        "translation_step_camera_mm": [float(v) for v in translation_step_mm],
        "rotation_step_camera_rotvec_deg": [float(math.degrees(v)) for v in rotvec_step_rad],
        "rotation_step_angle_deg": float(math.degrees(np.linalg.norm(rotvec_step_rad))),
        "before": before,
        "after": after,
        "delta_weighted_3d_error_mm": float(after["weighted_3d_error_mm"] - before["weighted_3d_error_mm"]),
        "improved": bool(after["weighted_3d_error_mm"] < before["weighted_3d_error_mm"]),
    }


def rank_step_candidates(
    *,
    p_mm: np.ndarray,
    axis: np.ndarray,
    target_distance_mm: float,
    translation_step_mm: np.ndarray,
    rotvec_step_rad: np.ndarray,
) -> dict[str, Any]:
    p_mm = np.asarray(p_mm, dtype=np.float64)
    axis = _unit(np.asarray(axis, dtype=np.float64))
    translation_step_mm = np.asarray(translation_step_mm, dtype=np.float64)
    rotvec_step_rad = np.asarray(rotvec_step_rad, dtype=np.float64)
    zero_t = np.zeros(3, dtype=np.float64)
    zero_r = np.zeros(3, dtype=np.float64)

    candidates = [
        simulate_candidate(
            name="translation_only",
            p_mm=p_mm,
            axis=axis,
            target_distance_mm=target_distance_mm,
            translation_step_mm=translation_step_mm,
            rotvec_step_rad=zero_r,
        ),
        simulate_candidate(
            name="rotation_only",
            p_mm=p_mm,
            axis=axis,
            target_distance_mm=target_distance_mm,
            translation_step_mm=zero_t,
            rotvec_step_rad=rotvec_step_rad,
        ),
        simulate_candidate(
            name="combined_translation_rotation",
            p_mm=p_mm,
            axis=axis,
            target_distance_mm=target_distance_mm,
            translation_step_mm=translation_step_mm,
            rotvec_step_rad=rotvec_step_rad,
        ),
    ]
    ranked = sorted(candidates, key=lambda item: item["after"]["weighted_3d_error_mm"])
    return {
        "best_candidate": ranked[0]["name"],
        "best_delta_weighted_3d_error_mm": ranked[0]["delta_weighted_3d_error_mm"],
        "candidates": ranked,
    }
