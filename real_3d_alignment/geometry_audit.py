from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def normalized(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize a zero vector.")
    return value / norm


def rotation_vector_between(
    source_axis: np.ndarray,
    target_axis: np.ndarray,
) -> np.ndarray:
    """Shortest world-frame rotation vector from source to target."""

    source = normalized(source_axis)
    target = normalized(target_axis)
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if sine <= 1e-12:
        if cosine > 0.0:
            return np.zeros(3, dtype=np.float64)
        helper = np.asarray(
            [1.0, 0.0, 0.0]
            if abs(source[0]) < 0.9
            else [0.0, 1.0, 0.0],
            dtype=np.float64,
        )
        return normalized(np.cross(source, helper)) * math.pi
    return cross / sine * math.atan2(sine, cosine)


def rotation_matrix_from_local_vector_deg(
    rotation_camera_deg: np.ndarray,
) -> np.ndarray:
    """Rodrigues rotation for a camera-local rotation vector."""

    rotation_vector = np.deg2rad(
        np.asarray(rotation_camera_deg, dtype=np.float64)
    )
    angle = float(np.linalg.norm(rotation_vector))
    if angle <= 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = rotation_vector / angle
    skew = np.asarray(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    return (
        np.eye(3)
        + math.sin(angle) * skew
        + (1.0 - math.cos(angle)) * (skew @ skew)
    )


def rotation_error_deg(
    source_rotation: np.ndarray,
    target_rotation: np.ndarray,
) -> float:
    """Geodesic angular distance between two rotation matrices."""

    relative = (
        np.asarray(target_rotation, dtype=np.float64).reshape(3, 3)
        @ np.asarray(source_rotation, dtype=np.float64).reshape(3, 3).T
    )
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def axis_error_after_local_rotation_deg(
    *,
    camera_rotation_world: np.ndarray,
    target_axis_world: np.ndarray,
    rotation_camera_deg: np.ndarray,
) -> float:
    """Truth axis error after an ideal local rotation, without running IK."""

    camera_rotation = np.asarray(
        camera_rotation_world,
        dtype=np.float64,
    ).reshape(3, 3)
    rotated = camera_rotation @ rotation_matrix_from_local_vector_deg(
        rotation_camera_deg
    )
    camera_axis = -rotated[:, 2]
    target_axis = normalized(target_axis_world)
    cosine = float(np.clip(np.dot(camera_axis, target_axis), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def signed_target_axis_tilt_camera_deg(
    camera_rotation_world: np.ndarray,
    target_axis_world: np.ndarray,
) -> tuple[float, float]:
    """Signed target-normal tilt relative to camera optical -z.

    Roll around the target normal is intentionally excluded. The two values
    are audit features, not simulator truth inputs for a controller.
    """

    rotation = np.asarray(camera_rotation_world, dtype=np.float64).reshape(
        3,
        3,
    )
    target_camera = normalized(rotation.T @ normalized(target_axis_world))
    forward = max(1e-12, -float(target_camera[2]))
    return (
        math.degrees(math.atan2(float(target_camera[0]), forward)),
        math.degrees(math.atan2(float(target_camera[1]), forward)),
    )


@dataclass(frozen=True)
class CoaxialPoseDelta:
    translation_camera_mm: tuple[float, float, float]
    rotation_camera_deg: tuple[float, float, float]


def privileged_coaxial_pose_delta(
    *,
    camera_position_world: np.ndarray,
    camera_rotation_world: np.ndarray,
    target_position_world: np.ndarray,
    target_axis_world: np.ndarray,
    target_standoff_mm: float,
) -> CoaxialPoseDelta:
    """Compute an offline truth-only delta for geometry auditing.

    This function is intentionally named ``privileged`` and must never be
    called by deployment or visual-control code.
    """

    rotation = np.asarray(camera_rotation_world, dtype=np.float64).reshape(
        3,
        3,
    )
    camera_axis = -rotation[:, 2]
    target_axis = normalized(target_axis_world)
    world_rotation_vector = rotation_vector_between(
        camera_axis,
        target_axis,
    )
    local_rotation_vector = rotation.T @ world_rotation_vector
    desired_camera_position = (
        np.asarray(target_position_world, dtype=np.float64)
        - target_axis * float(target_standoff_mm) * 1e-3
    )
    local_translation_mm = (
        rotation.T
        @ (
            desired_camera_position
            - np.asarray(camera_position_world, dtype=np.float64)
        )
        * 1000.0
    )
    return CoaxialPoseDelta(
        translation_camera_mm=tuple(
            float(value) for value in local_translation_mm
        ),
        rotation_camera_deg=tuple(
            float(value)
            for value in np.rad2deg(local_rotation_vector)
        ),
    )
