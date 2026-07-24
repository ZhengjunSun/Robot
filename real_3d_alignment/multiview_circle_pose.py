from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .geometry_audit import normalized
from .staged_alignment import ActiveMultiviewObservation


@dataclass(frozen=True)
class EllipseObservation:
    center_px: tuple[float, float]
    major_diameter_px: float
    minor_diameter_px: float
    major_angle_deg: float

    def __post_init__(self) -> None:
        if self.major_diameter_px <= 0.0 or self.minor_diameter_px <= 0.0:
            raise ValueError("Ellipse diameters must be positive.")


@dataclass(frozen=True)
class CalibratedCameraView:
    position_world_m: np.ndarray
    rotation_world: np.ndarray
    ellipse: EllipseObservation


@dataclass(frozen=True)
class MultiviewCirclePose:
    center_world_m: np.ndarray
    normal_world: np.ndarray
    rms_normalized_conic_residual: float
    covariance: np.ndarray
    covariance_condition: float
    view_count: int
    success: bool


def active_observation_from_multiview_pose(
    pose: MultiviewCirclePose,
    *,
    observation_id: int,
    camera_position_world_m: np.ndarray,
    tool_axis_world: np.ndarray,
    target_standoff_mm: float,
    all_views_reachable: bool,
    all_rings_detected: bool,
) -> ActiveMultiviewObservation:
    """Convert an estimated world circle into the fail-closed gate contract."""

    camera_position = np.asarray(
        camera_position_world_m,
        dtype=np.float64,
    ).reshape(3)
    tool_axis = normalized(tool_axis_world)
    relative = pose.center_world_m - camera_position
    axial_distance_mm = float(np.dot(relative, tool_axis) * 1000.0)
    lateral_vector = relative - np.dot(relative, tool_axis) * tool_axis
    normal = normalized(pose.normal_world)
    if float(np.dot(normal, tool_axis)) < 0.0:
        normal = -normal
    cosine = float(np.clip(np.dot(normal, tool_axis), -1.0, 1.0))
    return ActiveMultiviewObservation(
        observation_id=int(observation_id),
        view_count=pose.view_count,
        all_views_reachable=bool(all_views_reachable),
        all_rings_detected=bool(all_rings_detected),
        lateral_error_mm=float(np.linalg.norm(lateral_vector) * 1000.0),
        axis_error_deg=float(np.rad2deg(np.arccos(cosine))),
        standoff_error_mm=axial_distance_mm - float(target_standoff_mm),
        normalized_conic_residual=pose.rms_normalized_conic_residual,
        covariance_condition=pose.covariance_condition,
    )


def project_world_points(
    points_world_m: np.ndarray,
    view: CalibratedCameraView,
    *,
    focal_length_px: float,
    principal_point_px: tuple[float, float],
) -> np.ndarray:
    """Project using MuJoCo's x-right, y-up, optical -z convention."""

    points = np.asarray(points_world_m, dtype=np.float64).reshape(-1, 3)
    rotation = np.asarray(view.rotation_world, dtype=np.float64).reshape(3, 3)
    position = np.asarray(view.position_world_m, dtype=np.float64).reshape(3)
    local = (rotation.T @ (points - position).T).T
    depth = -local[:, 2]
    if np.any(depth <= 1e-6):
        raise ValueError("Circle projects behind the camera.")
    cx, cy = principal_point_px
    return np.column_stack(
        (
            cx + focal_length_px * local[:, 0] / depth,
            cy - focal_length_px * local[:, 1] / depth,
        )
    )


def _circle_basis(normal_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = normalized(normal_world)
    helper = (
        np.asarray([1.0, 0.0, 0.0])
        if abs(normal[0]) < 0.8
        else np.asarray([0.0, 1.0, 0.0])
    )
    first = normalized(np.cross(normal, helper))
    return first, np.cross(normal, first)


def sample_circle_world(
    center_world_m: np.ndarray,
    normal_world: np.ndarray,
    radius_m: float,
    *,
    sample_count: int = 48,
) -> np.ndarray:
    if radius_m <= 0.0:
        raise ValueError("Circle radius must be positive.")
    first, second = _circle_basis(normal_world)
    angles = np.linspace(0.0, 2.0 * np.pi, sample_count, endpoint=False)
    return (
        np.asarray(center_world_m, dtype=np.float64).reshape(1, 3)
        + float(radius_m)
        * (
            np.cos(angles)[:, None] * first
            + np.sin(angles)[:, None] * second
        )
    )


def ellipse_normalized_radius_residual(
    points_px: np.ndarray,
    ellipse: EllipseObservation,
) -> np.ndarray:
    shifted = (
        np.asarray(points_px, dtype=np.float64)
        - np.asarray(ellipse.center_px, dtype=np.float64)
    )
    angle = np.deg2rad(float(ellipse.major_angle_deg))
    major_direction = np.asarray([np.cos(angle), np.sin(angle)])
    minor_direction = np.asarray([-np.sin(angle), np.cos(angle)])
    local_major = shifted @ major_direction
    local_minor = shifted @ minor_direction
    radius = np.sqrt(
        np.square(local_major / (0.5 * ellipse.major_diameter_px))
        + np.square(local_minor / (0.5 * ellipse.minor_diameter_px))
    )
    return radius - 1.0


def estimate_multiview_circle_pose(
    views: list[CalibratedCameraView],
    *,
    radius_m: float,
    focal_length_px: float,
    principal_point_px: tuple[float, float],
    initial_center_world_m: np.ndarray,
    initial_normal_world: np.ndarray,
) -> MultiviewCirclePose:
    """Fit one fixed 3-D circle to calibrated ellipse observations.

    This estimator consumes only calibrated camera poses and image ellipses.
    Simulator target truth is not an input.
    """

    if len(views) < 2:
        raise ValueError("At least two calibrated views are required.")
    initial_normal = normalized(initial_normal_world)
    x0 = np.concatenate(
        (
            np.asarray(initial_center_world_m, dtype=np.float64).reshape(3),
            initial_normal,
        )
    )

    def residual(parameters: np.ndarray) -> np.ndarray:
        center = parameters[:3]
        raw_normal = parameters[3:]
        normal_norm = float(np.linalg.norm(raw_normal))
        if normal_norm <= 1e-9:
            return np.full(48 * len(views) + 1, 1e3)
        normal = raw_normal / normal_norm
        circle = sample_circle_world(center, normal, radius_m)
        values = []
        try:
            for view in views:
                pixels = project_world_points(
                    circle,
                    view,
                    focal_length_px=focal_length_px,
                    principal_point_px=principal_point_px,
                )
                values.append(
                    ellipse_normalized_radius_residual(
                        pixels,
                        view.ellipse,
                    )
                )
        except ValueError:
            return np.full(48 * len(views) + 1, 1e3)
        values.append(np.asarray([(normal_norm - 1.0) * 10.0]))
        return np.concatenate(values)

    result = least_squares(
        residual,
        x0,
        method="trf",
        max_nfev=400,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )
    normal = normalized(result.x[3:])
    if float(np.dot(normal, initial_normal)) < 0.0:
        normal = -normal
    jacobian_information = result.jac.T @ result.jac
    covariance = np.linalg.pinv(jacobian_information)
    condition = float(np.linalg.cond(jacobian_information))
    rms = float(np.sqrt(np.mean(np.square(residual(result.x)))))
    return MultiviewCirclePose(
        center_world_m=result.x[:3].copy(),
        normal_world=normal,
        rms_normalized_conic_residual=rms,
        covariance=covariance,
        covariance_condition=condition,
        view_count=len(views),
        success=bool(result.success and np.isfinite(condition)),
    )
