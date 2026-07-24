from __future__ import annotations

import cv2
import numpy as np
import pytest

from real_3d_alignment.multiview_circle_pose import (
    AdditionalCircleObservation,
    CalibratedCameraView,
    EllipseObservation,
    MultiviewCirclePose,
    active_observation_from_multiview_pose,
    estimate_multiview_circle_pose,
    estimate_multiview_circle_pose_multistart,
    ellipse_center_scale_residual,
    project_world_points,
    sample_circle_world,
)


def _ellipse_from_points(points_px: np.ndarray) -> EllipseObservation:
    center, axes, angle = cv2.fitEllipse(
        np.asarray(points_px, dtype=np.float32).reshape(-1, 1, 2)
    )
    first, second = axes
    if first >= second:
        major, minor, major_angle = first, second, angle
    else:
        major, minor, major_angle = second, first, angle + 90.0
    return EllipseObservation(
        center_px=(float(center[0]), float(center[1])),
        major_diameter_px=float(major),
        minor_diameter_px=float(minor),
        major_angle_deg=float(major_angle),
    )


def test_multiview_circle_recovers_center_and_normal() -> None:
    focal = 1250.0
    principal = (640.0, 480.0)
    radius = 0.00132
    center = np.asarray([0.0004, -0.0002, -0.022])
    normal = np.asarray([0.10, -0.14, -0.985])
    normal /= np.linalg.norm(normal)
    circle = sample_circle_world(center, normal, radius, sample_count=96)
    views = []
    for x in (-0.0015, 0.0015):
        placeholder = CalibratedCameraView(
            position_world_m=np.asarray([x, 0.0, 0.0]),
            rotation_world=np.eye(3),
            ellipse=EllipseObservation((0.0, 0.0), 1.0, 1.0, 0.0),
        )
        ellipse = _ellipse_from_points(
            project_world_points(
                circle,
                placeholder,
                focal_length_px=focal,
                principal_point_px=principal,
            )
        )
        views.append(
            CalibratedCameraView(
                position_world_m=placeholder.position_world_m,
                rotation_world=placeholder.rotation_world,
                ellipse=ellipse,
            )
        )
    estimate = estimate_multiview_circle_pose(
        views,
        radius_m=radius,
        focal_length_px=focal,
        principal_point_px=principal,
        initial_center_world_m=center + np.asarray([0.0003, -0.0002, 0.001]),
        initial_normal_world=np.asarray([0.0, 0.0, -1.0]),
    )
    assert estimate.success
    assert np.linalg.norm(estimate.center_world_m - center) * 1000.0 < 0.02
    angle = np.rad2deg(
        np.arccos(np.clip(np.dot(estimate.normal_world, normal), -1.0, 1.0))
    )
    assert angle < 0.2
    assert estimate.rms_normalized_conic_residual < 1e-4


def test_multiview_requires_two_views() -> None:
    with pytest.raises(ValueError, match="At least two"):
        estimate_multiview_circle_pose(
            [],
            radius_m=0.001,
            focal_length_px=1000.0,
            principal_point_px=(0.0, 0.0),
            initial_center_world_m=np.asarray([0.0, 0.0, -0.02]),
            initial_normal_world=np.asarray([0.0, 0.0, -1.0]),
        )


def test_pose_converts_to_active_gate_observation() -> None:
    pose = MultiviewCirclePose(
        center_world_m=np.asarray([0.0001, 0.0, -0.0221]),
        normal_world=np.asarray([0.0, 0.0, -1.0]),
        rms_normalized_conic_residual=0.003,
        covariance=np.eye(6),
        covariance_condition=1.0e7,
        view_count=3,
        success=True,
    )
    active = active_observation_from_multiview_pose(
        pose,
        observation_id=7,
        camera_position_world_m=np.zeros(3),
        tool_axis_world=np.asarray([0.0, 0.0, -1.0]),
        target_standoff_mm=22.0,
        all_views_reachable=True,
        all_rings_detected=True,
    )
    assert active.lateral_error_mm == pytest.approx(0.1)
    assert active.observation_id == 7
    assert active.axis_error_deg == pytest.approx(0.0)
    assert active.standoff_error_mm == pytest.approx(0.1)


def test_joint_inner_outer_multistart_recovers_tilted_plane() -> None:
    focal = 1250.0
    principal = (640.0, 480.0)
    outer_radius = 0.00132
    inner_radius = 0.00050
    center = np.asarray([0.0002, -0.0001, -0.022])
    normal = np.asarray([0.14, 0.08, -0.987])
    normal /= np.linalg.norm(normal)
    views = []
    for position in (
        np.asarray([-0.0015, 0.0, 0.0]),
        np.asarray([0.0, 0.0015, 0.0]),
        np.asarray([0.0015, -0.0015, 0.0]),
    ):
        placeholder = CalibratedCameraView(
            position_world_m=position,
            rotation_world=np.eye(3),
            ellipse=EllipseObservation((0.0, 0.0), 1.0, 1.0, 0.0),
        )
        outer = _ellipse_from_points(
            project_world_points(
                sample_circle_world(
                    center,
                    normal,
                    outer_radius,
                    sample_count=96,
                ),
                placeholder,
                focal_length_px=focal,
                principal_point_px=principal,
            )
        )
        inner = _ellipse_from_points(
            project_world_points(
                sample_circle_world(
                    center,
                    normal,
                    inner_radius,
                    sample_count=96,
                ),
                placeholder,
                focal_length_px=focal,
                principal_point_px=principal,
            )
        )
        views.append(
            CalibratedCameraView(
                position_world_m=position,
                rotation_world=np.eye(3),
                ellipse=outer,
                additional_circles=(
                    AdditionalCircleObservation(inner_radius, inner),
                ),
            )
        )
    estimate, candidates = estimate_multiview_circle_pose_multistart(
        views,
        radius_m=outer_radius,
        focal_length_px=focal,
        principal_point_px=principal,
        initial_center_world_m=center
        + np.asarray([0.0003, -0.0002, 0.001]),
    )
    assert len(candidates) == 3
    assert estimate.success
    assert np.linalg.norm(estimate.center_world_m - center) * 1000.0 < 0.02
    angle = np.rad2deg(
        np.arccos(np.clip(np.dot(estimate.normal_world, normal), -1.0, 1.0))
    )
    assert angle < 0.2


def test_center_scale_constraint_ignores_unstable_ellipse_angle() -> None:
    points = np.asarray(
        [[12.0, 10.0], [10.0, 12.0], [8.0, 10.0], [10.0, 8.0]]
    )
    first = EllipseObservation((10.0, 10.0), 4.0, 4.0, 0.0)
    rotated = EllipseObservation((10.0, 10.0), 4.0, 4.0, 137.0)
    assert ellipse_center_scale_residual(
        points,
        first,
    ) == pytest.approx(
        ellipse_center_scale_residual(points, rotated)
    )
