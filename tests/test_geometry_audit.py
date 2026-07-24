from __future__ import annotations

import numpy as np
import pytest

from real_3d_alignment.geometry_audit import (
    axis_error_after_local_rotation_deg,
    privileged_coaxial_pose_delta,
    rotation_error_deg,
    rotation_matrix_from_local_vector_deg,
    rotation_vector_between,
    signed_target_axis_tilt_camera_deg,
)


def test_rotation_vector_aligns_axes() -> None:
    source = np.asarray([0.0, 0.0, -1.0])
    target = np.asarray([0.0, 1.0, 0.0])
    vector = rotation_vector_between(source, target)
    assert np.rad2deg(np.linalg.norm(vector)) == pytest.approx(90.0)
    assert vector[0] > 0.0


def test_signed_axis_tilt_is_zero_when_target_matches_optical_axis() -> None:
    tilt = signed_target_axis_tilt_camera_deg(
        np.eye(3),
        np.asarray([0.0, 0.0, -1.0]),
    )
    assert tilt == pytest.approx((0.0, 0.0))


def test_privileged_coaxial_delta_has_expected_depth_sign() -> None:
    delta = privileged_coaxial_pose_delta(
        camera_position_world=np.zeros(3),
        camera_rotation_world=np.eye(3),
        target_position_world=np.asarray([0.0, 0.0, -0.020]),
        target_axis_world=np.asarray([0.0, 0.0, -1.0]),
        target_standoff_mm=22.0,
    )
    # Camera local +z points away from a target on optical local -z.
    assert delta.translation_camera_mm == pytest.approx((0.0, 0.0, 2.0))
    assert delta.rotation_camera_deg == pytest.approx((0.0, 0.0, 0.0))


def test_local_rotation_matrix_and_geodesic_error() -> None:
    rotation = rotation_matrix_from_local_vector_deg(
        np.asarray([6.0, 0.0, 0.0])
    )
    assert rotation_error_deg(np.eye(3), rotation) == pytest.approx(6.0)


def test_fixed_local_rotation_creates_axis_error_at_coaxial_pose() -> None:
    error = axis_error_after_local_rotation_deg(
        camera_rotation_world=np.eye(3),
        target_axis_world=np.asarray([0.0, 0.0, -1.0]),
        rotation_camera_deg=np.asarray([-6.0, 0.94, 0.0]),
    )
    assert error == pytest.approx(np.hypot(6.0, 0.94), abs=0.01)
