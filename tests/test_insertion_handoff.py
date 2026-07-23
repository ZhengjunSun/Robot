from __future__ import annotations

import pytest

from real_3d_alignment.insertion_handoff import (
    TraditionalInsertionHandoffController,
)
from real_3d_alignment.staged_alignment import FineObservation


def safe_fine() -> FineObservation:
    return FineObservation(
        image_center_px=(320.0, 240.0),
        outer_center_px=(320.1, 240.0),
        inner_center_px=(320.0, 240.0),
        lateral_error_mm=0.01,
        axis_error_deg=0.1,
        standoff_error_mm=0.1,
        reprojection_error_px=0.2,
        quality_gate_pass=True,
    )


def test_insertion_requires_explicit_handoff() -> None:
    decision = TraditionalInsertionHandoffController().decide(
        insertion_handoff_ready=False,
        fine=safe_fine(),
        current_extension_mm=0.0,
    )

    assert not decision.allow_motion
    assert decision.commanded_step_mm == 0.0


def test_insertion_step_is_bounded_and_stops_at_target() -> None:
    controller = TraditionalInsertionHandoffController()
    first = controller.decide(
        insertion_handoff_ready=True,
        fine=safe_fine(),
        current_extension_mm=0.0,
    )
    complete = controller.decide(
        insertion_handoff_ready=True,
        fine=safe_fine(),
        current_extension_mm=controller.target_extension_mm,
    )

    assert first.allow_motion
    assert first.commanded_step_mm <= 0.25
    assert complete.complete
    assert not complete.allow_motion


def test_clearance_gate_rejects_unsafe_visual_pose() -> None:
    unsafe = FineObservation(
        **{**safe_fine().__dict__, "lateral_error_mm": 0.24}
    )

    decision = TraditionalInsertionHandoffController().decide(
        insertion_handoff_ready=True,
        fine=unsafe,
        current_extension_mm=10.5,
    )

    assert not decision.allow_motion
    assert decision.reason == "visual_clearance_below_threshold"


def test_mujoco_wall_proxy_records_off_center_contact() -> None:
    pytest.importorskip("mujoco")
    from real_3d_alignment.mujoco_visual_env import (
        MujocoCoarseAlignmentPlant,
    )
    from real_3d_alignment.nih_baseline import NIH_HRA_SCENE

    plant = MujocoCoarseAlignmentPlant(
        NIH_HRA_SCENE,
        image_size_px=(320, 240),
        initial_camera_xy_mm=(0.35, 0.0),
    )
    try:
        plant.apply_camera_xyz_step((0.0, 0.0, -9.49))
        plant.apply_insertion_step(12.5)
        contact = plant.wall_contact_metrics()
    finally:
        plant.close()

    assert contact["wall_contact_detected"]
    assert contact["maximum_normal_force_n"] > 0.0
