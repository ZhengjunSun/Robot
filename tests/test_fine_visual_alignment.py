from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from real_3d_alignment.fine_vision import (
    FineImageBasedVisualServo,
    FineRingDetectorConfig,
    FineServoConfig,
    NihFineRingDetector,
)
from real_3d_alignment.nih_baseline import (
    NIH_HRA_SCENE,
    build_nih_coarse_servo,
    build_nih_fine_detector,
    build_nih_fine_servo,
    build_nih_m3_gate,
    build_nih_traditional_detector,
)
from real_3d_alignment.staged_alignment import AlignmentPhase
from real_3d_alignment.visual_loop import StagedVisualAlignmentLoop


def fine_ring_image(center: tuple[int, int] = (324, 237)) -> np.ndarray:
    image = np.full((480, 640, 3), 215, dtype=np.uint8)
    cv2.circle(image, center, 35, (25, 185, 225), -1)
    cv2.circle(image, center, 13, (12, 14, 18), -1)
    return image


def test_fine_detector_extracts_inner_outer_geometry_from_rgb() -> None:
    detector = NihFineRingDetector(
        FineRingDetectorConfig(
            focal_length_px=600.0,
            trocar_outer_radius_mm=1.32,
            target_standoff_mm=22.6,
        )
    )

    estimate = detector.detect(fine_ring_image())

    assert estimate is not None
    assert estimate.observation.outer_center_px == pytest.approx(
        (324.0, 237.0), abs=0.5
    )
    assert estimate.observation.inner_center_px == pytest.approx(
        (324.0, 237.0), abs=0.5
    )
    assert estimate.observation.quality_gate_pass
    assert estimate.observation.reprojection_error_px < 0.8


def test_fine_servo_bounds_lateral_and_standoff_commands() -> None:
    detector = NihFineRingDetector(
        FineRingDetectorConfig(
            focal_length_px=600.0,
            trocar_outer_radius_mm=1.32,
            target_standoff_mm=18.0,
        )
    )
    estimate = detector.detect(fine_ring_image(center=(350, 220)))
    assert estimate is not None
    servo = FineImageBasedVisualServo(
        FineServoConfig(
            focal_length_px=600.0,
            target_standoff_mm=18.0,
            maximum_lateral_step_mm=0.5,
            maximum_standoff_step_mm=0.75,
        )
    )

    command = servo.command(estimate.observation)

    assert np.linalg.norm(command.camera_xyz_mm[:2]) <= 0.5 + 1e-9
    assert abs(command.camera_xyz_mm[2]) <= 0.75 + 1e-9


def test_nih_m3_loop_reaches_five_frame_handoff() -> None:
    pytest.importorskip("mujoco")
    from real_3d_alignment.mujoco_visual_env import (
        MujocoCoarseAlignmentPlant,
    )

    plant = MujocoCoarseAlignmentPlant(
        Path(NIH_HRA_SCENE),
        image_size_px=(640, 480),
        initial_camera_xy_mm=(7.0, -5.0),
    )
    try:
        loop = StagedVisualAlignmentLoop(
            plant=plant,
            coarse_detector=build_nih_traditional_detector(),
            coarse_servo=build_nih_coarse_servo(480),
            gate=build_nih_m3_gate(),
            fine_provider=build_nih_fine_detector(480),
            fine_servo=build_nih_fine_servo(480),
        )

        result = loop.run(maximum_steps=60)
    finally:
        plant.close()

    assert result.stop_reason == "insertion_handoff_ready"
    assert result.final_decision.phase is AlignmentPhase.ALIGNED
    assert result.final_decision.stable_frames == 5
    assert result.final_decision.insertion_handoff_ready
