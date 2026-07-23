from __future__ import annotations

import cv2
import numpy as np
import pytest

from real_3d_alignment.coarse_vision import (
    CoarseImageBasedVisualServo,
    CoarseServoConfig,
    TraditionalRingDetector,
)
from real_3d_alignment.staged_alignment import (
    AlignmentPhase,
    AlignmentThresholds,
    StagedAlignmentGate,
)
from real_3d_alignment.visual_loop import StagedVisualAlignmentLoop


def ring_image(
    center: tuple[int, int] = (410, 180),
    *,
    size: tuple[int, int] = (640, 480),
) -> np.ndarray:
    width, height = size
    image = np.full((height, width, 3), 225, dtype=np.uint8)
    cv2.circle(image, center, 34, (215, 25, 18), -1)
    cv2.circle(image, center, 17, (15, 18, 22), -1)
    return image


def test_traditional_detector_uses_rgb_ring_center() -> None:
    detection = TraditionalRingDetector().detect(ring_image())

    assert detection is not None
    assert detection.observation.target_center_px == pytest.approx((410.0, 180.0), abs=0.5)
    assert detection.observation.confidence > 0.7
    assert detection.radius_px == pytest.approx(34.0, abs=1.0)


def test_traditional_detector_rejects_image_without_red_ring() -> None:
    image = np.full((480, 640, 3), 220, dtype=np.uint8)

    assert TraditionalRingDetector().detect(image) is None


def test_traditional_detector_tolerates_partial_ring_occlusion() -> None:
    image = ring_image(center=(320, 240))
    cv2.rectangle(image, (300, 205), (319, 275), (20, 22, 25), -1)

    detection = TraditionalRingDetector().detect(image)

    assert detection is not None
    assert detection.observation.confidence >= 0.5
    assert detection.observation.target_center_px == pytest.approx(
        (320.0, 240.0), abs=6.0
    )


def test_ibvs_command_is_bounded_and_has_camera_axis_signs() -> None:
    detection = TraditionalRingDetector().detect(ring_image())
    assert detection is not None
    servo = CoarseImageBasedVisualServo(
        CoarseServoConfig(focal_length_px=600.0, maximum_step_mm=1.0)
    )

    command = servo.command(detection)

    assert command.camera_xy_mm[0] > 0.0
    assert command.camera_xy_mm[1] > 0.0
    assert command.norm_mm <= 1.0 + 1e-9


class PixelPlant:
    def __init__(self) -> None:
        self.center = np.asarray([470.0, 150.0], dtype=np.float64)

    def capture_rgb(self) -> np.ndarray:
        center = tuple(int(round(value)) for value in self.center)
        return ring_image(center)

    def apply_camera_xy_step(self, command_mm: tuple[float, float]) -> None:
        # Image motion is opposite camera motion; v is opposite camera +y.
        self.center[0] -= 18.0 * command_mm[0]
        self.center[1] += 18.0 * command_mm[1]


def test_unified_loop_reaches_fine_region_using_pixels_only() -> None:
    plant = PixelPlant()
    loop = StagedVisualAlignmentLoop(
        plant=plant,
        coarse_detector=TraditionalRingDetector(),
        coarse_servo=CoarseImageBasedVisualServo(
            CoarseServoConfig(
                focal_length_px=600.0,
                gain=0.9,
                maximum_step_mm=2.5,
            )
        ),
        gate=StagedAlignmentGate(
            AlignmentThresholds(coarse_to_fine_center_error_px=12.0)
        ),
    )

    result = loop.run(maximum_steps=40)

    assert result.stop_reason == "coarse_alignment_complete_waiting_for_fine_provider"
    assert result.final_decision.phase is AlignmentPhase.FINE
    assert result.records[0].center_error_px is not None
    assert result.records[-1].center_error_px is not None
    assert result.records[-1].center_error_px < result.records[0].center_error_px
