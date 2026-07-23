from __future__ import annotations

from pathlib import Path

import numpy as np

from .coarse_vision import (
    CoarseImageBasedVisualServo,
    CoarseServoConfig,
    TraditionalRingDetector,
    TraditionalRingDetectorConfig,
)
from .staged_alignment import AlignmentThresholds, StagedAlignmentGate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NIH_HRA_SCENE = (
    PROJECT_ROOT
    / "3d_modeling"
    / "mujoco"
    / "single_arm_trocar_visual_alignment_nih_hra.xml"
)
NIH_HRA_EYE_SOURCE = "NIH 3DPX-020963 / HRA Visible Human female right eye v1.2"
EYE_IN_HAND_FOVY_DEG = 41.9741
TROCAR_FLANGE_OUTER_RADIUS_MM = 1.32


def eye_in_hand_focal_length_px(image_height_px: int) -> float:
    return float(
        0.5
        * image_height_px
        / np.tan(np.deg2rad(EYE_IN_HAND_FOVY_DEG) / 2.0)
    )


def build_nih_traditional_detector() -> TraditionalRingDetector:
    return TraditionalRingDetector(
        TraditionalRingDetectorConfig(
            hue_ranges=((82, 112),),
            saturation_minimum=55,
            value_minimum=35,
        )
    )


def build_nih_coarse_servo(
    image_height_px: int,
    *,
    maximum_step_mm: float = 1.5,
) -> CoarseImageBasedVisualServo:
    return CoarseImageBasedVisualServo(
        CoarseServoConfig(
            focal_length_px=eye_in_hand_focal_length_px(image_height_px),
            trocar_outer_radius_mm=TROCAR_FLANGE_OUTER_RADIUS_MM,
            maximum_step_mm=maximum_step_mm,
        )
    )


def build_nih_coarse_gate(
    *, transition_center_error_px: float
) -> StagedAlignmentGate:
    return StagedAlignmentGate(
        AlignmentThresholds(
            minimum_coarse_confidence=0.35,
            coarse_to_fine_center_error_px=transition_center_error_px,
        )
    )
