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
from .fine_vision import (
    FineImageBasedVisualServo,
    FineRingDetectorConfig,
    FineServoConfig,
    NihFineRingDetector,
)


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
M3_TARGET_STANDOFF_MM = 45.0


def eye_in_hand_focal_length_px(image_height_px: int) -> float:
    return float(
        0.5
        * image_height_px
        / np.tan(np.deg2rad(EYE_IN_HAND_FOVY_DEG) / 2.0)
    )


def build_nih_traditional_detector() -> TraditionalRingDetector:
    return TraditionalRingDetector(
        TraditionalRingDetectorConfig(
            hue_ranges=((94, 101),),
            saturation_minimum=55,
            value_minimum=35,
            maximum_radius_px=45.0,
            gaussian_blur_kernel_px=3,
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


def build_nih_fine_detector(image_height_px: int) -> NihFineRingDetector:
    image_scale = float(image_height_px) / 480.0
    return NihFineRingDetector(
        FineRingDetectorConfig(
            focal_length_px=eye_in_hand_focal_length_px(image_height_px),
            trocar_outer_radius_mm=TROCAR_FLANGE_OUTER_RADIUS_MM,
            target_standoff_mm=M3_TARGET_STANDOFF_MM,
            hue_ranges=((94, 101),),
            gaussian_blur_kernel_px=3,
            minimum_outer_area_px2=max(30.0, 180.0 * image_scale**2),
            minimum_inner_area_px2=max(4.0, 20.0 * image_scale**2),
            maximum_outer_diameter_px=120.0 * image_scale,
            maximum_outer_center_error_px=40.0 * image_scale,
        )
    )


def build_nih_fine_servo(image_height_px: int) -> FineImageBasedVisualServo:
    return FineImageBasedVisualServo(
        FineServoConfig(
            focal_length_px=eye_in_hand_focal_length_px(image_height_px),
            target_standoff_mm=M3_TARGET_STANDOFF_MM,
        )
    )


def build_nih_m3_gate() -> StagedAlignmentGate:
    return StagedAlignmentGate(
        AlignmentThresholds(
            minimum_coarse_confidence=0.35,
            coarse_to_fine_center_error_px=8.0,
            maximum_optical_outer_error_px=2.0,
            maximum_outer_inner_concentricity_px=1.5,
            maximum_lateral_error_mm=0.20,
            maximum_axis_error_deg=6.0,
            maximum_standoff_error_mm=0.30,
            maximum_reprojection_error_px=0.80,
            required_stable_frames=5,
        )
    )
