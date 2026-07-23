from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("mujoco")

from real_3d_alignment.coarse_vision import (
    CoarseImageBasedVisualServo,
    CoarseServoConfig,
    TraditionalRingDetector,
)
from real_3d_alignment.mujoco_visual_env import MujocoCoarseAlignmentPlant
from real_3d_alignment.staged_alignment import (
    AlignmentPhase,
    AlignmentThresholds,
    StagedAlignmentGate,
)
from real_3d_alignment.visual_loop import StagedVisualAlignmentLoop


@pytest.mark.parametrize(
    "initial_yz_mm",
    [
        (7.0, -5.0),
        (-7.0, -5.0),
        (6.0, 5.0),
        (-6.0, 5.0),
    ],
)
def test_mujoco_rgb_coarse_loop_reaches_fine_region(
    initial_yz_mm: tuple[float, float],
) -> None:
    root = Path(__file__).resolve().parents[1]
    plant = MujocoCoarseAlignmentPlant(
        root
        / "3d_modeling"
        / "mujoco"
        / "single_arm_trocar_visual_alignment.xml",
        image_size_px=(320, 240),
        initial_lateral_yz_mm=initial_yz_mm,
        settle_steps=20,
    )
    try:
        loop = StagedVisualAlignmentLoop(
            plant=plant,
            coarse_detector=TraditionalRingDetector(),
            coarse_servo=CoarseImageBasedVisualServo(
                CoarseServoConfig(
                    focal_length_px=312.7,
                    gain=0.65,
                    maximum_step_mm=1.5,
                )
            ),
            gate=StagedAlignmentGate(
                AlignmentThresholds(coarse_to_fine_center_error_px=8.0)
            ),
        )

        result = loop.run(maximum_steps=30)

        assert result.final_decision.phase is AlignmentPhase.FINE
        assert result.records[-1].center_error_px is not None
        assert result.records[-1].center_error_px <= 8.0
        assert plant.evaluation_lateral_error_mm() <= 1.6
    finally:
        plant.close()
