from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from real_3d_alignment.nih_baseline import (
    NIH_HRA_SCENE,
    build_nih_coarse_gate,
    build_nih_coarse_servo,
    build_nih_traditional_detector,
)
from real_3d_alignment.mujoco_visual_env import MujocoCoarseAlignmentPlant
from real_3d_alignment.staged_alignment import AlignmentPhase
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
    plant = MujocoCoarseAlignmentPlant(
        NIH_HRA_SCENE,
        image_size_px=(320, 240),
        initial_camera_xy_mm=initial_yz_mm,
        settle_steps=20,
    )
    try:
        loop = StagedVisualAlignmentLoop(
            plant=plant,
            coarse_detector=build_nih_traditional_detector(),
            coarse_servo=build_nih_coarse_servo(240),
            gate=build_nih_coarse_gate(transition_center_error_px=8.0),
        )

        result = loop.run(maximum_steps=30)

        assert result.final_decision.phase is AlignmentPhase.FINE
        assert result.records[-1].center_error_px is not None
        assert result.records[-1].center_error_px <= 8.0
        assert plant.evaluation_lateral_error_mm() <= 1.6
    finally:
        plant.close()
