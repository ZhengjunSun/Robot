from __future__ import annotations

import numpy as np
import pytest


def test_unified_meca500_uses_one_six_axis_state_for_both_views() -> None:
    pytest.importorskip("mujoco")
    from real_3d_alignment.meca500_visual_env import (
        Meca500VisualAlignmentPlant,
    )

    plant = Meca500VisualAlignmentPlant(
        image_size_px=(320, 240),
        settle_steps=20,
    )
    try:
        eye = plant.capture_rgb()
        world = plant.capture_overview_rgb()
        before = plant.joint_positions_rad()
        plant.apply_camera_translation_mm(
            (0.30, -0.20, 0.0),
            maximum_joint_step_deg=3.0,
        )
        after = plant.joint_positions_rad()

        assert eye.shape == (240, 320, 3)
        assert world.shape == (240, 320, 3)
        assert np.count_nonzero(np.abs(after - before) > 1e-7) >= 3
        assert np.isclose(np.linalg.norm(plant.tool_insertion_axis_world()), 1.0)
    finally:
        plant.close()
