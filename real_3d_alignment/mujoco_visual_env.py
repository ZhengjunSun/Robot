from __future__ import annotations

from pathlib import Path

import numpy as np


class MujocoCoarseAlignmentPlant:
    """RGB-only control boundary for the lightweight MuJoCo M0/M1 scene."""

    def __init__(
        self,
        xml_path: str | Path,
        *,
        image_size_px: tuple[int, int] = (640, 480),
        initial_lateral_yz_mm: tuple[float, float] = (7.0, -5.0),
        settle_steps: int = 20,
    ):
        try:
            import mujoco
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("MuJoCo is required for the visual alignment plant.") from exc

        self.mujoco = mujoco
        self.xml_path = Path(xml_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        width, height = image_size_px
        self.renderer = mujoco.Renderer(self.model, width=width, height=height)
        self.width = int(width)
        self.height = int(height)
        self.settle_steps = int(settle_steps)
        self.initial_lateral_yz_mm = initial_lateral_yz_mm
        self.reset()

    def reset(
        self, initial_lateral_yz_mm: tuple[float, float] | None = None
    ) -> None:
        self.mujoco.mj_resetData(self.model, self.data)
        lateral_yz = initial_lateral_yz_mm or self.initial_lateral_yz_mm
        self.data.qpos[:3] = np.asarray(
            [0.0, lateral_yz[0] * 1e-3, lateral_yz[1] * 1e-3],
            dtype=np.float64,
        )
        self.data.ctrl[:3] = self.data.qpos[:3]
        self.data.qvel[:3] = 0.0
        self.mujoco.mj_forward(self.model, self.data)
        self._settle()

    def _settle(self) -> None:
        for _ in range(max(1, self.settle_steps)):
            self.mujoco.mj_step(self.model, self.data)
        # mj_step integrates qpos at the end of its pipeline. Refresh derived
        # camera/site poses so the rendered frame and evaluation metrics refer
        # to the same state as data.qpos.
        self.mujoco.mj_forward(self.model, self.data)

    def capture_rgb(self) -> np.ndarray:
        self.renderer.update_scene(self.data, camera="eye_in_hand")
        return self.renderer.render().copy()

    def capture_overview_rgb(self) -> np.ndarray:
        self.renderer.update_scene(self.data, camera="overview")
        return self.renderer.render().copy()

    def apply_camera_xy_step(self, command_mm: tuple[float, float]) -> None:
        camera_x_mm, camera_y_mm = command_mm
        # Camera +x is world -y; camera +y is world +z.
        self.data.ctrl[1] = np.clip(
            self.data.qpos[1] - camera_x_mm * 1e-3,
            self.model.jnt_range[1, 0],
            self.model.jnt_range[1, 1],
        )
        self.data.ctrl[2] = np.clip(
            self.data.qpos[2] + camera_y_mm * 1e-3,
            self.model.jnt_range[2, 0],
            self.model.jnt_range[2, 1],
        )
        self._settle()

    def evaluation_lateral_error_mm(self) -> float:
        """Privileged metric for evaluation only; never used by the controller."""

        return float(np.linalg.norm(self.data.qpos[1:3]) * 1000.0)

    def close(self) -> None:
        self.renderer.close()
