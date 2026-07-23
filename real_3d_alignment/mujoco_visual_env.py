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
        initial_camera_xy_mm: tuple[float, float] = (7.0, -5.0),
        settle_steps: int = 20,
        camera_name: str = "eye_in_hand",
        evaluation_target_site: str = "trocar_center_evaluation_only",
        trocar_geom_names: tuple[str, ...] = (
            "trocar_outer_wall",
            "trocar_flange",
            "trocar_lumen_visual",
        ),
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
        self.initial_camera_xy_mm = initial_camera_xy_mm
        self.camera_name = str(camera_name)
        self.evaluation_target_site = str(evaluation_target_site)
        self.camera_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name
        )
        self.target_site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            self.evaluation_target_site,
        )
        if self.camera_id < 0 or self.target_site_id < 0:
            raise ValueError(
                "MuJoCo visual plant requires the configured camera and "
                "evaluation-only target site."
            )
        self.trocar_geom_ids = tuple(
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name
            )
            for geom_name in trocar_geom_names
        )
        if any(geom_id < 0 for geom_id in self.trocar_geom_ids):
            raise ValueError("Configured trocar segmentation geoms are missing.")
        self.eye_scene_option = mujoco.MjvOption()
        self.eye_scene_option.geomgroup[1] = 0
        self.eye_scene_option.sitegroup[1] = 0
        self.reset()

    def reset(
        self, initial_camera_xy_mm: tuple[float, float] | None = None
    ) -> None:
        self.mujoco.mj_resetData(self.model, self.data)
        camera_xy = initial_camera_xy_mm or self.initial_camera_xy_mm
        self.mujoco.mj_forward(self.model, self.data)
        camera_rotation = np.asarray(
            self.data.cam_xmat[self.camera_id], dtype=np.float64
        ).reshape(3, 3)
        world_offset_mm = (
            camera_rotation[:, 0] * float(camera_xy[0])
            + camera_rotation[:, 1] * float(camera_xy[1])
        )
        self.data.qpos[:3] = world_offset_mm * 1e-3
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
        self.renderer.update_scene(
            self.data,
            camera=self.camera_name,
            scene_option=self.eye_scene_option,
        )
        return self.renderer.render().copy()

    def capture_overview_rgb(self) -> np.ndarray:
        self.renderer.update_scene(self.data, camera="overview")
        return self.renderer.render().copy()

    def capture_trocar_segmentation_mask(self) -> np.ndarray:
        """Return privileged trocar pixels for dataset labels/evaluation only."""

        self.renderer.enable_segmentation_rendering()
        try:
            self.renderer.update_scene(
                self.data,
                camera=self.camera_name,
                scene_option=self.eye_scene_option,
            )
            segmentation = self.renderer.render().copy()
        finally:
            self.renderer.disable_segmentation_rendering()
        object_ids = segmentation[:, :, 0]
        return np.isin(object_ids, self.trocar_geom_ids)

    def apply_camera_xy_step(self, command_mm: tuple[float, float]) -> None:
        self.apply_camera_xyz_step((command_mm[0], command_mm[1], 0.0))

    def apply_camera_xyz_step(
        self, command_mm: tuple[float, float, float]
    ) -> None:
        camera_x_mm, camera_y_mm, camera_z_mm = command_mm
        camera_rotation = np.asarray(
            self.data.cam_xmat[self.camera_id], dtype=np.float64
        ).reshape(3, 3)
        world_step_mm = (
            camera_rotation[:, 0] * camera_x_mm
            + camera_rotation[:, 1] * camera_y_mm
            + camera_rotation[:, 2] * camera_z_mm
        )
        for index in range(3):
            self.data.ctrl[index] = np.clip(
                self.data.qpos[index] + world_step_mm[index] * 1e-3,
                self.model.jnt_range[index, 0],
                self.model.jnt_range[index, 1],
            )
        self._settle()

    def evaluation_lateral_error_mm(self) -> float:
        """Privileged metric for evaluation only; never used by the controller."""

        camera_rotation = np.asarray(
            self.data.cam_xmat[self.camera_id], dtype=np.float64
        ).reshape(3, 3)
        optical_axis = -camera_rotation[:, 2]
        delta = (
            np.asarray(self.data.site_xpos[self.target_site_id], dtype=np.float64)
            - np.asarray(self.data.cam_xpos[self.camera_id], dtype=np.float64)
        )
        lateral = delta - float(np.dot(delta, optical_axis)) * optical_axis
        return float(np.linalg.norm(lateral) * 1000.0)

    def close(self) -> None:
        self.renderer.close()
