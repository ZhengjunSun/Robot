from __future__ import annotations

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MECA500_NIH_PRESENTATION_SCENE = (
    ROOT
    / "3d_modeling"
    / "mujoco"
    / "meca500_r4_ophthalmic_visual_execution_scene.xml"
)


class Meca500PresentationRenderer:
    """Full-arm companion renderer for M4 videos.

    The M4 controller plant remains the RGB-driven Cartesian validation plant.
    This renderer only provides a synchronized, explicitly labelled Meca500
    kinematic visualization and is never read by the controller.
    """

    _ALIGNED_Q_DEG = np.asarray(
        [-90.377406, -21.5459251, -72.4287403, 0.0, 93.9103863, -88.3739],
        dtype=np.float64,
    )
    _SEARCH_Q_DEG = _ALIGNED_Q_DEG + np.asarray(
        [12.0, -5.0, 5.0, 5.0, -3.0, 0.0],
        dtype=np.float64,
    )
    _INSERTED_Q_DEG = np.asarray(
        [-90.377406, -28.7207737, -52.1346541, 0.0, 80.8554277, -88.3739],
        dtype=np.float64,
    )

    def __init__(
        self,
        *,
        width: int,
        height: int,
        target_extension_mm: float,
        xml_path: str | Path = MECA500_NIH_PRESENTATION_SCENE,
    ):
        try:
            import mujoco
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("MuJoCo is required for the presentation renderer.") from exc

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(Path(xml_path)))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(
            self.model,
            width=int(width),
            height=int(height),
        )
        self.target_extension_mm = float(target_extension_mm)
        self._joint_qpos_addresses = tuple(
            int(
                self.model.jnt_qposadr[
                    mujoco.mj_name2id(
                        self.model,
                        mujoco.mjtObj.mjOBJ_JOINT,
                        f"joint_{index}",
                    )
                ]
            )
            for index in range(1, 7)
        )

    def capture_rgb(
        self,
        *,
        alignment_progress: float,
        insertion_extension_mm: float,
    ) -> np.ndarray:
        alignment_alpha = float(np.clip(alignment_progress, 0.0, 1.0))
        insertion_alpha = float(
            np.clip(
                insertion_extension_mm / max(self.target_extension_mm, 1e-6),
                0.0,
                1.0,
            )
        )
        search_q = np.deg2rad(self._SEARCH_Q_DEG)
        aligned_q = np.deg2rad(self._ALIGNED_Q_DEG)
        inserted_q = np.deg2rad(self._INSERTED_Q_DEG)
        alignment_q = (1.0 - alignment_alpha) * search_q + alignment_alpha * aligned_q
        q = (1.0 - insertion_alpha) * alignment_q + insertion_alpha * inserted_q
        for address, value in zip(self._joint_qpos_addresses, q):
            self.data.qpos[address] = value
        self.mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(
            self.data,
            camera="full_arm_presentation",
        )
        return self.renderer.render().copy()

    def close(self) -> None:
        self.renderer.close()
