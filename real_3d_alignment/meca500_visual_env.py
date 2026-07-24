from __future__ import annotations

from pathlib import Path

import numpy as np

from .meca500_presentation import MECA500_NIH_PRESENTATION_SCENE


class Meca500VisualAlignmentPlant:
    """Unified six-axis Meca500 plant for control and both video views."""

    HOME_Q_DEG = np.zeros(6, dtype=np.float64)
    ALIGNED_Q_DEG = np.asarray(
        [
            -79.28978634,
            -21.21559892,
            -35.81736338,
            -22.21267786,
            62.74105957,
            177.21542446,
        ],
        dtype=np.float64,
    )
    SEARCH_Q_DEG = np.asarray(
        [
            -80.89260546,
            -41.51886461,
            6.30065561,
            -41.48512961,
            23.50213558,
            156.69935893,
        ],
        dtype=np.float64,
    )

    def __init__(
        self,
        xml_path: str | Path = MECA500_NIH_PRESENTATION_SCENE,
        *,
        image_size_px: tuple[int, int] = (640, 480),
        settle_steps: int = 100,
    ):
        try:
            import mujoco
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("MuJoCo is required for the Meca500 plant.") from exc

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(Path(xml_path)))
        self.data = mujoco.MjData(self.model)
        width, height = image_size_px
        self.renderer = mujoco.Renderer(
            self.model,
            width=int(width),
            height=int(height),
        )
        self.width = int(width)
        self.height = int(height)
        self.settle_steps = int(settle_steps)
        self.joint_ids = tuple(
            mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                f"joint_{index}",
            )
            for index in range(1, 7)
        )
        self.qpos_addresses = tuple(
            int(self.model.jnt_qposadr[joint_id])
            for joint_id in self.joint_ids
        )
        self.dof_addresses = tuple(
            int(self.model.jnt_dofadr[joint_id])
            for joint_id in self.joint_ids
        )
        self.actuator_ids = tuple(
            mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                f"joint_{index}_position",
            )
            for index in range(1, 7)
        )
        self.camera_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            "eye_in_hand",
        )
        self.camera_site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            "eye_in_hand_origin",
        )
        self.tool_tip_site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            "tool_tip",
        )
        self.trocar_site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            "trocar_center",
        )
        self.trocar_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "single_trocar_visual",
        )
        self.trocar_material_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_MATERIAL,
            "trocar_mat",
        )
        self.sclera_material_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_MATERIAL,
            "sclera_mat",
        )
        self.instrument_geom_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "instrument_shaft",
        )
        self.wall_geom_ids = tuple(
            mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                name,
            )
            for name in (
                "trocar_wall_pos_y",
                "trocar_wall_neg_y",
                "trocar_wall_pos_z",
                "trocar_wall_neg_z",
            )
        )
        ids = (
            *self.joint_ids,
            *self.actuator_ids,
            self.camera_id,
            self.camera_site_id,
            self.tool_tip_site_id,
            self.trocar_site_id,
            self.trocar_body_id,
            self.trocar_material_id,
            self.sclera_material_id,
            self.instrument_geom_id,
            *self.wall_geom_ids,
        )
        if any(value < 0 for value in ids):
            raise ValueError("The unified Meca500 scene is missing required objects.")
        self.eye_scene_option = mujoco.MjvOption()
        self.eye_scene_option.geomgroup[1] = 0
        self.eye_scene_option.geomgroup[2] = 0
        self.eye_scene_option.geomgroup[3] = 0
        self.eye_scene_option.sitegroup[1] = 0
        self._base_trocar_body_pos = self.model.body_pos[
            self.trocar_body_id
        ].copy()
        self._base_trocar_body_quat = self.model.body_quat[
            self.trocar_body_id
        ].copy()
        self._base_camera_fovy = float(self.model.cam_fovy[self.camera_id])
        self._base_trocar_rgba = self.model.mat_rgba[
            self.trocar_material_id
        ].copy()
        self._base_sclera_rgba = self.model.mat_rgba[
            self.sclera_material_id
        ].copy()
        self._base_light_diffuse = self.model.light_diffuse.copy()
        self._base_light_specular = self.model.light_specular.copy()
        self._last_contact_metrics = {
            "wall_contact_detected": False,
            "wall_contact_count": 0,
            "maximum_normal_force_n": 0.0,
        }
        self.reset()

    def reset_domain(self) -> None:
        self.model.body_pos[self.trocar_body_id] = self._base_trocar_body_pos
        self.model.body_quat[self.trocar_body_id] = (
            self._base_trocar_body_quat
        )
        self.model.cam_fovy[self.camera_id] = self._base_camera_fovy
        self.model.mat_rgba[self.trocar_material_id] = (
            self._base_trocar_rgba
        )
        self.model.mat_rgba[self.sclera_material_id] = self._base_sclera_rgba
        self.model.light_diffuse[:] = self._base_light_diffuse
        self.model.light_specular[:] = self._base_light_specular
        self.mujoco.mj_forward(self.model, self.data)

    def set_domain_randomization(
        self,
        *,
        trocar_translation_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
        trocar_rotation_deg_xyz: tuple[float, float, float] = (
            0.0,
            0.0,
            0.0,
        ),
        camera_fovy_scale: float = 1.0,
        trocar_rgb_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
        sclera_rgb_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
        light_intensity_scale: float = 1.0,
    ) -> None:
        """Apply reproducible model-domain changes for evaluation only."""

        self.reset_domain()
        self.model.body_pos[self.trocar_body_id] = (
            self._base_trocar_body_pos
            + np.asarray(trocar_translation_mm, dtype=np.float64) * 1e-3
        )
        delta = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        for axis, angle_deg in zip(
            np.eye(3, dtype=np.float64),
            np.asarray(
                trocar_rotation_deg_xyz,
                dtype=np.float64,
            ),
        ):
            if abs(float(angle_deg)) <= 1e-12:
                continue
            axis_quat = np.zeros(4, dtype=np.float64)
            self.mujoco.mju_axisAngle2Quat(
                axis_quat,
                axis,
                np.deg2rad(float(angle_deg)),
            )
            composed = np.zeros(4, dtype=np.float64)
            self.mujoco.mju_mulQuat(composed, axis_quat, delta)
            delta = composed
        randomized_quat = np.zeros(4, dtype=np.float64)
        self.mujoco.mju_mulQuat(
            randomized_quat,
            delta,
            self._base_trocar_body_quat,
        )
        self.model.body_quat[self.trocar_body_id] = randomized_quat
        self.model.cam_fovy[self.camera_id] = (
            self._base_camera_fovy * float(camera_fovy_scale)
        )
        for material_id, base, scale in (
            (
                self.trocar_material_id,
                self._base_trocar_rgba,
                trocar_rgb_scale,
            ),
            (
                self.sclera_material_id,
                self._base_sclera_rgba,
                sclera_rgb_scale,
            ),
        ):
            rgba = base.copy()
            rgba[:3] = np.clip(
                rgba[:3] * np.asarray(scale, dtype=np.float64),
                0.0,
                1.0,
            )
            self.model.mat_rgba[material_id] = rgba
        light_scale = float(light_intensity_scale)
        self.model.light_diffuse[:] = np.clip(
            self._base_light_diffuse * light_scale,
            0.0,
            1.0,
        )
        self.model.light_specular[:] = np.clip(
            self._base_light_specular * light_scale,
            0.0,
            1.0,
        )
        self.mujoco.mj_forward(self.model, self.data)

    def reset(self, q_deg: np.ndarray | None = None) -> None:
        self.mujoco.mj_resetData(self.model, self.data)
        q = np.deg2rad(
            self.SEARCH_Q_DEG if q_deg is None else np.asarray(q_deg)
        )
        self._set_joint_state(q)
        self._last_contact_metrics = {
            "wall_contact_detected": False,
            "wall_contact_count": 0,
            "maximum_normal_force_n": 0.0,
        }

    def joint_positions_rad(self) -> np.ndarray:
        return np.asarray(
            [self.data.qpos[address] for address in self.qpos_addresses],
            dtype=np.float64,
        )

    def joint_positions_deg(self) -> np.ndarray:
        return np.rad2deg(self.joint_positions_rad())

    def _set_joint_state(self, q: np.ndarray) -> None:
        for address, actuator_id, value in zip(
            self.qpos_addresses,
            self.actuator_ids,
            q,
        ):
            self.data.qpos[address] = float(value)
            self.data.ctrl[actuator_id] = float(value)
        self.data.qvel[:] = 0.0
        self.mujoco.mj_forward(self.model, self.data)

    def capture_rgb(self) -> np.ndarray:
        self.renderer.update_scene(
            self.data,
            camera="eye_in_hand",
            scene_option=self.eye_scene_option,
        )
        return self.renderer.render().copy()

    def capture_overview_rgb(self) -> np.ndarray:
        self.renderer.update_scene(
            self.data,
            camera="full_arm_presentation",
        )
        return self.renderer.render().copy()

    def capture_trocar_closeup_rgb(self) -> np.ndarray:
        self.renderer.update_scene(
            self.data,
            camera="trocar_closeup",
        )
        return self.renderer.render().copy()

    def camera_rotation_world(self) -> np.ndarray:
        return np.asarray(
            self.data.cam_xmat[self.camera_id],
            dtype=np.float64,
        ).reshape(3, 3)

    def tool_rotation_world(self) -> np.ndarray:
        return np.asarray(
            self.data.site_xmat[self.tool_tip_site_id],
            dtype=np.float64,
        ).reshape(3, 3)

    def tool_position_world(self) -> np.ndarray:
        return np.asarray(
            self.data.site_xpos[self.tool_tip_site_id],
            dtype=np.float64,
        ).copy()

    def tool_insertion_axis_world(self) -> np.ndarray:
        return self.tool_rotation_world()[:, 2].copy()

    def apply_camera_translation_mm(
        self,
        command_camera_xyz_mm: tuple[float, float, float],
        *,
        maximum_joint_step_deg: float = 2.0,
    ) -> np.ndarray:
        jacobian_position = np.zeros((3, self.model.nv), dtype=np.float64)
        jacobian_rotation = np.zeros((3, self.model.nv), dtype=np.float64)
        self.mujoco.mj_jacSite(
            self.model,
            self.data,
            jacobian_position,
            jacobian_rotation,
            self.camera_site_id,
        )
        camera_step = np.asarray(command_camera_xyz_mm, dtype=np.float64) * 1e-3
        world_step = self.camera_rotation_world() @ camera_step
        jacobian = jacobian_position[:, self.dof_addresses]
        delta_q = self._damped_least_squares(jacobian, world_step)
        return self.apply_joint_delta(
            delta_q,
            maximum_joint_step_deg=maximum_joint_step_deg,
        )

    def camera_twist_joint_delta(
        self,
        *,
        translation_camera_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotation_camera_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> np.ndarray:
        jacobian_position = np.zeros((3, self.model.nv), dtype=np.float64)
        jacobian_rotation = np.zeros((3, self.model.nv), dtype=np.float64)
        self.mujoco.mj_jacSite(
            self.model,
            self.data,
            jacobian_position,
            jacobian_rotation,
            self.camera_site_id,
        )
        camera_rotation = self.camera_rotation_world()
        desired_twist = np.concatenate(
            (
                camera_rotation
                @ (
                    np.asarray(translation_camera_mm, dtype=np.float64)
                    * 1e-3
                ),
                camera_rotation
                @ np.deg2rad(
                    np.asarray(rotation_camera_deg, dtype=np.float64)
                ),
            )
        )
        jacobian = np.vstack(
            (
                jacobian_position[:, self.dof_addresses],
                jacobian_rotation[:, self.dof_addresses],
            )
        )
        return self._damped_least_squares(jacobian, desired_twist)

    def apply_camera_twist(
        self,
        *,
        translation_camera_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotation_camera_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
        maximum_joint_step_deg: float = 1.5,
    ) -> np.ndarray:
        return self.apply_joint_delta(
            self.camera_twist_joint_delta(
                translation_camera_mm=translation_camera_mm,
                rotation_camera_deg=rotation_camera_deg,
            ),
            maximum_joint_step_deg=maximum_joint_step_deg,
        )

    def camera_pose_joint_delta(
        self,
        *,
        translation_camera_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotation_camera_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
        iterations: int = 8,
    ) -> np.ndarray:
        """Iterative IK for a camera-frame pose delta with position hold."""

        baseline_q = self.joint_positions_rad()
        baseline_position = np.asarray(
            self.data.site_xpos[self.camera_site_id],
            dtype=np.float64,
        ).copy()
        baseline_rotation = self.camera_rotation_world().copy()
        desired_position = baseline_position + baseline_rotation @ (
            np.asarray(translation_camera_mm, dtype=np.float64) * 1e-3
        )
        rotation_vector = np.deg2rad(
            np.asarray(rotation_camera_deg, dtype=np.float64)
        )
        angle = float(np.linalg.norm(rotation_vector))
        if angle <= 1e-12:
            local_rotation = np.eye(3)
        else:
            axis = rotation_vector / angle
            skew = np.asarray(
                [
                    [0.0, -axis[2], axis[1]],
                    [axis[2], 0.0, -axis[0]],
                    [-axis[1], axis[0], 0.0],
                ],
                dtype=np.float64,
            )
            local_rotation = (
                np.eye(3)
                + np.sin(angle) * skew
                + (1.0 - np.cos(angle)) * (skew @ skew)
            )
        desired_rotation = baseline_rotation @ local_rotation
        q = baseline_q.copy()
        try:
            for _ in range(max(1, int(iterations))):
                current_position = np.asarray(
                    self.data.site_xpos[self.camera_site_id],
                    dtype=np.float64,
                )
                current_rotation = self.camera_rotation_world()
                position_error = desired_position - current_position
                rotation_error_matrix = desired_rotation @ current_rotation.T
                rotation_error = 0.5 * np.asarray(
                    [
                        rotation_error_matrix[2, 1]
                        - rotation_error_matrix[1, 2],
                        rotation_error_matrix[0, 2]
                        - rotation_error_matrix[2, 0],
                        rotation_error_matrix[1, 0]
                        - rotation_error_matrix[0, 1],
                    ],
                    dtype=np.float64,
                )
                if (
                    np.linalg.norm(position_error) < 1e-7
                    and np.linalg.norm(rotation_error) < 1e-6
                ):
                    break
                jacobian_position = np.zeros(
                    (3, self.model.nv),
                    dtype=np.float64,
                )
                jacobian_rotation = np.zeros(
                    (3, self.model.nv),
                    dtype=np.float64,
                )
                self.mujoco.mj_jacSite(
                    self.model,
                    self.data,
                    jacobian_position,
                    jacobian_rotation,
                    self.camera_site_id,
                )
                jacobian = np.vstack(
                    (
                        jacobian_position[:, self.dof_addresses],
                        jacobian_rotation[:, self.dof_addresses],
                    )
                )
                q += self._damped_least_squares(
                    jacobian,
                    np.concatenate((position_error, rotation_error)),
                    damping=1e-7,
                )
                for index, joint_id in enumerate(self.joint_ids):
                    q[index] = np.clip(
                        q[index],
                        self.model.jnt_range[joint_id, 0],
                        self.model.jnt_range[joint_id, 1],
                    )
                self.probe_joint_configuration(q)
        finally:
            self.probe_joint_configuration(baseline_q)
        return q - baseline_q

    def apply_joint_delta(
        self,
        delta_q_rad: np.ndarray,
        *,
        maximum_joint_step_deg: float = 1.5,
        track_contacts: bool = False,
    ) -> np.ndarray:
        bound = np.deg2rad(float(maximum_joint_step_deg))
        delta = np.clip(
            np.asarray(delta_q_rad, dtype=np.float64),
            -bound,
            bound,
        )
        current = self.joint_positions_rad()
        target = current + delta
        for index, joint_id in enumerate(self.joint_ids):
            target[index] = np.clip(
                target[index],
                self.model.jnt_range[joint_id, 0],
                self.model.jnt_range[joint_id, 1],
            )
        for actuator_id, value in zip(self.actuator_ids, target):
            self.data.ctrl[actuator_id] = float(value)
        self._settle(track_contacts=track_contacts)
        return self.joint_positions_rad() - current

    def probe_joint_configuration(self, q_rad: np.ndarray) -> None:
        for address, value in zip(self.qpos_addresses, q_rad):
            self.data.qpos[address] = float(value)
        self.data.qvel[:] = 0.0
        self.mujoco.mj_forward(self.model, self.data)

    def move_tool_along_axis_mm(
        self,
        step_mm: float,
        *,
        maximum_joint_step_deg: float = 1.0,
    ) -> np.ndarray:
        jacobian_position = np.zeros((3, self.model.nv), dtype=np.float64)
        jacobian_rotation = np.zeros((3, self.model.nv), dtype=np.float64)
        self.mujoco.mj_jacSite(
            self.model,
            self.data,
            jacobian_position,
            jacobian_rotation,
            self.tool_tip_site_id,
        )
        jacobian = np.vstack(
            (
                jacobian_position[:, self.dof_addresses],
                jacobian_rotation[:, self.dof_addresses],
            )
        )
        insertion_axis = self.tool_rotation_world()[:, 2]
        desired_twist = np.concatenate(
            (insertion_axis * float(step_mm) * 1e-3, np.zeros(3))
        )
        delta_q = self._damped_least_squares(jacobian, desired_twist)
        return self.apply_joint_delta(
            delta_q,
            maximum_joint_step_deg=maximum_joint_step_deg,
            track_contacts=True,
        )

    @staticmethod
    def _damped_least_squares(
        jacobian: np.ndarray,
        target: np.ndarray,
        *,
        damping: float = 1e-5,
    ) -> np.ndarray:
        j = np.asarray(jacobian, dtype=np.float64)
        rhs = np.asarray(target, dtype=np.float64)
        return j.T @ np.linalg.solve(
            j @ j.T + damping * np.eye(j.shape[0]),
            rhs,
        )

    def _settle(self, *, track_contacts: bool = False) -> None:
        maximum_force = 0.0
        contact_count = 0
        for _ in range(max(1, self.settle_steps)):
            self.mujoco.mj_step(self.model, self.data)
            if track_contacts:
                metrics = self._current_contact_metrics()
                maximum_force = max(
                    maximum_force,
                    float(metrics["maximum_normal_force_n"]),
                )
                contact_count += int(metrics["wall_contact_count"])
        self.mujoco.mj_forward(self.model, self.data)
        if track_contacts:
            self._last_contact_metrics = {
                "wall_contact_detected": contact_count > 0,
                "wall_contact_count": contact_count,
                "maximum_normal_force_n": maximum_force,
            }

    def _current_contact_metrics(self) -> dict[str, float | int | bool]:
        maximum_force = 0.0
        count = 0
        for index in range(int(self.data.ncon)):
            contact = self.data.contact[index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if (
                self.instrument_geom_id not in pair
                or not any(wall in pair for wall in self.wall_geom_ids)
            ):
                continue
            force = np.zeros(6, dtype=np.float64)
            self.mujoco.mj_contactForce(
                self.model,
                self.data,
                index,
                force,
            )
            maximum_force = max(maximum_force, float(abs(force[0])))
            count += 1
        return {
            "wall_contact_detected": count > 0,
            "wall_contact_count": count,
            "maximum_normal_force_n": maximum_force,
        }

    def wall_contact_metrics(self) -> dict[str, float | int | bool]:
        current = self._current_contact_metrics()
        previous = self._last_contact_metrics
        return {
            "wall_contact_detected": bool(
                current["wall_contact_detected"]
                or previous["wall_contact_detected"]
            ),
            "wall_contact_count": (
                int(current["wall_contact_count"])
                + int(previous["wall_contact_count"])
            ),
            "maximum_normal_force_n": max(
                float(current["maximum_normal_force_n"]),
                float(previous["maximum_normal_force_n"]),
            ),
        }

    def evaluation_pose_errors(self) -> dict[str, float]:
        """Privileged evaluation only; never use these values for control."""

        camera_position = np.asarray(
            self.data.site_xpos[self.camera_site_id],
            dtype=np.float64,
        )
        target_position = np.asarray(
            self.data.site_xpos[self.trocar_site_id],
            dtype=np.float64,
        )
        camera_axis = -self.camera_rotation_world()[:, 2]
        target_axis = np.asarray(
            self.data.site_xmat[self.trocar_site_id],
            dtype=np.float64,
        ).reshape(3, 3)[:, 2]
        delta = target_position - camera_position
        lateral = delta - float(np.dot(delta, camera_axis)) * camera_axis
        cosine = float(np.clip(np.dot(camera_axis, target_axis), -1.0, 1.0))
        return {
            "lateral_error_mm": float(np.linalg.norm(lateral) * 1000.0),
            "axis_error_deg": float(np.degrees(np.arccos(cosine))),
            "axial_distance_mm": float(np.dot(delta, camera_axis) * 1000.0),
        }

    def close(self) -> None:
        self.renderer.close()
