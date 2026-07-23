from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fine_vision import FineRingEstimate, NihFineRingDetector
from .meca500_visual_env import Meca500VisualAlignmentPlant


@dataclass(frozen=True)
class SixAxisFineServoConfig:
    target_standoff_mm: float = 22.0
    center_scale_px: float = 20.0
    depth_scale_mm: float = 5.0
    anisotropy_scale: float = 0.10
    probe_step_deg: float = 0.25
    damping: float = 0.02
    maximum_joint_step_deg: float = 1.0
    maximum_joint_norm_deg: float = 2.5


@dataclass(frozen=True)
class SixAxisFineServoCommand:
    delta_q_rad: tuple[float, ...]
    feature_error: tuple[float, ...]
    valid_probe_columns: int


class NumericalImageJacobianFineServo:
    """Visual-only 5D ring-feature servo over all six Meca500 joints.

    The five observable features constrain image center (2), standoff (1), and
    ellipse anisotropy (2). Rotation about the trocar axis is intentionally a
    redundant degree of freedom because a circular ring cannot observe roll.
    """

    def __init__(self, config: SixAxisFineServoConfig | None = None):
        self.config = config or SixAxisFineServoConfig()

    def feature(
        self,
        estimate: FineRingEstimate,
        *,
        image_size_px: tuple[int, int],
    ) -> np.ndarray:
        width, height = image_size_px
        outer_x, outer_y = estimate.observation.outer_center_px
        ratio = float(
            estimate.outer_minor_diameter_px
            / max(estimate.outer_major_diameter_px, 1e-9)
        )
        anisotropy = max(0.0, 1.0 - ratio)
        angle = np.deg2rad(float(estimate.outer_angle_deg))
        return np.asarray(
            [
                (outer_x - 0.5 * width) / self.config.center_scale_px,
                (outer_y - 0.5 * height) / self.config.center_scale_px,
                (
                    estimate.estimated_depth_mm
                    - self.config.target_standoff_mm
                )
                / self.config.depth_scale_mm,
                anisotropy
                * np.cos(2.0 * angle)
                / self.config.anisotropy_scale,
                anisotropy
                * np.sin(2.0 * angle)
                / self.config.anisotropy_scale,
            ],
            dtype=np.float64,
        )

    def command(
        self,
        *,
        plant: Meca500VisualAlignmentPlant,
        detector: NihFineRingDetector,
        baseline_estimate: FineRingEstimate,
    ) -> SixAxisFineServoCommand | None:
        baseline_q = plant.joint_positions_rad()
        baseline_feature = self.feature(
            baseline_estimate,
            image_size_px=(plant.width, plant.height),
        )
        probe = np.deg2rad(self.config.probe_step_deg)
        jacobian = np.zeros((baseline_feature.size, 6), dtype=np.float64)
        valid_columns = 0
        try:
            for column in range(6):
                probe_q = baseline_q.copy()
                probe_q[column] += probe
                plant.probe_joint_configuration(probe_q)
                estimate = detector.detect(plant.capture_rgb())
                signed_probe = probe
                if estimate is None:
                    probe_q[column] = baseline_q[column] - probe
                    plant.probe_joint_configuration(probe_q)
                    estimate = detector.detect(plant.capture_rgb())
                    signed_probe = -probe
                if estimate is None:
                    continue
                jacobian[:, column] = (
                    self.feature(
                        estimate,
                        image_size_px=(plant.width, plant.height),
                    )
                    - baseline_feature
                ) / signed_probe
                valid_columns += 1
        finally:
            plant.probe_joint_configuration(baseline_q)
            detector.last_estimate = baseline_estimate

        if valid_columns < 4 or np.linalg.matrix_rank(jacobian) < 4:
            return None
        damping = float(self.config.damping)
        delta_q = -jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T
            + damping * np.eye(jacobian.shape[0]),
            baseline_feature,
        )
        per_joint_bound = np.deg2rad(self.config.maximum_joint_step_deg)
        delta_q = np.clip(delta_q, -per_joint_bound, per_joint_bound)
        norm_bound = np.deg2rad(self.config.maximum_joint_norm_deg)
        norm = float(np.linalg.norm(delta_q))
        if norm > norm_bound:
            delta_q *= norm_bound / norm
        return SixAxisFineServoCommand(
            delta_q_rad=tuple(float(value) for value in delta_q),
            feature_error=tuple(float(value) for value in baseline_feature),
            valid_probe_columns=valid_columns,
        )


@dataclass(frozen=True)
class ActiveOrientationServoConfig:
    probe_rotation_deg: float = 1.0
    maximum_rotation_step_deg: float = 1.5
    gradient_floor: float = 1e-5
    probe_recenter_iterations: int = 4
    probe_recenter_gain: float = 0.65
    # The finite raster and antialiasing leave a small nonzero ellipse
    # anisotropy even at the geometrically coaxial pose.
    target_anisotropy: float = 0.0048
    aligned_anisotropy_threshold: float = 0.0058


@dataclass(frozen=True)
class ActiveOrientationCommand:
    camera_rotation_xy_deg: tuple[float, float]
    anisotropy: float
    gradient_per_rad: tuple[float, float]
    valid_axes: int


class ActiveEllipseOrientationServo:
    """Minimize ring ellipse anisotropy using visual-only rotational probes."""

    def __init__(
        self,
        config: ActiveOrientationServoConfig | None = None,
    ):
        self.config = config or ActiveOrientationServoConfig()

    @staticmethod
    def anisotropy(estimate: FineRingEstimate) -> float:
        return float(
            max(
                0.0,
                1.0
                - estimate.outer_minor_diameter_px
                / max(estimate.outer_major_diameter_px, 1e-9),
            )
        )

    def command(
        self,
        *,
        plant: Meca500VisualAlignmentPlant,
        detector: NihFineRingDetector,
        baseline_estimate: FineRingEstimate,
    ) -> ActiveOrientationCommand | None:
        baseline_q = plant.joint_positions_rad()
        baseline_anisotropy = self.anisotropy(baseline_estimate)
        if (
            baseline_anisotropy
            <= self.config.aligned_anisotropy_threshold
        ):
            return ActiveOrientationCommand(
                camera_rotation_xy_deg=(0.0, 0.0),
                anisotropy=baseline_anisotropy,
                gradient_per_rad=(0.0, 0.0),
                valid_axes=2,
            )
        probe_deg = float(self.config.probe_rotation_deg)
        probe_rad = np.deg2rad(probe_deg)
        gradient = np.zeros(2, dtype=np.float64)
        valid_axes = 0
        try:
            for axis in range(2):
                samples: list[float | None] = []
                for sign in (1.0, -1.0):
                    rotation = [0.0, 0.0, 0.0]
                    rotation[axis] = sign * probe_deg
                    delta_q = plant.camera_pose_joint_delta(
                        rotation_camera_deg=tuple(rotation),
                    )
                    probe_q = baseline_q + delta_q
                    plant.probe_joint_configuration(probe_q)
                    estimate: FineRingEstimate | None = None
                    for _ in range(
                        max(0, self.config.probe_recenter_iterations)
                    ):
                        estimate = detector.detect(plant.capture_rgb())
                        if estimate is None:
                            break
                        center = np.asarray(
                            estimate.observation.outer_center_px,
                            dtype=np.float64,
                        )
                        image_center = np.asarray(
                            estimate.observation.image_center_px,
                            dtype=np.float64,
                        )
                        pixel_error = center - image_center
                        if np.linalg.norm(pixel_error) < 0.5:
                            break
                        depth = float(estimate.estimated_depth_mm)
                        focal = float(detector.config.focal_length_px)
                        lateral = (
                            self.config.probe_recenter_gain
                            * depth
                            / focal
                            * np.asarray(
                                [pixel_error[0], -pixel_error[1], 0.0],
                                dtype=np.float64,
                            )
                        )
                        recenter_delta = plant.camera_pose_joint_delta(
                            translation_camera_mm=tuple(lateral),
                        )
                        probe_q += recenter_delta
                        plant.probe_joint_configuration(probe_q)
                    estimate = detector.detect(plant.capture_rgb())
                    samples.append(
                        None if estimate is None else self.anisotropy(estimate)
                    )
                positive, negative = samples
                if positive is None or negative is None:
                    continue
                gradient[axis] = (positive - negative) / (2.0 * probe_rad)
                valid_axes += 1
        finally:
            plant.probe_joint_configuration(baseline_q)
            detector.last_estimate = baseline_estimate

        norm_squared = float(np.dot(gradient, gradient))
        if valid_axes < 2 or norm_squared < self.config.gradient_floor:
            return None
        anisotropy_error = max(
            0.0,
            baseline_anisotropy - self.config.target_anisotropy,
        )
        rotation_step_rad = (
            -anisotropy_error
            * gradient
            / (norm_squared + self.config.gradient_floor)
        )
        # Large corrections are useful in the far field. Close to a circular
        # projection, reduce the step with anisotropy so raster noise cannot
        # provoke another full 1.5-degree correction and displace the center.
        adaptive_maximum_deg = min(
            self.config.maximum_rotation_step_deg,
            max(0.12, 300.0 * anisotropy_error),
        )
        maximum = np.deg2rad(adaptive_maximum_deg)
        norm = float(np.linalg.norm(rotation_step_rad))
        if norm > maximum:
            rotation_step_rad *= maximum / norm
        return ActiveOrientationCommand(
            camera_rotation_xy_deg=tuple(
                float(value) for value in np.rad2deg(rotation_step_rad)
            ),
            anisotropy=baseline_anisotropy,
            gradient_per_rad=tuple(float(value) for value in gradient),
            valid_axes=valid_axes,
        )
