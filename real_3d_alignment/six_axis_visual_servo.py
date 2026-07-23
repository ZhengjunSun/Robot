from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .fine_vision import (
    FineRingDetectorConfig,
    FineRingEstimate,
    NihFineRingDetector,
    _fit_ellipse,
)
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
    target_concentricity_ratio: float = 0.005
    aligned_concentricity_ratio_threshold: float = 0.010
    concentricity_cost_weight: float = 0.75
    calibrated_outer_feature: tuple[float, float] = (
        -0.00053,
        -0.00035,
    )
    aligned_outer_feature_error: float = 0.00050
    # Offline eye-in-hand appearance calibration at 1280x960. The projected
    # flange becomes sub-pixel circular with a small repeatable residual axis
    # bias; this fixed correction is calibration data, not runtime truth.
    outer_residual_calibration_rotation_xy_deg: tuple[float, float] = (
        -6.00,
        0.94,
    )


@dataclass(frozen=True)
class ActiveOrientationCommand:
    camera_rotation_xy_deg: tuple[float, float]
    anisotropy: float
    gradient_per_rad: tuple[float, float]
    valid_axes: int


@dataclass(frozen=True)
class OuterEllipseEstimate:
    image_center_px: tuple[float, float]
    center_px: tuple[float, float]
    major_diameter_px: float
    minor_diameter_px: float
    angle_deg: float
    estimated_depth_mm: float

    @property
    def center_error_px(self) -> float:
        return float(
            np.linalg.norm(
                np.asarray(self.center_px)
                - np.asarray(self.image_center_px)
            )
        )


class NihOuterEllipseDetector:
    """Outer-ring-only pose cue used before the lumen becomes visible."""

    def __init__(self, config: FineRingDetectorConfig):
        self.config = config
        self.last_estimate: OuterEllipseEstimate | None = None

    def detect(self, image_rgb: np.ndarray) -> OuterEllipseEstimate | None:
        image = np.asarray(image_rgb)
        cfg = self.config
        blur_size = max(1, int(cfg.gaussian_blur_kernel_px))
        if blur_size % 2 == 0:
            blur_size += 1
        filtered = (
            image
            if blur_size == 1
            else cv2.GaussianBlur(image, (blur_size, blur_size), 0)
        )
        hsv = cv2.cvtColor(filtered, cv2.COLOR_RGB2HSV)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for lower_hue, upper_hue in cfg.hue_ranges:
            mask = cv2.bitwise_or(
                mask,
                cv2.inRange(
                    hsv,
                    (lower_hue, cfg.saturation_minimum, cfg.value_minimum),
                    (upper_hue, 255, 255),
                ),
            )
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        image_center = np.asarray(
            (0.5 * image.shape[1], 0.5 * image.shape[0]),
            dtype=np.float64,
        )
        candidates: list[tuple[float, np.ndarray]] = []
        for contour in contours:
            if (
                cv2.contourArea(contour) < cfg.minimum_outer_area_px2
                or len(contour) < 5
            ):
                continue
            center, major, minor, _ = _fit_ellipse(contour)
            if (
                major > cfg.maximum_outer_diameter_px
                or minor / max(major, 1e-9)
                < cfg.minimum_outer_aspect_ratio
                or np.linalg.norm(np.asarray(center) - image_center)
                > cfg.maximum_outer_center_error_px
            ):
                continue
            candidates.append((float(cv2.contourArea(contour)), contour))
        if not candidates:
            self.last_estimate = None
            return None
        contour = max(candidates, key=lambda item: item[0])[1]
        center, major, minor, angle = _fit_ellipse(contour)
        radius = 0.25 * (major + minor)
        estimate = OuterEllipseEstimate(
            image_center_px=tuple(float(value) for value in image_center),
            center_px=center,
            major_diameter_px=major,
            minor_diameter_px=minor,
            angle_deg=angle,
            estimated_depth_mm=(
                cfg.focal_length_px
                * cfg.trocar_outer_radius_mm
                / max(radius, 1e-9)
            ),
        )
        self.last_estimate = estimate
        return estimate


class ActiveOuterEllipseOrientationServo:
    """Reduce large axis error using only the entrance flange ellipse."""

    def __init__(
        self,
        config: ActiveOrientationServoConfig | None = None,
    ):
        self.config = config or ActiveOrientationServoConfig()

    @staticmethod
    def anisotropy(estimate: OuterEllipseEstimate) -> float:
        return float(
            max(
                0.0,
                1.0
                - estimate.minor_diameter_px
                / max(estimate.major_diameter_px, 1e-9),
            )
        )

    @classmethod
    def ellipse_feature(cls, estimate: OuterEllipseEstimate) -> np.ndarray:
        anisotropy = cls.anisotropy(estimate)
        angle = np.deg2rad(float(estimate.angle_deg))
        return anisotropy * np.asarray(
            [np.cos(2.0 * angle), np.sin(2.0 * angle)],
            dtype=np.float64,
        )

    def feature_error(self, estimate: OuterEllipseEstimate) -> np.ndarray:
        return self.ellipse_feature(estimate) - np.asarray(
            self.config.calibrated_outer_feature,
            dtype=np.float64,
        )

    def is_aligned(self, estimate: OuterEllipseEstimate) -> bool:
        return bool(
            np.linalg.norm(self.feature_error(estimate))
            <= self.config.aligned_outer_feature_error
        )

    def command(
        self,
        *,
        plant: Meca500VisualAlignmentPlant,
        detector: NihOuterEllipseDetector,
        baseline_estimate: OuterEllipseEstimate,
    ) -> ActiveOrientationCommand | None:
        baseline_q = plant.joint_positions_rad()
        baseline_anisotropy = self.anisotropy(baseline_estimate)
        if self.is_aligned(baseline_estimate):
            return ActiveOrientationCommand(
                camera_rotation_xy_deg=(0.0, 0.0),
                anisotropy=baseline_anisotropy,
                gradient_per_rad=(0.0, 0.0),
                valid_axes=2,
            )
        probe_deg = float(self.config.probe_rotation_deg)
        probe_rad = np.deg2rad(probe_deg)
        gradient = np.zeros(2, dtype=np.float64)
        feature_jacobian = np.zeros((2, 2), dtype=np.float64)
        valid_axes = 0
        try:
            for axis in range(2):
                samples: list[
                    tuple[np.ndarray, float] | None
                ] = []
                for sign in (1.0, -1.0):
                    rotation = [0.0, 0.0, 0.0]
                    rotation[axis] = sign * probe_deg
                    probe_q = baseline_q + plant.camera_pose_joint_delta(
                        rotation_camera_deg=tuple(rotation),
                    )
                    plant.probe_joint_configuration(probe_q)
                    estimate = detector.detect(plant.capture_rgb())
                    for _ in range(
                        max(0, self.config.probe_recenter_iterations)
                    ):
                        if estimate is None:
                            break
                        pixel_error = (
                            np.asarray(estimate.center_px)
                            - np.asarray(estimate.image_center_px)
                        )
                        if np.linalg.norm(pixel_error) < 0.5:
                            break
                        lateral = (
                            self.config.probe_recenter_gain
                            * estimate.estimated_depth_mm
                            / detector.config.focal_length_px
                            * np.asarray(
                                [pixel_error[0], -pixel_error[1], 0.0]
                            )
                        )
                        probe_q += plant.camera_pose_joint_delta(
                            translation_camera_mm=tuple(lateral),
                        )
                        plant.probe_joint_configuration(probe_q)
                        estimate = detector.detect(plant.capture_rgb())
                    samples.append(
                        None
                        if estimate is None
                        else (
                            self.ellipse_feature(estimate),
                            self.anisotropy(estimate),
                        )
                    )
                positive, negative = samples
                if positive is None or negative is None:
                    continue
                positive_feature, positive_anisotropy = positive
                negative_feature, negative_anisotropy = negative
                feature_jacobian[:, axis] = (
                    positive_feature - negative_feature
                ) / (2.0 * probe_rad)
                gradient[axis] = (
                    positive_anisotropy - negative_anisotropy
                ) / (2.0 * probe_rad)
                valid_axes += 1
        finally:
            plant.probe_joint_configuration(baseline_q)
            detector.last_estimate = baseline_estimate
        if valid_axes < 2:
            return None
        if np.linalg.matrix_rank(feature_jacobian) == 2:
            damping = float(self.config.gradient_floor)
            baseline_feature = self.feature_error(baseline_estimate)
            rotation = -feature_jacobian.T @ np.linalg.solve(
                feature_jacobian @ feature_jacobian.T
                + damping * np.eye(2),
                baseline_feature,
            )
        else:
            norm_squared = float(np.dot(gradient, gradient))
            if norm_squared < self.config.gradient_floor:
                return None
            error = max(
                0.0,
                baseline_anisotropy - self.config.target_anisotropy,
            )
            rotation = (
                -error
                * gradient
                / (norm_squared + self.config.gradient_floor)
            )
        maximum = np.deg2rad(self.config.maximum_rotation_step_deg)
        norm = float(np.linalg.norm(rotation))
        if norm > maximum:
            rotation *= maximum / norm
        return ActiveOrientationCommand(
            camera_rotation_xy_deg=tuple(
                float(value) for value in np.rad2deg(rotation)
            ),
            anisotropy=baseline_anisotropy,
            gradient_per_rad=tuple(float(value) for value in gradient),
            valid_axes=valid_axes,
        )


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

    def orientation_cost(self, estimate: FineRingEstimate) -> float:
        anisotropy_error = max(
            0.0,
            self.anisotropy(estimate) - self.config.target_anisotropy,
        )
        concentricity_ratio = float(
            estimate.observation.outer_inner_concentricity_px
            / max(estimate.outer_major_diameter_px, 1e-9)
        )
        concentricity_error = max(
            0.0,
            concentricity_ratio
            - self.config.target_concentricity_ratio,
        )
        return float(
            anisotropy_error
            + self.config.concentricity_cost_weight * concentricity_error
        )

    @staticmethod
    def concentricity_feature(estimate: FineRingEstimate) -> np.ndarray:
        outer = np.asarray(
            estimate.observation.outer_center_px,
            dtype=np.float64,
        )
        inner = np.asarray(
            estimate.observation.inner_center_px,
            dtype=np.float64,
        )
        return (inner - outer) / max(
            estimate.outer_major_diameter_px,
            1e-9,
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
        baseline_concentricity_ratio = float(
            baseline_estimate.observation.outer_inner_concentricity_px
            / max(baseline_estimate.outer_major_diameter_px, 1e-9)
        )
        if (
            baseline_anisotropy
            <= self.config.aligned_anisotropy_threshold
            and baseline_concentricity_ratio
            <= self.config.aligned_concentricity_ratio_threshold
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
        concentricity_jacobian = np.zeros((2, 2), dtype=np.float64)
        valid_axes = 0
        try:
            for axis in range(2):
                samples: list[
                    tuple[np.ndarray, float] | None
                ] = []
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
                        None
                        if estimate is None
                        else (
                            self.concentricity_feature(estimate),
                            self.orientation_cost(estimate),
                        )
                    )
                positive, negative = samples
                if positive is None or negative is None:
                    continue
                positive_feature, positive_cost = positive
                negative_feature, negative_cost = negative
                concentricity_jacobian[:, axis] = (
                    positive_feature - negative_feature
                ) / (2.0 * probe_rad)
                gradient[axis] = (
                    positive_cost - negative_cost
                ) / (2.0 * probe_rad)
                valid_axes += 1
        finally:
            plant.probe_joint_configuration(baseline_q)
            detector.last_estimate = baseline_estimate

        if valid_axes < 2:
            return None
        orientation_error = self.orientation_cost(baseline_estimate)
        if (
            baseline_concentricity_ratio
            > self.config.target_concentricity_ratio
            and np.linalg.matrix_rank(concentricity_jacobian) == 2
        ):
            baseline_feature = self.concentricity_feature(
                baseline_estimate
            )
            damping = float(self.config.gradient_floor)
            rotation_step_rad = -concentricity_jacobian.T @ np.linalg.solve(
                concentricity_jacobian @ concentricity_jacobian.T
                + damping * np.eye(2),
                baseline_feature,
            )
        else:
            norm_squared = float(np.dot(gradient, gradient))
            if norm_squared < self.config.gradient_floor:
                return None
            rotation_step_rad = (
                -orientation_error
                * gradient
                / (norm_squared + self.config.gradient_floor)
            )
        # Large corrections are useful in the far field. Close to a circular
        # projection, reduce the step with anisotropy so raster noise cannot
        # provoke another full 1.5-degree correction and displace the center.
        adaptive_maximum_deg = min(
            self.config.maximum_rotation_step_deg,
            max(0.12, 120.0 * orientation_error),
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
