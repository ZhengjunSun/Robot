from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .staged_alignment import FineObservation


@dataclass(frozen=True)
class FineRingDetectorConfig:
    focal_length_px: float
    trocar_outer_radius_mm: float
    target_standoff_mm: float
    hue_ranges: tuple[tuple[int, int], ...] = ((82, 112),)
    saturation_minimum: int = 55
    value_minimum: int = 35
    inner_value_maximum: int = 80
    minimum_outer_area_px2: float = 180.0
    minimum_inner_area_px2: float = 20.0
    maximum_outer_diameter_px: float = 120.0
    minimum_outer_aspect_ratio: float = 0.60
    maximum_outer_center_error_px: float = 40.0
    expected_inner_outer_ratio: float = 0.379
    maximum_ratio_error: float = 0.16
    maximum_fit_error_px: float = 0.8

    def __post_init__(self) -> None:
        if self.focal_length_px <= 0.0:
            raise ValueError("focal_length_px must be positive.")
        if self.trocar_outer_radius_mm <= 0.0:
            raise ValueError("trocar_outer_radius_mm must be positive.")
        if self.target_standoff_mm <= 0.0:
            raise ValueError("target_standoff_mm must be positive.")


@dataclass(frozen=True)
class FineRingEstimate:
    observation: FineObservation
    outer_major_diameter_px: float
    outer_minor_diameter_px: float
    inner_major_diameter_px: float
    inner_minor_diameter_px: float
    outer_angle_deg: float
    inner_angle_deg: float
    estimated_depth_mm: float
    ellipse_fit_error_px: float


def _fit_ellipse(contour: np.ndarray) -> tuple[tuple[float, float], float, float, float]:
    if len(contour) < 5:
        raise ValueError("At least five contour samples are required.")
    center, axes, angle_deg = cv2.fitEllipse(contour)
    first, second = (float(axes[0]), float(axes[1]))
    if first >= second:
        major, minor, major_angle = first, second, float(angle_deg)
    else:
        major, minor, major_angle = second, first, float(angle_deg) + 90.0
    return (
        (float(center[0]), float(center[1])),
        major,
        minor,
        major_angle,
    )


def _ellipse_fit_error_px(
    contour: np.ndarray,
    center: tuple[float, float],
    major: float,
    minor: float,
    angle_deg: float,
) -> float:
    points = contour.reshape(-1, 2).astype(np.float64)
    shifted = points - np.asarray(center, dtype=np.float64)
    angle = math.radians(angle_deg)
    rotation = np.asarray(
        [[math.cos(angle), math.sin(angle)], [-math.sin(angle), math.cos(angle)]],
        dtype=np.float64,
    )
    local = shifted @ rotation.T
    normalized_radius = np.sqrt(
        np.square(local[:, 0] / max(0.5 * major, 1e-6))
        + np.square(local[:, 1] / max(0.5 * minor, 1e-6))
    )
    return float(
        np.sqrt(np.mean(np.square(normalized_radius - 1.0)))
        * 0.25
        * (major + minor)
    )


class NihFineRingDetector:
    """Estimate inner/outer ring geometry from NIH-scene RGB pixels only."""

    def __init__(self, config: FineRingDetectorConfig):
        self.config = config
        self.last_estimate: FineRingEstimate | None = None

    def estimate(self, image_rgb: np.ndarray) -> FineObservation | None:
        estimate = self.detect(image_rgb)
        self.last_estimate = estimate
        return None if estimate is None else estimate.observation

    def detect(self, image_rgb: np.ndarray) -> FineRingEstimate | None:
        image = np.asarray(image_rgb)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("NihFineRingDetector expects an HxWx3 RGB image.")
        cfg = self.config
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        outer_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for lower_hue, upper_hue in cfg.hue_ranges:
            outer_mask = cv2.bitwise_or(
                outer_mask,
                cv2.inRange(
                    hsv,
                    (lower_hue, cfg.saturation_minimum, cfg.value_minimum),
                    (upper_hue, 255, 255),
                ),
            )
        kernel = np.ones((3, 3), dtype=np.uint8)
        outer_mask = cv2.morphologyEx(outer_mask, cv2.MORPH_OPEN, kernel)
        outer_mask = cv2.morphologyEx(outer_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(
            outer_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        outer_candidates: list[tuple[float, np.ndarray]] = []
        image_center = np.asarray(
            [0.5 * image.shape[1], 0.5 * image.shape[0]],
            dtype=np.float64,
        )
        for contour in contours:
            if (
                cv2.contourArea(contour) < cfg.minimum_outer_area_px2
                or len(contour) < 5
            ):
                continue
            center, major, minor, _ = _fit_ellipse(contour)
            center_error = float(
                np.linalg.norm(np.asarray(center) - image_center)
            )
            if (
                major > cfg.maximum_outer_diameter_px
                or minor / max(major, 1e-6)
                < cfg.minimum_outer_aspect_ratio
                or center_error > cfg.maximum_outer_center_error_px
            ):
                continue
            outer_candidates.append(
                (float(cv2.contourArea(contour)), contour)
            )
        if not outer_candidates:
            self.last_estimate = None
            return None
        outer_contour = max(outer_candidates, key=lambda item: item[0])[1]
        outer_center, outer_major, outer_minor, outer_angle = _fit_ellipse(
            outer_contour
        )

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        inner_mask = cv2.inRange(gray, 0, cfg.inner_value_maximum)
        roi = np.zeros_like(inner_mask)
        cv2.ellipse(
            roi,
            tuple(int(round(value)) for value in outer_center),
            (
                max(2, int(round(0.42 * outer_major))),
                max(2, int(round(0.42 * outer_minor))),
            ),
            outer_angle,
            0,
            360,
            255,
            -1,
        )
        inner_mask = cv2.bitwise_and(inner_mask, roi)
        inner_mask = cv2.morphologyEx(inner_mask, cv2.MORPH_OPEN, kernel)
        inner_contours, _ = cv2.findContours(
            inner_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        inner_candidates: list[tuple[float, np.ndarray, tuple, float, float, float]] = []
        for contour in inner_contours:
            if cv2.contourArea(contour) < cfg.minimum_inner_area_px2 or len(contour) < 5:
                continue
            center, major, minor, angle = _fit_ellipse(contour)
            ratio = major / max(outer_major, 1e-6)
            center_distance = float(
                np.linalg.norm(
                    np.asarray(center) - np.asarray(outer_center)
                )
            )
            score = (
                abs(ratio - cfg.expected_inner_outer_ratio)
                + 0.02 * center_distance
            )
            inner_candidates.append(
                (score, contour, center, major, minor, angle)
            )
        if not inner_candidates:
            self.last_estimate = None
            return None
        _, inner_contour, inner_center, inner_major, inner_minor, inner_angle = min(
            inner_candidates, key=lambda item: item[0]
        )

        outer_radius_px = 0.25 * (outer_major + outer_minor)
        depth_mm = (
            cfg.focal_length_px
            * cfg.trocar_outer_radius_mm
            / max(outer_radius_px, 1e-6)
        )
        image_center = (0.5 * image.shape[1], 0.5 * image.shape[0])
        pixel_delta = np.asarray(outer_center) - np.asarray(image_center)
        lateral_error_mm = float(
            np.linalg.norm(pixel_delta) * depth_mm / cfg.focal_length_px
        )
        # A one-pixel axis difference is below the angular resolution of this
        # small synthetic ring. Treat it as unresolved instead of reporting a
        # spurious several-degree tilt.
        aspect = float(
            np.clip(
                (outer_minor + 1.0) / max(outer_major, 1e-6),
                0.0,
                1.0,
            )
        )
        axis_error_deg = math.degrees(math.acos(aspect))
        outer_fit_error = _ellipse_fit_error_px(
            outer_contour,
            outer_center,
            outer_major,
            outer_minor,
            outer_angle,
        )
        inner_fit_error = _ellipse_fit_error_px(
            inner_contour,
            inner_center,
            inner_major,
            inner_minor,
            outer_angle,
        )
        fit_error = max(outer_fit_error, inner_fit_error)
        ratio_error = abs(
            inner_major / max(outer_major, 1e-6)
            - cfg.expected_inner_outer_ratio
        )
        quality_pass = bool(
            ratio_error <= cfg.maximum_ratio_error
            and fit_error <= cfg.maximum_fit_error_px
        )
        observation = FineObservation(
            image_center_px=image_center,
            outer_center_px=outer_center,
            inner_center_px=inner_center,
            lateral_error_mm=lateral_error_mm,
            axis_error_deg=axis_error_deg,
            standoff_error_mm=depth_mm - cfg.target_standoff_mm,
            reprojection_error_px=fit_error,
            quality_gate_pass=quality_pass,
        )
        estimate = FineRingEstimate(
            observation=observation,
            outer_major_diameter_px=outer_major,
            outer_minor_diameter_px=outer_minor,
            inner_major_diameter_px=inner_major,
            inner_minor_diameter_px=inner_minor,
            outer_angle_deg=outer_angle,
            inner_angle_deg=inner_angle,
            estimated_depth_mm=depth_mm,
            ellipse_fit_error_px=fit_error,
        )
        self.last_estimate = estimate
        return estimate


@dataclass(frozen=True)
class FineServoConfig:
    focal_length_px: float
    target_standoff_mm: float
    lateral_gain: float = 0.55
    standoff_gain: float = 0.50
    maximum_lateral_step_mm: float = 0.50
    maximum_standoff_step_mm: float = 0.75


@dataclass(frozen=True)
class FineServoCommand:
    camera_xyz_mm: tuple[float, float, float]
    center_error_px: tuple[float, float]
    standoff_error_mm: float


class FineImageBasedVisualServo:
    """Bounded 2.5D translation servo for centering and standoff."""

    def __init__(self, config: FineServoConfig):
        self.config = config

    def command(self, observation: FineObservation) -> FineServoCommand:
        image_center = np.asarray(observation.image_center_px, dtype=np.float64)
        outer_center = np.asarray(observation.outer_center_px, dtype=np.float64)
        pixel_error = outer_center - image_center
        depth_mm = self.config.target_standoff_mm + observation.standoff_error_mm
        lateral = (
            self.config.lateral_gain
            * depth_mm
            / self.config.focal_length_px
            * np.asarray([pixel_error[0], -pixel_error[1]], dtype=np.float64)
        )
        norm = float(np.linalg.norm(lateral))
        if norm > self.config.maximum_lateral_step_mm:
            lateral *= self.config.maximum_lateral_step_mm / norm
        # Camera +z points away from the target in the baseline scene.
        z_step = float(
            np.clip(
                -self.config.standoff_gain * observation.standoff_error_mm,
                -self.config.maximum_standoff_step_mm,
                self.config.maximum_standoff_step_mm,
            )
        )
        return FineServoCommand(
            camera_xyz_mm=(
                float(lateral[0]),
                float(lateral[1]),
                z_step,
            ),
            center_error_px=(float(pixel_error[0]), float(pixel_error[1])),
            standoff_error_mm=float(observation.standoff_error_mm),
        )
