from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .staged_alignment import CoarseObservation


@dataclass(frozen=True)
class RingDetection:
    observation: CoarseObservation
    radius_px: float
    contour_area_px2: float
    circularity: float


@dataclass(frozen=True)
class TraditionalRingDetectorConfig:
    minimum_area_px2: float = 80.0
    minimum_radius_px: float = 4.0
    minimum_circularity: float = 0.18
    saturation_minimum: int = 70
    value_minimum: int = 45
    morphology_kernel_px: int = 3


class TraditionalRingDetector:
    """Detect the red trocar outer ring from an RGB eye-in-hand frame.

    The implementation intentionally uses only image pixels. It does not accept
    simulator poses or target coordinates.
    """

    def __init__(self, config: TraditionalRingDetectorConfig | None = None):
        self.config = config or TraditionalRingDetectorConfig()

    def detect(self, image_rgb: np.ndarray) -> RingDetection | None:
        image = np.asarray(image_rgb)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("TraditionalRingDetector expects an HxWx3 RGB image.")

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        cfg = self.config
        lower_red = cv2.inRange(
            hsv,
            (0, cfg.saturation_minimum, cfg.value_minimum),
            (18, 255, 255),
        )
        upper_red = cv2.inRange(
            hsv,
            (165, cfg.saturation_minimum, cfg.value_minimum),
            (179, 255, 255),
        )
        mask = cv2.bitwise_or(lower_red, upper_red)
        kernel_size = max(1, int(cfg.morphology_kernel_px))
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        candidates: list[
            tuple[float, np.ndarray, tuple[float, float], float, float]
        ] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            perimeter = float(cv2.arcLength(contour, True))
            if area < cfg.minimum_area_px2 or perimeter <= 1e-9:
                continue
            circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
            circle_center, radius = cv2.minEnclosingCircle(contour)
            if (
                radius < cfg.minimum_radius_px
                or circularity < cfg.minimum_circularity
            ):
                continue
            candidates.append(
                (
                    area,
                    contour,
                    (float(circle_center[0]), float(circle_center[1])),
                    float(radius),
                    circularity,
                )
            )

        if not candidates:
            return None

        area, _, center, radius, circularity = max(
            candidates, key=lambda item: item[0]
        )
        height, width = image.shape[:2]
        area_score = float(np.clip(area / (np.pi * radius * radius), 0.0, 1.0))
        # Coarse acquisition must tolerate a shaft/shadow occluding part of the
        # ring. Coverage therefore carries more weight than perfect circularity;
        # M3 applies the strict inner/outer geometric quality gate.
        confidence = float(
            np.clip(0.25 + 0.25 * circularity + 0.50 * area_score, 0.0, 1.0)
        )
        return RingDetection(
            observation=CoarseObservation(
                image_size_px=(int(width), int(height)),
                target_center_px=center,
                confidence=confidence,
            ),
            radius_px=radius,
            contour_area_px2=area,
            circularity=circularity,
        )


@dataclass(frozen=True)
class CoarseServoConfig:
    focal_length_px: float
    trocar_outer_radius_mm: float = 2.0
    gain: float = 0.65
    maximum_step_mm: float = 2.5
    minimum_depth_mm: float = 5.0
    maximum_depth_mm: float = 200.0

    def __post_init__(self) -> None:
        if self.focal_length_px <= 0.0:
            raise ValueError("focal_length_px must be positive.")
        if self.trocar_outer_radius_mm <= 0.0:
            raise ValueError("trocar_outer_radius_mm must be positive.")
        if self.maximum_step_mm <= 0.0:
            raise ValueError("maximum_step_mm must be positive.")


@dataclass(frozen=True)
class CoarseServoCommand:
    camera_xy_mm: tuple[float, float]
    estimated_depth_mm: float
    center_error_px: tuple[float, float]

    @property
    def norm_mm(self) -> float:
        return float(np.linalg.norm(np.asarray(self.camera_xy_mm)))


class CoarseImageBasedVisualServo:
    """Small-angle IBVS translation from ring-center and apparent-size error."""

    def __init__(self, config: CoarseServoConfig):
        self.config = config

    def command(self, detection: RingDetection) -> CoarseServoCommand:
        observation = detection.observation
        image_center = np.asarray(observation.image_center_px, dtype=np.float64)
        target_center = np.asarray(observation.target_center_px, dtype=np.float64)
        error = target_center - image_center
        depth_mm = (
            self.config.focal_length_px
            * self.config.trocar_outer_radius_mm
            / max(detection.radius_px, 1e-6)
        )
        depth_mm = float(
            np.clip(
                depth_mm,
                self.config.minimum_depth_mm,
                self.config.maximum_depth_mm,
            )
        )
        # OpenCV v grows downward, while the camera y axis grows upward.
        raw = self.config.gain * depth_mm / self.config.focal_length_px * np.asarray(
            [error[0], -error[1]], dtype=np.float64
        )
        norm = float(np.linalg.norm(raw))
        if norm > self.config.maximum_step_mm:
            raw *= self.config.maximum_step_mm / norm
        return CoarseServoCommand(
            camera_xy_mm=(float(raw[0]), float(raw[1])),
            estimated_depth_mm=depth_mm,
            center_error_px=(float(error[0]), float(error[1])),
        )
