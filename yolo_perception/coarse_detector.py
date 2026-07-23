from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from real_3d_alignment.coarse_vision import RingDetection
from real_3d_alignment.staged_alignment import CoarseObservation


@dataclass(frozen=True)
class YoloCoarseDetectorConfig:
    target_class_ids: tuple[int, ...] = (0, 1)
    minimum_confidence: float = 0.25
    image_size: int = 640
    nms_iou: float = 0.50

    def __post_init__(self) -> None:
        if not self.target_class_ids:
            raise ValueError("target_class_ids must not be empty.")
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be in [0, 1].")


class YoloCoarseDetector:
    """Adapt an Ultralytics detector to the shared coarse-observation contract."""

    def __init__(
        self,
        model: Any,
        config: YoloCoarseDetectorConfig | None = None,
    ):
        self.model = model
        self.config = config or YoloCoarseDetectorConfig()

    @classmethod
    def from_weights(
        cls,
        weights: str | Path,
        config: YoloCoarseDetectorConfig | None = None,
    ) -> "YoloCoarseDetector":
        from ultralytics import YOLO

        return cls(YOLO(str(weights)), config)

    def detect(self, image_rgb: np.ndarray) -> RingDetection | None:
        image = np.asarray(image_rgb)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("YoloCoarseDetector expects an HxWx3 RGB image.")
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        results = self.model.predict(
            source=image_bgr,
            imgsz=self.config.image_size,
            conf=self.config.minimum_confidence,
            iou=self.config.nms_iou,
            verbose=False,
        )
        if not results:
            return None
        boxes = getattr(results[0], "boxes", None)
        if boxes is None:
            return None

        xyxy = self._to_numpy(getattr(boxes, "xyxy", None))
        classes = self._to_numpy(getattr(boxes, "cls", None))
        confidences = self._to_numpy(getattr(boxes, "conf", None))
        if xyxy is None or classes is None or confidences is None:
            return None

        candidates: list[tuple[float, float, np.ndarray, int]] = []
        target_classes = set(self.config.target_class_ids)
        for coords, class_id, confidence in zip(
            xyxy,
            classes.astype(int),
            confidences,
        ):
            if (
                int(class_id) not in target_classes
                or float(confidence) < self.config.minimum_confidence
            ):
                continue
            width = max(0.0, float(coords[2] - coords[0]))
            height = max(0.0, float(coords[3] - coords[1]))
            candidates.append(
                (
                    float(confidence),
                    width * height,
                    np.asarray(coords, dtype=np.float64),
                    int(class_id),
                )
            )
        if not candidates:
            return None

        confidence, area, coords, _ = max(
            candidates, key=lambda item: (item[0], item[1])
        )
        center = (
            float(0.5 * (coords[0] + coords[2])),
            float(0.5 * (coords[1] + coords[3])),
        )
        width = max(0.0, float(coords[2] - coords[0]))
        height = max(0.0, float(coords[3] - coords[1]))
        radius = 0.25 * (width + height)
        image_height, image_width = image.shape[:2]
        return RingDetection(
            observation=CoarseObservation(
                image_size_px=(int(image_width), int(image_height)),
                target_center_px=center,
                confidence=float(confidence),
            ),
            radius_px=float(radius),
            contour_area_px2=float(area),
            circularity=1.0,
        )

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray | None:
        if value is None:
            return None
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)
