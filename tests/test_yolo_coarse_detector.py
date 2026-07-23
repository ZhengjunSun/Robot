from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from yolo_perception.coarse_detector import (
    YoloCoarseDetector,
    YoloCoarseDetectorConfig,
)


class TensorLike:
    def __init__(self, values) -> None:
        self.values = np.asarray(values)

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FakeYolo:
    def __init__(self, boxes) -> None:
        self.boxes = boxes
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return [SimpleNamespace(boxes=self.boxes)]


def test_yolo_adapter_returns_shared_coarse_observation() -> None:
    model = FakeYolo(
        SimpleNamespace(
            xyxy=TensorLike([[100, 120, 180, 200], [300, 220, 340, 260]]),
            cls=TensorLike([2, 1]),
            conf=TensorLike([0.95, 0.80]),
        )
    )
    detector = YoloCoarseDetector(
        model,
        YoloCoarseDetectorConfig(target_class_ids=(1,)),
    )

    detection = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))

    assert detection is not None
    assert detection.observation.target_center_px == pytest.approx((320.0, 240.0))
    assert detection.observation.confidence == pytest.approx(0.80)
    assert detection.radius_px == pytest.approx(20.0)
    assert model.calls[0]["verbose"] is False


def test_yolo_adapter_rejects_non_target_or_low_confidence_boxes() -> None:
    model = FakeYolo(
        SimpleNamespace(
            xyxy=TensorLike([[100, 120, 180, 200]]),
            cls=TensorLike([3]),
            conf=TensorLike([0.90]),
        )
    )
    detector = YoloCoarseDetector(
        model,
        YoloCoarseDetectorConfig(target_class_ids=(1,), minimum_confidence=0.5),
    )

    assert detector.detect(np.zeros((240, 320, 3), dtype=np.uint8)) is None
