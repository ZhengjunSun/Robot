from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .staged_alignment import FineObservation


@dataclass(frozen=True)
class TemporalFineValidationConfig:
    """Safety-oriented temporal validation for near-field ring estimates."""

    window_size: int = 7
    minimum_valid_samples: int = 5
    maximum_center_range_px: float = 1.5
    maximum_lateral_range_mm: float = 0.08
    maximum_axis_range_deg: float = 1.5
    maximum_standoff_range_mm: float = 0.35

    def __post_init__(self) -> None:
        if self.window_size < 1:
            raise ValueError("window_size must be positive.")
        if not 1 <= self.minimum_valid_samples <= self.window_size:
            raise ValueError(
                "minimum_valid_samples must be within the temporal window."
            )
        limits = (
            self.maximum_center_range_px,
            self.maximum_lateral_range_mm,
            self.maximum_axis_range_deg,
            self.maximum_standoff_range_mm,
        )
        if any(value < 0.0 for value in limits):
            raise ValueError("Temporal spread limits must be non-negative.")


@dataclass(frozen=True)
class TemporalFineValidation:
    observation: FineObservation | None
    ready: bool
    sample_count: int
    reasons: tuple[str, ...]
    spreads: dict[str, float]


class TemporalFineObservationValidator:
    """Median-filter fresh observations and reject unstable windows.

    Missing or failed-quality observations clear the window. This class never
    turns stale data into insertion authorization.
    """

    def __init__(
        self,
        config: TemporalFineValidationConfig | None = None,
    ):
        self.config = config or TemporalFineValidationConfig()
        self._samples: deque[FineObservation] = deque(
            maxlen=self.config.window_size
        )

    def reset(self) -> None:
        self._samples.clear()

    def update(
        self,
        observation: FineObservation | None,
    ) -> TemporalFineValidation:
        if observation is None:
            self.reset()
            return TemporalFineValidation(
                observation=None,
                ready=False,
                sample_count=0,
                reasons=("fresh_fine_observation_missing",),
                spreads={},
            )
        if not observation.quality_gate_pass:
            self.reset()
            return TemporalFineValidation(
                observation=None,
                ready=False,
                sample_count=0,
                reasons=("fresh_fine_quality_gate_failed",),
                spreads={},
            )
        self._samples.append(observation)
        if len(self._samples) < self.config.minimum_valid_samples:
            return TemporalFineValidation(
                observation=None,
                ready=False,
                sample_count=len(self._samples),
                reasons=("collecting_temporal_window",),
                spreads={},
            )

        samples = tuple(self._samples)
        outer = np.asarray(
            [sample.outer_center_px for sample in samples],
            dtype=np.float64,
        )
        inner = np.asarray(
            [sample.inner_center_px for sample in samples],
            dtype=np.float64,
        )
        lateral = np.asarray(
            [sample.lateral_error_mm for sample in samples],
            dtype=np.float64,
        )
        axis = np.asarray(
            [sample.axis_error_deg for sample in samples],
            dtype=np.float64,
        )
        standoff = np.asarray(
            [sample.standoff_error_mm for sample in samples],
            dtype=np.float64,
        )
        reprojection = np.asarray(
            [sample.reprojection_error_px for sample in samples],
            dtype=np.float64,
        )
        outer_range = np.ptp(outer, axis=0)
        inner_range = np.ptp(inner, axis=0)
        spreads = {
            "outer_center_range_px": float(np.linalg.norm(outer_range)),
            "inner_center_range_px": float(np.linalg.norm(inner_range)),
            "lateral_range_mm": float(np.ptp(lateral)),
            "axis_range_deg": float(np.ptp(axis)),
            "standoff_range_mm": float(np.ptp(standoff)),
        }
        reasons: list[str] = []
        if max(
            spreads["outer_center_range_px"],
            spreads["inner_center_range_px"],
        ) > self.config.maximum_center_range_px:
            reasons.append("temporal_center_unstable")
        if (
            spreads["lateral_range_mm"]
            > self.config.maximum_lateral_range_mm
        ):
            reasons.append("temporal_lateral_unstable")
        if spreads["axis_range_deg"] > self.config.maximum_axis_range_deg:
            reasons.append("temporal_axis_unstable")
        if (
            spreads["standoff_range_mm"]
            > self.config.maximum_standoff_range_mm
        ):
            reasons.append("temporal_standoff_unstable")

        reference = samples[-1]
        filtered = FineObservation(
            image_center_px=reference.image_center_px,
            outer_center_px=tuple(
                float(value) for value in np.median(outer, axis=0)
            ),
            inner_center_px=tuple(
                float(value) for value in np.median(inner, axis=0)
            ),
            lateral_error_mm=float(np.median(lateral)),
            axis_error_deg=float(np.median(axis)),
            standoff_error_mm=float(np.median(standoff)),
            reprojection_error_px=float(np.median(reprojection)),
            quality_gate_pass=not reasons,
        )
        return TemporalFineValidation(
            observation=filtered,
            ready=not reasons,
            sample_count=len(samples),
            reasons=tuple(reasons),
            spreads=spreads,
        )


class ReobservationAction(str, Enum):
    MOVE = "move"
    HOLD_AND_REOBSERVE = "hold_and_reobserve"
    ABORT = "abort"


@dataclass(frozen=True)
class ReobservationDecision:
    action: ReobservationAction
    consecutive_failures: int
    reason: str


class FailClosedReobservationGate:
    """Pause on invalid vision; resume only from a fresh valid observation."""

    def __init__(self, maximum_reobservation_frames: int = 5):
        if maximum_reobservation_frames < 1:
            raise ValueError("maximum_reobservation_frames must be positive.")
        self.maximum_reobservation_frames = maximum_reobservation_frames
        self.consecutive_failures = 0

    def reset(self) -> None:
        self.consecutive_failures = 0

    def update(self, fresh_visual_authorization: bool) -> ReobservationDecision:
        if fresh_visual_authorization:
            recovered = self.consecutive_failures > 0
            self.consecutive_failures = 0
            return ReobservationDecision(
                action=ReobservationAction.MOVE,
                consecutive_failures=0,
                reason=(
                    "fresh_visual_authorization_recovered"
                    if recovered
                    else "fresh_visual_authorization_valid"
                ),
            )
        self.consecutive_failures += 1
        if self.consecutive_failures > self.maximum_reobservation_frames:
            return ReobservationDecision(
                action=ReobservationAction.ABORT,
                consecutive_failures=self.consecutive_failures,
                reason="visual_reobservation_budget_exhausted",
            )
        return ReobservationDecision(
            action=ReobservationAction.HOLD_AND_REOBSERVE,
            consecutive_failures=self.consecutive_failures,
            reason="fresh_visual_authorization_missing_hold_position",
        )
