from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class AlignmentPhase(str, Enum):
    SEARCH = "search"
    COARSE = "coarse_alignment"
    FINE = "fine_alignment"
    OBSERVE_ACTIVE = "active_multiview_observation"
    ALIGNED = "aligned"


@dataclass(frozen=True)
class CoarseObservation:
    image_size_px: tuple[int, int]
    target_center_px: tuple[float, float]
    confidence: float

    @property
    def image_center_px(self) -> tuple[float, float]:
        width, height = self.image_size_px
        return (0.5 * width, 0.5 * height)

    @property
    def center_error_px(self) -> float:
        return float(
            np.linalg.norm(
                np.asarray(self.target_center_px, dtype=np.float64)
                - np.asarray(self.image_center_px, dtype=np.float64)
            )
        )


@dataclass(frozen=True)
class FineObservation:
    image_center_px: tuple[float, float]
    outer_center_px: tuple[float, float]
    inner_center_px: tuple[float, float]
    lateral_error_mm: float
    axis_error_deg: float
    standoff_error_mm: float
    reprojection_error_px: float
    quality_gate_pass: bool

    @property
    def optical_outer_error_px(self) -> float:
        return float(
            np.linalg.norm(
                np.asarray(self.outer_center_px, dtype=np.float64)
                - np.asarray(self.image_center_px, dtype=np.float64)
            )
        )

    @property
    def outer_inner_concentricity_px(self) -> float:
        return float(
            np.linalg.norm(
                np.asarray(self.outer_center_px, dtype=np.float64)
                - np.asarray(self.inner_center_px, dtype=np.float64)
            )
        )


@dataclass(frozen=True)
class ActiveMultiviewObservation:
    observation_id: int
    view_count: int
    all_views_reachable: bool
    all_rings_detected: bool
    lateral_error_mm: float
    axis_error_deg: float
    standoff_error_mm: float
    normalized_conic_residual: float
    covariance_condition: float


@dataclass(frozen=True)
class AlignmentThresholds:
    minimum_coarse_confidence: float = 0.50
    coarse_to_fine_center_error_px: float = 30.0
    maximum_optical_outer_error_px: float = 2.0
    maximum_outer_inner_concentricity_px: float = 1.5
    maximum_lateral_error_mm: float = 0.20
    maximum_axis_error_deg: float = 1.0
    maximum_standoff_error_mm: float = 0.30
    maximum_reprojection_error_px: float = 0.50
    required_stable_frames: int = 5
    require_active_multiview_confirmation: bool = False
    minimum_active_view_count: int = 3
    maximum_active_lateral_error_mm: float = 0.20
    maximum_active_axis_error_deg: float = 2.0
    maximum_active_standoff_error_mm: float = 0.70
    maximum_active_normalized_conic_residual: float = 0.01
    maximum_active_covariance_condition: float = 3.0e8

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_coarse_confidence <= 1.0:
            raise ValueError("minimum_coarse_confidence must be in [0, 1].")
        if self.required_stable_frames < 1:
            raise ValueError("required_stable_frames must be positive.")
        if self.minimum_active_view_count < 2:
            raise ValueError("minimum_active_view_count must be at least 2.")
        numeric_limits = (
            self.coarse_to_fine_center_error_px,
            self.maximum_optical_outer_error_px,
            self.maximum_outer_inner_concentricity_px,
            self.maximum_lateral_error_mm,
            self.maximum_axis_error_deg,
            self.maximum_standoff_error_mm,
            self.maximum_reprojection_error_px,
            self.maximum_active_lateral_error_mm,
            self.maximum_active_axis_error_deg,
            self.maximum_active_standoff_error_mm,
            self.maximum_active_normalized_conic_residual,
            self.maximum_active_covariance_condition,
        )
        if any(value < 0.0 for value in numeric_limits):
            raise ValueError("Alignment thresholds must be nonnegative.")


@dataclass(frozen=True)
class AlignmentDecision:
    phase: AlignmentPhase
    hold_position: bool
    insertion_handoff_ready: bool
    stable_frames: int
    reasons: tuple[str, ...]
    metrics: dict[str, float | bool | int]


class StagedAlignmentGate:
    """Fail-closed coarse/fine alignment state machine.

    The gate does not command the robot and does not execute insertion. It only
    classifies the current observation and authorizes the existing insertion
    controller after a stable multi-frame fine-alignment confirmation.
    """

    def __init__(self, thresholds: AlignmentThresholds | None = None):
        self.thresholds = thresholds or AlignmentThresholds()
        self.stable_frames = 0
        self._last_active_observation_id: int | None = None

    def reset(self) -> None:
        self.stable_frames = 0
        self._last_active_observation_id = None

    def update(
        self,
        *,
        coarse: CoarseObservation | None,
        fine: FineObservation | None,
        active_multiview: ActiveMultiviewObservation | None = None,
    ) -> AlignmentDecision:
        if coarse is None:
            self.stable_frames = 0
            return self._decision(
                AlignmentPhase.SEARCH,
                hold_position=True,
                reasons=("coarse_target_missing",),
                metrics={},
            )

        coarse_metrics: dict[str, float | bool | int] = {
            "coarse_confidence": float(coarse.confidence),
            "coarse_center_error_px": coarse.center_error_px,
        }
        if coarse.confidence < self.thresholds.minimum_coarse_confidence:
            self.stable_frames = 0
            return self._decision(
                AlignmentPhase.SEARCH,
                hold_position=True,
                reasons=("coarse_confidence_below_threshold",),
                metrics=coarse_metrics,
            )

        if (
            coarse.center_error_px
            > self.thresholds.coarse_to_fine_center_error_px
        ):
            self.stable_frames = 0
            return self._decision(
                AlignmentPhase.COARSE,
                hold_position=False,
                reasons=("target_outside_fine_alignment_region",),
                metrics=coarse_metrics,
            )

        if fine is None:
            self.stable_frames = 0
            return self._decision(
                AlignmentPhase.FINE,
                hold_position=True,
                reasons=("fine_observation_missing",),
                metrics=coarse_metrics,
            )

        fine_metrics: dict[str, float | bool | int] = {
            **coarse_metrics,
            "optical_outer_error_px": fine.optical_outer_error_px,
            "outer_inner_concentricity_px": fine.outer_inner_concentricity_px,
            "lateral_error_mm": abs(float(fine.lateral_error_mm)),
            "axis_error_deg": abs(float(fine.axis_error_deg)),
            "standoff_error_mm": abs(float(fine.standoff_error_mm)),
            "reprojection_error_px": float(fine.reprojection_error_px),
            "quality_gate_pass": bool(fine.quality_gate_pass),
        }
        failed = self._failed_fine_checks(fine)
        if failed:
            self.stable_frames = 0
            return self._decision(
                AlignmentPhase.FINE,
                hold_position=not fine.quality_gate_pass,
                reasons=tuple(failed),
                metrics=fine_metrics,
            )

        if self.thresholds.require_active_multiview_confirmation:
            if active_multiview is None:
                self.stable_frames = 0
                return self._decision(
                    AlignmentPhase.OBSERVE_ACTIVE,
                    hold_position=True,
                    reasons=("active_multiview_observation_required",),
                    metrics=fine_metrics,
                )
            active_metrics: dict[str, float | bool | int] = {
                **fine_metrics,
                "active_observation_id": int(
                    active_multiview.observation_id
                ),
                "active_view_count": int(active_multiview.view_count),
                "active_all_views_reachable": bool(
                    active_multiview.all_views_reachable
                ),
                "active_all_rings_detected": bool(
                    active_multiview.all_rings_detected
                ),
                "active_lateral_error_mm": abs(
                    float(active_multiview.lateral_error_mm)
                ),
                "active_axis_error_deg": abs(
                    float(active_multiview.axis_error_deg)
                ),
                "active_standoff_error_mm": abs(
                    float(active_multiview.standoff_error_mm)
                ),
                "active_normalized_conic_residual": float(
                    active_multiview.normalized_conic_residual
                ),
                "active_covariance_condition": float(
                    active_multiview.covariance_condition
                ),
            }
            active_failed = self._failed_active_checks(active_multiview)
            if active_failed:
                self.stable_frames = 0
                return self._decision(
                    AlignmentPhase.OBSERVE_ACTIVE,
                    hold_position=True,
                    reasons=tuple(active_failed),
                    metrics=active_metrics,
                )
            if (
                self._last_active_observation_id is not None
                and active_multiview.observation_id
                <= self._last_active_observation_id
            ):
                self.stable_frames = 0
                return self._decision(
                    AlignmentPhase.OBSERVE_ACTIVE,
                    hold_position=True,
                    reasons=("active_multiview_observation_not_fresh",),
                    metrics=active_metrics,
                )
            self._last_active_observation_id = (
                active_multiview.observation_id
            )
            fine_metrics = active_metrics

        self.stable_frames += 1
        if self.stable_frames < self.thresholds.required_stable_frames:
            return self._decision(
                AlignmentPhase.FINE,
                hold_position=True,
                reasons=("waiting_for_stable_confirmation",),
                metrics=fine_metrics,
            )
        return self._decision(
            AlignmentPhase.ALIGNED,
            hold_position=True,
            reasons=("three_axis_alignment_confirmed",),
            metrics=fine_metrics,
        )

    def _failed_fine_checks(self, fine: FineObservation) -> list[str]:
        limits = self.thresholds
        failed: list[str] = []
        if not fine.quality_gate_pass:
            failed.append("fine_quality_gate_failed")
        if fine.optical_outer_error_px > limits.maximum_optical_outer_error_px:
            failed.append("camera_axis_not_centered_on_outer_ring")
        if (
            fine.outer_inner_concentricity_px
            > limits.maximum_outer_inner_concentricity_px
        ):
            failed.append("inner_outer_ring_not_concentric")
        if abs(fine.lateral_error_mm) > limits.maximum_lateral_error_mm:
            failed.append("lateral_error_above_threshold")
        if abs(fine.axis_error_deg) > limits.maximum_axis_error_deg:
            failed.append("axis_error_above_threshold")
        if abs(fine.standoff_error_mm) > limits.maximum_standoff_error_mm:
            failed.append("standoff_error_above_threshold")
        if fine.reprojection_error_px > limits.maximum_reprojection_error_px:
            failed.append("reprojection_error_above_threshold")
        return failed

    def _failed_active_checks(
        self,
        active: ActiveMultiviewObservation,
    ) -> list[str]:
        limits = self.thresholds
        failed: list[str] = []
        if active.observation_id < 0:
            failed.append("active_observation_id_invalid")
        if not active.all_views_reachable:
            failed.append("active_view_unreachable")
        if not active.all_rings_detected:
            failed.append("active_ring_detection_incomplete")
        if active.view_count < limits.minimum_active_view_count:
            failed.append("active_view_count_below_threshold")
        if (
            abs(active.lateral_error_mm)
            > limits.maximum_active_lateral_error_mm
        ):
            failed.append("active_lateral_error_above_threshold")
        if abs(active.axis_error_deg) > limits.maximum_active_axis_error_deg:
            failed.append("active_axis_error_above_threshold")
        if (
            abs(active.standoff_error_mm)
            > limits.maximum_active_standoff_error_mm
        ):
            failed.append("active_standoff_error_above_threshold")
        if (
            active.normalized_conic_residual
            > limits.maximum_active_normalized_conic_residual
        ):
            failed.append("active_conic_residual_above_threshold")
        if (
            not math.isfinite(active.covariance_condition)
            or active.covariance_condition
            > limits.maximum_active_covariance_condition
        ):
            failed.append("active_covariance_condition_above_threshold")
        return failed

    def _decision(
        self,
        phase: AlignmentPhase,
        *,
        hold_position: bool,
        reasons: tuple[str, ...],
        metrics: dict[str, float | bool | int],
    ) -> AlignmentDecision:
        return AlignmentDecision(
            phase=phase,
            hold_position=hold_position,
            insertion_handoff_ready=phase is AlignmentPhase.ALIGNED,
            stable_frames=self.stable_frames,
            reasons=reasons,
            metrics={**metrics, "stable_frames": self.stable_frames},
        )


def fine_observation_from_pose_report(
    report: dict[str, Any],
    *,
    image_size_px: tuple[int, int],
    target_standoff_mm: float,
    quality_gate_pass: bool,
) -> FineObservation:
    """Convert the existing ring-PnP report into the staged gate contract."""

    outer = report.get("outer_ellipse")
    inner = report.get("inner_ellipse")
    if not isinstance(outer, dict) or not isinstance(inner, dict):
        raise ValueError("Fine alignment requires both outer and inner ring estimates.")

    width, height = image_size_px
    pose = np.asarray(report["pose_camera_trocar_mm_deg"][:3], dtype=np.float64)
    axis = np.asarray(report["trocar_axis_camera"], dtype=np.float64)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-12:
        raise ValueError("Trocar axis must have nonzero norm.")
    axis = axis / axis_norm
    axis_error_deg = math.degrees(math.acos(float(np.clip(abs(axis[2]), 0.0, 1.0))))

    metrics = report.get("metrics", {})
    return FineObservation(
        image_center_px=(0.5 * width, 0.5 * height),
        outer_center_px=tuple(float(value) for value in outer["center"]),
        inner_center_px=tuple(float(value) for value in inner["center"]),
        lateral_error_mm=float(np.linalg.norm(pose[:2])),
        axis_error_deg=axis_error_deg,
        standoff_error_mm=float(pose[2] - target_standoff_mm),
        reprojection_error_px=float(metrics["mean_reprojection_error_px"]),
        quality_gate_pass=quality_gate_pass,
    )
