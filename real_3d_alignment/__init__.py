"""Unified coarse-to-fine real-alignment orchestration layer."""

from .coarse_vision import (
    CoarseImageBasedVisualServo,
    CoarseServoCommand,
    CoarseServoConfig,
    RingDetection,
    TraditionalRingDetector,
    TraditionalRingDetectorConfig,
)
from .pipeline import run_pipeline
from .staged_alignment import (
    AlignmentDecision,
    AlignmentPhase,
    AlignmentThresholds,
    CoarseObservation,
    FineObservation,
    StagedAlignmentGate,
    fine_observation_from_pose_report,
)
from .visual_loop import StagedVisualAlignmentLoop, VisualLoopRecord, VisualLoopResult

__all__ = [
    "AlignmentDecision",
    "AlignmentPhase",
    "AlignmentThresholds",
    "CoarseImageBasedVisualServo",
    "CoarseObservation",
    "CoarseServoCommand",
    "CoarseServoConfig",
    "FineObservation",
    "RingDetection",
    "StagedVisualAlignmentLoop",
    "StagedAlignmentGate",
    "TraditionalRingDetector",
    "TraditionalRingDetectorConfig",
    "VisualLoopRecord",
    "VisualLoopResult",
    "fine_observation_from_pose_report",
    "run_pipeline",
]
