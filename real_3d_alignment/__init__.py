"""Unified coarse-to-fine real-alignment orchestration layer."""

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

__all__ = [
    "AlignmentDecision",
    "AlignmentPhase",
    "AlignmentThresholds",
    "CoarseObservation",
    "FineObservation",
    "StagedAlignmentGate",
    "fine_observation_from_pose_report",
    "run_pipeline",
]
