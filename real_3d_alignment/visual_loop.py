from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Protocol

import numpy as np

from .coarse_vision import (
    CoarseImageBasedVisualServo,
    CoarseServoCommand,
    RingDetection,
    TraditionalRingDetector,
)
from .staged_alignment import (
    AlignmentDecision,
    AlignmentPhase,
    FineObservation,
    StagedAlignmentGate,
)


class EyeInHandPlant(Protocol):
    """Deployment-shaped boundary around a simulator or real robot."""

    def capture_rgb(self) -> np.ndarray: ...

    def apply_camera_xy_step(self, command_mm: tuple[float, float]) -> None: ...


class FineObservationProvider(Protocol):
    def estimate(self, image_rgb: np.ndarray) -> FineObservation | None: ...


@dataclass(frozen=True)
class VisualLoopRecord:
    step: int
    phase: str
    target_detected: bool
    center_error_px: float | None
    confidence: float | None
    radius_px: float | None
    command_camera_xy_mm: tuple[float, float]
    estimated_depth_mm: float | None
    stable_frames: int
    insertion_handoff_ready: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class VisualLoopResult:
    stop_reason: str
    records: tuple[VisualLoopRecord, ...]
    final_decision: AlignmentDecision

    def to_dict(self) -> dict:
        return {
            "stop_reason": self.stop_reason,
            "records": [asdict(record) for record in self.records],
            "final_decision": asdict(self.final_decision),
        }


class StagedVisualAlignmentLoop:
    """Unified RGB-driven coarse-to-fine loop.

    M0/M1 execute coarse image-based alignment. A fine provider can be plugged
    into the same loop in M3 without changing the detector or plant boundary.
    """

    def __init__(
        self,
        *,
        plant: EyeInHandPlant,
        coarse_detector: TraditionalRingDetector,
        coarse_servo: CoarseImageBasedVisualServo,
        gate: StagedAlignmentGate,
        fine_provider: FineObservationProvider | None = None,
        step_observer: Callable[
            [
                np.ndarray,
                RingDetection | None,
                CoarseServoCommand | None,
                AlignmentDecision,
                VisualLoopRecord,
            ],
            None,
        ]
        | None = None,
    ):
        self.plant = plant
        self.coarse_detector = coarse_detector
        self.coarse_servo = coarse_servo
        self.gate = gate
        self.fine_provider = fine_provider
        self.step_observer = step_observer

    def run(self, *, maximum_steps: int = 100) -> VisualLoopResult:
        if maximum_steps < 1:
            raise ValueError("maximum_steps must be positive.")
        records: list[VisualLoopRecord] = []
        final_decision: AlignmentDecision | None = None
        stop_reason = "maximum_steps_reached"

        for step in range(maximum_steps):
            image = self.plant.capture_rgb()
            detection = self.coarse_detector.detect(image)
            fine = (
                self.fine_provider.estimate(image)
                if detection is not None and self.fine_provider is not None
                else None
            )
            final_decision = self.gate.update(
                coarse=None if detection is None else detection.observation,
                fine=fine,
            )
            command: CoarseServoCommand | None = None
            if (
                detection is not None
                and final_decision.phase is AlignmentPhase.COARSE
                and not final_decision.hold_position
            ):
                command = self.coarse_servo.command(detection)

            record = self._record(step, detection, command, final_decision)
            records.append(record)
            if self.step_observer is not None:
                self.step_observer(
                    image, detection, command, final_decision, record
                )
            if final_decision.insertion_handoff_ready:
                stop_reason = "insertion_handoff_ready"
                break
            if final_decision.phase is AlignmentPhase.FINE and self.fine_provider is None:
                stop_reason = "coarse_alignment_complete_waiting_for_fine_provider"
                break
            if command is not None:
                self.plant.apply_camera_xy_step(command.camera_xy_mm)

        if final_decision is None:  # pragma: no cover - guarded by maximum_steps
            raise RuntimeError("Visual loop produced no decision.")
        return VisualLoopResult(
            stop_reason=stop_reason,
            records=tuple(records),
            final_decision=final_decision,
        )

    @staticmethod
    def _record(
        step: int,
        detection: RingDetection | None,
        command: CoarseServoCommand | None,
        decision: AlignmentDecision,
    ) -> VisualLoopRecord:
        return VisualLoopRecord(
            step=step,
            phase=decision.phase.value,
            target_detected=detection is not None,
            center_error_px=(
                None
                if detection is None
                else detection.observation.center_error_px
            ),
            confidence=(
                None if detection is None else detection.observation.confidence
            ),
            radius_px=None if detection is None else detection.radius_px,
            command_camera_xy_mm=(
                (0.0, 0.0) if command is None else command.camera_xy_mm
            ),
            estimated_depth_mm=(
                None if command is None else command.estimated_depth_mm
            ),
            stable_frames=decision.stable_frames,
            insertion_handoff_ready=decision.insertion_handoff_ready,
            reasons=decision.reasons,
        )
