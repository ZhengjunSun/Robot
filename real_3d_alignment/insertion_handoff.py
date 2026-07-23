from __future__ import annotations

from dataclasses import dataclass

from single_arm_precision_rl.clearance_contract import (
    ClearanceContract,
    ClearanceGeometry,
    ClearanceSample,
    ClearanceSampleResult,
)

from .staged_alignment import FineObservation


@dataclass(frozen=True)
class TraditionalInsertionConfig:
    calibrated_approach_mm: float = 10.0
    target_wall_traversal_mm: float = 2.5
    insertion_step_mm: float = 0.25
    minimum_robust_clearance_mm: float = 0.04
    uncertainty_margin_mm: float = 0.02


@dataclass(frozen=True)
class InsertionStepDecision:
    allow_motion: bool
    complete: bool
    commanded_step_mm: float
    insertion_depth_mm: float
    reason: str
    clearance: ClearanceSampleResult | None


class TraditionalInsertionHandoffController:
    """Fail-closed deterministic insertion after stable visual authorization."""

    def __init__(self, config: TraditionalInsertionConfig | None = None):
        self.config = config or TraditionalInsertionConfig()
        self.clearance_contract = ClearanceContract(
            geometry=ClearanceGeometry(
                port_inner_diameter_mm=1.0,
                tool_outer_diameter_mm=0.51,
                effective_wall_length_mm=2.5,
            )
        )
        self.clearance_contract.geometry.validate()

    @property
    def target_extension_mm(self) -> float:
        return (
            self.config.calibrated_approach_mm
            + self.config.target_wall_traversal_mm
        )

    def decide(
        self,
        *,
        insertion_handoff_ready: bool,
        fine: FineObservation | None,
        current_extension_mm: float,
    ) -> InsertionStepDecision:
        insertion_depth = max(
            0.0,
            float(current_extension_mm)
            - self.config.calibrated_approach_mm,
        )
        if not insertion_handoff_ready or fine is None:
            return InsertionStepDecision(
                False,
                False,
                0.0,
                insertion_depth,
                "visual_authorization_missing_or_revoked",
                None,
            )
        if not fine.quality_gate_pass:
            return InsertionStepDecision(
                False,
                False,
                0.0,
                insertion_depth,
                "fine_quality_gate_failed",
                None,
            )
        clearance = self.clearance_contract.evaluate_sample(
            ClearanceSample(
                lateral_error_mm=fine.lateral_error_mm,
                insertion_depth_mm=insertion_depth,
                axis_error_deg=fine.axis_error_deg,
                uncertainty_margin_mm=self.config.uncertainty_margin_mm,
            )
        )
        if clearance.robust_clearance_mm <= self.config.minimum_robust_clearance_mm:
            return InsertionStepDecision(
                False,
                False,
                0.0,
                insertion_depth,
                "visual_clearance_below_threshold",
                clearance,
            )
        remaining = max(
            0.0,
            self.target_extension_mm - float(current_extension_mm),
        )
        if remaining <= 1e-6:
            return InsertionStepDecision(
                False,
                True,
                0.0,
                insertion_depth,
                "target_insertion_depth_reached",
                clearance,
            )
        step = min(self.config.insertion_step_mm, remaining)
        return InsertionStepDecision(
            True,
            False,
            step,
            insertion_depth,
            "bounded_insertion_step_authorized",
            clearance,
        )
