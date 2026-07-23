from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .geometry import clip_norm


ACTION_DIM = 6
TRANSLATION_SLICE = slice(0, 3)
ROTATION_SLICE = slice(3, 5)
INSERT_INDEX = 5


def zero_action() -> np.ndarray:
    return np.zeros(ACTION_DIM, dtype=np.float64)


@dataclass
class GeometricAlignmentController:
    """Bounded one-step task-space controller for single-arm precision alignment."""

    cfg: dict[str, Any]
    safety_cfg: dict[str, Any]

    def action(self, observed: dict[str, Any]) -> np.ndarray:
        action = zero_action()
        if not bool(observed.get("quality_gate_pass", True)):
            return action

        xy = np.asarray(observed["observed_lateral_xy_mm"], dtype=np.float64)
        depth = float(observed["observed_depth_error_mm"])
        axis_xy = np.asarray(observed["observed_axis_error_xy_deg"], dtype=np.float64)

        translation = np.asarray(
            [
                float(self.cfg["k_lateral"]) * xy[0],
                float(self.cfg["k_lateral"]) * xy[1],
                float(self.cfg["k_depth"]) * depth,
            ],
            dtype=np.float64,
        )
        action[TRANSLATION_SLICE] = clip_norm(
            translation,
            float(self.safety_cfg["max_base_translation_step_mm"]),
        )

        rotation = float(self.cfg["k_axis"]) * axis_xy
        action[ROTATION_SLICE] = clip_norm(
            rotation,
            float(self.safety_cfg["max_base_rotation_step_deg"]),
        )

        if self._advance_gate(observed):
            action[INSERT_INDEX] = min(
                float(self.cfg["insertion_step_mm"]),
                float(self.safety_cfg["max_base_insert_step_mm"]),
                max(0.0, float(observed["remaining_insert_depth_mm"])),
            )

        phase_cfg = self.cfg.get("phase_aware_insert", {}) or {}
        if bool(phase_cfg.get("enabled", False)):
            # Once the visual estimate reaches the port plane, continuing to
            # regulate signed plane error pulls the instrument back out. The
            # phase controller freezes that axial centering term and advances
            # only while the stricter visual pose gate is satisfied. The G96
            # runtime clearance gate remains the final safety authority.
            pose_safe = (
                float(observed["observed_lateral_error_mm"])
                <= float(phase_cfg.get("entry_lateral_mm", 0.12))
                and float(observed["observed_axis_error_deg"])
                <= float(phase_cfg.get("entry_axis_deg", 0.50))
            )
            reached_plane = float(observed["observed_depth_error_mm"]) <= float(
                phase_cfg.get("entry_plane_depth_mm", 0.30)
            )
            # The deployable observation contract exposes remaining insertion
            # depth, rather than simulator-only contact state. Derive progress
            # from that task-plan quantity so phase latching works equally in
            # the task-space and MuJoCo execution paths.
            target_depth = max(0.0, float(observed.get("target_insert_depth_mm", 0.0)))
            remaining_depth = max(0.0, float(observed.get("remaining_insert_depth_mm", target_depth)))
            inserted_depth = max(0.0, target_depth - remaining_depth)
            axial_hold_depth = max(
                0.0, float(phase_cfg.get("axial_hold_after_insert_mm", 0.01))
            )
            insertion_phase_started = inserted_depth >= axial_hold_depth
            # A noisy plane estimate must not pull a tool back out after a
            # measurable insertion has started. Lateral/axis corrections stay
            # active and the downstream visual safety gate still decides
            # whether the next forward micro-step is admissible.
            phase_entry_ready = pose_safe and reached_plane
            if phase_entry_ready or insertion_phase_started:
                action[2] = 0.0
                # After entering the insertion phase, keep proposing a
                # bounded forward micro-step. The later G96 gate evaluates
                # every proposal against the current visual estimate and can
                # cap or reject it, so this phase latch never bypasses safety.
                action[INSERT_INDEX] = min(
                    float(phase_cfg.get("insertion_step_mm", self.cfg["insertion_step_mm"])),
                    float(self.safety_cfg["max_base_insert_step_mm"]),
                    max(0.0, float(observed["remaining_insert_depth_mm"])),
                )
        return action

    def _advance_gate(self, observed: dict[str, Any]) -> bool:
        return (
            float(observed["observed_lateral_error_mm"]) <= float(self.cfg["advance_gate_lateral_mm"])
            and abs(float(observed["observed_depth_error_mm"])) <= float(self.cfg["advance_gate_depth_mm"])
            and float(observed["observed_axis_error_deg"]) <= float(self.cfg["advance_gate_axis_deg"])
        )
