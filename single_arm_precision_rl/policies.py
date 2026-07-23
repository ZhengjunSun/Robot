from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .controllers import ACTION_DIM, INSERT_INDEX
from .config import resolve_project_path


class ResidualPolicy(Protocol):
    name: str

    def reset(self, seed: int | None = None) -> None:
        ...

    def act(self, observation: np.ndarray, info: dict) -> np.ndarray:
        ...


@dataclass
class ZeroResidualPolicy:
    name: str = "zero"

    def reset(self, seed: int | None = None) -> None:
        return None

    def act(self, observation: np.ndarray, info: dict) -> np.ndarray:
        return np.zeros(ACTION_DIM, dtype=np.float64)


class RandomResidualPolicy:
    name = "random"

    def __init__(self, low: np.ndarray, high: np.ndarray, scale: float = 1.0):
        self.low = np.asarray(low, dtype=np.float64)
        self.high = np.asarray(high, dtype=np.float64)
        self.scale = float(scale)
        self.rng = np.random.default_rng()

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def act(self, observation: np.ndarray, info: dict) -> np.ndarray:
        return self.rng.uniform(self.low, self.high) * self.scale


class TremorAdvancePolicy:
    """Hand-written micro-step policy used only as an interpretable sanity baseline."""

    name = "tremor"

    def __init__(self, low: np.ndarray, high: np.ndarray):
        self.low = np.asarray(low, dtype=np.float64)
        self.high = np.asarray(high, dtype=np.float64)
        self.step = 0

    def reset(self, seed: int | None = None) -> None:
        self.step = 0

    def act(self, observation: np.ndarray, info: dict) -> np.ndarray:
        self.step += 1
        action = np.zeros(ACTION_DIM, dtype=np.float64)
        observed = info.get("observation_metrics", {})
        lateral = float(observed.get("observed_lateral_error_mm", 0.0))
        depth_abs = abs(float(observed.get("observed_depth_error_mm", 0.0)))
        axis = float(observed.get("observed_axis_error_deg", 0.0))
        gate = bool(observed.get("quality_gate_pass", False))
        if gate and lateral < 0.65 and depth_abs < 0.65 and axis < 3.2:
            phase = 1.0 if self.step % 2 == 0 else -1.0
            action[0] = 0.05 * phase
            action[1] = -0.05 * phase
            action[5] = min(0.18, self.high[5])
        return np.clip(action, self.low, self.high)


class PIDResidualPolicy:
    """Small fixed-gain PID-style residual baseline.

    This is a traditional-control comparison, not an RL policy. It adds a
    hand-tuned proportional/derivative correction on top of the geometric base
    action so we can quantify whether learned residuals beat a simple residual
    controller.
    """

    name = "pid"

    def __init__(self, low: np.ndarray, high: np.ndarray):
        self.low = np.asarray(low, dtype=np.float64)
        self.high = np.asarray(high, dtype=np.float64)
        self.previous_error: np.ndarray | None = None

    def reset(self, seed: int | None = None) -> None:
        self.previous_error = None

    def act(self, observation: np.ndarray, info: dict) -> np.ndarray:
        observed = info.get("observation_metrics", {})
        if not bool(observed.get("quality_gate_pass", False)):
            return np.zeros(ACTION_DIM, dtype=np.float64)

        error = np.asarray(
            [
                *observed.get("observed_lateral_xy_mm", [0.0, 0.0]),
                observed.get("observed_depth_error_mm", 0.0),
                *observed.get("observed_axis_error_xy_deg", [0.0, 0.0]),
            ],
            dtype=np.float64,
        )
        derivative = np.zeros_like(error) if self.previous_error is None else error - self.previous_error
        self.previous_error = error

        action = np.zeros(ACTION_DIM, dtype=np.float64)
        action[0:2] = 0.075 * error[0:2] + 0.030 * derivative[0:2]
        action[2] = 0.055 * error[2] + 0.020 * derivative[2]
        action[3:5] = 0.030 * error[3:5] + 0.012 * derivative[3:5]

        lateral = float(observed.get("observed_lateral_error_mm", 0.0))
        depth_abs = abs(float(observed.get("observed_depth_error_mm", 0.0)))
        axis = float(observed.get("observed_axis_error_deg", 0.0))
        if lateral < 0.65 and depth_abs < 0.65 and axis < 3.2:
            action[INSERT_INDEX] = 0.12
        return np.clip(action, self.low, self.high)


class DampedIKResidualPolicy:
    """Damped pseudo-IK residual baseline for conservative fine alignment."""

    name = "damped_ik"

    def __init__(self, low: np.ndarray, high: np.ndarray):
        self.low = np.asarray(low, dtype=np.float64)
        self.high = np.asarray(high, dtype=np.float64)

    def reset(self, seed: int | None = None) -> None:
        return None

    def act(self, observation: np.ndarray, info: dict) -> np.ndarray:
        observed = info.get("observation_metrics", {})
        metrics = info.get("metrics", {})
        if not bool(observed.get("quality_gate_pass", False)):
            return np.zeros(ACTION_DIM, dtype=np.float64)

        xy = np.asarray(observed.get("observed_lateral_xy_mm", [0.0, 0.0]), dtype=np.float64)
        depth = float(observed.get("observed_depth_error_mm", 0.0))
        axis_xy = np.asarray(observed.get("observed_axis_error_xy_deg", [0.0, 0.0]), dtype=np.float64)

        lateral = float(observed.get("observed_lateral_error_mm", float(np.linalg.norm(xy))))
        depth_abs = abs(depth)
        axis = float(observed.get("observed_axis_error_deg", float(np.linalg.norm(axis_xy))))
        normalized_error = lateral / 3.0 + depth_abs / 3.0 + axis / 18.0
        damping = 1.0 / (1.0 + 0.8 * normalized_error * normalized_error)

        min_margin = float(metrics.get("min_joint_limit_margin_deg", 180.0))
        joint_scale = float(np.clip((min_margin - 12.0) / 48.0, 0.0, 1.0))
        scale = damping * joint_scale

        action = np.zeros(ACTION_DIM, dtype=np.float64)
        action[0:2] = scale * 0.090 * xy
        action[2] = scale * 0.065 * depth
        action[3:5] = scale * 0.040 * axis_xy
        if lateral < 0.60 and depth_abs < 0.60 and axis < 3.0:
            action[INSERT_INDEX] = 0.10 * max(joint_scale, 0.0)
        return np.clip(action, self.low, self.high)


class AuxiliaryTriggerDepthPolicy:
    """Explicit Stage-G8 trigger/depth advisor trained from MuJoCo labels.

    This is a diagnostic behavior policy, not the final RL method. It checks
    whether supervised trigger/depth labels are informative enough before the
    same signal is promoted into SAC warm-start or constrained auxiliary losses.
    """

    name = "aux_trigger_depth"

    def __init__(self, low: np.ndarray, high: np.ndarray, cfg: dict[str, Any] | None = None):
        self.low = np.asarray(low, dtype=np.float64)
        self.high = np.asarray(high, dtype=np.float64)
        self.cfg = cfg or {}
        model_cfg = self.cfg.get("auxiliary_trigger_depth_model", {})
        model_path = str(model_cfg.get("path", "")).strip()
        if not model_path:
            raise ValueError("auxiliary_trigger_depth_model.path is required for aux_trigger_depth policy")
        with resolve_project_path(model_path).open("r", encoding="utf-8") as stream:
            self.model = json.load(stream)
        if str(self.model.get("model_type", "")) != "standardized_auxiliary_trigger_depth":
            raise ValueError(f"Unsupported auxiliary trigger/depth model: {self.model.get('model_type')}")
        self.step = 0

    def reset(self, seed: int | None = None) -> None:
        self.step = 0

    def act(self, observation: np.ndarray, info: dict) -> np.ndarray:
        self.step += 1
        observed = info.get("observation_metrics", {})
        metrics = info.get("metrics", {})
        if not bool(observed.get("quality_gate_pass", False)):
            return np.zeros(ACTION_DIM, dtype=np.float64)

        values = np.asarray(
            [self._feature_value(name, observed, metrics, info) for name in self.model["feature_names"]],
            dtype=np.float64,
        )
        mean = np.asarray(self.model["feature_mean"], dtype=np.float64)
        std = np.asarray(self.model["feature_std"], dtype=np.float64)
        if values.shape != mean.shape:
            raise ValueError(f"auxiliary feature shape mismatch: values={values.shape}, mean={mean.shape}")
        z = (values - mean) / np.where(std < 1e-9, 1.0, std)

        safe_prob = self._classifier_probability("safe_trigger_label", z)
        unsafe_prob = self._classifier_probability("unsafe_trigger_label", z)
        missed_prob = self._classifier_probability("missed_safe_trigger_label", z)
        target_insert_step = self._regression_value("target_insert_step_mm", z)

        model_cfg = self.cfg.get("auxiliary_trigger_depth_model", {})
        min_insert_step = float(model_cfg.get("min_insert_step_mm", 0.02))
        margin = float(model_cfg.get("safe_probability_margin", self.model.get("safe_probability_margin", 0.04)))
        safe_threshold = float(
            model_cfg.get(
                "safe_threshold",
                self.model.get("classifiers", {}).get("safe_trigger_label", {}).get("threshold", 0.5),
            )
        )
        unsafe_threshold = float(
            model_cfg.get(
                "unsafe_threshold",
                self.model.get("classifiers", {}).get("unsafe_trigger_label", {}).get("threshold", 0.5),
            )
        )
        remaining = float(observed.get("remaining_insert_depth_mm", 0.0))
        trigger = (
            safe_prob >= safe_threshold
            and unsafe_prob < unsafe_threshold
            and safe_prob >= unsafe_prob + margin
            and target_insert_step >= min_insert_step
            and remaining > 1e-9
        )
        if bool(model_cfg.get("allow_missed_safe_trigger_boost", True)) and not trigger:
            missed_threshold = float(
                model_cfg.get(
                    "missed_threshold",
                    self.model.get("classifiers", {}).get("missed_safe_trigger_label", {}).get("threshold", 0.5),
                )
            )
            trigger = (
                missed_prob >= missed_threshold
                and unsafe_prob < unsafe_threshold
                and target_insert_step >= min_insert_step
                and remaining > 1e-9
            )

        action = self._alignment_residual(observed, metrics, info)
        if trigger:
            action[INSERT_INDEX] = float(model_cfg.get("trigger_output", self.model.get("trigger_output", 1.0)))
        elif self._safe_continuation_trigger(model_cfg, observed, metrics, z, unsafe_prob):
            action[INSERT_INDEX] = float(model_cfg.get("trigger_output", self.model.get("trigger_output", 1.0)))
        return np.clip(action, self.low, self.high)

    def _classifier_probability(self, name: str, z: np.ndarray) -> float:
        clf = self.model.get("classifiers", {}).get(name)
        if not clf:
            return 0.0
        weights = np.asarray(clf["weights"], dtype=np.float64)
        logit = float(z @ weights + float(clf["bias"]))
        return float(1.0 / (1.0 + np.exp(-np.clip(logit, -50.0, 50.0))))

    def _regression_value(self, name: str, z: np.ndarray) -> float:
        reg = self.model.get("regressors", {}).get(name)
        if not reg:
            return 0.0
        weights = np.asarray(reg["weights"], dtype=np.float64)
        value = float(z @ weights + float(reg["bias"]))
        if name in {"lateral_feedback_x_mm", "lateral_feedback_y_mm"}:
            model_cfg = self.cfg.get("auxiliary_trigger_depth_model", {})
            max_feedback = float(model_cfg.get("lateral_feedback_vector_max_mm", 0.12))
            return float(np.clip(value, -max_feedback, max_feedback))
        return float(np.clip(value, 0.0, float(self.model.get("max_insert_step_mm", 0.12))))

    def _feature_value(
        self,
        name: str,
        observed: dict[str, Any],
        metrics: dict[str, Any],
        info: dict[str, Any],
    ) -> float:
        lateral = float(observed.get("observed_lateral_error_mm", 0.0))
        depth_abs = abs(float(observed.get("observed_depth_error_mm", 0.0)))
        axis = float(observed.get("observed_axis_error_deg", 0.0))
        insertion = float(metrics.get("insertion_depth_mm", 0.0))
        target = max(float(self.cfg.get("task", {}).get("target_insert_depth_mm", 8.0)), 1e-9)
        remaining = float(observed.get("remaining_insert_depth_mm", max(0.0, target - insertion)))
        action_info = info.get("action_info", {}) or {}
        mujoco = info.get("mujoco_metrics", {}) or {}
        if name == "lateral_error_mm":
            return lateral
        if name == "depth_abs_error_mm":
            return depth_abs
        if name == "axis_error_deg":
            return axis
        if name == "insertion_depth_mm":
            return insertion
        if name == "remaining_insert_depth_mm":
            return remaining
        if name == "insertion_fraction":
            return insertion / target
        if name == "remaining_fraction":
            return remaining / target
        if name == "phase_insert":
            phase_cfg = self.cfg.get("phase_action_gate", {})
            lateral_gate = float(phase_cfg.get("align_lateral_mm", self.cfg.get("safety", {}).get("allow_insert_lateral_mm", 0.45)))
            depth_gate = float(phase_cfg.get("depth_gate_mm", self.cfg.get("safety", {}).get("allow_insert_depth_mm", 0.45)))
            axis_gate = float(phase_cfg.get("align_axis_deg", self.cfg.get("safety", {}).get("allow_insert_axis_deg", 2.0)))
            return 1.0 if lateral <= lateral_gate and depth_abs <= depth_gate and axis <= axis_gate else 0.0
        if name == "step_fraction":
            max_steps = max(float(self.cfg.get("episode", {}).get("max_steps", 180)), 1.0)
            return float(metrics.get("step", self.step)) / max_steps
        if name == "contact_risk_probability":
            return float(action_info.get("contact_risk_probability", 0.0))
        if name == "mujoco_before_lateral_error_mm":
            return float(mujoco.get("lateral_error_mm", 0.0))
        if name == "mujoco_before_insertion_depth_mm":
            return float(mujoco.get("insertion_depth_mm", 0.0))
        if name == "mujoco_before_plane_depth_error_mm":
            return float(mujoco.get("plane_depth_error_mm", 0.0))
        if name == "insert_trigger_active":
            return 1.0 if bool(action_info.get("insert_trigger_active", False)) else 0.0
        if name == "insert_trigger_safe":
            return 1.0 if bool(action_info.get("insert_trigger_safe", False)) else 0.0
        if name == "insert_trigger_blocked":
            return 1.0 if bool(action_info.get("insert_trigger_blocked", False)) else 0.0
        if name == "final_action_insert_mm":
            action = info.get("final_action", [0.0] * ACTION_DIM)
            try:
                return float(action[INSERT_INDEX])
            except (IndexError, TypeError, ValueError):
                return 0.0
        return 0.0

    def _safe_continuation_trigger(
        self,
        model_cfg: dict[str, Any],
        observed: dict[str, Any],
        metrics: dict[str, Any],
        z: np.ndarray,
        unsafe_prob: float,
    ) -> bool:
        if not bool(model_cfg.get("use_depth_completion_head_for_trigger", False)):
            return False
        remaining = float(observed.get("remaining_insert_depth_mm", 0.0))
        if remaining <= 1e-9:
            return False
        insertion = float(metrics.get("insertion_depth_mm", 0.0))
        min_started = float(model_cfg.get("continuation_min_insertion_mm", 0.10))
        if insertion < min_started:
            return False
        if unsafe_prob >= float(model_cfg.get("continuation_unsafe_threshold", model_cfg.get("unsafe_threshold", 0.65))):
            return False
        lateral = float(observed.get("observed_lateral_error_mm", 0.0))
        depth_abs = abs(float(observed.get("observed_depth_error_mm", 0.0)))
        axis = float(observed.get("observed_axis_error_deg", 0.0))
        if lateral > float(model_cfg.get("continuation_lateral_gate_mm", 0.26)):
            return False
        if depth_abs > float(model_cfg.get("continuation_depth_gate_mm", 0.22)):
            return False
        if axis > float(model_cfg.get("continuation_axis_gate_deg", 1.05)):
            return False
        depth_need = self._regression_value("depth_completion_need_mm", z)
        return depth_need >= float(model_cfg.get("depth_completion_trigger_threshold_mm", 0.20))

    def _alignment_residual(
        self,
        observed: dict[str, Any],
        metrics: dict[str, Any],
        info: dict[str, Any],
    ) -> np.ndarray:
        model_cfg = self.cfg.get("auxiliary_trigger_depth_model", {})
        gain = float(model_cfg.get("alignment_gain", self.model.get("alignment_gain", 0.08)))
        max_residual = float(model_cfg.get("max_alignment_residual_mm", self.model.get("max_alignment_residual_mm", 0.08)))
        xy = np.asarray(observed.get("observed_lateral_xy_mm", [0.0, 0.0]), dtype=np.float64)
        depth = float(observed.get("observed_depth_error_mm", 0.0))
        axis_xy = np.asarray(observed.get("observed_axis_error_xy_deg", [0.0, 0.0]), dtype=np.float64)
        action = np.zeros(ACTION_DIM, dtype=np.float64)
        action[0:2] = np.clip(gain * xy, -max_residual, max_residual)
        action[2] = np.clip(0.5 * gain * depth, -max_residual, max_residual)
        action[3:5] = np.clip(0.35 * gain * axis_xy, -0.20, 0.20)
        if bool(model_cfg.get("use_regression_lateral_feedback_head", False)):
            values = np.asarray(
                [self._feature_value(name, observed, metrics, info) for name in self.model["feature_names"]],
                dtype=np.float64,
            )
            mean = np.asarray(self.model["feature_mean"], dtype=np.float64)
            std = np.asarray(self.model["feature_std"], dtype=np.float64)
            z = (values - mean) / np.where(std < 1e-9, 1.0, std)
            feedback_need = self._regression_value("lateral_feedback_need_mm", z)
            if feedback_need >= float(model_cfg.get("lateral_feedback_need_threshold_mm", 0.03)):
                xy_norm = float(np.linalg.norm(xy))
                if xy_norm > 1e-9:
                    direction = xy / xy_norm
                    delta = (
                        float(model_cfg.get("lateral_feedback_head_gain", 0.6))
                        * min(feedback_need, float(model_cfg.get("lateral_feedback_head_max_mm", 0.12)))
                        * direction
                    )
                    action[0:2] += delta
        if bool(model_cfg.get("use_regression_lateral_feedback_vector_head", False)):
            values = np.asarray(
                [self._feature_value(name, observed, metrics, info) for name in self.model["feature_names"]],
                dtype=np.float64,
            )
            mean = np.asarray(self.model["feature_mean"], dtype=np.float64)
            std = np.asarray(self.model["feature_std"], dtype=np.float64)
            z = (values - mean) / np.where(std < 1e-9, 1.0, std)
            feedback = np.asarray(
                [
                    self._regression_value("lateral_feedback_x_mm", z),
                    self._regression_value("lateral_feedback_y_mm", z),
                ],
                dtype=np.float64,
            )
            feedback_norm = float(np.linalg.norm(feedback))
            if feedback_norm >= float(model_cfg.get("lateral_feedback_vector_threshold_mm", 0.01)):
                gain = float(model_cfg.get("lateral_feedback_vector_gain", 1.0))
                max_delta = float(model_cfg.get("lateral_feedback_vector_max_mm", 0.12))
                action[0:2] += np.clip(gain * feedback, -max_delta, max_delta)
        if bool(model_cfg.get("use_mujoco_feedback_to_policy", False)):
            mujoco = info.get("mujoco_metrics", {}) or {}
            lateral = np.asarray(
                [
                    (float(mujoco.get("tip_y_m", 0.0)) - float(mujoco.get("port_y_m", 0.0))) * 1000.0,
                    (float(mujoco.get("tip_z_m", 0.0)) - float(mujoco.get("port_z_m", 0.0))) * 1000.0,
                ],
                dtype=np.float64,
            )
            if float(np.linalg.norm(lateral)) >= float(model_cfg.get("mujoco_feedback_min_lateral_mm", 0.02)):
                delta = -float(model_cfg.get("mujoco_feedback_gain", 0.8)) * lateral
                delta = np.clip(
                    delta,
                    -float(model_cfg.get("mujoco_feedback_max_mm", 0.12)),
                    float(model_cfg.get("mujoco_feedback_max_mm", 0.12)),
                )
                action[0:2] += delta
        action[0:2] = np.clip(action[0:2], -max_residual, max_residual)
        action[2] = np.clip(action[2], -max_residual, max_residual)
        return action


def make_policy(
    name: str,
    low: np.ndarray,
    high: np.ndarray,
    *,
    random_scale: float = 1.0,
    config: dict[str, Any] | None = None,
) -> ResidualPolicy:
    if name == "zero":
        return ZeroResidualPolicy()
    if name == "random":
        return RandomResidualPolicy(low, high, scale=random_scale)
    if name == "tremor":
        return TremorAdvancePolicy(low, high)
    if name == "pid":
        return PIDResidualPolicy(low, high)
    if name == "damped_ik":
        return DampedIKResidualPolicy(low, high)
    if name == "aux_trigger_depth":
        return AuxiliaryTriggerDepthPolicy(low, high, config)
    raise ValueError(f"Unknown residual policy: {name}")
