from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import load_config, resolve_project_path
from .controllers import (
    ACTION_DIM,
    INSERT_INDEX,
    ROTATION_SLICE,
    TRANSLATION_SLICE,
    GeometricAlignmentController,
    zero_action,
)
from .geometry import clip_norm, weighted_precision_error


def _deep_update(base: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in values.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


@dataclass
class SingleArmState:
    lateral_xy_mm: np.ndarray
    depth_error_mm: float
    axis_error_xy_deg: np.ndarray
    insertion_depth_mm: float
    previous_action: np.ndarray
    previous_observed_error: np.ndarray
    joint_angles_deg: np.ndarray
    joint_velocities_deg_s: np.ndarray
    task_space_velocity: np.ndarray


class SingleArmPrecisionEnv:
    """Task-space single-arm trocar alignment and insertion environment.

    The environment models the part the advisor emphasized: after the
    eye-in-hand camera has found the trocar/port, the arm should align and
    advance with very small bounded corrections. The geometric controller is
    the nominal one-step command; RL supplies only a residual micro-motion.
    """

    action_dim = ACTION_DIM
    base_observation_dim = 47
    observation_dim = 47

    def __init__(self, config: dict[str, Any] | None = None, seed: int | None = None):
        self.cfg = self._apply_curriculum(config or load_config())
        self.task_cfg = self.cfg["task"]
        self.episode_cfg = self.cfg["episode"]
        self.reward_cfg = self.cfg["reward"]
        self.perception_cfg = self.cfg.get("perception", {"enabled": False})
        self.randomization_cfg = self.cfg.get("randomization", {"enabled": False})
        self.dynamics_cfg = self.cfg["dynamics"]
        self.safety_cfg = self.cfg["safety"]
        self.observation_cfg = self.cfg.get("observation", {"normalize": False})
        self.contact_observation_cfg = self.observation_cfg.get("contact_aware", {"enabled": False})
        self.contact_observation_features = list(
            self.contact_observation_cfg.get(
                "features",
                [
                    "current_contact_trigger_probability",
                    "current_contact_trigger_above_threshold",
                    "previous_contact_risk_probability",
                    "previous_contact_risk_above_threshold",
                    "previous_contact_helper_depth_step_mm",
                    "previous_contact_helper_feedback_norm_mm",
                    "previous_contact_helper_depth_active",
                    "previous_contact_helper_feedback_active",
                    "previous_contact_helper_depth_mismatch_mm",
                    "previous_insert_trigger_active",
                    "previous_insert_trigger_safe",
                    "previous_insert_trigger_blocked",
                    "previous_insert_trigger_missed",
                    "previous_advance_blocked",
                    "previous_phase_insert",
                ],
            )
        )
        self.observation_dim = self.base_observation_dim + (
            len(self.contact_observation_features) if bool(self.contact_observation_cfg.get("enabled", False)) else 0
        )
        self.joint_model_cfg = self.cfg.get("joint_model", {"enabled": False})
        self.action_space_cfg = self.cfg.get("action_space", {"mode": "task_residual"})
        self.phase_action_gate_cfg = self.cfg.get("phase_action_gate", {"enabled": False})
        self.contact_risk_model_cfg = self.cfg.get("contact_risk_model", {"enabled": False})
        self.contact_risk_model = self._load_contact_risk_model()
        self.contact_risk_models_cfg = self.cfg.get("contact_risk_models", {})
        self.contact_risk_models = self._load_contact_risk_models()
        self.contact_trigger_model_cfg = self.cfg.get("contact_trigger_model", {"enabled": False})
        self.contact_trigger_model = self._load_contact_trigger_model()
        self.contact_helper_model_cfg = self.cfg.get("contact_helper_model", {"enabled": False})
        self.contact_helper_model = self._load_contact_helper_model()
        self.controller = GeometricAlignmentController(self.cfg["controller"], self.safety_cfg)
        self.rng = np.random.default_rng(seed)
        self.state: SingleArmState | None = None
        self.steps = 0
        self.done = False
        self.current_observation: dict[str, Any] | None = None
        self.previous_action_info: dict[str, Any] = self._default_previous_action_info()

    def _default_previous_action_info(self) -> dict[str, Any]:
        return {
            "contact_risk_probability": 0.0,
            "contact_risk_above_threshold": False,
            "contact_helper_depth_step_mm": 0.0,
            "contact_helper_feedback_norm_mm": 0.0,
            "contact_helper_depth_active": False,
            "contact_helper_feedback_active": False,
            "contact_helper_depth_mismatch_mm": 0.0,
            "insert_trigger_active": False,
            "insert_trigger_safe": False,
            "insert_trigger_blocked": False,
            "insert_trigger_missed": False,
            "advance_blocked": False,
            "phase_action_gate_mode": "off",
            "requested_insert_step_mm": 0.0,
            "final_insert_step_mm": 0.0,
            "insert_cap_margin_mm": 0.0,
        }

    def _load_contact_risk_model(self) -> dict[str, Any] | None:
        if not bool(self.contact_risk_model_cfg.get("enabled", False)):
            return None
        model_path = str(self.contact_risk_model_cfg.get("path", "")).strip()
        if not model_path:
            raise ValueError("contact_risk_model.enabled is true but no model path was provided.")
        path = resolve_project_path(model_path)
        with path.open("r", encoding="utf-8") as stream:
            model = json.load(stream)
        if str(model.get("model_type", "")) != "standardized_logistic_contact_risk":
            raise ValueError(f"Unsupported contact risk model type: {model.get('model_type')}")
        return model

    def _load_contact_risk_models(self) -> dict[str, dict[str, Any]]:
        if not isinstance(self.contact_risk_models_cfg, dict):
            return {}
        loaded: dict[str, dict[str, Any]] = {}
        for name, cfg in self.contact_risk_models_cfg.items():
            if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
                continue
            model_path = str(cfg.get("path", "")).strip()
            if not model_path:
                raise ValueError(f"contact_risk_models.{name}.enabled is true but no model path was provided.")
            path = resolve_project_path(model_path)
            with path.open("r", encoding="utf-8") as stream:
                model = json.load(stream)
            if str(model.get("model_type", "")) != "standardized_logistic_contact_risk":
                raise ValueError(f"Unsupported named contact risk model type for {name}: {model.get('model_type')}")
            loaded[str(name)] = {"config": dict(cfg), "model": model}
        return loaded

    def _load_contact_trigger_model(self) -> dict[str, Any] | None:
        if not bool(self.contact_trigger_model_cfg.get("enabled", False)):
            return None
        model_path = str(self.contact_trigger_model_cfg.get("path", "")).strip()
        if not model_path:
            raise ValueError("contact_trigger_model.enabled is true but no model path was provided.")
        path = resolve_project_path(model_path)
        with path.open("r", encoding="utf-8") as stream:
            model = json.load(stream)
        if str(model.get("model_type", "")) != "standardized_logistic_trigger":
            raise ValueError(f"Unsupported contact trigger model type: {model.get('model_type')}")
        return model

    def _load_contact_helper_model(self) -> dict[str, Any] | None:
        if not bool(self.contact_helper_model_cfg.get("enabled", False)):
            return None
        model_path = str(self.contact_helper_model_cfg.get("path", "")).strip()
        if not model_path:
            raise ValueError("contact_helper_model.enabled is true but no model path was provided.")
        path = resolve_project_path(model_path)
        with path.open("r", encoding="utf-8") as stream:
            model = json.load(stream)
        if str(model.get("model_type", "")) != "standardized_ridge_contact_helper_auxiliary":
            raise ValueError(f"Unsupported contact helper model type: {model.get('model_type')}")
        return model

    def _apply_curriculum(self, config: dict[str, Any]) -> dict[str, Any]:
        cfg = copy.deepcopy(config)
        curriculum = cfg.get("curriculum", {})
        if not bool(curriculum.get("enabled", False)):
            return cfg
        stage = str(curriculum.get("stage", ""))
        stage_cfg = curriculum.get("stages", {}).get(stage)
        if not isinstance(stage_cfg, dict):
            raise ValueError(f"Unknown curriculum stage: {stage}")
        for section, values in stage_cfg.items():
            if isinstance(values, dict):
                cfg[section] = _deep_update(cfg.get(section, {}), values)
        cfg.setdefault("curriculum", {})["active_stage"] = stage
        return cfg

    def residual_action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        mode = str(self.action_space_cfg.get("mode", "task_residual"))
        if mode == "joint_residual_proxy":
            joint_step = float(self.action_space_cfg.get("max_joint_residual_step_deg", 1.0))
            high = np.full(ACTION_DIM, joint_step, dtype=np.float32)
            return -high, high
        if mode == "hybrid_insert_trigger":
            high = np.zeros(ACTION_DIM, dtype=np.float32)
            high[TRANSLATION_SLICE] = float(self.safety_cfg["max_residual_translation_step_mm"])
            high[ROTATION_SLICE] = float(self.safety_cfg["max_residual_rotation_step_deg"])
            high[INSERT_INDEX] = float(self.action_space_cfg.get("trigger_bound", 1.0))
            return -high, high
        high = np.zeros(ACTION_DIM, dtype=np.float32)
        high[TRANSLATION_SLICE] = float(self.safety_cfg["max_residual_translation_step_mm"])
        high[ROTATION_SLICE] = float(self.safety_cfg["max_residual_rotation_step_deg"])
        high[INSERT_INDEX] = float(self.safety_cfg["max_residual_insert_step_mm"])
        return -high, high

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        initial = self.cfg["initial_state"]
        lateral_xy = np.asarray(initial["lateral_xy_mm"], dtype=np.float64)
        depth_error = float(initial["depth_error_mm"])
        axis_xy = np.asarray(initial["axis_error_xy_deg"], dtype=np.float64)

        if bool(self.randomization_cfg.get("enabled", False)):
            lateral_xy = self.rng.uniform(
                -float(self.randomization_cfg["lateral_xy_mm"]),
                float(self.randomization_cfg["lateral_xy_mm"]),
                size=2,
            )
            depth_error = float(
                self.rng.uniform(
                    -float(self.randomization_cfg["depth_error_mm"]),
                    float(self.randomization_cfg["depth_error_mm"]),
                )
            )
            axis_xy = self.rng.uniform(
                -float(self.randomization_cfg["axis_error_deg"]),
                float(self.randomization_cfg["axis_error_deg"]),
                size=2,
            )

        initial_sampling = self.cfg.get("initial_state_sampling", {}) or {}
        if bool(initial_sampling.get("enabled", False)):
            # The clinical workflow starts outside the trocar plane. Keep this
            # curriculum contract separate from runtime safety logic so it
            # cannot masquerade as a contact-completion helper.
            lateral_bounds = np.asarray(
                initial_sampling.get("lateral_xy_bounds_mm", [-3.0, 3.0]),
                dtype=np.float64,
            )
            axis_bounds = np.asarray(
                initial_sampling.get("axis_error_bounds_deg", [-12.0, 12.0]),
                dtype=np.float64,
            )
            depth_bounds = np.asarray(
                initial_sampling.get("depth_error_bounds_mm", [0.6, 3.0]),
                dtype=np.float64,
            )
            insertion_bounds = np.asarray(
                initial_sampling.get("insertion_depth_bounds_mm", [0.0, 0.0]),
                dtype=np.float64,
            )
            if (
                lateral_bounds.shape != (2,)
                or axis_bounds.shape != (2,)
                or depth_bounds.shape != (2,)
                or insertion_bounds.shape != (2,)
                or depth_bounds[0] <= 0.0
                or insertion_bounds[0] < 0.0
                or insertion_bounds[0] > insertion_bounds[1]
            ):
                raise ValueError("Invalid initial_state_sampling bounds.")
            lateral_xy = self.rng.uniform(lateral_bounds[0], lateral_bounds[1], size=2)
            axis_xy = self.rng.uniform(axis_bounds[0], axis_bounds[1], size=2)
            depth_error = float(self.rng.uniform(depth_bounds[0], depth_bounds[1]))
            initial_insertion_depth = float(
                self.rng.uniform(insertion_bounds[0], insertion_bounds[1])
            )
        else:
            initial_insertion_depth = float(initial.get("insertion_depth_mm", 0.0))

        self.state = SingleArmState(
            lateral_xy_mm=lateral_xy,
            depth_error_mm=depth_error,
            axis_error_xy_deg=axis_xy,
            insertion_depth_mm=initial_insertion_depth,
            previous_action=zero_action(),
            previous_observed_error=np.zeros(5, dtype=np.float64),
            joint_angles_deg=self._initial_joint_angles(),
            joint_velocities_deg_s=np.zeros(6, dtype=np.float64),
            task_space_velocity=zero_action(),
        )
        self.steps = 0
        self.done = False
        self.previous_action_info = self._default_previous_action_info()
        self.current_observation = self._observe()
        self.state.previous_observed_error = self._observed_error_vector(self.current_observation)
        info = {
            "metrics": self.metrics(),
            "observation_metrics": self.current_observation,
            "raw_observation_vector": self.raw_observation_vector().tolist(),
        }
        return self.observation_vector(), info

    def metrics(self) -> dict[str, Any]:
        if self.state is None:
            raise RuntimeError("Call reset before metrics.")
        lateral = float(np.linalg.norm(self.state.lateral_xy_mm))
        depth_abs = abs(float(self.state.depth_error_mm))
        axis_error = float(np.linalg.norm(self.state.axis_error_xy_deg))
        remaining = max(0.0, float(self.task_cfg["target_insert_depth_mm"]) - float(self.state.insertion_depth_mm))
        precision_error = weighted_precision_error(
            lateral,
            self.state.depth_error_mm,
            axis_error,
            float(self.task_cfg["target_standoff_mm"]),
        )
        joint_margins = self._joint_limit_margins()
        joint_velocity_abs = np.abs(self.state.joint_velocities_deg_s)
        return {
            "lateral_xy_mm": self.state.lateral_xy_mm.tolist(),
            "depth_error_mm": float(self.state.depth_error_mm),
            "axis_error_xy_deg": self.state.axis_error_xy_deg.tolist(),
            "insertion_depth_mm": float(self.state.insertion_depth_mm),
            "remaining_insert_depth_mm": remaining,
            "lateral_error_mm": lateral,
            "depth_abs_error_mm": depth_abs,
            "axis_error_deg": axis_error,
            "weighted_precision_error_mm": precision_error,
            "joint_angles_deg": self.state.joint_angles_deg.tolist(),
            "joint_velocities_deg_s": self.state.joint_velocities_deg_s.tolist(),
            "joint_limit_margins_deg": joint_margins.tolist(),
            "min_joint_limit_margin_deg": float(np.min(joint_margins)),
            "mean_abs_joint_velocity_deg_s": float(np.mean(joint_velocity_abs)),
            "task_space_velocity": self.state.task_space_velocity.tolist(),
            "step": int(self.steps),
            "time_s": float(self.steps) * float(self.episode_cfg["step_time_s"]),
            "curriculum_stage": self.cfg.get("curriculum", {}).get("active_stage", "none"),
        }

    def observation_vector(self) -> np.ndarray:
        return self._observation_vector(normalize=bool(self.observation_cfg.get("normalize", False)))

    def raw_observation_vector(self) -> np.ndarray:
        return self._observation_vector(normalize=False)

    def _observation_vector(self, *, normalize: bool) -> np.ndarray:
        if self.state is None or self.current_observation is None:
            raise RuntimeError("Call reset before observation_vector.")
        obs = self.current_observation
        current_error = self._observed_error_vector(obs)
        error_delta = current_error - self.state.previous_observed_error
        joint_margins = self._joint_limit_margins()
        base_raw = [
            *obs["observed_lateral_xy_mm"],
            obs["observed_depth_error_mm"],
            *obs["observed_axis_error_xy_deg"],
            self.state.insertion_depth_mm,
            obs["remaining_insert_depth_mm"],
            obs["observed_lateral_error_mm"],
            abs(obs["observed_depth_error_mm"]),
            obs["observed_axis_error_deg"],
            obs["confidence"],
            1.0 if obs["quality_gate_pass"] else 0.0,
            *error_delta.tolist(),
            *self.state.previous_action.tolist(),
            *self.state.joint_angles_deg.tolist(),
            *self.state.joint_velocities_deg_s.tolist(),
            *joint_margins.tolist(),
            *self.state.task_space_velocity.tolist(),
        ]
        if bool(self.contact_observation_cfg.get("enabled", False)):
            base_raw.extend(self._contact_observation_raw_features(obs))
        raw = np.asarray(
            base_raw,
            dtype=np.float32,
        )
        if not normalize:
            return raw
        return self._normalize_observation(raw)

    def _normalize_observation(self, raw: np.ndarray) -> np.ndarray:
        scales = self.observation_cfg.get("scales", {})
        lateral = float(scales.get("lateral_mm", self.task_cfg["hard_lateral_mm"]))
        depth = float(scales.get("depth_mm", self.task_cfg["hard_depth_mm"]))
        axis = float(scales.get("axis_deg", self.task_cfg["hard_axis_deg"]))
        insertion = float(scales.get("insertion_depth_mm", self.task_cfg["target_insert_depth_mm"]))
        trans_action = float(scales.get("translation_action_mm", 1.0))
        rot_action = float(scales.get("rotation_action_deg", 1.0))
        insert_action = float(scales.get("insert_action_mm", 1.0))
        joint_angle = float(scales.get("joint_angle_deg", 180.0))
        joint_velocity = float(scales.get("joint_velocity_deg_s", 120.0))
        joint_margin = float(scales.get("joint_limit_margin_deg", 180.0))
        vel_trans = float(scales.get("task_velocity_translation_mm_s", 15.0))
        vel_rot = float(scales.get("task_velocity_rotation_deg_s", 24.0))
        vel_insert = float(scales.get("task_velocity_insert_mm_s", 5.0))

        out: list[float] = []
        out.extend((raw[0:2] / lateral).tolist())
        out.append(float(raw[2] / depth))
        out.extend((raw[3:5] / axis).tolist())
        out.append(float(raw[5] / insertion))
        out.append(float(raw[6] / insertion))
        out.append(float(raw[7] / lateral))
        out.append(float(raw[8] / depth))
        out.append(float(raw[9] / axis))
        out.append(float(raw[10]))
        out.append(float(raw[11]))
        out.extend((raw[12:14] / lateral).tolist())
        out.append(float(raw[14] / depth))
        out.extend((raw[15:17] / axis).tolist())
        out.extend((raw[17:20] / trans_action).tolist())
        out.extend((raw[20:22] / rot_action).tolist())
        out.append(float(raw[22] / insert_action))
        out.extend((raw[23:29] / joint_angle).tolist())
        out.extend((raw[29:35] / joint_velocity).tolist())
        out.extend((raw[35:41] / joint_margin).tolist())
        out.extend((raw[41:44] / vel_trans).tolist())
        out.extend((raw[44:46] / vel_rot).tolist())
        out.append(float(raw[46] / vel_insert))
        if len(raw) > self.base_observation_dim:
            out.extend(self._normalize_contact_observation(raw[self.base_observation_dim :]).tolist())
        return np.clip(np.asarray(out, dtype=np.float32), -1.0, 1.0)

    def _contact_observation_raw_features(self, observed: dict[str, Any]) -> list[float]:
        current_trigger = self._contact_trigger_model_info(observed)
        previous = self.previous_action_info
        current_error = self._observed_error_vector(observed)
        previous_error = self.state.previous_observed_error if self.state is not None else np.zeros(5, dtype=np.float64)
        error_delta = current_error - previous_error
        previous_action = self.state.previous_action if self.state is not None else zero_action()
        target_depth = max(float(self.task_cfg["target_insert_depth_mm"]), 1e-9)
        insertion_depth = float(self.state.insertion_depth_mm) if self.state is not None else 0.0
        remaining_depth = max(0.0, target_depth - insertion_depth)
        phase = self._current_contact_phase(observed)
        values = []
        for name in self.contact_observation_features:
            if name == "current_contact_trigger_probability":
                value = float(current_trigger.get("contact_trigger_probability", 0.0))
            elif name == "current_contact_trigger_above_threshold":
                value = 1.0 if bool(current_trigger.get("contact_trigger_above_threshold", False)) else 0.0
            elif name == "previous_contact_risk_probability":
                value = float(previous.get("contact_risk_probability", 0.0))
            elif name == "previous_contact_risk_above_threshold":
                value = 1.0 if bool(previous.get("contact_risk_above_threshold", False)) else 0.0
            elif name == "previous_contact_helper_depth_step_mm":
                value = float(previous.get("contact_helper_depth_step_mm", 0.0))
            elif name == "previous_contact_helper_feedback_norm_mm":
                value = float(previous.get("contact_helper_feedback_norm_mm", 0.0))
            elif name == "previous_contact_helper_depth_active":
                value = 1.0 if bool(previous.get("contact_helper_depth_active", False)) else 0.0
            elif name == "previous_contact_helper_feedback_active":
                value = 1.0 if bool(previous.get("contact_helper_feedback_active", False)) else 0.0
            elif name == "previous_contact_helper_depth_mismatch_mm":
                value = float(previous.get("contact_helper_depth_mismatch_mm", 0.0))
            elif name == "previous_insert_trigger_active":
                value = 1.0 if bool(previous.get("insert_trigger_active", False)) else 0.0
            elif name == "previous_insert_trigger_safe":
                value = 1.0 if bool(previous.get("insert_trigger_safe", False)) else 0.0
            elif name == "previous_insert_trigger_blocked":
                value = 1.0 if bool(previous.get("insert_trigger_blocked", False)) else 0.0
            elif name == "previous_insert_trigger_missed":
                value = 1.0 if bool(previous.get("insert_trigger_missed", False)) else 0.0
            elif name == "previous_advance_blocked":
                value = 1.0 if bool(previous.get("advance_blocked", False)) else 0.0
            elif name == "previous_phase_insert":
                value = 1.0 if str(previous.get("phase_action_gate_mode", "")) == "insert" else 0.0
            elif name == "current_contact_frame_lateral_x_mm":
                value = float(current_error[0])
            elif name == "current_contact_frame_lateral_y_mm":
                value = float(current_error[1])
            elif name == "current_contact_frame_lateral_norm_mm":
                value = float(np.linalg.norm(current_error[0:2]))
            elif name == "current_lateral_drift_x_mm":
                value = float(error_delta[0])
            elif name == "current_lateral_drift_y_mm":
                value = float(error_delta[1])
            elif name == "current_lateral_drift_norm_mm":
                value = float(np.linalg.norm(error_delta[0:2]))
            elif name == "previous_lateral_action_x_mm":
                value = float(previous_action[0])
            elif name == "previous_lateral_action_y_mm":
                value = float(previous_action[1])
            elif name == "previous_lateral_action_norm_mm":
                value = float(np.linalg.norm(previous_action[0:2]))
            elif name == "previous_insert_step_mm":
                value = float(previous_action[INSERT_INDEX])
            elif name == "previous_requested_insert_step_mm":
                value = float(previous.get("requested_insert_step_mm", 0.0))
            elif name == "previous_insert_cap_margin_mm":
                value = float(previous.get("insert_cap_margin_mm", 0.0))
            elif name == "current_insertion_fraction":
                value = insertion_depth / target_depth
            elif name == "current_remaining_fraction":
                value = remaining_depth / target_depth
            elif name == "current_remaining_insert_depth_mm":
                value = remaining_depth
            elif name == "current_over_insert_margin_mm":
                value = max(0.0, insertion_depth - target_depth)
            elif name == "current_phase_pre_insert":
                value = 1.0 if phase == "pre_insert" else 0.0
            elif name == "current_phase_insert_contact":
                value = 1.0 if phase == "insert_contact" else 0.0
            elif name == "current_phase_near_target":
                value = 1.0 if phase == "near_target" else 0.0
            elif name == "current_phase_insert":
                value = 1.0 if phase in {"insert_contact", "near_target"} else 0.0
            else:
                value = 0.0
            values.append(float(value))
        return values

    def _normalize_contact_observation(self, raw: np.ndarray) -> np.ndarray:
        scales = self.observation_cfg.get("scales", {})
        contact_depth = float(scales.get("contact_helper_depth_step_mm", self.safety_cfg.get("max_base_insert_step_mm", 0.1)))
        contact_feedback = float(scales.get("contact_helper_feedback_mm", self.safety_cfg.get("allow_insert_lateral_mm", 0.25)))
        lateral_drift = float(scales.get("contact_lateral_drift_mm", contact_feedback))
        lateral_action = float(scales.get("contact_lateral_action_mm", scales.get("translation_action_mm", 1.0)))
        insertion_depth = float(scales.get("insertion_depth_mm", self.task_cfg["target_insert_depth_mm"]))
        insert_action = float(scales.get("insert_action_mm", 1.0))
        values = []
        for name, value in zip(self.contact_observation_features, raw):
            scale = 1.0
            if name in {"previous_contact_helper_depth_step_mm", "previous_contact_helper_depth_mismatch_mm"}:
                scale = max(contact_depth, 1e-9)
            elif name == "previous_contact_helper_feedback_norm_mm":
                scale = max(contact_feedback, 1e-9)
            elif name in {
                "current_contact_frame_lateral_x_mm",
                "current_contact_frame_lateral_y_mm",
                "current_contact_frame_lateral_norm_mm",
            }:
                scale = max(contact_feedback, 1e-9)
            elif name in {"current_lateral_drift_x_mm", "current_lateral_drift_y_mm", "current_lateral_drift_norm_mm"}:
                scale = max(lateral_drift, 1e-9)
            elif name in {"previous_lateral_action_x_mm", "previous_lateral_action_y_mm", "previous_lateral_action_norm_mm"}:
                scale = max(lateral_action, 1e-9)
            elif name in {"previous_insert_step_mm", "previous_requested_insert_step_mm", "previous_insert_cap_margin_mm"}:
                scale = max(insert_action, 1e-9)
            elif name in {"current_remaining_insert_depth_mm", "current_over_insert_margin_mm"}:
                scale = max(insertion_depth, 1e-9)
            values.append(float(value) / scale)
        return np.asarray(values, dtype=np.float32)

    def _current_contact_phase(self, observed: dict[str, Any]) -> str:
        if self.state is None:
            return "pre_insert"
        phase_cfg = self.cfg.get("auxiliary_constrained_sac", {}).get("phase_conditional_objective", {})
        target = max(float(self.task_cfg["target_insert_depth_mm"]), 1e-9)
        insertion = float(self.state.insertion_depth_mm)
        remaining = max(0.0, target - insertion)
        started_eps = float(phase_cfg.get("insert_started_depth_eps_mm", 1e-6))
        near_remaining = float(phase_cfg.get("near_target_remaining_mm", 0.12))
        near_fraction = float(phase_cfg.get("near_target_fraction", 0.97))
        previous = self.previous_action_info
        insert_started = bool(
            insertion > started_eps
            or float(previous.get("final_insert_step_mm", 0.0)) > started_eps
            or bool(previous.get("insert_trigger_active", False))
        )
        near_target = bool(remaining <= near_remaining or insertion / target >= near_fraction)
        if not insert_started:
            return "pre_insert"
        if near_target:
            return "near_target"
        return "insert_contact"

    def step(self, residual_action: np.ndarray | list[float]) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        if self.state is None or self.current_observation is None:
            raise RuntimeError("Call reset before step.")
        if self.done:
            raise RuntimeError("Episode is done. Call reset before stepping again.")

        before = self.metrics()
        observed = self.current_observation
        base_action = self.controller.action(observed)
        raw_residual = np.asarray(residual_action, dtype=np.float64).copy()
        clipped_residual_command = self._clip_residual(raw_residual)
        residual_gate_scale = self._residual_gate_scale(observed)
        effective_residual_command = clipped_residual_command * residual_gate_scale
        if str(self.action_space_cfg.get("mode", "task_residual")) == "hybrid_insert_trigger":
            trigger_cfg = self.action_space_cfg.get("hybrid_insert_trigger", {})
            trigger_scale = float(trigger_cfg.get("trigger_gate_scale", 1.0))
            effective_residual_command[INSERT_INDEX] = clipped_residual_command[INSERT_INDEX] * trigger_scale

        refused_by_quality = not bool(observed.get("quality_gate_pass", True))
        if refused_by_quality:
            effective_residual_command[:] = 0.0
            residual_gate_scale = 0.0
        residual = self._residual_command_to_task(effective_residual_command)
        final_action = self._combine_and_project(base_action, residual, observed, effective_residual_command)
        previous_action = self.state.previous_action.copy()
        previous_observed_error = self._observed_error_vector(observed)

        self._apply_action(final_action)
        self.steps += 1
        after = self.metrics()

        failure_reason = self._failure_reason(after)
        success = self._success(after)
        timeout = self.steps >= int(self.episode_cfg["max_steps"])
        self.done = success or failure_reason is not None or timeout

        action_info = self._action_info(
            base_action,
            raw_residual,
            clipped_residual_command,
            effective_residual_command,
            residual,
            final_action,
            previous_action,
            observed,
            residual_gate_scale,
        )
        action_info.update(self._contact_risk_model_info(before, after, final_action, action_info))
        action_info.update(self._contact_helper_model_info(before, after, final_action, action_info))
        reward, reward_components = self._reward(
            before,
            after,
            final_action,
            previous_action,
            action_info,
            success,
            failure_reason,
            timeout,
        )
        self.state.previous_action = final_action
        self.state.previous_observed_error = previous_observed_error
        self.previous_action_info = dict(action_info)
        self.current_observation = self._observe()

        info = {
            "before_metrics": before,
            "metrics": after,
            "observation_metrics": self.current_observation,
            "raw_observation_vector": self.raw_observation_vector().tolist(),
            "base_action": base_action.tolist(),
            "raw_residual_action": raw_residual.tolist(),
            "clipped_residual_action": clipped_residual_command.tolist(),
            "effective_residual_command": effective_residual_command.tolist(),
            "effective_residual_action": residual.tolist(),
            "final_action": final_action.tolist(),
            "action_info": action_info,
            "reward_components": reward_components,
            "success": success,
            "failure_reason": failure_reason,
            "timeout": timeout,
            "refused_by_quality": refused_by_quality,
        }
        return self.observation_vector(), float(reward), self.done, info

    def _observe(self) -> dict[str, Any]:
        true = self.metrics()
        if not bool(self.perception_cfg.get("enabled", False)):
            return self._observation_from_values(
                np.asarray(true["lateral_xy_mm"], dtype=np.float64),
                float(true["depth_error_mm"]),
                np.asarray(true["axis_error_xy_deg"], dtype=np.float64),
                confidence=1.0,
                dropout=False,
            )

        dropout = bool(self.rng.random() < float(self.perception_cfg.get("dropout_probability", 0.0)))
        xy = np.asarray(true["lateral_xy_mm"], dtype=np.float64) + self.rng.normal(
            0.0,
            float(self.perception_cfg["center_noise_std_mm"]),
            size=2,
        )
        depth = float(true["depth_error_mm"]) + float(
            self.rng.normal(0.0, float(self.perception_cfg["depth_noise_std_mm"]))
        )
        axis_xy = np.asarray(true["axis_error_xy_deg"], dtype=np.float64) + self.rng.normal(
            0.0,
            float(self.perception_cfg["axis_noise_std_deg"]),
            size=2,
        )
        # The task starts after the trocar/port has been found. Confidence is
        # therefore a detector/ROI quality proxy, not an alignment-quality
        # proxy; large correctable pose errors must not permanently freeze the
        # controller.
        error_scale = (
            0.02 * float(np.linalg.norm(xy))
            + 0.015 * abs(depth)
            + 0.008 * float(np.linalg.norm(axis_xy))
        )
        confidence = float(np.clip(0.96 - error_scale + self.rng.normal(0.0, float(self.perception_cfg["confidence_noise_std"])), 0.0, 1.0))
        return self._observation_from_values(xy, depth, axis_xy, confidence=confidence, dropout=dropout)

    def _observation_from_values(
        self,
        xy: np.ndarray,
        depth: float,
        axis_xy: np.ndarray,
        *,
        confidence: float,
        dropout: bool,
    ) -> dict[str, Any]:
        confidence = 0.0 if dropout else float(confidence)
        gate = (not dropout) and confidence >= float(self.perception_cfg.get("min_confidence", 0.0))
        remaining = max(0.0, float(self.task_cfg["target_insert_depth_mm"]) - float(self.state.insertion_depth_mm))
        return {
            "observed_lateral_xy_mm": xy.astype(float).tolist(),
            "observed_depth_error_mm": float(depth),
            "observed_axis_error_xy_deg": axis_xy.astype(float).tolist(),
            "observed_lateral_error_mm": float(np.linalg.norm(xy)),
            "observed_axis_error_deg": float(np.linalg.norm(axis_xy)),
            # Commanded insertion progress is available from the robot task
            # plan/encoder estimate. It is intentionally separate from any
            # MuJoCo contact truth and lets phase logic avoid retracting after
            # a safe insertion has begun.
            "insertion_depth_mm": float(self.state.insertion_depth_mm),
            "target_insert_depth_mm": float(self.task_cfg["target_insert_depth_mm"]),
            "remaining_insert_depth_mm": remaining,
            "confidence": confidence,
            "quality_gate_pass": bool(gate),
            "dropout": bool(dropout),
        }

    def _observed_error_vector(self, observed: dict[str, Any]) -> np.ndarray:
        return np.asarray(
            [
                *observed["observed_lateral_xy_mm"],
                observed["observed_depth_error_mm"],
                *observed["observed_axis_error_xy_deg"],
            ],
            dtype=np.float64,
        )

    def _clip_residual(self, residual: np.ndarray) -> np.ndarray:
        low, high = self.residual_action_bounds()
        return np.clip(residual, low.astype(np.float64), high.astype(np.float64))

    def _residual_command_to_task(self, residual_command: np.ndarray) -> np.ndarray:
        mode = str(self.action_space_cfg.get("mode", "task_residual"))
        if mode == "task_residual":
            return self._clip_task_residual(np.asarray(residual_command, dtype=np.float64))
        if mode == "hybrid_insert_trigger":
            residual = np.asarray(residual_command, dtype=np.float64).copy()
            residual[INSERT_INDEX] = 0.0
            return self._clip_task_residual(residual)
        if mode != "joint_residual_proxy":
            raise ValueError(f"Unknown action_space.mode: {mode}")

        command = np.asarray(residual_command, dtype=np.float64)
        if "joint_to_task_gain" in self.action_space_cfg:
            matrix = np.asarray(self.action_space_cfg["joint_to_task_gain"], dtype=np.float64)
        else:
            task_to_joint = np.asarray(self.joint_model_cfg.get("task_to_joint_gain", np.eye(6)), dtype=np.float64)
            if task_to_joint.shape != (6, 6):
                raise ValueError("joint_model.task_to_joint_gain must be a 6x6 matrix")
            matrix = np.linalg.pinv(task_to_joint)
        if matrix.shape != (6, 6):
            raise ValueError("action_space.joint_to_task_gain must be a 6x6 matrix")
        return self._clip_task_residual(matrix @ command)

    def _clip_task_residual(self, residual: np.ndarray) -> np.ndarray:
        high = np.zeros(ACTION_DIM, dtype=np.float64)
        high[TRANSLATION_SLICE] = float(self.safety_cfg["max_residual_translation_step_mm"])
        high[ROTATION_SLICE] = float(self.safety_cfg["max_residual_rotation_step_deg"])
        high[INSERT_INDEX] = float(self.safety_cfg["max_residual_insert_step_mm"])
        return np.clip(np.asarray(residual, dtype=np.float64), -high, high)

    def _residual_gate_scale(self, observed: dict[str, Any]) -> float:
        gate_cfg = self.cfg.get("residual_gate", {})
        if not bool(gate_cfg.get("enabled", False)):
            return 1.0
        lateral = float(observed["observed_lateral_error_mm"])
        depth = abs(float(observed["observed_depth_error_mm"]))
        axis = float(observed["observed_axis_error_deg"])
        thresholds = np.asarray(
            [
                max(float(gate_cfg.get("full_scale_lateral_mm", self.safety_cfg["allow_insert_lateral_mm"])), 1e-9),
                max(float(gate_cfg.get("full_scale_depth_mm", self.safety_cfg["allow_insert_depth_mm"])), 1e-9),
                max(float(gate_cfg.get("full_scale_axis_deg", self.safety_cfg["allow_insert_axis_deg"])), 1e-9),
            ],
            dtype=np.float64,
        )
        ratio = max(float(lateral / thresholds[0]), float(depth / thresholds[1]), float(axis / thresholds[2]), 1.0)
        max_scale = float(gate_cfg.get("max_scale", 1.0))
        far_scale = float(gate_cfg.get("far_scale", 0.0))
        scale = max(far_scale, max_scale / ratio)
        remaining = float(observed.get("remaining_insert_depth_mm", 0.0))
        if remaining <= float(gate_cfg.get("post_insert_remaining_mm", 0.05)):
            scale = min(scale, float(gate_cfg.get("post_insert_scale", scale)))
        return float(np.clip(scale, 0.0, 1.0))

    def _combine_and_project(
        self,
        base_action: np.ndarray,
        residual: np.ndarray,
        observed: dict[str, Any],
        residual_command: np.ndarray | None = None,
    ) -> np.ndarray:
        mode = str(self.action_space_cfg.get("mode", "task_residual"))
        action = base_action + residual
        action[TRANSLATION_SLICE] = clip_norm(action[TRANSLATION_SLICE], float(self.safety_cfg["max_base_translation_step_mm"]))
        action[ROTATION_SLICE] = clip_norm(action[ROTATION_SLICE], float(self.safety_cfg["max_base_rotation_step_deg"]))
        if mode == "hybrid_insert_trigger":
            hybrid_cfg = self.action_space_cfg.get("hybrid_insert_trigger", {})
            nominal_base_insert_enabled = bool(
                hybrid_cfg.get("nominal_base_insert_enabled", False)
            )
            nominal_base_requested = max(0.0, float(base_action[INSERT_INDEX]))
            action[INSERT_INDEX] = 0.0
            action = self._apply_phase_action_gate(action, observed)
            trigger_info = self._hybrid_insert_trigger_info(residual_command, observed)
            nominal_base_safe = bool(
                nominal_base_insert_enabled
                and nominal_base_requested > 1e-9
                and self._safe_to_insert(observed)
                and self._phase_action_gate_mode(observed) in ("insert", "off")
            )
            if nominal_base_safe:
                action[INSERT_INDEX] = min(
                    nominal_base_requested,
                    float(self.safety_cfg["max_base_insert_step_mm"]),
                    float(observed["remaining_insert_depth_mm"]),
                )
            if trigger_info["trigger_active"] and trigger_info["trigger_safe"]:
                action[INSERT_INDEX] = max(
                    float(action[INSERT_INDEX]),
                    min(
                        float(trigger_info["safe_insert_step_mm"]),
                        float(observed["remaining_insert_depth_mm"]),
                    ),
                )
            trigger_info.update(
                {
                    "nominal_base_insert_enabled": nominal_base_insert_enabled,
                    "nominal_base_insert_requested_mm": nominal_base_requested,
                    "nominal_base_insert_safe": nominal_base_safe,
                    "nominal_base_insert_active": bool(action[INSERT_INDEX] > 1e-9 and nominal_base_safe),
                }
            )
            return action
        action[INSERT_INDEX] = float(
            np.clip(
                action[INSERT_INDEX],
                0.0,
                float(self.safety_cfg["max_base_insert_step_mm"]) + float(self.safety_cfg["max_residual_insert_step_mm"]),
            )
        )
        action = self._apply_phase_action_gate(action, observed)
        if not self._safe_to_insert(observed):
            action[INSERT_INDEX] = 0.0
        action[INSERT_INDEX] = min(action[INSERT_INDEX], float(observed["remaining_insert_depth_mm"]))
        return action

    def _hybrid_insert_trigger_info(
        self,
        residual_command: np.ndarray | None,
        observed: dict[str, Any],
    ) -> dict[str, Any]:
        cfg = self.action_space_cfg.get("hybrid_insert_trigger", {})
        command = np.asarray(residual_command if residual_command is not None else zero_action(), dtype=np.float64)
        trigger_value = float(command[INSERT_INDEX])
        threshold = float(cfg.get("trigger_threshold", 0.0))
        trigger_active = trigger_value > threshold
        phase_mode = self._phase_action_gate_mode(observed)
        gate_safe = self._safe_to_insert(observed)
        phase_safe = phase_mode in ("insert", "off")
        trigger_opportunity_safe = bool(gate_safe and phase_safe)
        trigger_safe = bool(trigger_active and gate_safe and phase_safe)
        remaining = max(0.0, float(observed.get("remaining_insert_depth_mm", 0.0)))
        max_insert_step = float(
            cfg.get(
                "insert_step_mm",
                self.controller.cfg.get(
                    "insertion_step_mm", self.safety_cfg["max_base_insert_step_mm"]
                ),
            )
        )
        insert_step_mode = str(cfg.get("insert_step_mode", "fixed")).lower()
        insert_step_fraction = 1.0
        requested_insert_step = max_insert_step
        if insert_step_mode == "continuous":
            trigger_bound = max(
                float(self.action_space_cfg.get("trigger_bound", 1.0)),
                threshold + 1e-6,
            )
            insert_step_fraction = float(
                np.clip((trigger_value - threshold) / (trigger_bound - threshold), 0.0, 1.0)
            )
            insert_step_fraction = insert_step_fraction ** max(
                float(cfg.get("insert_step_power", 1.0)), 1e-6
            )
            min_insert_step = max(0.0, float(cfg.get("min_insert_step_mm", 0.0)))
            requested_insert_step = (
                min_insert_step
                + insert_step_fraction * max(0.0, max_insert_step - min_insert_step)
            )
        elif insert_step_mode != "fixed":
            raise ValueError(f"Unknown hybrid insert_step_mode: {insert_step_mode}")
        insert_step = min(
            requested_insert_step,
            float(self.safety_cfg["max_base_insert_step_mm"]) + float(self.safety_cfg["max_residual_insert_step_mm"]),
            remaining,
        )
        return {
            "trigger_value": trigger_value,
            "trigger_threshold": threshold,
            "trigger_active": bool(trigger_active),
            "trigger_safe": bool(trigger_safe),
            "trigger_blocked": bool(trigger_active and not trigger_safe),
            "trigger_missed": bool((not trigger_active) and trigger_opportunity_safe and remaining > 1e-9),
            "trigger_opportunity_safe": bool(trigger_opportunity_safe and remaining > 1e-9),
            "safe_insert_step_mm": float(insert_step),
            "insert_step_mode": insert_step_mode,
            "insert_step_fraction": float(insert_step_fraction),
            "trigger_phase_mode": phase_mode,
            "trigger_gate_safe": bool(gate_safe),
        }

    def _apply_phase_action_gate(self, action: np.ndarray, observed: dict[str, Any]) -> np.ndarray:
        if not bool(self.phase_action_gate_cfg.get("enabled", False)):
            return action
        gated = np.asarray(action, dtype=np.float64).copy()
        mode = self._phase_action_gate_mode(observed)
        if mode == "align":
            max_depth_step = float(self.phase_action_gate_cfg.get("max_align_depth_step_mm", 0.0))
            gated[2] = float(np.clip(gated[2], -max_depth_step, max_depth_step))
            gated[INSERT_INDEX] = 0.0
        elif mode == "depth":
            gated[INSERT_INDEX] = 0.0
        elif mode == "quality_refuse":
            gated[:] = 0.0
        return gated

    def _phase_action_gate_mode(self, observed: dict[str, Any]) -> str:
        if not bool(self.phase_action_gate_cfg.get("enabled", False)):
            return "off"
        if not bool(observed.get("quality_gate_pass", True)):
            return "quality_refuse"
        lateral = float(observed["observed_lateral_error_mm"])
        axis = float(observed["observed_axis_error_deg"])
        depth = abs(float(observed["observed_depth_error_mm"]))
        if (
            lateral > float(self.phase_action_gate_cfg.get("align_lateral_mm", self.safety_cfg["allow_insert_lateral_mm"]))
            or axis > float(self.phase_action_gate_cfg.get("align_axis_deg", self.safety_cfg["allow_insert_axis_deg"]))
        ):
            return "align"
        if depth > float(self.phase_action_gate_cfg.get("depth_gate_mm", self.safety_cfg["allow_insert_depth_mm"])):
            return "depth"
        return "insert"

    def _safe_to_insert(self, observed: dict[str, Any]) -> bool:
        return (
            bool(observed.get("quality_gate_pass", True))
            and float(observed["observed_lateral_error_mm"]) <= float(self.safety_cfg["allow_insert_lateral_mm"])
            and abs(float(observed["observed_depth_error_mm"])) <= float(self.safety_cfg["allow_insert_depth_mm"])
            and float(observed["observed_axis_error_deg"]) <= float(self.safety_cfg["allow_insert_axis_deg"])
        )

    def _apply_action(self, action: np.ndarray) -> None:
        if self.state is None:
            raise RuntimeError("Call reset before _apply_action.")
        dyn = self.dynamics_cfg
        step_time = float(self.episode_cfg["step_time_s"])
        self.state.task_space_velocity = np.asarray(action, dtype=np.float64) / max(step_time, 1e-9)
        self._apply_joint_proxy(action)
        translation_noise = self.rng.normal(0.0, float(dyn["translation_execution_noise_mm"]), size=3)
        rotation_noise = self.rng.normal(0.0, float(dyn["rotation_execution_noise_deg"]), size=2)
        self.state.lateral_xy_mm = self.state.lateral_xy_mm - action[:2] + translation_noise[:2]
        self.state.depth_error_mm = float(self.state.depth_error_mm - action[2] + translation_noise[2])
        self.state.axis_error_xy_deg = self.state.axis_error_xy_deg - action[3:5] + rotation_noise

        insert = max(0.0, float(action[INSERT_INDEX]))
        self.state.insertion_depth_mm = min(
            float(self.task_cfg["target_insert_depth_mm"]),
            float(self.state.insertion_depth_mm) + insert,
        )

        self.state.lateral_xy_mm += self.rng.normal(0.0, float(dyn["tremor_lateral_std_mm"]), size=2)
        self.state.depth_error_mm += float(self.rng.normal(0.0, float(dyn["tremor_depth_std_mm"])))
        self.state.axis_error_xy_deg += self.rng.normal(0.0, float(dyn["tremor_axis_std_deg"]), size=2)

        if insert > 0.0:
            lateral = float(np.linalg.norm(self.state.lateral_xy_mm))
            axis = float(np.linalg.norm(self.state.axis_error_xy_deg))
            lateral_excess = max(0.0, lateral - float(self.task_cfg["success_lateral_mm"]))
            axis_excess = max(0.0, axis - float(self.task_cfg["success_axis_deg"]))
            if lateral_excess > 0.0 and lateral > 1e-9:
                self.state.lateral_xy_mm += (
                    self.state.lateral_xy_mm
                    / lateral
                    * insert
                    * lateral_excess
                    * float(dyn["contact_lateral_gain"])
                )
            if axis_excess > 0.0 and axis > 1e-9:
                self.state.axis_error_xy_deg += (
                    self.state.axis_error_xy_deg
                    / axis
                    * insert
                    * axis_excess
                    * float(dyn["contact_axis_gain"])
                )

    def _initial_joint_angles(self) -> np.ndarray:
        values = self.joint_model_cfg.get("initial_angles_deg", [0.0] * 6)
        angles = np.asarray(values, dtype=np.float64)
        if angles.shape != (6,):
            raise ValueError("joint_model.initial_angles_deg must contain 6 values")
        return angles

    def _joint_limit_margins(self) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("Call reset before _joint_limit_margins.")
        lower = np.asarray(self.joint_model_cfg.get("lower_limits_deg", [-180.0] * 6), dtype=np.float64)
        upper = np.asarray(self.joint_model_cfg.get("upper_limits_deg", [180.0] * 6), dtype=np.float64)
        return np.minimum(self.state.joint_angles_deg - lower, upper - self.state.joint_angles_deg)

    def _apply_joint_proxy(self, action: np.ndarray) -> None:
        if self.state is None or not bool(self.joint_model_cfg.get("enabled", False)):
            return
        matrix = np.asarray(self.joint_model_cfg.get("task_to_joint_gain", np.eye(6)), dtype=np.float64)
        if matrix.shape != (6, 6):
            raise ValueError("joint_model.task_to_joint_gain must be a 6x6 matrix")
        delta = matrix @ np.asarray(action, dtype=np.float64)
        step_time = max(float(self.episode_cfg["step_time_s"]), 1e-9)
        measured_velocity = delta / step_time
        alpha = float(self.joint_model_cfg.get("velocity_filter_alpha", 1.0))
        self.state.joint_velocities_deg_s = (
            (1.0 - alpha) * self.state.joint_velocities_deg_s + alpha * measured_velocity
        )
        self.state.joint_angles_deg = self.state.joint_angles_deg + delta

    def _failure_reason(self, metrics: dict[str, Any]) -> str | None:
        if float(metrics["lateral_error_mm"]) > float(self.task_cfg["hard_lateral_mm"]):
            return "lateral_hard_limit"
        if float(metrics["depth_abs_error_mm"]) > float(self.task_cfg["hard_depth_mm"]):
            return "depth_hard_limit"
        if float(metrics["axis_error_deg"]) > float(self.task_cfg["hard_axis_deg"]):
            return "axis_hard_limit"
        if bool(self.joint_model_cfg.get("enabled", False)) and float(metrics["min_joint_limit_margin_deg"]) < float(self.joint_model_cfg.get("hard_margin_deg", 0.0)):
            return "joint_limit_margin"
        return None

    def _success(self, metrics: dict[str, Any]) -> bool:
        return (
            float(metrics["insertion_depth_mm"]) >= float(self.task_cfg["target_insert_depth_mm"]) - 1e-9
            and float(metrics["lateral_error_mm"]) <= float(self.task_cfg["success_lateral_mm"])
            and float(metrics["depth_abs_error_mm"]) <= float(self.task_cfg["success_depth_mm"])
            and float(metrics["axis_error_deg"]) <= float(self.task_cfg["success_axis_deg"])
        )

    def _action_info(
        self,
        base_action: np.ndarray,
        raw_residual: np.ndarray,
        clipped_residual_command: np.ndarray,
        effective_residual_command: np.ndarray,
        effective_residual: np.ndarray,
        final_action: np.ndarray,
        previous_action: np.ndarray,
        observed: dict[str, Any],
        residual_gate_scale: float,
    ) -> dict[str, Any]:
        mode = str(self.action_space_cfg.get("mode", "task_residual"))
        hybrid_trigger = (
            self._hybrid_insert_trigger_info(effective_residual_command, observed)
            if mode == "hybrid_insert_trigger"
            else {}
        )
        requested_insert = (
            float(hybrid_trigger.get("safe_insert_step_mm", 0.0))
            if hybrid_trigger.get("trigger_active", False)
            else float(base_action[INSERT_INDEX] + effective_residual[INSERT_INDEX])
        )
        final_insert = float(final_action[INSERT_INDEX])
        insert_cap_margin = max(0.0, requested_insert - final_insert)
        advance_blocked = requested_insert > 1e-9 and final_insert <= 1e-9 and not self._safe_to_insert(observed)
        phase_gate_mode = self._phase_action_gate_mode(observed)
        advance_block_risk = self._advance_block_risk(observed) if advance_blocked else 0.0
        return {
            "residual_action_space_mode": str(self.action_space_cfg.get("mode", "task_residual")),
            "phase_action_gate_mode": phase_gate_mode,
            "phase_action_gate_enabled": bool(self.phase_action_gate_cfg.get("enabled", False)),
            "raw_residual_norm": float(np.linalg.norm(raw_residual)),
            "clipped_residual_command_norm": float(np.linalg.norm(clipped_residual_command)),
            "effective_residual_command_norm": float(np.linalg.norm(effective_residual_command)),
            "effective_residual_norm": float(np.linalg.norm(effective_residual)),
            "residual_gate_scale": float(residual_gate_scale),
            "final_action_norm": float(np.linalg.norm(final_action)),
            "final_translation_norm_mm": float(np.linalg.norm(final_action[TRANSLATION_SLICE])),
            "final_rotation_norm_deg": float(np.linalg.norm(final_action[ROTATION_SLICE])),
            "final_insert_step_mm": final_insert,
            "requested_insert_step_mm": float(requested_insert),
            "insert_cap_margin_mm": float(insert_cap_margin),
            "action_smoothness_norm": float(np.linalg.norm(final_action - previous_action)),
            "advance_blocked": bool(advance_blocked),
            "advance_block_risk": float(advance_block_risk),
            "residual_was_clipped": bool(np.linalg.norm(raw_residual - clipped_residual_command) > 1e-9),
            "residual_was_gated": bool(np.linalg.norm(clipped_residual_command - effective_residual_command) > 1e-9),
            "insert_trigger_value": float(hybrid_trigger.get("trigger_value", 0.0)),
            "insert_trigger_threshold": float(hybrid_trigger.get("trigger_threshold", 0.0)),
            "insert_trigger_active": bool(hybrid_trigger.get("trigger_active", False)),
            "insert_trigger_safe": bool(hybrid_trigger.get("trigger_safe", False)),
            "insert_trigger_blocked": bool(hybrid_trigger.get("trigger_blocked", False)),
            "insert_trigger_missed": bool(hybrid_trigger.get("trigger_missed", False)),
            "insert_trigger_opportunity_safe": bool(hybrid_trigger.get("trigger_opportunity_safe", False)),
            "insert_trigger_phase_mode": str(hybrid_trigger.get("trigger_phase_mode", phase_gate_mode)),
            "insert_trigger_gate_safe": bool(hybrid_trigger.get("trigger_gate_safe", self._safe_to_insert(observed))),
            **self._contact_trigger_model_info(observed),
        }

    def _contact_risk_model_info(
        self,
        before: dict[str, Any],
        after: dict[str, Any],
        action: np.ndarray,
        action_info: dict[str, Any],
    ) -> dict[str, Any]:
        info = {
            "contact_risk_model_enabled": False,
            "contact_risk_probability": 0.0,
            "contact_risk_logit": 0.0,
            "contact_risk_above_threshold": False,
        }
        if self.contact_risk_model is not None:
            info.update(
                self._predict_contact_risk_model(
                    model=self.contact_risk_model,
                    model_cfg=self.contact_risk_model_cfg,
                    before=before,
                    after=after,
                    action=action,
                    action_info=action_info,
                    prefix="contact_risk",
                )
            )
        for name, payload in self.contact_risk_models.items():
            model_cfg = payload["config"]
            prefix = str(model_cfg.get("output_prefix", f"{name}_risk"))
            info.update(
                self._predict_contact_risk_model(
                    model=payload["model"],
                    model_cfg=model_cfg,
                    before=before,
                    after=after,
                    action=action,
                    action_info=action_info,
                    prefix=prefix,
                )
            )
        return info

    def _predict_contact_risk_model(
        self,
        *,
        model: dict[str, Any],
        model_cfg: dict[str, Any],
        before: dict[str, Any],
        after: dict[str, Any],
        action: np.ndarray,
        action_info: dict[str, Any],
        prefix: str,
    ) -> dict[str, Any]:
        insert_stage_only = bool(model_cfg.get("insertion_stage_only", True))
        insertion_stage = (
            float(before.get("insertion_depth_mm", 0.0)) > 1e-9
            or float(after.get("insertion_depth_mm", 0.0)) > 1e-9
            or abs(float(action[INSERT_INDEX])) > 1e-9
            or (
                bool(model_cfg.get("include_trigger_active", True))
                and bool(action_info.get("insert_trigger_active", False))
            )
            or (
                bool(model_cfg.get("include_trigger_safe", True))
                and bool(action_info.get("insert_trigger_safe", False))
            )
        )
        if insert_stage_only and not insertion_stage:
            return {
                f"{prefix}_model_enabled": True,
                f"{prefix}_probability": 0.0,
                f"{prefix}_logit": 0.0,
                f"{prefix}_above_threshold": False,
            }
        feature_names = list(model.get("feature_names", []))
        values = np.asarray(
            [
                self._contact_risk_feature_value(
                    name,
                    before=before,
                    after=after,
                    action=action,
                    action_info=action_info,
                )
                for name in feature_names
            ],
            dtype=np.float64,
        )
        mean = np.asarray(model["feature_mean"], dtype=np.float64)
        std = np.asarray(model["feature_std"], dtype=np.float64)
        weights = np.asarray(model["weights"], dtype=np.float64)
        if values.shape != mean.shape or values.shape != weights.shape:
            raise ValueError(
                "contact risk feature shape mismatch: "
                f"values={values.shape}, mean={mean.shape}, weights={weights.shape}"
            )
        z = (values - mean) / np.where(std < 1e-9, 1.0, std)
        logit = float(z @ weights + float(model.get("bias", 0.0)))
        probability = float(1.0 / (1.0 + np.exp(-np.clip(logit, -50.0, 50.0))))
        threshold = float(
            model_cfg.get(
                "threshold",
                model.get("threshold", 0.5),
            )
        )
        result = {
            f"{prefix}_model_enabled": True,
            f"{prefix}_probability": probability,
            f"{prefix}_logit": logit,
            f"{prefix}_threshold": threshold,
            f"{prefix}_above_threshold": bool(probability >= threshold),
        }
        if prefix == "contact_risk":
            result.update(
                {
                    "contact_risk_model_enabled": True,
                    "contact_risk_probability": probability,
                    "contact_risk_logit": logit,
                    "contact_risk_threshold": threshold,
                    "contact_risk_above_threshold": bool(probability >= threshold),
                }
            )
        return result

    def _contact_trigger_model_info(self, observed: dict[str, Any]) -> dict[str, Any]:
        if self.contact_trigger_model is None or self.state is None:
            return {
                "contact_trigger_model_enabled": False,
                "contact_trigger_probability": 0.0,
                "contact_trigger_logit": 0.0,
                "contact_trigger_above_threshold": False,
            }
        feature_names = list(self.contact_trigger_model.get("feature_names", []))
        values = np.asarray(
            [self._contact_trigger_feature_value(name, observed=observed) for name in feature_names],
            dtype=np.float64,
        )
        mean = np.asarray(self.contact_trigger_model["feature_mean"], dtype=np.float64)
        std = np.asarray(self.contact_trigger_model["feature_std"], dtype=np.float64)
        weights = np.asarray(self.contact_trigger_model["weights"], dtype=np.float64)
        if values.shape != mean.shape or values.shape != weights.shape:
            raise ValueError(
                "contact trigger feature shape mismatch: "
                f"values={values.shape}, mean={mean.shape}, weights={weights.shape}"
            )
        z = (values - mean) / np.where(std < 1e-9, 1.0, std)
        logit = float(z @ weights + float(self.contact_trigger_model.get("bias", 0.0)))
        probability = float(1.0 / (1.0 + np.exp(-np.clip(logit, -50.0, 50.0))))
        threshold = float(
            self.contact_trigger_model_cfg.get(
                "threshold",
                self.contact_trigger_model.get("threshold", 0.5),
            )
        )
        return {
            "contact_trigger_model_enabled": True,
            "contact_trigger_probability": probability,
            "contact_trigger_logit": logit,
            "contact_trigger_threshold": threshold,
            "contact_trigger_above_threshold": bool(probability >= threshold),
        }

    def _contact_helper_model_info(
        self,
        before: dict[str, Any],
        after: dict[str, Any],
        action: np.ndarray,
        action_info: dict[str, Any],
    ) -> dict[str, Any]:
        if self.contact_helper_model is None:
            return {
                "contact_helper_model_enabled": False,
                "contact_helper_depth_step_mm": 0.0,
                "contact_helper_feedback_x_mm": 0.0,
                "contact_helper_feedback_y_mm": 0.0,
                "contact_helper_feedback_norm_mm": 0.0,
                "contact_helper_depth_active": False,
                "contact_helper_feedback_active": False,
                "contact_helper_depth_mismatch_mm": 0.0,
                "contact_helper_depth_floor_applied": False,
            }
        feature_names = list(self.contact_helper_model.get("feature_names", []))
        values = np.asarray(
            [
                self._contact_helper_feature_value(
                    name,
                    before=before,
                    after=after,
                    action=action,
                    action_info=action_info,
                )
                for name in feature_names
            ],
            dtype=np.float64,
        )
        mean = np.asarray(self.contact_helper_model["feature_mean"], dtype=np.float64)
        std = np.asarray(self.contact_helper_model["feature_std"], dtype=np.float64)
        weights = np.asarray(self.contact_helper_model["weights"], dtype=np.float64)
        bias = np.asarray(self.contact_helper_model["bias"], dtype=np.float64)
        if values.shape != mean.shape or values.shape[0] != weights.shape[0]:
            raise ValueError(
                "contact helper feature shape mismatch: "
                f"values={values.shape}, mean={mean.shape}, weights={weights.shape}"
            )
        z = (values - mean) / np.where(std < 1e-9, 1.0, std)
        prediction = z @ weights + bias
        max_depth_step = float(
            self.contact_helper_model_cfg.get(
                "max_depth_step_mm",
                self.contact_helper_model.get("max_depth_step_mm", 0.10),
            )
        )
        max_feedback = float(
            self.contact_helper_model_cfg.get(
                "max_feedback_mm",
                self.contact_helper_model.get("max_feedback_mm", 0.12),
            )
        )
        depth_step = float(np.clip(prediction[0], 0.0, max_depth_step))
        depth_floor_applied = False
        safe_depth_floor = float(self.contact_helper_model_cfg.get("safe_trigger_depth_floor_mm", 0.0))
        if safe_depth_floor > 0.0:
            insertion_started = float(before.get("insertion_depth_mm", 0.0)) > 1e-9
            safe_trigger = bool(action_info.get("insert_trigger_safe", False))
            active_trigger = bool(action_info.get("insert_trigger_active", False))
            contact_trigger = bool(action_info.get("contact_trigger_above_threshold", False))
            floor_requires_trigger = bool(
                self.contact_helper_model_cfg.get("depth_floor_requires_trigger_active", True)
            )
            trigger_context = safe_trigger and (active_trigger or contact_trigger or not floor_requires_trigger)
            if insertion_started or trigger_context:
                target_depth = float(self.task_cfg["target_insert_depth_mm"])
                remaining_depth = max(0.0, target_depth - float(before.get("insertion_depth_mm", 0.0)))
                floor_value = min(max_depth_step, safe_depth_floor, remaining_depth)
                if depth_step < floor_value:
                    depth_step = float(floor_value)
                    depth_floor_applied = True
        feedback = np.asarray(prediction[1:3], dtype=np.float64)
        feedback = clip_norm(feedback, max_feedback)
        feedback_norm = float(np.linalg.norm(feedback))
        actual_insert = max(0.0, float(action[INSERT_INDEX]))
        depth_active_threshold = float(self.contact_helper_model_cfg.get("depth_active_threshold_mm", 0.01))
        feedback_active_threshold = float(self.contact_helper_model_cfg.get("feedback_active_threshold_mm", 0.01))
        depth_mismatch = abs(actual_insert - depth_step) if (
            depth_step > depth_active_threshold or actual_insert > 1e-9
        ) else 0.0
        return {
            "contact_helper_model_enabled": True,
            "contact_helper_depth_step_mm": depth_step,
            "contact_helper_feedback_x_mm": float(feedback[0]),
            "contact_helper_feedback_y_mm": float(feedback[1]),
            "contact_helper_feedback_norm_mm": feedback_norm,
            "contact_helper_depth_active": bool(depth_step > depth_active_threshold),
            "contact_helper_feedback_active": bool(feedback_norm > feedback_active_threshold),
            "contact_helper_depth_mismatch_mm": float(depth_mismatch),
            "contact_helper_depth_floor_applied": bool(depth_floor_applied),
        }

    def _contact_trigger_feature_value(self, name: str, *, observed: dict[str, Any]) -> float:
        if self.state is None:
            return 0.0
        target = max(float(self.task_cfg["target_insert_depth_mm"]), 1e-9)
        insertion = float(self.state.insertion_depth_mm)
        remaining = max(0.0, target - insertion)
        if name == "lateral_error_mm":
            return float(observed.get("observed_lateral_error_mm", 0.0))
        if name == "depth_abs_error_mm":
            return abs(float(observed.get("observed_depth_error_mm", 0.0)))
        if name == "axis_error_deg":
            return float(observed.get("observed_axis_error_deg", 0.0))
        if name == "insertion_depth_mm":
            return insertion
        if name == "remaining_insert_depth_mm":
            return remaining
        if name == "insertion_fraction":
            return insertion / target
        if name == "remaining_fraction":
            return remaining / target
        if name == "confidence":
            return float(observed.get("confidence", 1.0))
        if name == "quality_gate_pass":
            return 1.0 if bool(observed.get("quality_gate_pass", True)) else 0.0
        return 0.0

    def _contact_risk_feature_value(
        self,
        name: str,
        *,
        before: dict[str, Any],
        after: dict[str, Any],
        action: np.ndarray,
        action_info: dict[str, Any],
    ) -> float:
        target = max(float(self.task_cfg["target_insert_depth_mm"]), 1e-9)
        if name == "lateral_error_mm":
            return float(after.get("lateral_error_mm", 0.0))
        if name == "depth_abs_error_mm":
            return float(after.get("depth_abs_error_mm", 0.0))
        if name == "axis_error_deg":
            return float(after.get("axis_error_deg", 0.0))
        if name == "insertion_depth_mm":
            return float(after.get("insertion_depth_mm", 0.0))
        if name == "remaining_insert_depth_mm":
            return max(0.0, target - float(after.get("insertion_depth_mm", 0.0)))
        if name == "mujoco_remaining_insert_depth_mm":
            return max(0.0, target - float(after.get("insertion_depth_mm", 0.0)))
        if name == "insertion_fraction":
            return float(after.get("insertion_depth_mm", 0.0)) / target
        if name == "mujoco_insertion_fraction":
            return float(after.get("insertion_depth_mm", 0.0)) / target
        if name == "final_action_x_mm":
            return float(action[0])
        if name == "final_action_y_mm":
            return float(action[1])
        if name == "final_action_depth_mm":
            return float(action[2])
        if name == "final_action_insert_mm":
            return float(action[INSERT_INDEX])
        if name == "raw_final_action_x_mm":
            return float(action[0])
        if name == "raw_final_action_y_mm":
            return float(action[1])
        if name == "raw_final_action_depth_mm":
            return float(action[2])
        if name == "raw_final_action_insert_mm":
            return float(action[INSERT_INDEX])
        if name == "insert_trigger_active":
            return 1.0 if bool(action_info.get("insert_trigger_active", False)) else 0.0
        if name == "insert_trigger_safe":
            return 1.0 if bool(action_info.get("insert_trigger_safe", False)) else 0.0
        if name == "insert_trigger_blocked":
            return 1.0 if bool(action_info.get("insert_trigger_blocked", False)) else 0.0
        if name == "mujoco_before_lateral_error_mm":
            return float(before.get("lateral_error_mm", 0.0))
        if name == "mujoco_before_insertion_depth_mm":
            return float(before.get("insertion_depth_mm", 0.0))
        if name == "mujoco_before_lateral_y_mm":
            lateral_xy = before.get("lateral_xy_mm", [0.0, 0.0])
            return float(lateral_xy[0])
        if name == "mujoco_before_lateral_z_mm":
            lateral_xy = before.get("lateral_xy_mm", [0.0, 0.0])
            return float(lateral_xy[1])
        if name == "step_fraction":
            return float(after.get("step", 0.0)) / max(float(self.episode_cfg.get("max_steps", 1)), 1.0)
        return 0.0

    def _contact_helper_feature_value(
        self,
        name: str,
        *,
        before: dict[str, Any],
        after: dict[str, Any],
        action: np.ndarray,
        action_info: dict[str, Any],
    ) -> float:
        target = max(float(self.task_cfg["target_insert_depth_mm"]), 1e-9)
        insertion = float(after.get("insertion_depth_mm", 0.0))
        lateral_xy = after.get("lateral_xy_mm", [0.0, 0.0])
        if name == "lateral_error_mm":
            return float(after.get("lateral_error_mm", 0.0))
        if name == "depth_abs_error_mm":
            return float(after.get("depth_abs_error_mm", 0.0))
        if name == "axis_error_deg":
            return float(after.get("axis_error_deg", 0.0))
        if name == "insertion_depth_mm":
            return insertion
        if name == "remaining_insert_depth_mm":
            return max(0.0, target - insertion)
        if name == "lateral_x_mm":
            return float(lateral_xy[0])
        if name == "lateral_y_mm":
            return float(lateral_xy[1])
        if name == "insertion_fraction":
            return insertion / target
        if name == "remaining_fraction":
            return max(0.0, target - insertion) / target
        if name == "final_action_insert_mm":
            return float(action[INSERT_INDEX])
        if name == "final_action_x_mm":
            return float(action[0])
        if name == "final_action_y_mm":
            return float(action[1])
        if name == "insert_trigger_active":
            return 1.0 if bool(action_info.get("insert_trigger_active", False)) else 0.0
        if name == "insert_trigger_safe":
            return 1.0 if bool(action_info.get("insert_trigger_safe", False)) else 0.0
        if name == "insert_trigger_blocked":
            return 1.0 if bool(action_info.get("insert_trigger_blocked", False)) else 0.0
        if name == "step_fraction":
            return float(after.get("step", 0.0)) / max(float(self.episode_cfg.get("max_steps", 1)), 1.0)
        return self._contact_risk_feature_value(
            name,
            before=before,
            after=after,
            action=action,
            action_info=action_info,
        )

    def _advance_block_risk(self, observed: dict[str, Any]) -> float:
        """Scale blocked-insertion cost by how unsafe the attempted advance was."""
        if not bool(observed.get("quality_gate_pass", True)):
            return 1.0
        lateral = float(observed["observed_lateral_error_mm"])
        depth = abs(float(observed["observed_depth_error_mm"]))
        axis = float(observed["observed_axis_error_deg"])
        lateral_limit = max(
            float(self.phase_action_gate_cfg.get("align_lateral_mm", self.safety_cfg["allow_insert_lateral_mm"])),
            1e-9,
        )
        depth_limit = max(
            float(self.phase_action_gate_cfg.get("depth_gate_mm", self.safety_cfg["allow_insert_depth_mm"])),
            1e-9,
        )
        axis_limit = max(
            float(self.phase_action_gate_cfg.get("align_axis_deg", self.safety_cfg["allow_insert_axis_deg"])),
            1e-9,
        )
        ratio = max(lateral / lateral_limit, depth / depth_limit, axis / axis_limit)
        return float(np.clip(ratio - 1.0, 0.0, 3.0))

    def _reward(
        self,
        before: dict[str, Any],
        after: dict[str, Any],
        action: np.ndarray,
        previous_action: np.ndarray,
        action_info: dict[str, Any],
        success: bool,
        failure_reason: str | None,
        timeout: bool,
    ) -> tuple[float, dict[str, float]]:
        cfg = self.reward_cfg
        precision_improvement = float(before["weighted_precision_error_mm"]) - float(after["weighted_precision_error_mm"])
        insert_progress = float(after["insertion_depth_mm"]) - float(before["insertion_depth_mm"])
        joint_velocity_cost = float(after.get("mean_abs_joint_velocity_deg_s", 0.0))
        joint_margin = float(after.get("min_joint_limit_margin_deg", 180.0))
        joint_limit_risk = max(0.0, 1.0 - joint_margin / 60.0)
        residual_norm = float(action_info.get("effective_residual_norm", action_info.get("clipped_residual_norm", 0.0)))
        precision_regression = max(0.0, -precision_improvement)
        action_reversal = self._action_reversal(previous_action, action)
        post_insert_precision = self._post_insert_precision_excess(after, success)
        action_smoothness = float(np.linalg.norm(action - previous_action))
        near_target_smoothness = action_smoothness if self._near_target_reward_zone(after) else 0.0
        insert_trigger_active = bool(action_info.get("insert_trigger_active", False))
        insert_trigger_safe = bool(action_info.get("insert_trigger_safe", False))
        insert_trigger_blocked = bool(action_info.get("insert_trigger_blocked", False))
        insert_trigger_missed = bool(action_info.get("insert_trigger_missed", False))
        insert_trigger_opportunity_safe = bool(action_info.get("insert_trigger_opportunity_safe", False))
        trigger_value = float(action_info.get("insert_trigger_value", 0.0))
        trigger_threshold = max(abs(float(action_info.get("insert_trigger_threshold", 0.0))), 1e-9)
        contact_trigger_probability = float(action_info.get("contact_trigger_probability", 0.0))
        contact_trigger_threshold = float(
            action_info.get(
                "contact_trigger_threshold",
                self.contact_trigger_model_cfg.get("threshold", 0.5),
            )
        )
        contact_trigger_above_threshold = bool(action_info.get("contact_trigger_above_threshold", False))
        contact_trigger_confidence = float(np.clip(contact_trigger_probability, 0.0, 1.0))
        target_insert_depth = max(float(self.task_cfg["target_insert_depth_mm"]), 1e-9)
        remaining_fraction = float(after.get("remaining_insert_depth_mm", 0.0)) / target_insert_depth
        trigger_positive_fraction = float(np.clip(max(0.0, trigger_value) / trigger_threshold, 0.0, 1.0))
        trigger_signed_fraction = float(np.clip(trigger_value / trigger_threshold, -1.0, 1.0))
        post_insert_lateral_drift = max(0.0, float(after["lateral_error_mm"]) - float(before["lateral_error_mm"])) if insert_progress > 1e-9 else 0.0
        contact_proxy_excess = 0.0
        if insert_progress > 1e-9:
            lateral_excess = max(0.0, float(after["lateral_error_mm"]) - float(self.task_cfg["success_lateral_mm"]))
            axis_excess = max(0.0, float(after["axis_error_deg"]) - float(self.task_cfg["success_axis_deg"]))
            contact_proxy_excess = lateral_excess + 0.2 * axis_excess
        contact_label_proxy_risk = self._contact_label_proxy_risk(
            before=before,
            after=after,
            action=action,
            action_info=action_info,
            insert_progress=insert_progress,
            post_insert_lateral_drift=post_insert_lateral_drift,
            action_smoothness=action_smoothness,
        )
        insert_started = bool(float(before["insertion_depth_mm"]) > 1e-9 or float(after["insertion_depth_mm"]) > 1e-9)
        depth_objective_context = bool(insert_started or (insert_trigger_active and insert_trigger_safe))
        safe_depth_progress_context = bool(
            insert_started or (insert_trigger_active and insert_trigger_safe and contact_trigger_above_threshold)
        )
        depth_completion_need = max(0.0, remaining_fraction) if depth_objective_context else 0.0
        contact_risk_probability = float(action_info.get("contact_risk_probability", 0.0))
        contact_feedback_need = 0.0
        if insert_started or insert_progress > 1e-9:
            lateral_ratio = float(after["lateral_error_mm"]) / max(float(self.task_cfg["success_lateral_mm"]), 1e-9)
            drift_ratio = post_insert_lateral_drift / max(float(cfg.get("helper_lateral_drift_warning_mm", 0.03)), 1e-9)
            contact_feedback_need = lateral_ratio + 0.35 * contact_risk_probability + 0.50 * drift_ratio
            contact_feedback_need = float(np.clip(contact_feedback_need, 0.0, float(cfg.get("max_helper_feedback_need", 5.0))))
        helper_depth_step = float(action_info.get("contact_helper_depth_step_mm", 0.0))
        helper_feedback_norm = float(action_info.get("contact_helper_feedback_norm_mm", 0.0))
        helper_depth_mismatch = float(action_info.get("contact_helper_depth_mismatch_mm", 0.0))
        helper_depth_follow = min(max(0.0, float(action[INSERT_INDEX])), max(0.0, helper_depth_step))
        components = {
            "precision_improvement": float(cfg["precision_improvement"]) * precision_improvement,
            "insert_progress": float(cfg["insert_progress"]) * insert_progress,
            "contact_depth_progress_bonus": 0.0,
            "helper_depth_completion_need_penalty": -float(cfg.get("helper_depth_completion_need_penalty", 0.0))
            * depth_completion_need,
            "helper_feedback_need_penalty": -float(cfg.get("helper_feedback_need_penalty", 0.0)) * contact_feedback_need,
            "contact_helper_depth_follow_bonus": float(cfg.get("contact_helper_depth_follow_bonus", 0.0))
            * helper_depth_follow,
            "contact_helper_depth_target_penalty": -float(cfg.get("contact_helper_depth_target_penalty", 0.0))
            * helper_depth_mismatch,
            "contact_helper_feedback_need_penalty": -float(cfg.get("contact_helper_feedback_need_penalty", 0.0))
            * helper_feedback_norm,
            "lateral_penalty": -float(cfg["lateral_penalty"]) * float(after["lateral_error_mm"]),
            "depth_penalty": -float(cfg["depth_penalty"]) * float(after["depth_abs_error_mm"]),
            "axis_penalty": -float(cfg["axis_penalty"]) * float(after["axis_error_deg"]),
            "step_penalty": -float(cfg["step_penalty"]),
            "action_penalty": -float(cfg["action_penalty"]) * float(np.linalg.norm(action)),
            "smoothness_penalty": -float(cfg["smoothness_penalty"]) * action_smoothness,
            "near_target_smoothness_penalty": -float(cfg.get("near_target_smoothness_penalty", 0.0)) * near_target_smoothness,
            "residual_penalty": -float(cfg.get("residual_penalty", 0.0)) * residual_norm,
            "precision_regression_penalty": -float(cfg.get("precision_regression_penalty", 0.0)) * precision_regression,
            "action_reversal_penalty": -float(cfg.get("action_reversal_penalty", 0.0)) * action_reversal,
            "post_insert_precision_penalty": -float(cfg.get("post_insert_precision_penalty", 0.0)) * post_insert_precision,
            "joint_velocity_penalty": -float(cfg.get("joint_velocity_penalty", 0.0)) * joint_velocity_cost,
            "joint_limit_penalty": -float(cfg.get("joint_limit_penalty", 0.0)) * joint_limit_risk,
            "advance_block_penalty": 0.0,
            "insert_trigger_alignment_bonus": 0.0,
            "safe_insert_continuation_bonus": 0.0,
            "safe_trigger_margin_bonus": 0.0,
            "unsafe_trigger_margin_penalty": 0.0,
            "blocked_trigger_penalty": 0.0,
            "insert_trigger_missed_penalty": 0.0,
            "post_insert_lateral_drift_penalty": -float(cfg.get("post_insert_lateral_drift_penalty", 0.0)) * post_insert_lateral_drift,
            "contact_proxy_penalty": -float(cfg.get("contact_proxy_penalty", 0.0)) * contact_proxy_excess,
            "contact_label_proxy_penalty": -float(cfg.get("contact_label_proxy_penalty", 0.0)) * contact_label_proxy_risk,
            "contact_risk_model_penalty": -float(cfg.get("contact_risk_model_penalty", 0.0)) * contact_risk_probability,
            "contact_trigger_model_safe_bonus": 0.0,
            "contact_trigger_model_missed_penalty": 0.0,
            "contact_trigger_model_unsafe_penalty": 0.0,
            "quality_refuse_penalty": 0.0,
            "success_bonus": 0.0,
            "timeout_penalty": 0.0,
            "timeout_insert_shortfall_penalty": 0.0,
            "failure_penalty": 0.0,
        }
        if bool(action_info["advance_blocked"]):
            risk = float(action_info.get("advance_block_risk", 0.0))
            components["advance_block_penalty"] = -float(cfg["advance_block_penalty"])
            components["premature_advance_penalty"] = -float(cfg.get("premature_advance_block_penalty", 0.0)) * risk
        if insert_trigger_active and insert_trigger_safe:
            components["insert_trigger_alignment_bonus"] = float(cfg.get("insert_trigger_alignment_bonus", 0.0))
            components["safe_insert_continuation_bonus"] = (
                float(cfg.get("safe_insert_continuation_bonus", 0.0)) * max(0.0, insert_progress)
            )
            if contact_trigger_above_threshold:
                components["contact_trigger_model_safe_bonus"] = (
                    float(cfg.get("contact_trigger_model_safe_bonus", 0.0))
                    * contact_trigger_confidence
                    * max(0.0, insert_progress)
                )
        if safe_depth_progress_context:
            components["contact_depth_progress_bonus"] = (
                float(cfg.get("contact_depth_progress_bonus", 0.0))
                * max(0.0, insert_progress)
                * (1.0 + max(0.0, remaining_fraction))
            )
        if insert_trigger_opportunity_safe:
            components["safe_trigger_margin_bonus"] = (
                float(cfg.get("safe_trigger_margin_bonus", 0.0)) * trigger_signed_fraction
            )
        elif trigger_value > 0.0:
            risk = float(action_info.get("advance_block_risk", 1.0))
            components["unsafe_trigger_margin_penalty"] = (
                -float(cfg.get("unsafe_trigger_margin_penalty", 0.0)) * trigger_positive_fraction * max(1.0, risk)
            )
        if insert_trigger_blocked:
            risk = float(action_info.get("advance_block_risk", 1.0))
            components["blocked_trigger_penalty"] = -float(cfg.get("blocked_trigger_penalty", 0.0)) * max(1.0, risk)
            components["premature_insert_trigger_penalty"] = -float(cfg.get("premature_insert_trigger_penalty", 0.0)) * max(1.0, risk)
        if insert_trigger_missed:
            components["insert_trigger_missed_penalty"] = -float(cfg.get("insert_trigger_missed_penalty", 0.0))
        if contact_trigger_above_threshold and insert_trigger_missed:
            components["contact_trigger_model_missed_penalty"] = (
                -float(cfg.get("contact_trigger_model_missed_penalty", 0.0))
                * contact_trigger_confidence
                * max(remaining_fraction, 0.05)
            )
        if insert_trigger_active and not contact_trigger_above_threshold:
            unsafe_fraction = float(np.clip((contact_trigger_threshold - contact_trigger_probability) / max(contact_trigger_threshold, 1e-9), 0.0, 1.0))
            components["contact_trigger_model_unsafe_penalty"] = (
                -float(cfg.get("contact_trigger_model_unsafe_penalty", 0.0))
                * max(trigger_positive_fraction, 0.25)
                * max(unsafe_fraction, 0.05)
            )
        if self.current_observation is not None and not bool(self.current_observation.get("quality_gate_pass", True)):
            components["quality_refuse_penalty"] = -float(cfg["quality_refuse_penalty"])
        if success:
            components["success_bonus"] = float(cfg["success_bonus"])
        if timeout and not success:
            components["timeout_penalty"] = -float(cfg.get("timeout_penalty", 0.0))
            components["timeout_insert_shortfall_penalty"] = (
                -float(cfg.get("timeout_insert_shortfall_penalty", 0.0))
                * float(after.get("remaining_insert_depth_mm", 0.0))
            )
        if failure_reason is not None:
            components["failure_penalty"] = -float(cfg["failure_penalty"])
        reward = float(sum(components.values()))
        components["total"] = reward
        return reward, components

    def _contact_label_proxy_risk(
        self,
        *,
        before: dict[str, Any],
        after: dict[str, Any],
        action: np.ndarray,
        action_info: dict[str, Any],
        insert_progress: float,
        post_insert_lateral_drift: float,
        action_smoothness: float,
    ) -> float:
        cfg = self.cfg.get("contact_label_proxy", {})
        if not bool(cfg.get("enabled", False)):
            return 0.0
        trigger_active = bool(action_info.get("insert_trigger_active", False))
        if insert_progress <= 1e-9 and not trigger_active:
            return 0.0

        def excess_ratio(value: float, limit: float) -> float:
            return max(0.0, float(value) / max(float(limit), 1e-9) - 1.0)

        lateral_ref = max(float(before["lateral_error_mm"]), float(after["lateral_error_mm"]))
        depth_ref = max(float(before["depth_abs_error_mm"]), float(after["depth_abs_error_mm"]))
        axis_ref = max(float(before["axis_error_deg"]), float(after["axis_error_deg"]))
        insert_step = max(0.0, float(action[INSERT_INDEX]))

        lateral_risk = excess_ratio(lateral_ref, float(cfg.get("safe_lateral_mm", self.task_cfg["success_lateral_mm"])))
        depth_risk = excess_ratio(depth_ref, float(cfg.get("safe_depth_mm", self.safety_cfg["allow_insert_depth_mm"])))
        axis_risk = excess_ratio(axis_ref, float(cfg.get("safe_axis_deg", self.task_cfg["success_axis_deg"])))
        step_risk = excess_ratio(insert_step, float(cfg.get("max_insert_step_mm", self.safety_cfg["max_base_insert_step_mm"])))
        drift_risk = float(post_insert_lateral_drift) / max(float(cfg.get("drift_warning_mm", 0.03)), 1e-9)
        smoothness_risk = excess_ratio(action_smoothness, float(cfg.get("smoothness_warning_norm", 0.90)))
        blocked_risk = float(cfg.get("blocked_trigger_risk", 0.25)) if bool(action_info.get("insert_trigger_blocked", False)) else 0.0
        missed_risk = float(cfg.get("missed_trigger_risk", 0.10)) if bool(action_info.get("insert_trigger_missed", False)) else 0.0

        risk = (
            float(cfg.get("lateral_weight", 1.00)) * lateral_risk
            + float(cfg.get("depth_weight", 0.70)) * depth_risk
            + float(cfg.get("axis_weight", 0.50)) * axis_risk
            + float(cfg.get("step_weight", 0.40)) * step_risk
            + float(cfg.get("drift_weight", 0.80)) * drift_risk
            + float(cfg.get("smoothness_weight", 0.30)) * smoothness_risk
            + blocked_risk
            + missed_risk
        )
        return float(np.clip(risk, 0.0, float(cfg.get("max_risk", 5.0))))

    def _near_target_reward_zone(self, metrics: dict[str, Any]) -> bool:
        cfg = self.reward_cfg
        lateral_limit = float(cfg.get("near_target_lateral_mm", self.safety_cfg["allow_insert_lateral_mm"]))
        depth_limit = float(cfg.get("near_target_depth_mm", self.safety_cfg["allow_insert_depth_mm"]))
        axis_limit = float(cfg.get("near_target_axis_deg", self.safety_cfg["allow_insert_axis_deg"]))
        return (
            float(metrics["lateral_error_mm"]) <= lateral_limit
            and float(metrics["depth_abs_error_mm"]) <= depth_limit
            and float(metrics["axis_error_deg"]) <= axis_limit
        )

    def _action_reversal(self, previous_action: np.ndarray, action: np.ndarray) -> float:
        previous_motion = np.asarray([previous_action[0], previous_action[1], previous_action[2], previous_action[5]])
        current_motion = np.asarray([action[0], action[1], action[2], action[5]])
        previous_norm = float(np.linalg.norm(previous_motion))
        current_norm = float(np.linalg.norm(current_motion))
        if previous_norm <= 1e-9 or current_norm <= 1e-9:
            return 0.0
        cosine = float(np.dot(previous_motion, current_motion) / (previous_norm * current_norm))
        return 1.0 if cosine < -0.25 else 0.0

    def _post_insert_precision_excess(self, metrics: dict[str, Any], success: bool) -> float:
        if success:
            return 0.0
        if float(metrics["insertion_depth_mm"]) < float(self.task_cfg["target_insert_depth_mm"]) - 1e-9:
            return 0.0
        lateral_excess = max(0.0, float(metrics["lateral_error_mm"]) - float(self.task_cfg["success_lateral_mm"]))
        depth_excess = max(0.0, float(metrics["depth_abs_error_mm"]) - float(self.task_cfg["success_depth_mm"]))
        axis_excess = max(0.0, float(metrics["axis_error_deg"]) - float(self.task_cfg["success_axis_deg"]))
        return lateral_excess + depth_excess + 0.2 * axis_excess
