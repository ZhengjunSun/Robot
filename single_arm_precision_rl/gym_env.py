from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import load_config
from .environment import SingleArmPrecisionEnv
from .policies import make_policy


class SingleArmPrecisionGymEnv(gym.Env):
    """Gymnasium adapter for single-arm precision alignment."""

    metadata = {"render_modes": []}

    def __init__(self, config: dict[str, Any] | None = None, config_path: str | None = None):
        super().__init__()
        cfg = config or load_config(config_path or "config/single_arm_precision_alignment.yaml")
        self.core = SingleArmPrecisionEnv(cfg)
        action_low, action_high = self.core.residual_action_bounds()
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)
        if bool(self.core.observation_cfg.get("normalize", False)):
            obs_high = np.ones(self.core.observation_dim, dtype=np.float32)
        else:
            obs_high = np.full(self.core.observation_dim, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)
        self.teacher_cfg = self.core.cfg.get("teacher_warm_start", cfg.get("teacher_warm_start", {"enabled": False}))
        self.teacher = None
        if bool(self.teacher_cfg.get("enabled", False)):
            self.teacher = make_policy(
                str(self.teacher_cfg.get("policy", "tremor")),
                action_low,
                action_high,
                random_scale=1.0,
                config=self.core.cfg,
            )
        self._last_observation: np.ndarray | None = None
        self._last_info: dict[str, Any] | None = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if self.teacher is not None:
            self.teacher.reset(seed)
        observation, info = self.core.reset(seed=seed)
        self._last_observation = observation.astype(np.float32)
        self._last_info = info
        return observation.astype(np.float32), info

    def step(self, action):
        agent_action = np.asarray(action, dtype=np.float64)
        effective_action, teacher_action = self._combine_teacher_action(agent_action)
        observation, reward, done, info = self.core.step(effective_action)
        if teacher_action is not None:
            info["teacher_warm_start"] = {
                "enabled": True,
                "policy": str(self.teacher_cfg.get("policy", "tremor")),
                "teacher_blend": float(self.teacher_cfg.get("teacher_blend", 1.0)),
                "correction_scale": float(self.teacher_cfg.get("correction_scale", 1.0)),
                "agent_action": agent_action.tolist(),
                "teacher_action": teacher_action.tolist(),
                "effective_action": effective_action.tolist(),
            }
        self._last_observation = observation.astype(np.float32)
        self._last_info = info
        terminated = bool(info["success"] or info["failure_reason"] is not None)
        truncated = bool(info["timeout"] and not terminated)
        return observation.astype(np.float32), float(reward), terminated, truncated, info

    def _combine_teacher_action(self, agent_action: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        if self.teacher is None or self._last_observation is None or self._last_info is None:
            return agent_action, None
        teacher_action = np.asarray(self.teacher.act(self._last_observation, self._last_info), dtype=np.float64)
        teacher_blend = float(self.teacher_cfg.get("teacher_blend", 1.0))
        correction_scale = float(self.teacher_cfg.get("correction_scale", 1.0))
        effective = teacher_blend * teacher_action + correction_scale * agent_action
        return effective, teacher_action
