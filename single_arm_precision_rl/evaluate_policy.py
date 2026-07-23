from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config, resolve_project_path
from .episode_diagnostics import EpisodeDiagnostics
from .environment import SingleArmPrecisionEnv
from .policies import make_policy


def run_episode(env: SingleArmPrecisionEnv, policy_name: str, seed: int, random_scale: float) -> dict[str, Any]:
    low, high = env.residual_action_bounds()
    policy = make_policy(policy_name, low, high, random_scale=random_scale, config=env.cfg)
    policy.reset(seed)
    observation, info = env.reset(seed=seed)
    diagnostics = EpisodeDiagnostics.from_config(env.cfg)
    diagnostics.start(info["metrics"])
    total_reward = 0.0
    reward_component_totals: dict[str, float] = defaultdict(float)
    done = False
    quality_refusals = 0
    advance_blocks = 0
    residual_clip_count = 0
    insert_trigger_active_count = 0
    insert_trigger_safe_count = 0
    insert_trigger_blocked_count = 0
    insert_trigger_missed_count = 0
    smoothness_total = 0.0
    residual_norm_total = 0.0
    contact_risk_probability_total = 0.0
    contact_risk_active_count = 0
    contact_trigger_probability_total = 0.0
    contact_trigger_active_count = 0
    contact_helper_depth_step_total = 0.0
    contact_helper_depth_active_count = 0
    contact_helper_depth_floor_count = 0
    contact_helper_feedback_norm_total = 0.0
    contact_helper_feedback_active_count = 0
    while not done:
        residual_action = policy.act(observation, info)
        observation, reward, done, info = env.step(residual_action)
        diagnostics.update(info)
        total_reward += reward
        for key, value in info.get("reward_components", {}).items():
            reward_component_totals[key] += float(value)
        quality_refusals += int(info.get("refused_by_quality", False))
        action_info = info.get("action_info", {})
        advance_blocks += int(action_info.get("advance_blocked", False))
        residual_clip_count += int(action_info.get("residual_was_clipped", False))
        insert_trigger_active_count += int(action_info.get("insert_trigger_active", False))
        insert_trigger_safe_count += int(action_info.get("insert_trigger_safe", False))
        insert_trigger_blocked_count += int(action_info.get("insert_trigger_blocked", False))
        insert_trigger_missed_count += int(action_info.get("insert_trigger_missed", False))
        smoothness_total += float(action_info.get("action_smoothness_norm", 0.0))
        residual_norm_total += float(action_info.get("effective_residual_norm", action_info.get("clipped_residual_norm", 0.0)))
        contact_risk_probability_total += float(action_info.get("contact_risk_probability", 0.0))
        contact_risk_active_count += int(action_info.get("contact_risk_above_threshold", False))
        contact_trigger_probability_total += float(action_info.get("contact_trigger_probability", 0.0))
        contact_trigger_active_count += int(action_info.get("contact_trigger_above_threshold", False))
        contact_helper_depth_step_total += float(action_info.get("contact_helper_depth_step_mm", 0.0))
        contact_helper_depth_active_count += int(action_info.get("contact_helper_depth_active", False))
        contact_helper_depth_floor_count += int(action_info.get("contact_helper_depth_floor_applied", False))
        contact_helper_feedback_norm_total += float(action_info.get("contact_helper_feedback_norm_mm", 0.0))
        contact_helper_feedback_active_count += int(action_info.get("contact_helper_feedback_active", False))

    metrics = info["metrics"]
    steps = max(1, int(metrics["step"]))
    return {
        "seed": seed,
        "policy": policy_name,
        "steps": int(metrics["step"]),
        "time_s": float(metrics["time_s"]),
        "total_reward": float(total_reward),
        "reward_components": {key: float(value) for key, value in sorted(reward_component_totals.items())},
        "success": bool(info["success"]),
        "failure_reason": info["failure_reason"],
        "timeout": bool(info["timeout"]),
        "quality_refusals": int(quality_refusals),
        "advance_blocks": int(advance_blocks),
        "residual_clip_count": int(residual_clip_count),
        "insert_trigger_active_count": int(insert_trigger_active_count),
        "insert_trigger_safe_count": int(insert_trigger_safe_count),
        "insert_trigger_blocked_count": int(insert_trigger_blocked_count),
        "insert_trigger_missed_count": int(insert_trigger_missed_count),
        "mean_action_smoothness_norm": float(smoothness_total / steps),
        "mean_residual_norm": float(residual_norm_total / steps),
        "mean_contact_risk_probability": float(contact_risk_probability_total / steps),
        "contact_risk_active_count": int(contact_risk_active_count),
        "mean_contact_trigger_probability": float(contact_trigger_probability_total / steps),
        "contact_trigger_active_count": int(contact_trigger_active_count),
        "mean_contact_helper_depth_step_mm": float(contact_helper_depth_step_total / steps),
        "contact_helper_depth_active_count": int(contact_helper_depth_active_count),
        "contact_helper_depth_floor_count": int(contact_helper_depth_floor_count),
        "mean_contact_helper_feedback_norm_mm": float(contact_helper_feedback_norm_total / steps),
        "contact_helper_feedback_active_count": int(contact_helper_feedback_active_count),
        "final_lateral_error_mm": float(metrics["lateral_error_mm"]),
        "final_depth_abs_error_mm": float(metrics["depth_abs_error_mm"]),
        "final_axis_error_deg": float(metrics["axis_error_deg"]),
        "final_weighted_precision_error_mm": float(metrics["weighted_precision_error_mm"]),
        "final_insertion_depth_mm": float(metrics["insertion_depth_mm"]),
        "final_min_joint_limit_margin_deg": float(metrics["min_joint_limit_margin_deg"]),
        "final_mean_abs_joint_velocity_deg_s": float(metrics["mean_abs_joint_velocity_deg_s"]),
        **diagnostics.summary(),
    }


def summarize(episodes: list[dict[str, Any]], *, config_path: Path, policy_name: str) -> dict[str, Any]:
    failures = Counter(item["failure_reason"] or "none" for item in episodes)

    def mean(key: str) -> float:
        return float(np.mean([item[key] for item in episodes])) if episodes else 0.0

    reward_keys = sorted({key for item in episodes for key in item.get("reward_components", {})})
    mean_reward_components = {
        key: float(np.mean([item.get("reward_components", {}).get(key, 0.0) for item in episodes]))
        for key in reward_keys
    }

    return {
        "config": str(config_path),
        "policy": policy_name,
        "episode_count": len(episodes),
        "success_count": int(sum(1 for item in episodes if item["success"])),
        "timeout_count": int(sum(1 for item in episodes if item["timeout"])),
        "failure_reasons": dict(failures),
        "mean_steps": mean("steps"),
        "mean_time_s": mean("time_s"),
        "mean_total_reward": mean("total_reward"),
        "mean_quality_refusals": mean("quality_refusals"),
        "mean_advance_blocks": mean("advance_blocks"),
        "mean_residual_clip_count": mean("residual_clip_count"),
        "mean_insert_trigger_active_count": mean("insert_trigger_active_count"),
        "mean_insert_trigger_safe_count": mean("insert_trigger_safe_count"),
        "mean_insert_trigger_blocked_count": mean("insert_trigger_blocked_count"),
        "mean_insert_trigger_missed_count": mean("insert_trigger_missed_count"),
        "mean_action_smoothness_norm": mean("mean_action_smoothness_norm"),
        "mean_residual_norm": mean("mean_residual_norm"),
        "mean_contact_risk_probability": mean("mean_contact_risk_probability"),
        "mean_contact_risk_active_count": mean("contact_risk_active_count"),
        "mean_contact_trigger_probability": mean("mean_contact_trigger_probability"),
        "mean_contact_trigger_active_count": mean("contact_trigger_active_count"),
        "mean_contact_helper_depth_step_mm": mean("mean_contact_helper_depth_step_mm"),
        "mean_contact_helper_depth_active_count": mean("contact_helper_depth_active_count"),
        "mean_contact_helper_depth_floor_count": mean("contact_helper_depth_floor_count"),
        "mean_contact_helper_feedback_norm_mm": mean("mean_contact_helper_feedback_norm_mm"),
        "mean_contact_helper_feedback_active_count": mean("contact_helper_feedback_active_count"),
        "mean_reward_components": mean_reward_components,
        "mean_final_lateral_error_mm": mean("final_lateral_error_mm"),
        "mean_final_depth_abs_error_mm": mean("final_depth_abs_error_mm"),
        "mean_final_axis_error_deg": mean("final_axis_error_deg"),
        "mean_final_weighted_precision_error_mm": mean("final_weighted_precision_error_mm"),
        "mean_final_insertion_depth_mm": mean("final_insertion_depth_mm"),
        "mean_final_min_joint_limit_margin_deg": mean("final_min_joint_limit_margin_deg"),
        "mean_final_mean_abs_joint_velocity_deg_s": mean("final_mean_abs_joint_velocity_deg_s"),
        "mean_task_path_length_mm": mean("task_path_length_mm"),
        "mean_translation_path_length_mm": mean("translation_path_length_mm"),
        "mean_insert_path_length_mm": mean("insert_path_length_mm"),
        "mean_ideal_task_path_length_mm": mean("ideal_task_path_length_mm"),
        "mean_path_efficiency_ratio": mean("path_efficiency_ratio"),
        "mean_joint_path_length_deg": mean("joint_path_length_deg"),
        "mean_episode_min_joint_limit_margin_deg": mean("episode_min_joint_limit_margin_deg"),
        "mean_joint_limit_near_count": mean("joint_limit_near_count"),
        "mean_abs_joint_velocity_trace_deg_s": mean("mean_abs_joint_velocity_trace_deg_s"),
        "mean_max_abs_joint_velocity_deg_s": mean("max_abs_joint_velocity_deg_s"),
        "mean_precision_regression_count": mean("precision_regression_count"),
        "mean_action_direction_reversal_count": mean("action_direction_reversal_count"),
        "mean_near_target_step_count": mean("near_target_step_count"),
        "mean_near_target_action_smoothness_norm": mean("near_target_mean_action_smoothness_norm"),
        "mean_near_target_residual_norm": mean("near_target_mean_residual_norm"),
        "mean_near_target_oscillation_count": mean("near_target_oscillation_count"),
        "mean_phase_gate_off_count": mean("phase_gate_off_count"),
        "mean_phase_gate_quality_refuse_count": mean("phase_gate_quality_refuse_count"),
        "mean_phase_gate_align_count": mean("phase_gate_align_count"),
        "mean_phase_gate_depth_count": mean("phase_gate_depth_count"),
        "mean_phase_gate_insert_count": mean("phase_gate_insert_count"),
        "mean_phase_gate_unknown_count": mean("phase_gate_unknown_count"),
        "mean_phase_gate_align_ratio": mean("phase_gate_align_ratio"),
        "mean_phase_gate_depth_ratio": mean("phase_gate_depth_ratio"),
        "mean_phase_gate_insert_ratio": mean("phase_gate_insert_ratio"),
        "mean_phase_gate_quality_refuse_ratio": mean("phase_gate_quality_refuse_ratio"),
        "episodes": episodes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate residual policies in the single-arm precision task.")
    parser.add_argument("--config", default="config/single_arm_precision_alignment.yaml")
    parser.add_argument(
        "--policy",
        choices=["zero", "random", "tremor", "pid", "damped_ik", "aux_trigger_depth"],
        default="zero",
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--random-scale", type=float, default=1.0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    env = SingleArmPrecisionEnv(cfg, seed=args.seed)
    episodes = [
        run_episode(env, args.policy, args.seed + idx, args.random_scale)
        for idx in range(int(args.episodes))
    ]
    summary = summarize(
        episodes,
        config_path=resolve_project_path(args.config),
        policy_name=args.policy,
    )
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        output = resolve_project_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
