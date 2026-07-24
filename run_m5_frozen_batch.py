from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from real_3d_alignment.fine_vision import FineRingEstimate
from real_3d_alignment.insertion_handoff import (
    TraditionalInsertionHandoffController,
)
from real_3d_alignment.meca500_visual_env import Meca500VisualAlignmentPlant
from real_3d_alignment.nih_baseline import (
    NIH_HRA_EYE_SOURCE,
    build_nih_coarse_servo,
    build_nih_fine_detector,
    build_nih_fine_servo,
    build_nih_traditional_detector,
)
from real_3d_alignment.scene_contract import TROCAR_TILT_DEG
from real_3d_alignment.six_axis_visual_servo import (
    ActiveEllipseOrientationServo,
    ActiveOuterEllipseOrientationServo,
    NihOuterEllipseDetector,
)
from real_3d_alignment.staged_alignment import (
    AlignmentThresholds,
    StagedAlignmentGate,
)
from run_mujoco_meca500_full_flow import calibrated_fine_estimate
from single_arm_precision_rl.clearance_contract import ClearanceSample


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "output" / "m5_frozen_500"
TARGET_STANDOFF_MM = 22.0
_WORKER_PLANT: Meca500VisualAlignmentPlant | None = None
_WORKER_HEIGHT = 0
_WORKER_MAXIMUM_ALIGNMENT_STEPS = 0


@dataclass(frozen=True)
class DomainSample:
    initial_joint_delta_deg: tuple[float, ...]
    trocar_translation_mm: tuple[float, float, float]
    trocar_rotation_deg_xyz: tuple[float, float, float]
    camera_fovy_scale: float
    principal_point_shift_px: tuple[float, float]
    rgb_gain: float
    rgb_noise_std: float
    blur_kernel: int
    trocar_rgb_scale: tuple[float, float, float]
    sclera_rgb_scale: tuple[float, float, float]
    occlusion_probability: float
    occlusion_fraction: float


class PerturbedEyeInHandPlant:
    """Image-domain perturbations around the deployment-shaped plant.

    All geometric commands are delegated to the six-axis MuJoCo plant.  Only
    the RGB stream is altered; simulator pose truth is never exposed here.
    """

    def __init__(
        self,
        plant: Meca500VisualAlignmentPlant,
        *,
        sample: DomainSample,
        rng: np.random.Generator,
    ):
        self.plant = plant
        self.sample = sample
        self.rng = rng
        self.occluded_frames = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.plant, name)

    def capture_rgb(self) -> np.ndarray:
        image = self.plant.capture_rgb().astype(np.float32)
        image *= float(self.sample.rgb_gain)
        if self.sample.rgb_noise_std > 0.0:
            image += self.rng.normal(
                0.0,
                self.sample.rgb_noise_std,
                image.shape,
            )
        image = np.clip(image, 0.0, 255.0).astype(np.uint8)
        if self.sample.blur_kernel > 1:
            image = cv2.GaussianBlur(
                image,
                (self.sample.blur_kernel, self.sample.blur_kernel),
                0.0,
            )
        shift_x, shift_y = self.sample.principal_point_shift_px
        if abs(shift_x) > 1e-9 or abs(shift_y) > 1e-9:
            image = cv2.warpAffine(
                image,
                np.asarray(
                    [[1.0, 0.0, shift_x], [0.0, 1.0, shift_y]],
                    dtype=np.float32,
                ),
                (image.shape[1], image.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
        if self.rng.random() < self.sample.occlusion_probability:
            height, width = image.shape[:2]
            side = max(
                2,
                int(round(min(height, width) * self.sample.occlusion_fraction)),
            )
            x0 = int(self.rng.integers(0, max(1, width - side + 1)))
            y0 = int(self.rng.integers(0, max(1, height - side + 1)))
            image[y0 : y0 + side, x0 : x0 + side] = 0
            self.occluded_frames += 1
        return image


def sample_domain(
    rng: np.random.Generator,
    *,
    image_height: int,
) -> DomainSample:
    image_scale = image_height / 960.0
    return DomainSample(
        initial_joint_delta_deg=tuple(
            float(value) for value in rng.uniform(-1.5, 1.5, size=6)
        ),
        trocar_translation_mm=tuple(
            float(value)
            for value in rng.uniform(
                (-0.60, -0.60, -0.35),
                (0.60, 0.60, 0.35),
            )
        ),
        trocar_rotation_deg_xyz=tuple(
            float(value) for value in rng.uniform(-2.0, 2.0, size=3)
        ),
        camera_fovy_scale=float(rng.uniform(0.985, 1.015)),
        principal_point_shift_px=tuple(
            float(value)
            for value in rng.uniform(-2.0 * image_scale, 2.0 * image_scale, 2)
        ),
        rgb_gain=float(rng.uniform(0.92, 1.08)),
        rgb_noise_std=float(rng.uniform(0.0, 2.0)),
        blur_kernel=int(rng.choice((1, 1, 1, 3))),
        trocar_rgb_scale=tuple(
            float(value) for value in rng.uniform(0.92, 1.08, size=3)
        ),
        sclera_rgb_scale=tuple(
            float(value) for value in rng.uniform(0.92, 1.08, size=3)
        ),
        occlusion_probability=float(rng.uniform(0.0, 0.025)),
        occlusion_fraction=float(rng.uniform(0.05, 0.12)),
    )


def nominal_domain() -> DomainSample:
    return DomainSample(
        initial_joint_delta_deg=(0.0,) * 6,
        trocar_translation_mm=(0.0, 0.0, 0.0),
        trocar_rotation_deg_xyz=(0.0, 0.0, 0.0),
        camera_fovy_scale=1.0,
        principal_point_shift_px=(0.0, 0.0),
        rgb_gain=1.0,
        rgb_noise_std=0.0,
        blur_kernel=1,
        trocar_rgb_scale=(1.0, 1.0, 1.0),
        sclera_rgb_scale=(1.0, 1.0, 1.0),
        occlusion_probability=0.0,
        occlusion_fraction=0.0,
    )


def build_components(image_height: int):
    coarse_detector = build_nih_traditional_detector()
    coarse_detector.config = replace(
        coarse_detector.config,
        hue_ranges=((88, 101),),
        minimum_radius_px=8.0 * image_height / 960.0,
        maximum_radius_px=300.0 * image_height / 960.0,
    )
    fine_detector = build_nih_fine_detector(image_height)
    fine_detector.config = replace(
        fine_detector.config,
        hue_ranges=((88, 101),),
        target_standoff_mm=TARGET_STANDOFF_MM,
        maximum_outer_diameter_px=600.0 * image_height / 960.0,
        maximum_outer_center_error_px=260.0 * image_height / 960.0,
        minimum_outer_aspect_ratio=0.40,
    )
    fine_servo = build_nih_fine_servo(image_height)
    fine_servo.config = replace(
        fine_servo.config,
        target_standoff_mm=TARGET_STANDOFF_MM,
        lateral_gain=0.25,
        standoff_gain=0.25,
        maximum_lateral_step_mm=0.20,
        maximum_standoff_step_mm=0.30,
    )
    orientation_servo = ActiveEllipseOrientationServo()
    outer_detector = NihOuterEllipseDetector(fine_detector.config)
    outer_servo = ActiveOuterEllipseOrientationServo(
        orientation_servo.config
    )
    gate = StagedAlignmentGate(
        AlignmentThresholds(
            minimum_coarse_confidence=0.35,
            coarse_to_fine_center_error_px=100.0 * image_height / 960.0,
            maximum_optical_outer_error_px=2.5 * image_height / 960.0,
            maximum_outer_inner_concentricity_px=1.5 * image_height / 960.0,
            maximum_lateral_error_mm=0.20,
            maximum_axis_error_deg=6.0,
            maximum_standoff_error_mm=0.70,
            maximum_reprojection_error_px=0.80 * image_height / 960.0,
            required_stable_frames=5,
        )
    )
    return (
        coarse_detector,
        build_nih_coarse_servo(image_height),
        fine_detector,
        fine_servo,
        orientation_servo,
        outer_detector,
        outer_servo,
        gate,
    )


def _calibrate(
    raw: FineRingEstimate | None,
    orientation_servo: ActiveEllipseOrientationServo,
    *,
    standoff_mm: float,
) -> FineRingEstimate | None:
    if raw is None:
        return None
    return calibrated_fine_estimate(
        raw,
        target_standoff_mm=standoff_mm,
        aligned_anisotropy_threshold=(
            orientation_servo.config.aligned_anisotropy_threshold
        ),
        target_anisotropy=orientation_servo.config.target_anisotropy,
        aligned_concentricity_ratio_threshold=(
            orientation_servo.config.aligned_concentricity_ratio_threshold
        ),
        require_orientation_convergence=False,
    )


def _apply_pose_translation(
    plant: PerturbedEyeInHandPlant,
    command_mm: tuple[float, float, float],
    *,
    joint_limit_deg: float = 3.0,
) -> None:
    delta_q = plant.camera_pose_joint_delta(
        translation_camera_mm=command_mm,
        iterations=12,
    )
    plant.apply_joint_delta(delta_q, maximum_joint_step_deg=joint_limit_deg)


def run_episode(
    *,
    plant: Meca500VisualAlignmentPlant,
    episode: int,
    episode_seed: int,
    image_height: int,
    maximum_alignment_steps: int,
    nominal: bool = False,
) -> dict[str, Any]:
    rng = np.random.default_rng(episode_seed)
    sample = (
        nominal_domain()
        if nominal
        else sample_domain(rng, image_height=image_height)
    )
    plant.set_domain_randomization(
        trocar_translation_mm=sample.trocar_translation_mm,
        trocar_rotation_deg_xyz=sample.trocar_rotation_deg_xyz,
        camera_fovy_scale=sample.camera_fovy_scale,
        trocar_rgb_scale=sample.trocar_rgb_scale,
        sclera_rgb_scale=sample.sclera_rgb_scale,
    )
    initial_q = (
        plant.SEARCH_Q_DEG
        + np.asarray(sample.initial_joint_delta_deg, dtype=np.float64)
    )
    plant.reset(initial_q)
    perturbed = PerturbedEyeInHandPlant(plant, sample=sample, rng=rng)
    (
        coarse_detector,
        coarse_servo,
        fine_detector,
        fine_servo,
        orientation_servo,
        outer_detector,
        outer_servo,
        gate,
    ) = build_components(image_height)
    insertion = TraditionalInsertionHandoffController()

    started = time.perf_counter()
    tool_positions = [plant.tool_position_world()]
    center_history: list[float] = []
    phase_counts: Counter[str] = Counter()
    failure_reasons: Counter[str] = Counter()
    target_acquired = False
    coarse_success = False
    fine_success = False
    target_lost_frames = 0
    outer_residual_applied = False
    alignment_steps = 0
    final_estimate: FineRingEstimate | None = None
    final_decision = None

    for step in range(maximum_alignment_steps):
        alignment_steps = step + 1
        image = perturbed.capture_rgb()
        coarse = coarse_detector.detect(image)
        if coarse is None:
            target_lost_frames += 1
        else:
            target_acquired = True
        fine_raw = None if coarse is None else fine_detector.detect(image)
        outer_raw = None if coarse is None else outer_detector.detect(image)
        final_estimate = _calibrate(
            fine_raw,
            orientation_servo,
            standoff_mm=TARGET_STANDOFF_MM,
        )
        final_decision = gate.update(
            coarse=None if coarse is None else coarse.observation,
            fine=(
                None
                if final_estimate is None
                else final_estimate.observation
            ),
        )
        phase_counts[final_decision.phase.value] += 1
        failure_reasons.update(final_decision.reasons)
        if coarse is not None:
            center_history.append(float(coarse.observation.center_error_px))
            coarse_success = coarse_success or (
                coarse.observation.center_error_px
                <= gate.thresholds.coarse_to_fine_center_error_px
            )
        if final_decision.insertion_handoff_ready:
            fine_success = True
            break
        if final_decision.stable_frames > 0:
            continue
        if coarse is None:
            continue
        if (
            coarse.observation.center_error_px
            > gate.thresholds.coarse_to_fine_center_error_px
        ):
            command = coarse_servo.command(coarse)
            camera_command = np.asarray(
                (*command.camera_xy_mm, 0.0),
                dtype=np.float64,
            )
            _apply_pose_translation(perturbed, tuple(camera_command))
            tool_positions.append(plant.tool_position_world())
            continue
        if outer_raw is not None and not outer_servo.is_aligned(outer_raw):
            if outer_raw.center_error_px > 2.5 * image_height / 960.0:
                command = coarse_servo.command(coarse)
                camera_command = np.asarray(
                    (*command.camera_xy_mm, 0.0),
                    dtype=np.float64,
                )
                _apply_pose_translation(perturbed, tuple(camera_command))
                tool_positions.append(plant.tool_position_world())
                continue
            outer_command = outer_servo.command(
                plant=perturbed,
                detector=outer_detector,
                baseline_estimate=outer_raw,
            )
            if outer_command is not None:
                rotation = np.asarray(
                    (*outer_command.camera_rotation_xy_deg, 0.0),
                    dtype=np.float64,
                )
                if np.linalg.norm(rotation) > 1e-9:
                    delta_q = plant.camera_pose_joint_delta(
                        rotation_camera_deg=tuple(rotation),
                    )
                    plant.apply_joint_delta(
                        delta_q,
                        maximum_joint_step_deg=10.0,
                    )
                    tool_positions.append(plant.tool_position_world())
            continue
        if outer_raw is not None and not outer_residual_applied:
            correction = (
                outer_servo.config.outer_residual_calibration_rotation_xy_deg
            )
            delta_q = plant.camera_pose_joint_delta(
                rotation_camera_deg=(*correction, 0.0),
                iterations=20,
            )
            plant.apply_joint_delta(delta_q, maximum_joint_step_deg=25.0)
            tool_positions.append(plant.tool_position_world())
            outer_residual_applied = True
            continue
        if fine_raw is None:
            continue
        fine_command = fine_servo.command(fine_raw.observation)
        camera_command = np.asarray(
            fine_command.camera_xyz_mm,
            dtype=np.float64,
        )
        _apply_pose_translation(perturbed, tuple(camera_command))
        tool_positions.append(plant.tool_position_world())

    alignment_positions = np.asarray(tool_positions, dtype=np.float64)
    alignment_displacements = np.diff(alignment_positions, axis=0)
    alignment_path_length_mm = float(
        np.linalg.norm(alignment_displacements, axis=1).sum() * 1000.0
    )
    alignment_reversals = sum(
        int(float(np.dot(a, b)) < 0.0)
        for a, b in zip(
            alignment_displacements,
            alignment_displacements[1:],
        )
        if np.linalg.norm(a) > 1e-9 and np.linalg.norm(b) > 1e-9
    )

    insertion_complete = False
    insertion_stop_reason = "alignment_not_authorized"
    insertion_steps = 0
    extension_mm = 0.0
    clearance_samples: list[ClearanceSample] = []
    insertion_origin = None
    insertion_axis = None

    if fine_success:
        insertion_origin = plant.tool_position_world()
        insertion_axis = plant.tool_insertion_axis_world()
        insertion_gate = StagedAlignmentGate(
            replace(
                gate.thresholds,
                maximum_optical_outer_error_px=(
                    16.0 * image_height / 960.0
                ),
                maximum_outer_inner_concentricity_px=(
                    3.0 * image_height / 960.0
                ),
                required_stable_frames=1,
            )
        )
        for insertion_step in range(80):
            insertion_steps = insertion_step + 1
            image = perturbed.capture_rgb()
            coarse = coarse_detector.detect(image)
            fine_raw = (
                None if coarse is None else fine_detector.detect(image)
            )
            estimate = _calibrate(
                fine_raw,
                orientation_servo,
                standoff_mm=TARGET_STANDOFF_MM - extension_mm,
            )
            visual = insertion_gate.update(
                coarse=None if coarse is None else coarse.observation,
                fine=None if estimate is None else estimate.observation,
            )
            decision = insertion.decide(
                insertion_handoff_ready=visual.insertion_handoff_ready,
                fine=None if estimate is None else estimate.observation,
                current_extension_mm=extension_mm,
            )
            insertion_stop_reason = decision.reason
            if estimate is not None:
                clearance_samples.append(
                    ClearanceSample(
                        lateral_error_mm=(
                            estimate.observation.lateral_error_mm
                        ),
                        insertion_depth_mm=decision.insertion_depth_mm,
                        axis_error_deg=estimate.observation.axis_error_deg,
                        uncertainty_margin_mm=(
                            insertion.config.uncertainty_margin_mm
                        ),
                    )
                )
            if decision.complete:
                insertion_complete = True
                break
            if not decision.allow_motion:
                break
            plant.move_tool_along_axis_mm(decision.commanded_step_mm)
            tool_positions.append(plant.tool_position_world())
            extension_mm = float(
                np.dot(
                    plant.tool_position_world() - insertion_origin,
                    insertion_axis,
                )
                * 1000.0
            )

    contact = plant.wall_contact_metrics()
    terminal = clearance_samples[-1] if clearance_samples else None
    clearance = (
        None
        if terminal is None
        else insertion.clearance_contract.evaluate_episode(
            clearance_samples,
            terminal_sample=terminal,
            target_insert_depth_mm=insertion.config.target_wall_traversal_mm,
            max_contact_force_n=float(contact["maximum_normal_force_n"]),
            wall_contact_steps=int(contact["wall_contact_count"]),
            full_trajectory_available=True,
        )
    )
    all_positions = np.asarray(tool_positions, dtype=np.float64)
    total_tcp_path_length_mm = float(
        np.linalg.norm(np.diff(all_positions, axis=0), axis=1).sum() * 1000.0
    )
    final_truth = plant.evaluation_pose_errors()
    final_metrics = (
        {} if final_decision is None else dict(final_decision.metrics)
    )
    full_flow_success = bool(
        insertion_complete
        and clearance is not None
        and clearance.certified_success
    )
    return {
        "episode": episode,
        "episode_seed": episode_seed,
        **asdict(sample),
        "target_acquired": target_acquired,
        "coarse_success": coarse_success,
        "fine_success": fine_success,
        "outer_residual_calibration_applied": outer_residual_applied,
        "full_flow_success": full_flow_success,
        "alignment_steps": alignment_steps,
        "insertion_steps": insertion_steps,
        "insertion_extension_mm": extension_mm,
        "alignment_path_length_mm": alignment_path_length_mm,
        "total_tcp_path_length_mm": total_tcp_path_length_mm,
        "alignment_reversal_count": alignment_reversals,
        "target_lost_frames": target_lost_frames,
        "occluded_frames": perturbed.occluded_frames,
        "phase_counts": dict(phase_counts),
        "failure_reasons": dict(failure_reasons),
        "insertion_stop_reason": insertion_stop_reason,
        "final_visual_metrics": final_metrics,
        "final_evaluation_only_pose_errors": final_truth,
        "contact": contact,
        "clearance_contract": (
            None if clearance is None else clearance.to_dict()
        ),
        "elapsed_s": time.perf_counter() - started,
        "center_overshoot_px": (
            None
            if not center_history
            else float(max(center_history) - min(center_history))
        ),
        "terminal_center_jitter_px": (
            None
            if len(center_history) < 5
            else float(np.std(center_history[-5:]))
        ),
    }


def _initialize_worker(
    width: int,
    height: int,
    settle_steps: int,
    maximum_alignment_steps: int,
) -> None:
    global _WORKER_PLANT
    global _WORKER_HEIGHT
    global _WORKER_MAXIMUM_ALIGNMENT_STEPS
    _WORKER_PLANT = Meca500VisualAlignmentPlant(
        image_size_px=(width, height),
        settle_steps=settle_steps,
    )
    _WORKER_HEIGHT = height
    _WORKER_MAXIMUM_ALIGNMENT_STEPS = maximum_alignment_steps


def _run_worker_episode(payload: tuple[int, int, bool]) -> dict[str, Any]:
    episode, episode_seed, nominal = payload
    if _WORKER_PLANT is None:
        raise RuntimeError("M5 worker plant was not initialized.")
    return run_episode(
        plant=_WORKER_PLANT,
        episode=episode,
        episode_seed=episode_seed,
        image_height=_WORKER_HEIGHT,
        maximum_alignment_steps=_WORKER_MAXIMUM_ALIGNMENT_STEPS,
        nominal=nominal,
    )


def summarize(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    def rate(key: str) -> float:
        return float(np.mean([bool(item[key]) for item in episodes]))

    successes = [item for item in episodes if item["full_flow_success"]]
    failure_counter: Counter[str] = Counter()
    for item in episodes:
        if item["full_flow_success"]:
            continue
        failure_counter.update(item["failure_reasons"])
        if item["insertion_stop_reason"] != "target_insertion_depth_reached":
            failure_counter[item["insertion_stop_reason"]] += 1

    def values(path: tuple[str, ...], source=episodes) -> list[float]:
        output: list[float] = []
        for item in source:
            value: Any = item
            for key in path:
                if value is None:
                    break
                value = value.get(key)
            if value is not None and math.isfinite(float(value)):
                output.append(float(value))
        return output

    lateral = values(("final_evaluation_only_pose_errors", "lateral_error_mm"))
    axis = values(("final_evaluation_only_pose_errors", "axis_error_deg"))
    clearance = values(
        ("clearance_contract", "minimum_robust_clearance_mm"),
        successes,
    )
    return {
        "target_acquisition_rate": rate("target_acquired"),
        "coarse_success_rate": rate("coarse_success"),
        "fine_success_rate": rate("fine_success"),
        "full_flow_success_rate": rate("full_flow_success"),
        "episodes": len(episodes),
        "successful_episodes": len(successes),
        "mean_alignment_steps": float(
            np.mean([item["alignment_steps"] for item in episodes])
        ),
        "mean_alignment_path_length_mm": float(
            np.mean([item["alignment_path_length_mm"] for item in episodes])
        ),
        "mean_alignment_reversal_count": float(
            np.mean([item["alignment_reversal_count"] for item in episodes])
        ),
        "mean_elapsed_s": float(
            np.mean([item["elapsed_s"] for item in episodes])
        ),
        "final_lateral_error_mm": {
            "mean": float(np.mean(lateral)),
            "p95": float(np.quantile(lateral, 0.95)),
            "max": float(np.max(lateral)),
        },
        "final_axis_error_deg": {
            "mean": float(np.mean(axis)),
            "p95": float(np.quantile(axis, 0.95)),
            "max": float(np.max(axis)),
        },
        "minimum_robust_clearance_mm": (
            None if not clearance else float(np.min(clearance))
        ),
        "total_wall_contact_steps": int(
            sum(item["contact"]["wall_contact_count"] for item in episodes)
        ),
        "over_insert_count": int(
            sum(
                item["insertion_extension_mm"] > 12.75
                for item in episodes
            )
        ),
        "failure_reason_counts": dict(failure_counter.most_common()),
    }


def write_outputs(
    *,
    output_dir: Path,
    episodes: list[dict[str, Any]],
    seed: int,
    width: int,
    height: int,
) -> tuple[Path, Path]:
    summary = summarize(episodes)
    report = {
        "timestamp": datetime.now().isoformat(),
        "status": "M5_frozen_six_axis_baseline",
        "official_m5_frozen_baseline": len(episodes) >= 500,
        "episodes": len(episodes),
        "seed": seed,
        "image_size_px": [width, height],
        "anatomy": NIH_HRA_EYE_SOURCE,
        "eye_orientation": "upright",
        "nominal_trocar_tilt_deg": TROCAR_TILT_DEG,
        "controller_inputs": [
            "perturbed_eye_in_hand_rgb",
            "inner_outer_ellipse_geometry",
            "six_joint_state",
            "commanded_insertion_extension",
        ],
        "privileged_truth_used_for_control": False,
        "privileged_truth_use": "episode evaluation and aggregation only",
        "randomization_contract": {
            "initial_joint_delta_deg": [-1.5, 1.5],
            "trocar_translation_mm_xyz": [
                [-0.60, 0.60],
                [-0.60, 0.60],
                [-0.35, 0.35],
            ],
            "trocar_rotation_deg_xyz": [-2.0, 2.0],
            "camera_fovy_scale": [0.985, 1.015],
            "principal_point_shift_px_at_960p": [-2.0, 2.0],
            "rgb_gain": [0.92, 1.08],
            "rgb_noise_std": [0.0, 2.0],
            "gaussian_blur_kernel": [1, 3],
            "material_rgb_scale": [0.92, 1.08],
            "per_frame_occlusion_probability": [0.0, 0.025],
            "occlusion_fraction_of_short_side": [0.05, 0.12],
        },
        "summary": summary,
        "episode_results": episodes,
        "limitations": [
            "Simulation-only frozen baseline; independent physical camera images remain required.",
            "The joint-zero to observation-waypoint motion is outside this alignment benchmark.",
            "The fixed ellipse residual calibration is frozen and is not refit per episode.",
        ],
    }
    report_path = output_dir / "m5_frozen_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    csv_path = output_dir / "m5_frozen_episodes.csv"
    rows = []
    for item in episodes:
        row = {}
        for key, value in item.items():
            row[key] = (
                json.dumps(value, ensure_ascii=False)
                if isinstance(value, (dict, list, tuple))
                else value
            )
        rows.append(row)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return report_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="M5 frozen 500-episode six-axis RGB baseline."
    )
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--settle-steps", type=int, default=100)
    parser.add_argument("--maximum-alignment-steps", type=int, default=220)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume missing episode indices from an existing report.",
    )
    parser.add_argument(
        "--nominal",
        action="store_true",
        help="Disable randomization for controller regression diagnostics.",
    )
    args = parser.parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be positive.")
    if args.workers < 1:
        raise ValueError("--workers must be positive.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    master_rng = np.random.default_rng(args.seed)
    episode_seeds = [
        int(value)
        for value in master_rng.integers(
            0,
            2**32 - 1,
            size=args.episodes,
        )
    ]
    episodes: list[dict[str, Any]] = []
    report_path_existing = output_dir / "m5_frozen_report.json"
    if args.resume and report_path_existing.exists():
        existing = json.loads(
            report_path_existing.read_text(encoding="utf-8")
        )
        if int(existing["seed"]) != args.seed:
            raise ValueError("Existing report seed does not match --seed.")
        if list(existing["image_size_px"]) != [args.width, args.height]:
            raise ValueError(
                "Existing report image size does not match this run."
            )
        episodes = list(existing.get("episode_results", []))
    completed_indices = {int(item["episode"]) for item in episodes}
    payloads = [
        (index, episode_seed, args.nominal)
        for index, episode_seed in enumerate(episode_seeds)
        if index not in completed_indices
    ]

    def accept(result: dict[str, Any]) -> None:
        episodes.append(result)
        completed = len(episodes)
        print(
            f"[{completed}/{args.episodes}] "
            f"episode={result['episode']} "
            f"acquired={result['target_acquired']} "
            f"aligned={result['fine_success']} "
            f"success={result['full_flow_success']} "
            f"steps={result['alignment_steps']} "
            f"elapsed={result['elapsed_s']:.2f}s",
            flush=True,
        )
        if (
            completed % max(1, args.checkpoint_every) == 0
            or completed == args.episodes
        ):
            episodes.sort(key=lambda item: int(item["episode"]))
            write_outputs(
                output_dir=output_dir,
                episodes=episodes,
                seed=args.seed,
                width=args.width,
                height=args.height,
            )

    if args.workers == 1:
        plant = Meca500VisualAlignmentPlant(
            image_size_px=(args.width, args.height),
            settle_steps=args.settle_steps,
        )
        try:
            for index, episode_seed, nominal in payloads:
                accept(
                    run_episode(
                        plant=plant,
                        episode=index,
                        episode_seed=episode_seed,
                        image_height=args.height,
                        maximum_alignment_steps=(
                            args.maximum_alignment_steps
                        ),
                        nominal=nominal,
                    )
                )
        finally:
            plant.reset_domain()
            plant.close()
    else:
        context = mp.get_context("spawn")
        with context.Pool(
            processes=args.workers,
            initializer=_initialize_worker,
            initargs=(
                args.width,
                args.height,
                args.settle_steps,
                args.maximum_alignment_steps,
            ),
        ) as pool:
            for result in pool.imap_unordered(
                _run_worker_episode,
                payloads,
                chunksize=1,
            ):
                accept(result)

    episodes.sort(key=lambda item: int(item["episode"]))
    report_path, csv_path = write_outputs(
        output_dir=output_dir,
        episodes=episodes,
        seed=args.seed,
        width=args.width,
        height=args.height,
    )
    print(json.dumps(summarize(episodes), ensure_ascii=False, indent=2))
    print(f"Report: {report_path}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
