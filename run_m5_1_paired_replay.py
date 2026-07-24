from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import run_m5_frozen_batch as m5
from real_3d_alignment.fine_vision import FineRingEstimate
from real_3d_alignment.insertion_handoff import (
    TraditionalInsertionHandoffController,
)
from real_3d_alignment.meca500_visual_env import Meca500VisualAlignmentPlant
from real_3d_alignment.staged_alignment import StagedAlignmentGate
from real_3d_alignment.temporal_validation import (
    FailClosedReobservationGate,
    ReobservationAction,
    TemporalFineObservationValidator,
    TemporalFineValidationConfig,
)
from single_arm_precision_rl.clearance_contract import ClearanceSample


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = (
    ROOT / "output" / "m5_frozen_500" / "m5_1_replay_manifest.json"
)
DEFAULT_BASELINE = (
    ROOT / "output" / "m5_frozen_500" / "m5_frozen_report.json"
)
DEFAULT_OUTPUT = ROOT / "output" / "m5_1_paired_replay"
_WORKER_PLANT: Meca500VisualAlignmentPlant | None = None
_WORKER_HEIGHT = 0
_WORKER_MAXIMUM_ALIGNMENT_STEPS = 0


def m5_1_temporal_config(image_height: int) -> TemporalFineValidationConfig:
    scale = image_height / 960.0
    return TemporalFineValidationConfig(
        window_size=5,
        minimum_valid_samples=3,
        maximum_center_range_px=1.5 * scale,
        maximum_lateral_range_mm=0.08,
        maximum_axis_range_deg=1.5,
        maximum_standoff_range_mm=0.35,
    )


def _reset_after_motion(
    validator: TemporalFineObservationValidator,
    gate: StagedAlignmentGate,
) -> None:
    validator.reset()
    gate.reset()


def _apply_rotation(
    plant: m5.PerturbedEyeInHandPlant,
    rotation_xy_deg: tuple[float, float],
    *,
    maximum_joint_step_deg: float = 10.0,
) -> bool:
    rotation = np.asarray((*rotation_xy_deg, 0.0), dtype=np.float64)
    if np.linalg.norm(rotation) <= 1e-9:
        return False
    delta_q = plant.camera_pose_joint_delta(
        rotation_camera_deg=tuple(rotation),
        iterations=20,
    )
    plant.apply_joint_delta(
        delta_q,
        maximum_joint_step_deg=maximum_joint_step_deg,
    )
    return True


def run_m5_1_episode(
    *,
    plant: Meca500VisualAlignmentPlant,
    episode: int,
    episode_seed: int,
    image_height: int,
    maximum_alignment_steps: int,
) -> dict[str, Any]:
    """Replay one M5 seed with M5.1 perception and fail-closed insertion."""

    rng = np.random.default_rng(episode_seed)
    sample = m5.sample_domain(
        rng,
        image_height=image_height,
        stratum=m5.RANDOMIZATION_STRATA[
            episode % len(m5.RANDOMIZATION_STRATA)
        ],
    )
    plant.set_domain_randomization(
        trocar_translation_mm=sample.trocar_translation_mm,
        trocar_rotation_deg_xyz=sample.trocar_rotation_deg_xyz,
        camera_fovy_scale=sample.camera_fovy_scale,
        trocar_rgb_scale=sample.trocar_rgb_scale,
        sclera_rgb_scale=sample.sclera_rgb_scale,
        light_intensity_scale=sample.light_intensity_scale,
    )
    initial_q = (
        plant.SEARCH_Q_DEG
        + np.asarray(sample.initial_joint_delta_deg, dtype=np.float64)
    )
    plant.reset(initial_q)
    perturbed = m5.PerturbedEyeInHandPlant(
        plant,
        sample=sample,
        rng=rng,
    )
    (
        coarse_detector,
        coarse_servo,
        fine_detector,
        fine_servo,
        fine_orientation_servo,
        outer_detector,
        outer_orientation_servo,
        gate,
    ) = m5.build_components(image_height)
    insertion = TraditionalInsertionHandoffController()
    temporal = TemporalFineObservationValidator(
        m5_1_temporal_config(image_height)
    )

    started = time.perf_counter()
    tool_positions = [plant.tool_position_world()]
    center_history: list[float] = []
    phase_counts: Counter[str] = Counter()
    failure_reason_frames: Counter[str] = Counter()
    temporal_reason_frames: Counter[str] = Counter()
    target_acquired = False
    coarse_success = False
    target_lost_frames = 0
    alignment_steps = 0
    final_estimate: FineRingEstimate | None = None
    final_decision = None
    fine_success = False
    outer_orientation_steps = 0
    inner_outer_orientation_steps = 0

    for step in range(maximum_alignment_steps):
        alignment_steps = step + 1
        image = perturbed.capture_rgb()
        coarse = coarse_detector.detect(image)
        if coarse is None:
            target_lost_frames += 1
        else:
            target_acquired = True
            center_history.append(
                float(coarse.observation.center_error_px)
            )
            coarse_success = coarse_success or (
                coarse.observation.center_error_px
                <= gate.thresholds.coarse_to_fine_center_error_px
            )
        fine_raw = None if coarse is None else fine_detector.detect(image)
        outer_raw = None if coarse is None else outer_detector.detect(image)
        calibrated = (
            None
            if fine_raw is None
            else m5.calibrated_fine_estimate(
                fine_raw,
                target_standoff_mm=m5.TARGET_STANDOFF_MM,
                aligned_anisotropy_threshold=(
                    fine_orientation_servo.config
                    .aligned_anisotropy_threshold
                ),
                target_anisotropy=(
                    fine_orientation_servo.config.target_anisotropy
                ),
                aligned_concentricity_ratio_threshold=(
                    fine_orientation_servo.config
                    .aligned_concentricity_ratio_threshold
                ),
                require_orientation_convergence=True,
            )
        )
        temporal_result = temporal.update(
            None if calibrated is None else calibrated.observation
        )
        temporal_reason_frames.update(temporal_result.reasons)
        final_estimate = (
            None
            if calibrated is None
            else replace(
                calibrated,
                observation=(
                    calibrated.observation
                    if temporal_result.observation is None
                    else temporal_result.observation
                ),
            )
        )
        final_decision = gate.update(
            coarse=None if coarse is None else coarse.observation,
            fine=(
                temporal_result.observation
                if temporal_result.ready
                else None
            ),
        )
        phase_counts[final_decision.phase.value] += 1
        failure_reason_frames.update(final_decision.reasons)
        if final_decision.insertion_handoff_ready:
            fine_success = True
            break
        if final_decision.stable_frames > 0:
            # Hold the pose while collecting the remaining consecutive
            # authorization frames. Moving here would invalidate the window.
            continue
        if coarse is None:
            temporal.reset()
            continue
        if (
            coarse.observation.center_error_px
            > gate.thresholds.coarse_to_fine_center_error_px
        ):
            command = coarse_servo.command(coarse)
            m5._apply_pose_translation(
                perturbed,
                (*command.camera_xy_mm, 0.0),
            )
            tool_positions.append(plant.tool_position_world())
            _reset_after_motion(temporal, gate)
            continue
        if outer_raw is None:
            temporal.reset()
            continue
        if outer_raw.center_error_px > 2.5 * image_height / 960.0:
            command = coarse_servo.command(coarse)
            m5._apply_pose_translation(
                perturbed,
                (*command.camera_xy_mm, 0.0),
            )
            tool_positions.append(plant.tool_position_world())
            _reset_after_motion(temporal, gate)
            continue
        if not outer_orientation_servo.is_aligned(outer_raw):
            command = outer_orientation_servo.command(
                plant=perturbed,
                detector=outer_detector,
                baseline_estimate=outer_raw,
            )
            if command is not None and _apply_rotation(
                perturbed,
                command.camera_rotation_xy_deg,
            ):
                outer_orientation_steps += 1
                tool_positions.append(plant.tool_position_world())
                _reset_after_motion(temporal, gate)
            continue
        if fine_raw is None:
            temporal.reset()
            continue
        anisotropy = fine_orientation_servo.anisotropy(fine_raw)
        concentricity_ratio = float(
            fine_raw.observation.outer_inner_concentricity_px
            / max(fine_raw.outer_major_diameter_px, 1e-9)
        )
        orientation_converged = bool(
            anisotropy
            <= fine_orientation_servo.config.aligned_anisotropy_threshold
            and concentricity_ratio
            <= fine_orientation_servo.config
            .aligned_concentricity_ratio_threshold
        )
        if not orientation_converged:
            command = fine_orientation_servo.command(
                plant=perturbed,
                detector=fine_detector,
                baseline_estimate=fine_raw,
            )
            if command is not None and _apply_rotation(
                perturbed,
                command.camera_rotation_xy_deg,
            ):
                inner_outer_orientation_steps += 1
                tool_positions.append(plant.tool_position_world())
                _reset_after_motion(temporal, gate)
            continue
        if not temporal_result.ready:
            continue
        translation = fine_servo.command(temporal_result.observation)
        if np.linalg.norm(translation.camera_xyz_mm) > 1e-9:
            m5._apply_pose_translation(
                perturbed,
                translation.camera_xyz_mm,
            )
            tool_positions.append(plant.tool_position_world())
            _reset_after_motion(temporal, gate)

    insertion_complete = False
    insertion_stop_reason = "alignment_not_authorized"
    insertion_steps = 0
    insertion_capture_attempts = 0
    reobservation_holds = 0
    reobservation_recoveries = 0
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
        insertion_temporal = TemporalFineObservationValidator(
            m5_1_temporal_config(image_height)
        )
        reobserve = FailClosedReobservationGate(
            maximum_reobservation_frames=5
        )
        for insertion_step in range(80):
            insertion_steps = insertion_step + 1
            insertion_temporal.reset()
            insertion_gate.reset()
            reobserve.reset()
            decision = None
            for _ in range(8):
                insertion_capture_attempts += 1
                image = perturbed.capture_rgb()
                coarse = coarse_detector.detect(image)
                fine_raw = (
                    None
                    if coarse is None
                    else fine_detector.detect(image)
                )
                estimate = (
                    None
                    if fine_raw is None
                    else m5.calibrated_fine_estimate(
                        fine_raw,
                        target_standoff_mm=(
                            m5.TARGET_STANDOFF_MM - extension_mm
                        ),
                        aligned_anisotropy_threshold=(
                            fine_orientation_servo.config
                            .aligned_anisotropy_threshold
                        ),
                        target_anisotropy=(
                            fine_orientation_servo.config.target_anisotropy
                        ),
                        aligned_concentricity_ratio_threshold=(
                            fine_orientation_servo.config
                            .aligned_concentricity_ratio_threshold
                        ),
                        require_orientation_convergence=True,
                    )
                )
                temporal_result = insertion_temporal.update(
                    None if estimate is None else estimate.observation
                )
                visual = insertion_gate.update(
                    coarse=(
                        None if coarse is None else coarse.observation
                    ),
                    fine=(
                        temporal_result.observation
                        if temporal_result.ready
                        else None
                    ),
                )
                fresh_ready = bool(
                    temporal_result.ready
                    and visual.insertion_handoff_ready
                )
                recovery = reobserve.update(fresh_ready)
                if (
                    recovery.action
                    is ReobservationAction.HOLD_AND_REOBSERVE
                ):
                    reobservation_holds += 1
                    continue
                if recovery.action is ReobservationAction.ABORT:
                    insertion_stop_reason = recovery.reason
                    break
                if recovery.reason.endswith("_recovered"):
                    reobservation_recoveries += 1
                filtered = temporal_result.observation
                decision = insertion.decide(
                    insertion_handoff_ready=True,
                    fine=filtered,
                    current_extension_mm=extension_mm,
                )
                insertion_stop_reason = decision.reason
                if filtered is not None:
                    clearance_samples.append(
                        ClearanceSample(
                            lateral_error_mm=filtered.lateral_error_mm,
                            insertion_depth_mm=decision.insertion_depth_mm,
                            axis_error_deg=filtered.axis_error_deg,
                            uncertainty_margin_mm=(
                                insertion.config.uncertainty_margin_mm
                            ),
                        )
                    )
                break
            if decision is None:
                if not insertion_stop_reason:
                    insertion_stop_reason = (
                        "visual_reobservation_budget_exhausted"
                    )
                break
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
    positions = np.asarray(tool_positions, dtype=np.float64)
    displacements = np.diff(positions, axis=0)
    final_truth = plant.evaluation_pose_errors()
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
        "full_flow_success": full_flow_success,
        "alignment_steps": alignment_steps,
        "insertion_steps": insertion_steps,
        "insertion_capture_attempts": insertion_capture_attempts,
        "insertion_extension_mm": extension_mm,
        "outer_orientation_steps": outer_orientation_steps,
        "inner_outer_orientation_steps": inner_outer_orientation_steps,
        "reobservation_holds": reobservation_holds,
        "reobservation_recoveries": reobservation_recoveries,
        "target_lost_frames": target_lost_frames,
        "occluded_frames": perturbed.occluded_frames,
        "phase_counts": dict(phase_counts),
        "failure_reason_frame_counts": dict(failure_reason_frames),
        "temporal_reason_frame_counts": dict(temporal_reason_frames),
        "terminal_alignment_reasons": (
            []
            if final_decision is None
            else list(final_decision.reasons)
        ),
        "insertion_stop_reason": insertion_stop_reason,
        "final_visual_metrics": (
            {} if final_decision is None else dict(final_decision.metrics)
        ),
        "final_evaluation_only_pose_errors": final_truth,
        "contact": contact,
        "clearance_contract": (
            None if clearance is None else clearance.to_dict()
        ),
        "alignment_path_length_mm": float(
            np.linalg.norm(displacements, axis=1).sum() * 1000.0
        ),
        "elapsed_s": time.perf_counter() - started,
        "controller_inputs": [
            "perturbed_eye_in_hand_rgb",
            "inner_outer_ellipse_geometry",
            "six_joint_state",
            "commanded_insertion_extension",
        ],
        "privileged_truth_used_for_control": False,
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


def _run_worker(payload: tuple[int, int]) -> dict[str, Any]:
    if _WORKER_PLANT is None:
        raise RuntimeError("M5.1 worker plant was not initialized.")
    episode, episode_seed = payload
    return run_m5_1_episode(
        plant=_WORKER_PLANT,
        episode=episode,
        episode_seed=episode_seed,
        image_height=_WORKER_HEIGHT,
        maximum_alignment_steps=_WORKER_MAXIMUM_ALIGNMENT_STEPS,
    )


def paired_summary(
    baseline: dict[str, Any],
    candidate: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_by_episode = {
        int(item["episode"]): item
        for item in baseline["episode_results"]
    }
    pairs = [
        (baseline_by_episode[int(item["episode"])], item)
        for item in candidate
    ]

    def rate(items: list[dict[str, Any]], key: str) -> float:
        return float(np.mean([bool(item[key]) for item in items]))

    baseline_items = [pair[0] for pair in pairs]
    return {
        "paired_episodes": len(pairs),
        "baseline": {
            "fine_success_rate": rate(baseline_items, "fine_success"),
            "full_flow_success_rate": rate(
                baseline_items,
                "full_flow_success",
            ),
        },
        "m5_1": {
            "fine_success_rate": rate(candidate, "fine_success"),
            "full_flow_success_rate": rate(
                candidate,
                "full_flow_success",
            ),
            "wall_contact_steps": int(
                sum(
                    item["contact"]["wall_contact_count"]
                    for item in candidate
                )
            ),
            "over_insert_count": int(
                sum(
                    item["insertion_extension_mm"] > 12.75
                    for item in candidate
                )
            ),
        },
        "transitions": dict(
            Counter(
                (
                    f"baseline_{'success' if old['full_flow_success'] else 'fail'}"
                    f"_to_m5_1_{'success' if new['full_flow_success'] else 'fail'}"
                )
                for old, new in pairs
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="M5.1 same-seed paired perception replay."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=DEFAULT_BASELINE,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--settle-steps", type=int, default=100)
    parser.add_argument("--maximum-alignment-steps", type=int, default=260)
    parser.add_argument("--maximum-cases", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Validate inputs and list paired cases without MuJoCo.",
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    baseline = json.loads(
        args.baseline_report.read_text(encoding="utf-8")
    )
    if (
        manifest["source_protocol_sha256"]
        != baseline["protocol_sha256"]
    ):
        raise ValueError(
            "Replay manifest and M5 baseline protocol hashes differ."
        )
    cases = list(manifest["episodes"])
    if args.maximum_cases > 0:
        cases = cases[: args.maximum_cases]
    payloads = [
        (int(case["episode"]), int(case["episode_seed"]))
        for case in cases
    ]
    baseline_by_episode = {
        int(item["episode"]): item
        for item in baseline["episode_results"]
    }
    missing = [
        episode
        for episode, _ in payloads
        if episode not in baseline_by_episode
    ]
    if missing:
        raise ValueError(
            f"Baseline report lacks replay episodes: {missing}"
        )
    if args.list_only:
        print(
            json.dumps(
                {
                    "source_protocol_sha256": (
                        manifest["source_protocol_sha256"]
                    ),
                    "cases": payloads,
                },
                indent=2,
            )
        )
        return

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    results: list[dict[str, Any]] = []
    with ctx.Pool(
        processes=args.workers,
        initializer=_initialize_worker,
        initargs=(
            args.width,
            args.height,
            args.settle_steps,
            args.maximum_alignment_steps,
        ),
        maxtasksperchild=1,
    ) as pool:
        for result in pool.imap(_run_worker, payloads):
            results.append(result)
            print(
                f"[{len(results)}/{len(payloads)}] "
                f"episode={result['episode']} "
                f"aligned={result['fine_success']} "
                f"success={result['full_flow_success']} "
                f"stop={result['insertion_stop_reason']}",
                flush=True,
            )
            report = {
                "timestamp": datetime.now().astimezone().isoformat(),
                "status": "M5.1_paired_perception_replay",
                "source_m5_protocol_sha256": (
                    manifest["source_protocol_sha256"]
                ),
                "source_m5_report": str(
                    args.baseline_report.resolve()
                ),
                "privileged_truth_used_for_control": False,
                "candidate_changes": [
                    "active_outer_then_inner_outer_orientation",
                    "temporal_fine_observation_validation",
                    "fresh_observation_after_every_insertion_step",
                    "fail_closed_hold_reobserve_or_abort",
                ],
                "summary": paired_summary(baseline, results),
                "episode_results": results,
            }
            (output_dir / "m5_1_paired_report.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )


if __name__ == "__main__":
    mp.freeze_support()
    main()
