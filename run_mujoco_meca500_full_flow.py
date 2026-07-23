from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from datetime import datetime
from pathlib import Path

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
from real_3d_alignment.scene_contract import (
    CAMERA_CENTER_BGR,
    INNER_CENTER_BGR,
    NEEDLE_VISIBLE_LENGTH_MM,
    OUTER_CENTER_BGR,
    TROCAR_TILT_DEG,
)
from real_3d_alignment.six_axis_visual_servo import (
    ActiveEllipseOrientationServo,
)
from real_3d_alignment.staged_alignment import (
    AlignmentThresholds,
    StagedAlignmentGate,
)
from single_arm_precision_rl.clearance_contract import ClearanceSample


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "output" / "m4_meca500_unified_full_flow"
TARGET_STANDOFF_MM = 22.0


def calibrated_fine_estimate(
    estimate: FineRingEstimate,
    *,
    target_standoff_mm: float,
    aligned_anisotropy_threshold: float,
    target_anisotropy: float,
    require_orientation_convergence: bool = True,
) -> FineRingEstimate:
    """Apply the frozen visual calibration without simulator pose truth."""

    ratio = float(
        estimate.outer_minor_diameter_px
        / max(estimate.outer_major_diameter_px, 1e-9)
    )
    anisotropy = max(0.0, 1.0 - ratio)
    corrected_anisotropy = max(
        0.0,
        anisotropy - float(target_anisotropy),
    )
    axis_error_deg = math.degrees(
        math.acos(float(np.clip(1.0 - corrected_anisotropy, 0.0, 1.0)))
    )
    observation = replace(
        estimate.observation,
        axis_error_deg=axis_error_deg,
        standoff_error_mm=(
            estimate.estimated_depth_mm - float(target_standoff_mm)
        ),
        quality_gate_pass=bool(
            estimate.observation.quality_gate_pass
            and (
                not require_orientation_convergence
                or anisotropy <= aligned_anisotropy_threshold
            )
        ),
    )
    return replace(estimate, observation=observation)


def annotate_eye(
    image_rgb: np.ndarray,
    *,
    phase: str,
    step: int,
    metrics: dict,
    reason: str,
    estimate: FineRingEstimate | None,
    extension_mm: float,
) -> np.ndarray:
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    height, width = image.shape[:2]
    camera = (width // 2, height // 2)
    cv2.drawMarker(image, camera, CAMERA_CENTER_BGR, cv2.MARKER_CROSS, 34, 3)
    cv2.putText(
        image,
        "CAMERA CENTER",
        (camera[0] - 180, camera[1] - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        CAMERA_CENTER_BGR,
        2,
        cv2.LINE_AA,
    )
    if estimate is not None:
        outer = tuple(
            int(round(value)) for value in estimate.observation.outer_center_px
        )
        inner = tuple(
            int(round(value)) for value in estimate.observation.inner_center_px
        )
        cv2.line(image, camera, outer, CAMERA_CENTER_BGR, 2)
        cv2.line(image, outer, inner, (255, 255, 255), 2)
        for center, major, minor, angle, color, label, offset in (
            (
                outer,
                estimate.outer_major_diameter_px,
                estimate.outer_minor_diameter_px,
                estimate.outer_angle_deg,
                OUTER_CENTER_BGR,
                "OUTER CENTER",
                (14, 25),
            ),
            (
                inner,
                estimate.inner_major_diameter_px,
                estimate.inner_minor_diameter_px,
                estimate.inner_angle_deg,
                INNER_CENTER_BGR,
                "INNER CENTER",
                (14, -18),
            ),
        ):
            cv2.ellipse(
                image,
                center,
                (
                    max(1, int(round(0.5 * major))),
                    max(1, int(round(0.5 * minor))),
                ),
                float(angle),
                0,
                360,
                color,
                3,
            )
            cv2.circle(image, center, 6, color, -1)
            cv2.putText(
                image,
                label,
                (center[0] + offset[0], center[1] + offset[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                color,
                2,
                cv2.LINE_AA,
            )
    cv2.rectangle(image, (10, 10), (width - 10, 154), (18, 18, 18), -1)
    lines = (
        "EYE-IN-HAND RGB | CONTROL INPUT",
        f"step={step} phase={phase}",
        (
            "center={:.2f}px lateral={:.3f}mm axis={:.2f}deg".format(
                float(metrics.get("optical_outer_error_px", float("nan"))),
                float(metrics.get("lateral_error_mm", float("nan"))),
                float(metrics.get("axis_error_deg", float("nan"))),
            )
        ),
        f"insertion={extension_mm:.2f}mm | {reason}",
    )
    for index, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (24, 38 + 33 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.64,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )
    return image


def annotate_world(
    image_rgb: np.ndarray,
    *,
    phase: str,
    step: int,
    q_deg: np.ndarray,
    truth: dict[str, float],
    extension_mm: float,
) -> np.ndarray:
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    height, width = image.shape[:2]
    cv2.rectangle(image, (10, 10), (width - 10, 188), (18, 18, 18), -1)
    lines = (
        "FULL MECA500 | SAME MjModel / SAME FRAME",
        (
            f"NIH EYE | TROCAR TILT={TROCAR_TILT_DEG:.0f}deg"
            f" | NEEDLE={NEEDLE_VISIBLE_LENGTH_MM:.0f}mm"
        ),
        f"step={step} phase={phase} insertion={extension_mm:.2f}mm",
        "q(deg)=" + " ".join(f"{value:6.1f}" for value in q_deg),
        (
            "EVAL ONLY: lateral={:.3f}mm axis={:.2f}deg".format(
                truth["lateral_error_mm"],
                truth["axis_error_deg"],
            )
        ),
    )
    for index, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (24, 38 + 33 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.61,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )
    return image


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified full-six-axis RGB alignment and insertion demo."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--maximum-alignment-steps", type=int, default=70)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "m4_meca500_unified_six_axis_full_flow.mp4"
    report_path = output_dir / "m4_meca500_unified_report.json"
    plant = Meca500VisualAlignmentPlant(
        image_size_px=(args.width, args.height),
        settle_steps=100,
    )
    coarse_detector = build_nih_traditional_detector()
    coarse_detector.config = replace(
        coarse_detector.config,
        hue_ranges=((88, 101),),
        minimum_radius_px=8.0 * args.height / 960.0,
        # The same ring grows by more than 2x during the 12.5 mm axial
        # insertion. Keep coarse presence monitoring valid throughout.
        maximum_radius_px=300.0 * args.height / 960.0,
    )
    coarse_servo = build_nih_coarse_servo(args.height)
    fine_detector = build_nih_fine_detector(args.height)
    fine_detector.config = replace(
        fine_detector.config,
        hue_ranges=((88, 101),),
        target_standoff_mm=TARGET_STANDOFF_MM,
        maximum_outer_diameter_px=600.0 * args.height / 960.0,
        maximum_outer_center_error_px=260.0 * args.height / 960.0,
        minimum_outer_aspect_ratio=0.40,
    )
    fine_servo = build_nih_fine_servo(args.height)
    fine_servo.config = replace(
        fine_servo.config,
        target_standoff_mm=TARGET_STANDOFF_MM,
    )
    orientation_servo = ActiveEllipseOrientationServo()
    gate = StagedAlignmentGate(
        AlignmentThresholds(
            minimum_coarse_confidence=0.35,
            # The fine detector remains reliable throughout this ROI. Keeping
            # the controller in fine mode prevents a rotational correction
            # from bouncing the state machine back to coarse alignment.
            coarse_to_fine_center_error_px=100.0 * args.height / 960.0,
            maximum_optical_outer_error_px=2.5 * args.height / 960.0,
            maximum_outer_inner_concentricity_px=1.5 * args.height / 960.0,
            maximum_lateral_error_mm=0.20,
            maximum_axis_error_deg=6.0,
            maximum_standoff_error_mm=0.30,
            maximum_reprojection_error_px=0.80 * args.height / 960.0,
            required_stable_frames=5,
        )
    )
    insertion = TraditionalInsertionHandoffController()
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (2 * args.width, args.height),
    )
    if not writer.isOpened():
        plant.close()
        raise RuntimeError(f"Could not open video writer: {video_path}")

    trajectory: list[dict] = []
    extension_mm = 0.0
    final_estimate: FineRingEstimate | None = None
    final_decision = None

    def record_frame(
        *,
        phase: str,
        step: int,
        metrics: dict,
        reason: str,
        estimate: FineRingEstimate | None,
    ) -> None:
        eye_rgb = plant.capture_rgb()
        world_rgb = plant.capture_overview_rgb()
        truth = plant.evaluation_pose_errors()
        q_deg = plant.joint_positions_deg()
        eye = annotate_eye(
            eye_rgb,
            phase=phase,
            step=step,
            metrics=metrics,
            reason=reason,
            estimate=estimate,
            extension_mm=extension_mm,
        )
        world = annotate_world(
            world_rgb,
            phase=phase,
            step=step,
            q_deg=q_deg,
            truth=truth,
            extension_mm=extension_mm,
        )
        writer.write(np.concatenate((eye, world), axis=1))
        trajectory.append(
            {
                "phase": phase,
                "step": step,
                "extension_mm": extension_mm,
                "joint_positions_deg": q_deg.tolist(),
                "visual_metrics": metrics,
                "evaluation_only_pose_errors": truth,
                "reason": reason,
            }
        )

    stop_reason = "alignment_failed"
    clearance_samples: list[ClearanceSample] = []
    insertion_origin_world: np.ndarray | None = None
    insertion_axis_world: np.ndarray | None = None
    try:
        for step in range(args.maximum_alignment_steps):
            image = plant.capture_rgb()
            coarse = coarse_detector.detect(image)
            fine_raw = (
                None if coarse is None else fine_detector.detect(image)
            )
            final_estimate = (
                None
                if fine_raw is None
                else calibrated_fine_estimate(
                    fine_raw,
                    target_standoff_mm=TARGET_STANDOFF_MM,
                    aligned_anisotropy_threshold=(
                        orientation_servo.config.aligned_anisotropy_threshold
                    ),
                    target_anisotropy=(
                        orientation_servo.config.target_anisotropy
                    ),
                )
            )
            final_decision = gate.update(
                coarse=None if coarse is None else coarse.observation,
                fine=(
                    None
                    if final_estimate is None
                    else final_estimate.observation
                ),
            )
            record_frame(
                phase=final_decision.phase.value,
                step=step,
                metrics=final_decision.metrics,
                reason=final_decision.reasons[0],
                estimate=final_estimate,
            )
            if final_decision.insertion_handoff_ready:
                stop_reason = "visual_alignment_authorized"
                break
            if coarse is None:
                continue
            if (
                coarse.observation.center_error_px
                > gate.thresholds.coarse_to_fine_center_error_px
            ):
                command = coarse_servo.command(coarse)
                plant.apply_camera_translation_mm(
                    (*command.camera_xy_mm, 0.0),
                    maximum_joint_step_deg=3.0,
                )
                continue
            if fine_raw is None:
                continue
            translation = fine_servo.command(fine_raw.observation)
            plant.apply_camera_translation_mm(
                translation.camera_xyz_mm,
                maximum_joint_step_deg=3.0,
            )
            recentered = fine_detector.detect(plant.capture_rgb())
            if recentered is None:
                continue
            orientation = orientation_servo.command(
                plant=plant,
                detector=fine_detector,
                baseline_estimate=recentered,
            )
            if orientation is None:
                continue
            if np.linalg.norm(orientation.camera_rotation_xy_deg) > 1e-9:
                delta_q = plant.camera_pose_joint_delta(
                    rotation_camera_deg=(
                        *orientation.camera_rotation_xy_deg,
                        0.0,
                    )
                )
                plant.apply_joint_delta(
                    delta_q,
                    maximum_joint_step_deg=10.0,
                )

        if final_decision is None or not final_decision.insertion_handoff_ready:
            raise RuntimeError("Six-axis visual alignment did not authorize insertion.")

        insertion_origin_world = plant.tool_position_world()
        insertion_axis_world = plant.tool_insertion_axis_world()
        insertion_gate = StagedAlignmentGate(
            replace(
                gate.thresholds,
                # A fixed pixel threshold becomes artificially tighter as the
                # ring grows during approach. The millimetre lateral limit
                # remains unchanged and is the safety-relevant quantity.
                maximum_optical_outer_error_px=8.0 * args.height / 960.0,
                maximum_outer_inner_concentricity_px=(
                    3.0 * args.height / 960.0
                ),
                required_stable_frames=1,
            )
        )
        for insertion_step in range(80):
            image = plant.capture_rgb()
            coarse = coarse_detector.detect(image)
            fine_raw = (
                None if coarse is None else fine_detector.detect(image)
            )
            expected_standoff = TARGET_STANDOFF_MM - extension_mm
            final_estimate = (
                None
                if fine_raw is None
                else calibrated_fine_estimate(
                    fine_raw,
                    target_standoff_mm=expected_standoff,
                    aligned_anisotropy_threshold=(
                        orientation_servo.config.aligned_anisotropy_threshold
                    ),
                    target_anisotropy=(
                        orientation_servo.config.target_anisotropy
                    ),
                    # Alignment already passed the stricter anisotropy gate.
                    # During axial motion the ring grows in the image, so the
                    # normal axis-error threshold handles scale-dependent
                    # raster variation while geometric fit quality remains
                    # fail-closed.
                    require_orientation_convergence=False,
                )
            )
            final_decision = insertion_gate.update(
                coarse=None if coarse is None else coarse.observation,
                fine=(
                    None
                    if final_estimate is None
                    else final_estimate.observation
                ),
            )
            decision = insertion.decide(
                insertion_handoff_ready=(
                    final_decision.insertion_handoff_ready
                ),
                fine=(
                    None
                    if final_estimate is None
                    else final_estimate.observation
                ),
                current_extension_mm=extension_mm,
            )
            record_frame(
                phase="insertion",
                step=insertion_step,
                metrics=final_decision.metrics,
                reason=decision.reason,
                estimate=final_estimate,
            )
            if final_estimate is not None:
                clearance_samples.append(
                    ClearanceSample(
                        lateral_error_mm=(
                            final_estimate.observation.lateral_error_mm
                        ),
                        insertion_depth_mm=decision.insertion_depth_mm,
                        axis_error_deg=(
                            final_estimate.observation.axis_error_deg
                        ),
                        uncertainty_margin_mm=(
                            insertion.config.uncertainty_margin_mm
                        ),
                    )
                )
            if decision.complete:
                stop_reason = decision.reason
                break
            if not decision.allow_motion:
                stop_reason = decision.reason
                break
            plant.move_tool_along_axis_mm(decision.commanded_step_mm)
            extension_mm = float(
                np.dot(
                    plant.tool_position_world() - insertion_origin_world,
                    insertion_axis_world,
                )
                * 1000.0
            )
        else:
            stop_reason = "maximum_insertion_steps_reached"
    finally:
        writer.release()

    contact = plant.wall_contact_metrics()
    terminal = clearance_samples[-1] if clearance_samples else None
    clearance = (
        None
        if terminal is None
        else insertion.clearance_contract.evaluate_episode(
            clearance_samples,
            terminal_sample=terminal,
            target_insert_depth_mm=(
                insertion.config.target_wall_traversal_mm
            ),
            max_contact_force_n=float(contact["maximum_normal_force_n"]),
            wall_contact_steps=int(contact["wall_contact_count"]),
            full_trajectory_available=True,
        )
    )
    report = {
        "timestamp": datetime.now().isoformat(),
        "evidence_level": (
            "M4 unified full-six-axis MuJoCo RGB alignment and insertion"
        ),
        "anatomy": NIH_HRA_EYE_SOURCE,
        "trocar_tilt_deg": TROCAR_TILT_DEG,
        "needle_visible_length_mm": NEEDLE_VISIBLE_LENGTH_MM,
        "same_model_data_for_both_video_panes": True,
        "controlled_degrees_of_freedom": 6,
        "controller_inputs": [
            "eye_in_hand_rgb",
            "inner_outer_ellipse_geometry",
            "six_joint_state",
            "commanded_insertion_extension",
        ],
        "privileged_truth_used_for_control": False,
        "privileged_truth_use": "report and world-pane evaluation overlay only",
        "stop_reason": stop_reason,
        "final_evaluation_only_pose_errors": plant.evaluation_pose_errors(),
        "contact": contact,
        "clearance_contract": None if clearance is None else clearance.to_dict(),
        "video": str(video_path),
        "trajectory": trajectory,
        "limitations": [
            "This is a MuJoCo simulation result, not a physical-robot validation.",
            "Ellipse anisotropy calibration is specific to this camera, trocar geometry, and render resolution.",
            "MuJoCo contact uses hidden tilted square-wall proxies; analytic clearance uses the circular lumen contract.",
        ],
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    plant.close()
    print(f"Stop reason: {stop_reason}")
    print(f"Final evaluation pose: {report['final_evaluation_only_pose_errors']}")
    print(f"Contact: {contact}")
    print(f"Video: {video_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
