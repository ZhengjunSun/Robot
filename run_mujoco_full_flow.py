from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from real_3d_alignment.insertion_handoff import (
    TraditionalInsertionHandoffController,
)
from real_3d_alignment.meca500_presentation import (
    Meca500PresentationRenderer,
)
from real_3d_alignment.mujoco_visual_env import MujocoCoarseAlignmentPlant
from real_3d_alignment.fine_vision import FineRingEstimate
from real_3d_alignment.nih_baseline import (
    NIH_HRA_EYE_SOURCE,
    NIH_HRA_SCENE,
    build_nih_coarse_servo,
    build_nih_fine_detector,
    build_nih_fine_servo,
    build_nih_m3_gate,
    build_nih_traditional_detector,
)
from real_3d_alignment.visual_loop import StagedVisualAlignmentLoop
from single_arm_precision_rl.clearance_contract import ClearanceSample


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "output" / "m4_nih_hra_full_flow"


def annotate(
    image_rgb: np.ndarray,
    *,
    title: str,
    phase: str,
    step: int,
    metrics: dict,
    insertion_extension_mm: float,
    reason: str,
    fine_estimate: FineRingEstimate | None = None,
    draw_alignment_centers: bool = False,
    compact_header: bool = False,
) -> np.ndarray:
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    height, width = image.shape[:2]
    camera_center = (width // 2, height // 2)
    if draw_alignment_centers:
        camera_color = (255, 255, 0)
        outer_color = (0, 255, 0)
        inner_color = (0, 165, 255)
        cv2.drawMarker(
            image,
            camera_center,
            camera_color,
            cv2.MARKER_CROSS,
            26,
            2,
        )
        cv2.putText(
            image,
            "CAMERA CENTER",
            (camera_center[0] - 145, camera_center[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            camera_color,
            1,
            cv2.LINE_AA,
        )
        if fine_estimate is not None:
            outer_center = tuple(
                int(round(value))
                for value in fine_estimate.observation.outer_center_px
            )
            inner_center = tuple(
                int(round(value))
                for value in fine_estimate.observation.inner_center_px
            )
            cv2.line(image, camera_center, outer_center, camera_color, 1)
            cv2.line(image, outer_center, inner_center, (255, 255, 255), 1)
            for (
                center,
                major,
                minor,
                angle,
                color,
                label,
                label_offset,
            ) in (
                (
                    outer_center,
                    fine_estimate.outer_major_diameter_px,
                    fine_estimate.outer_minor_diameter_px,
                    fine_estimate.outer_angle_deg,
                    outer_color,
                    "OUTER CENTER",
                    (12, 18),
                ),
                (
                    inner_center,
                    fine_estimate.inner_major_diameter_px,
                    fine_estimate.inner_minor_diameter_px,
                    fine_estimate.inner_angle_deg,
                    inner_color,
                    "INNER CENTER",
                    (12, -12),
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
                    2,
                )
                cv2.circle(image, center, 4, color, -1)
                cv2.putText(
                    image,
                    label,
                    (center[0] + label_offset[0], center[1] + label_offset[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    color,
                    1,
                    cv2.LINE_AA,
                )
        cv2.drawMarker(
            image,
            camera_center,
            camera_color,
            cv2.MARKER_CROSS,
            26,
            2,
        )
        cv2.rectangle(
            image,
            (8, height - 37),
            (width - 8, height - 8),
            (20, 20, 20),
            -1,
        )
        legend = (
            ("CAMERA", camera_color),
            ("OUTER", outer_color),
            ("INNER", inner_color),
        )
        for index, (label, color) in enumerate(legend):
            x = 20 + 150 * index
            cv2.circle(image, (x, height - 22), 4, color, -1)
            cv2.putText(
                image,
                label,
                (x + 10, height - 17),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                color,
                1,
                cv2.LINE_AA,
            )
    if compact_header:
        cv2.rectangle(image, (8, 8), (width - 8, 82), (20, 20, 20), -1)
        lines = [
            title,
            f"step={step} phase={phase}",
            f"extension={insertion_extension_mm:.3f}mm",
        ]
    else:
        cv2.rectangle(image, (8, 8), (width - 8, 130), (20, 20, 20), -1)
        lines = [
            title,
            f"step={step} phase={phase}",
            (
                "fine={:.3f}px lateral={:.3f}mm axis={:.3f}deg".format(
                    float(metrics.get("optical_outer_error_px", float("nan"))),
                    float(metrics.get("lateral_error_mm", float("nan"))),
                    float(metrics.get("axis_error_deg", float("nan"))),
                )
            ),
            f"extension={insertion_extension_mm:.3f}mm",
            reason,
        ]
    for index, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (18, 29 + 23 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
    return image


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the M4 NIH/HRA RGB alignment-to-insertion full flow."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--initial-camera-x-mm", type=float, default=7.0)
    parser.add_argument("--initial-camera-y-mm", type=float, default=-5.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plant = MujocoCoarseAlignmentPlant(
        NIH_HRA_SCENE,
        image_size_px=(args.width, args.height),
        initial_camera_xy_mm=(
            args.initial_camera_x_mm,
            args.initial_camera_y_mm,
        ),
    )
    coarse_detector = build_nih_traditional_detector()
    fine_detector = build_nih_fine_detector(args.height)
    gate = build_nih_m3_gate()
    insertion = TraditionalInsertionHandoffController()
    presentation = Meca500PresentationRenderer(
        width=args.width,
        height=args.height,
        target_extension_mm=insertion.target_extension_mm,
    )
    video_path = output_dir / "m4_rgb_alignment_safe_insertion.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (2 * args.width, args.height),
    )
    if not writer.isOpened():
        plant.close()
        presentation.close()
        raise RuntimeError(f"Could not open video writer: {video_path}")

    trajectory: list[dict] = []

    def write_pair(
        image_rgb: np.ndarray,
        *,
        phase: str,
        step: int,
        metrics: dict,
        reason: str,
        fine_estimate: FineRingEstimate | None,
        alignment_progress: float,
    ) -> None:
        extension = plant.insertion_extension_mm()
        eye = annotate(
            image_rgb,
            title="eye-in-hand RGB controller view",
            phase=phase,
            step=step,
            metrics=metrics,
            insertion_extension_mm=extension,
            reason=reason,
            fine_estimate=fine_estimate,
            draw_alignment_centers=True,
        )
        world = annotate(
            presentation.capture_rgb(
                alignment_progress=alignment_progress,
                insertion_extension_mm=extension,
            ),
            title="FULL MECA500 KINEMATIC VIEW (not control input)",
            phase=phase,
            step=step,
            metrics=metrics,
            insertion_extension_mm=extension,
            reason=reason,
            compact_header=True,
        )
        writer.write(np.concatenate([eye, world], axis=1))

    def alignment_observer(image, detection, command, decision, record) -> None:
        write_pair(
            image,
            phase=record.phase,
            step=record.step,
            metrics=decision.metrics,
            reason=record.reasons[0],
            fine_estimate=fine_detector.last_estimate,
            alignment_progress=min(1.0, record.step / 24.0),
        )
        trajectory.append(
            {
                "stage": "alignment",
                "step": record.step,
                "phase": record.phase,
                "insertion_handoff_ready": record.insertion_handoff_ready,
                "metrics": decision.metrics,
                "insertion_extension_mm": plant.insertion_extension_mm(),
            }
        )

    alignment_loop = StagedVisualAlignmentLoop(
        plant=plant,
        coarse_detector=coarse_detector,
        coarse_servo=build_nih_coarse_servo(args.height),
        gate=gate,
        fine_provider=fine_detector,
        fine_servo=build_nih_fine_servo(args.height),
        step_observer=alignment_observer,
    )
    contact_steps = 0
    maximum_contact_force_n = 0.0
    clearance_samples: list[ClearanceSample] = []
    full_flow_stop_reason = "alignment_failed"
    try:
        alignment_result = alignment_loop.run(maximum_steps=80)
        if not alignment_result.final_decision.insertion_handoff_ready:
            raise RuntimeError("Alignment did not authorize insertion.")

        for insertion_step in range(80):
            image = plant.capture_rgb()
            coarse = coarse_detector.detect(image)
            fine = fine_detector.estimate(image) if coarse is not None else None
            visual_decision = gate.update(
                coarse=None if coarse is None else coarse.observation,
                fine=fine,
            )
            step_decision = insertion.decide(
                insertion_handoff_ready=(
                    visual_decision.insertion_handoff_ready
                ),
                fine=fine,
                current_extension_mm=plant.insertion_extension_mm(),
            )
            metrics = visual_decision.metrics
            write_pair(
                image,
                phase="insertion",
                step=insertion_step,
                metrics=metrics,
                reason=step_decision.reason,
                fine_estimate=fine_detector.last_estimate,
                alignment_progress=1.0,
            )
            contact = plant.wall_contact_metrics()
            contact_steps += int(contact["wall_contact_detected"])
            maximum_contact_force_n = max(
                maximum_contact_force_n,
                float(contact["maximum_normal_force_n"]),
            )
            trajectory.append(
                {
                    "stage": "insertion",
                    "step": insertion_step,
                    "visual_decision": {
                        "insertion_handoff_ready": (
                            visual_decision.insertion_handoff_ready
                        ),
                        "reasons": list(visual_decision.reasons),
                        "metrics": metrics,
                    },
                    "insertion_decision": {
                        "allow_motion": step_decision.allow_motion,
                        "complete": step_decision.complete,
                        "commanded_step_mm": step_decision.commanded_step_mm,
                        "insertion_depth_mm": step_decision.insertion_depth_mm,
                        "reason": step_decision.reason,
                        "robust_clearance_mm": (
                            None
                            if step_decision.clearance is None
                            else step_decision.clearance.robust_clearance_mm
                        ),
                    },
                    "insertion_extension_mm": plant.insertion_extension_mm(),
                    "contact": contact,
                }
            )
            if fine is not None:
                clearance_samples.append(
                    ClearanceSample(
                        lateral_error_mm=fine.lateral_error_mm,
                        insertion_depth_mm=step_decision.insertion_depth_mm,
                        axis_error_deg=fine.axis_error_deg,
                        uncertainty_margin_mm=(
                            insertion.config.uncertainty_margin_mm
                        ),
                    )
                )
            if step_decision.complete:
                full_flow_stop_reason = "target_insertion_depth_reached"
                break
            if not step_decision.allow_motion:
                full_flow_stop_reason = step_decision.reason
                break
            plant.apply_insertion_step(step_decision.commanded_step_mm)
        else:
            full_flow_stop_reason = "maximum_insertion_steps_reached"
    finally:
        writer.release()
        plant.close()
        presentation.close()

    terminal_sample = clearance_samples[-1]
    clearance_result = insertion.clearance_contract.evaluate_episode(
        clearance_samples,
        terminal_sample=terminal_sample,
        target_insert_depth_mm=insertion.config.target_wall_traversal_mm,
        max_contact_force_n=maximum_contact_force_n,
        wall_contact_steps=contact_steps,
        full_trajectory_available=True,
    )
    report = {
        "timestamp": datetime.now().isoformat(),
        "evidence_level": (
            "M4 NIH/HRA RGB-driven alignment and deterministic insertion baseline"
        ),
        "anatomy": NIH_HRA_EYE_SOURCE,
        "controller_inputs": [
            "eye_in_hand_rgb",
            "inner_outer_ellipse_geometry",
            "robot_insertion_joint_state",
        ],
        "privileged_truth_used_for_control": False,
        "privileged_truth_use": "MuJoCo contact evaluation and right-pane presentation only",
        "stop_reason": full_flow_stop_reason,
        "alignment": alignment_result.to_dict(),
        "clearance_contract": clearance_result.to_dict(),
        "maximum_contact_force_n": maximum_contact_force_n,
        "wall_contact_steps": contact_steps,
        "video": str(video_path),
        "trajectory": trajectory,
        "limitations": [
            "The insertion plant is Cartesian, not the full six-axis robot.",
            "The right video pane is a synchronized Meca500 kinematic presentation and is not the controller plant.",
            "MuJoCo contact uses a hidden axis-aligned square-wall proxy; analytic clearance uses the circular lumen contract.",
            "Rotational visual correction remains pending; this scene starts with parallel camera and trocar axes.",
        ],
    }
    report_path = output_dir / "m4_full_flow_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Stop reason: {full_flow_stop_reason}")
    print(f"Certified geometric/contact success: {clearance_result.certified_success}")
    print(f"Minimum robust clearance mm: {clearance_result.minimum_robust_clearance_mm}")
    print(f"Wall contact steps: {contact_steps}")
    print(f"Maximum contact force N: {maximum_contact_force_n}")
    print(f"Video: {video_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
