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
from real_3d_alignment.mujoco_visual_env import MujocoCoarseAlignmentPlant
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
) -> np.ndarray:
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.rectangle(image, (8, 8), (image.shape[1] - 8, 130), (20, 20, 20), -1)
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
    video_path = output_dir / "m4_rgb_alignment_safe_insertion.mp4"
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

    def write_pair(
        image_rgb: np.ndarray,
        *,
        phase: str,
        step: int,
        metrics: dict,
        reason: str,
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
        )
        world = annotate(
            plant.capture_overview_rgb(),
            title="world/contact evaluation view (not control input)",
            phase=phase,
            step=step,
            metrics=metrics,
            insertion_extension_mm=extension,
            reason=reason,
        )
        writer.write(np.concatenate([eye, world], axis=1))

    def alignment_observer(image, detection, command, decision, record) -> None:
        write_pair(
            image,
            phase=record.phase,
            step=record.step,
            metrics=decision.metrics,
            reason=record.reasons[0],
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
        "privileged_truth_use": "world video and MuJoCo contact evaluation only",
        "stop_reason": full_flow_stop_reason,
        "alignment": alignment_result.to_dict(),
        "clearance_contract": clearance_result.to_dict(),
        "maximum_contact_force_n": maximum_contact_force_n,
        "wall_contact_steps": contact_steps,
        "video": str(video_path),
        "trajectory": trajectory,
        "limitations": [
            "The insertion plant is Cartesian, not the full six-axis robot.",
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
