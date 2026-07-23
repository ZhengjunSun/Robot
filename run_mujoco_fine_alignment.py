from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from real_3d_alignment.mujoco_visual_env import MujocoCoarseAlignmentPlant
from real_3d_alignment.fine_vision import FineRingEstimate
from real_3d_alignment.nih_baseline import (
    M3_TARGET_STANDOFF_MM,
    NIH_HRA_EYE_SOURCE,
    NIH_HRA_SCENE,
    build_nih_coarse_servo,
    build_nih_fine_detector,
    build_nih_fine_servo,
    build_nih_m3_gate,
    build_nih_traditional_detector,
)
from real_3d_alignment.visual_loop import (
    StagedVisualAlignmentLoop,
    VisualLoopRecord,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "output" / "m3_nih_hra_fine_alignment"


def label_frame(
    image_rgb: np.ndarray,
    *,
    title: str,
    record: VisualLoopRecord,
    metrics: dict,
    fine_estimate: FineRingEstimate | None = None,
) -> np.ndarray:
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    height, width = image.shape[:2]
    cv2.drawMarker(
        image,
        (width // 2, height // 2),
        (255, 220, 40),
        cv2.MARKER_CROSS,
        22,
        2,
    )
    if fine_estimate is not None:
        for center, major, minor, angle, color in (
            (
                fine_estimate.observation.outer_center_px,
                fine_estimate.outer_major_diameter_px,
                fine_estimate.outer_minor_diameter_px,
                fine_estimate.outer_angle_deg,
                (0, 255, 0),
            ),
            (
                fine_estimate.observation.inner_center_px,
                fine_estimate.inner_major_diameter_px,
                fine_estimate.inner_minor_diameter_px,
                fine_estimate.inner_angle_deg,
                (0, 180, 255),
            ),
        ):
            cv2.ellipse(
                image,
                tuple(int(round(value)) for value in center),
                (
                    max(1, int(round(0.5 * major))),
                    max(1, int(round(0.5 * minor))),
                ),
                angle,
                0,
                360,
                color,
                2,
            )
    cv2.rectangle(image, (8, 8), (width - 8, 135), (20, 20, 20), -1)
    lines = [
        title,
        f"step={record.step} phase={record.phase} stable={record.stable_frames}/5",
        f"coarse_px={record.center_error_px if record.center_error_px is not None else 'missing'}",
        (
            "fine_px={:.3f} lateral={:.3f}mm".format(
                float(metrics.get("optical_outer_error_px", float("nan"))),
                float(metrics.get("lateral_error_mm", float("nan"))),
            )
        ),
        (
            "axis={:.3f}deg standoff={:.3f}mm fit={:.3f}px".format(
                float(metrics.get("axis_error_deg", float("nan"))),
                float(metrics.get("standoff_error_mm", float("nan"))),
                float(metrics.get("reprojection_error_px", float("nan"))),
            )
        ),
    ]
    for index, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (18, 29 + 24 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
    return image


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run M3 NIH/HRA RGB-driven coarse-to-fine alignment."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--initial-camera-x-mm", type=float, default=7.0)
    parser.add_argument("--initial-camera-y-mm", type=float, default=-5.0)
    parser.add_argument("--maximum-steps", type=int, default=80)
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
    fine_detector = build_nih_fine_detector(args.height)
    video_path = output_dir / "m3_coarse_to_fine_alignment.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (2 * args.width, args.height),
    )
    if not writer.isOpened():
        plant.close()
        raise RuntimeError(f"Could not open video writer: {video_path}")

    metrics_rows: list[dict] = []

    def observe_step(
        image_rgb: np.ndarray,
        detection,
        command,
        decision,
        record: VisualLoopRecord,
    ) -> None:
        metrics = decision.metrics
        estimate = fine_detector.last_estimate
        eye = label_frame(
            image_rgb,
            title="eye-in-hand RGB: coarse + inner/outer fine alignment",
            record=record,
            metrics=metrics,
            fine_estimate=estimate,
        )
        overview = label_frame(
            plant.capture_overview_rgb(),
            title="evaluation overview (not a controller input)",
            record=record,
            metrics=metrics,
        )
        writer.write(np.concatenate([eye, overview], axis=1))
        metrics_rows.append(
            {
                "step": record.step,
                "phase": record.phase,
                "coarse_center_error_px": record.center_error_px,
                "fine_center_error_px": metrics.get(
                    "optical_outer_error_px"
                ),
                "outer_inner_concentricity_px": metrics.get(
                    "outer_inner_concentricity_px"
                ),
                "lateral_error_mm": metrics.get("lateral_error_mm"),
                "axis_error_deg": metrics.get("axis_error_deg"),
                "standoff_error_mm": metrics.get("standoff_error_mm"),
                "ellipse_fit_error_px": metrics.get(
                    "reprojection_error_px"
                ),
                "stable_frames": record.stable_frames,
                "insertion_handoff_ready": record.insertion_handoff_ready,
                "coarse_command_xy_mm": list(
                    record.command_camera_xy_mm
                ),
                "fine_command_xyz_mm": list(
                    record.fine_command_camera_xyz_mm
                ),
                "estimated_depth_mm": (
                    None
                    if estimate is None
                    else estimate.estimated_depth_mm
                ),
                "evaluation_only_lateral_error_mm": (
                    plant.evaluation_lateral_error_mm()
                ),
            }
        )

    loop = StagedVisualAlignmentLoop(
        plant=plant,
        coarse_detector=build_nih_traditional_detector(),
        coarse_servo=build_nih_coarse_servo(args.height),
        gate=build_nih_m3_gate(),
        fine_provider=fine_detector,
        fine_servo=build_nih_fine_servo(args.height),
        step_observer=observe_step,
    )
    try:
        result = loop.run(maximum_steps=args.maximum_steps)
    finally:
        writer.release()
        plant.close()

    report = {
        "timestamp": datetime.now().isoformat(),
        "evidence_level": (
            "M3 RGB-driven NIH/HRA MuJoCo coarse-to-fine translation baseline"
        ),
        "anatomy": NIH_HRA_EYE_SOURCE,
        "target_standoff_mm": M3_TARGET_STANDOFF_MM,
        "controller_inputs": [
            "eye_in_hand_rgb",
            "traditional_outer_ring_detection",
            "inner_outer_ellipse_geometry",
        ],
        "privileged_truth_used_for_control": False,
        "privileged_truth_use": "evaluation and overview only",
        "insertion_executed": False,
        "stop_reason": result.stop_reason,
        "insertion_handoff_ready": (
            result.final_decision.insertion_handoff_ready
        ),
        "video": str(video_path),
        "visual_loop": result.to_dict(),
        "metrics_rows": metrics_rows,
        "limitations": [
            "The lightweight Cartesian M3 plant actuates x/y/standoff translation.",
            "Trocar and camera axes are mechanically pre-aligned in this scene; axis error is visually estimated and enforced as a hard gate but rotational correction is not yet actuated.",
            "The NIH/HRA eye is a visual anatomy layer, not a clinical tissue or collision model.",
        ],
    }
    report_path = output_dir / "m3_fine_alignment_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    csv_path = output_dir / "m3_fine_alignment_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer_csv = csv.DictWriter(
            stream, fieldnames=list(metrics_rows[0])
        )
        writer_csv.writeheader()
        writer_csv.writerows(metrics_rows)

    print(f"Stop reason: {result.stop_reason}")
    print(f"Steps: {len(result.records)}")
    print(f"Insertion handoff ready: {result.final_decision.insertion_handoff_ready}")
    print(f"Final metrics: {json.dumps(result.final_decision.metrics)}")
    print(f"Video: {video_path}")
    print(f"Report: {report_path}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
