from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from real_3d_alignment.mujoco_visual_env import MujocoCoarseAlignmentPlant
from real_3d_alignment.nih_baseline import (
    NIH_HRA_EYE_SOURCE,
    NIH_HRA_SCENE,
    build_nih_coarse_gate,
    build_nih_coarse_servo,
    build_nih_traditional_detector,
)
from real_3d_alignment.staged_alignment import AlignmentDecision
from real_3d_alignment.visual_loop import StagedVisualAlignmentLoop, VisualLoopRecord
from yolo_perception.coarse_detector import (
    YoloCoarseDetector,
    YoloCoarseDetectorConfig,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_XML = NIH_HRA_SCENE
DEFAULT_OUTPUT = ROOT / "output" / "mujoco_coarse_alignment_m1_nih_hra"


def annotate(
    image_rgb: np.ndarray,
    *,
    step: int,
    center: tuple[float, float] | None,
    error_px: float | None,
    lateral_error_mm: float,
    phase: str,
    detector_name: str,
) -> np.ndarray:
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    height, width = image.shape[:2]
    cv2.drawMarker(
        image,
        (width // 2, height // 2),
        (255, 220, 40),
        cv2.MARKER_CROSS,
        24,
        2,
    )
    if center is not None:
        cv2.circle(image, tuple(int(round(value)) for value in center), 7, (40, 255, 40), 2)
    cv2.rectangle(image, (8, 8), (510, 108), (20, 20, 20), -1)
    lines = [
        f"RGB-driven {detector_name} coarse alignment",
        f"step={step} phase={phase}",
        f"pixel_error={error_px if error_px is not None else 'missing'}",
        f"privileged_eval_lateral={lateral_error_mm:.3f} mm",
    ]
    for index, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (18, 31 + 23 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
    return image


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an RGB-driven MuJoCo coarse-alignment baseline."
    )
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--initial-camera-x-mm", type=float, default=7.0)
    parser.add_argument("--initial-camera-y-mm", type=float, default=-5.0)
    parser.add_argument("--maximum-steps", type=int, default=80)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument(
        "--detector",
        choices=("traditional", "yolo"),
        default="traditional",
    )
    parser.add_argument("--yolo-model", type=Path, default=None)
    parser.add_argument(
        "--yolo-target-classes",
        default="0,1",
        help="Comma-separated YOLO class ids accepted as the trocar ROI.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plant = MujocoCoarseAlignmentPlant(
        args.xml,
        image_size_px=(args.width, args.height),
        initial_camera_xy_mm=(
            args.initial_camera_x_mm,
            args.initial_camera_y_mm,
        ),
    )
    if args.detector == "traditional":
        detector = build_nih_traditional_detector()
    else:
        if args.yolo_model is None:
            plant.close()
            raise ValueError("--yolo-model is required when --detector=yolo.")
        target_classes = tuple(
            int(value.strip())
            for value in args.yolo_target_classes.split(",")
            if value.strip()
        )
        detector = YoloCoarseDetector.from_weights(
            args.yolo_model,
            YoloCoarseDetectorConfig(
                target_class_ids=target_classes,
                minimum_confidence=0.25,
                image_size=max(args.width, args.height),
            ),
        )
    servo = build_nih_coarse_servo(args.height)
    gate = build_nih_coarse_gate(transition_center_error_px=12.0)

    video_path = output_dir / f"m2_{args.detector}_coarse_alignment.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (args.width, args.height),
    )
    if not writer.isOpened():
        plant.close()
        raise RuntimeError(f"Could not open video writer: {video_path}")

    evaluation_records: list[dict] = []

    def observe_step(
        image_rgb: np.ndarray,
        detection,
        command,
        decision: AlignmentDecision,
        record: VisualLoopRecord,
    ) -> None:
        lateral_error = plant.evaluation_lateral_error_mm()
        frame = annotate(
            image_rgb,
            step=record.step,
            center=(
                None if detection is None else detection.observation.target_center_px
            ),
            error_px=record.center_error_px,
            lateral_error_mm=lateral_error,
            phase=decision.phase.value,
            detector_name=args.detector,
        )
        writer.write(frame)
        evaluation_records.append(
            {
                "step": record.step,
                "evaluation_only_lateral_error_mm": lateral_error,
                "detection_details": (
                    None if detection is None else asdict(detection)
                ),
                "command_details": None if command is None else asdict(command),
            }
        )

    loop = StagedVisualAlignmentLoop(
        plant=plant,
        coarse_detector=detector,
        coarse_servo=servo,
        gate=gate,
        step_observer=observe_step,
    )
    try:
        result = loop.run(maximum_steps=args.maximum_steps)
    finally:
        writer.release()
        plant.close()

    milestone = "M1" if args.detector == "traditional" else "M2"
    report = {
        "timestamp": datetime.now().isoformat(),
        "evidence_level": (
            f"{milestone} RGB-driven NIH/HRA MuJoCo coarse-alignment baseline"
        ),
        "anatomy": {
            "source": NIH_HRA_EYE_SOURCE,
            "use": "visual_only",
        },
        "trocar_contract": {
            "outer_diameter_mm": 2.0,
            "inner_diameter_mm": 1.0,
            "wall_length_mm": 2.5,
            "flange_outer_diameter_mm": 2.64,
        },
        "controller_inputs": [
            "eye_in_hand_rgb",
            f"{args.detector}_trocar_detection",
        ],
        "detector": args.detector,
        "yolo_model": None if args.yolo_model is None else str(args.yolo_model),
        "perception_rendering": {
            "shaft_visual_hidden": True,
            "reason": "M1 perception isolation; collision and overview retain shaft",
        },
        "privileged_truth_used_for_control": False,
        "privileged_truth_use": "evaluation_only",
        "stop_reason": result.stop_reason,
        "video": str(video_path),
        "visual_loop": result.to_dict(),
        "evaluation_records": evaluation_records,
    }
    report_path = output_dir / f"{milestone.lower()}_coarse_alignment_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Stop reason: {result.stop_reason}")
    print(f"Steps: {len(result.records)}")
    if result.records:
        print("Final pixel error:", result.records[-1].center_error_px)
        print(
            "Final evaluation-only lateral error mm:",
            evaluation_records[-1]["evaluation_only_lateral_error_mm"],
        )
    print(f"Video: {video_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
