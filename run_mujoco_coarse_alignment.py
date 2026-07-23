from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from real_3d_alignment.coarse_vision import (
    CoarseImageBasedVisualServo,
    CoarseServoConfig,
    TraditionalRingDetector,
)
from real_3d_alignment.mujoco_visual_env import MujocoCoarseAlignmentPlant
from real_3d_alignment.staged_alignment import (
    AlignmentDecision,
    AlignmentThresholds,
    StagedAlignmentGate,
)
from real_3d_alignment.visual_loop import StagedVisualAlignmentLoop, VisualLoopRecord


ROOT = Path(__file__).resolve().parent
DEFAULT_XML = (
    ROOT
    / "3d_modeling"
    / "mujoco"
    / "single_arm_trocar_visual_alignment.xml"
)
DEFAULT_OUTPUT = ROOT / "output" / "mujoco_coarse_alignment_m1"


def annotate(
    image_rgb: np.ndarray,
    *,
    step: int,
    center: tuple[float, float] | None,
    error_px: float | None,
    lateral_error_mm: float,
    phase: str,
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
        "M1 RGB-driven traditional coarse alignment",
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
        description="Run the M1 traditional RGB-driven MuJoCo coarse alignment baseline."
    )
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--initial-y-mm", type=float, default=7.0)
    parser.add_argument("--initial-z-mm", type=float, default=-5.0)
    parser.add_argument("--maximum-steps", type=int, default=80)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plant = MujocoCoarseAlignmentPlant(
        args.xml,
        image_size_px=(args.width, args.height),
        initial_lateral_yz_mm=(args.initial_y_mm, args.initial_z_mm),
    )
    detector = TraditionalRingDetector()
    focal_px = 0.5 * args.height / np.tan(np.deg2rad(41.9741) / 2.0)
    servo = CoarseImageBasedVisualServo(
        CoarseServoConfig(
            focal_length_px=float(focal_px),
            maximum_step_mm=1.5,
        )
    )
    gate = StagedAlignmentGate(
        AlignmentThresholds(coarse_to_fine_center_error_px=12.0)
    )

    video_path = output_dir / "m1_traditional_coarse_alignment.mp4"
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

    report = {
        "timestamp": datetime.now().isoformat(),
        "evidence_level": "M1 RGB-driven MuJoCo coarse-alignment baseline",
        "controller_inputs": [
            "eye_in_hand_rgb",
            "traditional_red_ring_detection",
        ],
        "privileged_truth_used_for_control": False,
        "privileged_truth_use": "evaluation_only",
        "stop_reason": result.stop_reason,
        "video": str(video_path),
        "visual_loop": result.to_dict(),
        "evaluation_records": evaluation_records,
    }
    report_path = output_dir / "m1_coarse_alignment_report.json"
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
