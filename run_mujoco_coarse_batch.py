from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from real_3d_alignment.mujoco_visual_env import MujocoCoarseAlignmentPlant
from real_3d_alignment.nih_baseline import (
    NIH_HRA_EYE_SOURCE,
    NIH_HRA_SCENE,
    build_nih_coarse_gate,
    build_nih_coarse_servo,
    build_nih_traditional_detector,
)
from real_3d_alignment.staged_alignment import AlignmentPhase
from real_3d_alignment.visual_loop import StagedVisualAlignmentLoop
from yolo_perception.coarse_detector import (
    YoloCoarseDetector,
    YoloCoarseDetectorConfig,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "output" / "mujoco_coarse_batch_m1_nih_hra"


def summarize_numeric(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def reversal_count(commands: list[tuple[float, float]]) -> int:
    reversals = 0
    for previous, current in zip(commands, commands[1:]):
        if float(np.dot(previous, current)) < 0.0:
            reversals += 1
    return reversals


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run randomized NIH/HRA coarse-alignment episodes."
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--offset-limit-mm", type=float, default=7.0)
    parser.add_argument("--maximum-steps", type=int, default=30)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--detector",
        choices=("traditional", "yolo"),
        default="traditional",
    )
    parser.add_argument("--yolo-model", type=Path, default=None)
    parser.add_argument("--yolo-target-classes", default="0")
    args = parser.parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be positive.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    plant = MujocoCoarseAlignmentPlant(
        NIH_HRA_SCENE,
        image_size_px=(args.width, args.height),
        initial_camera_xy_mm=(0.0, 0.0),
    )
    if args.detector == "traditional":
        detector = build_nih_traditional_detector()
    else:
        if args.yolo_model is None:
            plant.close()
            raise ValueError("--yolo-model is required when --detector=yolo.")
        detector = YoloCoarseDetector.from_weights(
            args.yolo_model,
            YoloCoarseDetectorConfig(
                target_class_ids=tuple(
                    int(value.strip())
                    for value in args.yolo_target_classes.split(",")
                    if value.strip()
                ),
                minimum_confidence=0.25,
                image_size=max(args.width, args.height),
            ),
        )
    servo = build_nih_coarse_servo(args.height)
    episodes: list[dict] = []

    try:
        for episode in range(args.episodes):
            initial = tuple(
                float(value)
                for value in rng.uniform(
                    -args.offset_limit_mm,
                    args.offset_limit_mm,
                    size=2,
                )
            )
            plant.reset(initial)
            initial_error_mm = plant.evaluation_lateral_error_mm()
            loop = StagedVisualAlignmentLoop(
                plant=plant,
                coarse_detector=detector,
                coarse_servo=servo,
                gate=build_nih_coarse_gate(transition_center_error_px=8.0),
            )
            result = loop.run(maximum_steps=args.maximum_steps)
            commands = [
                record.command_camera_xy_mm
                for record in result.records
                if float(np.linalg.norm(record.command_camera_xy_mm)) > 0.0
            ]
            success = result.final_decision.phase is AlignmentPhase.FINE
            episodes.append(
                {
                    "episode": episode,
                    "initial_camera_x_mm": initial[0],
                    "initial_camera_y_mm": initial[1],
                    "initial_lateral_error_mm": initial_error_mm,
                    "success": success,
                    "stop_reason": result.stop_reason,
                    "steps": len(result.records),
                    "target_detected_first_frame": result.records[0].target_detected,
                    "target_loss_frames": sum(
                        not record.target_detected for record in result.records
                    ),
                    "initial_center_error_px": result.records[0].center_error_px,
                    "final_center_error_px": result.records[-1].center_error_px,
                    "final_lateral_error_mm": plant.evaluation_lateral_error_mm(),
                    "path_length_camera_mm": float(
                        sum(np.linalg.norm(command) for command in commands)
                    ),
                    "reversal_count": reversal_count(commands),
                }
            )
    finally:
        plant.close()

    successes = [record for record in episodes if record["success"]]
    milestone = "M1" if args.detector == "traditional" else "M2"
    report = {
        "timestamp": datetime.now().isoformat(),
        "evidence_level": (
            f"{milestone} randomized RGB-driven NIH/HRA MuJoCo coarse baseline"
        ),
        "anatomy": NIH_HRA_EYE_SOURCE,
        "privileged_truth_used_for_control": False,
        "detector": args.detector,
        "yolo_model": None if args.yolo_model is None else str(args.yolo_model),
        "seed": args.seed,
        "episode_count": len(episodes),
        "success_count": len(successes),
        "success_rate": len(successes) / len(episodes),
        "configuration": {
            "offset_limit_mm": args.offset_limit_mm,
            "maximum_steps": args.maximum_steps,
            "image_size_px": [args.width, args.height],
            "transition_center_error_px": 8.0,
        },
        "summary": {
            "steps_success": summarize_numeric(
                [float(record["steps"]) for record in successes]
            ),
            "final_center_error_px_success": summarize_numeric(
                [
                    float(record["final_center_error_px"])
                    for record in successes
                    if record["final_center_error_px"] is not None
                ]
            ),
            "final_lateral_error_mm_success": summarize_numeric(
                [float(record["final_lateral_error_mm"]) for record in successes]
            ),
            "path_length_camera_mm_success": summarize_numeric(
                [float(record["path_length_camera_mm"]) for record in successes]
            ),
            "reversal_count_success": summarize_numeric(
                [float(record["reversal_count"]) for record in successes]
            ),
        },
        "failure_count": len(episodes) - len(successes),
        "failures": [record for record in episodes if not record["success"]],
        "episodes": episodes,
    }
    json_path = output_dir / f"{milestone.lower()}_randomized_batch_report.json"
    csv_path = output_dir / f"{milestone.lower()}_randomized_batch_episodes.csv"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(episodes[0]))
        writer.writeheader()
        writer.writerows(episodes)

    print(f"Episodes: {len(episodes)}")
    print(f"Success: {len(successes)} ({report['success_rate']:.3%})")
    print(f"Failures: {report['failure_count']}")
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
