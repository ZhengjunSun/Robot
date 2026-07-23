from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

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


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "output" / "m3_nih_hra_randomized_batch"


def numeric_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run randomized M3 NIH/HRA coarse-to-fine episodes."
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--xy-limit-mm", type=float, default=7.0)
    parser.add_argument("--z-limit-mm", type=float, default=3.0)
    parser.add_argument("--maximum-steps", type=int, default=70)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    plant = MujocoCoarseAlignmentPlant(
        NIH_HRA_SCENE,
        image_size_px=(args.width, args.height),
        initial_camera_xy_mm=(0.0, 0.0),
    )
    episodes: list[dict] = []
    try:
        for episode in range(args.episodes):
            xy = rng.uniform(-args.xy_limit_mm, args.xy_limit_mm, size=2)
            z = float(rng.uniform(-args.z_limit_mm, args.z_limit_mm))
            plant.reset((float(xy[0]), float(xy[1])))
            plant.apply_camera_xyz_step((0.0, 0.0, z))
            loop = StagedVisualAlignmentLoop(
                plant=plant,
                coarse_detector=build_nih_traditional_detector(),
                coarse_servo=build_nih_coarse_servo(args.height),
                gate=build_nih_m3_gate(),
                fine_provider=build_nih_fine_detector(args.height),
                fine_servo=build_nih_fine_servo(args.height),
            )
            result = loop.run(maximum_steps=args.maximum_steps)
            metrics = result.final_decision.metrics
            episodes.append(
                {
                    "episode": episode,
                    "initial_x_mm": float(xy[0]),
                    "initial_y_mm": float(xy[1]),
                    "initial_z_offset_mm": z,
                    "success": result.final_decision.insertion_handoff_ready,
                    "stop_reason": result.stop_reason,
                    "steps": len(result.records),
                    "final_fine_center_error_px": metrics.get(
                        "optical_outer_error_px"
                    ),
                    "final_concentricity_px": metrics.get(
                        "outer_inner_concentricity_px"
                    ),
                    "final_lateral_error_mm": metrics.get(
                        "lateral_error_mm"
                    ),
                    "final_axis_error_deg": metrics.get("axis_error_deg"),
                    "final_standoff_error_mm": metrics.get(
                        "standoff_error_mm"
                    ),
                    "final_fit_error_px": metrics.get(
                        "reprojection_error_px"
                    ),
                    "stable_frames": result.final_decision.stable_frames,
                    "target_loss_frames": sum(
                        not record.target_detected
                        for record in result.records
                    ),
                }
            )
    finally:
        plant.close()

    successes = [item for item in episodes if item["success"]]
    metric_keys = (
        "steps",
        "final_fine_center_error_px",
        "final_concentricity_px",
        "final_lateral_error_mm",
        "final_axis_error_deg",
        "final_standoff_error_mm",
        "final_fit_error_px",
        "target_loss_frames",
    )
    report = {
        "timestamp": datetime.now().isoformat(),
        "evidence_level": (
            "M3 randomized NIH/HRA MuJoCo RGB-driven translation baseline"
        ),
        "anatomy": NIH_HRA_EYE_SOURCE,
        "privileged_truth_used_for_control": False,
        "seed": args.seed,
        "episode_count": len(episodes),
        "success_count": len(successes),
        "success_rate": len(successes) / max(len(episodes), 1),
        "randomization": {
            "initial_xy_limit_mm": args.xy_limit_mm,
            "initial_z_limit_mm": args.z_limit_mm,
        },
        "summary_success": {
            key: numeric_summary(
                [
                    float(item[key])
                    for item in successes
                    if item[key] is not None
                ]
            )
            for key in metric_keys
        },
        "failures": [item for item in episodes if not item["success"]],
        "episodes": episodes,
        "limitations": [
            "Randomization covers Cartesian x/y/standoff only.",
            "Rotational actuation and six-axis dynamics remain pending.",
        ],
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "m3_randomized_batch_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"Success: {len(successes)}/{len(episodes)} "
        f"({report['success_rate']:.3%})"
    )
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
