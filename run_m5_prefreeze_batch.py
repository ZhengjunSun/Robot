from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

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
from real_3d_alignment.scene_contract import TROCAR_TILT_DEG
from real_3d_alignment.visual_loop import StagedVisualAlignmentLoop


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "output" / "m5_prefreeze_readiness"


class PhotometricPlant:
    """Deployment-shaped RGB perturbation wrapper around the MuJoCo plant."""

    def __init__(
        self,
        plant: MujocoCoarseAlignmentPlant,
        *,
        gain: float,
        noise_std: float,
        rng: np.random.Generator,
    ):
        self.plant = plant
        self.gain = float(gain)
        self.noise_std = float(noise_std)
        self.rng = rng

    def capture_rgb(self) -> np.ndarray:
        image = self.plant.capture_rgb().astype(np.float32)
        if self.gain != 1.0:
            image *= self.gain
        if self.noise_std > 0.0:
            image += self.rng.normal(0.0, self.noise_std, image.shape)
        return np.clip(image, 0.0, 255.0).astype(np.uint8)

    def apply_camera_xy_step(self, command_mm: tuple[float, float]) -> None:
        self.plant.apply_camera_xy_step(command_mm)

    def apply_camera_xyz_step(
        self,
        command_mm: tuple[float, float, float],
    ) -> None:
        self.plant.apply_camera_xyz_step(command_mm)


def _path_length_mm(records) -> float:
    total = 0.0
    for record in records:
        total += float(np.linalg.norm(record.command_camera_xy_mm))
        total += float(np.linalg.norm(record.fine_command_camera_xyz_mm))
    return total


def _reversal_count(records) -> int:
    commands: list[np.ndarray] = []
    for record in records:
        coarse = np.asarray(
            [*record.command_camera_xy_mm, 0.0],
            dtype=np.float64,
        )
        fine = np.asarray(record.fine_command_camera_xyz_mm, dtype=np.float64)
        command = fine if np.linalg.norm(fine) > 0.0 else coarse
        if np.linalg.norm(command) > 1e-9:
            commands.append(command)
    return sum(
        int(float(np.dot(previous, current)) < 0.0)
        for previous, current in zip(commands, commands[1:])
    )


def run_episode(
    *,
    base_plant: MujocoCoarseAlignmentPlant,
    initial_xy_mm: tuple[float, float],
    gain: float,
    noise_std: float,
    rng: np.random.Generator,
    image_height: int,
) -> dict:
    base_plant.reset(initial_xy_mm)
    plant = PhotometricPlant(
        base_plant,
        gain=gain,
        noise_std=noise_std,
        rng=rng,
    )
    coarse_detector = build_nih_traditional_detector()
    fine_detector = build_nih_fine_detector(image_height)
    gate = build_nih_m3_gate()
    insertion = TraditionalInsertionHandoffController()
    loop = StagedVisualAlignmentLoop(
        plant=plant,
        coarse_detector=coarse_detector,
        coarse_servo=build_nih_coarse_servo(image_height),
        gate=gate,
        fine_provider=fine_detector,
        fine_servo=build_nih_fine_servo(image_height),
    )
    start = time.perf_counter()
    alignment = loop.run(maximum_steps=80)
    acquired = any(record.target_detected for record in alignment.records)
    coarse_success = any(
        record.phase in {"fine_alignment", "aligned"}
        for record in alignment.records
    )
    insertion_steps = 0
    insertion_complete = False
    minimum_clearance_mm: float | None = None
    wall_contact_steps = 0
    maximum_contact_force_n = 0.0
    insertion_stop_reason = "alignment_not_authorized"

    if alignment.final_decision.insertion_handoff_ready:
        for insertion_step in range(80):
            image = plant.capture_rgb()
            coarse = coarse_detector.detect(image)
            fine = fine_detector.estimate(image) if coarse is not None else None
            visual_decision = gate.update(
                coarse=None if coarse is None else coarse.observation,
                fine=fine,
            )
            decision = insertion.decide(
                insertion_handoff_ready=visual_decision.insertion_handoff_ready,
                fine=fine,
                current_extension_mm=base_plant.insertion_extension_mm(),
            )
            insertion_steps = insertion_step + 1
            if decision.clearance is not None:
                clearance = float(decision.clearance.robust_clearance_mm)
                minimum_clearance_mm = (
                    clearance
                    if minimum_clearance_mm is None
                    else min(minimum_clearance_mm, clearance)
                )
            contact = base_plant.wall_contact_metrics()
            wall_contact_steps += int(contact["wall_contact_detected"])
            maximum_contact_force_n = max(
                maximum_contact_force_n,
                float(contact["maximum_normal_force_n"]),
            )
            insertion_stop_reason = decision.reason
            if decision.complete:
                insertion_complete = True
                break
            if not decision.allow_motion:
                break
            base_plant.apply_insertion_step(decision.commanded_step_mm)

    elapsed_s = time.perf_counter() - start
    final_metrics = dict(alignment.final_decision.metrics)
    return {
        "initial_camera_x_mm": initial_xy_mm[0],
        "initial_camera_y_mm": initial_xy_mm[1],
        "rgb_gain": gain,
        "rgb_noise_std": noise_std,
        "target_acquired": acquired,
        "coarse_success": coarse_success,
        "fine_success": alignment.final_decision.insertion_handoff_ready,
        "full_flow_success": bool(
            insertion_complete
            and wall_contact_steps == 0
            and minimum_clearance_mm is not None
            and minimum_clearance_mm
            > insertion.config.minimum_robust_clearance_mm
        ),
        "alignment_stop_reason": alignment.stop_reason,
        "insertion_stop_reason": insertion_stop_reason,
        "alignment_steps": len(alignment.records),
        "insertion_steps": insertion_steps,
        "alignment_path_length_mm": _path_length_mm(alignment.records),
        "alignment_reversal_count": _reversal_count(alignment.records),
        "target_lost_frames": sum(
            int(not record.target_detected) for record in alignment.records
        ),
        "minimum_robust_clearance_mm": minimum_clearance_mm,
        "wall_contact_steps": wall_contact_steps,
        "maximum_contact_force_n": maximum_contact_force_n,
        "elapsed_s": elapsed_s,
        "final_metrics": final_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the pre-freeze M5 readiness batch. This is not the official "
            "500-episode frozen baseline."
        )
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--xy-range-mm", type=float, default=7.0)
    parser.add_argument("--gain-min", type=float, default=0.95)
    parser.add_argument("--gain-max", type=float, default=1.05)
    parser.add_argument("--noise-std-max", type=float, default=2.0)
    args = parser.parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be positive.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    master_rng = np.random.default_rng(args.seed)
    plant = MujocoCoarseAlignmentPlant(
        NIH_HRA_SCENE,
        image_size_px=(args.width, args.height),
        initial_camera_xy_mm=(0.0, 0.0),
    )
    episodes: list[dict] = []
    try:
        for index in range(args.episodes):
            episode_seed = int(master_rng.integers(0, 2**32 - 1))
            rng = np.random.default_rng(episode_seed)
            initial_xy_mm = tuple(
                float(value)
                for value in rng.uniform(
                    -args.xy_range_mm,
                    args.xy_range_mm,
                    size=2,
                )
            )
            result = run_episode(
                base_plant=plant,
                initial_xy_mm=initial_xy_mm,
                gain=float(rng.uniform(args.gain_min, args.gain_max)),
                noise_std=float(rng.uniform(0.0, args.noise_std_max)),
                rng=rng,
                image_height=args.height,
            )
            result["episode"] = index
            result["episode_seed"] = episode_seed
            episodes.append(result)
            print(
                f"[{index + 1}/{args.episodes}] "
                f"success={result['full_flow_success']} "
                f"alignment_steps={result['alignment_steps']}"
            )
    finally:
        plant.close()

    def rate(field: str) -> float:
        return float(np.mean([bool(item[field]) for item in episodes]))

    successful = [item for item in episodes if item["full_flow_success"]]
    summary = {
        "target_acquisition_rate": rate("target_acquired"),
        "coarse_success_rate": rate("coarse_success"),
        "fine_success_rate": rate("fine_success"),
        "full_flow_success_rate": rate("full_flow_success"),
        "mean_alignment_steps": float(
            np.mean([item["alignment_steps"] for item in episodes])
        ),
        "mean_alignment_path_length_mm": float(
            np.mean([item["alignment_path_length_mm"] for item in episodes])
        ),
        "mean_alignment_reversal_count": float(
            np.mean([item["alignment_reversal_count"] for item in episodes])
        ),
        "minimum_robust_clearance_mm": (
            None
            if not successful
            else float(
                min(
                    item["minimum_robust_clearance_mm"]
                    for item in successful
                    if item["minimum_robust_clearance_mm"] is not None
                )
            )
        ),
        "total_wall_contact_steps": int(
            sum(item["wall_contact_steps"] for item in episodes)
        ),
        "maximum_contact_force_n": float(
            max(item["maximum_contact_force_n"] for item in episodes)
        ),
        "mean_elapsed_s": float(
            np.mean([item["elapsed_s"] for item in episodes])
        ),
    }
    report = {
        "timestamp": datetime.now().isoformat(),
        "status": "M5_pre_freeze_readiness_only",
        "official_m5_frozen_baseline": False,
        "episodes": args.episodes,
        "seed": args.seed,
        "anatomy": NIH_HRA_EYE_SOURCE,
        "trocar_tilt_deg": TROCAR_TILT_DEG,
        "randomization": {
            "initial_camera_xy_mm": [-args.xy_range_mm, args.xy_range_mm],
            "rgb_gain": [args.gain_min, args.gain_max],
            "rgb_noise_std": [0.0, args.noise_std_max],
        },
        "controller_inputs": [
            "perturbed_eye_in_hand_rgb",
            "inner_outer_ellipse_geometry",
            "robot_insertion_joint_state",
        ],
        "privileged_truth_used_for_control": False,
        "not_official_m5_reasons": [
            "full six-axis closed-loop control is pending",
            "active rotational visual correction is pending",
            "independent real-camera image validation is pending",
            "material, occlusion, calibration, and variable trocar-pose randomization are not yet covered",
        ],
        "summary": summary,
        "episode_results": episodes,
    }
    report_path = output_dir / "m5_prefreeze_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    csv_path = output_dir / "m5_prefreeze_episodes.csv"
    flat_rows = [
        {
            **{key: value for key, value in item.items() if key != "final_metrics"},
            **{
                f"final_{key}": value
                for key, value in item["final_metrics"].items()
            },
        }
        for item in episodes
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Report: {report_path}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
