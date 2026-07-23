from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from handeye_common import (
    ROOT,
    load_json,
    project_transform_to_se3,
    rotation_error_deg,
    transform_inverse,
    write_json,
)
from solve_handeye_calibration import METHODS, solve_method


OUTPUT_DIR = ROOT / "outputs"
DEFAULT_EXTRINSIC = ROOT / "config" / "camera_extrinsic_initial.json"


def random_transform(rng: np.random.Generator, translation_center: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    euler = rng.uniform(low=[-50, -35, -70], high=[50, 45, 70], size=3)
    transform[:3, :3] = R.from_euler("xyz", euler, degrees=True).as_matrix()
    transform[:3, 3] = translation_center + rng.normal(0.0, [0.04, 0.04, 0.03], size=3)
    return transform


def make_synthetic_samples(sample_count: int, noise_translation_m: float, noise_rotation_deg: float) -> tuple[list[dict], np.ndarray]:
    rng = np.random.default_rng(20260612)
    extrinsic = load_json(DEFAULT_EXTRINSIC)
    T_link6_camera_true = project_transform_to_se3(np.array(extrinsic["T_link6_camera"], dtype=np.float64))

    T_base_target = np.eye(4)
    T_base_target[:3, :3] = R.from_euler("xyz", [0.0, 0.0, 0.0], degrees=True).as_matrix()
    T_base_target[:3, 3] = np.array([0.18, 0.02, 0.08], dtype=np.float64)

    samples = []
    for index in range(sample_count):
        T_base_link6 = random_transform(rng, np.array([0.14, -0.02, 0.12], dtype=np.float64))
        T_camera_target = transform_inverse(T_link6_camera_true) @ transform_inverse(T_base_link6) @ T_base_target

        if noise_translation_m > 0.0:
            T_camera_target[:3, 3] += rng.normal(0.0, noise_translation_m, size=3)
        if noise_rotation_deg > 0.0:
            noise_R = R.from_rotvec(
                rng.normal(0.0, np.deg2rad(noise_rotation_deg), size=3)
            ).as_matrix()
            T_camera_target[:3, :3] = noise_R @ T_camera_target[:3, :3]

        samples.append(
            {
                "path": Path(f"synthetic_{index:03d}.json"),
                "T_base_link6": T_base_link6,
                "T_camera_target": T_camera_target,
                "mean_reprojection_error_px": None,
            }
        )

    return samples, T_link6_camera_true


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate hand-eye pipeline with synthetic samples.")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--noise-translation-m", type=float, default=0.0)
    parser.add_argument("--noise-rotation-deg", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "handeye_synthetic_validation.json")
    args = parser.parse_args()

    samples, true_transform = make_synthetic_samples(
        sample_count=args.samples,
        noise_translation_m=args.noise_translation_m,
        noise_rotation_deg=args.noise_rotation_deg,
    )

    results = []
    for name, method_id in METHODS.items():
        try:
            result = solve_method(samples, name, method_id)
            estimated = np.array(result["T_link6_camera"], dtype=np.float64)
            result["true_comparison"] = {
                "translation_error_m": float(np.linalg.norm(estimated[:3, 3] - true_transform[:3, 3])),
                "rotation_error_deg": rotation_error_deg(true_transform, estimated),
            }
            results.append(result)
        except Exception as exc:
            results.append({"method": name, "error": f"{type(exc).__name__}: {exc}"})

    valid = [item for item in results if "error" not in item]
    best = min(
        valid,
        key=lambda item: (
            item["true_comparison"]["translation_error_m"],
            item["true_comparison"]["rotation_error_deg"],
        ),
    )
    report = {
        "timestamp": datetime.now().isoformat(),
        "sample_count": args.samples,
        "noise_translation_m": args.noise_translation_m,
        "noise_rotation_deg": args.noise_rotation_deg,
        "true_T_link6_camera": true_transform.tolist(),
        "best_method": best["method"],
        "best_translation_error_m": best["true_comparison"]["translation_error_m"],
        "best_rotation_error_deg": best["true_comparison"]["rotation_error_deg"],
        "results": results,
    }
    write_json(args.output, report)
    print("Synthetic hand-eye validation written:", args.output)
    print("Best method:", report["best_method"])
    print("Translation error [m]:", report["best_translation_error_m"])
    print("Rotation error [deg]:", report["best_rotation_error_deg"])


if __name__ == "__main__":
    main()
