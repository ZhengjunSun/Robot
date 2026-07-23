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
    matrix_to_pose_mm_deg,
    rotation_error_deg,
    transform_inverse,
    write_json,
)


OUTPUT_DIR = ROOT / "outputs"


METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def load_samples(dataset_dir: Path) -> list[dict]:
    samples = []
    for path in sorted(dataset_dir.glob("sample_*.json")):
        sample = load_json(path)
        robot = sample.get("robot", {})
        detection = sample.get("detection", {})
        if not robot.get("T_base_link6") or not detection.get("T_camera_target"):
            continue
        samples.append(
            {
                "path": path,
                "T_base_link6": np.array(robot["T_base_link6"], dtype=np.float64),
                "T_camera_target": np.array(detection["T_camera_target"], dtype=np.float64),
                "mean_reprojection_error_px": detection.get("mean_reprojection_error_px"),
            }
        )
    return samples


def handeye_inputs(samples: list[dict]) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []
    for sample in samples:
        T_base_link6 = sample["T_base_link6"]
        T_camera_target = sample["T_camera_target"]
        R_gripper2base.append(T_base_link6[:3, :3])
        t_gripper2base.append(T_base_link6[:3, 3].reshape(3, 1))
        R_target2cam.append(T_camera_target[:3, :3])
        t_target2cam.append(T_camera_target[:3, 3].reshape(3, 1))
    return R_gripper2base, t_gripper2base, R_target2cam, t_target2cam


def solve_method(samples: list[dict], method_name: str, method_id: int) -> dict:
    Rg, tg, Rt, tt = handeye_inputs(samples)
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        Rg,
        tg,
        Rt,
        tt,
        method=method_id,
    )
    T_link6_camera = np.eye(4)
    T_link6_camera[:3, :3] = R_cam2gripper
    T_link6_camera[:3, 3] = np.asarray(t_cam2gripper, dtype=float).reshape(3)

    base_T_targets = []
    for sample in samples:
        base_T_targets.append(sample["T_base_link6"] @ T_link6_camera @ sample["T_camera_target"])

    translations = np.array([transform[:3, 3] for transform in base_T_targets])
    mean_translation = translations.mean(axis=0)
    translation_errors = np.linalg.norm(translations - mean_translation, axis=1)

    rotations = [transform[:3, :3] for transform in base_T_targets]
    mean_rotation = R.from_matrix(rotations).mean().as_matrix()
    rotation_errors = np.array([
        rotation_error_deg(np.block([[mean_rotation, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]]),
                           np.block([[rot, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]]))
        for rot in rotations
    ])

    return {
        "method": method_name,
        "T_link6_camera": T_link6_camera.tolist(),
        "pose_link6_camera_mm_deg": matrix_to_pose_mm_deg(T_link6_camera),
        "target_consistency": {
            "translation_error_m_mean": float(translation_errors.mean()),
            "translation_error_m_max": float(translation_errors.max()),
            "rotation_error_deg_mean": float(rotation_errors.mean()),
            "rotation_error_deg_max": float(rotation_errors.max()),
        },
    }


def solve(dataset_dir: Path, min_samples: int) -> dict:
    samples = load_samples(dataset_dir)
    if len(samples) < min_samples:
        raise RuntimeError(f"Need at least {min_samples} valid samples, got {len(samples)}")

    results = []
    for name, method_id in METHODS.items():
        try:
            result = solve_method(samples, name, method_id)
            results.append(result)
        except Exception as exc:
            results.append({"method": name, "error": f"{type(exc).__name__}: {exc}"})

    valid_results = [item for item in results if "error" not in item]
    if not valid_results:
        raise RuntimeError("All hand-eye methods failed.")

    best = min(
        valid_results,
        key=lambda item: (
            item["target_consistency"]["translation_error_m_mean"],
            item["target_consistency"]["rotation_error_deg_mean"],
        ),
    )
    return {
        "timestamp": datetime.now().isoformat(),
        "dataset_dir": str(dataset_dir.as_posix()),
        "sample_count": len(samples),
        "best_method": best["method"],
        "best_T_link6_camera": best["T_link6_camera"],
        "best_pose_link6_camera_mm_deg": best["pose_link6_camera_mm_deg"],
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve Meca500 eye-in-hand calibration from collected samples.")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--min-samples", type=int, default=8)
    parser.add_argument("--output-json", type=Path, default=OUTPUT_DIR / "handeye_calibration_result.json")
    parser.add_argument(
        "--output-config",
        type=Path,
        default=ROOT / "config" / "camera_extrinsic_handeye_latest.json",
    )
    args = parser.parse_args()

    report = solve(args.dataset_dir, args.min_samples)
    write_json(args.output_json, report)

    config = {
        "name": "Solved eye-in-hand camera extrinsic",
        "status": "solved_from_real_handeye_samples",
        "source_dataset": str(args.dataset_dir.as_posix()),
        "method": report["best_method"],
        "units": "meters",
        "T_link6_camera": report["best_T_link6_camera"],
        "pose_link6_camera_mm_deg": report["best_pose_link6_camera_mm_deg"],
        "validation": next(item for item in report["results"] if item.get("method") == report["best_method"])["target_consistency"],
    }
    write_json(args.output_config, config)
    print("Hand-eye calibration solved.")
    print("Report:", args.output_json)
    print("Config:", args.output_config)
    print("Best method:", report["best_method"])


if __name__ == "__main__":
    main()
