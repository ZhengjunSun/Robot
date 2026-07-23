from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from handeye_common import camera_matrix, dist_coeffs, load_json, matrix_from_rvec_tvec, rotation_error_deg, write_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMERA_CONFIG = ROOT / "config" / "handeye_calibration_config.json"
DEFAULT_TROCAR_CONFIG = ROOT / "config" / "trocar_model_initial.json"
DEFAULT_KEYPOINTS = ROOT / "assets" / "trocar" / "trocar_parametric_initial_keypoints.json"
OUTPUT_DIR = ROOT / "outputs" / "trocar_pose_pnp_baseline"


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return ROOT.parent / path


def load_keypoints(path: Path, ring_names: list[str], angles: list[float]) -> dict[str, list[float]]:
    data = load_json(path)
    points = data["points"]
    selected: dict[str, list[float]] = {}
    for ring_name in ring_names:
        for angle in angles:
            key = f"{ring_name}_{int(angle):03d}"
            if key not in points:
                raise KeyError(f"Missing model keypoint: {key}")
            selected[key] = points[key]
    return selected


def transform_to_rvec_tvec(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rvec = cv2.Rodrigues(transform[:3, :3])[0].reshape(3, 1)
    tvec = transform[:3, 3].reshape(3, 1)
    return rvec, tvec


def safe_imwrite(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise RuntimeError(f"Failed to encode image for {path}")
    encoded.tofile(str(path))


def make_synthetic_observation(model_points: dict[str, list[float]], camera_cfg: dict, noise_px: float) -> tuple[dict, np.ndarray]:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = R.from_euler("XYZ", [12.0, -9.0, 18.0], degrees=True).as_matrix()
    transform[:3, 3] = np.array([0.010, -0.012, 0.220], dtype=np.float64)
    rvec, tvec = transform_to_rvec_tvec(transform)

    names = list(model_points.keys())
    object_points = np.array([model_points[name] for name in names], dtype=np.float64)
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix(camera_cfg), dist_coeffs(camera_cfg))
    image_points = projected.reshape(-1, 2)
    if noise_px > 0.0:
        rng = np.random.default_rng(42)
        image_points = image_points + rng.normal(0.0, noise_px, size=image_points.shape)
    return {name: image_points[index].tolist() for index, name in enumerate(names)}, transform


def load_annotation(path: Path) -> tuple[dict[str, list[float]], Path | None]:
    data = load_json(path)
    image_path = data.get("image")
    resolved_image = resolve_path(Path(image_path)) if image_path else None
    points = data.get("points") or data.get("image_points")
    if not points:
        raise ValueError("Annotation JSON must contain 'points' or 'image_points'.")
    return points, resolved_image


def solve_pose(model_points: dict[str, list[float]], image_points: dict[str, list[float]], camera_cfg: dict) -> tuple[np.ndarray, dict]:
    common_names = [name for name in model_points.keys() if name in image_points and image_points[name] is not None]
    if len(common_names) < 4:
        raise ValueError(f"Need at least 4 valid 2D-3D correspondences; got {len(common_names)}.")

    object_array = np.array([model_points[name] for name in common_names], dtype=np.float64)
    image_array = np.array([image_points[name] for name in common_names], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(
        object_array,
        image_array,
        camera_matrix(camera_cfg),
        dist_coeffs(camera_cfg),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError("cv2.solvePnP failed.")

    transform = matrix_from_rvec_tvec(rvec, tvec)
    reprojected, _ = cv2.projectPoints(object_array, rvec, tvec, camera_matrix(camera_cfg), dist_coeffs(camera_cfg))
    reprojection_errors = np.linalg.norm(reprojected.reshape(-1, 2) - image_array, axis=1)
    metrics = {
        "used_keypoints": common_names,
        "used_keypoint_count": len(common_names),
        "mean_reprojection_error_px": float(np.mean(reprojection_errors)),
        "max_reprojection_error_px": float(np.max(reprojection_errors)),
    }
    return transform, metrics


def draw_overlay(
    image_path: Path | None,
    image_points: dict[str, list[float]],
    model_points: dict[str, list[float]],
    transform: np.ndarray,
    camera_cfg: dict,
    output_path: Path,
) -> None:
    width = int(camera_cfg["camera"]["image_width"])
    height = int(camera_cfg["camera"]["image_height"])
    image = cv2.imread(str(image_path)) if image_path and image_path.exists() else np.full((height, width, 3), 245, dtype=np.uint8)

    rvec, tvec = transform_to_rvec_tvec(transform)
    names = [name for name in model_points.keys() if name in image_points and image_points[name] is not None]
    object_array = np.array([model_points[name] for name in names], dtype=np.float64)
    projected, _ = cv2.projectPoints(object_array, rvec, tvec, camera_matrix(camera_cfg), dist_coeffs(camera_cfg))
    projected = projected.reshape(-1, 2)

    for index, name in enumerate(names):
        measured = tuple(int(round(value)) for value in image_points[name])
        reproj = tuple(int(round(value)) for value in projected[index])
        cv2.circle(image, measured, 4, (30, 120, 255), -1)
        cv2.circle(image, reproj, 3, (20, 180, 60), -1)
        cv2.line(image, measured, reproj, (80, 80, 80), 1)
        if index % 2 == 0:
            cv2.putText(image, name[-3:], reproj, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (20, 60, 20), 1, cv2.LINE_AA)

    axis_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.035, 0.0, 0.0],
            [0.0, 0.035, 0.0],
            [0.0, 0.0, 0.055],
        ],
        dtype=np.float64,
    )
    axis_projected, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix(camera_cfg), dist_coeffs(camera_cfg))
    axis_projected = axis_projected.reshape(-1, 2).astype(int)
    origin = tuple(axis_projected[0])
    cv2.line(image, origin, tuple(axis_projected[1]), (0, 0, 255), 2)
    cv2.line(image, origin, tuple(axis_projected[2]), (0, 180, 0), 2)
    cv2.line(image, origin, tuple(axis_projected[3]), (255, 0, 0), 2)

    safe_imwrite(output_path, image)


def write_annotation_template(path: Path, model_points: dict[str, list[float]]) -> None:
    template = {
        "image": "optional/path/to/image.png",
        "frame": "trocar front aperture keypoints; fill pixel coordinates as [u, v]",
        "points": {name: None for name in model_points.keys()},
    }
    write_json(path, template)


def main() -> None:
    parser = argparse.ArgumentParser(description="First 6D trocar pose baseline using known 3D ring keypoints and solvePnP.")
    parser.add_argument("--camera-config", type=Path, default=DEFAULT_CAMERA_CONFIG)
    parser.add_argument("--trocar-config", type=Path, default=DEFAULT_TROCAR_CONFIG)
    parser.add_argument("--keypoints", type=Path, default=DEFAULT_KEYPOINTS)
    parser.add_argument("--annotation", type=Path, default=None)
    parser.add_argument("--synthetic", action="store_true", help="Run a synthetic self-check instead of loading annotation.")
    parser.add_argument("--noise-px", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    camera_cfg = load_json(args.camera_config)
    trocar_cfg = load_json(args.trocar_config)
    angles = [float(value) for value in trocar_cfg["pose_estimation"]["keypoint_angles_deg"]]
    ring_names = list(trocar_cfg["pose_estimation"]["use_rings"])
    model_points = load_keypoints(args.keypoints, ring_names, angles)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_annotation_template(args.output_dir / "trocar_pnp_annotation_template.json", model_points)

    true_transform = None
    image_path = None
    mode = "annotation"
    if args.synthetic or args.annotation is None:
        mode = "synthetic"
        image_points, true_transform = make_synthetic_observation(model_points, camera_cfg, args.noise_px)
    else:
        image_points, image_path = load_annotation(args.annotation)

    estimated_transform, metrics = solve_pose(model_points, image_points, camera_cfg)
    overlay_path = args.output_dir / f"trocar_pose_pnp_{mode}_overlay.png"
    draw_overlay(image_path, image_points, model_points, estimated_transform, camera_cfg, overlay_path)

    report = {
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "camera_config": str(args.camera_config),
        "trocar_config": str(args.trocar_config),
        "keypoints": str(args.keypoints),
        "annotation": str(args.annotation) if args.annotation else None,
        "T_camera_trocar": estimated_transform.tolist(),
        "translation_camera_trocar_m": estimated_transform[:3, 3].tolist(),
        "rotation_matrix_camera_trocar": estimated_transform[:3, :3].tolist(),
        "metrics": metrics,
        "overlay": str(overlay_path),
        "annotation_template": str(args.output_dir / "trocar_pnp_annotation_template.json"),
    }
    if true_transform is not None:
        report["synthetic_true_T_camera_trocar"] = true_transform.tolist()
        report["synthetic_translation_error_m"] = float(np.linalg.norm(estimated_transform[:3, 3] - true_transform[:3, 3]))
        report["synthetic_rotation_error_deg"] = rotation_error_deg(estimated_transform, true_transform)

    report_path = args.output_dir / f"trocar_pose_pnp_{mode}_report.json"
    write_json(report_path, report)

    print("Trocar PnP baseline finished.")
    print("Mode:", mode)
    print("Report:", report_path)
    print("Overlay:", overlay_path)
    print("Mean reprojection error [px]:", metrics["mean_reprojection_error_px"])
    if true_transform is not None:
        print("Synthetic translation error [m]:", report["synthetic_translation_error_m"])
        print("Synthetic rotation error [deg]:", report["synthetic_rotation_error_deg"])


if __name__ == "__main__":
    main()
