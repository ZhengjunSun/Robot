from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from handeye_common import (
    camera_matrix,
    dist_coeffs,
    load_json,
    matrix_from_rvec_tvec,
    matrix_to_pose_mm_deg,
    project_transform_to_se3,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMERA_CONFIG = ROOT / "config" / "handeye_calibration_config.json"
DEFAULT_TROCAR_CONFIG = ROOT / "config" / "trocar_model_measured_20260612.json"
DEFAULT_EXTRINSIC = ROOT / "config" / "camera_extrinsic_colleague_20260612.json"
OUTPUT_DIR = ROOT / "outputs" / "trocar_pose_from_ring"


def safe_imwrite(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise RuntimeError(f"Failed to encode image for {path}")
    encoded.tofile(str(path))


def safe_imread(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def load_observation(observation_dir: Path | None, image_path: Path | None) -> tuple[Path, dict | None]:
    if observation_dir is not None:
        metadata_path = observation_dir / "metadata.json"
        metadata = load_json(metadata_path) if metadata_path.exists() else None
        return observation_dir / "camera_rgb.png", metadata
    if image_path is None:
        raise ValueError("Use --observation-dir or --image.")
    return image_path, None


def fit_ellipse_from_contour(contour: np.ndarray) -> dict:
    if len(contour) < 5:
        raise ValueError("Need at least 5 contour points to fit ellipse.")
    (cx, cy), (d1, d2), angle_deg = cv2.fitEllipse(contour)
    major = max(float(d1), float(d2))
    minor = min(float(d1), float(d2))
    # cv2 angle is tied to the returned first axis. Shift by 90 deg when d2 is the major axis.
    theta = math.radians(float(angle_deg) + (90.0 if d2 > d1 else 0.0))
    return {
        "center": [float(cx), float(cy)],
        "major_diameter_px": major,
        "minor_diameter_px": minor,
        "angle_deg": math.degrees(theta),
        "theta_rad": theta,
    }


def ellipse_point(ellipse: dict, angle_deg: float) -> list[float]:
    theta = ellipse["theta_rad"]
    local = np.array(
        [
            0.5 * ellipse["major_diameter_px"] * math.cos(math.radians(angle_deg)),
            0.5 * ellipse["minor_diameter_px"] * math.sin(math.radians(angle_deg)),
        ],
        dtype=np.float64,
    )
    rotation = np.array(
        [
            [math.cos(theta), -math.sin(theta)],
            [math.sin(theta), math.cos(theta)],
        ],
        dtype=np.float64,
    )
    point = np.array(ellipse["center"], dtype=np.float64) + rotation @ local
    return point.tolist()


def detect_outer_orange(image: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([5, 80, 40]), np.array([35, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dark_mask = cv2.inRange(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 0, 70)
    candidates = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 30.0 or len(contour) < 5:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        aspect = w / max(h, 1)
        if aspect < 0.35 or aspect > 2.8:
            continue
        ellipse = fit_ellipse_from_contour(contour)
        cx, cy = ellipse["center"]
        support_radius = max(6, int(round(0.55 * ellipse["major_diameter_px"])))
        roi = np.zeros_like(dark_mask)
        cv2.circle(roi, (int(round(cx)), int(round(cy))), support_radius, 255, -1)
        dark_support = int(cv2.countNonZero(cv2.bitwise_and(dark_mask, roi)))
        center_score = -abs(cx - image.shape[1] / 2.0) - abs(cy - image.shape[0] / 2.0)
        candidates.append((dark_support > 0, dark_support, area, center_score, contour))
    if not candidates:
        raise RuntimeError("No orange outer ring contour found.")
    contour = sorted(candidates, key=lambda item: (item[0], item[1], item[2], item[3]), reverse=True)[0][4]
    return fit_ellipse_from_contour(contour), contour, mask


def detect_outer_edge(
    image: np.ndarray,
    trocar_cfg: dict,
    expected_size_range_px: tuple[float, float] = (8.0, 50.0),
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Edge-based backup detection: Canny + contour finding + circularity filter."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_center = np.array([image.shape[1] / 2.0, image.shape[0] / 2.0])
    candidates = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 20.0 or len(contour) < 5:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter < 1.0:
            continue
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        if circularity < 0.5:
            continue
        try:
            ellipse = fit_ellipse_from_contour(contour)
        except ValueError:
            continue
        major = ellipse["major_diameter_px"]
        minor = ellipse["minor_diameter_px"]
        if major < expected_size_range_px[0] or major > expected_size_range_px[1]:
            continue
        if major / max(minor, 1.0) > 1.5:
            continue
        cx, cy = ellipse["center"]
        center_dist = math.sqrt((cx - img_center[0]) ** 2 + (cy - img_center[1]) ** 2)
        # Prefer centered, circular contours
        score = circularity * 100.0 - center_dist * 0.1
        candidates.append((score, circularity, area, contour, ellipse))

    if not candidates:
        raise RuntimeError("No circular contour found via edge detection.")

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _, _, _, contour, ellipse = candidates[0]
    return ellipse, contour, edges


def detect_outer_hough(
    image: np.ndarray,
    expected_radius_range_px: tuple[int, int] = (5, 30),
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Hough Circle Transform backup detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.0,
        minDist=expected_radius_range_px[0],
        param1=50, param2=15,
        minRadius=expected_radius_range_px[0],
        maxRadius=expected_radius_range_px[1],
    )
    if circles is None or len(circles[0]) == 0:
        raise RuntimeError("No circle found via Hough Transform.")

    img_center = np.array([image.shape[1] / 2.0, image.shape[0] / 2.0])
    best = None
    best_dist = float("inf")
    for c in circles[0]:
        dist = math.sqrt((c[0] - img_center[0]) ** 2 + (c[1] - img_center[1]) ** 2)
        if dist < best_dist:
            best_dist = dist
            best = c

    cx, cy, radius = float(best[0]), float(best[1]), float(best[2])
    diameter = radius * 2.0
    # Construct equivalent ellipse dict (circular = major == minor == diameter)
    ellipse = {
        "center": [cx, cy],
        "major_diameter_px": diameter,
        "minor_diameter_px": diameter,
        "angle_deg": 0.0,
        "theta_rad": 0.0,
    }

    # Visualization mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (int(round(cx)), int(round(cy))), int(round(radius)), 255, 2)
    return ellipse, np.array([[[cx, cy]]], dtype=np.int32), mask


def detect_ring_with_fallback(
    image: np.ndarray,
    trocar_cfg: dict,
    detection_method: str = "auto",
) -> tuple[dict, np.ndarray, np.ndarray, str]:
    """Fallback chain: color -> edge -> hough."""
    methods = []
    if detection_method == "auto":
        methods = [("color", detect_outer_orange), ("edge", detect_outer_edge), ("hough", detect_outer_hough)]
    elif detection_method == "color":
        methods = [("color", detect_outer_orange)]
    elif detection_method == "edge":
        methods = [("edge", detect_outer_edge)]
    elif detection_method == "hough":
        methods = [("hough", detect_outer_hough)]
    else:
        raise ValueError(f"Unknown detection method: {detection_method}")

    errors = []
    for name, func in methods:
        try:
            if name == "edge":
                ellipse, contour, mask = func(image, trocar_cfg)
            elif name == "hough":
                ellipse, contour, mask = func(image)
            else:
                ellipse, contour, mask = func(image)
            return ellipse, contour, mask, name
        except RuntimeError as exc:
            errors.append(f"{name}: {exc}")
            continue

    raise RuntimeError("All detection methods failed:\n" + "\n".join(errors))


def detect_inner_dark(image: np.ndarray, outer_ellipse: dict, trocar_cfg: dict) -> tuple[dict, np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, 0, 70)
    cx, cy = outer_ellipse["center"]
    radius = 0.75 * outer_ellipse["major_diameter_px"]
    roi_mask = np.zeros_like(mask)
    cv2.circle(roi_mask, (int(round(cx)), int(round(cy))), int(round(radius)), 255, -1)
    mask = cv2.bitwise_and(mask, roi_mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    target_ratio = float(trocar_cfg["dimensions"]["inner_radius_m"]) / float(trocar_cfg["dimensions"]["outer_radius_m"])
    candidates = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 10.0 or len(contour) < 5:
            continue
        ellipse = fit_ellipse_from_contour(contour)
        dist = float(np.linalg.norm(np.array(ellipse["center"]) - np.array(outer_ellipse["center"])))
        if dist > 0.35 * outer_ellipse["major_diameter_px"]:
            continue
        ratio = ellipse["major_diameter_px"] / outer_ellipse["major_diameter_px"]
        if ratio < 0.15 or ratio > 0.75:
            continue
        score = -abs(ratio - target_ratio) * 100.0 - dist / max(outer_ellipse["major_diameter_px"], 1.0)
        candidates.append((score, area, contour, ellipse))
    if not candidates:
        raise RuntimeError("No dark inner bore contour found near the orange ring.")
    _, _, contour, ellipse = sorted(candidates, key=lambda item: (item[0], item[1]), reverse=True)[0]
    # Use outer ellipse orientation for consistent point correspondences.
    ellipse["theta_rad"] = outer_ellipse["theta_rad"]
    ellipse["angle_deg"] = outer_ellipse["angle_deg"]
    return ellipse, contour, mask


def build_correspondences(
    trocar_cfg: dict,
    outer_ellipse: dict,
    inner_ellipse: dict,
    use_inner_ring: bool,
) -> tuple[np.ndarray, np.ndarray, dict]:
    dims = trocar_cfg["dimensions"]
    angles = [float(value) for value in trocar_cfg["pose_estimation"]["keypoint_angles_deg"]]
    object_points = []
    image_points = []
    annotation = {}
    rings = [("outer_front", float(dims["outer_radius_m"]), outer_ellipse)]
    if use_inner_ring:
        rings.append(("inner_front", float(dims["inner_radius_m"]), inner_ellipse))
    for ring_name, radius, ellipse in rings:
        for angle in angles:
            key = f"{ring_name}_{int(angle):03d}"
            obj = [radius * math.cos(math.radians(angle)), radius * math.sin(math.radians(angle)), 0.0]
            img = ellipse_point(ellipse, angle)
            object_points.append(obj)
            image_points.append(img)
            annotation[key] = img
    return np.array(object_points, dtype=np.float64), np.array(image_points, dtype=np.float64), annotation


def solve_pose(object_points: np.ndarray, image_points: np.ndarray, camera_cfg: dict) -> tuple[np.ndarray, dict]:
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix(camera_cfg),
        dist_coeffs(camera_cfg),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError("solvePnP failed.")
    transform = matrix_from_rvec_tvec(rvec, tvec)
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix(camera_cfg), dist_coeffs(camera_cfg))
    errors = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
    return transform, {
        "mean_reprojection_error_px": float(errors.mean()),
        "max_reprojection_error_px": float(errors.max()),
        "used_point_count": int(len(object_points)),
    }


def draw_overlay(
    image: np.ndarray,
    outer_ellipse: dict,
    inner_ellipse: dict | None,
    object_points: np.ndarray,
    transform: np.ndarray,
    camera_cfg: dict,
) -> np.ndarray:
    overlay = image.copy()
    for ellipse, color in [(outer_ellipse, (0, 165, 255)), (inner_ellipse, (0, 0, 0))]:
        if ellipse is None:
            continue
        center = tuple(int(round(v)) for v in ellipse["center"])
        axes = (
            int(round(0.5 * ellipse["major_diameter_px"])),
            int(round(0.5 * ellipse["minor_diameter_px"])),
        )
        cv2.ellipse(overlay, center, axes, ellipse["angle_deg"], 0, 360, color, 2)

    rvec = cv2.Rodrigues(transform[:3, :3])[0]
    tvec = transform[:3, 3].reshape(3, 1)
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix(camera_cfg), dist_coeffs(camera_cfg))
    for point in projected.reshape(-1, 2):
        cv2.circle(overlay, tuple(int(round(v)) for v in point), 2, (0, 255, 0), -1)
    cv2.drawFrameAxes(overlay, camera_matrix(camera_cfg), dist_coeffs(camera_cfg), rvec, tvec, 0.003)
    return overlay


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate first real T_camera_trocar from orange/dark ring ellipse detection.")
    parser.add_argument("--observation-dir", type=Path, default=None)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--camera-config", type=Path, default=DEFAULT_CAMERA_CONFIG)
    parser.add_argument("--trocar-config", type=Path, default=DEFAULT_TROCAR_CONFIG)
    parser.add_argument("--extrinsic", type=Path, default=DEFAULT_EXTRINSIC)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--use-inner-ring",
        action="store_true",
        help="Also use the dark inner bore ellipse. Disabled by default because the inner boundary is often shadow-contaminated.",
    )
    parser.add_argument(
        "--detection-method",
        choices=["color", "edge", "hough", "auto"],
        default="color",
        help="Ring detection method. 'auto' falls back: color -> edge -> hough.",
    )
    args = parser.parse_args()

    image_path, metadata = load_observation(args.observation_dir, args.image)
    image = safe_imread(image_path)
    if image is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    camera_cfg = load_json(args.camera_config)
    trocar_cfg = load_json(args.trocar_config)
    extrinsic = load_json(args.extrinsic)

    outer_ellipse, outer_contour, outer_mask, method_used = detect_ring_with_fallback(
        image, trocar_cfg, detection_method=args.detection_method,
    )
    inner_ellipse = None
    inner_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    inner_detection_error = None
    try:
        inner_ellipse, inner_contour, inner_mask = detect_inner_dark(image, outer_ellipse, trocar_cfg)
    except RuntimeError as exc:
        inner_detection_error = str(exc)
        if args.use_inner_ring:
            raise
    object_points, image_points, annotation = build_correspondences(
        trocar_cfg,
        outer_ellipse,
        inner_ellipse,
        use_inner_ring=args.use_inner_ring,
    )
    T_camera_trocar, metrics = solve_pose(object_points, image_points, camera_cfg)

    report = {
        "timestamp": datetime.now().isoformat(),
        "status": "ok",
        "image": str(image_path),
        "camera_config": str(args.camera_config),
        "trocar_config": str(args.trocar_config),
        "extrinsic": str(args.extrinsic),
        "measurement": trocar_cfg.get("measurement"),
        "outer_ellipse": {k: v for k, v in outer_ellipse.items() if k != "theta_rad"},
        "inner_ellipse": None if inner_ellipse is None else {k: v for k, v in inner_ellipse.items() if k != "theta_rad"},
        "inner_detection_error": inner_detection_error,
        "detection_method_used": method_used,
        "use_inner_ring_for_pnp": args.use_inner_ring,
        "annotation_points": annotation,
        "T_camera_trocar": T_camera_trocar.tolist(),
        "pose_camera_trocar_mm_deg": matrix_to_pose_mm_deg(T_camera_trocar),
        "translation_camera_trocar_m": T_camera_trocar[:3, 3].tolist(),
        "trocar_axis_camera": T_camera_trocar[:3, 2].tolist(),
        "metrics": metrics,
        "limitations": [
            "The visible trocar entrance is a rotationally symmetric ring, so roll around the bore axis is weakly observable and treated as an ellipse-axis convention.",
            "The 2D ring points are sampled from fitted ellipses, not from manually verified physical keypoints.",
            "Use this as the first baseline; final experiments should include measured dimensions and repeated validation images.",
        ],
    }

    if metadata and metadata.get("robot", {}).get("success"):
        T_base_link6 = np.array(metadata["robot"]["T_base_link6"], dtype=np.float64)
        T_link6_camera = project_transform_to_se3(np.array(extrinsic["T_link6_camera"], dtype=np.float64))
        T_base_trocar = T_base_link6 @ T_link6_camera @ T_camera_trocar
        report["T_base_trocar"] = T_base_trocar.tolist()
        report["pose_base_trocar_mm_deg"] = matrix_to_pose_mm_deg(T_base_trocar)
        report["trocar_axis_base"] = T_base_trocar[:3, 2].tolist()
        report["robot"] = metadata["robot"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlay = draw_overlay(image, outer_ellipse, inner_ellipse, object_points, T_camera_trocar, camera_cfg)
    overlay_path = args.output_dir / "real_trocar_ring_pose_overlay.png"
    safe_imwrite(overlay_path, overlay)
    safe_imwrite(args.output_dir / "outer_orange_mask.png", outer_mask)
    safe_imwrite(args.output_dir / "inner_dark_mask.png", inner_mask)

    annotation_path = args.output_dir / "real_trocar_ring_annotation.json"
    write_json(
        annotation_path,
        {
            "image": str(image_path),
            "points": annotation,
            "source": "automatic ellipse detector",
            "trocar_config": str(args.trocar_config),
        },
    )
    report["overlay"] = str(overlay_path)
    report["annotation"] = str(annotation_path)
    report_path = args.output_dir / "real_trocar_ring_pose_report.json"
    write_json(report_path, report)

    print("Real trocar pose estimated.")
    print("Report:", report_path)
    print("Overlay:", overlay_path)
    print("Pose camera->trocar [mm,deg]:", report["pose_camera_trocar_mm_deg"])
    if "pose_base_trocar_mm_deg" in report:
        print("Pose base->trocar [mm,deg]:", report["pose_base_trocar_mm_deg"])
    print("Mean reprojection error [px]:", metrics["mean_reprojection_error_px"])


if __name__ == "__main__":
    main()
