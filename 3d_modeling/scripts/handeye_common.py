from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
DEFAULT_CONFIG = ROOT / "config" / "handeye_calibration_config.json"


ARUCO_DICTIONARIES = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_APRILTAG_36H11": cv2.aruco.DICT_APRILTAG_36H11,
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def camera_matrix(config: dict) -> np.ndarray:
    camera = config["camera"]
    return np.array(
        [
            [camera["fx"], 0.0, camera["cx"]],
            [0.0, camera["fy"], camera["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def dist_coeffs(config: dict) -> np.ndarray:
    return np.array(config["camera"].get("dist_coeffs", [0, 0, 0, 0, 0]), dtype=np.float64)


def aruco_dictionary(config: dict) -> cv2.aruco.Dictionary:
    name = config["target"]["dictionary"]
    if name not in ARUCO_DICTIONARIES:
        raise ValueError(f"Unsupported ArUco dictionary: {name}")
    return cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARIES[name])


def create_charuco_board(config: dict) -> cv2.aruco.CharucoBoard:
    target = config["target"]
    return cv2.aruco.CharucoBoard(
        (int(target["squares_x"]), int(target["squares_y"])),
        float(target["square_length_m"]),
        float(target["marker_length_m"]),
        aruco_dictionary(config),
    )


def pose_mm_deg_to_matrix(pose_mm_deg: list[float]) -> np.ndarray:
    x, y, z, alpha, beta, gamma = [float(value) for value in pose_mm_deg]
    transform = np.eye(4)
    transform[:3, 3] = np.array([x, y, z], dtype=float) / 1000.0
    transform[:3, :3] = R.from_euler("XYZ", [alpha, beta, gamma], degrees=True).as_matrix()
    return transform


def matrix_to_pose_mm_deg(transform: np.ndarray) -> list[float]:
    xyz_mm = transform[:3, 3] * 1000.0
    euler = R.from_matrix(transform[:3, :3]).as_euler("XYZ", degrees=True)
    return [float(value) for value in np.concatenate([xyz_mm, euler])]


def matrix_from_rvec_tvec(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = cv2.Rodrigues(np.asarray(rvec, dtype=float).reshape(3, 1))[0]
    transform[:3, 3] = np.asarray(tvec, dtype=float).reshape(3)
    return transform


def rvec_tvec_from_matrix(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rvec = cv2.Rodrigues(transform[:3, :3])[0].reshape(3, 1)
    tvec = transform[:3, 3].reshape(3, 1)
    return rvec, tvec


def rotation_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    delta = a[:3, :3].T @ b[:3, :3]
    trace_value = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
    return float(math.degrees(math.acos(trace_value)))


def transform_inverse(transform: np.ndarray) -> np.ndarray:
    inv = np.eye(4)
    inv[:3, :3] = transform[:3, :3].T
    inv[:3, 3] = -inv[:3, :3] @ transform[:3, 3]
    return inv


def project_rotation_to_so3(rotation: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.asarray(rotation, dtype=np.float64))
    projected = u @ vt
    if np.linalg.det(projected) < 0:
        u[:, -1] *= -1.0
        projected = u @ vt
    return projected


def project_transform_to_se3(transform: np.ndarray) -> np.ndarray:
    projected = np.array(transform, dtype=np.float64).copy()
    projected[:3, :3] = project_rotation_to_so3(projected[:3, :3])
    return projected


def to_plain(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [to_plain(item) for item in value]
    if hasattr(value, "data"):
        return to_plain(value.data)
    if hasattr(value, "__dict__"):
        return {key: to_plain(item) for key, item in value.__dict__.items() if not key.startswith("_")}
    try:
        return list(value)
    except TypeError:
        return str(value)


def detect_charuco_pose(image: np.ndarray, config: dict) -> dict:
    board = create_charuco_board(config)
    detector = cv2.aruco.CharucoDetector(board)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)

    detected_count = 0 if charuco_ids is None else int(len(charuco_ids))
    result = {
        "success": False,
        "detected_charuco_corners": detected_count,
        "detected_marker_count": 0 if marker_ids is None else int(len(marker_ids)),
        "message": "",
    }
    if charuco_ids is None or detected_count < int(config["target"]["min_corners"]):
        result["message"] = f"Not enough ChArUco corners: {detected_count}"
        return result

    object_points, image_points = board.matchImagePoints(charuco_corners, charuco_ids)
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix(config),
        dist_coeffs(config),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        result["message"] = "solvePnP failed"
        return result

    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix(config), dist_coeffs(config))
    reproj = np.linalg.norm(projected.reshape(-1, 2) - image_points.reshape(-1, 2), axis=1)

    overlay = image.copy()
    cv2.aruco.drawDetectedCornersCharuco(overlay, charuco_corners, charuco_ids)
    cv2.drawFrameAxes(overlay, camera_matrix(config), dist_coeffs(config), rvec, tvec, 0.05)

    transform = matrix_from_rvec_tvec(rvec, tvec)
    result.update(
        {
            "success": True,
            "message": "ok",
            "rvec": rvec.reshape(3).tolist(),
            "tvec_m": tvec.reshape(3).tolist(),
            "T_camera_target": transform.tolist(),
            "mean_reprojection_error_px": float(np.mean(reproj)),
            "max_reprojection_error_px": float(np.max(reproj)),
            "overlay_bgr": overlay,
        }
    )
    return result
