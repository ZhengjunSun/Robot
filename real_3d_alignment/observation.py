from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import PROJECT_ROOT, load_config, load_json, resolve_project_path, write_json


SCRIPTS_3D = PROJECT_ROOT / "3d_modeling" / "scripts"
if str(SCRIPTS_3D) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_3D))

from capture_handeye_samples import read_robot_state  # noqa: E402
from meca500_fk import forward_kinematics, load_mdh_config  # noqa: E402


def safe_imwrite(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise RuntimeError(f"Failed to encode image for {path}")
    encoded.tofile(str(path))


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {index}")
    return cap


def robot_state_from_joints(
    joints_deg: list[float],
    mdh_config_path: Path,
    cart_pose_mm_deg: list[float] | None,
) -> dict[str, Any]:
    mdh_config = load_mdh_config(mdh_config_path)
    T_base_link6 = forward_kinematics(np.deg2rad(np.array(joints_deg, dtype=float)), mdh_config)[-1]
    return {
        "success": True,
        "joint_angles_deg": joints_deg,
        "cart_pose_mm_deg": cart_pose_mm_deg,
        "actual_pose_mm_deg": cart_pose_mm_deg,
        "rt_target_joint_angles_deg": joints_deg,
        "rt_target_cart_pose_mm_deg": cart_pose_mm_deg,
        "T_base_link6": T_base_link6.tolist(),
        "T_base_link6_source": "manual_joint_fk",
        "error": None,
    }


def capture_live_observation(
    *,
    config_path: Path,
    output_dir: Path,
    note: str = "",
    robot_timeout_s: float = 3.0,
    warmup_frames: int = 10,
    manual_joints_deg: list[float] | None = None,
    manual_cart_pose_mm_deg: list[float] | None = None,
) -> Path:
    cfg = load_config(config_path)
    paths = cfg["paths"]
    camera_config = resolve_project_path(paths["camera_config"])
    extrinsic_config = resolve_project_path(paths["extrinsic_config"])

    handeye_cfg = load_json(camera_config)
    extrinsic = load_json(extrinsic_config)
    camera_cfg = handeye_cfg["camera"]
    robot_cfg = handeye_cfg["robot"]

    mdh_config_path = Path(handeye_cfg["robot_kinematics"]["mdh_config"])
    if not mdh_config_path.is_absolute():
        mdh_config_path = PROJECT_ROOT / mdh_config_path

    output_dir.mkdir(parents=True, exist_ok=True)
    cap = open_camera(
        int(camera_cfg["index"]),
        int(camera_cfg["image_width"]),
        int(camera_cfg["image_height"]),
    )
    try:
        frame = None
        ok = False
        for _ in range(max(1, int(warmup_frames))):
            ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("Camera opened but frame read failed.")
    finally:
        cap.release()

    if manual_joints_deg is not None:
        robot_state = robot_state_from_joints(manual_joints_deg, mdh_config_path, manual_cart_pose_mm_deg)
    else:
        robot_state = read_robot_state(
            ip=robot_cfg["ip_address"],
            timeout_s=robot_timeout_s,
            mdh_config_path=mdh_config_path,
        )

    image_path = output_dir / "camera_rgb.png"
    safe_imwrite(image_path, frame)

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "moves_robot": False,
        "note": note,
        "image": str(image_path),
        "camera_config_path": str(camera_config),
        "camera_config": camera_cfg,
        "extrinsic_config_path": str(extrinsic_config),
        "extrinsic": extrinsic,
        "robot": robot_state,
        "next_step": "Run run_real_3d_alignment.py on this observation folder for pose, gate, and dry-run control.",
    }
    write_json(output_dir / "metadata.json", metadata)
    return output_dir
