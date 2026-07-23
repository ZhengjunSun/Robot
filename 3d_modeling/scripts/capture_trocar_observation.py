from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from capture_handeye_samples import read_robot_state
from handeye_common import DEFAULT_CONFIG, load_json, write_json
from meca500_fk import forward_kinematics, load_mdh_config


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRINSIC = ROOT / "config" / "camera_extrinsic_colleague_20260612.json"
OUTPUT_ROOT = ROOT / "outputs" / "trocar_observations"


def safe_imwrite(path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise RuntimeError(f"Failed to encode image for {path}")
    encoded.tofile(str(path))


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {index}")
    return cap


def parse_six_floats(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if len(values) != 6:
        raise argparse.ArgumentTypeError("Expected six comma-separated values.")
    return values


def robot_state_from_joints(joints_deg: list[float], mdh_config_path: Path, cart_pose_mm_deg: list[float] | None) -> dict:
    mdh_config = load_mdh_config(mdh_config_path)
    T_base_link6 = forward_kinematics(np.deg2rad(np.array(joints_deg, dtype=float)), mdh_config)[-1]
    return {
        "success": True,
        "joint_angles_deg": joints_deg,
        "cart_pose_mm_deg": cart_pose_mm_deg,
        "T_base_link6": T_base_link6.tolist(),
        "T_base_link6_source": "manual_joint_fk",
        "error": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture one read-only real trocar observation: camera image + robot state.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--extrinsic", type=Path, default=DEFAULT_EXTRINSIC)
    parser.add_argument("--robot-timeout-s", type=float, default=3.0)
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--note", type=str, default="")
    parser.add_argument("--joints-deg", type=parse_six_floats, default=None)
    parser.add_argument("--cart-pose-mm-deg", type=parse_six_floats, default=None)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    config = load_json(args.config)
    extrinsic = load_json(args.extrinsic)
    camera_cfg = config["camera"]
    robot_cfg = config["robot"]
    mdh_config_path = Path(config["robot_kinematics"]["mdh_config"])
    if not mdh_config_path.is_absolute():
        mdh_config_path = ROOT.parent / mdh_config_path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / f"trocar_obs_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = open_camera(
        int(camera_cfg["index"]),
        int(camera_cfg["image_width"]),
        int(camera_cfg["image_height"]),
    )
    try:
        frame = None
        ok = False
        for _ in range(max(1, args.warmup_frames)):
            ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("Camera opened but frame read failed.")
    finally:
        cap.release()

    if args.joints_deg is not None:
        robot_state = robot_state_from_joints(args.joints_deg, mdh_config_path, args.cart_pose_mm_deg)
    else:
        robot_state = read_robot_state(
            ip=robot_cfg["ip_address"],
            timeout_s=args.robot_timeout_s,
            mdh_config_path=mdh_config_path,
        )

    image_path = output_dir / "camera_rgb.png"
    safe_imwrite(image_path, frame)

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "moves_robot": False,
        "note": args.note,
        "image": str(image_path.relative_to(ROOT).as_posix()),
        "camera_config": config["camera"],
        "extrinsic_config_path": str(args.extrinsic),
        "extrinsic": extrinsic,
        "robot": robot_state,
        "next_step": "Annotate trocar rim keypoints or run an automatic detector to estimate T_camera_trocar.",
    }
    metadata_path = output_dir / "metadata.json"
    write_json(metadata_path, metadata)

    print("Trocar observation captured.")
    print("Image:", image_path)
    print("Metadata:", metadata_path)
    print("Robot state:", "OK" if robot_state.get("success") else robot_state.get("error"))


if __name__ == "__main__":
    main()
