from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from handeye_common import (
    DEFAULT_CONFIG,
    detect_charuco_pose,
    load_json,
    pose_mm_deg_to_matrix,
    to_plain,
    write_json,
)
from meca500_fk import forward_kinematics, load_mdh_config


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "calibration" / "handeye_samples"


def read_robot_state(ip: str, timeout_s: float, mdh_config_path: Path) -> dict:
    from mecademicpy.robot import Robot

    robot = Robot()
    try:
        robot.Connect(address=ip, enable_synchronous_mode=True, timeout=timeout_s)
        status = to_plain(robot.GetStatusRobot(synchronous_update=True, timeout=timeout_s))
        safety_status = to_plain(robot.GetSafetyStatus(synchronous_update=True, timeout=timeout_s))
        actual_pose = to_plain(robot.GetPose())
        target_joints = to_plain(
            robot.GetRtTargetJointPos(
                include_timestamp=False,
                synchronous_update=True,
                timeout=timeout_s,
            )
        )
        target_cart_pose = to_plain(
            robot.GetRtTargetCartPos(
                include_timestamp=False,
                synchronous_update=True,
                timeout=timeout_s,
            )
        )

        if actual_pose and len(actual_pose) == 6:
            T_base_link6 = pose_mm_deg_to_matrix(actual_pose)
            source = "actual_get_pose_xyz_xyz_euler"
        elif target_cart_pose and len(target_cart_pose) == 6:
            T_base_link6 = pose_mm_deg_to_matrix(target_cart_pose)
            source = "rt_target_cart_pose_xyz_xyz_euler"
        elif target_joints and len(target_joints) == 6:
            mdh_config = load_mdh_config(mdh_config_path)
            T_base_link6 = forward_kinematics(np.deg2rad(np.array(target_joints, dtype=float)), mdh_config)[-1]
            source = "rt_target_joint_fk"
        else:
            raise RuntimeError("Robot returned neither six joints nor six cartesian pose values.")

        return {
            "success": True,
            "joint_angles_deg": target_joints,
            "cart_pose_mm_deg": actual_pose,
            "actual_pose_mm_deg": actual_pose,
            "rt_target_joint_angles_deg": target_joints,
            "rt_target_cart_pose_mm_deg": target_cart_pose,
            "status": status,
            "safety_status": safety_status,
            "T_base_link6": T_base_link6.tolist(),
            "T_base_link6_source": source,
            "error": None,
        }
    except Exception as exc:
        return {
            "success": False,
            "joint_angles_deg": None,
            "cart_pose_mm_deg": None,
            "T_base_link6": None,
            "T_base_link6_source": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        try:
            robot.Disconnect()
        except Exception:
            pass


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {index}")
    return cap


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect read-only hand-eye calibration samples.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--interval-s", type=float, default=1.0)
    parser.add_argument("--robot-timeout-s", type=float, default=2.0)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    args = parser.parse_args()

    config = load_json(args.config)
    camera_cfg = config["camera"]
    robot_cfg = config["robot"]
    mdh_config_path = Path(config["robot_kinematics"]["mdh_config"])
    if not mdh_config_path.is_absolute():
        mdh_config_path = ROOT.parent / mdh_config_path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_dir = args.dataset_dir or (DATASET_ROOT / f"handeye_{timestamp}")
    image_dir = dataset_dir / "images"
    overlay_dir = dataset_dir / "overlays"
    image_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    session = {
        "timestamp": datetime.now().isoformat(),
        "config": str(args.config.relative_to(ROOT).as_posix()) if args.config.is_absolute() else args.config.as_posix(),
        "dataset_dir": str(dataset_dir.relative_to(ROOT).as_posix()),
        "moves_robot": False,
        "samples_requested": args.samples,
        "samples_saved": 0,
        "samples": [],
        "errors": [],
    }

    try:
        cap = open_camera(int(camera_cfg["index"]), int(camera_cfg["image_width"]), int(camera_cfg["image_height"]))
    except Exception as exc:
        session["errors"].append(f"camera_open_failed: {type(exc).__name__}: {exc}")
        write_json(dataset_dir / "session.json", session)
        print("Camera open failed:", exc)
        print("Session:", dataset_dir / "session.json")
        sys.exit(1)

    try:
        for index in range(args.samples):
            for _ in range(3):
                cap.read()
            ok, frame = cap.read()
            if not ok or frame is None:
                session["errors"].append(f"sample_{index}: camera_read_failed")
                continue

            detection = detect_charuco_pose(frame, config)
            robot_state = read_robot_state(
                ip=robot_cfg["ip_address"],
                timeout_s=args.robot_timeout_s,
                mdh_config_path=mdh_config_path,
            )

            if not detection["success"]:
                session["errors"].append(f"sample_{index}: detection_failed: {detection['message']}")
            if not robot_state["success"]:
                session["errors"].append(f"sample_{index}: robot_state_failed: {robot_state['error']}")

            if detection["success"] and robot_state["success"]:
                sample_id = f"sample_{session['samples_saved']:03d}"
                image_path = image_dir / f"{sample_id}.png"
                overlay_path = overlay_dir / f"{sample_id}_overlay.png"
                sample_path = dataset_dir / f"{sample_id}.json"
                cv2.imwrite(str(image_path), frame)
                cv2.imwrite(str(overlay_path), detection.pop("overlay_bgr"))

                sample = {
                    "sample_id": sample_id,
                    "timestamp": datetime.now().isoformat(),
                    "image": str(image_path.relative_to(dataset_dir).as_posix()),
                    "overlay": str(overlay_path.relative_to(dataset_dir).as_posix()),
                    "detection": detection,
                    "robot": robot_state,
                }
                write_json(sample_path, sample)
                session["samples"].append(str(sample_path.relative_to(dataset_dir).as_posix()))
                session["samples_saved"] += 1
                print(f"Saved {sample_id}: reproj={detection['mean_reprojection_error_px']:.3f}px")
            else:
                print(f"Skipped sample {index}: detection={detection['success']} robot={robot_state['success']}")

            if index != args.samples - 1:
                time.sleep(args.interval_s)
    finally:
        cap.release()
        write_json(dataset_dir / "session.json", session)

    print("Hand-eye collection finished.")
    print("Dataset:", dataset_dir)
    print("Saved samples:", session["samples_saved"])


if __name__ == "__main__":
    main()
