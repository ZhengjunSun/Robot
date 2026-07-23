from __future__ import annotations

import argparse
import json
import socket
import sys
from datetime import datetime
from pathlib import Path

import cv2

from handeye_common import DEFAULT_CONFIG, detect_charuco_pose, load_json, write_json


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "preflight"


def safe_imwrite(path: Path, image) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        return False
    encoded.tofile(str(path))
    return True


def check_imports() -> dict:
    modules = ["cv2", "numpy", "scipy", "pybullet", "mecademicpy"]
    result = {}
    for module_name in modules:
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "available")
            result[module_name] = {"success": True, "version": str(version)}
        except Exception as exc:
            result[module_name] = {"success": False, "error": f"{type(exc).__name__}: {exc}"}
    return result


def check_files(config: dict) -> dict:
    target_dir = ROOT / "calibration" / "targets"
    required = [
        ROOT / "config" / "handeye_calibration_config.json",
        target_dir / "charuco_7x5_30mm_exact_300dpi.png",
        target_dir / "charuco_7x5_30mm_a4_landscape_300dpi.png",
        target_dir / "charuco_7x5_30mm_metadata.json",
    ]
    mdh_path = Path(config["robot_kinematics"]["mdh_config"])
    if not mdh_path.is_absolute():
        mdh_path = ROOT.parent / mdh_path
    required.append(mdh_path)

    return {
        str(path.relative_to(ROOT.parent).as_posix()): {
            "exists": path.exists(),
            "length": path.stat().st_size if path.exists() else None,
        }
        for path in required
    }


def check_robot_port(ip: str, port: int, timeout_s: float) -> dict:
    started = datetime.now()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout_s)
    try:
        sock.connect((ip, port))
        return {
            "success": True,
            "ip": ip,
            "port": port,
            "timeout_s": timeout_s,
            "started_at": started.isoformat(),
            "error": None,
        }
    except Exception as exc:
        return {
            "success": False,
            "ip": ip,
            "port": port,
            "timeout_s": timeout_s,
            "started_at": started.isoformat(),
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        sock.close()


def check_robot_state(ip: str, timeout_s: float, config: dict) -> dict:
    from capture_handeye_samples import read_robot_state

    mdh_config_path = Path(config["robot_kinematics"]["mdh_config"])
    if not mdh_config_path.is_absolute():
        mdh_config_path = ROOT.parent / mdh_config_path
    return read_robot_state(ip=ip, timeout_s=timeout_s, mdh_config_path=mdh_config_path)


def check_camera(config: dict, warmup_frames: int) -> dict:
    camera_cfg = config["camera"]
    index = int(camera_cfg["index"])
    width = int(camera_cfg["image_width"])
    height = int(camera_cfg["image_height"])

    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    result = {
        "success": False,
        "camera_index": index,
        "requested_width": width,
        "requested_height": height,
        "frame_path": None,
        "charuco_detection": None,
        "error": None,
    }
    try:
        if not cap.isOpened():
            result["error"] = f"Cannot open camera index {index}"
            return result

        frame = None
        ok = False
        for _ in range(max(1, warmup_frames)):
            ok, frame = cap.read()
        if not ok or frame is None:
            result["error"] = "Camera opened but frame read failed"
            return result

        frame_path = OUTPUT_DIR / "preflight_camera_frame.png"
        safe_imwrite(frame_path, frame)
        result["frame_path"] = str(frame_path.relative_to(ROOT).as_posix())
        result["actual_width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        result["actual_height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        detection = detect_charuco_pose(frame, config)
        overlay = detection.pop("overlay_bgr", None)
        if overlay is not None:
            overlay_path = OUTPUT_DIR / "preflight_charuco_overlay.png"
            safe_imwrite(overlay_path, overlay)
            detection["overlay_path"] = str(overlay_path.relative_to(ROOT).as_posix())
        result["charuco_detection"] = detection
        result["success"] = True
        return result
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    finally:
        cap.release()


def summarize(report: dict) -> list[str]:
    lines = []
    imports_ok = all(item["success"] for item in report["imports"].values())
    files_ok = all(item["exists"] for item in report["files"].values())
    camera = report.get("camera")
    robot_port = report.get("robot_port")

    lines.append(f"Imports: {'OK' if imports_ok else 'CHECK'}")
    lines.append(f"Files: {'OK' if files_ok else 'CHECK'}")
    if camera is None:
        lines.append("Camera: SKIPPED")
    else:
        det = camera.get("charuco_detection") or {}
        det_status = "detected" if det.get("success") else "not detected"
        lines.append(f"Camera: {'OK' if camera.get('success') else 'CHECK'} ({det_status})")
    if robot_port is None:
        lines.append("Robot port: SKIPPED")
    else:
        lines.append(f"Robot port: {'OK' if robot_port.get('success') else 'CHECK'} ({robot_port['ip']}:{robot_port['port']})")
    if report.get("robot_state") is not None:
        lines.append(f"Robot state read: {'OK' if report['robot_state'].get('success') else 'CHECK'}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only preflight checks before Meca500 hand-eye sample collection.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--skip-camera", action="store_true")
    parser.add_argument("--skip-robot-port", action="store_true")
    parser.add_argument("--read-robot-state", action="store_true", help="Read current robot joints/TCP if the port is reachable. This does not move the robot.")
    parser.add_argument("--robot-port", type=int, default=10000)
    parser.add_argument("--robot-timeout-s", type=float, default=3.0)
    parser.add_argument("--camera-warmup-frames", type=int, default=10)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "preflight_handeye_readiness.json")
    args = parser.parse_args()

    config = load_json(args.config)
    robot_ip = config["robot"]["ip_address"]

    report = {
        "timestamp": datetime.now().isoformat(),
        "moves_robot": False,
        "config": str(args.config),
        "imports": check_imports(),
        "files": check_files(config),
        "camera": None,
        "robot_port": None,
        "robot_state": None,
    }

    if not args.skip_camera:
        report["camera"] = check_camera(config, args.camera_warmup_frames)

    if not args.skip_robot_port:
        report["robot_port"] = check_robot_port(robot_ip, args.robot_port, args.robot_timeout_s)
        if args.read_robot_state and report["robot_port"]["success"]:
            report["robot_state"] = check_robot_state(robot_ip, args.robot_timeout_s, config)

    report["summary"] = summarize(report)
    write_json(args.output, report)

    for line in report["summary"]:
        print(line)
    print("Report:", args.output)

    hard_fail = False
    hard_fail = hard_fail or not all(item["success"] for item in report["imports"].values())
    hard_fail = hard_fail or not all(item["exists"] for item in report["files"].values())
    if report["camera"] is not None:
        hard_fail = hard_fail or not report["camera"]["success"]
    if report["robot_port"] is not None:
        hard_fail = hard_fail or not report["robot_port"]["success"]
    sys.exit(1 if hard_fail else 0)


if __name__ == "__main__":
    main()
