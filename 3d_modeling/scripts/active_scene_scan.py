from __future__ import annotations

import argparse
import json
import socket
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "active_scene_scan"
CONFIRM_TOKEN = "ACTIVE_SCENE_SCAN_OK"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def to_plain(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {key: to_plain(item) for key, item in value.items()}
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


def tcp_probe(ip: str, port: int, timeout_s: float) -> dict[str, Any]:
    start = datetime.now()
    try:
        with socket.create_connection((ip, port), timeout=timeout_s):
            elapsed_ms = (datetime.now() - start).total_seconds() * 1000.0
            return {"success": True, "elapsed_ms": elapsed_ms, "error": None}
    except Exception as exc:
        elapsed_ms = (datetime.now() - start).total_seconds() * 1000.0
        return {"success": False, "elapsed_ms": elapsed_ms, "error": f"{type(exc).__name__}: {exc}"}


def build_offsets(pattern: str, step_mm: float, z_step_mm: float) -> list[list[float]]:
    if pattern == "cross":
        offsets = [
            [0.0, 0.0, 0.0],
            [step_mm, 0.0, 0.0],
            [-step_mm, 0.0, 0.0],
            [0.0, step_mm, 0.0],
            [0.0, -step_mm, 0.0],
        ]
    elif pattern == "grid3x3":
        offsets = [[ix * step_mm, iy * step_mm, 0.0] for iy in [-1, 0, 1] for ix in [-1, 0, 1]]
    elif pattern == "grid3x3_z":
        offsets = [[ix * step_mm, iy * step_mm, 0.0] for iy in [-1, 0, 1] for ix in [-1, 0, 1]]
        offsets += [[0.0, 0.0, z_step_mm], [0.0, 0.0, -z_step_mm]]
    else:
        raise ValueError(f"Unsupported scan pattern: {pattern}")

    deduped: list[list[float]] = []
    seen: set[tuple[float, float, float]] = set()
    for offset in offsets:
        key = tuple(round(float(v), 9) for v in offset)
        if key not in seen:
            deduped.append(offset)
            seen.add(key)
    return deduped


def pose_delta_norm(a: list[float], b: list[float]) -> float:
    pa = np.asarray(a[:3], dtype=np.float64)
    pb = np.asarray(b[:3], dtype=np.float64)
    return float(np.linalg.norm(pb - pa))


def build_plan(current_pose: list[float], args: argparse.Namespace) -> list[dict[str, Any]]:
    offsets = build_offsets(args.pattern, args.step_mm, args.z_step_mm)
    plan = []
    for index, offset in enumerate(offsets):
        target = list(current_pose)
        target[0] = float(current_pose[0] + offset[0])
        target[1] = float(current_pose[1] + offset[1])
        target[2] = float(current_pose[2] + offset[2])
        plan.append(
            {
                "index": index,
                "name": f"{args.pattern}_{index:02d}",
                "offset_wrf_mm": [float(v) for v in offset],
                "target_pose_mm_deg": target,
                "radius_from_start_mm": pose_delta_norm(current_pose, target),
            }
        )
    if args.return_to_start:
        plan.append(
            {
                "index": len(plan),
                "name": "return_to_start",
                "offset_wrf_mm": [0.0, 0.0, 0.0],
                "target_pose_mm_deg": list(current_pose),
                "radius_from_start_mm": 0.0,
            }
        )
    return plan


def validate_plan(plan: list[dict[str, Any]], current_pose: list[float], args: argparse.Namespace) -> tuple[dict[str, bool], list[str]]:
    checks: dict[str, bool] = {}
    reasons: list[str] = []
    max_radius = max((record["radius_from_start_mm"] for record in plan), default=0.0)
    max_consecutive = 0.0
    prev = current_pose
    for record in plan:
        target = record["target_pose_mm_deg"]
        max_consecutive = max(max_consecutive, pose_delta_norm(prev, target))
        prev = target
    min_z = min((record["target_pose_mm_deg"][2] for record in plan), default=current_pose[2])
    orientation_unchanged = all(
        np.linalg.norm(np.asarray(record["target_pose_mm_deg"][3:], dtype=np.float64) - np.asarray(current_pose[3:], dtype=np.float64))
        <= 1e-9
        for record in plan
    )
    checks["max_radius_within_limit"] = max_radius <= args.max_radius_mm + 1e-9
    checks["max_consecutive_step_within_limit"] = max_consecutive <= args.max_consecutive_step_mm + 1e-9
    checks["min_z_above_limit"] = min_z >= args.min_link6_z_mm - 1e-9
    checks["orientation_unchanged"] = orientation_unchanged
    checks["confirm_token_matches"] = args.confirm_token == CONFIRM_TOKEN
    checks["execute_flag_present"] = bool(args.execute_scan)
    if not checks["max_radius_within_limit"]:
        reasons.append("Scan radius exceeds --max-radius-mm.")
    if not checks["max_consecutive_step_within_limit"]:
        reasons.append("Consecutive move exceeds --max-consecutive-step-mm.")
    if not checks["min_z_above_limit"]:
        reasons.append("A target link6 Z is below --min-link6-z-mm.")
    if not checks["orientation_unchanged"]:
        reasons.append("This scan script only supports translation-only targets.")
    if args.execute_scan and not checks["confirm_token_matches"]:
        reasons.append(f"Execution requested but --confirm-token is not {CONFIRM_TOKEN}.")
    return checks, reasons


def capture_image(camera_index: int, output_path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return {"success": False, "image": str(output_path), "error": f"Could not open camera index {camera_index}"}
    try:
        for _ in range(5):
            cap.read()
            time.sleep(0.03)
        ok, frame = cap.read()
        if not ok or frame is None:
            return {"success": False, "image": str(output_path), "error": "Camera read failed"}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        encoded_ok, encoded = cv2.imencode(".png", frame)
        if not encoded_ok:
            return {"success": False, "image": str(output_path), "error": "PNG encode failed"}
        encoded.tofile(str(output_path))
        return {"success": True, "image": str(output_path), "shape": list(frame.shape), "error": None}
    finally:
        cap.release()


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Active Scene Scan Report",
        "",
        f"Timestamp: `{report['timestamp']}`",
        f"Status: `{report['status']}`",
        f"Moves robot: `{report['moves_robot']}`",
        f"Pattern: `{report['parameters']['pattern']}`",
        f"Step mm: `{report['parameters']['step_mm']}`",
        "",
        "## Current Robot Pose",
        "",
        f"- Current pose mm/deg: `{report.get('current_pose_mm_deg')}`",
        f"- Current joints deg: `{report.get('current_joints_deg')}`",
        "",
        "## Plan",
        "",
        "| # | name | offset WRF mm | target pose mm/deg | radius mm |",
        "|---:|---|---|---|---:|",
    ]
    for record in report["plan"]:
        lines.append(
            f"| {record['index']} | `{record['name']}` | `{record['offset_wrf_mm']}` | "
            f"`{record['target_pose_mm_deg']}` | {record['radius_from_start_mm']:.3f} |"
        )
    lines += ["", "## Checks", ""]
    for key, value in report.get("checks", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    if report.get("refusal_reasons"):
        lines += ["", "## Refusal Reasons", ""]
        for reason in report["refusal_reasons"]:
            lines.append(f"- {reason}")
    if report.get("captures"):
        lines += ["", "## Captures", "", "| # | name | moved | image | pose |", "|---:|---|---|---|---|"]
        for item in report["captures"]:
            image = item.get("capture", {}).get("image")
            lines.append(
                f"| {item['index']} | `{item['name']}` | `{item.get('move_status')}` | "
                f"`{image}` | `{item.get('pose_after_mm_deg')}` |"
            )
    if report.get("error"):
        lines += ["", "## Error", "", f"- `{report['error']['type']}`: {report['error']['message']}"]
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview or execute an active multi-view scan with Meca500 and camera 0.")
    parser.add_argument("--ip", default="192.168.0.100")
    parser.add_argument("--port", type=int, default=10000)
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pattern", choices=["cross", "grid3x3", "grid3x3_z"], default="cross")
    parser.add_argument("--step-mm", type=float, default=1.0)
    parser.add_argument("--z-step-mm", type=float, default=0.5)
    parser.add_argument("--settle-s", type=float, default=0.4)
    parser.add_argument("--cart-lin-vel-mm-s", type=float, default=1.0)
    parser.add_argument("--max-radius-mm", type=float, default=2.0)
    parser.add_argument("--max-consecutive-step-mm", type=float, default=3.0)
    parser.add_argument("--min-link6-z-mm", type=float, default=20.0)
    parser.add_argument("--return-to-start", action="store_true")
    parser.add_argument("--execute-scan", action="store_true")
    parser.add_argument("--confirm-token", default="")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"active_scan_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "status": "preview",
        "moves_robot": False,
        "script": "active_scene_scan.py",
        "parameters": {
            "pattern": args.pattern,
            "step_mm": args.step_mm,
            "z_step_mm": args.z_step_mm,
            "camera_index": args.camera_index,
            "settle_s": args.settle_s,
            "cart_lin_vel_mm_s": args.cart_lin_vel_mm_s,
            "max_radius_mm": args.max_radius_mm,
            "max_consecutive_step_mm": args.max_consecutive_step_mm,
            "return_to_start": args.return_to_start,
            "execute_scan": args.execute_scan,
        },
        "network_probe": tcp_probe(args.ip, args.port, args.timeout_s),
        "plan": [],
        "checks": {},
        "refusal_reasons": [],
        "captures": [],
    }

    if not report["network_probe"]["success"]:
        report["status"] = "refused"
        report["refusal_reasons"].append("Robot TCP port is not reachable.")
        write_json(run_dir / "active_scene_scan_report.json", report)
        write_markdown(run_dir / "active_scene_scan_report.md", report)
        print("Active scene scan report:", run_dir)
        print("Status:", report["status"])
        return

    from mecademicpy.robot import Robot

    robot = Robot()
    try:
        robot.Connect(address=args.ip, enable_synchronous_mode=True, timeout=args.timeout_s)
        status = to_plain(robot.GetStatusRobot(synchronous_update=True, timeout=args.timeout_s))
        safety_status = to_plain(robot.GetSafetyStatus(synchronous_update=True, timeout=args.timeout_s))
        current_pose = to_plain(robot.GetPose())
        current_joints = to_plain(robot.GetJoints())
        report["robot_status_before"] = status
        report["robot_safety_status_before"] = safety_status
        report["current_pose_mm_deg"] = current_pose
        report["current_joints_deg"] = current_joints
        plan = build_plan(current_pose, args)
        checks, reasons = validate_plan(plan, current_pose, args)
        report["plan"] = plan
        report["checks"].update(checks)
        report["checks"].update(
            {
                "activation_state": bool(status.get("activation_state")) if isinstance(status, dict) else False,
                "homing_state": bool(status.get("homing_state")) if isinstance(status, dict) else False,
                "error_status_clear": not bool(status.get("error_status")) if isinstance(status, dict) else False,
                "brakes_released": not bool(status.get("brakes_engaged")) if isinstance(status, dict) else False,
            }
        )
        for key in ["activation_state", "homing_state", "error_status_clear", "brakes_released"]:
            if not report["checks"][key]:
                reasons.append(f"Robot readiness check failed: {key}")
        report["refusal_reasons"] = reasons

        if not args.execute_scan:
            report["status"] = "preview"
            report["moves_robot"] = False
            capture = capture_image(args.camera_index, run_dir / "preview_current_camera_rgb.png")
            report["preview_capture"] = capture
            return
        if reasons:
            report["status"] = "refused"
            report["moves_robot"] = False
            return

        robot.SetCartLinVel(args.cart_lin_vel_mm_s)
        report["status"] = "executing"
        report["moves_robot"] = True
        for record in plan:
            target = record["target_pose_mm_deg"]
            robot.MoveLin(*target)
            robot.WaitIdle()
            time.sleep(args.settle_s)
            pose_after = to_plain(robot.GetPose())
            joints_after = to_plain(robot.GetJoints())
            capture = capture_image(args.camera_index, run_dir / f"{record['index']:02d}_{record['name']}_camera_rgb.png")
            report["captures"].append(
                {
                    "index": record["index"],
                    "name": record["name"],
                    "target_pose_mm_deg": target,
                    "pose_after_mm_deg": pose_after,
                    "joints_after_deg": joints_after,
                    "move_status": "executed",
                    "capture": capture,
                }
            )
        report["status"] = "executed"
    except Exception as exc:
        report["status"] = "error"
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        try:
            robot.Disconnect()
        except Exception:
            pass
        write_json(run_dir / "active_scene_scan_report.json", report)
        write_markdown(run_dir / "active_scene_scan_report.md", report)
        print("Active scene scan report:", run_dir)
        print("Status:", report["status"])
        print("Moves robot:", report["moves_robot"])
        for reason in report.get("refusal_reasons", []):
            print("Refused:", reason)


if __name__ == "__main__":
    main()
