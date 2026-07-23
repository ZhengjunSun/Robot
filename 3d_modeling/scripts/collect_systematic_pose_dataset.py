#!/usr/bin/env python3
"""Systematic multi-position, multi-frame trocar pose dataset collector.

Moves the robot to predefined positions at varying distances and lateral offsets,
captures N frames at each position, and runs pose estimation on every frame.
This produces a dataset for analyzing pose estimation robustness and jitter.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import socket
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "systematic_pose_dataset"
CONFIRM_TOKEN = "SYSTEMATIC_POSE_DATASET_OK"

# Default camera config (colleague calibration)
DEFAULT_CAMERA_CONFIG = ROOT / "config" / "handeye_calibration_config.json"
DEFAULT_TROCAR_CONFIG = ROOT / "config" / "trocar_model_measured_20260612.json"
DEFAULT_EXTRINSIC = ROOT / "config" / "camera_extrinsic_colleague_20260612.json"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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
        return {k: to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_plain(value.tolist())
    if hasattr(value, "data"):
        return to_plain(value.data)
    if hasattr(value, "__dict__"):
        return {k: to_plain(v) for k, v in value.__dict__.items() if not k.startswith("_")}
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


def safe_imwrite(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise RuntimeError(f"Failed to encode image for {path}")
    encoded.tofile(str(path))


def build_position_plan(
    current_pose: list[float],
    distance_offsets_mm: list[float],
    lateral_offsets_mm: list[list[float]],
    frames_per_position: int,
) -> list[dict[str, Any]]:
    """Build plan: cartesian product of distance x lateral offsets, each with N frames."""
    plan = []
    idx = 0
    for d_off in distance_offsets_mm:
        for lat in lateral_offsets_mm:
            for frame_i in range(frames_per_position):
                target = list(current_pose)
                target[0] = float(current_pose[0] + lat[0])
                target[1] = float(current_pose[1] + lat[1])
                target[2] = float(current_pose[2] + d_off)
                radius = math.sqrt(
                    (target[0] - current_pose[0]) ** 2
                    + (target[1] - current_pose[1]) ** 2
                    + (target[2] - current_pose[2]) ** 2
                )
                plan.append({
                    "index": idx,
                    "position_index": len(plan) // frames_per_position,
                    "frame_index": frame_i,
                    "name": f"dZ{d_off:+.0f}_dX{lat[0]:+.0f}_dY{lat[1]:+.0f}_f{frame_i:02d}",
                    "distance_offset_mm": d_off,
                    "lateral_offset_mm": lat,
                    "target_pose_mm_deg": target,
                    "radius_from_start_mm": radius,
                })
                idx += 1
    return plan


def compress_plan(plan: list[dict]) -> list[dict]:
    """Compress plan to unique robot positions (one per position, not per frame)."""
    seen_targets = set()
    compressed = []
    for entry in plan:
        key = tuple(entry["target_pose_mm_deg"])
        if key not in seen_targets:
            seen_targets.add(key)
            compressed.append(entry)
    return compressed


def validate_plan(
    plan: list[dict],
    current_pose: list[float],
    max_radius_mm: float,
    max_consecutive_step_mm: float,
    min_link6_z_mm: float,
) -> tuple[dict[str, bool], list[str]]:
    checks: dict[str, bool] = {}
    reasons: list[str] = []

    max_radius = max((e["radius_from_start_mm"] for e in plan), default=0.0)
    compressed = compress_plan(plan)
    max_consecutive = 0.0
    prev = current_pose
    for e in compressed:
        target = e["target_pose_mm_deg"]
        step = math.sqrt(
            (target[0] - prev[0]) ** 2
            + (target[1] - prev[1]) ** 2
            + (target[2] - prev[2]) ** 2
        )
        max_consecutive = max(max_consecutive, step)
        prev = target

    min_z = min((e["target_pose_mm_deg"][2] for e in plan), default=current_pose[2])
    orientation_ok = all(
        math.sqrt(sum((e["target_pose_mm_deg"][3 + i] - current_pose[3 + i]) ** 2 for i in range(3))) <= 1e-9
        for e in plan
    )

    checks["max_radius_within_limit"] = max_radius <= max_radius_mm + 1e-9
    checks["max_consecutive_step_within_limit"] = max_consecutive <= max_consecutive_step_mm + 1e-9
    checks["min_z_above_limit"] = min_z >= min_link6_z_mm - 1e-9
    checks["orientation_unchanged"] = orientation_ok

    if not checks["max_radius_within_limit"]:
        reasons.append(f"Max radius {max_radius:.2f}mm exceeds limit {max_radius_mm}mm.")
    if not checks["max_consecutive_step_within_limit"]:
        reasons.append(f"Max consecutive step {max_consecutive:.2f}mm exceeds limit {max_consecutive_step_mm}mm.")
    if not checks["min_z_above_limit"]:
        reasons.append(f"Min Z {min_z:.2f}mm below limit {min_link6_z_mm}mm.")
    if not checks["orientation_unchanged"]:
        reasons.append("Orientation must remain unchanged (translation-only).")

    return checks, reasons


def capture_frame(camera_index: int, output_path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return {"success": False, "image": str(output_path), "error": f"Camera {camera_index} failed"}
    try:
        for _ in range(5):
            cap.read()
            time.sleep(0.03)
        ok, frame = cap.read()
        if not ok or frame is None:
            return {"success": False, "image": str(output_path), "error": "Camera read failed"}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        safe_imwrite(output_path, frame)
        return {"success": True, "image": str(output_path), "shape": list(frame.shape), "error": None}
    finally:
        cap.release()


def estimate_pose_for_frame(
    frame_path: Path,
    camera_cfg: dict,
    trocar_cfg: dict,
    detection_method: str = "auto",
) -> dict[str, Any]:
    """Run pose estimation on a single frame using functions from estimate_trocar_pose_from_ring."""
    from estimate_trocar_pose_from_ring import (
        detect_ring_with_fallback,
        build_correspondences,
        solve_pose,
        safe_imread,
    )
    from handeye_common import matrix_to_pose_mm_deg

    image = safe_imread(frame_path)
    if image is None:
        return {"status": "image_read_failed", "error": f"Cannot read {frame_path}"}

    try:
        outer_ellipse, outer_contour, outer_mask, method_used = detect_ring_with_fallback(
            image, trocar_cfg, detection_method=detection_method,
        )
    except RuntimeError as exc:
        return {"status": "detection_failed", "error": str(exc), "detection_method": None}

    try:
        inner_ellipse = None
        object_points, image_points, annotation = build_correspondences(
            trocar_cfg, outer_ellipse, inner_ellipse, use_inner_ring=False,
        )
        T_camera_trocar, metrics = solve_pose(object_points, image_points, camera_cfg)
        pose_mm_deg = matrix_to_pose_mm_deg(T_camera_trocar)
        return {
            "status": "ok",
            "detection_method": method_used,
            "pose_camera_trocar_mm_deg": pose_mm_deg,
            "T_camera_trocar": T_camera_trocar.tolist(),
            "translation_camera_trocar_m": T_camera_trocar[:3, 3].tolist(),
            "trocar_axis_camera": T_camera_trocar[:3, 2].tolist(),
            "mean_reprojection_error_px": metrics["mean_reprojection_error_px"],
            "max_reprojection_error_px": metrics["max_reprojection_error_px"],
            "outer_ellipse": {k: v for k, v in outer_ellipse.items() if k != "theta_rad"},
        }
    except Exception as exc:
        return {"status": "pose_solve_failed", "error": str(exc), "detection_method": method_used}


def write_dataset_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Systematic Pose Dataset Report",
        "",
        f"Timestamp: `{report['timestamp']}`",
        f"Status: `{report['status']}`",
        f"Moves robot: `{report.get('moves_robot', False)}`",
        "",
        "## Parameters",
        "",
        f"- Distance offsets (mm): `{report['parameters']['distance_offsets_mm']}`",
        f"- Lateral offsets (mm): `{report['parameters']['lateral_offsets_mm']}`",
        f"- Frames per position: `{report['parameters']['frames_per_position']}`",
        f"- Detection method: `{report['parameters']['detection_method']}`",
        f"- Total planned frames: `{report['parameters']['total_planned_frames']}`",
        "",
        "## Current Robot Pose",
        "",
        f"- Pose mm/deg: `{report.get('current_pose_mm_deg')}`",
        f"- Joints deg: `{report.get('current_joints_deg')}`",
        "",
        "## Plan Summary",
        "",
        f"- Unique positions: `{report.get('unique_position_count', 'N/A')}`",
        f"- Max radius: `{report.get('max_radius_mm', 'N/A')} mm`",
        f"- Max consecutive step: `{report.get('max_consecutive_step_mm', 'N/A')} mm`",
        "",
    ]

    if report.get("checks"):
        lines += ["", "## Safety Checks", ""]
        for key, value in report["checks"].items():
            lines.append(f"- `{key}`: `{value}`")
    if report.get("refusal_reasons"):
        lines += ["", "## Refusal Reasons", ""]
        for reason in report["refusal_reasons"]:
            lines.append(f"- {reason}")

    # Per-position summary
    positions = report.get("positions", [])
    if positions:
        lines += [
            "",
            "## Position Results",
            "",
            "| # | name | dZ mm | dX mm | dY mm | frames | detected | depth_mean mm | depth_std mm | reproj_mean px |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for pos in positions:
            det = pos.get("frames_detected", 0)
            total = pos.get("frames_captured", 0)
            depth_mean = pos.get("depth_mean_mm")
            depth_std = pos.get("depth_std_mm")
            reproj_mean = pos.get("reproj_mean_px")
            lines.append(
                f"| {pos.get('index', '?')} | `{pos.get('name', '?')}` | "
                f"{pos.get('distance_offset_mm', 0):+.1f} | "
                f"{pos.get('lateral_offset_mm', [0, 0])[0]:+.1f} | "
                f"{pos.get('lateral_offset_mm', [0, 0])[1]:+.1f} | "
                f"{det}/{total} | "
                f"{depth_mean:.3f} | {depth_std:.3f} | "
                f"{reproj_mean:.4f} |"
            )

    if report.get("error"):
        lines += ["", "## Error", "", f"- `{report['error']}`"]

    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_dataset_csv(path: Path, frames: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(frames[0].keys()))
        writer.writeheader()
        writer.writerows(frames)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect systematic multi-position, multi-frame trocar pose dataset.",
    )
    # Safety
    parser.add_argument("--execute-scan", action="store_true", help="Actually move the robot")
    parser.add_argument("--confirm-token", default="", help=f"Must be '{CONFIRM_TOKEN}' to execute")

    # Robot
    parser.add_argument("--ip", default="192.168.0.100")
    parser.add_argument("--port", type=int, default=10000)
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument("--settle-s", type=float, default=0.4)
    parser.add_argument("--cart-lin-vel-mm-s", type=float, default=1.0)

    # Position plan
    parser.add_argument(
        "--distance-offsets-mm", default="-3,-2,-1,0,1,2,3",
        help="Comma-separated Z offsets in mm",
    )
    parser.add_argument(
        "--lateral-offsets-mm", default="0,0 1,0 0,1",
        help="Space-separated X,Y pairs in mm",
    )
    parser.add_argument("--frames-per-position", type=int, default=10)
    parser.add_argument("--inter-frame-delay-s", type=float, default=0.1)

    # Safety limits
    parser.add_argument("--max-radius-mm", type=float, default=10.0)
    parser.add_argument("--max-consecutive-step-mm", type=float, default=3.0)
    parser.add_argument("--min-link6-z-mm", type=float, default=20.0)

    # Camera & detection
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--detection-method", choices=["color", "edge", "hough", "auto"], default="color")
    parser.add_argument("--camera-config", type=Path, default=DEFAULT_CAMERA_CONFIG)
    parser.add_argument("--trocar-config", type=Path, default=DEFAULT_TROCAR_CONFIG)
    parser.add_argument("--extrinsic", type=Path, default=DEFAULT_EXTRINSIC)

    # Output
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    # Parse offset lists
    distance_offsets = [float(x) for x in args.distance_offsets_mm.split(",")]
    lateral_offsets = []
    for pair in args.lateral_offsets_mm.split():
        parts = pair.split(",")
        lateral_offsets.append([float(parts[0]), float(parts[1])])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"systematic_dataset_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "status": "preview",
        "moves_robot": False,
        "script": "collect_systematic_pose_dataset.py",
        "parameters": {
            "distance_offsets_mm": distance_offsets,
            "lateral_offsets_mm": lateral_offsets,
            "frames_per_position": args.frames_per_position,
            "detection_method": args.detection_method,
            "total_planned_frames": len(distance_offsets) * len(lateral_offsets) * args.frames_per_position,
            "camera_index": args.camera_index,
            "settle_s": args.settle_s,
            "cart_lin_vel_mm_s": args.cart_lin_vel_mm_s,
        },
        "network_probe": tcp_probe(args.ip, args.port, args.timeout_s),
        "plan": [],
        "checks": {},
        "refusal_reasons": [],
        "positions": [],
        "frames": [],
    }

    # Network check
    if not report["network_probe"]["success"]:
        report["status"] = "refused"
        report["refusal_reasons"].append("Robot TCP port unreachable.")
        write_json(run_dir / "dataset_report.json", report)
        write_dataset_markdown(run_dir / "dataset_report.md", report)
        print("Report:", run_dir)
        print("Status:", report["status"])
        return

    from mecademicpy.robot import Robot

    robot = Robot()
    try:
        robot.Connect(address=args.ip, enable_synchronous_mode=True, timeout=args.timeout_s)
        status = to_plain(robot.GetStatusRobot(synchronous_update=True, timeout=args.timeout_s))
        safety = to_plain(robot.GetSafetyStatus(synchronous_update=True, timeout=args.timeout_s))
        current_pose = to_plain(robot.GetPose())
        current_joints = to_plain(robot.GetJoints())
        report["robot_status"] = status
        report["robot_safety"] = safety
        report["current_pose_mm_deg"] = current_pose
        report["current_joints_deg"] = current_joints

        # Build and validate plan
        plan = build_position_plan(
            current_pose, distance_offsets, lateral_offsets, args.frames_per_position,
        )
        compressed = compress_plan(plan)
        report["unique_position_count"] = len(compressed)
        report["max_radius_mm"] = max((e["radius_from_start_mm"] for e in plan), default=0.0)

        max_consec = 0.0
        prev = current_pose
        for e in compressed:
            t = e["target_pose_mm_deg"]
            step = math.sqrt(sum((t[i] - prev[i]) ** 2 for i in range(3)))
            max_consec = max(max_consec, step)
            prev = t
        report["max_consecutive_step_mm"] = max_consec

        checks, reasons = validate_plan(
            plan, current_pose,
            args.max_radius_mm, args.max_consecutive_step_mm, args.min_link6_z_mm,
        )
        report["checks"] = checks
        report["refusal_reasons"] = reasons

        # Robot readiness
        for key in ["activation_state", "homing_state", "error_status_clear", "brakes_released"]:
            if isinstance(status, dict):
                val = bool(status.get(
                    key if key != "error_status_clear" else "error_status",
                    False if key != "error_status_clear" else True,
                ))
                if key == "error_status_clear":
                    val = not bool(status.get("error_status", False))
                if key == "brakes_released":
                    val = not bool(status.get("brakes_engaged", True))
                report["checks"][key] = val
                if not val:
                    reasons.append(f"Robot not ready: {key}")

        # Preview mode
        if not args.execute_scan:
            report["status"] = "preview"
            report["moves_robot"] = False
            # Capture a preview image
            preview = capture_frame(args.camera_index, run_dir / "preview_current_camera_rgb.png")
            report["preview_capture"] = preview
            write_json(run_dir / "dataset_report.json", report)
            write_dataset_markdown(run_dir / "dataset_report.md", report)
            print("PREVIEW MODE - no robot motion.")
            print("Report:", run_dir)
            print(f"Positions: {len(compressed)}, Frames: {len(plan)}")
            for r in report["refusal_reasons"]:
                print(f"  Refused: {r}")
            return

        # Check confirmation token
        if args.confirm_token != CONFIRM_TOKEN:
            report["status"] = "refused"
            report["refusal_reasons"].append(
                f"Confirmation token mismatch. Required: '{CONFIRM_TOKEN}'"
            )
            write_json(run_dir / "dataset_report.json", report)
            write_dataset_markdown(run_dir / "dataset_report.md", report)
            print("REFUSED - wrong confirmation token.")
            return

        if report["refusal_reasons"]:
            report["status"] = "refused"
            write_json(run_dir / "dataset_report.json", report)
            write_dataset_markdown(run_dir / "dataset_report.md", report)
            print("REFUSED:")
            for r in report["refusal_reasons"]:
                print(f"  {r}")
            return

        # ===== EXECUTION =====
        robot.SetCartLinVel(args.cart_lin_vel_mm_s)
        report["status"] = "executing"
        report["moves_robot"] = True

        # Load configs for pose estimation
        camera_cfg = load_json(args.camera_config)
        trocar_cfg = load_json(args.trocar_config)

        all_frames_csv = []
        visited_positions: dict[tuple, list[dict]] = {}  # target_tuple -> list of frame results

        # Group plan entries by position
        positions_grouped: dict[int, list[dict]] = {}
        for entry in plan:
            pos_idx = entry["position_index"]
            positions_grouped.setdefault(pos_idx, []).append(entry)

        for pos_idx in sorted(positions_grouped.keys()):
            entries = positions_grouped[pos_idx]
            first_entry = entries[0]
            target = first_entry["target_pose_mm_deg"]
            pos_name = first_entry["name"].rsplit("_f", 1)[0]
            pos_dir = run_dir / f"pos_{pos_idx:03d}_{pos_name}"
            pos_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n[{pos_idx + 1}/{len(positions_grouped)}] Moving to {pos_name}...")
            print(f"  Target: {[f'{v:.2f}' for v in target]}")

            # Move robot
            robot.MoveLin(*target)
            robot.WaitIdle()
            time.sleep(args.settle_s)

            actual_pose = to_plain(robot.GetPose())
            actual_joints = to_plain(robot.GetJoints())

            pos_result = {
                "index": pos_idx,
                "name": pos_name,
                "distance_offset_mm": first_entry["distance_offset_mm"],
                "lateral_offset_mm": first_entry["lateral_offset_mm"],
                "target_pose_mm_deg": target,
                "actual_pose_mm_deg": actual_pose,
                "actual_joints_deg": actual_joints,
                "frames": [],
            }

            # Capture N frames
            for frame_entry in entries:
                fi = frame_entry["frame_index"]
                frame_path = pos_dir / f"frame_{fi:03d}.png"
                pose_path = pos_dir / f"frame_{fi:03d}_pose.json"

                # Capture image
                capture_result = capture_frame(args.camera_index, frame_path)

                # Run pose estimation
                pose_result = estimate_pose_for_frame(
                    frame_path, camera_cfg, trocar_cfg,
                    detection_method=args.detection_method,
                )

                # Save per-frame pose report
                frame_report = {
                    "frame_index": fi,
                    "position_index": pos_idx,
                    "position_name": pos_name,
                    "distance_offset_mm": first_entry["distance_offset_mm"],
                    "lateral_offset_mm": first_entry["lateral_offset_mm"],
                    "robot_pose_mm_deg": actual_pose,
                    "robot_joints_deg": actual_joints,
                    "capture": capture_result,
                    "pose_estimation": pose_result,
                }
                write_json(pose_path, frame_report)

                # Collect for CSV
                csv_row = {
                    "position_index": pos_idx,
                    "position_name": pos_name,
                    "distance_offset_mm": first_entry["distance_offset_mm"],
                    "lateral_x_mm": first_entry["lateral_offset_mm"][0],
                    "lateral_y_mm": first_entry["lateral_offset_mm"][1],
                    "frame_index": fi,
                    "capture_success": capture_result.get("success", False),
                    "detection_status": pose_result.get("status"),
                    "detection_method": pose_result.get("detection_method"),
                    "camera_x_mm": pose_result.get("pose_camera_trocar_mm_deg", [None])[0] if pose_result.get("pose_camera_trocar_mm_deg") else None,
                    "camera_y_mm": pose_result.get("pose_camera_trocar_mm_deg", [None, None])[1] if pose_result.get("pose_camera_trocar_mm_deg") else None,
                    "camera_z_mm": pose_result.get("pose_camera_trocar_mm_deg", [None, None, None])[2] if pose_result.get("pose_camera_trocar_mm_deg") else None,
                    "reprojection_error_px": pose_result.get("mean_reprojection_error_px"),
                }
                all_frames_csv.append(csv_row)
                pos_result["frames"].append(pose_result)

                status_str = pose_result.get("status", "?")
                z_mm = pose_result.get("pose_camera_trocar_mm_deg", [None, None, None])[2] if pose_result.get("pose_camera_trocar_mm_deg") else None
                if z_mm is not None:
                    print(f"  Frame {fi}: {status_str}, z={z_mm:.2f}mm, reproj={pose_result.get('mean_reprojection_error_px', '?'):.4f}px")
                else:
                    print(f"  Frame {fi}: {status_str}")

            # Compute per-position stats
            successful_poses = [
                f for f in pos_result["frames"]
                if f.get("status") == "ok" and f.get("pose_camera_trocar_mm_deg")
            ]
            depths = [f["pose_camera_trocar_mm_deg"][2] for f in successful_poses]
            reproj_errors = [f.get("mean_reprojection_error_px", 0) for f in successful_poses]
            pos_result["frames_captured"] = len(pos_result["frames"])
            pos_result["frames_detected"] = len(successful_poses)
            if depths:
                pos_result["depth_mean_mm"] = float(np.mean(depths))
                pos_result["depth_std_mm"] = float(np.std(depths))
                pos_result["reproj_mean_px"] = float(np.mean(reproj_errors))
            report["positions"].append(pos_result)

        # Save CSV
        write_dataset_csv(run_dir / "dataset_summary.csv", all_frames_csv)

        report["status"] = "completed"
        report["total_frames_collected"] = len(all_frames_csv)

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
        write_json(run_dir / "dataset_report.json", report)
        write_dataset_markdown(run_dir / "dataset_report.md", report)
        print("\n" + "=" * 60)
        print("Dataset report:", run_dir)
        print("Status:", report["status"])
        if report.get("total_frames_collected"):
            print(f"Total frames: {report['total_frames_collected']}")


if __name__ == "__main__":
    main()
