#!/usr/bin/env python3
"""Batch closed-loop alignment experiments on real Meca500 robot.

Supports 4 modes: 2d_center, 3d_one_step, 3d_pbvs, 3d_filtered.
Displaces robot to create misalignment, then runs alignment loop,
recording full trajectory and final accuracy.
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
DEFAULT_OUTPUT = ROOT / "outputs" / "alignment_experiments"
CONFIRM_TOKEN = "ALIGNMENT_EXPERIMENT_BATCH_OK"

DEFAULT_CAMERA_CONFIG = ROOT / "config" / "handeye_calibration_config.json"
DEFAULT_TROCAR_CONFIG = ROOT / "config" / "trocar_model_measured_20260612.json"
DEFAULT_EXTRINSIC = ROOT / "config" / "camera_extrinsic_colleague_20260612.json"


# ── Utilities ──────────────────────────────────────────────────────────────

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
    try:
        return list(value)
    except TypeError:
        return str(value)


def tcp_probe(ip: str, port: int, timeout_s: float) -> dict[str, Any]:
    start = datetime.now()
    try:
        with socket.create_connection((ip, port), timeout=timeout_s):
            return {"success": True, "elapsed_ms": (datetime.now() - start).total_seconds() * 1000.0, "error": None}
    except Exception as exc:
        return {"success": False, "elapsed_ms": (datetime.now() - start).total_seconds() * 1000.0, "error": str(exc)}


def safe_imwrite(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise RuntimeError(f"Failed to encode {path}")
    encoded.tofile(str(path))


# ── Pose Estimation ───────────────────────────────────────────────────────

def estimate_pose_from_camera(camera_index: int, camera_cfg: dict, trocar_cfg: dict, detection_method: str = "color") -> dict[str, Any]:
    """Capture image and run PnP pose estimation. Returns pose report dict."""
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return {"status": "camera_failed"}
    try:
        for _ in range(5):
            cap.read()
            time.sleep(0.03)
        ok, frame = cap.read()
        if not ok or frame is None:
            return {"status": "capture_failed"}
    finally:
        cap.release()

    from estimate_trocar_pose_from_ring import detect_ring_with_fallback, build_correspondences, solve_pose
    from handeye_common import matrix_to_pose_mm_deg

    try:
        ellipse, _, _, method_used = detect_ring_with_fallback(frame, trocar_cfg, detection_method=detection_method)
    except RuntimeError as exc:
        return {"status": "detection_failed", "error": str(exc)}

    try:
        obj_pts, img_pts, _ = build_correspondences(trocar_cfg, ellipse, None, use_inner_ring=False)
        T_cam_trocar, metrics = solve_pose(obj_pts, img_pts, camera_cfg)
        pose_mm = matrix_to_pose_mm_deg(T_cam_trocar)
        return {
            "status": "ok",
            "detection_method": method_used,
            "pose_camera_trocar_mm_deg": pose_mm,
            "T_camera_trocar": T_cam_trocar.tolist(),
            "trocar_axis_camera": T_cam_trocar[:3, 2].tolist(),
            "mean_reprojection_error_px": metrics["mean_reprojection_error_px"],
            "max_reprojection_error_px": metrics["max_reprojection_error_px"],
            "metrics": metrics,
            "outer_ellipse": {k: v for k, v in ellipse.items() if k != "theta_rad"},
            "ellipse_center": ellipse["center"],
        }
    except Exception as exc:
        return {"status": "pose_failed", "error": str(exc)}


def detect_ring_center_only(camera_index: int, trocar_cfg: dict) -> dict[str, Any]:
    """Capture image and detect orange ring center (2D mode). Returns center pixel coords."""
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return {"status": "camera_failed"}
    try:
        for _ in range(5):
            cap.read()
            time.sleep(0.03)
        ok, frame = cap.read()
        if not ok or frame is None:
            return {"status": "capture_failed"}
    finally:
        cap.release()

    from estimate_trocar_pose_from_ring import detect_outer_orange
    try:
        ellipse, _, _ = detect_outer_orange(frame)
        cx, cy = ellipse["center"]
        return {"status": "ok", "center_px": [cx, cy], "major_px": ellipse["major_diameter_px"]}
    except RuntimeError as exc:
        return {"status": "detection_failed", "error": str(exc)}


# ── Error Computation ─────────────────────────────────────────────────────

def axis_angle_deg(axis: list[float]) -> float:
    a = np.array(axis, dtype=np.float64)
    a = a / max(np.linalg.norm(a), 1e-12)
    return float(math.degrees(math.acos(float(np.clip(a[2], -1.0, 1.0)))))


def compute_errors(pose_mm_deg: list[float], target_distance_mm: float) -> dict:
    x, y, z = pose_mm_deg[0], pose_mm_deg[1], pose_mm_deg[2]
    axis = [0, 0, 1]  # will be overridden if available
    lateral = math.sqrt(x * x + y * y)
    depth = z - target_distance_mm
    return {
        "camera_x_mm": x, "camera_y_mm": y, "camera_z_mm": z,
        "lateral_error_mm": lateral,
        "depth_error_mm": depth,
        "axis_angle_error_deg": 0.0,
        "weighted_3d_error_mm": math.sqrt(lateral ** 2 + depth ** 2),
    }


def compute_errors_from_report(report: dict, target_distance_mm: float) -> dict:
    pose = report.get("pose_camera_trocar_mm_deg")
    if pose is None:
        return {"lateral_error_mm": float("inf"), "depth_error_mm": float("inf"),
                "axis_angle_error_deg": float("inf"), "weighted_3d_error_mm": float("inf")}
    x, y, z = pose[0], pose[1], pose[2]
    axis = report.get("trocar_axis_camera", [0, 0, 1])
    lateral = math.sqrt(x * x + y * y)
    depth = z - target_distance_mm
    aa = axis_angle_deg(axis)
    return {
        "camera_x_mm": x, "camera_y_mm": y, "camera_z_mm": z,
        "lateral_error_mm": lateral,
        "depth_error_mm": depth,
        "axis_angle_error_deg": aa,
        "weighted_3d_error_mm": math.sqrt(lateral ** 2 + depth ** 2 + (target_distance_mm * math.sin(math.radians(aa))) ** 2),
        "mean_reprojection_error_px": report.get("mean_reprojection_error_px"),
        "detection_method": report.get("detection_method"),
        "detection_status": report.get("status"),
    }


def build_quality_gate(args: argparse.Namespace):
    if args.disable_quality_gate:
        return None
    from pose_quality_gate import GateState, QualityGateThresholds

    return GateState(QualityGateThresholds(
        max_reprojection_error_px=args.gate_max_reprojection_error_px,
        max_translation_jump_mm=args.gate_max_translation_jump_mm,
        max_depth_jump_mm=args.gate_max_depth_jump_mm,
        max_axis_jump_deg=args.gate_max_axis_jump_deg,
    ))


def apply_quality_gate(obs: dict[str, Any], gate) -> dict[str, Any]:
    if gate is None or obs.get("status") != "ok":
        return {"accepted": True, "status": "disabled_or_not_applicable", "reasons": []}
    result = gate.update(obs)
    obs["quality_gate"] = result
    if not result.get("accepted"):
        obs["status"] = "quality_gate_rejected"
        obs["quality_gate_reasons"] = result.get("reasons", [])
    return result


# ── Control Law ────────────────────────────────────────────────────────────

def clip_vector(v: np.ndarray, max_norm: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n <= max_norm or n <= 1e-12 else v * (max_norm / n)


def compute_correction_base_mm(
    errors: dict, k_xy: float, k_z: float, max_step_mm: float,
    T_base_camera: np.ndarray,
) -> np.ndarray:
    """Camera-frame proportional correction converted to base frame. Translation only."""
    dx_cam = k_xy * errors["camera_x_mm"]
    dy_cam = k_xy * errors["camera_y_mm"]
    dz_cam = k_z * errors["depth_error_mm"]
    step_cam = clip_vector(np.array([dx_cam, dy_cam, dz_cam], dtype=np.float64), max_step_mm)
    step_base_m = T_base_camera[:3, :3] @ (step_cam / 1000.0)
    return step_base_m * 1000.0


def compute_2d_correction_base_mm(
    center_px: list[float], fx: float, fy: float, z_est_mm: float,
    max_step_mm: float, T_base_camera: np.ndarray,
) -> np.ndarray:
    """2D pixel-center correction converted to base frame."""
    img_cx, img_cy = 320.0, 240.0
    dx_px = center_px[0] - img_cx
    dy_px = center_px[1] - img_cy
    dx_mm = dx_px * z_est_mm / fx
    dy_mm = dy_px * z_est_mm / fy
    step_cam = clip_vector(np.array([dx_mm, dy_mm, 0.0], dtype=np.float64), max_step_mm)
    step_base_m = T_base_camera[:3, :3] @ (step_cam / 1000.0)
    return step_base_m * 1000.0


# ── Experiment Execution ──────────────────────────────────────────────────

def run_single_experiment(
    robot, camera_index: int, args: argparse.Namespace,
    experiment_id: int, mode: str, offset_mm: list[float],
    reference_pose: list[float], camera_cfg: dict, trocar_cfg: dict,
    T_link6_camera: np.ndarray,
) -> dict[str, Any]:
    """Run one alignment experiment. Robot is moved by offset, then aligned back."""
    from handeye_common import pose_mm_deg_to_matrix, project_transform_to_se3, transform_inverse
    from trocar_pose_filter import PoseHistory

    exp_start = time.time()
    displaced = list(reference_pose)
    displaced[0] += offset_mm[0]
    displaced[1] += offset_mm[1]
    displaced[2] += offset_mm[2]

    # Move to displaced position
    robot.MoveLin(*displaced)
    robot.WaitIdle()
    time.sleep(args.settle_s)
    actual_displaced = list(robot.GetPose())

    # Build T_base_camera for frame conversion
    T_base_link6 = pose_mm_deg_to_matrix(actual_displaced)
    T_base_camera = T_base_link6 @ T_link6_camera

    # Initial observation
    if mode in ("2d_center",):
        obs = detect_ring_center_only(camera_index, trocar_cfg)
    else:
        obs = estimate_pose_from_camera(camera_index, camera_cfg, trocar_cfg, args.detection_method)

    initial_errors = {}
    trajectory = []
    filter_history = PoseHistory(window_size=7, ema_alpha=0.4)
    quality_gate = build_quality_gate(args)
    z_est_2d = 34.0  # estimated Z for 2D Jacobian

    if mode != "2d_center":
        apply_quality_gate(obs, quality_gate)

    if obs.get("status") == "ok" and mode != "2d_center":
        initial_errors = compute_errors_from_report(obs, args.target_distance_mm)
        trajectory.append({"iteration": 0, "timestamp": datetime.now().isoformat(),
                           "robot_pose_mm_deg": actual_displaced, **initial_errors})
    elif mode == "2d_center" and obs.get("status") == "ok":
        cx, cy = obs["center_px"]
        px_off = math.sqrt((cx - 320) ** 2 + (cy - 240) ** 2)
        initial_errors = {"pixel_offset": px_off, "center_px": obs["center_px"]}
        pose_obs = estimate_pose_from_camera(camera_index, camera_cfg, trocar_cfg, args.detection_method)
        if pose_obs.get("status") == "ok":
            initial_errors.update(compute_errors_from_report(pose_obs, args.target_distance_mm))
            initial_errors["common_3d_eval_status"] = "ok"
        else:
            initial_errors["common_3d_eval_status"] = pose_obs.get("status")
        trajectory.append({"iteration": 0, "timestamp": datetime.now().isoformat(),
                           "robot_pose_mm_deg": actual_displaced, "pixel_offset": px_off,
                           "center_px": obs["center_px"], "detection_status": "ok"})
    else:
        initial_errors = {"detection_status": obs.get("status"), "error": obs.get("error")}
        trajectory.append({"iteration": 0, "detection_status": "failed",
                           "robot_pose_mm_deg": actual_displaced})

    converged = False
    current_pose = list(actual_displaced)

    # ── 3d_one_step: single step ──
    if mode == "3d_one_step":
        if obs.get("status") != "ok":
            total_time = time.time() - exp_start
            return {
                "experiment_id": experiment_id,
                "mode": mode,
                "offset_mm": offset_mm,
                "reference_pose_mm_deg": reference_pose,
                "displaced_pose_mm_deg": actual_displaced,
                "initial_errors": initial_errors,
                "final_errors": {"detection_status": obs.get("status"), "quality_gate": obs.get("quality_gate")},
                "converged": False,
                "iterations": len(trajectory) - 1,
                "total_time_s": round(total_time, 2),
                "trajectory": trajectory,
            }
        T_bl6 = pose_mm_deg_to_matrix(current_pose)
        T_bc = T_bl6 @ T_link6_camera
        step = compute_correction_base_mm(initial_errors, args.k_xy, args.k_z, args.max_step_mm, T_bc)
        new_pose = list(current_pose)
        new_pose[0] += step[0]
        new_pose[1] += step[1]
        new_pose[2] += step[2]
        if new_pose[2] < args.min_z_mm:
            new_pose[2] = args.min_z_mm
        robot.MoveLin(*new_pose)
        robot.WaitIdle()
        time.sleep(args.settle_s)
        current_pose = list(robot.GetPose())
        # Final observation
        obs = estimate_pose_from_camera(camera_index, camera_cfg, trocar_cfg, args.detection_method)
        final_errors = compute_errors_from_report(obs, args.target_distance_mm) if obs.get("status") == "ok" else {}
        converged = (final_errors.get("lateral_error_mm", 999) < 2.0)
        trajectory.append({**final_errors, "iteration": 1, "timestamp": datetime.now().isoformat(),
                           "robot_pose_mm_deg": current_pose})

    # ── Iterative modes ──
    elif mode in ("2d_center", "3d_pbvs", "3d_filtered"):
        for iteration in range(1, args.max_iterations + 1):
            # Capture
            if mode == "2d_center":
                obs = detect_ring_center_only(camera_index, trocar_cfg)
            else:
                obs = estimate_pose_from_camera(camera_index, camera_cfg, trocar_cfg, args.detection_method)

            if mode != "2d_center":
                apply_quality_gate(obs, quality_gate)

            if obs.get("status") != "ok":
                trajectory.append({"iteration": iteration, "detection_status": "failed",
                                   "robot_pose_mm_deg": current_pose,
                                   "quality_gate": obs.get("quality_gate"),
                                   "quality_gate_reasons": obs.get("quality_gate_reasons")})
                break

            # Compute errors
            if mode == "2d_center":
                cx, cy = obs["center_px"]
                px_off = math.sqrt((cx - 320) ** 2 + (cy - 240) ** 2)
                errors = {
                    "pixel_offset": px_off,
                    "center_px": obs["center_px"],
                    "detection_status": "ok",
                }
                if px_off < args.convergence_px:
                    converged = True
            else:
                errors = compute_errors_from_report(obs, args.target_distance_mm)
                if mode == "3d_filtered":
                    filter_result = filter_history.update(obs)
                    if filter_result.get("filtered_translation_camera_trocar_m"):
                        ft = filter_result["filtered_translation_camera_trocar_m"]
                        errors["camera_x_mm"] = ft[0] * 1000
                        errors["camera_y_mm"] = ft[1] * 1000
                        errors["camera_z_mm"] = ft[2] * 1000
                        errors["lateral_error_mm"] = math.sqrt(errors["camera_x_mm"] ** 2 + errors["camera_y_mm"] ** 2)
                        errors["depth_error_mm"] = errors["camera_z_mm"] - args.target_distance_mm
                        errors["confidence"] = filter_result.get("confidence", 0.0)
                        errors["filter_status"] = filter_result.get("status")
                        errors["filter_inlier_count"] = filter_result.get("inlier_count")
                    else:
                        errors["confidence"] = filter_result.get("confidence", 0.0)
                        errors["filter_status"] = filter_result.get("status")

                if (errors.get("lateral_error_mm", 999) < args.lateral_tol_mm
                        and abs(errors.get("depth_error_mm", 999)) < args.depth_tol_mm):
                    converged = True

            step_info = {"iteration": iteration, "timestamp": datetime.now().isoformat(),
                         "robot_pose_mm_deg": list(current_pose), **errors}

            if converged:
                trajectory.append(step_info)
                break

            # Compute correction
            T_bl6 = pose_mm_deg_to_matrix(current_pose)
            T_bc = T_bl6 @ T_link6_camera

            if mode == "2d_center":
                step = compute_2d_correction_base_mm(
                    obs["center_px"], camera_cfg["camera"]["fx"], camera_cfg["camera"]["fy"],
                    z_est_2d, args.max_step_mm, T_bc)
            else:
                step = compute_correction_base_mm(
                    errors, args.k_xy, args.k_z, args.max_step_mm, T_bc)

            step_norm = float(np.linalg.norm(step))
            step_info["correction_base_mm"] = step.tolist()
            step_info["link6_step_mm"] = step_norm
            trajectory.append(step_info)

            # Safety: don't move if step too small or Z too low
            if step_norm < 0.01:
                break
            new_pose = list(current_pose)
            new_pose[0] += step[0]
            new_pose[1] += step[1]
            new_pose[2] += step[2]
            if new_pose[2] < args.min_z_mm:
                new_pose[2] = args.min_z_mm

            # Max total displacement check
            disp_from_start = math.sqrt(sum((new_pose[i] - displaced[i]) ** 2 for i in range(3)))
            if disp_from_start > args.max_total_displacement_mm:
                break

            robot.MoveLin(*new_pose)
            robot.WaitIdle()
            time.sleep(args.inter_step_s)
            current_pose = list(robot.GetPose())

    total_time = time.time() - exp_start

    # Final observation
    if mode != "2d_center":
        obs = estimate_pose_from_camera(camera_index, camera_cfg, trocar_cfg, args.detection_method)
        final_gate = apply_quality_gate(obs, quality_gate)
        final_errors = compute_errors_from_report(obs, args.target_distance_mm) if obs.get("status") == "ok" else {"detection_status": obs.get("status")}
        final_errors["quality_gate"] = final_gate
    else:
        obs = detect_ring_center_only(camera_index, trocar_cfg)
        if obs.get("status") == "ok":
            cx, cy = obs["center_px"]
            final_errors = {"pixel_offset": math.sqrt((cx - 320) ** 2 + (cy - 240) ** 2)}
            pose_obs = estimate_pose_from_camera(camera_index, camera_cfg, trocar_cfg, args.detection_method)
            if pose_obs.get("status") == "ok":
                final_errors.update(compute_errors_from_report(pose_obs, args.target_distance_mm))
                final_errors["common_3d_eval_status"] = "ok"
            else:
                final_errors["common_3d_eval_status"] = pose_obs.get("status")
        else:
            final_errors = {"detection_status": obs.get("status")}

    return {
        "experiment_id": experiment_id,
        "mode": mode,
        "offset_mm": offset_mm,
        "reference_pose_mm_deg": reference_pose,
        "displaced_pose_mm_deg": actual_displaced,
        "initial_errors": initial_errors,
        "final_errors": final_errors,
        "converged": converged,
        "iterations": len(trajectory) - 1,
        "total_time_s": round(total_time, 2),
        "trajectory": trajectory,
    }


# ── Report Generation ────────────────────────────────────────────────────

def write_batch_markdown(path: Path, experiments: list[dict], args: argparse.Namespace) -> None:
    lines = [
        "# Alignment Experiment Batch Report",
        "",
        f"Generated: `{datetime.now().isoformat()}`",
        f"Total experiments: `{len(experiments)}`",
        "",
        "## Parameters",
        "",
        f"- Modes: `{args.modes}`",
        f"- Max iterations: `{args.max_iterations}`",
        f"- K_xy: {args.k_xy}, K_z: {args.k_z}",
        f"- Max step: `{args.max_step_mm}` mm",
        f"- Convergence: lateral < `{args.lateral_tol_mm}` mm, depth < `{args.depth_tol_mm}` mm",
        f"- Target distance: `{args.target_distance_mm}` mm",
        "",
        "## Per-Experiment Results",
        "",
        "| # | mode | offset mm | converged | iterations | time s | init_weighted | final_weighted | init_lateral | final_lateral |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for exp in experiments:
        ie = exp.get("initial_errors", {})
        fe = exp.get("final_errors", {})
        iw = ie.get("weighted_3d_error_mm", float("inf"))
        fw = fe.get("weighted_3d_error_mm", float("inf"))
        il = ie.get("lateral_error_mm", float("inf"))
        fl = fe.get("lateral_error_mm", float("inf"))
        off = exp["offset_mm"]
        lines.append(
            f"| {exp['experiment_id']} | `{exp['mode']}` | "
            f"[{off[0]:+.0f},{off[1]:+.0f},{off[2]:+.0f}] | "
            f"{exp['converged']} | {exp['iterations']} | {exp['total_time_s']} | "
            f"{iw:.2f} | {fw:.2f} | {il:.2f} | {fl:.2f} |"
        )

    # Per-mode summary
    lines += ["", "## Per-Mode Summary", ""]
    modes_seen = sorted(set(e["mode"] for e in experiments))
    for mode in modes_seen:
        mode_exps = [e for e in experiments if e["mode"] == mode]
        converged = sum(1 for e in mode_exps if e["converged"])
        total = len(mode_exps)
        iters = [e["iterations"] for e in mode_exps]
        finals = [e.get("final_errors", {}).get("weighted_3d_error_mm", float("inf")) for e in mode_exps]
        finals_clean = [v for v in finals if v < 100]
        lines.append(
            f"### `{mode}` ({converged}/{total} converged)",
        )
        lines.append(f"- Mean iterations: `{np.mean(iters):.1f}`")
        if finals_clean:
            lines.append(f"- Mean final weighted error: `{np.mean(finals_clean):.3f} mm`")
            lines.append(f"- Min final weighted error: `{np.min(finals_clean):.3f} mm`")

    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_batch_csv(path: Path, experiments: list[dict]) -> None:
    rows = []
    for e in experiments:
        ie = e.get("initial_errors", {})
        fe = e.get("final_errors", {})
        rows.append({
            "experiment_id": e["experiment_id"],
            "mode": e["mode"],
            "offset_x_mm": e["offset_mm"][0],
            "offset_y_mm": e["offset_mm"][1],
            "offset_z_mm": e["offset_mm"][2],
            "offset_norm_mm": math.sqrt(sum(v ** 2 for v in e["offset_mm"])),
            "converged": e["converged"],
            "iterations": e["iterations"],
            "total_time_s": e["total_time_s"],
            "initial_lateral_mm": ie.get("lateral_error_mm"),
            "initial_weighted_mm": ie.get("weighted_3d_error_mm"),
            "final_lateral_mm": fe.get("lateral_error_mm"),
            "final_depth_mm": fe.get("depth_error_mm"),
            "final_weighted_mm": fe.get("weighted_3d_error_mm"),
            "final_axis_deg": fe.get("axis_angle_error_deg"),
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Batch closed-loop alignment experiments on real Meca500.")
    parser.add_argument("--execute-experiments", action="store_true")
    parser.add_argument("--confirm-token", default="")
    parser.add_argument("--modes", default="3d_pbvs,3d_filtered,2d_center",
                        help="Comma-separated experiment modes")
    parser.add_argument("--offsets", default="3,0,0 -3,0,0 0,3,0 0,-3,0 5,0,0 -5,0,0 3,3,0 -3,-3,0",
                        help="Space-separated X,Y,Z offsets in mm")
    parser.add_argument("--repetitions", type=int, default=1)

    # Robot
    parser.add_argument("--ip", default="192.168.0.100")
    parser.add_argument("--port", type=int, default=10000)
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument("--cart-lin-vel-mm-s", type=float, default=2.0)
    parser.add_argument("--settle-s", type=float, default=0.4)
    parser.add_argument("--inter-step-s", type=float, default=0.3)

    # Control
    parser.add_argument("--k-xy", type=float, default=0.55)
    parser.add_argument("--k-z", type=float, default=0.45)
    parser.add_argument("--max-step-mm", type=float, default=1.0)
    parser.add_argument("--max-iterations", type=int, default=25)
    parser.add_argument("--max-total-displacement-mm", type=float, default=15.0)
    parser.add_argument("--min-z-mm", type=float, default=20.0)
    parser.add_argument("--target-distance-mm", type=float, default=34.0)
    parser.add_argument("--lateral-tol-mm", type=float, default=1.0)
    parser.add_argument("--depth-tol-mm", type=float, default=2.0)
    parser.add_argument("--convergence-px", type=float, default=5.0)

    # Camera & detection
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--detection-method", default="color")
    parser.add_argument("--disable-quality-gate", action="store_true")
    parser.add_argument("--gate-max-reprojection-error-px", type=float, default=0.35)
    parser.add_argument("--gate-max-translation-jump-mm", type=float, default=3.0)
    parser.add_argument("--gate-max-depth-jump-mm", type=float, default=2.0)
    parser.add_argument("--gate-max-axis-jump-deg", type=float, default=12.0)
    parser.add_argument("--camera-config", type=Path, default=DEFAULT_CAMERA_CONFIG)
    parser.add_argument("--trocar-config", type=Path, default=DEFAULT_TROCAR_CONFIG)
    parser.add_argument("--extrinsic", type=Path, default=DEFAULT_EXTRINSIC)

    # Output
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    # Parse
    modes = [m.strip() for m in args.modes.split(",")]
    offsets = []
    for pair in args.offsets.split():
        parts = pair.split(",")
        offsets.append([float(parts[0]), float(parts[1]), float(parts[2])])

    # Build experiment list
    experiments_plan = []
    eid = 0
    for rep in range(args.repetitions):
        for mode in modes:
            for off in offsets:
                experiments_plan.append({"id": eid, "mode": mode, "offset_mm": off, "rep": rep})
                eid += 1

    print(f"Experiment plan: {len(experiments_plan)} experiments")
    print(f"  Modes: {modes}")
    print(f"  Offsets: {len(offsets)}")
    print(f"  Repetitions: {args.repetitions}")

    if not args.execute_experiments:
        print("\nPREVIEW MODE. Add --execute-experiments and --confirm-token to run.")
        return

    if args.confirm_token != CONFIRM_TOKEN:
        print(f"Wrong token. Required: '{CONFIRM_TOKEN}'")
        return

    # Load configs
    from handeye_common import project_transform_to_se3
    camera_cfg = load_json(args.camera_config)
    trocar_cfg = load_json(args.trocar_config)
    extrinsic = load_json(args.extrinsic)
    T_link6_camera = project_transform_to_se3(np.array(extrinsic["T_link6_camera"], dtype=np.float64))

    # Network check
    probe = tcp_probe(args.ip, args.port, args.timeout_s)
    if not probe["success"]:
        print("Robot unreachable:", probe["error"])
        return
    print("Network: OK")

    # Connect
    from mecademicpy.robot import Robot
    robot = Robot()
    robot.Connect(address=args.ip, enable_synchronous_mode=True, timeout=args.timeout_s)
    robot.SetCartLinVel(args.cart_lin_vel_mm_s)

    status = to_plain(robot.GetStatusRobot(synchronous_update=True, timeout=args.timeout_s))
    if isinstance(status, dict):
        if not status.get("activation_state"):
            print("ERROR: Robot not activated.")
            robot.Disconnect()
            return
        if not status.get("homing_state"):
            print("ERROR: Robot not homed.")
            robot.Disconnect()
            return

    reference_pose = to_plain(robot.GetPose())
    print(f"Reference pose: {[f'{v:.2f}' for v in reference_pose]}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = args.output_dir / f"batch_{timestamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    results = []
    try:
        for i, plan in enumerate(experiments_plan):
            exp_label = f"[{i + 1}/{len(experiments_plan)}] {plan['mode']} offset={plan['offset_mm']}"
            print(f"\n{'=' * 60}")
            print(exp_label)
            print(f"{'=' * 60}")

            exp_result = run_single_experiment(
                robot, args.camera_index, args, plan["id"], plan["mode"], plan["offset_mm"],
                reference_pose, camera_cfg, trocar_cfg, T_link6_camera,
            )

            # Save per-experiment
            exp_path = batch_dir / f"experiment_{plan['id']:03d}_{plan['mode']}_off_{plan['offset_mm'][0]:+.0f}_{plan['offset_mm'][1]:+.0f}_{plan['offset_mm'][2]:+.0f}.json"
            write_json(exp_path, exp_result)

            converged_str = "CONVERGED" if exp_result["converged"] else "NOT CONVERGED"
            fw = exp_result.get("final_errors", {}).get("weighted_3d_error_mm", "?")
            fw_str = f"{fw:.2f}" if isinstance(fw, (int, float)) else str(fw)
            print(f"  {converged_str} | {exp_result['iterations']} iters | {exp_result['total_time_s']}s | final_weighted={fw_str}mm")
            results.append(exp_result)

    except Exception as exc:
        print(f"\nERROR: {exc}")
        traceback.print_exc()
    finally:
        robot.Disconnect()

    # Save batch summary
    params = dict((k, str(v)) for k, v in vars(args).items())
    write_json(batch_dir / "batch_summary.json", {"experiments": results, "parameters": params})
    write_batch_csv(batch_dir / "batch_summary.csv", results)
    write_batch_markdown(batch_dir / "batch_report.md", results, args)

    print(f"\n{'=' * 60}")
    print(f"Batch complete: {batch_dir}")
    print(f"Total experiments: {len(results)}")
    for mode in modes:
        mode_res = [r for r in results if r["mode"] == mode]
        conv = sum(1 for r in mode_res if r["converged"])
        print(f"  {mode}: {conv}/{len(mode_res)} converged")


if __name__ == "__main__":
    main()
