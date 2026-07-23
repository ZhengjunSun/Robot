from __future__ import annotations

import math
import socket
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from .config import PROJECT_ROOT, load_json, write_json


def to_plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
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


def wrap_angle_delta_deg(delta: np.ndarray) -> np.ndarray:
    return (delta + 180.0) % 360.0 - 180.0


def pose_delta_mm_deg(a: list[float], b: list[float]) -> dict[str, Any]:
    pa = np.asarray(a[:3], dtype=np.float64)
    pb = np.asarray(b[:3], dtype=np.float64)
    ra = np.asarray(a[3:], dtype=np.float64)
    rb = np.asarray(b[3:], dtype=np.float64)
    raw = rb - ra
    wrapped = wrap_angle_delta_deg(raw)
    true_rotation_delta_deg = float(
        math.degrees((R.from_euler("XYZ", ra, degrees=True).inv() * R.from_euler("XYZ", rb, degrees=True)).magnitude())
    )
    return {
        "position_delta_mm": [float(v) for v in (pb - pa)],
        "position_delta_norm_mm": float(np.linalg.norm(pb - pa)),
        "orientation_raw_delta_deg": [float(v) for v in raw],
        "orientation_raw_delta_norm_deg": float(np.linalg.norm(raw)),
        "orientation_wrapped_delta_deg": [float(v) for v in wrapped],
        "orientation_wrapped_delta_norm_deg": float(np.linalg.norm(wrapped)),
        "orientation_true_delta_deg": true_rotation_delta_deg,
    }


def tcp_probe(ip: str, port: int, timeout_s: float) -> dict[str, Any]:
    start = datetime.now()
    try:
        with socket.create_connection((ip, port), timeout=timeout_s):
            elapsed_ms = (datetime.now() - start).total_seconds() * 1000.0
            return {"success": True, "elapsed_ms": elapsed_ms, "error": None}
    except Exception as exc:
        elapsed_ms = (datetime.now() - start).total_seconds() * 1000.0
        return {"success": False, "elapsed_ms": elapsed_ms, "error": f"{type(exc).__name__}: {exc}"}


def timestamp_age_seconds(timestamp: Any) -> float | None:
    if not timestamp:
        return None
    try:
        value = datetime.fromisoformat(str(timestamp))
    except ValueError:
        return None
    return max(0.0, (datetime.now() - value).total_seconds())


def resolve_existing_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    candidate = PROJECT_ROOT / path
    return candidate if candidate.exists() else path


def select_predicted_candidate(camera_control: dict[str, Any], candidate_name: str) -> dict[str, Any] | None:
    ranking = camera_control.get("candidate_ranking") or {}
    for candidate in ranking.get("candidates") or []:
        if candidate.get("name") == candidate_name:
            return candidate
    return None


def _translation_only_target(command: dict[str, Any]) -> list[float] | None:
    current_pose = command.get("current_pose_base_link6_mm_deg")
    base_step = command.get("translation_step_base_mm")
    if not current_pose or not base_step:
        return None
    target = [float(v) for v in current_pose]
    target[:3] = [float(current_pose[i]) + float(base_step[i]) for i in range(3)]
    return target


def build_execution_plan(
    *,
    run_report_path: Path,
    config: dict[str, Any],
    candidate_name: str,
    execute_requested: bool,
    confirm_token: str,
    allow_orientation_step: bool,
    max_link6_delta_mm: float,
    max_rotation_step_deg: float,
    max_current_pose_position_error_mm: float,
    max_current_pose_orientation_error_deg: float,
    cart_lin_vel_mm_s: float,
    max_run_age_s: float,
    robot_ip: str,
    robot_port: int,
    timeout_s: float,
) -> dict[str, Any]:
    run_report_path = resolve_existing_path(run_report_path)
    run_report = load_json(run_report_path)
    robot_cfg = config.get("robot") or {}
    control_cfg = config.get("control") or {}
    quality_gate = run_report.get("quality_gate") or {}
    camera_control = run_report.get("camera_frame_control") or {}
    link6_control = run_report.get("link6_dry_run_control") or {}
    link6_command = link6_control.get("dry_run_command") or {}
    predicted_candidate = select_predicted_candidate(camera_control, candidate_name)
    run_age_s = timestamp_age_seconds(run_report.get("timestamp"))

    current_pose = link6_command.get("current_pose_base_link6_mm_deg")
    if candidate_name == "translation_only":
        execution_mode = "translation_only_keep_current_orientation"
        target_pose = _translation_only_target(link6_command)
    elif candidate_name == "combined_translation_rotation":
        execution_mode = "translation_and_orientation"
        target_pose = link6_command.get("candidate_pose_base_link6_mm_deg")
    else:
        execution_mode = "unsupported"
        target_pose = None

    pose_delta = None
    if current_pose and target_pose:
        pose_delta = pose_delta_mm_deg(current_pose, target_pose)

    return {
        "timestamp": datetime.now().isoformat(),
        "status": "preview",
        "moves_robot": False,
        "script": "execute_real_3d_step.py",
        "run_report": str(run_report_path),
        "config": str(config.get("_config_path")),
        "execute_requested": bool(execute_requested),
        "confirm_token_ok": confirm_token == str(robot_cfg.get("confirmation_token", "")),
        "config_execution_enabled": bool(robot_cfg.get("execution_enabled")),
        "robot": {
            "ip": robot_ip,
            "port": robot_port,
            "timeout_s": timeout_s,
        },
        "limits": {
            "max_link6_delta_mm": max_link6_delta_mm,
            "max_rotation_step_deg": max_rotation_step_deg,
            "rotation_numeric_tolerance_deg": 0.01,
            "max_current_pose_position_error_mm": max_current_pose_position_error_mm,
            "max_current_pose_orientation_error_deg": max_current_pose_orientation_error_deg,
            "cart_lin_vel_mm_s": cart_lin_vel_mm_s,
            "max_run_age_s": max_run_age_s,
            "min_link6_z_mm": float(control_cfg.get("min_link6_z_mm", -math.inf)),
            "min_predicted_improvement_mm": 0.0,
        },
        "source_run": {
            "status": run_report.get("status"),
            "timestamp": run_report.get("timestamp"),
            "age_seconds": run_age_s,
            "run_dir": run_report.get("run_dir"),
            "pose_report": run_report.get("pose_report"),
            "quality_gate": {
                "status": quality_gate.get("status"),
                "accepted": quality_gate.get("accepted"),
                "score": quality_gate.get("score"),
                "reasons": quality_gate.get("reasons") or [],
                "warnings": quality_gate.get("warnings") or [],
            },
            "camera_control_status": camera_control.get("status"),
            "camera_control_warnings": camera_control.get("warnings") or [],
            "link6_control_status": link6_control.get("status"),
            "link6_checks": link6_control.get("checks") or {},
        },
        "candidate": {
            "name": candidate_name,
            "execution_mode": execution_mode,
            "allow_orientation_step": bool(allow_orientation_step),
            "current_pose_base_link6_mm_deg": current_pose,
            "execution_target_pose_base_link6_mm_deg": target_pose,
            "execution_pose_delta": pose_delta,
            "translation_step_base_mm": link6_command.get("translation_step_base_mm"),
            "translation_step_camera_mm": link6_command.get("translation_step_camera_mm"),
            "rotation_step_camera_rotvec_deg": link6_command.get("rotation_step_camera_rotvec_deg"),
            "rotation_step_angle_deg": link6_command.get("rotation_step_angle_deg"),
            "predicted_effect": predicted_candidate,
        },
        "checks": {},
        "refusal_reasons": [],
        "execution_note": "",
    }


def refuse(report: dict[str, Any], reason: str) -> None:
    if reason not in report["refusal_reasons"]:
        report["refusal_reasons"].append(reason)


def validate_static_gates(report: dict[str, Any]) -> dict[str, Any]:
    source = report["source_run"]
    candidate = report["candidate"]
    checks = report["checks"]
    limits = report["limits"]
    predicted = candidate.get("predicted_effect") or {}
    pose_delta = candidate.get("execution_pose_delta") or {}
    target_pose = candidate.get("execution_target_pose_base_link6_mm_deg")
    link6_checks = source.get("link6_checks") or {}

    checks["dry_run_complete"] = source.get("status") == "dry_run_complete"
    checks["quality_gate_accepted"] = bool(source["quality_gate"].get("accepted"))
    checks["camera_control_status_ok"] = source.get("camera_control_status") == "ok"
    checks["camera_control_has_no_warnings"] = not bool(source.get("camera_control_warnings"))
    checks["link6_control_status_ok"] = source.get("link6_control_status") == "ok"
    checks["link6_static_checks_passed"] = bool(link6_checks.get("all_passed"))
    checks["candidate_supported"] = candidate.get("name") in {"translation_only", "combined_translation_rotation"}
    checks["target_pose_available"] = bool(target_pose and len(target_pose) == 6)
    checks["predicted_candidate_found"] = bool(predicted)
    checks["predicted_candidate_improves"] = bool(predicted.get("improved"))
    checks["predicted_delta_is_negative"] = float(predicted.get("delta_weighted_3d_error_mm", math.inf)) < 0.0
    checks["orientation_permission_matches_candidate"] = (
        candidate.get("name") != "combined_translation_rotation" or bool(candidate.get("allow_orientation_step"))
    )
    checks["execution_link6_step_within_limit"] = (
        float(pose_delta.get("position_delta_norm_mm", math.inf)) <= float(limits["max_link6_delta_mm"]) + 1e-9
    )
    checks["execution_rotation_step_within_limit"] = (
        float(pose_delta.get("orientation_true_delta_deg", math.inf))
        <= float(limits["max_rotation_step_deg"]) + float(limits["rotation_numeric_tolerance_deg"])
    )
    checks["candidate_link6_z_above_min"] = (
        bool(target_pose) and float(target_pose[2]) >= float(limits["min_link6_z_mm"])
    )
    max_run_age_s = float(limits.get("max_run_age_s", 0.0) or 0.0)
    run_age_s = source.get("age_seconds")
    checks["dry_run_age_within_execution_limit"] = (
        max_run_age_s <= 0.0 or (run_age_s is not None and float(run_age_s) <= max_run_age_s)
    )

    reason_map = {
        "dry_run_complete": "Source run_report is not a completed dry-run.",
        "quality_gate_accepted": "Quality gate is not accepted.",
        "camera_control_status_ok": "Camera-frame control status is not ok.",
        "camera_control_has_no_warnings": "Camera-frame control contains warnings.",
        "link6_control_status_ok": "Link6 dry-run control is not ok, usually because robot metadata is missing.",
        "link6_static_checks_passed": "Link6 dry-run static checks did not all pass.",
        "candidate_supported": "Candidate name is not supported by this bridge.",
        "target_pose_available": "Execution target pose is missing.",
        "predicted_candidate_found": "Selected camera-frame predicted candidate was not found.",
        "predicted_candidate_improves": "Selected candidate is not predicted to improve weighted 3D error.",
        "predicted_delta_is_negative": "Selected candidate predicted delta is not negative.",
        "orientation_permission_matches_candidate": "Combined translation+rotation requires --allow-orientation-step.",
        "execution_link6_step_within_limit": "Execution link6 step exceeds the configured limit.",
        "execution_rotation_step_within_limit": "Execution orientation step exceeds the configured limit.",
        "candidate_link6_z_above_min": "Execution target link6 z is below the configured minimum.",
    }
    for key, message in reason_map.items():
        if not checks.get(key):
            refuse(report, message)

    checks["explicit_execute_flag_present"] = bool(report["execute_requested"])
    checks["confirmation_token_matches"] = bool(report["confirm_token_ok"])
    checks["config_execution_enabled"] = bool(report["config_execution_enabled"])
    if report["execute_requested"]:
        if not checks["confirmation_token_matches"]:
            refuse(report, "Execution requested but confirmation token does not match config robot.confirmation_token.")
        if not checks["config_execution_enabled"]:
            refuse(report, "Execution requested but config robot.execution_enabled is false.")
        if not checks["dry_run_age_within_execution_limit"]:
            refuse(report, "Execution requested but source dry-run is older than max_run_age_s.")

    if report["execute_requested"]:
        report["status"] = "refused" if report["refusal_reasons"] else "ready_to_execute"
    else:
        report["status"] = "preview_blocked" if report["refusal_reasons"] else "preview_ready"
        report["execution_note"] = "Preview only. Add --execute-one-step, an enabled config, and the exact token to move the robot."
    return report


def execute_if_allowed(report: dict[str, Any]) -> dict[str, Any]:
    if not report["execute_requested"] or report["refusal_reasons"]:
        report["moves_robot"] = False
        return report

    robot_cfg = report["robot"]
    probe = tcp_probe(str(robot_cfg["ip"]), int(robot_cfg["port"]), float(robot_cfg["timeout_s"]))
    report["network_probe"] = probe
    report["checks"]["tcp_port_reachable"] = probe["success"]
    if not probe["success"]:
        refuse(report, "Robot TCP port is not reachable.")
        report["status"] = "refused"
        report["moves_robot"] = False
        return report

    from mecademicpy.robot import Robot

    robot = Robot()
    try:
        robot.Connect(address=robot_cfg["ip"], enable_synchronous_mode=True, timeout=robot_cfg["timeout_s"])
        status = to_plain(robot.GetStatusRobot(synchronous_update=True, timeout=robot_cfg["timeout_s"]))
        safety_status = to_plain(robot.GetSafetyStatus(synchronous_update=True, timeout=robot_cfg["timeout_s"]))
        current_pose_live = to_plain(robot.GetPose())
        report["robot_status_before"] = status
        report["robot_safety_status_before"] = safety_status
        report["current_pose_live_mm_deg"] = current_pose_live

        ready_checks = {
            "activation_state": bool(status.get("activation_state")) if isinstance(status, dict) else False,
            "homing_state": bool(status.get("homing_state")) if isinstance(status, dict) else False,
            "error_status_clear": not bool(status.get("error_status")) if isinstance(status, dict) else False,
            "brakes_released": not bool(status.get("brakes_engaged")) if isinstance(status, dict) else False,
        }
        report["checks"].update(ready_checks)
        for key, ok in ready_checks.items():
            if not ok:
                refuse(report, f"Robot readiness check failed: {key}")
        if report["refusal_reasons"]:
            report["status"] = "refused"
            return report

        expected_current = report["candidate"]["current_pose_base_link6_mm_deg"]
        if not current_pose_live or len(current_pose_live) != 6:
            refuse(report, "Live robot pose is missing or malformed.")
            report["status"] = "refused"
            return report

        current_error = pose_delta_mm_deg(expected_current, current_pose_live)
        report["current_pose_vs_dry_run"] = current_error
        position_ok = (
            current_error["position_delta_norm_mm"]
            <= float(report["limits"]["max_current_pose_position_error_mm"]) + 1e-9
        )
        orientation_ok = (
            current_error["orientation_true_delta_deg"]
            <= float(report["limits"]["max_current_pose_orientation_error_deg"]) + 1e-9
        )
        report["checks"]["current_pose_position_matches_dry_run"] = position_ok
        report["checks"]["current_pose_orientation_matches_dry_run"] = orientation_ok
        if not position_ok:
            refuse(report, "Live robot position does not match the dry-run current pose.")
        if not orientation_ok:
            refuse(report, "Live robot orientation does not match the dry-run current pose.")
        if report["refusal_reasons"]:
            report["status"] = "refused"
            return report

        target_pose = report["candidate"]["execution_target_pose_base_link6_mm_deg"]
        robot.SetCartLinVel(float(report["limits"]["cart_lin_vel_mm_s"]))
        report["status"] = "executing"
        report["moves_robot"] = True
        robot.MoveLin(*target_pose)
        robot.WaitIdle()
        final_pose = to_plain(robot.GetPose())
        report["final_pose_live_mm_deg"] = final_pose
        report["final_pose_vs_target"] = pose_delta_mm_deg(target_pose, final_pose)
        report["status"] = "executed"
        return report
    except Exception as exc:
        report["status"] = "error"
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        return report
    finally:
        try:
            robot.Disconnect()
        except Exception:
            pass


def write_execution_markdown(path: Path, report: dict[str, Any]) -> None:
    candidate = report["candidate"]
    delta = candidate.get("execution_pose_delta") or {}
    predicted = candidate.get("predicted_effect") or {}
    lines = [
        "# Real 3D One-Step Bridge Report",
        "",
        f"Generated: `{report['timestamp']}`",
        "",
        "## Status",
        "",
        f"- Status: `{report['status']}`",
        f"- Moves robot: `{report['moves_robot']}`",
        f"- Execute requested: `{report['execute_requested']}`",
        f"- Config execution enabled: `{report['config_execution_enabled']}`",
        f"- Confirmation token ok: `{report['confirm_token_ok']}`",
        "",
        "## Candidate",
        "",
        f"- Name: `{candidate['name']}`",
        f"- Mode: `{candidate['execution_mode']}`",
        f"- Current pose: `{candidate['current_pose_base_link6_mm_deg']}`",
        f"- Target pose: `{candidate['execution_target_pose_base_link6_mm_deg']}`",
        f"- Position delta norm: `{delta.get('position_delta_norm_mm')}` mm",
        f"- True orientation delta: `{delta.get('orientation_true_delta_deg')}` deg",
        f"- Wrapped Euler delta norm: `{delta.get('orientation_wrapped_delta_norm_deg')}` deg",
        f"- Predicted weighted-error delta: `{predicted.get('delta_weighted_3d_error_mm')}` mm",
        "",
        "## Checks",
        "",
    ]
    for key, value in report["checks"].items():
        lines.append(f"- `{key}`: `{value}`")
    if report["refusal_reasons"]:
        lines += ["", "## Refusal Reasons", ""]
        for reason in report["refusal_reasons"]:
            lines.append(f"- {reason}")
    if report.get("network_probe"):
        lines += [
            "",
            "## Network",
            "",
            f"- TCP reachable: `{report['network_probe']['success']}`",
            f"- Error: `{report['network_probe']['error']}`",
        ]
    if report.get("error"):
        lines += ["", "## Error", "", f"- {report['error']['type']}: {report['error']['message']}"]
    lines.append("")
    write_json(path.with_suffix(".json"), report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
