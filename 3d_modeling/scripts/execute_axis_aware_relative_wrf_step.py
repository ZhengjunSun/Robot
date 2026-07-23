from __future__ import annotations

import argparse
import json
import math
import socket
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs" / "control_3d" / "axis_aware_dry_run_after_step2_025deg" / "axis_aware_dry_run_candidates.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "control_3d" / "axis_aware_relative_wrf_execution"
CONFIRM_TOKEN = "MOVE_MECA500_ONE_RELATIVE_SMALL_STEP"


def load_json(path: Path) -> dict[str, Any]:
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
    return {
        "position_delta_mm": (pb - pa).tolist(),
        "position_delta_norm_mm": float(np.linalg.norm(pb - pa)),
        "orientation_raw_delta_deg": raw.tolist(),
        "orientation_raw_delta_norm_deg": float(np.linalg.norm(raw)),
        "orientation_wrapped_delta_deg": wrapped.tolist(),
        "orientation_wrapped_delta_norm_deg": float(np.linalg.norm(wrapped)),
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


def select_latest_candidate(dry_run: dict[str, Any], name: str) -> dict[str, Any]:
    latest = dry_run.get("latest", {})
    candidates = latest.get("candidates", [])
    for candidate in candidates:
        if candidate.get("name") == name:
            return candidate
    available = [candidate.get("name") for candidate in candidates]
    raise ValueError(f"Candidate {name!r} not found. Available: {available}")


def build_report(args: argparse.Namespace, dry_run: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    latest = dry_run.get("latest", {})
    risk = candidate.get("euler_risk", {})
    wrapped_delta = risk.get("wrapped_euler_delta_deg", [0.0, 0.0, 0.0])
    if not args.allow_orientation_step:
        wrapped_delta = [0.0, 0.0, 0.0]
    relative_wrf_command = {
        "x_mm": float(candidate["position_delta_base_link6_mm"][0]),
        "y_mm": float(candidate["position_delta_base_link6_mm"][1]),
        "z_mm": float(candidate["position_delta_base_link6_mm"][2]),
        "alpha_deg": float(wrapped_delta[0]),
        "beta_deg": float(wrapped_delta[1]),
        "gamma_deg": float(wrapped_delta[2]),
    }
    return {
        "timestamp": datetime.now().isoformat(),
        "status": "preview",
        "moves_robot": False,
        "script": "execute_axis_aware_relative_wrf_step.py",
        "input": str(args.input),
        "candidate_name": candidate.get("name"),
        "execute_requested": bool(args.execute_one_step),
        "confirm_token_ok": args.confirm_token == CONFIRM_TOKEN,
        "ip": args.ip,
        "port": args.port,
        "limits": {
            "max_link6_delta_mm": args.max_link6_delta_mm,
            "max_rotation_step_deg": args.max_rotation_step_deg,
            "max_current_pose_position_error_mm": args.max_current_pose_position_error_mm,
            "max_current_pose_orientation_error_deg": args.max_current_pose_orientation_error_deg,
            "cart_lin_vel_mm_s": args.cart_lin_vel_mm_s,
        },
        "latest": {
            "report": latest.get("report"),
            "image": latest.get("image"),
            "current_error": latest.get("current_error", {}),
            "best_candidate": latest.get("best_candidate"),
            "best_delta_weighted_3d_error_mm": latest.get("best_delta_weighted_3d_error_mm"),
        },
        "candidate": {
            "current_pose_base_link6_mm_deg": candidate.get("current_pose_base_link6_mm_deg"),
            "candidate_pose_base_link6_mm_deg": candidate.get("candidate_pose_base_link6_mm_deg"),
            "position_delta_base_link6_mm": candidate.get("position_delta_base_link6_mm"),
            "position_delta_norm_mm": candidate.get("position_delta_norm_mm"),
            "rotation_step_camera_rotvec_deg": candidate.get("rotation_step_camera_rotvec_deg"),
            "rotation_axis_base": candidate.get("rotation_axis_base"),
            "rotation_step_angle_deg": candidate.get("rotation_step_angle_deg"),
            "euler_risk": risk,
            "error_prediction": candidate.get("error_prediction"),
            "relative_wrf_command": relative_wrf_command,
        },
        "checks": {},
        "refusal_reasons": [],
    }


def refuse(report: dict[str, Any], reason: str) -> None:
    report["status"] = "refused"
    report.setdefault("refusal_reasons", []).append(reason)


def validate_static_gates(report: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    candidate = report["candidate"]
    risk = candidate.get("euler_risk", {})
    checks = report["checks"]
    position_delta_norm = float(candidate.get("position_delta_norm_mm", math.inf))
    wrapped_norm = float(risk.get("wrapped_euler_delta_norm_deg", math.inf))
    true_rot = float(risk.get("true_rotation_delta_deg", math.inf))
    checks["execution_link6_step_within_limit"] = position_delta_norm <= args.max_link6_delta_mm + 1e-9
    checks["wrapped_orientation_step_within_limit"] = wrapped_norm <= args.max_rotation_step_deg + 1e-9
    checks["true_rotation_step_within_limit"] = true_rot <= args.max_rotation_step_deg + 1e-9
    checks["safe_for_orientation_command"] = bool(risk.get("safe_for_orientation_command"))
    checks["wrapped_euler_jump_clear"] = not bool(risk.get("wrapped_euler_jump_risk"))
    checks["near_xyz_euler_singularity_clear"] = not bool(risk.get("near_xyz_euler_singularity"))
    checks["explicit_execute_flag_present"] = bool(args.execute_one_step)
    checks["confirmation_token_matches"] = args.confirm_token == CONFIRM_TOKEN
    checks["orientation_step_allowed"] = bool(args.allow_orientation_step)

    if not checks["execution_link6_step_within_limit"]:
        refuse(report, "Relative WRF link6 step exceeds --max-link6-delta-mm.")
    if args.allow_orientation_step and not checks["wrapped_orientation_step_within_limit"]:
        refuse(report, "Wrapped relative orientation step exceeds --max-rotation-step-deg.")
    if args.allow_orientation_step and not checks["true_rotation_step_within_limit"]:
        refuse(report, "True relative rotation step exceeds --max-rotation-step-deg.")
    if args.allow_orientation_step and not checks["safe_for_orientation_command"]:
        refuse(report, "Candidate is not marked safe for orientation command.")
    if args.allow_orientation_step and not checks["wrapped_euler_jump_clear"]:
        refuse(report, "Wrapped Euler delta has jump risk.")
    if args.allow_orientation_step and not checks["near_xyz_euler_singularity_clear"]:
        refuse(report, "Candidate is near XYZ Euler singularity.")
    if args.execute_one_step and not checks["confirmation_token_matches"]:
        refuse(report, f"Execution requested but --confirm-token does not match required token: {CONFIRM_TOKEN}")
    return report


def execute_if_allowed(report: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if not args.execute_one_step:
        report["moves_robot"] = False
        report["execution_note"] = "Preview only. Add --execute-one-step and the exact confirmation token to move the robot."
        return report
    if report["refusal_reasons"]:
        report["moves_robot"] = False
        return report

    probe = tcp_probe(args.ip, args.port, args.timeout_s)
    report["network_probe"] = probe
    report["checks"]["tcp_port_reachable"] = probe["success"]
    if not probe["success"]:
        refuse(report, "Robot TCP port is not reachable.")
        return report

    from mecademicpy.robot import Robot

    robot = Robot()
    try:
        robot.Connect(address=args.ip, enable_synchronous_mode=True, timeout=args.timeout_s)
        status = to_plain(robot.GetStatusRobot(synchronous_update=True, timeout=args.timeout_s))
        safety_status = to_plain(robot.GetSafetyStatus(synchronous_update=True, timeout=args.timeout_s))
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
            return report

        expected_current = report["candidate"]["current_pose_base_link6_mm_deg"]
        current_error = pose_delta_mm_deg(expected_current, current_pose_live)
        report["current_pose_vs_confirmation"] = current_error
        current_position_ok = current_error["position_delta_norm_mm"] <= args.max_current_pose_position_error_mm + 1e-9
        current_orientation_ok = (
            current_error["orientation_wrapped_delta_norm_deg"] <= args.max_current_pose_orientation_error_deg + 1e-9
        )
        report["checks"]["current_pose_position_matches_confirmation"] = current_position_ok
        report["checks"]["current_pose_orientation_matches_confirmation"] = current_orientation_ok
        if not current_position_ok:
            refuse(report, "Live robot pose does not match candidate current position.")
            return report
        if args.allow_orientation_step and not current_orientation_ok:
            refuse(report, "Live robot orientation does not match candidate current orientation.")
            return report

        command = report["candidate"]["relative_wrf_command"]
        robot.SetCartLinVel(args.cart_lin_vel_mm_s)
        report["moves_robot"] = True
        report["status"] = "executing"
        robot.MoveLinRelWrf(
            command["x_mm"],
            command["y_mm"],
            command["z_mm"],
            command["alpha_deg"],
            command["beta_deg"],
            command["gamma_deg"],
        )
        robot.WaitIdle()
        final_pose = to_plain(robot.GetPose())
        report["final_pose_live_mm_deg"] = final_pose
        report["final_pose_vs_expected_absolute_target"] = pose_delta_mm_deg(
            report["candidate"]["candidate_pose_base_link6_mm_deg"],
            final_pose,
        )
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


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    candidate = report["candidate"]
    command = candidate["relative_wrf_command"]
    risk = candidate.get("euler_risk", {})
    lines = [
        "# Axis-Aware Relative WRF 单步执行记录",
        "",
        f"生成时间：{report['timestamp']}",
        "",
        "## 状态",
        "",
        f"- 状态：`{report['status']}`",
        f"- 是否移动机械臂：`{report['moves_robot']}`",
        f"- 候选：`{report['candidate_name']}`",
        f"- 执行请求：`{report['execute_requested']}`",
        f"- 确认口令状态：`{'正确' if report['confirm_token_ok'] else '错误或未提供'}`",
        "",
        "## Relative WRF 命令",
        "",
        f"- x/y/z mm：`[{command['x_mm']:.6f}, {command['y_mm']:.6f}, {command['z_mm']:.6f}]`",
        f"- alpha/beta/gamma deg：`[{command['alpha_deg']:.6f}, {command['beta_deg']:.6f}, {command['gamma_deg']:.6f}]`",
        f"- link6 位移范数：`{candidate['position_delta_norm_mm']:.6f} mm`",
        f"- wrapped 欧拉角变化范数：`{risk.get('wrapped_euler_delta_norm_deg'):.6f} deg`",
        f"- true rotation delta：`{risk.get('true_rotation_delta_deg'):.6f} deg`",
        f"- raw euler jump risk：`{risk.get('raw_euler_jump_risk')}`",
        "",
        "## 绝对姿态参考",
        "",
        f"- 当前位姿：`{candidate['current_pose_base_link6_mm_deg']}`",
        f"- 候选绝对位姿：`{candidate['candidate_pose_base_link6_mm_deg']}`",
        "",
        "## 检查",
        "",
    ]
    for key, value in report["checks"].items():
        lines.append(f"- `{key}`: `{value}`")
    if report.get("network_probe"):
        lines += [
            "",
            "## 网络",
            "",
            f"- TCP 可达：`{report['network_probe']['success']}`",
            f"- 错误：`{report['network_probe']['error']}`",
        ]
    if report["refusal_reasons"]:
        lines += ["", "## 拒绝原因", ""]
        for reason in report["refusal_reasons"]:
            lines.append(f"- {reason}")
    if report.get("error"):
        lines += ["", "## 错误", "", f"- {report['error']['type']}: {report['error']['message']}"]
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview or execute one axis-aware relative WRF step on a real Meca500.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--candidate-name", default="combined_translation_rotation")
    parser.add_argument("--ip", default="192.168.0.100")
    parser.add_argument("--port", type=int, default=10000)
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--execute-one-step", action="store_true")
    parser.add_argument("--confirm-token", default="")
    parser.add_argument("--allow-orientation-step", action="store_true")
    parser.add_argument("--max-link6-delta-mm", type=float, default=0.8)
    parser.add_argument("--max-rotation-step-deg", type=float, default=0.3)
    parser.add_argument("--max-current-pose-position-error-mm", type=float, default=0.75)
    parser.add_argument("--max-current-pose-orientation-error-deg", type=float, default=0.3)
    parser.add_argument("--cart-lin-vel-mm-s", type=float, default=1.0)
    args = parser.parse_args()

    dry_run = load_json(args.input)
    candidate = select_latest_candidate(dry_run, args.candidate_name)
    report = build_report(args, dry_run, candidate)
    report = validate_static_gates(report, args)
    report = execute_if_allowed(report, args)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = args.output_dir / f"relative_wrf_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "relative_wrf_execution_report.json", report)
    write_markdown(run_dir / "relative_wrf_execution_report.md", report)

    print("Relative WRF execution report:", run_dir)
    print("Status:", report["status"])
    print("Moves robot:", report["moves_robot"])
    for reason in report.get("refusal_reasons", []):
        print("Refused:", reason)


if __name__ == "__main__":
    main()
