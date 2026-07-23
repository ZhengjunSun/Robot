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
DEFAULT_CONFIRMATION = ROOT / "outputs" / "control_3d" / "real_step_confirmation" / "real_step_confirmation.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "control_3d" / "real_step_execution"
CONFIRM_TOKEN = "MOVE_MECA500_ONE_SMALL_STEP"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
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


def tcp_probe(ip: str, port: int, timeout_s: float) -> dict:
    start = datetime.now()
    try:
        with socket.create_connection((ip, port), timeout=timeout_s):
            elapsed_ms = (datetime.now() - start).total_seconds() * 1000.0
            return {"success": True, "elapsed_ms": elapsed_ms, "error": None}
    except Exception as exc:
        elapsed_ms = (datetime.now() - start).total_seconds() * 1000.0
        return {"success": False, "elapsed_ms": elapsed_ms, "error": f"{type(exc).__name__}: {exc}"}


def pose_delta_mm_deg(a: list[float], b: list[float]) -> dict:
    pa = np.asarray(a[:3], dtype=np.float64)
    pb = np.asarray(b[:3], dtype=np.float64)
    ra = np.asarray(a[3:], dtype=np.float64)
    rb = np.asarray(b[3:], dtype=np.float64)
    return {
        "position_delta_mm": (pb - pa).tolist(),
        "position_delta_norm_mm": float(np.linalg.norm(pb - pa)),
        "orientation_euler_delta_deg": (rb - ra).tolist(),
        "orientation_euler_delta_norm_deg": float(np.linalg.norm(rb - ra)),
    }


def select_candidate(sheet: dict, one_based_index: int) -> dict:
    candidates = sheet.get("candidates", [])
    if one_based_index < 1 or one_based_index > len(candidates):
        raise ValueError(f"Candidate index {one_based_index} out of range. Available: {len(candidates)}")
    return candidates[one_based_index - 1]


def status_text(value: Any) -> str:
    return str(to_plain(value))


def refuse(report: dict, reason: str) -> dict:
    report["status"] = "refused"
    report.setdefault("refusal_reasons", []).append(reason)
    return report


def build_report(args: argparse.Namespace, sheet: dict, candidate: dict) -> dict:
    command = candidate["dry_run_command"]
    current_pose = command["current_pose_base_link6_mm_deg"]
    requested_target_pose = command["candidate_pose_base_link6_mm_deg"]
    execution_target_pose = list(requested_target_pose)
    execution_mode = "translation_and_orientation"
    if not args.allow_orientation_step:
        execution_mode = "translation_only_keep_current_orientation"
        base_step = np.asarray(command["translation_step_base_mm"], dtype=np.float64)
        execution_target_pose = list(current_pose)
        execution_target_pose[:3] = (np.asarray(current_pose[:3], dtype=np.float64) + base_step).tolist()
    return {
        "timestamp": datetime.now().isoformat(),
        "status": "preview",
        "moves_robot": False,
        "script": "execute_3d_alignment_one_step.py",
        "confirmation_sheet": str(args.confirmation),
        "candidate_index": args.candidate_index,
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
        "candidate": {
            "image": candidate.get("image"),
            "current_pose_base_link6_mm_deg": current_pose,
            "requested_target_pose_base_link6_mm_deg": requested_target_pose,
            "execution_target_pose_base_link6_mm_deg": execution_target_pose,
            "execution_mode": execution_mode,
            "requested_pose_delta": pose_delta_mm_deg(current_pose, requested_target_pose),
            "execution_pose_delta": pose_delta_mm_deg(current_pose, execution_target_pose),
            "camera_step_mm": command["translation_step_camera_mm"],
            "base_step_mm": command["translation_step_base_mm"],
            "requested_rotation_step_angle_deg": command["rotation_step_angle_deg"],
            "requested_link6_delta_norm_mm": command["link6_delta_norm_mm"],
        },
        "checks": {},
        "refusal_reasons": [],
    }


def validate_static_gates(report: dict, args: argparse.Namespace) -> dict:
    candidate = report["candidate"]
    checks = report["checks"]
    checks["execution_link6_step_within_limit"] = (
        candidate["execution_pose_delta"]["position_delta_norm_mm"] <= args.max_link6_delta_mm + 1e-9
    )
    checks["execution_orientation_step_within_limit"] = (
        candidate["execution_pose_delta"]["orientation_euler_delta_norm_deg"] <= args.max_rotation_step_deg + 1e-9
    )
    checks["dry_run_rotation_step_within_limit"] = (
        candidate["requested_rotation_step_angle_deg"] <= args.max_rotation_step_deg + 1e-9
    )
    checks["explicit_execute_flag_present"] = bool(args.execute_one_step)
    checks["confirmation_token_matches"] = args.confirm_token == CONFIRM_TOKEN
    checks["orientation_step_allowed"] = bool(args.allow_orientation_step)

    if not checks["execution_link6_step_within_limit"]:
        refuse(report, "Execution link6 step exceeds --max-link6-delta-mm.")
    if not checks["execution_orientation_step_within_limit"]:
        refuse(report, "Execution orientation step exceeds --max-rotation-step-deg.")
    if args.allow_orientation_step and not checks["dry_run_rotation_step_within_limit"]:
        refuse(report, "Requested dry-run rotation step exceeds --max-rotation-step-deg.")
    if args.execute_one_step and not checks["confirmation_token_matches"]:
        refuse(report, f"Execution requested but --confirm-token does not match required token: {CONFIRM_TOKEN}")
    if not args.execute_one_step:
        report["status"] = "preview"
    return report


def execute_if_allowed(report: dict, args: argparse.Namespace) -> dict:
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
        status = robot.GetStatusRobot(synchronous_update=True, timeout=args.timeout_s)
        safety_status = robot.GetSafetyStatus(synchronous_update=True, timeout=args.timeout_s)
        status_plain = to_plain(status)
        safety_plain = to_plain(safety_status)
        current_pose_live = to_plain(robot.GetPose())
        report["robot_status_before"] = status_plain
        report["robot_safety_status_before"] = safety_plain
        report["current_pose_live_mm_deg"] = current_pose_live
        robot_ready_checks = {
            "activation_state": bool(status_plain.get("activation_state")) if isinstance(status_plain, dict) else False,
            "homing_state": bool(status_plain.get("homing_state")) if isinstance(status_plain, dict) else False,
            "error_status_clear": not bool(status_plain.get("error_status")) if isinstance(status_plain, dict) else False,
            "brakes_released": not bool(status_plain.get("brakes_engaged")) if isinstance(status_plain, dict) else False,
        }
        report["checks"].update(robot_ready_checks)
        if not robot_ready_checks["activation_state"]:
            refuse(report, "Robot is not activated.")
        if not robot_ready_checks["homing_state"]:
            refuse(report, "Robot is not homed.")
        if not robot_ready_checks["error_status_clear"]:
            refuse(report, "Robot error status is not clear.")
        if not robot_ready_checks["brakes_released"]:
            refuse(report, "Robot brakes are engaged.")
        if report["refusal_reasons"]:
            return report

        expected_current = report["candidate"]["current_pose_base_link6_mm_deg"]
        current_error = pose_delta_mm_deg(expected_current, current_pose_live)
        report["current_pose_vs_confirmation"] = current_error
        current_position_ok = current_error["position_delta_norm_mm"] <= args.max_current_pose_position_error_mm + 1e-9
        current_orientation_ok = (
            current_error["orientation_euler_delta_norm_deg"] <= args.max_current_pose_orientation_error_deg + 1e-9
        )
        report["checks"]["current_pose_position_matches_confirmation"] = current_position_ok
        report["checks"]["current_pose_orientation_matches_confirmation"] = current_orientation_ok
        if not current_position_ok:
            refuse(report, "Live robot pose does not match the confirmation sheet current pose.")
            return report
        if args.allow_orientation_step and not current_orientation_ok:
            refuse(report, "Live robot orientation does not match the confirmation sheet current orientation.")
            return report

        target_pose = report["candidate"]["execution_target_pose_base_link6_mm_deg"]
        robot.SetCartLinVel(args.cart_lin_vel_mm_s)
        report["moves_robot"] = True
        report["status"] = "executing"
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


def write_markdown(path: Path, report: dict) -> None:
    candidate = report["candidate"]
    token_text = "未请求执行，预览模式无需口令"
    if report["execute_requested"]:
        token_text = "正确" if report["confirm_token_ok"] else "错误"
    lines = [
        "# 真实 Meca500 3D 对齐单步执行记录",
        "",
        f"生成时间：{report['timestamp']}",
        "",
        "## 状态",
        "",
        f"- 状态：`{report['status']}`",
        f"- 是否移动机械臂：`{report['moves_robot']}`",
        f"- 候选序号：`{report['candidate_index']}`",
        f"- 执行请求：`{report['execute_requested']}`",
        f"- 确认口令状态：`{token_text}`",
        f"- 执行模式：`{candidate['execution_mode']}`",
        "",
        "## 候选动作",
        "",
        f"- 当前确认表位姿：`{candidate['current_pose_base_link6_mm_deg']}`",
        f"- dry-run 原始目标位姿：`{candidate['requested_target_pose_base_link6_mm_deg']}`",
        f"- 实际执行目标位姿：`{candidate['execution_target_pose_base_link6_mm_deg']}`",
        f"- 实际执行 link6 位移范数：`{candidate['execution_pose_delta']['position_delta_norm_mm']:.6f} mm`",
        f"- 实际执行欧拉角变化范数：`{candidate['execution_pose_delta']['orientation_euler_delta_norm_deg']:.6f} deg`",
        f"- dry-run 姿态步长：`{candidate['requested_rotation_step_angle_deg']:.6f} deg`",
        f"- 相机坐标步长：`{candidate['camera_step_mm']}`",
        f"- 基坐标步长：`{candidate['base_step_mm']}`",
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
    parser = argparse.ArgumentParser(description="Preview or execute one gated 3D alignment step on a real Meca500.")
    parser.add_argument("--confirmation", type=Path, default=DEFAULT_CONFIRMATION)
    parser.add_argument("--candidate-index", type=int, default=1)
    parser.add_argument("--ip", default="192.168.0.100")
    parser.add_argument("--port", type=int, default=10000)
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--execute-one-step", action="store_true")
    parser.add_argument("--confirm-token", default="")
    parser.add_argument(
        "--allow-orientation-step",
        action="store_true",
        help="Also execute the dry-run orientation step. By default only the translation step is executed.",
    )
    parser.add_argument("--max-link6-delta-mm", type=float, default=2.0)
    parser.add_argument("--max-rotation-step-deg", type=float, default=2.0)
    parser.add_argument("--max-current-pose-position-error-mm", type=float, default=0.75)
    parser.add_argument("--max-current-pose-orientation-error-deg", type=float, default=0.5)
    parser.add_argument("--cart-lin-vel-mm-s", type=float, default=2.0)
    args = parser.parse_args()

    sheet = load_json(args.confirmation)
    candidate = select_candidate(sheet, args.candidate_index)
    report = build_report(args, sheet, candidate)
    report = validate_static_gates(report, args)
    report = execute_if_allowed(report, args)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"one_step_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "one_step_execution_report.json", report)
    write_markdown(run_dir / "one_step_execution_report.md", report)

    print("One-step execution report:", run_dir)
    print("Status:", report["status"])
    print("Moves robot:", report["moves_robot"])
    for reason in report.get("refusal_reasons", []):
        print("Refused:", reason)


if __name__ == "__main__":
    main()
