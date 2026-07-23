from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "control_3d" / "robot_prepare"
CONFIRM_TOKEN = "PREPARE_MECA500_FOR_SMALL_STEP"


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


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def snapshot(robot, timeout_s: float) -> dict:
    status = to_plain(robot.GetStatusRobot(synchronous_update=True, timeout=timeout_s))
    safety_status = to_plain(robot.GetSafetyStatus(synchronous_update=True, timeout=timeout_s))
    pose = to_plain(robot.GetPose())
    return {
        "timestamp": datetime.now().isoformat(),
        "status": status,
        "safety_status": safety_status,
        "pose_mm_deg": pose,
    }


def status_flag(snapshot_data: dict, name: str) -> bool:
    status = snapshot_data.get("status") or {}
    return bool(status.get(name))


def prepare_robot(args: argparse.Namespace) -> dict:
    report = {
        "timestamp": datetime.now().isoformat(),
        "moves_robot": False,
        "ip": args.ip,
        "execute_requested": bool(args.execute_prepare),
        "confirm_token_ok": args.confirm_token == CONFIRM_TOKEN,
        "status": "preview",
        "steps": [],
        "refusal_reasons": [],
    }
    if not args.execute_prepare:
        report["refusal_reasons"].append("Preview only. Add --execute-prepare and confirmation token to activate/home.")
        return report
    if args.confirm_token != CONFIRM_TOKEN:
        report["status"] = "refused"
        report["refusal_reasons"].append(f"Confirmation token mismatch. Required: {CONFIRM_TOKEN}")
        return report

    from mecademicpy.robot import Robot

    robot = Robot()
    try:
        robot.Connect(address=args.ip, enable_synchronous_mode=True, timeout=args.timeout_s)
        before = snapshot(robot, args.timeout_s)
        report["before"] = before

        if (before.get("status") or {}).get("error_status"):
            robot.ClearError()
            report["steps"].append("ClearError")
            robot.WaitErrorReset(timeout=args.timeout_s)

        if not status_flag(before, "activation_state"):
            robot.ActivateRobot()
            report["moves_robot"] = True
            report["steps"].append("ActivateRobot")
            robot.WaitActivated(timeout=args.activation_timeout_s)

        after_activation = snapshot(robot, args.timeout_s)
        report["after_activation"] = after_activation

        if not status_flag(after_activation, "homing_state"):
            try:
                robot.SetJointVel(args.home_joint_vel_deg_s)
                report["steps"].append(f"SetJointVel({args.home_joint_vel_deg_s})")
            except Exception as exc:
                report["steps"].append(f"SetJointVel skipped: {type(exc).__name__}: {exc}")
            robot.Home()
            report["moves_robot"] = True
            report["steps"].append("Home")
            robot.WaitHomed(timeout=args.home_timeout_s)

        after = snapshot(robot, args.timeout_s)
        report["after"] = after
        ready = {
            "activation_state": status_flag(after, "activation_state"),
            "homing_state": status_flag(after, "homing_state"),
            "error_status_clear": not status_flag(after, "error_status"),
            "brakes_released": not status_flag(after, "brakes_engaged"),
        }
        report["ready_checks"] = ready
        report["status"] = "ready" if all(ready.values()) else "not_ready"
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
    lines = [
        "# Meca500 执行前准备记录",
        "",
        f"生成时间：{report['timestamp']}",
        "",
        f"- 状态：`{report['status']}`",
        f"- 是否可能移动机械臂：`{report['moves_robot']}`",
        f"- 执行请求：`{report['execute_requested']}`",
        f"- 口令正确：`{report['confirm_token_ok']}`",
        "",
        "## 步骤",
        "",
    ]
    for step in report.get("steps", []):
        lines.append(f"- {step}")
    if not report.get("steps"):
        lines.append("- 无")
    if report.get("ready_checks"):
        lines += ["", "## Ready 检查", ""]
        for key, value in report["ready_checks"].items():
            lines.append(f"- `{key}`: `{value}`")
    if report.get("before"):
        lines += ["", "## 准备前位姿", "", f"`{report['before'].get('pose_mm_deg')}`"]
    if report.get("after"):
        lines += ["", "## 准备后位姿", "", f"`{report['after'].get('pose_mm_deg')}`"]
    if report.get("refusal_reasons"):
        lines += ["", "## 拒绝原因", ""]
        for reason in report["refusal_reasons"]:
            lines.append(f"- {reason}")
    if report.get("error"):
        lines += ["", "## 错误", "", f"- {report['error']['type']}: {report['error']['message']}"]
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Activate/home Meca500 before a gated one-step execution.")
    parser.add_argument("--ip", default="192.168.0.100")
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument("--activation-timeout-s", type=float, default=30.0)
    parser.add_argument("--home-timeout-s", type=float, default=90.0)
    parser.add_argument("--home-joint-vel-deg-s", type=float, default=5.0)
    parser.add_argument("--execute-prepare", action="store_true")
    parser.add_argument("--confirm-token", default="")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = prepare_robot(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"prepare_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "prepare_report.json", report)
    write_markdown(run_dir / "prepare_report.md", report)

    print("Prepare report:", run_dir)
    print("Status:", report["status"])
    print("Moves robot:", report["moves_robot"])
    for reason in report.get("refusal_reasons", []):
        print("Refused:", reason)


if __name__ == "__main__":
    main()
