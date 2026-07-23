from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from real_3d_alignment.config import DEFAULT_CONFIG, load_config, resolve_project_path
from real_3d_alignment.execution_bridge import (
    build_execution_plan,
    execute_if_allowed,
    validate_static_gates,
    write_execution_markdown,
)


def resolve_run_report(run_report: Path | None, run_dir: Path | None) -> Path:
    if run_report is not None and run_dir is not None:
        raise ValueError("Use either --run-report or --run-dir, not both.")
    if run_dir is not None:
        return run_dir / "run_report.json"
    if run_report is not None:
        return run_report
    raise ValueError("Use --run-report or --run-dir.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview or execute one gated real 3D alignment step. Preview mode never moves the robot."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-report", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument(
        "--candidate-name",
        choices=["translation_only", "combined_translation_rotation"],
        default="translation_only",
        help="First onsite validation should normally use translation_only.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--execute-one-step", action="store_true")
    parser.add_argument("--confirm-token", default="")
    parser.add_argument("--allow-orientation-step", action="store_true")
    parser.add_argument("--robot-ip", default=None)
    parser.add_argument("--robot-port", type=int, default=10000)
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument("--max-link6-delta-mm", type=float, default=None)
    parser.add_argument("--max-rotation-step-deg", type=float, default=None)
    parser.add_argument("--max-current-pose-position-error-mm", type=float, default=0.75)
    parser.add_argument("--max-current-pose-orientation-error-deg", type=float, default=0.5)
    parser.add_argument("--max-run-age-s", type=float, default=None)
    parser.add_argument("--cart-lin-vel-mm-s", type=float, default=1.0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    control = cfg["control"]
    robot = cfg["robot"]
    output_root = resolve_project_path(cfg["paths"]["output_root"])
    output_dir = args.output_dir or (output_root / "execution_bridge")
    run_report = resolve_run_report(args.run_report, args.run_dir)
    max_link6_delta_mm = (
        float(args.max_link6_delta_mm)
        if args.max_link6_delta_mm is not None
        else float(control["max_link6_translation_step_mm"])
    )
    max_rotation_step_deg = (
        float(args.max_rotation_step_deg)
        if args.max_rotation_step_deg is not None
        else float(control["max_rotation_step_deg"])
    )

    report = build_execution_plan(
        run_report_path=run_report,
        config=cfg,
        candidate_name=args.candidate_name,
        execute_requested=args.execute_one_step,
        confirm_token=args.confirm_token,
        allow_orientation_step=args.allow_orientation_step,
        max_link6_delta_mm=max_link6_delta_mm,
        max_rotation_step_deg=max_rotation_step_deg,
        max_current_pose_position_error_mm=float(args.max_current_pose_position_error_mm),
        max_current_pose_orientation_error_deg=float(args.max_current_pose_orientation_error_deg),
        cart_lin_vel_mm_s=float(args.cart_lin_vel_mm_s),
        max_run_age_s=float(args.max_run_age_s if args.max_run_age_s is not None else robot.get("max_dry_run_age_s", 300.0)),
        robot_ip=str(args.robot_ip or robot["ip_address"]),
        robot_port=int(args.robot_port),
        timeout_s=float(args.timeout_s),
    )
    report = validate_static_gates(report)
    report = execute_if_allowed(report)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = output_dir / f"real_3d_step_{timestamp}"
    report_path = report_dir / "real_3d_step_report.md"
    write_execution_markdown(report_path, report)

    print("real_3d_step report:", report_dir)
    print("status:", report["status"])
    print("moves_robot:", report["moves_robot"])
    for reason in report.get("refusal_reasons", []):
        print("refused:", reason)
    if report.get("execute_requested") and report.get("status") != "executed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
