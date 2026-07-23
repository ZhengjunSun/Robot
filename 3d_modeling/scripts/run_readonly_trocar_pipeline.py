from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
DEFAULT_OUTPUT = ROOT / "outputs" / "live_readonly_pipeline"
OBSERVATION_ROOT = ROOT / "outputs" / "trocar_observations"


def run_step(name: str, command: list[str], required: bool = True) -> dict:
    started = datetime.now()
    completed = subprocess.run(
        command,
        cwd=str(WORKSPACE),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    ended = datetime.now()
    record = {
        "name": name,
        "command": command,
        "required": required,
        "returncode": completed.returncode,
        "success": completed.returncode == 0,
        "started": started.isoformat(),
        "ended": ended.isoformat(),
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }
    if required and completed.returncode != 0:
        record["fatal"] = True
    return record


def latest_observation_from_stdout(stdout: str) -> Path | None:
    for line in stdout.splitlines():
        if line.startswith("Image:"):
            image_path = Path(line.split(":", 1)[1].strip())
            if image_path.exists():
                return image_path.parent
    return None


def latest_observation_from_filesystem() -> Path | None:
    candidates = sorted(
        OBSERVATION_ROOT.glob("trocar_obs_*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if (candidate / "camera_rgb.png").exists() and (candidate / "metadata.json").exists():
            return candidate
    return None


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# 只读戳卡 3D 链路运行记录",
        "",
        f"生成时间：{report['timestamp']}",
        "",
        "## 状态",
        "",
        f"- 总体状态：`{report['status']}`",
        f"- 观测目录：`{report.get('observation_dir')}`",
        f"- 位姿输出目录：`{report.get('pose_output_dir')}`",
        "",
        "## 步骤",
        "",
        "| # | step | required | success | returncode |",
        "|---:|---|---|---|---:|",
    ]
    for index, step in enumerate(report["steps"], start=1):
        lines.append(
            f"| {index} | `{step['name']}` | {step['required']} | {step['success']} | {step['returncode']} |"
        )
    lines += [
        "",
        "## 关键输出",
        "",
    ]
    for item in report["outputs"]:
        lines.append(f"- `{item}`")
    if report.get("notes"):
        lines += ["", "## 备注", ""]
        for note in report["notes"]:
            lines.append(f"- {note}")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the read-only trocar pipeline: network diagnostics, observation, 6D pose, 3D errors, simulation, dry-run, confirmation."
    )
    parser.add_argument("--observation-dir", type=Path, default=None, help="Use an existing observation directory instead of capturing.")
    parser.add_argument("--note", default="readonly full pipeline")
    parser.add_argument("--skip-network-diagnostics", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_root / f"pipeline_run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    python = sys.executable
    steps = []
    notes = [
        "本链路是只读流程，不发送任何机械臂运动命令。",
        "如果机器人状态读取失败，仍可完成相机坐标系位姿估计；但该样本会跳过基坐标系 dry-run 命令生成。",
    ]

    if not args.skip_network_diagnostics:
        steps.append(
            run_step(
                "network_diagnostics",
                [
                    python,
                    str(ROOT / "scripts" / "diagnose_meca500_network.py"),
                    "--ip",
                    "192.168.0.100",
                    "--port",
                    "10000",
                    "--timeout-s",
                    "5",
                ],
                required=False,
            )
        )

    observation_dir = args.observation_dir
    if observation_dir is None:
        capture = run_step(
            "capture_trocar_observation",
            [
                python,
                str(ROOT / "scripts" / "capture_trocar_observation.py"),
                "--note",
                args.note,
            ],
            required=True,
        )
        steps.append(capture)
        if capture["success"]:
            observation_dir = latest_observation_from_stdout(capture["stdout"])
            if observation_dir is None:
                observation_dir = latest_observation_from_filesystem()
        if observation_dir is None:
            capture["fatal"] = True
            capture["success"] = False
            notes.append("Capture succeeded but no observation directory could be resolved.")

    if observation_dir is not None and not observation_dir.is_absolute():
        observation_dir = (WORKSPACE / observation_dir).resolve()
    pose_output_dir = None

    if observation_dir is not None:
        pose_output_dir = ROOT / "outputs" / "trocar_pose_from_ring" / observation_dir.name
        steps.append(
            run_step(
                "estimate_trocar_pose_from_ring",
                [
                    python,
                    str(ROOT / "scripts" / "estimate_trocar_pose_from_ring.py"),
                    "--observation-dir",
                    str(observation_dir),
                    "--output-dir",
                    str(pose_output_dir),
                ],
                required=True,
            )
        )

    if any(step.get("fatal") for step in steps):
        status = "failed"
    else:
        for name, command in [
            ("compute_trocar_3d_control_errors", [python, str(ROOT / "scripts" / "compute_trocar_3d_control_errors.py")]),
            ("simulate_3d_alignment_controller", [python, str(ROOT / "scripts" / "simulate_3d_alignment_controller.py")]),
            ("generate_3d_alignment_dry_run_commands", [python, str(ROOT / "scripts" / "generate_3d_alignment_dry_run_commands.py")]),
            ("prepare_real_step_confirmation_sheet", [python, str(ROOT / "scripts" / "prepare_real_step_confirmation_sheet.py")]),
        ]:
            step = run_step(name, command, required=True)
            steps.append(step)
            if step.get("fatal"):
                break
        status = "ok" if not any(step.get("fatal") for step in steps) else "failed"

    outputs = [
        str(ROOT / "outputs" / "network_diagnostics" / "meca500_network_diagnostics_latest.md"),
        str(ROOT / "outputs" / "control_3d" / "trocar_3d_control_errors.md"),
        str(ROOT / "outputs" / "control_3d" / "closed_loop_sim" / "closed_loop_simulation.md"),
        str(ROOT / "outputs" / "control_3d" / "dry_run_commands" / "dry_run_commands.md"),
        str(ROOT / "outputs" / "control_3d" / "real_step_confirmation" / "real_step_confirmation.md"),
    ]
    if pose_output_dir is not None:
        outputs.insert(1, str(pose_output_dir / "real_trocar_ring_pose_report.json"))
        outputs.insert(2, str(pose_output_dir / "real_trocar_ring_pose_overlay.png"))

    report = {
        "timestamp": datetime.now().isoformat(),
        "status": status,
        "observation_dir": None if observation_dir is None else str(observation_dir),
        "pose_output_dir": None if pose_output_dir is None else str(pose_output_dir),
        "steps": steps,
        "outputs": outputs,
        "notes": notes,
    }
    write_json(run_dir / "pipeline_summary.json", report)
    write_markdown(run_dir / "pipeline_summary.md", report)

    print("Readonly trocar pipeline summary:", run_dir)
    print("Status:", status)
    for step in steps:
        print(f"- {step['name']}: {'OK' if step['success'] else 'FAILED'}")


if __name__ == "__main__":
    main()
