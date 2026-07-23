from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs" / "control_3d" / "dry_run_commands" / "dry_run_commands.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "control_3d" / "real_step_confirmation"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def fmt_vec(values: list[float], digits: int = 3) -> str:
    return "[" + ", ".join(f"{float(value):.{digits}f}" for value in values) + "]"


def build_sheet(dry_run: dict) -> dict:
    executable = [
        record
        for record in dry_run.get("records", [])
        if record.get("status") == "ok" and record.get("checks", {}).get("all_passed")
    ]
    skipped = [
        record
        for record in dry_run.get("records", [])
        if not (record.get("status") == "ok" and record.get("checks", {}).get("all_passed"))
    ]
    return {
        "timestamp": datetime.now().isoformat(),
        "source": dry_run.get("timestamp"),
        "execution_enabled": False,
        "sample_count": dry_run.get("sample_count", 0),
        "candidate_count": len(executable),
        "skipped_count": len(skipped),
        "parameters": dry_run.get("parameters", {}),
        "candidates": executable,
        "skipped": skipped,
        "required_human_checks": [
            "确认机器人控制端口 192.168.0.100:10000 可连接，且没有被其他程序占用。",
            "确认 Meca500 处于低速安全模式，急停按钮可随时触达。",
            "确认戳卡和周围物体固定，且戳卡入口仍在相机视野内。",
            "确认表中基坐标方向与现场观察直觉一致。",
            "确认候选 link6 位姿在允许工作空间内，并且远离碰撞。",
            "每次只执行一个小步；执行后必须重新采图、重新估计 `T_camera_trocar`、重新生成 dry-run 报告。",
        ],
    }


def write_markdown(path: Path, sheet: dict) -> None:
    lines = [
        "# 真实 Meca500 单步执行前确认表",
        "",
        f"生成时间：{sheet['timestamp']}",
        "",
        "## 状态",
        "",
        "- 本文件只用于人工确认，不执行任何机械臂运动。",
        f"- 执行开关：`{sheet['execution_enabled']}`",
        f"- dry-run 样本数：`{sheet['sample_count']}`",
        f"- 可候选执行样本数：`{sheet['candidate_count']}`",
        f"- 跳过样本数：`{sheet['skipped_count']}`",
        "",
        "## 候选单步",
        "",
        "| # | current link6 pose mm/deg | candidate link6 pose mm/deg | link6 delta mm | camera step mm | base step mm | rot deg |",
        "|---:|---|---|---|---|---|---:|",
    ]
    for index, record in enumerate(sheet["candidates"], start=1):
        cmd = record["dry_run_command"]
        lines.append(
            f"| {index} | `{fmt_vec(cmd['current_pose_base_link6_mm_deg'])}` | "
            f"`{fmt_vec(cmd['candidate_pose_base_link6_mm_deg'])}` | "
            f"`{fmt_vec(cmd['link6_delta_mm'])}` | "
            f"`{fmt_vec(cmd['translation_step_camera_mm'])}` | "
            f"`{fmt_vec(cmd['translation_step_base_mm'])}` | "
            f"{cmd['rotation_step_angle_deg']:.3f} |"
        )

    lines += [
        "",
        "## 跳过样本",
        "",
        "| # | image | reason |",
        "|---:|---|---|",
    ]
    for index, record in enumerate(sheet["skipped"], start=1):
        lines.append(f"| {index} | `{record.get('image')}` | {record.get('reason', 'failed checks')} |")

    lines += [
        "",
        "## 人工确认项",
        "",
    ]
    for item in sheet["required_human_checks"]:
        lines.append(f"- [ ] {item}")

    lines += [
        "",
        "## 执行原则",
        "",
        "即使后续增加 `--execute` 分支，也只能执行一小步；每一步后必须重新采集图像、重新估计 `T_camera_trocar`、重新生成 dry-run 报告。",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a human confirmation sheet before any real Meca500 3D alignment step.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    dry_run = load_json(args.input)
    sheet = build_sheet(dry_run)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "real_step_confirmation.json", sheet)
    write_markdown(args.output_dir / "real_step_confirmation.md", sheet)

    print("Real-step confirmation sheet written:", args.output_dir)
    print("Candidate steps:", sheet["candidate_count"], "Skipped:", sheet["skipped_count"])


if __name__ == "__main__":
    main()
