from __future__ import annotations

import argparse
import csv
import math
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from compute_trocar_3d_control_errors import collect_reports, load_json, write_json
from handeye_common import matrix_to_pose_mm_deg, transform_inverse


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs" / "trocar_pose_from_ring"
DEFAULT_EXTRINSIC = ROOT / "config" / "camera_extrinsic_colleague_20260612.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "control_3d" / "dry_run_commands"


def axis_angle_deg(axis: np.ndarray) -> float:
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    dot = float(np.clip(axis[2], -1.0, 1.0))
    return float(math.degrees(math.acos(dot)))


def rotation_from_z_to_axis(axis: np.ndarray) -> np.ndarray:
    target = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    dot = float(np.clip(np.dot(target, axis), -1.0, 1.0))
    if dot > 0.999999:
        return np.zeros(3, dtype=np.float64)
    if dot < -0.999999:
        return np.array([math.pi, 0.0, 0.0], dtype=np.float64)
    rot_axis = np.cross(target, axis)
    rot_axis = rot_axis / max(np.linalg.norm(rot_axis), 1e-12)
    return rot_axis * math.acos(dot)


def clip_vector(vector: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= max_norm or norm <= 1e-12:
        return vector
    return vector * (max_norm / norm)


def relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT).as_posix())
    except ValueError:
        return str(path)


def build_command(
    report_path: Path,
    T_link6_camera: np.ndarray,
    target_distance_mm: float,
    k_xy: float,
    k_z: float,
    k_rot: float,
    max_translation_step_mm: float,
    max_rotation_step_deg: float,
    max_link6_translation_step_mm: float,
    min_link6_z_mm: float,
) -> dict:
    data = load_json(report_path)
    robot = data.get("robot") or {}
    if not robot.get("success") or "T_base_link6" not in robot:
        return {
            "report": relative_path(report_path),
            "image": data.get("image"),
            "status": "skipped",
            "reason": "Pose report has no successful robot T_base_link6 metadata.",
        }

    T_camera_trocar = np.array(data["T_camera_trocar"], dtype=np.float64)
    T_base_link6 = np.array(robot["T_base_link6"], dtype=np.float64)
    T_base_camera = T_base_link6 @ T_link6_camera

    p_camera_mm = T_camera_trocar[:3, 3] * 1000.0
    axis_camera = T_camera_trocar[:3, 2]
    axis_camera = axis_camera / max(np.linalg.norm(axis_camera), 1e-12)

    lateral_error_mm = float(np.linalg.norm(p_camera_mm[:2]))
    depth_error_mm = float(p_camera_mm[2] - target_distance_mm)
    axis_error_deg = axis_angle_deg(axis_camera)
    weighted_error_mm = float(
        math.sqrt(
            lateral_error_mm**2
            + depth_error_mm**2
            + (target_distance_mm * math.sin(math.radians(axis_error_deg))) ** 2
        )
    )

    raw_translation_step_camera_mm = np.array(
        [k_xy * p_camera_mm[0], k_xy * p_camera_mm[1], k_z * depth_error_mm],
        dtype=np.float64,
    )
    translation_step_camera_mm = clip_vector(raw_translation_step_camera_mm, max_translation_step_mm)

    rotvec_full_camera = rotation_from_z_to_axis(axis_camera)
    raw_rotvec_step_camera = rotvec_full_camera * k_rot
    max_rot_rad = math.radians(max_rotation_step_deg)
    rot_norm = float(np.linalg.norm(raw_rotvec_step_camera))
    if rot_norm > max_rot_rad and rot_norm > 1e-12:
        rotvec_step_camera = raw_rotvec_step_camera * (max_rot_rad / rot_norm)
    else:
        rotvec_step_camera = raw_rotvec_step_camera
    R_delta_camera = R.from_rotvec(rotvec_step_camera).as_matrix()

    translation_step_base_m = T_base_camera[:3, :3] @ (translation_step_camera_mm / 1000.0)
    T_base_camera_next = T_base_camera.copy()
    T_base_camera_next[:3, :3] = T_base_camera[:3, :3] @ R_delta_camera
    T_base_camera_next[:3, 3] = T_base_camera[:3, 3] + translation_step_base_m
    T_base_link6_next = T_base_camera_next @ transform_inverse(T_link6_camera)

    link6_delta_mm = (T_base_link6_next[:3, 3] - T_base_link6[:3, 3]) * 1000.0
    link6_delta_norm_mm = float(np.linalg.norm(link6_delta_mm))
    camera_step_norm_mm = float(np.linalg.norm(translation_step_camera_mm))
    rotation_step_deg = float(math.degrees(np.linalg.norm(rotvec_step_camera)))
    candidate_pose = matrix_to_pose_mm_deg(T_base_link6_next)
    current_pose = matrix_to_pose_mm_deg(T_base_link6)

    checks = {
        "camera_translation_step_within_limit": camera_step_norm_mm <= max_translation_step_mm + 1e-9,
        "link6_translation_step_within_limit": link6_delta_norm_mm <= max_link6_translation_step_mm + 1e-9,
        "rotation_step_within_limit": rotation_step_deg <= max_rotation_step_deg + 1e-9,
        "candidate_link6_z_above_min": candidate_pose[2] >= min_link6_z_mm,
    }
    checks["all_passed"] = all(checks.values())

    return {
        "report": relative_path(report_path),
        "image": data.get("image"),
        "status": "ok",
        "target_distance_mm": target_distance_mm,
        "current_errors": {
            "camera_position_mm": [float(v) for v in p_camera_mm],
            "lateral_error_mm": lateral_error_mm,
            "depth_error_mm": depth_error_mm,
            "axis_angle_error_deg": axis_error_deg,
            "weighted_3d_error_mm": weighted_error_mm,
            "pnp_mean_reprojection_error_px": data.get("metrics", {}).get("mean_reprojection_error_px"),
        },
        "dry_run_command": {
            "meaning": "Move the eye-in-hand camera by this small body-frame step to reduce T_camera_trocar error. This report does not execute any robot command.",
            "translation_step_camera_mm": [float(v) for v in translation_step_camera_mm],
            "translation_step_base_mm": [float(v) for v in (translation_step_base_m * 1000.0)],
            "rotation_step_camera_rotvec_deg": [float(math.degrees(v)) for v in rotvec_step_camera],
            "rotation_step_angle_deg": rotation_step_deg,
            "candidate_pose_base_link6_mm_deg": [float(v) for v in candidate_pose],
            "current_pose_base_link6_mm_deg": [float(v) for v in current_pose],
            "link6_delta_mm": [float(v) for v in link6_delta_mm],
            "link6_delta_norm_mm": link6_delta_norm_mm,
        },
        "limits": {
            "max_translation_step_mm": max_translation_step_mm,
            "max_rotation_step_deg": max_rotation_step_deg,
            "max_link6_translation_step_mm": max_link6_translation_step_mm,
            "min_link6_z_mm": min_link6_z_mm,
        },
        "checks": checks,
        "notes": [
            "This is a dry-run candidate generated from the current 3D pose estimate.",
            "Review direction and scale onsite before enabling any real motion.",
            "The candidate pose includes a small orientation step about the camera frame; real execution should still pass robot-side workspace, speed, and collision checks.",
        ],
    }


def write_csv(path: Path, records: list[dict]) -> None:
    rows = []
    for index, record in enumerate(records, start=1):
        if record.get("status") != "ok":
            rows.append(
                {
                    "sample_index": index,
                    "status": record.get("status"),
                    "image": record.get("image"),
                    "reason": record.get("reason"),
                }
            )
            continue
        cmd = record["dry_run_command"]
        err = record["current_errors"]
        rows.append(
            {
                "sample_index": index,
                "status": record["status"],
                "image": record.get("image"),
                "lateral_error_mm": err["lateral_error_mm"],
                "depth_error_mm": err["depth_error_mm"],
                "axis_angle_error_deg": err["axis_angle_error_deg"],
                "weighted_3d_error_mm": err["weighted_3d_error_mm"],
                "camera_step_x_mm": cmd["translation_step_camera_mm"][0],
                "camera_step_y_mm": cmd["translation_step_camera_mm"][1],
                "camera_step_z_mm": cmd["translation_step_camera_mm"][2],
                "base_step_x_mm": cmd["translation_step_base_mm"][0],
                "base_step_y_mm": cmd["translation_step_base_mm"][1],
                "base_step_z_mm": cmd["translation_step_base_mm"][2],
                "rotation_step_angle_deg": cmd["rotation_step_angle_deg"],
                "link6_delta_norm_mm": cmd["link6_delta_norm_mm"],
                "checks_all_passed": record["checks"]["all_passed"],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# 3D 对齐 dry-run 控制命令",
        "",
        f"生成时间：{report['timestamp']}",
        "",
        "## 说明",
        "",
        "本报告只根据真实 `T_camera_trocar` 估计结果生成保守的小步控制建议，不向真实 Meca500 发送运动命令。",
        "",
        "命令含义：",
        "",
        "```text",
        "translation_step_camera_mm: 建议相机在自身坐标系中移动的微小位移",
        "translation_step_base_mm: 上述位移换算到 Meca500 基坐标系后的方向",
        "candidate_pose_base_link6_mm_deg: 若采用该小步后的 link6 候选位姿",
        "```",
        "",
        "## 参数",
        "",
        f"- 目标相机-戳卡距离：`{report['parameters']['target_distance_mm']:.3f} mm`",
        f"- 最大相机平移步长：`{report['parameters']['max_translation_step_mm']:.3f} mm`",
        f"- 最大姿态步长：`{report['parameters']['max_rotation_step_deg']:.3f} deg`",
        f"- 最大 link6 平移步长：`{report['parameters']['max_link6_translation_step_mm']:.3f} mm`",
        "",
        "## 样本命令",
        "",
        "| # | ok | lateral mm | depth mm | axis deg | cam step mm | base step mm | rot deg | link6 step mm | note |",
        "|---:|---|---:|---:|---:|---|---|---:|---:|---|",
    ]
    for index, record in enumerate(report["records"], start=1):
        if record.get("status") != "ok":
            reason = record.get("reason", "skipped")
            lines.append(f"| {index} | False |  |  |  |  |  |  |  | {reason} |")
            continue
        err = record["current_errors"]
        cmd = record["dry_run_command"]
        cam_step = ", ".join(f"{v:.3f}" for v in cmd["translation_step_camera_mm"])
        base_step = ", ".join(f"{v:.3f}" for v in cmd["translation_step_base_mm"])
        lines.append(
            f"| {index} | {record['checks']['all_passed']} | "
            f"{err['lateral_error_mm']:.3f} | {err['depth_error_mm']:.3f} | {err['axis_angle_error_deg']:.3f} | "
            f"`[{cam_step}]` | `[{base_step}]` | {cmd['rotation_step_angle_deg']:.3f} | {cmd['link6_delta_norm_mm']:.3f} |  |"
        )

    lines += [
        "",
        "## 安全状态",
        "",
        f"- 通过 dry-run 安全检查的样本数：`{report['passed_count']} / {report['sample_count']}`",
        "- 当前结果仍然不能直接代表可执行真机命令；真机执行前需要现场确认方向、速度、边界和急停状态。",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dry-run 3D alignment motion suggestions without moving Meca500.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--extrinsic", type=Path, default=DEFAULT_EXTRINSIC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-distance-mm", type=float, default=18.0)
    parser.add_argument("--k-xy", type=float, default=0.35)
    parser.add_argument("--k-z", type=float, default=0.35)
    parser.add_argument("--k-rot", type=float, default=0.25)
    parser.add_argument("--max-translation-step-mm", type=float, default=1.0)
    parser.add_argument("--max-rotation-step-deg", type=float, default=2.0)
    parser.add_argument("--max-link6-translation-step-mm", type=float, default=2.5)
    parser.add_argument("--min-link6-z-mm", type=float, default=20.0)
    args = parser.parse_args()

    extrinsic = load_json(args.extrinsic)
    T_link6_camera = np.array(extrinsic["T_link6_camera"], dtype=np.float64)

    reports = collect_reports(args.input_dir)
    records = [
        build_command(
            report_path=path,
            T_link6_camera=T_link6_camera,
            target_distance_mm=args.target_distance_mm,
            k_xy=args.k_xy,
            k_z=args.k_z,
            k_rot=args.k_rot,
            max_translation_step_mm=args.max_translation_step_mm,
            max_rotation_step_deg=args.max_rotation_step_deg,
            max_link6_translation_step_mm=args.max_link6_translation_step_mm,
            min_link6_z_mm=args.min_link6_z_mm,
        )
        for path in reports
    ]
    passed_count = sum(1 for record in records if record.get("checks", {}).get("all_passed"))
    output = {
        "timestamp": datetime.now().isoformat(),
        "input_dir": str(args.input_dir),
        "extrinsic": str(args.extrinsic),
        "sample_count": len(records),
        "passed_count": int(passed_count),
        "parameters": {
            "target_distance_mm": args.target_distance_mm,
            "k_xy": args.k_xy,
            "k_z": args.k_z,
            "k_rot": args.k_rot,
            "max_translation_step_mm": args.max_translation_step_mm,
            "max_rotation_step_deg": args.max_rotation_step_deg,
            "max_link6_translation_step_mm": args.max_link6_translation_step_mm,
            "min_link6_z_mm": args.min_link6_z_mm,
        },
        "records": records,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "dry_run_commands.json", output)
    write_csv(args.output_dir / "dry_run_commands.csv", records)
    write_markdown(args.output_dir / "dry_run_commands.md", output)

    print("3D dry-run commands written:", args.output_dir)
    print("Samples:", len(records), "Passed dry-run checks:", passed_count)


if __name__ == "__main__":
    main()
