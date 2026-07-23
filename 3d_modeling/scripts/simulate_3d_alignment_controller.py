from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from compute_trocar_3d_control_errors import collect_reports, load_json, write_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs" / "trocar_pose_from_ring"
DEFAULT_OUTPUT = ROOT / "outputs" / "control_3d" / "closed_loop_sim"


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
    angle = math.acos(dot)
    return rot_axis * angle


def clip_vector(vector: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= max_norm or norm <= 1e-12:
        return vector
    return vector * (max_norm / norm)


def simulate_one(
    report_path: Path,
    target_distance_mm: float,
    max_iterations: int,
    k_xy: float,
    k_z: float,
    k_rot: float,
    max_translation_step_mm: float,
    max_rotation_step_deg: float,
    lateral_tol_mm: float,
    depth_tol_mm: float,
    axis_tol_deg: float,
) -> dict:
    data = load_json(report_path)
    p = np.array(data["translation_camera_trocar_m"], dtype=np.float64) * 1000.0
    axis = np.array(data["trocar_axis_camera"], dtype=np.float64)
    axis = axis / max(np.linalg.norm(axis), 1e-12)

    trajectory = []
    converged = False
    for iteration in range(max_iterations + 1):
        lateral = float(np.linalg.norm(p[:2]))
        depth = float(p[2] - target_distance_mm)
        axis_error = axis_angle_deg(axis)
        weighted = float(
            math.sqrt(
                lateral**2
                + depth**2
                + (target_distance_mm * math.sin(math.radians(axis_error))) ** 2
            )
        )
        trajectory.append(
            {
                "iteration": iteration,
                "camera_x_mm": float(p[0]),
                "camera_y_mm": float(p[1]),
                "camera_z_mm": float(p[2]),
                "lateral_error_mm": lateral,
                "depth_error_mm": depth,
                "axis_angle_error_deg": axis_error,
                "weighted_3d_error_mm": weighted,
            }
        )
        if lateral <= lateral_tol_mm and abs(depth) <= depth_tol_mm and axis_error <= axis_tol_deg:
            converged = True
            break

        translation_cmd = np.array([k_xy * p[0], k_xy * p[1], k_z * depth], dtype=np.float64)
        translation_cmd = clip_vector(translation_cmd, max_translation_step_mm)

        rotvec_full = rotation_from_z_to_axis(axis)
        rotvec_step = rotvec_full * k_rot
        max_rot = math.radians(max_rotation_step_deg)
        rot_norm = float(np.linalg.norm(rotvec_step))
        if rot_norm > max_rot and rot_norm > 1e-12:
            rotvec_step = rotvec_step * (max_rot / rot_norm)
        R_delta = R.from_rotvec(rotvec_step).as_matrix()

        # Camera moves by translation_cmd in its own frame; object coordinates decrease by that amount.
        # Then camera rotates by R_delta, so object coordinates are expressed in the new camera frame.
        p = R_delta.T @ (p - translation_cmd)
        axis = R_delta.T @ axis
        axis = axis / max(np.linalg.norm(axis), 1e-12)

    return {
        "report": str(report_path.relative_to(ROOT).as_posix()),
        "image": data.get("image"),
        "converged": converged,
        "iterations": trajectory[-1]["iteration"],
        "initial": trajectory[0],
        "final": trajectory[-1],
        "trajectory": trajectory,
    }


def numeric_summary(records: list[dict], key_path: tuple[str, ...]) -> dict:
    values = []
    for record in records:
        value = record
        for key in key_path:
            value = value.get(key)
            if value is None:
                break
        if value is not None:
            values.append(value)
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def write_csv(path: Path, simulations: list[dict]) -> None:
    rows = []
    for sample_index, sim in enumerate(simulations, start=1):
        for item in sim["trajectory"]:
            rows.append({"sample_index": sample_index, "image": sim["image"], **item})
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def draw_convergence_plot(path: Path, simulations: list[dict], width: int = 1000, height: int = 620) -> None:
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    margin_left, margin_right, margin_top, margin_bottom = 80, 30, 40, 80
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    all_values = [
        item["weighted_3d_error_mm"]
        for sim in simulations
        for item in sim["trajectory"]
    ]
    max_iter = max((len(sim["trajectory"]) - 1 for sim in simulations), default=1)
    max_value = max(max(all_values, default=1.0), 1.0)
    colors = [(220, 70, 50), (40, 150, 60), (60, 90, 230), (180, 80, 180), (60, 160, 180)]

    cv2.rectangle(image, (margin_left, margin_top), (margin_left + plot_w, margin_top + plot_h), (30, 30, 30), 1)
    for tick in range(6):
        y = margin_top + plot_h - int(plot_h * tick / 5)
        value = max_value * tick / 5
        cv2.line(image, (margin_left - 5, y), (margin_left + plot_w, y), (230, 230, 230), 1)
        cv2.putText(image, f"{value:.1f}", (10, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)
    cv2.putText(image, "weighted 3D error (mm)", (margin_left, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2)
    cv2.putText(image, "iteration", (margin_left + plot_w // 2 - 30, height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2)

    for index, sim in enumerate(simulations):
        points = []
        for item in sim["trajectory"]:
            x = margin_left + int(plot_w * item["iteration"] / max(max_iter, 1))
            y = margin_top + plot_h - int(plot_h * item["weighted_3d_error_mm"] / max_value)
            points.append((x, y))
        color = colors[index % len(colors)]
        for a, b in zip(points, points[1:]):
            cv2.line(image, a, b, color, 2, cv2.LINE_AA)
        if points:
            cv2.circle(image, points[0], 4, color, -1)
            cv2.circle(image, points[-1], 4, color, -1)
            cv2.putText(image, f"s{index + 1}", (points[-1][0] + 5, points[-1][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise RuntimeError(f"Failed to encode {path}")
    encoded.tofile(str(path))


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# 3D 几何闭环控制仿真 baseline",
        "",
        f"生成时间：{report['timestamp']}",
        "",
        "## 控制目标",
        "",
        f"- 目标距离：`{report['controller']['target_distance_mm']:.3f} mm`",
        "- 横向误差趋近 `0 mm`。",
        "- 距离误差趋近 `0 mm`。",
        "- 戳卡孔轴与相机光轴夹角趋近 `0 deg`。",
        "",
        "## 控制器",
        "",
        "当前是相机坐标系下的几何小步控制器，用于验证 3D 误差定义和闭环收敛性，不直接发送给真实机械臂。",
        "",
        "## 汇总",
        "",
        f"- 样本数：`{report['sample_count']}`",
        f"- 收敛样本数：`{report['converged_count']}`",
    ]
    for key, label in [
        ("initial_weighted_error_mm", "初始综合误差 mm"),
        ("final_weighted_error_mm", "最终综合误差 mm"),
        ("iterations", "迭代次数"),
    ]:
        stats = report["summary"].get(key, {})
        if stats.get("count", 0):
            lines.append(
                f"- {label}: mean={stats['mean']:.6f}, std={stats['std']:.6f}, "
                f"min={stats['min']:.6f}, max={stats['max']:.6f}"
            )

    lines += [
        "",
        "## 样本结果",
        "",
        "| # | converged | iter | initial weighted mm | final weighted mm | final lateral mm | final depth mm | final axis deg |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for index, sim in enumerate(report["simulations"], start=1):
        lines.append(
            f"| {index} | {sim['converged']} | {sim['iterations']} | "
            f"{sim['initial']['weighted_3d_error_mm']:.3f} | {sim['final']['weighted_3d_error_mm']:.3f} | "
            f"{sim['final']['lateral_error_mm']:.3f} | {sim['final']['depth_error_mm']:.3f} | "
            f"{sim['final']['axis_angle_error_deg']:.3f} |"
        )

    lines += [
        "",
        "## 图",
        "",
        "```text",
        "3d_modeling/outputs/control_3d/closed_loop_sim/convergence_plot.png",
        "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate a camera-frame 3D alignment controller from real T_camera_trocar samples.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-distance-mm", type=float, default=18.0)
    parser.add_argument("--max-iterations", type=int, default=40)
    parser.add_argument("--k-xy", type=float, default=0.55)
    parser.add_argument("--k-z", type=float, default=0.45)
    parser.add_argument("--k-rot", type=float, default=0.35)
    parser.add_argument("--max-translation-step-mm", type=float, default=8.0)
    parser.add_argument("--max-rotation-step-deg", type=float, default=8.0)
    parser.add_argument("--lateral-tol-mm", type=float, default=0.05)
    parser.add_argument("--depth-tol-mm", type=float, default=0.10)
    parser.add_argument("--axis-tol-deg", type=float, default=0.5)
    args = parser.parse_args()

    reports = collect_reports(args.input_dir)
    simulations = [
        simulate_one(
            path,
            target_distance_mm=args.target_distance_mm,
            max_iterations=args.max_iterations,
            k_xy=args.k_xy,
            k_z=args.k_z,
            k_rot=args.k_rot,
            max_translation_step_mm=args.max_translation_step_mm,
            max_rotation_step_deg=args.max_rotation_step_deg,
            lateral_tol_mm=args.lateral_tol_mm,
            depth_tol_mm=args.depth_tol_mm,
            axis_tol_deg=args.axis_tol_deg,
        )
        for path in reports
    ]

    summary = {
        "initial_weighted_error_mm": numeric_summary(simulations, ("initial", "weighted_3d_error_mm")),
        "final_weighted_error_mm": numeric_summary(simulations, ("final", "weighted_3d_error_mm")),
        "iterations": numeric_summary(simulations, ("iterations",)),
        "final_lateral_error_mm": numeric_summary(simulations, ("final", "lateral_error_mm")),
        "final_depth_error_mm": numeric_summary(simulations, ("final", "depth_error_mm")),
        "final_axis_angle_error_deg": numeric_summary(simulations, ("final", "axis_angle_error_deg")),
    }
    output = {
        "timestamp": datetime.now().isoformat(),
        "input_dir": str(args.input_dir),
        "sample_count": len(simulations),
        "converged_count": int(sum(1 for sim in simulations if sim["converged"])),
        "controller": {
            "target_distance_mm": args.target_distance_mm,
            "max_iterations": args.max_iterations,
            "k_xy": args.k_xy,
            "k_z": args.k_z,
            "k_rot": args.k_rot,
            "max_translation_step_mm": args.max_translation_step_mm,
            "max_rotation_step_deg": args.max_rotation_step_deg,
            "lateral_tol_mm": args.lateral_tol_mm,
            "depth_tol_mm": args.depth_tol_mm,
            "axis_tol_deg": args.axis_tol_deg,
        },
        "summary": summary,
        "simulations": simulations,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "closed_loop_simulation.json", output)
    write_csv(args.output_dir / "closed_loop_trajectories.csv", simulations)
    draw_convergence_plot(args.output_dir / "convergence_plot.png", simulations)
    write_markdown(args.output_dir / "closed_loop_simulation.md", output)

    print("3D closed-loop simulation written:", args.output_dir)
    print("Samples:", len(simulations), "Converged:", output["converged_count"])
    print("Final weighted error summary:", summary["final_weighted_error_mm"])


if __name__ == "__main__":
    main()
