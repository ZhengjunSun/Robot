from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from compute_trocar_3d_control_errors import collect_reports, load_json, write_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs" / "trocar_pose_from_ring"
DEFAULT_OUTPUT = ROOT / "outputs" / "control_3d" / "axis_aware_sim"


def axis_angle_deg(axis: np.ndarray) -> float:
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    return float(math.degrees(math.acos(float(np.clip(axis[2], -1.0, 1.0)))))


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


def error_metrics(p_mm: np.ndarray, axis: np.ndarray, target_distance_mm: float) -> dict:
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    lateral = float(np.linalg.norm(p_mm[:2]))
    depth = float(p_mm[2] - target_distance_mm)
    axis_deg = axis_angle_deg(axis)
    axis_equiv = float(target_distance_mm * math.sin(math.radians(axis_deg)))
    weighted = float(math.sqrt(lateral**2 + depth**2 + axis_equiv**2))
    return {
        "camera_position_mm": p_mm.tolist(),
        "trocar_axis_camera": axis.tolist(),
        "lateral_error_mm": lateral,
        "depth_error_mm": depth,
        "axis_angle_error_deg": axis_deg,
        "axis_equivalent_mm": axis_equiv,
        "weighted_3d_error_mm": weighted,
    }


def simulate_candidate(
    name: str,
    p_mm: np.ndarray,
    axis: np.ndarray,
    target_distance_mm: float,
    translation_step_mm: np.ndarray,
    rotvec_step: np.ndarray,
) -> dict:
    R_delta = R.from_rotvec(rotvec_step).as_matrix()
    next_p = R_delta.T @ (p_mm - translation_step_mm)
    next_axis = R_delta.T @ axis
    before = error_metrics(p_mm, axis, target_distance_mm)
    after = error_metrics(next_p, next_axis, target_distance_mm)
    return {
        "name": name,
        "translation_step_camera_mm": translation_step_mm.tolist(),
        "rotation_step_camera_rotvec_deg": [float(math.degrees(v)) for v in rotvec_step],
        "rotation_step_angle_deg": float(math.degrees(np.linalg.norm(rotvec_step))),
        "before": before,
        "after": after,
        "delta_weighted_3d_error_mm": after["weighted_3d_error_mm"] - before["weighted_3d_error_mm"],
        "improved": after["weighted_3d_error_mm"] < before["weighted_3d_error_mm"],
    }


def simulate_report(
    path: Path,
    target_distance_mm: float,
    max_translation_step_mm: float,
    max_rotation_step_deg: float,
    k_xy: float,
    k_z: float,
    k_rot: float,
) -> dict:
    data = load_json(path)
    p = np.asarray(data["translation_camera_trocar_m"], dtype=np.float64) * 1000.0
    axis = np.asarray(data["trocar_axis_camera"], dtype=np.float64)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    depth = p[2] - target_distance_mm
    raw_translation = np.array([k_xy * p[0], k_xy * p[1], k_z * depth], dtype=np.float64)
    translation = clip_vector(raw_translation, max_translation_step_mm)

    full_rotvec = rotation_from_z_to_axis(axis) * k_rot
    max_rot_rad = math.radians(max_rotation_step_deg)
    if np.linalg.norm(full_rotvec) > max_rot_rad:
        rotvec = full_rotvec * (max_rot_rad / max(np.linalg.norm(full_rotvec), 1e-12))
    else:
        rotvec = full_rotvec

    zero_t = np.zeros(3, dtype=np.float64)
    zero_r = np.zeros(3, dtype=np.float64)
    candidates = [
        simulate_candidate("translation_only", p, axis, target_distance_mm, translation, zero_r),
        simulate_candidate("rotation_only", p, axis, target_distance_mm, zero_t, rotvec),
        simulate_candidate("combined_translation_rotation", p, axis, target_distance_mm, translation, rotvec),
    ]
    best = min(candidates, key=lambda item: item["after"]["weighted_3d_error_mm"])
    return {
        "report": str(path),
        "image": data.get("image"),
        "target_distance_mm": target_distance_mm,
        "parameters": {
            "max_translation_step_mm": max_translation_step_mm,
            "max_rotation_step_deg": max_rotation_step_deg,
            "k_xy": k_xy,
            "k_z": k_z,
            "k_rot": k_rot,
        },
        "candidates": candidates,
        "best_candidate": best["name"],
        "best_delta_weighted_3d_error_mm": best["delta_weighted_3d_error_mm"],
    }


def write_markdown(path: Path, output: dict) -> None:
    latest = output["latest"]
    lines = [
        "# Axis-Aware 单步候选仿真",
        "",
        f"生成时间：{output['timestamp']}",
        "",
        f"- 输入图像：`{latest['image']}`",
        f"- 最优候选：`{latest['best_candidate']}`",
        f"- 最优综合误差变化：`{latest['best_delta_weighted_3d_error_mm']:.3f} mm`",
        "",
        "## 候选对比",
        "",
        "| candidate | trans step camera mm | rot deg | before weighted | after weighted | delta | after lateral | after depth | after axis deg |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in latest["candidates"]:
        trans = ", ".join(f"{v:.3f}" for v in c["translation_step_camera_mm"])
        after = c["after"]
        lines.append(
            f"| {c['name']} | `[{trans}]` | {c['rotation_step_angle_deg']:.3f} | "
            f"{c['before']['weighted_3d_error_mm']:.3f} | {after['weighted_3d_error_mm']:.3f} | "
            f"{c['delta_weighted_3d_error_mm']:.3f} | {after['lateral_error_mm']:.3f} | "
            f"{after['depth_error_mm']:.3f} | {after['axis_angle_error_deg']:.3f} |"
        )
    lines += [
        "",
        "## 结论",
        "",
        "这是相机坐标系离线一步预测，不会移动真实机械臂。真实执行姿态步长前，必须先解决旋转向量到 Meca500 姿态命令的安全映射，并继续使用单步确认表。",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate one-step axis-aware candidates from the latest trocar pose.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-distance-mm", type=float, default=18.0)
    parser.add_argument("--max-translation-step-mm", type=float, default=0.75)
    parser.add_argument("--max-rotation-step-deg", type=float, default=1.0)
    parser.add_argument("--k-xy", type=float, default=0.35)
    parser.add_argument("--k-z", type=float, default=0.35)
    parser.add_argument("--k-rot", type=float, default=0.25)
    args = parser.parse_args()

    reports = collect_reports(args.input_dir)
    if not reports:
        raise RuntimeError(f"No real_trocar_ring_pose_report.json found under {args.input_dir}")
    records = [
        simulate_report(
            path,
            target_distance_mm=args.target_distance_mm,
            max_translation_step_mm=args.max_translation_step_mm,
            max_rotation_step_deg=args.max_rotation_step_deg,
            k_xy=args.k_xy,
            k_z=args.k_z,
            k_rot=args.k_rot,
        )
        for path in reports
    ]
    output = {
        "timestamp": datetime.now().isoformat(),
        "input_dir": str(args.input_dir),
        "sample_count": len(records),
        "latest": records[-1],
        "records": records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "axis_aware_candidates.json", output)
    write_markdown(args.output_dir / "axis_aware_candidates.md", output)

    latest = output["latest"]
    print("Axis-aware simulation written:", args.output_dir)
    print("Best candidate:", latest["best_candidate"])
    print("Best delta weighted mm:", latest["best_delta_weighted_3d_error_mm"])


if __name__ == "__main__":
    main()
