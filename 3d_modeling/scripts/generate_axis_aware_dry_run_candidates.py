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
DEFAULT_OUTPUT = ROOT / "outputs" / "control_3d" / "axis_aware_dry_run"


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


def wrapped_euler_delta_deg(a: list[float], b: list[float]) -> list[float]:
    delta = np.asarray(b[3:], dtype=np.float64) - np.asarray(a[3:], dtype=np.float64)
    wrapped = (delta + 180.0) % 360.0 - 180.0
    return [float(v) for v in wrapped]


def true_rotation_delta_deg(T_a: np.ndarray, T_b: np.ndarray) -> float:
    delta = T_a[:3, :3].T @ T_b[:3, :3]
    return float(np.linalg.norm(R.from_matrix(delta).as_rotvec()) * 180.0 / math.pi)


def error_metrics(p_mm: np.ndarray, axis: np.ndarray, target_distance_mm: float) -> dict:
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    lateral = float(np.linalg.norm(p_mm[:2]))
    depth = float(p_mm[2] - target_distance_mm)
    axis_deg = axis_angle_deg(axis)
    axis_equiv = float(target_distance_mm * math.sin(math.radians(axis_deg)))
    weighted = float(math.sqrt(lateral**2 + depth**2 + axis_equiv**2))
    return {
        "camera_position_mm": [float(v) for v in p_mm],
        "trocar_axis_camera": [float(v) for v in axis],
        "lateral_error_mm": lateral,
        "depth_error_mm": depth,
        "axis_angle_error_deg": axis_deg,
        "axis_equivalent_mm": axis_equiv,
        "weighted_3d_error_mm": weighted,
    }


def simulate_error_after(
    p_mm: np.ndarray,
    axis: np.ndarray,
    target_distance_mm: float,
    translation_camera_mm: np.ndarray,
    rotvec_camera: np.ndarray,
) -> dict:
    R_delta = R.from_rotvec(rotvec_camera).as_matrix()
    next_p = R_delta.T @ (p_mm - translation_camera_mm)
    next_axis = R_delta.T @ axis
    before = error_metrics(p_mm, axis, target_distance_mm)
    after = error_metrics(next_p, next_axis, target_distance_mm)
    return {
        "before": before,
        "after": after,
        "delta_weighted_3d_error_mm": after["weighted_3d_error_mm"] - before["weighted_3d_error_mm"],
        "improved": after["weighted_3d_error_mm"] < before["weighted_3d_error_mm"],
    }


def euler_risk(current_pose: list[float], candidate_pose: list[float], current_T: np.ndarray, candidate_T: np.ndarray) -> dict:
    raw_delta = np.asarray(candidate_pose[3:], dtype=np.float64) - np.asarray(current_pose[3:], dtype=np.float64)
    wrapped_delta = np.asarray(wrapped_euler_delta_deg(current_pose, candidate_pose), dtype=np.float64)
    true_delta = true_rotation_delta_deg(current_T, candidate_T)
    current_beta = float(current_pose[4])
    candidate_beta = float(candidate_pose[4])
    near_singularity = min(abs(abs(current_beta) - 90.0), abs(abs(candidate_beta) - 90.0)) < 5.0
    jump_risk = float(np.linalg.norm(raw_delta)) > max(10.0, 5.0 * max(true_delta, 1e-6))
    wrapped_jump_risk = float(np.linalg.norm(wrapped_delta)) > max(10.0, 5.0 * max(true_delta, 1e-6))
    return {
        "current_euler_deg": [float(v) for v in current_pose[3:]],
        "candidate_euler_deg": [float(v) for v in candidate_pose[3:]],
        "raw_euler_delta_deg": [float(v) for v in raw_delta],
        "wrapped_euler_delta_deg": [float(v) for v in wrapped_delta],
        "raw_euler_delta_norm_deg": float(np.linalg.norm(raw_delta)),
        "wrapped_euler_delta_norm_deg": float(np.linalg.norm(wrapped_delta)),
        "true_rotation_delta_deg": true_delta,
        "near_xyz_euler_singularity": near_singularity,
        "raw_euler_jump_risk": jump_risk,
        "wrapped_euler_jump_risk": wrapped_jump_risk,
        "safe_for_orientation_command": (not near_singularity) and (not wrapped_jump_risk) and true_delta <= 1.5,
    }


def make_candidate(
    name: str,
    T_base_link6: np.ndarray,
    T_link6_camera: np.ndarray,
    T_base_camera: np.ndarray,
    p_camera_mm: np.ndarray,
    axis_camera: np.ndarray,
    target_distance_mm: float,
    translation_camera_mm: np.ndarray,
    rotvec_camera: np.ndarray,
) -> dict:
    R_delta_camera = R.from_rotvec(rotvec_camera).as_matrix()
    T_base_camera_next = T_base_camera.copy()
    T_base_camera_next[:3, :3] = T_base_camera[:3, :3] @ R_delta_camera
    T_base_camera_next[:3, 3] = T_base_camera[:3, 3] + T_base_camera[:3, :3] @ (translation_camera_mm / 1000.0)
    T_base_link6_next = T_base_camera_next @ transform_inverse(T_link6_camera)

    current_pose = matrix_to_pose_mm_deg(T_base_link6)
    candidate_pose = matrix_to_pose_mm_deg(T_base_link6_next)
    base_translation_mm = T_base_camera[:3, :3] @ translation_camera_mm
    base_rotation_axis = None
    rotation_angle = float(np.linalg.norm(rotvec_camera))
    if rotation_angle > 1e-12:
        base_rotation_axis = (T_base_camera[:3, :3] @ (rotvec_camera / rotation_angle)).tolist()
    else:
        base_rotation_axis = [0.0, 0.0, 0.0]

    error_prediction = simulate_error_after(
        p_camera_mm,
        axis_camera,
        target_distance_mm,
        translation_camera_mm,
        rotvec_camera,
    )
    risk = euler_risk(current_pose, candidate_pose, T_base_link6, T_base_link6_next)
    position_delta = (T_base_link6_next[:3, 3] - T_base_link6[:3, 3]) * 1000.0
    return {
        "name": name,
        "current_pose_base_link6_mm_deg": [float(v) for v in current_pose],
        "candidate_pose_base_link6_mm_deg": [float(v) for v in candidate_pose],
        "position_delta_base_link6_mm": [float(v) for v in position_delta],
        "position_delta_norm_mm": float(np.linalg.norm(position_delta)),
        "translation_step_camera_mm": [float(v) for v in translation_camera_mm],
        "translation_step_base_mm": [float(v) for v in base_translation_mm],
        "rotation_step_camera_rotvec_deg": [float(math.degrees(v)) for v in rotvec_camera],
        "rotation_step_angle_deg": float(math.degrees(rotation_angle)),
        "rotation_axis_base": [float(v) for v in base_rotation_axis],
        "error_prediction": error_prediction,
        "euler_risk": risk,
        "recommended_for_real_orientation_test": (
            name != "translation_only"
            and error_prediction["improved"]
            and risk["safe_for_orientation_command"]
        ),
    }


def generate_for_report(
    path: Path,
    T_link6_camera: np.ndarray,
    target_distance_mm: float,
    max_translation_step_mm: float,
    max_rotation_step_deg: float,
    k_xy: float,
    k_z: float,
    k_rot: float,
) -> dict:
    data = load_json(path)
    robot = data.get("robot") or {}
    if not robot.get("success") or "T_base_link6" not in robot:
        return {
            "report": str(path),
            "image": data.get("image"),
            "status": "skipped",
            "reason": "Pose report has no successful robot T_base_link6 metadata.",
        }
    T_base_link6 = np.asarray(robot["T_base_link6"], dtype=np.float64)
    T_base_camera = T_base_link6 @ T_link6_camera
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
        make_candidate(
            "translation_only",
            T_base_link6,
            T_link6_camera,
            T_base_camera,
            p,
            axis,
            target_distance_mm,
            translation,
            zero_r,
        ),
        make_candidate(
            "rotation_only",
            T_base_link6,
            T_link6_camera,
            T_base_camera,
            p,
            axis,
            target_distance_mm,
            zero_t,
            rotvec,
        ),
        make_candidate(
            "combined_translation_rotation",
            T_base_link6,
            T_link6_camera,
            T_base_camera,
            p,
            axis,
            target_distance_mm,
            translation,
            rotvec,
        ),
    ]
    best = min(candidates, key=lambda item: item["error_prediction"]["after"]["weighted_3d_error_mm"])
    return {
        "report": str(path),
        "image": data.get("image"),
        "status": "ok",
        "robot": robot,
        "target_distance_mm": target_distance_mm,
        "parameters": {
            "max_translation_step_mm": max_translation_step_mm,
            "max_rotation_step_deg": max_rotation_step_deg,
            "k_xy": k_xy,
            "k_z": k_z,
            "k_rot": k_rot,
        },
        "current_error": error_metrics(p, axis, target_distance_mm),
        "candidates": candidates,
        "best_candidate": best["name"],
        "best_delta_weighted_3d_error_mm": best["error_prediction"]["delta_weighted_3d_error_mm"],
    }


def write_csv(path: Path, records: list[dict]) -> None:
    rows = []
    for sample_index, record in enumerate(records, start=1):
        if record.get("status") != "ok":
            rows.append({"sample_index": sample_index, "status": record.get("status"), "reason": record.get("reason")})
            continue
        for candidate in record["candidates"]:
            risk = candidate["euler_risk"]
            pred = candidate["error_prediction"]
            rows.append(
                {
                    "sample_index": sample_index,
                    "candidate": candidate["name"],
                    "position_delta_norm_mm": candidate["position_delta_norm_mm"],
                    "rotation_step_angle_deg": candidate["rotation_step_angle_deg"],
                    "before_weighted_mm": pred["before"]["weighted_3d_error_mm"],
                    "after_weighted_mm": pred["after"]["weighted_3d_error_mm"],
                    "delta_weighted_mm": pred["delta_weighted_3d_error_mm"],
                    "true_rotation_delta_deg": risk["true_rotation_delta_deg"],
                    "wrapped_euler_delta_norm_deg": risk["wrapped_euler_delta_norm_deg"],
                    "near_xyz_euler_singularity": risk["near_xyz_euler_singularity"],
                    "wrapped_euler_jump_risk": risk["wrapped_euler_jump_risk"],
                    "safe_for_orientation_command": risk["safe_for_orientation_command"],
                    "recommended_for_real_orientation_test": candidate["recommended_for_real_orientation_test"],
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


def write_markdown(path: Path, output: dict) -> None:
    latest = output["latest"]
    lines = [
        "# Axis-Aware Dry-Run 候选报告",
        "",
        f"生成时间：{output['timestamp']}",
        "",
        "## 最新样本",
        "",
        f"- 图像：`{latest.get('image')}`",
        f"- 状态：`{latest.get('status')}`",
    ]
    if latest.get("status") != "ok":
        lines += [f"- 跳过原因：`{latest.get('reason')}`", ""]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    err = latest["current_error"]
    lines += [
        f"- 当前综合误差：`{err['weighted_3d_error_mm']:.3f} mm`",
        f"- 当前横向误差：`{err['lateral_error_mm']:.3f} mm`",
        f"- 当前深度误差：`{err['depth_error_mm']:.3f} mm`",
        f"- 当前轴向夹角：`{err['axis_angle_error_deg']:.3f} deg`",
        f"- 最优 dry-run 候选：`{latest['best_candidate']}`",
        f"- 最优预测综合误差变化：`{latest['best_delta_weighted_3d_error_mm']:.3f} mm`",
        "",
        "## 候选表",
        "",
        "| candidate | pos step mm | rot deg | base rot axis | wrapped euler delta norm | singular risk | jump risk | safe orientation | predicted delta | recommended |",
        "|---|---:|---:|---|---:|---|---|---|---:|---|",
    ]
    for candidate in latest["candidates"]:
        risk = candidate["euler_risk"]
        axis = ", ".join(f"{v:.3f}" for v in candidate["rotation_axis_base"])
        lines.append(
            f"| {candidate['name']} | {candidate['position_delta_norm_mm']:.3f} | "
            f"{candidate['rotation_step_angle_deg']:.3f} | `[{axis}]` | "
            f"{risk['wrapped_euler_delta_norm_deg']:.3f} | {risk['near_xyz_euler_singularity']} | "
            f"{risk['wrapped_euler_jump_risk']} | {risk['safe_for_orientation_command']} | "
            f"{candidate['error_prediction']['delta_weighted_3d_error_mm']:.3f} | "
            f"{candidate['recommended_for_real_orientation_test']} |"
        )
    lines += [
        "",
        "## 最新候选详情",
        "",
    ]
    for candidate in latest["candidates"]:
        lines += [
            f"### {candidate['name']}",
            "",
            f"- 相机平移步长 mm：`{candidate['translation_step_camera_mm']}`",
            f"- 基坐标平移步长 mm：`{candidate['translation_step_base_mm']}`",
            f"- 相机旋转向量 deg：`{candidate['rotation_step_camera_rotvec_deg']}`",
            f"- 基坐标旋转轴：`{candidate['rotation_axis_base']}`",
            f"- 当前 Meca500 pose：`{candidate['current_pose_base_link6_mm_deg']}`",
            f"- 候选 Meca500 pose：`{candidate['candidate_pose_base_link6_mm_deg']}`",
            f"- 欧拉角风险：`{candidate['euler_risk']}`",
            "",
        ]
    lines += [
        "## 结论",
        "",
        "本报告仅为 dry-run，不移动真实机械臂。真实执行姿态候选前，应优先选择 `safe_for_orientation_command=True` 且 `recommended_for_real_orientation_test=True` 的候选，并继续使用单步确认和执行后重采图机制。",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate axis-aware dry-run candidates with Meca500 orientation risk checks.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--extrinsic", type=Path, default=DEFAULT_EXTRINSIC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-distance-mm", type=float, default=18.0)
    parser.add_argument("--max-translation-step-mm", type=float, default=0.75)
    parser.add_argument("--max-rotation-step-deg", type=float, default=1.0)
    parser.add_argument("--k-xy", type=float, default=0.35)
    parser.add_argument("--k-z", type=float, default=0.35)
    parser.add_argument("--k-rot", type=float, default=0.25)
    args = parser.parse_args()

    extrinsic = load_json(args.extrinsic)
    T_link6_camera = np.asarray(extrinsic["T_link6_camera"], dtype=np.float64)
    reports = collect_reports(args.input_dir)
    if not reports:
        raise RuntimeError(f"No real_trocar_ring_pose_report.json found under {args.input_dir}")
    records = [
        generate_for_report(
            path,
            T_link6_camera=T_link6_camera,
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
        "extrinsic": str(args.extrinsic),
        "sample_count": len(records),
        "latest": records[-1],
        "records": records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "axis_aware_dry_run_candidates.json", output)
    write_csv(args.output_dir / "axis_aware_dry_run_candidates.csv", records)
    write_markdown(args.output_dir / "axis_aware_dry_run_candidates.md", output)

    latest = output["latest"]
    print("Axis-aware dry-run written:", args.output_dir)
    print("Latest status:", latest.get("status"))
    if latest.get("status") == "ok":
        print("Best candidate:", latest["best_candidate"])
        print("Best delta weighted mm:", latest["best_delta_weighted_3d_error_mm"])


if __name__ == "__main__":
    main()
