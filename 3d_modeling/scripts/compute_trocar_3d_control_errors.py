from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs" / "trocar_pose_from_ring"
DEFAULT_OUTPUT = ROOT / "outputs" / "control_3d"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def collect_reports(input_dir: Path) -> list[Path]:
    paths = sorted(input_dir.rglob("real_trocar_ring_pose_report.json"))
    latest_by_image: dict[str, Path] = {}
    for path in paths:
        data = load_json(path)
        image = data.get("image", str(path))
        current = latest_by_image.get(image)
        if current is None or path.stat().st_mtime >= current.stat().st_mtime:
            latest_by_image[image] = path
    return sorted(latest_by_image.values(), key=lambda p: p.stat().st_mtime)


def axis_angle_deg(axis: np.ndarray, target_axis: np.ndarray) -> float:
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    target_axis = target_axis / max(np.linalg.norm(target_axis), 1e-12)
    dot = float(np.clip(np.dot(axis, target_axis), -1.0, 1.0))
    return float(math.degrees(math.acos(dot)))


def evaluate_report(path: Path, target_distance_mm: float) -> dict:
    data = load_json(path)
    pose_camera = data["pose_camera_trocar_mm_deg"]
    pose_base = data.get("pose_base_trocar_mm_deg", [None] * 6)
    translation = np.array(pose_camera[:3], dtype=np.float64)
    axis_camera = np.array(data["trocar_axis_camera"], dtype=np.float64)
    target_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    lateral_error = float(np.linalg.norm(translation[:2]))
    depth_error = float(translation[2] - target_distance_mm)
    axis_error = axis_angle_deg(axis_camera, target_axis)
    weighted_error = float(
        math.sqrt(
            lateral_error**2
            + depth_error**2
            + (target_distance_mm * math.sin(math.radians(axis_error))) ** 2
        )
    )

    return {
        "report": str(path.relative_to(ROOT).as_posix()),
        "image": data.get("image"),
        "target_distance_mm": target_distance_mm,
        "camera_x_mm": float(translation[0]),
        "camera_y_mm": float(translation[1]),
        "camera_z_mm": float(translation[2]),
        "base_x_mm": pose_base[0],
        "base_y_mm": pose_base[1],
        "base_z_mm": pose_base[2],
        "lateral_error_mm": lateral_error,
        "depth_error_mm": depth_error,
        "axis_angle_error_deg": axis_error,
        "weighted_3d_error_mm": weighted_error,
        "trocar_axis_camera": axis_camera.tolist(),
        "pnp_mean_reprojection_error_px": data.get("metrics", {}).get("mean_reprojection_error_px"),
    }


def numeric_summary(records: list[dict], key: str) -> dict:
    values = np.asarray([record[key] for record in records if record.get(key) is not None], dtype=np.float64)
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def write_csv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# 戳卡 3D 控制误差定义与样本评估",
        "",
        f"生成时间：{report['timestamp']}",
        "",
        f"目标相机-戳卡距离：`{report['target_distance_mm']:.3f} mm`",
        "",
        "## 误差定义",
        "",
        "- 横向误差：`sqrt(x^2 + y^2)`，其中 `[x, y, z]` 是戳卡入口中心在相机坐标系下的位置。",
        "- 距离误差：`z - z_target`。",
        "- 孔轴夹角误差：戳卡孔轴与相机光轴 `[0, 0, 1]` 的夹角。",
        "- 综合 3D 误差：横向误差、距离误差和轴线误差对应横向位移的加权范数。",
        "",
        "## 汇总",
        "",
    ]
    for key, label in [
        ("lateral_error_mm", "横向误差 mm"),
        ("depth_error_mm", "距离误差 mm"),
        ("axis_angle_error_deg", "轴线夹角 deg"),
        ("weighted_3d_error_mm", "综合 3D 误差 mm"),
    ]:
        stats = report["summary"].get(key, {})
        if stats.get("count", 0):
            lines.append(
                f"- {label}: mean={stats['mean']:.6f}, std={stats['std']:.6f}, "
                f"min={stats['min']:.6f}, max={stats['max']:.6f}"
            )

    lines += [
        "",
        "## 样本",
        "",
        "| # | image | lateral mm | depth mm | axis deg | weighted mm |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for index, record in enumerate(report["records"], start=1):
        lines.append(
            f"| {index} | `{record['image']}` | {record['lateral_error_mm']:.3f} | "
            f"{record['depth_error_mm']:.3f} | {record['axis_angle_error_deg']:.3f} | "
            f"{record['weighted_3d_error_mm']:.3f} |"
        )

    lines += [
        "",
        "## 控制含义",
        "",
        "后续 3D 闭环控制不再直接最小化图像中心偏差，而是使 `T_camera_trocar` 满足：",
        "",
        "```text",
        "x -> 0",
        "y -> 0",
        "z -> z_target",
        "trocar_axis_camera -> [0, 0, 1]",
        "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute 3D control errors from real trocar pose estimates.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-distance-mm", type=float, default=18.0)
    args = parser.parse_args()

    reports = collect_reports(args.input_dir)
    records = [evaluate_report(path, args.target_distance_mm) for path in reports]
    summary_keys = [
        "lateral_error_mm",
        "depth_error_mm",
        "axis_angle_error_deg",
        "weighted_3d_error_mm",
        "pnp_mean_reprojection_error_px",
    ]
    output = {
        "timestamp": datetime.now().isoformat(),
        "input_dir": str(args.input_dir),
        "target_distance_mm": args.target_distance_mm,
        "sample_count": len(records),
        "summary": {key: numeric_summary(records, key) for key in summary_keys},
        "records": records,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "trocar_3d_control_errors.json", output)
    write_csv(args.output_dir / "trocar_3d_control_errors.csv", records)
    write_markdown(args.output_dir / "trocar_3d_control_errors.md", output)

    print("3D control errors written:", args.output_dir)
    print("Samples:", len(records))
    if records:
        print("Weighted error summary:", output["summary"]["weighted_3d_error_mm"])


if __name__ == "__main__":
    main()
