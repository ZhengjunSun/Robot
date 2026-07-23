from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs" / "trocar_pose_from_ring"
DEFAULT_OUTPUT = ROOT / "outputs" / "trocar_pose_summary"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def collect_reports(input_dir: Path) -> list[Path]:
    paths = sorted(input_dir.rglob("real_trocar_ring_pose_report.json"))
    # Keep the latest result per source image in case a report was regenerated in-place.
    latest_by_image: dict[str, Path] = {}
    for path in paths:
        data = load_json(path)
        image = data.get("image", str(path))
        current = latest_by_image.get(image)
        if current is None or path.stat().st_mtime >= current.stat().st_mtime:
            latest_by_image[image] = path
    return sorted(latest_by_image.values(), key=lambda p: p.stat().st_mtime)


def flatten_report(path: Path) -> dict:
    data = load_json(path)
    camera_pose = data.get("pose_camera_trocar_mm_deg", [None] * 6)
    base_pose = data.get("pose_base_trocar_mm_deg", [None] * 6)
    axis_camera = data.get("trocar_axis_camera", [None] * 3)
    axis_base = data.get("trocar_axis_base", [None] * 3)
    metrics = data.get("metrics", {})
    return {
        "report": str(path.relative_to(ROOT).as_posix()),
        "timestamp": data.get("timestamp"),
        "image": data.get("image"),
        "mean_reprojection_error_px": metrics.get("mean_reprojection_error_px"),
        "max_reprojection_error_px": metrics.get("max_reprojection_error_px"),
        "used_point_count": metrics.get("used_point_count"),
        "camera_x_mm": camera_pose[0],
        "camera_y_mm": camera_pose[1],
        "camera_z_mm": camera_pose[2],
        "camera_rx_deg": camera_pose[3],
        "camera_ry_deg": camera_pose[4],
        "camera_rz_deg": camera_pose[5],
        "base_x_mm": base_pose[0],
        "base_y_mm": base_pose[1],
        "base_z_mm": base_pose[2],
        "base_rx_deg": base_pose[3],
        "base_ry_deg": base_pose[4],
        "base_rz_deg": base_pose[5],
        "axis_camera_x": axis_camera[0],
        "axis_camera_y": axis_camera[1],
        "axis_camera_z": axis_camera[2],
        "axis_base_x": axis_base[0],
        "axis_base_y": axis_base[1],
        "axis_base_z": axis_base[2],
        "outer_major_diameter_px": data.get("outer_ellipse", {}).get("major_diameter_px"),
        "outer_minor_diameter_px": data.get("outer_ellipse", {}).get("minor_diameter_px"),
    }


def numeric_summary(records: list[dict], key: str) -> dict:
    values = np.array([record[key] for record in records if record.get(key) is not None], dtype=float)
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
        "# 真实戳卡 6D 位姿估计汇总",
        "",
        f"生成时间：{report['timestamp']}",
        "",
        f"样本数：{report['sample_count']}",
        "",
        "## 汇总指标",
        "",
    ]
    for key, title in [
        ("mean_reprojection_error_px", "平均重投影误差 px"),
        ("camera_z_mm", "相机坐标系 Z 距离 mm"),
        ("base_x_mm", "Base X mm"),
        ("base_y_mm", "Base Y mm"),
        ("base_z_mm", "Base Z mm"),
    ]:
        summary = report["summary"].get(key, {})
        if summary.get("count", 0) == 0:
            continue
        lines.append(
            f"- {title}: mean={summary['mean']:.6f}, std={summary['std']:.6f}, "
            f"min={summary['min']:.6f}, max={summary['max']:.6f}"
        )

    lines += [
        "",
        "## 样本",
        "",
        "| # | image | reproj px | camera xyz mm | base xyz mm |",
        "|---:|---|---:|---|---|",
    ]
    for index, record in enumerate(report["records"], start=1):
        lines.append(
            f"| {index} | `{record['image']}` | {record['mean_reprojection_error_px']:.3f} | "
            f"[{record['camera_x_mm']:.3f}, {record['camera_y_mm']:.3f}, {record['camera_z_mm']:.3f}] | "
            f"[{record['base_x_mm']:.3f}, {record['base_y_mm']:.3f}, {record['base_z_mm']:.3f}] |"
        )

    lines += [
        "",
        "## 说明",
        "",
        "当前结果使用橙色外圈椭圆拟合 + PnP。戳卡入口近似圆环具有绕轴旋转对称性，因此最可信的是入口中心位置和孔轴方向，绕轴滚转角仅作为算法约定。",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize real trocar ring pose estimates.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    reports = collect_reports(args.input_dir)
    records = [flatten_report(path) for path in reports]
    summary_keys = [
        "mean_reprojection_error_px",
        "max_reprojection_error_px",
        "camera_x_mm",
        "camera_y_mm",
        "camera_z_mm",
        "base_x_mm",
        "base_y_mm",
        "base_z_mm",
        "outer_major_diameter_px",
        "outer_minor_diameter_px",
    ]
    summary = {key: numeric_summary(records, key) for key in summary_keys}
    report = {
        "timestamp": datetime.now().isoformat(),
        "input_dir": str(args.input_dir),
        "sample_count": len(records),
        "summary": summary,
        "records": records,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "trocar_pose_summary.json", report)
    write_csv(args.output_dir / "trocar_pose_summary.csv", records)
    write_markdown(args.output_dir / "trocar_pose_summary.md", report)

    print("Trocar pose summary written:", args.output_dir)
    print("Samples:", len(records))
    if records:
        print("Mean reprojection summary:", summary["mean_reprojection_error_px"])


if __name__ == "__main__":
    main()
