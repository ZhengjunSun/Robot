#!/usr/bin/env python3
"""Control-time quality gate for single-frame trocar pose estimates."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = ROOT / "outputs" / "systematic_pose_dataset"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "pose_quality_gate"


@dataclass
class QualityGateThresholds:
    max_reprojection_error_px: float = 0.35
    min_depth_mm: float = 15.0
    max_depth_mm: float = 80.0
    min_major_diameter_px: float = 8.0
    max_major_diameter_px: float = 80.0
    max_ellipse_aspect_ratio: float = 1.8
    max_translation_jump_mm: float = 3.0
    max_depth_jump_mm: float = 2.0
    max_axis_jump_deg: float = 12.0


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def latest_dataset(root: Path) -> Path:
    candidates = sorted(root.glob("systematic_dataset_*"))
    if not candidates:
        raise FileNotFoundError(f"No systematic_dataset_* directories found in {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def pose_from_record(report: dict[str, Any]) -> dict[str, Any]:
    """Normalize different pose-report layouts into one gate input."""
    if "pose_estimation" in report:
        report = report["pose_estimation"]
    metrics = report.get("metrics") or {}
    pose = report.get("pose_camera_trocar_mm_deg")
    axis = report.get("trocar_axis_camera")
    reproj = report.get("mean_reprojection_error_px", metrics.get("mean_reprojection_error_px"))
    max_reproj = report.get("max_reprojection_error_px", metrics.get("max_reprojection_error_px"))
    return {
        "status": report.get("status"),
        "pose_camera_trocar_mm_deg": pose,
        "trocar_axis_camera": axis,
        "mean_reprojection_error_px": reproj,
        "max_reprojection_error_px": max_reproj,
        "outer_ellipse": report.get("outer_ellipse") or {},
        "detection_method": report.get("detection_method") or report.get("detection_method_used"),
    }


def axis_delta_deg(a: list[float] | None, b: list[float] | None) -> float | None:
    if a is None or b is None:
        return None
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na <= 1e-12 or nb <= 1e-12:
        return None
    va /= na
    vb /= nb
    return float(math.degrees(math.acos(float(np.clip(np.dot(va, vb), -1.0, 1.0)))))


def score_from_checks(reasons: list[str], normalized: dict[str, Any], thresholds: QualityGateThresholds) -> float:
    if reasons:
        return 0.0
    score = 1.0
    reproj = normalized.get("mean_reprojection_error_px")
    if reproj is not None:
        score -= 0.35 * min(max(float(reproj) / thresholds.max_reprojection_error_px, 0.0), 1.0)
    ellipse = normalized.get("outer_ellipse") or {}
    major = ellipse.get("major_diameter_px")
    minor = ellipse.get("minor_diameter_px")
    if major and minor:
        aspect = float(major) / max(float(minor), 1e-9)
        score -= 0.20 * min(max((aspect - 1.0) / (thresholds.max_ellipse_aspect_ratio - 1.0), 0.0), 1.0)
    return round(max(0.0, min(1.0, score)), 4)


def evaluate_pose_quality(
    report: dict[str, Any],
    previous_accepted: dict[str, Any] | None = None,
    thresholds: QualityGateThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or QualityGateThresholds()
    normalized = pose_from_record(report)
    reasons: list[str] = []
    warnings: list[str] = []

    if normalized.get("status") != "ok":
        reasons.append(f"pose_status_not_ok:{normalized.get('status')}")

    pose = normalized.get("pose_camera_trocar_mm_deg")
    if not pose or len(pose) < 3:
        reasons.append("missing_pose")
    else:
        z_mm = float(pose[2])
        if z_mm < thresholds.min_depth_mm or z_mm > thresholds.max_depth_mm:
            reasons.append("depth_out_of_range")

    axis = normalized.get("trocar_axis_camera")
    if not axis or len(axis) != 3:
        reasons.append("missing_axis")
    else:
        axis_norm = float(np.linalg.norm(np.asarray(axis, dtype=np.float64)))
        if axis_norm < 0.5 or axis_norm > 1.5:
            reasons.append("axis_norm_invalid")

    reproj = normalized.get("mean_reprojection_error_px")
    if reproj is None:
        reasons.append("missing_reprojection_error")
    elif float(reproj) > thresholds.max_reprojection_error_px:
        reasons.append("reprojection_too_high")

    ellipse = normalized.get("outer_ellipse") or {}
    major = ellipse.get("major_diameter_px")
    minor = ellipse.get("minor_diameter_px")
    if major is None or minor is None:
        warnings.append("missing_outer_ellipse")
    else:
        major_f = float(major)
        minor_f = float(minor)
        if major_f < thresholds.min_major_diameter_px or major_f > thresholds.max_major_diameter_px:
            reasons.append("ellipse_size_out_of_range")
        if minor_f <= 0.0:
            reasons.append("ellipse_minor_invalid")
        elif major_f / minor_f > thresholds.max_ellipse_aspect_ratio:
            reasons.append("ellipse_aspect_too_large")

    if previous_accepted is not None and pose and len(pose) >= 3:
        prev = pose_from_record(previous_accepted)
        prev_pose = prev.get("pose_camera_trocar_mm_deg")
        if prev_pose and len(prev_pose) >= 3:
            jump = float(np.linalg.norm(np.asarray(pose[:3], dtype=np.float64) - np.asarray(prev_pose[:3], dtype=np.float64)))
            depth_jump = abs(float(pose[2]) - float(prev_pose[2]))
            if jump > thresholds.max_translation_jump_mm:
                reasons.append("translation_jump_too_large")
            if depth_jump > thresholds.max_depth_jump_mm:
                reasons.append("depth_jump_too_large")
        axis_jump = axis_delta_deg(axis, prev.get("trocar_axis_camera"))
        if axis_jump is not None and axis_jump > thresholds.max_axis_jump_deg:
            reasons.append("axis_jump_too_large")

    accepted = len(reasons) == 0
    return {
        "accepted": accepted,
        "status": "accepted" if accepted else "rejected",
        "score": score_from_checks(reasons, normalized, thresholds),
        "reasons": reasons,
        "warnings": warnings,
        "thresholds": asdict(thresholds),
        "normalized": {
            "detection_method": normalized.get("detection_method"),
            "mean_reprojection_error_px": normalized.get("mean_reprojection_error_px"),
            "max_reprojection_error_px": normalized.get("max_reprojection_error_px"),
            "pose_camera_trocar_mm_deg": normalized.get("pose_camera_trocar_mm_deg"),
            "trocar_axis_camera": normalized.get("trocar_axis_camera"),
            "outer_ellipse": normalized.get("outer_ellipse"),
        },
    }


class GateState:
    """Stateful gate that compares each pose with the last accepted pose."""

    def __init__(self, thresholds: QualityGateThresholds | None = None) -> None:
        self.thresholds = thresholds or QualityGateThresholds()
        self.previous_accepted: dict[str, Any] | None = None

    def update(self, report: dict[str, Any]) -> dict[str, Any]:
        result = evaluate_pose_quality(report, self.previous_accepted, self.thresholds)
        if result["accepted"]:
            self.previous_accepted = report
        return result


def load_dataset_pose_reports(dataset_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    reports = []
    for path in sorted(dataset_dir.glob("pos_*/frame_*_pose.json")):
        reports.append((path, load_json(path)))
    if not reports:
        raise FileNotFoundError(f"No frame_*_pose.json files found under {dataset_dir}")
    return reports


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def report_from_comparison_row(row: dict[str, Any]) -> dict[str, Any]:
    pose = [
        parse_float(row.get("camera_x_mm")),
        parse_float(row.get("camera_y_mm")),
        parse_float(row.get("camera_z_mm")),
        0.0,
        0.0,
        0.0,
    ]
    if any(v is None for v in pose[:3]):
        pose_value = None
    else:
        pose_value = pose
    axis = [
        parse_float(row.get("axis_x")),
        parse_float(row.get("axis_y")),
        parse_float(row.get("axis_z")),
    ]
    axis_value = None if any(v is None for v in axis) else axis
    return {
        "status": row.get("status"),
        "detection_method": row.get("method_used") or row.get("method_requested"),
        "pose_camera_trocar_mm_deg": pose_value,
        "trocar_axis_camera": axis_value,
        "mean_reprojection_error_px": parse_float(row.get("mean_reprojection_error_px")),
        "max_reprojection_error_px": parse_float(row.get("max_reprojection_error_px")),
        "outer_ellipse": {
            "center": [
                parse_float(row.get("ellipse_center_x_px")),
                parse_float(row.get("ellipse_center_y_px")),
            ],
            "major_diameter_px": parse_float(row.get("ellipse_major_px")),
            "minor_diameter_px": parse_float(row.get("ellipse_minor_px")),
            "angle_deg": parse_float(row.get("ellipse_angle_deg")),
        },
    }


def validate_comparison_records(records_csv: Path, output_dir: Path, thresholds: QualityGateThresholds) -> dict[str, Any]:
    with records_csv.open("r", encoding="utf-8-sig") as f:
        records = list(csv.DictReader(f))
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in records:
        key = (row.get("method_requested", "unknown"), row.get("position_name", "unknown"))
        grouped.setdefault(key, []).append(row)

    rows = []
    method_counts: dict[str, dict[str, int]] = {}
    reason_counts: dict[str, int] = {}
    for (method, position_name), items in sorted(grouped.items()):
        gate = GateState(thresholds)
        for row in sorted(items, key=lambda r: int(parse_float(r.get("frame_index")) or 0)):
            report = report_from_comparison_row(row)
            result = gate.update(report)
            counts = method_counts.setdefault(method, {"total": 0, "accepted": 0, "rejected": 0})
            counts["total"] += 1
            if result["accepted"]:
                counts["accepted"] += 1
            else:
                counts["rejected"] += 1
                for reason in result["reasons"]:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
            rows.append({
                "method_requested": method,
                "position_name": position_name,
                "frame_index": row.get("frame_index"),
                "accepted": result["accepted"],
                "score": result["score"],
                "reasons": ";".join(result["reasons"]),
                "status": row.get("status"),
                "camera_z_mm": row.get("camera_z_mm"),
                "mean_reprojection_error_px": row.get("mean_reprojection_error_px"),
                "ellipse_major_px": row.get("ellipse_major_px"),
                "ellipse_minor_px": row.get("ellipse_minor_px"),
            })

    report = {
        "timestamp": datetime.now().isoformat(),
        "comparison_records_csv": str(records_csv),
        "thresholds": asdict(thresholds),
        "total_records": len(records),
        "method_counts": method_counts,
        "reason_counts": reason_counts,
        "records_csv": str(output_dir / "quality_gate_comparison_records.csv"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "quality_gate_comparison_records.csv", rows)
    write_json(output_dir / "quality_gate_comparison_report.json", report)
    write_comparison_markdown(output_dir / "quality_gate_comparison_report.md", report)
    return report


def write_comparison_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Pose Quality Gate on Detection Comparison",
        "",
        f"Generated: `{report['timestamp']}`",
        f"Records: `{report['comparison_records_csv']}`",
        "",
        "## Method Acceptance",
        "",
        "| method | accepted/total | acceptance rate |",
        "|---|---:|---:|",
    ]
    for method, counts in sorted(report["method_counts"].items()):
        rate = counts["accepted"] / max(counts["total"], 1)
        lines.append(f"| `{method}` | {counts['accepted']}/{counts['total']} | {rate:.2%} |")
    lines += ["", "## Rejection Reasons", ""]
    if report["reason_counts"]:
        for reason, count in sorted(report["reason_counts"].items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- `{reason}`: {count}")
    else:
        lines.append("- None")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def validate_dataset(dataset_dir: Path, output_dir: Path, thresholds: QualityGateThresholds) -> dict[str, Any]:
    reports = load_dataset_pose_reports(dataset_dir)
    rows = []
    grouped: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    for path, data in reports:
        grouped.setdefault(data.get("position_name", path.parent.name), []).append((path, data))

    accepted = 0
    rejected = 0
    reason_counts: dict[str, int] = {}
    for position_name, items in sorted(grouped.items()):
        gate = GateState(thresholds)
        for path, data in sorted(items):
            result = gate.update(data)
            if result["accepted"]:
                accepted += 1
            else:
                rejected += 1
                for reason in result["reasons"]:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
            norm = result["normalized"]
            pose = norm.get("pose_camera_trocar_mm_deg") or [None, None, None]
            rows.append({
                "file": str(path),
                "position_name": position_name,
                "frame_index": data.get("frame_index"),
                "accepted": result["accepted"],
                "score": result["score"],
                "reasons": ";".join(result["reasons"]),
                "warnings": ";".join(result["warnings"]),
                "camera_x_mm": pose[0],
                "camera_y_mm": pose[1],
                "camera_z_mm": pose[2],
                "mean_reprojection_error_px": norm.get("mean_reprojection_error_px"),
                "ellipse_major_px": (norm.get("outer_ellipse") or {}).get("major_diameter_px"),
                "ellipse_minor_px": (norm.get("outer_ellipse") or {}).get("minor_diameter_px"),
            })

    report = {
        "timestamp": datetime.now().isoformat(),
        "dataset_dir": str(dataset_dir),
        "thresholds": asdict(thresholds),
        "total_reports": len(reports),
        "accepted": accepted,
        "rejected": rejected,
        "acceptance_rate": accepted / max(len(reports), 1),
        "reason_counts": reason_counts,
        "records_csv": str(output_dir / "quality_gate_records.csv"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "quality_gate_records.csv", rows)
    write_json(output_dir / "quality_gate_report.json", report)
    write_markdown(output_dir / "quality_gate_report.md", report)
    return report


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Pose Quality Gate Validation Report",
        "",
        f"Generated: `{report['timestamp']}`",
        f"Dataset: `{report['dataset_dir']}`",
        f"Total reports: `{report['total_reports']}`",
        f"Accepted: `{report['accepted']}`",
        f"Rejected: `{report['rejected']}`",
        f"Acceptance rate: `{report['acceptance_rate']:.2%}`",
        "",
        "## Thresholds",
        "",
    ]
    for key, value in report["thresholds"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines += ["", "## Rejection Reasons", ""]
    if report["reason_counts"]:
        for reason, count in sorted(report["reason_counts"].items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- `{reason}`: {count}")
    else:
        lines.append("- None")
    lines += [
        "",
        "## Control Use",
        "",
        "This gate should be evaluated after pose estimation and before any robot correction command.",
        "A rejected pose must be logged and skipped instead of entering the controller.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the trocar pose quality gate on a dataset.")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--comparison-records-csv", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-reprojection-error-px", type=float, default=0.35)
    parser.add_argument("--max-translation-jump-mm", type=float, default=3.0)
    parser.add_argument("--max-depth-jump-mm", type=float, default=2.0)
    parser.add_argument("--max-axis-jump-deg", type=float, default=12.0)
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / f"quality_gate_{stamp}"
    thresholds = QualityGateThresholds(
        max_reprojection_error_px=args.max_reprojection_error_px,
        max_translation_jump_mm=args.max_translation_jump_mm,
        max_depth_jump_mm=args.max_depth_jump_mm,
        max_axis_jump_deg=args.max_axis_jump_deg,
    )
    if args.comparison_records_csv:
        report = validate_comparison_records(args.comparison_records_csv, output_dir, thresholds)
        print("Quality gate comparison validation complete.")
        print("Output:", output_dir)
        for method, counts in sorted(report["method_counts"].items()):
            rate = counts["accepted"] / max(counts["total"], 1)
            print(f"  {method}: {counts['accepted']}/{counts['total']} accepted ({rate:.2%})")
        if report["reason_counts"]:
            print("Rejection reasons:", report["reason_counts"])
    else:
        dataset_dir = args.dataset_dir if args.dataset_dir else latest_dataset(args.dataset_root)
        report = validate_dataset(dataset_dir, output_dir, thresholds)
        print("Quality gate validation complete.")
        print("Output:", output_dir)
        print(f"Accepted: {report['accepted']}/{report['total_reports']} ({report['acceptance_rate']:.2%})")
        if report["reason_counts"]:
            print("Rejection reasons:", report["reason_counts"])


if __name__ == "__main__":
    main()
