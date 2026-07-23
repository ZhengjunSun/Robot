#!/usr/bin/env python3
"""Multi-frame trocar pose filter with outlier rejection and confidence scoring.

Addresses the single-frame PnP depth jumping problem by maintaining a sliding
window of pose estimates, rejecting outliers, and outputting a stable filtered
pose with a confidence score.
"""

from __future__ import annotations

import argparse
import csv
import math
from datetime import datetime
from pathlib import Path

import numpy as np

from handeye_common import load_json, write_json, matrix_to_pose_mm_deg


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs" / "trocar_pose_from_ring"
DEFAULT_OUTPUT = ROOT / "outputs" / "pose_filtered"


class PoseHistory:
    """Sliding-window pose filter with outlier rejection and EMA smoothing."""

    def __init__(
        self,
        window_size: int = 7,
        ema_alpha: float = 0.4,
        max_reprojection_error_px: float = 0.5,
        max_translation_jump_mm: float = 5.0,
        max_axis_angle_jump_deg: float = 15.0,
        min_samples_for_filter: int = 3,
    ) -> None:
        self.window_size = window_size
        self.ema_alpha = ema_alpha
        self.max_reprojection_error_px = max_reprojection_error_px
        self.max_translation_jump_mm = max_translation_jump_mm
        self.max_axis_angle_jump_deg = max_axis_angle_jump_deg
        self.min_samples_for_filter = min_samples_for_filter
        self.history: list[dict] = []
        self.filtered_translation: np.ndarray | None = None
        self.filtered_axis: np.ndarray | None = None

    def reset(self) -> None:
        self.history.clear()
        self.filtered_translation = None
        self.filtered_axis = None

    def update(self, pose_report: dict) -> dict:
        """Add a pose report, run filtering, return filtered result."""
        raw_pose = pose_report.get("pose_camera_trocar_mm_deg")
        raw_reproj = pose_report.get("metrics", {}).get("mean_reprojection_error_px", float("inf"))
        raw_axis = pose_report.get("trocar_axis_camera")
        raw_T = pose_report.get("T_camera_trocar")

        if raw_pose is None or raw_axis is None:
            return {"status": "invalid_input", "confidence": 0.0}

        entry = {
            "pose_mm_deg": list(raw_pose),
            "translation_mm": np.array(raw_pose[:3], dtype=np.float64),
            "axis": np.array(raw_axis, dtype=np.float64),
            "reprojection_error_px": float(raw_reproj),
            "T_camera_trocar": np.array(raw_T, dtype=np.float64) if raw_T is not None else None,
        }
        self.history.append(entry)
        if len(self.history) > self.window_size:
            self.history.pop(0)

        # Need enough samples before filtering
        if len(self.history) < self.min_samples_for_filter:
            return {
                "status": "insufficient_samples",
                "sample_count": len(self.history),
                "confidence": 0.0,
                "raw_pose_camera_trocar_mm_deg": raw_pose,
            }

        # Stage 1: reject reprojection outliers
        inliers = self._reject_reprojection_outliers()
        if len(inliers) < self.min_samples_for_filter:
            return {
                "status": "insufficient_inliers",
                "sample_count": len(self.history),
                "inlier_count": len(inliers),
                "confidence": 0.0,
                "raw_pose_camera_trocar_mm_deg": raw_pose,
            }

        # Stage 2: reject pose consistency outliers
        inliers = self._reject_pose_outliers(inliers)
        if len(inliers) < self.min_samples_for_filter:
            return {
                "status": "insufficient_inliers_after_consistency",
                "sample_count": len(self.history),
                "inlier_count": len(inliers),
                "confidence": 0.0,
                "raw_pose_camera_trocar_mm_deg": raw_pose,
            }

        # Median filter
        median_trans, median_axis = self._median_filter(inliers)

        # EMA smoothing
        filtered_trans, filtered_axis = self._ema_smooth(median_trans, median_axis)

        self.filtered_translation = filtered_trans
        self.filtered_axis = filtered_axis

        # Reconstruct filtered T_camera_trocar from median inlier
        filtered_T = self._reconstruct_transform(inliers, filtered_trans, filtered_axis)

        # Confidence
        confidence = self._compute_confidence(
            len(inliers), len(self.history),
            [e["reprojection_error_px"] for e in inliers],
        )

        return {
            "status": "ok",
            "filtered_pose_camera_trocar_mm_deg": matrix_to_pose_mm_deg(filtered_T),
            "filtered_T_camera_trocar": filtered_T.tolist(),
            "filtered_translation_camera_trocar_m": filtered_trans.tolist(),
            "filtered_trocar_axis_camera": filtered_axis.tolist(),
            "confidence": round(confidence, 4),
            "sample_count": len(self.history),
            "inlier_count": len(inliers),
            "outlier_count": len(self.history) - len(inliers),
            "input_reprojection_error_px": float(raw_reproj),
            "raw_pose_camera_trocar_mm_deg": raw_pose,
        }

    def get_filtered_pose(self) -> dict | None:
        if self.filtered_translation is None:
            return None
        return {
            "filtered_translation_camera_trocar_m": self.filtered_translation.tolist(),
            "filtered_trocar_axis_camera": self.filtered_axis.tolist(),
        }

    def _reject_reprojection_outliers(self) -> list[dict]:
        return [
            e for e in self.history
            if e["reprojection_error_px"] <= self.max_reprojection_error_px
        ]

    def _reject_pose_outliers(self, inliers: list[dict]) -> list[dict]:
        translations = np.array([e["translation_mm"] for e in inliers])
        axes = np.array([e["axis"] for e in inliers])

        med_trans = np.median(translations, axis=0)
        med_axis = np.median(axes, axis=0)
        med_axis_norm = med_axis / max(np.linalg.norm(med_axis), 1e-12)

        result = []
        for e in inliers:
            trans_dev = np.linalg.norm(e["translation_mm"] - med_trans)
            if trans_dev > self.max_translation_jump_mm:
                continue
            axis_norm = e["axis"] / max(np.linalg.norm(e["axis"]), 1e-12)
            angle = math.degrees(math.acos(float(np.clip(np.dot(axis_norm, med_axis_norm), -1.0, 1.0))))
            if angle > self.max_axis_angle_jump_deg:
                continue
            result.append(e)
        return result

    def _median_filter(self, inliers: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        translations = np.array([e["translation_mm"] for e in inliers])
        axes = np.array([e["axis"] for e in inliers])
        med_trans = np.median(translations, axis=0)
        med_axis = np.median(axes, axis=0)
        norm = np.linalg.norm(med_axis)
        if norm > 1e-12:
            med_axis = med_axis / norm
        return med_trans, med_axis

    def _ema_smooth(self, median_trans: np.ndarray, median_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.filtered_translation is None:
            return median_trans.copy(), median_axis.copy()

        alpha = self.ema_alpha
        smoothed_trans = alpha * median_trans + (1 - alpha) * self.filtered_translation

        # SLERP for axis direction
        dot = float(np.clip(np.dot(self.filtered_axis, median_axis), -1.0, 1.0))
        omega = math.acos(dot)
        if omega < 1e-6:
            smoothed_axis = self.filtered_axis.copy()
        else:
            sin_omega = math.sin(omega)
            smoothed_axis = (math.sin((1 - alpha) * omega) / sin_omega) * self.filtered_axis + \
                            (math.sin(alpha * omega) / sin_omega) * median_axis
            norm = np.linalg.norm(smoothed_axis)
            if norm > 1e-12:
                smoothed_axis = smoothed_axis / norm

        return smoothed_trans, smoothed_axis

    def _reconstruct_transform(
        self, inliers: list[dict],
        filtered_trans: np.ndarray,
        filtered_axis: np.ndarray,
    ) -> np.ndarray:
        # Use the best inlier's rotation as base, replace z-axis with filtered axis
        best = min(inliers, key=lambda e: e["reprojection_error_px"])
        T = best["T_camera_trocar"]
        if T is None:
            T = np.eye(4)
        result = T.copy()
        result[:3, 3] = filtered_trans / 1000.0  # mm → m
        # Replace the Z column (trocar axis) with filtered axis
        result[:3, 2] = filtered_axis
        # Re-orthogonalize: X = Y × Z, Y = Z × X
        z = filtered_axis / max(np.linalg.norm(filtered_axis), 1e-12)
        x = np.cross(result[:3, 1], z)
        x = x / max(np.linalg.norm(x), 1e-12)
        y = np.cross(z, x)
        result[:3, 0] = x
        result[:3, 1] = y
        return result

    def _compute_confidence(
        self, inlier_count: int, total_count: int, reproj_errors: list[float],
    ) -> float:
        inlier_ratio = inlier_count / max(total_count, 1)
        if reproj_errors:
            mean_reproj = sum(reproj_errors) / len(reproj_errors)
            reproj_quality = max(0.0, 1.0 - mean_reproj / self.max_reprojection_error_px)
        else:
            reproj_quality = 0.0
        return 0.5 * inlier_ratio + 0.3 * reproj_quality + 0.2


def filter_pose_reports_from_files(
    report_paths: list[Path],
    window_size: int = 7,
    ema_alpha: float = 0.4,
) -> dict:
    """Batch-mode: process a list of pose report files sequentially."""
    history = PoseHistory(window_size=window_size, ema_alpha=ema_alpha)
    trace = []
    for path in report_paths:
        data = load_json(path)
        result = history.update(data)
        trace.append({
            "report": str(path.name),
            "raw_z_mm": data.get("pose_camera_trocar_mm_deg", [0, 0, 0])[2] if data.get("pose_camera_trocar_mm_deg") else None,
            "filtered_z_mm": result.get("filtered_pose_camera_trocar_mm_deg", [None, None, None])[2] if result.get("filtered_pose_camera_trocar_mm_deg") else None,
            "confidence": result.get("confidence", 0.0),
            "status": result.get("status"),
            "raw_reproj_px": data.get("metrics", {}).get("mean_reprojection_error_px"),
        })
    final = history.get_filtered_pose()
    return {
        "trace": trace,
        "final_filtered": final,
        "total_processed": len(report_paths),
    }


def collect_reports(input_dir: Path) -> list[Path]:
    """Find all pose report JSON files, deduplicated by latest per image."""
    paths = sorted(input_dir.rglob("real_trocar_ring_pose_report.json"))
    latest_by_image: dict[str, Path] = {}
    for path in paths:
        data = load_json(path)
        image = data.get("image", str(path))
        current = latest_by_image.get(image)
        if current is None or path.stat().st_mtime >= current.stat().st_mtime:
            latest_by_image[image] = path
    return sorted(latest_by_image.values(), key=lambda p: p.stat().st_mtime)


def write_markdown(path: Path, result: dict, args: argparse.Namespace) -> None:
    lines = [
        "# Multi-Frame Pose Filter Report",
        "",
        f"Generated: `{datetime.now().isoformat()}`",
        f"Window size: `{args.window_size}`",
        f"EMA alpha: `{args.ema_alpha}`",
        f"Max reprojection error: `{args.max_reprojection_error_px}` px",
        f"Max translation jump: `{args.max_translation_jump_mm}` mm",
        f"Total reports processed: `{result['total_processed']}`",
        "",
        "## Filter Trace",
        "",
        "| # | report | raw_z (mm) | filtered_z (mm) | confidence | status | raw_reproj (px) |",
        "|---:|---|---:|---:|---:|---|---:|",
    ]
    for i, row in enumerate(result["trace"], 1):
        raw_z = f"{row['raw_z_mm']:.3f}" if row['raw_z_mm'] is not None else "N/A"
        filt_z = f"{row['filtered_z_mm']:.3f}" if row['filtered_z_mm'] is not None else "N/A"
        reproj = f"{row['raw_reproj_px']:.4f}" if row['raw_reproj_px'] is not None else "N/A"
        lines.append(
            f"| {i} | `{row['report']}` | {raw_z} | "
            f"{filt_z} | {row['confidence']:.3f} | "
            f"`{row['status']}` | {reproj} |"
        )
    final = result.get("final_filtered")
    if final:
        lines += [
            "",
            "## Final Filtered Pose",
            "",
            f"- Translation (m): `{final['filtered_translation_camera_trocar_m']}`",
            f"- Axis: `{final['filtered_trocar_axis_camera']}`",
        ]
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv_trace(path: Path, trace: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not trace:
        path.write_text("", encoding="utf-8")
        return
    fields = list(trace[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(trace)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-frame trocar pose filter with outlier rejection.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--window-size", type=int, default=7, help="Sliding window size")
    parser.add_argument("--ema-alpha", type=float, default=0.4, help="EMA smoothing (lower = smoother)")
    parser.add_argument("--max-reprojection-error-px", type=float, default=0.5)
    parser.add_argument("--max-translation-jump-mm", type=float, default=5.0)
    parser.add_argument("--max-axis-angle-jump-deg", type=float, default=15.0)
    parser.add_argument("--min-samples", type=int, default=3)
    args = parser.parse_args()

    report_paths = collect_reports(args.input_dir)
    if not report_paths:
        print("No pose report files found in:", args.input_dir)
        return

    print(f"Found {len(report_paths)} pose reports.")

    result = filter_pose_reports_from_files(
        report_paths,
        window_size=args.window_size,
        ema_alpha=args.ema_alpha,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "pose_filter_report.json", result)
    write_csv_trace(args.output_dir / "pose_filter_trace.csv", result["trace"])
    write_markdown(args.output_dir / "pose_filter_report.md", result, args)

    # Print summary
    final = result.get("final_filtered")
    if final:
        print(f"Final filtered translation (m): {final['filtered_translation_camera_trocar_m']}")
        print(f"Final filtered axis: {final['filtered_trocar_axis_camera']}")
    print(f"Total processed: {result['total_processed']}")

    # Print last few traces for inspection
    for row in result["trace"][-5:]:
        print(f"  {row['report']}: raw_z={row['raw_z_mm']:.3f}mm -> "
              f"filtered_z={row['filtered_z_mm']:.3f}mm, "
              f"conf={row['confidence']:.3f}, status={row['status']}")

    print("Output:", args.output_dir)


if __name__ == "__main__":
    main()
