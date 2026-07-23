#!/usr/bin/env python3
"""Analyze systematic pose dataset: per-position stats, filter comparison, thesis plots.

Processes the dataset from collect_systematic_pose_dataset.py and produces:
- Per-position statistics (detection rate, jitter, depth accuracy)
- Filter comparison (unfiltered vs multi-frame filtered)
- Publication-quality plots (matplotlib)
- Markdown report for thesis inclusion
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "outputs" / "systematic_pose_dataset"
DEFAULT_OUTPUT = ROOT / "outputs" / "pose_dataset_analysis"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_latest_dataset(base_dir: Path) -> Path:
    """Find the latest dataset directory."""
    datasets = sorted(base_dir.glob("systematic_dataset_*"))
    if not datasets:
        raise FileNotFoundError(f"No dataset directories found in {base_dir}")
    # Pick the latest by modification time
    return max(datasets, key=lambda p: p.stat().st_mtime)


def compute_position_statistics(frames: list[dict]) -> dict:
    """Compute statistics for frames at a single robot position."""
    total = len(frames)
    successful = [f for f in frames if f.get("detection_status") == "ok"]
    detected = len(successful)

    # Extract metrics from successful frames
    depths = [f.get("camera_z_mm") for f in successful if f.get("camera_z_mm") is not None]
    lateral_xs = [f.get("camera_x_mm") for f in successful if f.get("camera_x_mm") is not None]
    lateral_ys = [f.get("camera_y_mm") for f in successful if f.get("camera_y_mm") is not None]
    reproj_errors = [f.get("reprojection_error_px") for f in successful if f.get("reprojection_error_px") is not None]

    result = {
        "total_frames": total,
        "detected_frames": detected,
        "detection_success_rate": detected / max(total, 1),
    }

    if depths:
        depths_arr = np.array(depths, dtype=np.float64)
        median_depth = float(np.median(depths_arr))
        deviations = np.abs(depths_arr - median_depth)
        result["depth_mean_mm"] = float(np.mean(depths_arr))
        result["depth_std_mm"] = float(np.std(depths_arr))
        result["depth_median_mm"] = median_depth
        result["depth_min_mm"] = float(np.min(depths_arr))
        result["depth_max_mm"] = float(np.max(depths_arr))
        result["depth_range_mm"] = float(np.max(depths_arr) - np.min(depths_arr))
        # Outlier: frames where depth deviates > 3*std from median
        if result["depth_std_mm"] > 0:
            outlier_mask = deviations > 3.0 * result["depth_std_mm"]
            result["depth_outlier_count"] = int(outlier_mask.sum())
        else:
            result["depth_outlier_count"] = 0
        # Jitter RMS
        result["depth_jitter_rms_mm"] = float(np.sqrt(np.mean((depths_arr - np.mean(depths_arr)) ** 2)))
    else:
        for key in ["depth_mean_mm", "depth_std_mm", "depth_median_mm", "depth_min_mm",
                     "depth_max_mm", "depth_range_mm", "depth_outlier_count", "depth_jitter_rms_mm"]:
            result[key] = None

    if lateral_xs and lateral_ys:
        lateral = np.sqrt(np.array(lateral_xs) ** 2 + np.array(lateral_ys) ** 2)
        result["lateral_mean_mm"] = float(np.mean(lateral))
        result["lateral_std_mm"] = float(np.std(lateral))
        result["lateral_jitter_rms_mm"] = float(np.sqrt(np.mean((lateral - np.mean(lateral)) ** 2)))
    else:
        result["lateral_mean_mm"] = None
        result["lateral_std_mm"] = None
        result["lateral_jitter_rms_mm"] = None

    if reproj_errors:
        result["reproj_mean_px"] = float(np.mean(reproj_errors))
        result["reproj_std_px"] = float(np.std(reproj_errors))
        result["reproj_max_px"] = float(np.max(reproj_errors))
    else:
        result["reproj_mean_px"] = None
        result["reproj_std_px"] = None
        result["reproj_max_px"] = None

    # Detection method breakdown
    methods = {}
    for f in successful:
        m = f.get("detection_method", "unknown")
        methods[m] = methods.get(m, 0) + 1
    result["detection_methods"] = methods

    return result


def apply_filter_to_position(frames: list[dict], window_size: int = 7, ema_alpha: float = 0.4) -> list[dict]:
    """Run PoseHistory filter on frames within one position."""
    from trocar_pose_filter import PoseHistory

    history = PoseHistory(window_size=window_size, ema_alpha=ema_alpha)
    filtered_results = []
    for frame in frames:
        if frame.get("detection_status") != "ok":
            filtered_results.append(None)
            continue
        # Build a pose report dict that PoseHistory expects
        pose_report = {
            "pose_camera_trocar_mm_deg": [
                frame.get("camera_x_mm", 0),
                frame.get("camera_y_mm", 0),
                frame.get("camera_z_mm", 0),
                0, 0, 0,  # Euler angles not in CSV
            ],
            "trocar_axis_camera": [0, 0, 1],  # Approximation
            "metrics": {"mean_reprojection_error_px": frame.get("reprojection_error_px", 0)},
            "T_camera_trocar": None,
        }
        result = history.update(pose_report)
        filtered_results.append(result)
    return filtered_results


def compute_dataset_summary(position_stats: list[dict]) -> dict:
    """Aggregate statistics across all positions."""
    total_frames = sum(p["total_frames"] for p in position_stats)
    total_detected = sum(p["detected_frames"] for p in position_stats)
    total_outliers = sum(p.get("depth_outlier_count", 0) or 0 for p in position_stats)

    depth_stds = [p["depth_std_mm"] for p in position_stats if p.get("depth_std_mm") is not None]
    lateral_stds = [p["lateral_std_mm"] for p in position_stats if p.get("lateral_std_mm") is not None]
    reproj_means = [p["reproj_mean_px"] for p in position_stats if p.get("reproj_mean_px") is not None]

    # Detection method totals
    method_totals: dict[str, int] = {}
    for p in position_stats:
        for m, c in p.get("detection_methods", {}).items():
            method_totals[m] = method_totals.get(m, 0) + c

    return {
        "total_positions": len(position_stats),
        "total_frames": total_frames,
        "total_detected": total_detected,
        "overall_detection_rate": total_detected / max(total_frames, 1),
        "total_depth_outliers": total_outliers,
        "mean_depth_std_mm": float(np.mean(depth_stds)) if depth_stds else None,
        "max_depth_std_mm": float(np.max(depth_stds)) if depth_stds else None,
        "mean_lateral_std_mm": float(np.mean(lateral_stds)) if lateral_stds else None,
        "mean_reproj_px": float(np.mean(reproj_means)) if reproj_means else None,
        "detection_method_totals": method_totals,
    }


def compare_filtered_unfiltered(
    position_stats: list[dict],
    filtered_stats: list[dict],
) -> dict:
    """Compare unfiltered vs filtered metrics."""
    comparison = []
    for u, f in zip(position_stats, filtered_stats):
        row = {
            "position": u.get("name", "?"),
            "dZ_mm": u.get("distance_offset_mm"),
        }
        row["unfiltered_depth_std_mm"] = u.get("depth_std_mm")
        row["filtered_depth_std_mm"] = f.get("depth_std_mm")
        if u.get("depth_std_mm") and f.get("depth_std_mm") and u["depth_std_mm"] > 0:
            row["depth_std_reduction_pct"] = (1.0 - f["depth_std_mm"] / u["depth_std_mm"]) * 100.0
        else:
            row["depth_std_reduction_pct"] = None
        comparison.append(row)
    return comparison


def generate_plots(
    output_dir: Path,
    dataset_csv_rows: list[dict],
    position_stats: list[dict],
    comparison: list[dict] | None,
) -> list[str]:
    """Generate analysis plots using matplotlib."""
    plot_paths = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots.")
        return plot_paths

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Depth vs distance offset
    if position_stats:
        offsets = [p.get("distance_offset_mm", 0) for p in position_stats]
        depth_means = [p.get("depth_mean_mm", 0) for p in position_stats]
        depth_stds = [p.get("depth_std_mm", 0) for p in position_stats]
        depth_stds = [s if s else 0 for s in depth_stds]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(offsets, depth_means, yerr=depth_stds, fmt="o-", capsize=3, label="Measured depth")
        ax.set_xlabel("Distance offset from start (mm)")
        ax.set_ylabel("Measured depth (mm)")
        ax.set_title("Depth Estimate vs Robot Distance Offset")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = plots_dir / "depth_vs_distance.png"
        fig.savefig(str(path), dpi=150)
        plt.close(fig)
        plot_paths.append(str(path))

    # 2. Jitter by distance (depth std)
    if position_stats:
        offsets = [p.get("distance_offset_mm", 0) for p in position_stats]
        jitters = [p.get("depth_std_mm", 0) for p in position_stats]
        jitters = [j if j else 0 for j in jitters]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(range(len(offsets)), jitters, tick_label=[f"{o:+.0f}" for o in offsets])
        ax.set_xlabel("Distance offset (mm)")
        ax.set_ylabel("Depth jitter std (mm)")
        ax.set_title("Per-Position Depth Jitter")
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        path = plots_dir / "jitter_by_distance.png"
        fig.savefig(str(path), dpi=150)
        plt.close(fig)
        plot_paths.append(str(path))

    # 3. Detection success rate per position
    if position_stats:
        names = [p.get("name", f"pos_{i}") for i, p in enumerate(position_stats)]
        rates = [p.get("detection_success_rate", 0) for p in position_stats]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(len(names)), rates)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Detection success rate")
        ax.set_title("Detection Success Rate per Position")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        path = plots_dir / "detection_success_rate.png"
        fig.savefig(str(path), dpi=150)
        plt.close(fig)
        plot_paths.append(str(path))

    # 4. Filter comparison (unfiltered vs filtered jitter)
    if comparison:
        labels = [c.get("position", "?") for c in comparison]
        unfiltered = [c.get("unfiltered_depth_std_mm", 0) or 0 for c in comparison]
        filtered = [c.get("filtered_depth_std_mm", 0) or 0 for c in comparison]

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w / 2, unfiltered, w, label="Unfiltered", alpha=0.7)
        ax.bar(x + w / 2, filtered, w, label="Filtered", alpha=0.7)
        ax.set_ylabel("Depth std (mm)")
        ax.set_title("Filter Effect on Depth Jitter")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        path = plots_dir / "filter_comparison.png"
        fig.savefig(str(path), dpi=150)
        plt.close(fig)
        plot_paths.append(str(path))

    # 5. Outlier histogram: all depth deviations from position median
    all_deviations = []
    for p in position_stats:
        if p.get("depth_median_mm") is not None:
            # Re-compute from frames
            frames_at_pos = [
                f for f in dataset_csv_rows
                if f.get("position_name") == p.get("name")
                and f.get("camera_z_mm") is not None
            ]
            if frames_at_pos:
                depths_arr = np.array([f["camera_z_mm"] for f in frames_at_pos])
                all_deviations.extend((depths_arr - np.median(depths_arr)).tolist())

    if all_deviations:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(all_deviations, bins=30, edgecolor="black", alpha=0.7)
        ax.axvline(0, color="red", linestyle="--", label="Zero deviation")
        ax.set_xlabel("Depth deviation from position median (mm)")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Depth Deviations")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = plots_dir / "depth_deviation_histogram.png"
        fig.savefig(str(path), dpi=150)
        plt.close(fig)
        plot_paths.append(str(path))

    # 6. Reprojection error histogram
    all_reproj = [f.get("reprojection_error_px") for f in dataset_csv_rows if f.get("reprojection_error_px") is not None]
    if all_reproj:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(all_reproj, bins=30, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Reprojection error (px)")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Reprojection Errors")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = plots_dir / "reprojection_error_histogram.png"
        fig.savefig(str(path), dpi=150)
        plt.close(fig)
        plot_paths.append(str(path))

    return plot_paths


def write_analysis_markdown(
    path: Path,
    dataset_summary: dict,
    position_stats: list[dict],
    comparison: list[dict] | None,
    plot_paths: list[str],
) -> None:
    lines = [
        "# Pose Estimation Robustness Analysis Report",
        "",
        f"Generated: `{datetime.now().isoformat()}`",
        "",
        "## Dataset Overview",
        "",
        f"- Total positions: `{dataset_summary['total_positions']}`",
        f"- Total frames: `{dataset_summary['total_frames']}`",
        f"- Total detected: `{dataset_summary['total_detected']}`",
        f"- Overall detection rate: `{dataset_summary['overall_detection_rate']:.2%}`",
        f"- Total depth outliers (>3σ): `{dataset_summary.get('total_depth_outliers', 0)}`",
        "",
        "## Aggregate Statistics",
        "",
    ]

    if dataset_summary.get("mean_depth_std_mm") is not None:
        lines.append(f"- Mean depth jitter (std): `{dataset_summary['mean_depth_std_mm']:.4f} mm`")
        lines.append(f"- Max depth jitter (std): `{dataset_summary['max_depth_std_mm']:.4f} mm`")
    if dataset_summary.get("mean_lateral_std_mm") is not None:
        lines.append(f"- Mean lateral jitter (std): `{dataset_summary['mean_lateral_std_mm']:.4f} mm`")
    if dataset_summary.get("mean_reproj_px") is not None:
        lines.append(f"- Mean reprojection error: `{dataset_summary['mean_reproj_px']:.4f} px`")

    if dataset_summary.get("detection_method_totals"):
        lines += ["", "## Detection Method Breakdown", ""]
        for method, count in dataset_summary["detection_method_totals"].items():
            lines.append(f"- `{method}`: {count} frames")

    lines += ["", "## Per-Position Statistics", ""]
    lines.append(
        "| # | position | dZ mm | detected | depth_mean mm | depth_std mm | jitter_rms mm | lateral_std mm | reproj_mean px | outliers |"
    )
    lines.append(
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for i, p in enumerate(position_stats):
        lines.append(
            f"| {i + 1} | `{p.get('name', '?')}` | "
            f"{p.get('distance_offset_mm', 0):+.1f} | "
            f"{p['detected_frames']}/{p['total_frames']} | "
            f"{p.get('depth_mean_mm', 'N/A'):.3f} | "
            f"{p.get('depth_std_mm', 'N/A'):.4f} | "
            f"{p.get('depth_jitter_rms_mm', 'N/A'):.4f} | "
            f"{p.get('lateral_std_mm', 'N/A'):.4f} | "
            f"{p.get('reproj_mean_px', 'N/A'):.4f} | "
            f"{p.get('depth_outlier_count', 0)} |"
        )

    if comparison:
        lines += ["", "## Filter Comparison", ""]
        lines.append(
            "| position | dZ mm | unfiltered_std mm | filtered_std mm | reduction % |"
        )
        lines.append(
            "|---|---:|---:|---:|---:|"
        )
        for c in comparison:
            red = f"{c['depth_std_reduction_pct']:.1f}%" if c.get("depth_std_reduction_pct") is not None else "N/A"
            lines.append(
                f"| `{c['position']}` | {c.get('dZ_mm', 0):+.1f} | "
                f"{c.get('unfiltered_depth_std_mm', 'N/A'):.4f} | "
                f"{c.get('filtered_depth_std_mm', 'N/A'):.4f} | {red} |"
            )

    if plot_paths:
        lines += ["", "## Generated Plots", ""]
        for p in plot_paths:
            lines.append(f"- ![]({Path(p).name})")

    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze systematic pose dataset.")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--apply-filter", action="store_true", help="Also run filter for comparison")
    parser.add_argument("--window-size", type=int, default=7)
    parser.add_argument("--ema-alpha", type=float, default=0.4)
    parser.add_argument("--target-distance-mm", type=float, default=18.0)
    parser.add_argument("--plots", action="store_true", default=True)
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    # Find dataset
    if args.dataset_dir:
        dataset_dir = args.dataset_dir
    else:
        dataset_dir = load_latest_dataset(DEFAULT_DATASET)

    print(f"Loading dataset from: {dataset_dir}")

    # Load CSV
    csv_path = dataset_dir / "dataset_summary.csv"
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        print("Run collect_systematic_pose_dataset.py first.")
        return

    rows = []
    with csv_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric strings
            for key in ["distance_offset_mm", "lateral_x_mm", "lateral_y_mm",
                        "frame_index", "camera_x_mm", "camera_y_mm", "camera_z_mm",
                        "reprojection_error_px"]:
                if row.get(key) and row[key] not in ("None", ""):
                    try:
                        row[key] = float(row[key])
                    except ValueError:
                        row[key] = None
                else:
                    row[key] = None
            rows.append(row)

    print(f"Loaded {len(rows)} frame records.")

    # Group by position
    positions: dict[str, list[dict]] = {}
    for row in rows:
        name = row.get("position_name", "unknown")
        positions.setdefault(name, []).append(row)

    # Compute per-position stats
    position_stats = []
    for name in sorted(positions.keys()):
        frames = positions[name]
        stats = compute_position_statistics(frames)
        stats["name"] = name
        if frames:
            stats["distance_offset_mm"] = frames[0].get("distance_offset_mm", 0)
            stats["lateral_offset_mm"] = [frames[0].get("lateral_x_mm", 0), frames[0].get("lateral_y_mm", 0)]
        position_stats.append(stats)

    # Dataset summary
    dataset_summary = compute_dataset_summary(position_stats)
    print(f"Detection rate: {dataset_summary['overall_detection_rate']:.2%}")
    print(f"Mean depth std: {dataset_summary.get('mean_depth_std_mm', 'N/A')} mm")

    # Filter comparison
    comparison = None
    filtered_stats = []
    if args.apply_filter:
        print("Applying multi-frame filter...")
        for name in sorted(positions.keys()):
            frames = positions[name]
            filtered = apply_filter_to_position(
                frames, window_size=args.window_size, ema_alpha=args.ema_alpha,
            )
            # Extract filtered depths for stats
            filtered_depths = []
            for r in filtered:
                if r and r.get("status") == "ok" and r.get("filtered_pose_camera_trocar_mm_deg"):
                    filtered_depths.append(r["filtered_pose_camera_trocar_mm_deg"][2])
            f_stats = {
                "name": name,
                "depth_std_mm": float(np.std(filtered_depths)) if len(filtered_depths) > 1 else None,
            }
            filtered_stats.append(f_stats)
        comparison = compare_filtered_unfiltered(position_stats, filtered_stats)
        print("Filter comparison computed.")

    # Generate plots
    plot_paths = []
    if args.plots and not args.no_plots:
        print("Generating plots...")
        plot_paths = generate_plots(args.output_dir, rows, position_stats, comparison)
        print(f"Generated {len(plot_paths)} plots.")

    # Write outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Per-position CSV
    per_pos_path = args.output_dir / "per_position_stats.csv"
    with per_pos_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "name", "distance_offset_mm", "total_frames", "detected_frames",
            "detection_success_rate", "depth_mean_mm", "depth_std_mm",
            "depth_median_mm", "depth_range_mm", "depth_outlier_count",
            "depth_jitter_rms_mm", "lateral_mean_mm", "lateral_std_mm",
            "reproj_mean_px", "reproj_std_px",
        ])
        writer.writeheader()
        for p in position_stats:
            row = {k: p.get(k) for k in [
                "name", "distance_offset_mm", "total_frames", "detected_frames",
                "detection_success_rate", "depth_mean_mm", "depth_std_mm",
                "depth_median_mm", "depth_range_mm", "depth_outlier_count",
                "depth_jitter_rms_mm", "lateral_mean_mm", "lateral_std_mm",
                "reproj_mean_px", "reproj_std_px",
            ]}
            writer.writerow(row)

    # Analysis report JSON
    analysis_report = {
        "timestamp": datetime.now().isoformat(),
        "dataset_dir": str(dataset_dir),
        "dataset_summary": dataset_summary,
        "position_stats": position_stats,
        "comparison": comparison,
        "plot_paths": plot_paths,
    }
    write_json(args.output_dir / "analysis_report.json", analysis_report)

    # Markdown report
    write_analysis_markdown(
        args.output_dir / "analysis_report.md",
        dataset_summary, position_stats, comparison, plot_paths,
    )

    print(f"\nAnalysis output: {args.output_dir}")
    print(f"  - per_position_stats.csv")
    print(f"  - analysis_report.json")
    print(f"  - analysis_report.md")
    if plot_paths:
        print(f"  - plots/ ({len(plot_paths)} figures)")


if __name__ == "__main__":
    main()
