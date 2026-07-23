#!/usr/bin/env python3
"""Analyze alignment experiment batch results: per-mode stats, plots, thesis tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "outputs" / "alignment_experiments"
DEFAULT_OUTPUT = ROOT / "outputs" / "alignment_analysis"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_latest_batch(base_dir: Path) -> Path:
    dirs = sorted(base_dir.glob("batch_*"))
    if not dirs:
        raise FileNotFoundError(f"No batch directories in {base_dir}")
    return max(dirs, key=lambda p: p.stat().st_mtime)


def load_batch_experiments(batch_dir: Path) -> list[dict]:
    summary_path = batch_dir / "batch_summary.json"
    if summary_path.exists():
        data = load_json(summary_path)
        experiments = data.get("experiments", [])
        if experiments:
            return experiments
    experiment_paths = sorted(batch_dir.glob("experiment_*.json"))
    return [load_json(path) for path in experiment_paths]


def mode_summary(experiments: list[dict]) -> dict:
    total = len(experiments)
    converged = sum(1 for e in experiments if e["converged"])
    iters = [e["iterations"] for e in experiments]
    times = [e["total_time_s"] for e in experiments]
    finals_w = [e.get("final_errors", {}).get("weighted_3d_error_mm", float("inf")) for e in experiments]
    finals_w_clean = [v for v in finals_w if v < 100]
    finals_l = [e.get("final_errors", {}).get("lateral_error_mm", float("inf")) for e in experiments]
    finals_l_clean = [v for v in finals_l if v < 100]
    initials_w = [e.get("initial_errors", {}).get("weighted_3d_error_mm", float("inf")) for e in experiments]
    initials_w_clean = [v for v in initials_w if v < 100]
    improvements = [i - f for i, f in zip(initials_w_clean, finals_w_clean)]

    return {
        "total": total,
        "converged": converged,
        "success_rate": converged / max(total, 1),
        "mean_iterations": float(np.mean(iters)) if iters else None,
        "std_iterations": float(np.std(iters)) if len(iters) > 1 else None,
        "mean_time_s": float(np.mean(times)) if times else None,
        "mean_final_weighted_mm": float(np.mean(finals_w_clean)) if finals_w_clean else None,
        "std_final_weighted_mm": float(np.std(finals_w_clean)) if len(finals_w_clean) > 1 else None,
        "min_final_weighted_mm": float(np.min(finals_w_clean)) if finals_w_clean else None,
        "mean_final_lateral_mm": float(np.mean(finals_l_clean)) if finals_l_clean else None,
        "mean_improvement_mm": float(np.mean(improvements)) if improvements else None,
    }


def offset_analysis(experiments: list[dict]) -> list[dict]:
    rows = []
    for e in experiments:
        off_norm = math.sqrt(sum(v ** 2 for v in e["offset_mm"]))
        rows.append({
            "mode": e["mode"],
            "offset_norm_mm": off_norm,
            "offset_mm": e["offset_mm"],
            "converged": e["converged"],
            "iterations": e["iterations"],
            "initial_weighted": e.get("initial_errors", {}).get("weighted_3d_error_mm"),
            "final_weighted": e.get("final_errors", {}).get("weighted_3d_error_mm"),
        })
    return rows


def fmt_optional(value, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def generate_plots(output_dir: Path, experiments: list[dict]) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots.")
        return []

    plot_paths = []
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    modes = sorted(set(e["mode"] for e in experiments))
    colors = {"2d_center": "#e74c3c", "3d_one_step": "#3498db", "3d_pbvs": "#2ecc71", "3d_filtered": "#9b59b6"}
    mode_colors = [colors.get(m, "#888888") for m in modes]

    # 1. Convergence curves per mode (weighted error over iterations)
    fig, ax = plt.subplots(figsize=(10, 6))
    for mode in modes:
        mode_exps = [e for e in experiments if e["mode"] == mode and len(e.get("trajectory", [])) > 1]
        for exp in mode_exps:
            traj = exp["trajectory"]
            weighted = [t.get("weighted_3d_error_mm", 0) for t in traj if t.get("weighted_3d_error_mm") is not None]
            if len(weighted) > 1:
                ax.plot(weighted, alpha=0.4, color=colors.get(mode, "#888"), linewidth=1)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Weighted 3D error (mm)")
    ax.set_title("Convergence Trajectories by Mode")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    for mode in modes:
        ax.plot([], [], color=colors.get(mode, "#888"), linewidth=2, label=mode)
    ax.legend()
    fig.tight_layout()
    path = plots_dir / "convergence_trajectories.png"
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    plot_paths.append(str(path))

    # 2. Final error by mode (box plot)
    fig, ax = plt.subplots(figsize=(8, 5))
    data_by_mode = []
    labels = []
    for mode in modes:
        finals = [e.get("final_errors", {}).get("weighted_3d_error_mm", float("inf"))
                   for e in experiments if e["mode"] == mode]
        finals = [v for v in finals if v < 100]
        if finals:
            data_by_mode.append(finals)
            labels.append(mode)
    if data_by_mode:
        ax.boxplot(data_by_mode, labels=labels, patch_artist=True)
        ax.set_ylabel("Final weighted 3D error (mm)")
        ax.set_title("Final Alignment Error by Method")
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        path = plots_dir / "final_error_by_mode.png"
        fig.savefig(str(path), dpi=150)
        plt.close(fig)
        plot_paths.append(str(path))

    # 3. Iterations by mode
    fig, ax = plt.subplots(figsize=(8, 5))
    iter_data = []
    iter_labels = []
    for mode in modes:
        iters = [e["iterations"] for e in experiments if e["mode"] == mode]
        if iters:
            iter_data.append(iters)
            iter_labels.append(mode)
    if iter_data:
        ax.boxplot(iter_data, labels=iter_labels, patch_artist=True)
        ax.set_ylabel("Iterations to converge")
        ax.set_title("Iterations Needed by Method")
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        path = plots_dir / "iterations_by_mode.png"
        fig.savefig(str(path), dpi=150)
        plt.close(fig)
        plot_paths.append(str(path))

    # 4. Success rate by offset magnitude
    fig, ax = plt.subplots(figsize=(8, 5))
    offset_bins = [(0, 3), (3, 6), (6, 10)]
    for mode in modes:
        rates = []
        bin_labels = []
        for lo, hi in offset_bins:
            exps = [e for e in experiments if e["mode"] == mode
                    and lo <= math.sqrt(sum(v ** 2 for v in e["offset_mm"])) < hi]
            rate = sum(1 for e in exps if e["converged"]) / max(len(exps), 1) * 100
            rates.append(rate)
            bin_labels.append(f"{lo}-{hi}mm")
        ax.plot(range(len(bin_labels)), rates, "o-", label=mode, color=colors.get(mode, "#888"))
    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels)
    ax.set_ylabel("Convergence rate (%)")
    ax.set_xlabel("Initial offset magnitude")
    ax.set_ylim(0, 110)
    ax.set_title("Convergence Rate vs Initial Offset")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = plots_dir / "success_rate_by_offset.png"
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    plot_paths.append(str(path))

    # 5. Time per experiment by mode
    fig, ax = plt.subplots(figsize=(8, 5))
    time_data = []
    time_labels = []
    for mode in modes:
        times = [e["total_time_s"] for e in experiments if e["mode"] == mode]
        if times:
            time_data.append(times)
            time_labels.append(mode)
    if time_data:
        ax.boxplot(time_data, labels=time_labels, patch_artist=True)
        ax.set_ylabel("Time (seconds)")
        ax.set_title("Experiment Duration by Method")
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        path = plots_dir / "time_by_mode.png"
        fig.savefig(str(path), dpi=150)
        plt.close(fig)
        plot_paths.append(str(path))

    return plot_paths


def write_thesis_markdown(path: Path, summaries: dict[str, dict], experiments: list[dict], plot_paths: list[str]) -> None:
    lines = [
        "# Closed-Loop Alignment Experiment Analysis",
        "",
        f"Generated: `{datetime.now().isoformat()}`",
        "",
        f"Total experiments: `{sum(s['total'] for s in summaries.values())}`",
        "",
        "## Methods Compared",
        "",
        "| Method | Experiments | Converged | Success Rate | Mean Iterations | Mean Time (s) | Mean Final Error (mm) | Min Final Error (mm) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, s in summaries.items():
        lines.append(
            f"| `{mode}` | {s['total']} | {s['converged']} | "
            f"{s['success_rate']:.0%} | "
            f"{fmt_optional(s['mean_iterations'], 1)} | "
            f"{fmt_optional(s['mean_time_s'], 1)} | "
            f"{fmt_optional(s['mean_final_weighted_mm'])} | "
            f"{fmt_optional(s['min_final_weighted_mm'])} |"
        )

    lines += [
        "",
        "## Key Findings",
        "",
    ]
    # Auto-generate findings
    valid_accuracy = {
        mode: summary for mode, summary in summaries.items()
        if summary.get("mean_final_weighted_mm") is not None
    }
    best_mode = min(valid_accuracy.items(), key=lambda kv: kv[1].get("mean_final_weighted_mm", 999)) if valid_accuracy else None
    fastest_candidates = valid_accuracy if valid_accuracy else summaries
    fastest_mode = min(fastest_candidates.items(), key=lambda kv: kv[1].get("mean_iterations", 999))
    if best_mode:
        lines.append(f"- Best accuracy: `{best_mode[0]}` (mean {best_mode[1].get('mean_final_weighted_mm', 0):.3f} mm)")
    else:
        lines.append("- Best accuracy: N/A (no common 3D final error available)")
    lines.append(f"- Fastest convergence: `{fastest_mode[0]}` (mean {fastest_mode[1].get('mean_iterations', '?'):.1f} iterations)")

    if "3d_filtered" in summaries and "3d_pbvs" in summaries:
        s_f = summaries["3d_filtered"]
        s_p = summaries["3d_pbvs"]
        if s_f.get("mean_final_weighted_mm") and s_p.get("mean_final_weighted_mm"):
            improvement = (1 - s_f["mean_final_weighted_mm"] / s_p["mean_final_weighted_mm"]) * 100
            lines.append(f"- Filter improvement over PBVS: {improvement:.1f}% mean final error reduction")

    lines += ["", "## Detailed Per-Experiment Results", "",
        "| # | mode | offset mm | converged | iters | time(s) | initial_wt | final_wt | final_lat |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for e in experiments:
        iw = e.get("initial_errors", {}).get("weighted_3d_error_mm")
        fw = e.get("final_errors", {}).get("weighted_3d_error_mm")
        fl = e.get("final_errors", {}).get("lateral_error_mm")
        off = e["offset_mm"]
        lines.append(
            f"| {e['experiment_id']} | `{e['mode']}` | "
            f"[{off[0]:+.0f},{off[1]:+.0f},{off[2]:+.0f}] | "
            f"{'Yes' if e['converged'] else 'No'} | {e['iterations']} | "
            f"{e['total_time_s']} | "
            f"{fmt_optional(iw, 2)} | {fmt_optional(fw, 2)} | {fmt_optional(fl, 2)} |"
        )

    if plot_paths:
        lines += ["", "## Figures", ""]
        for p in plot_paths:
            lines.append(f"![]({Path(p).name})")

    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze alignment experiment batch results.")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    batch_dir = args.dataset_dir or load_latest_batch(DEFAULT_DATASET)
    print(f"Loading batch from: {batch_dir}")

    experiments = load_batch_experiments(batch_dir)
    if not experiments:
        print("No experiments found in batch directory.")
        return

    print(f"Loaded {len(experiments)} experiments.")

    summaries = {}
    modes = sorted(set(e["mode"] for e in experiments))
    for mode in modes:
        mode_exps = [e for e in experiments if e["mode"] == mode]
        summaries[mode] = mode_summary(mode_exps)

    # Print summary
    print("\nPer-Mode Summary:")
    for mode, s in summaries.items():
        print(f"  {mode:20s}: {s['converged']}/{s['total']} converged "
              f"({s['success_rate']:.0%}), mean_iters={s['mean_iterations']:.1f}, "
              f"mean_final={fmt_optional(s['mean_final_weighted_mm'])}mm")

    # Generate plots
    print("\nGenerating plots...")
    plot_paths = generate_plots(args.output_dir, experiments)

    # Save analysis
    analysis = {
        "timestamp": datetime.now().isoformat(),
        "batch_dir": str(batch_dir),
        "summaries": summaries,
        "experiment_count": len(experiments),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "analysis_report.json", analysis)
    write_thesis_markdown(args.output_dir / "thesis_report.md", summaries, experiments, plot_paths)

    print(f"\nAnalysis output: {args.output_dir}")
    print(f"  thesis_report.md")
    if plot_paths:
        print(f"  plots/ ({len(plot_paths)} figures)")


if __name__ == "__main__":
    main()
