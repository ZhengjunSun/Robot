from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from real_3d_alignment.config import DEFAULT_CONFIG, load_config, resolve_project_path
from real_3d_alignment.step_evaluation import (
    evaluate_step,
    resolve_bridge_report,
    resolve_run_report,
    write_step_evaluation_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare pre/post real 3D alignment dry-run reports after one robot step."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pre-run-report", type=Path, default=None)
    parser.add_argument("--pre-run-dir", type=Path, default=None)
    parser.add_argument("--post-run-report", type=Path, default=None)
    parser.add_argument("--post-run-dir", type=Path, default=None)
    parser.add_argument("--bridge-report", type=Path, default=None)
    parser.add_argument("--bridge-dir", type=Path, default=None)
    parser.add_argument("--target-distance-mm", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    target_distance_mm = (
        float(args.target_distance_mm)
        if args.target_distance_mm is not None
        else float(cfg["control"]["target_distance_mm"])
    )
    output_root = resolve_project_path(cfg["paths"]["output_root"])
    output_dir = args.output_dir or (output_root / "step_evaluation")

    pre_run_report = resolve_run_report(args.pre_run_report, args.pre_run_dir)
    post_run_report = resolve_run_report(args.post_run_report, args.post_run_dir)
    bridge_report = resolve_bridge_report(args.bridge_report, args.bridge_dir)

    report = evaluate_step(
        pre_run_report=pre_run_report,
        post_run_report=post_run_report,
        target_distance_mm=target_distance_mm,
        bridge_report=bridge_report,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = output_dir / f"real_3d_step_eval_{timestamp}"
    report_path = report_dir / "real_3d_step_evaluation.md"
    write_step_evaluation_markdown(report_path, report)

    print("real_3d_step evaluation:", report_dir)
    print("improved:", report["improved"])
    print("weighted_delta_mm:", report["delta"]["weighted_3d_error_mm"])
    if report.get("prediction"):
        print("predicted_delta_mm:", report["prediction"].get("predicted_delta_weighted_3d_error_mm"))
        print("prediction_error_mm:", report["prediction"].get("prediction_error_weighted_delta_mm"))


if __name__ == "__main__":
    main()
