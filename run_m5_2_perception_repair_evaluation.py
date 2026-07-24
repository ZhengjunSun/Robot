from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from run_m5_2_gate_evaluation import (
    FAILURE_SEEDS,
    NOMINAL_GRID,
    run_case,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "output" / "m5_2_perception_repair_paired"
BASELINE_ARGS = (
    "--intrinsics-mode",
    "nominal",
    "--ring-mode",
    "outer",
    "--normal-init",
    "single",
    "--view-offset-xy-mm",
    "-1.5",
    "0.0",
    "--view-offset-xy-mm",
    "0.0",
    "1.5",
    "--view-offset-xy-mm",
    "1.5",
    "-1.5",
)
CANDIDATE_ARGS = (
    "--intrinsics-mode",
    "calibrated",
    "--ring-mode",
    "joint",
    "--normal-init",
    "multistart",
    "--view-offset-xy-mm",
    "-3.0",
    "0.0",
    "--view-offset-xy-mm",
    "0.0",
    "3.0",
    "--view-offset-xy-mm",
    "3.0",
    "-3.0",
)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row["completed"]]
    normal = np.asarray(
        [row["normal_estimation_error_deg"] for row in completed]
    )
    center = np.asarray(
        [row["center_estimation_error_mm"] for row in completed]
    )
    return {
        "completed": len(completed),
        "false_authorizations": sum(
            int(row["false_authorization"]) for row in completed
        ),
        "false_rejections": sum(
            int(row["false_rejection"]) for row in completed
        ),
        "normal_error_median_deg": float(np.median(normal)),
        "normal_error_p95_deg": float(np.percentile(normal, 95)),
        "normal_error_max_deg": float(np.max(normal)),
        "center_error_median_mm": float(np.median(center)),
        "center_error_p95_mm": float(np.percentile(center, 95)),
        "center_error_max_mm": float(np.max(center)),
    }


def evaluate_variant(
    root: Path,
    *,
    extra_args: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows = []
    for tilt_x, tilt_y in NOMINAL_GRID:
        rows.append(
            run_case(
                root / f"grid_x{tilt_x:+.1f}_y{tilt_y:+.1f}",
                tilt_x_deg=tilt_x,
                tilt_y_deg=tilt_y,
                extra_args=extra_args,
            )
        )
    for episode, seed in FAILURE_SEEDS:
        rows.append(
            run_case(
                root / f"failure_episode_{episode}_seed_{seed}",
                tilt_x_deg=6.0,
                tilt_y_deg=0.0,
                episode=episode,
                episode_seed=seed,
                extra_args=extra_args,
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired M5.2 calibration/ring/multistart repair test."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = evaluate_variant(
        output_dir / "baseline",
        extra_args=BASELINE_ARGS,
    )
    candidate = evaluate_variant(
        output_dir / "candidate",
        extra_args=CANDIDATE_ARGS,
    )
    baseline_summary = summarize(baseline)
    candidate_summary = summarize(candidate)
    report = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "status": "M5.2_perception_repair_paired_evaluation",
        "case_count_per_variant": len(baseline),
        "privileged_truth_used_for_control": False,
        "baseline_configuration": list(BASELINE_ARGS),
        "candidate_configuration": list(CANDIDATE_ARGS),
        "baseline_summary": baseline_summary,
        "candidate_summary": candidate_summary,
        "improvement": {
            "normal_p95_reduction_deg": (
                baseline_summary["normal_error_p95_deg"]
                - candidate_summary["normal_error_p95_deg"]
            ),
            "center_p95_reduction_mm": (
                baseline_summary["center_error_p95_mm"]
                - candidate_summary["center_error_p95_mm"]
            ),
        },
        "baseline_cases": baseline,
        "candidate_cases": candidate,
    }
    report_path = output_dir / "m5_2_perception_repair_paired_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "baseline": baseline_summary,
                "candidate": candidate_summary,
                "improvement": report["improvement"],
            },
            indent=2,
        )
    )
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
