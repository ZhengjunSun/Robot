from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "output" / "m5_2_gate_evaluation"
NOMINAL_GRID = (
    (0.0, -6.0),
    (0.0, 0.0),
    (0.0, 6.0),
    (6.0, -6.0),
    (6.0, 0.0),
    (6.0, 6.0),
)
FAILURE_SEEDS = (
    (47, 202371628),
    (7, 616336834),
    (22, 455730289),
    (139, 1160727935),
    (79, 6324905),
)


def run_case(
    output_dir: Path,
    *,
    tilt_x_deg: float,
    tilt_y_deg: float,
    episode: int | None = None,
    episode_seed: int | None = None,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(ROOT / "run_m5_2_multiview_smoke.py"),
        "--tilt-deg",
        str(tilt_x_deg),
        "--tilt-y-deg",
        str(tilt_y_deg),
        "--output-dir",
        str(output_dir),
    ]
    if episode is not None and episode_seed is not None:
        command.extend(
            [
                "--episode",
                str(episode),
                "--episode-seed",
                str(episode_seed),
            ]
        )
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        return {
            "completed": False,
            "returncode": completed.returncode,
            "stderr_tail": completed.stderr[-2000:],
            "tilt_xy_deg": [tilt_x_deg, tilt_y_deg],
            "episode": episode,
            "episode_seed": episode_seed,
        }
    report_path = output_dir / "m5_2_multiview_smoke_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    truth_axis = float(
        report["offline_truth_evaluation"]["tool_target_axis_error_deg"]
    )
    authorized = bool(
        report["state_machine"]["insertion_handoff_ready"]
    )
    should_authorize = truth_axis <= 2.0
    return {
        "completed": True,
        "tilt_xy_deg": [tilt_x_deg, tilt_y_deg],
        "episode": episode,
        "episode_seed": episode_seed,
        "randomization_stratum": (
            None
            if report["domain_sample"] is None
            else report["domain_sample"]["randomization_stratum"]
        ),
        "truth_axis_error_deg": truth_axis,
        "estimated_axis_error_deg": float(
            report["active_multiview_observation"]["axis_error_deg"]
        ),
        "normal_estimation_error_deg": float(
            report["offline_truth_evaluation"]["normal_error_deg"]
        ),
        "center_estimation_error_mm": float(
            report["offline_truth_evaluation"]["center_error_mm"]
        ),
        "covariance_condition": float(
            report["active_multiview_observation"][
                "covariance_condition"
            ]
        ),
        "state_phase": report["state_machine"]["phase"],
        "reasons": report["state_machine"]["reasons"],
        "authorized": authorized,
        "should_authorize": should_authorize,
        "false_authorization": bool(authorized and not should_authorize),
        "false_rejection": bool(not authorized and should_authorize),
        "report": str(report_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="M5.2 active-multiview gate grid and failure-seed test."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    grid_rows = []
    for tilt_x, tilt_y in NOMINAL_GRID:
        case_dir = (
            output_dir / f"grid_x{tilt_x:+.1f}_y{tilt_y:+.1f}"
        )
        grid_rows.append(
            run_case(
                case_dir,
                tilt_x_deg=tilt_x,
                tilt_y_deg=tilt_y,
            )
        )

    failure_rows = []
    for episode, seed in FAILURE_SEEDS:
        case_dir = output_dir / f"failure_episode_{episode}_seed_{seed}"
        failure_rows.append(
            run_case(
                case_dir,
                tilt_x_deg=6.0,
                tilt_y_deg=0.0,
                episode=episode,
                episode_seed=seed,
            )
        )

    rows = grid_rows + failure_rows
    completed = [row for row in rows if row["completed"]]
    report = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "status": "M5.2_active_multiview_gate_evaluation",
        "scope": (
            "near-field active multiview perception and authorization only; "
            "not a full-flow M5 replay"
        ),
        "privileged_truth_used_for_control": False,
        "privileged_truth_use": (
            "offline expected authorization and pose-error scoring only"
        ),
        "nominal_grid": grid_rows,
        "representative_failure_seeds": failure_rows,
        "summary": {
            "requested_cases": len(rows),
            "completed_cases": len(completed),
            "false_authorizations": sum(
                int(row["false_authorization"]) for row in completed
            ),
            "false_rejections": sum(
                int(row["false_rejection"]) for row in completed
            ),
            "authorized_cases": sum(
                int(row["authorized"]) for row in completed
            ),
            "maximum_normal_estimation_error_deg": max(
                (
                    row["normal_estimation_error_deg"]
                    for row in completed
                ),
                default=None,
            ),
            "maximum_center_estimation_error_mm": max(
                (
                    row["center_estimation_error_mm"]
                    for row in completed
                ),
                default=None,
            ),
        },
    }
    report_path = output_dir / "m5_2_gate_evaluation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], indent=2))
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
