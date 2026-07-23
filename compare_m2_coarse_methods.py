from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_TRADITIONAL = (
    ROOT
    / "output"
    / "m2_matched_traditional_nih_hra_10"
    / "m1_randomized_batch_report.json"
)
DEFAULT_YOLO = (
    ROOT
    / "output"
    / "m2_yolo_randomized_batch_nih_hra_10"
    / "m1_randomized_batch_report.json"
)
DEFAULT_OUTPUT = ROOT / "output" / "m2_coarse_comparison_nih_hra"


def load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(records: list[dict], key: str) -> float:
    return float(np.mean([float(record[key]) for record in records]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare M2 traditional and YOLO coarse loops on matched episodes."
    )
    parser.add_argument("--traditional", type=Path, default=DEFAULT_TRADITIONAL)
    parser.add_argument("--yolo", type=Path, default=DEFAULT_YOLO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    reports = {
        "traditional": load_report(args.traditional.resolve()),
        "yolo": load_report(args.yolo.resolve()),
    }
    reference = reports["traditional"]
    for name, report in reports.items():
        if report["seed"] != reference["seed"]:
            raise ValueError(f"{name} seed does not match.")
        if report["episode_count"] != reference["episode_count"]:
            raise ValueError(f"{name} episode count does not match.")
        initials = [
            (record["initial_camera_x_mm"], record["initial_camera_y_mm"])
            for record in report["episodes"]
        ]
        reference_initials = [
            (record["initial_camera_x_mm"], record["initial_camera_y_mm"])
            for record in reference["episodes"]
        ]
        if initials != reference_initials:
            raise ValueError(f"{name} initial conditions do not match.")

    comparison: dict[str, dict] = {}
    for name, report in reports.items():
        successful = [record for record in report["episodes"] if record["success"]]
        comparison[name] = {
            "episodes": report["episode_count"],
            "success_rate": report["success_rate"],
            "mean_steps_success": mean(successful, "steps"),
            "mean_final_center_error_px_success": mean(
                successful, "final_center_error_px"
            ),
            "mean_final_lateral_error_mm_success": mean(
                successful, "final_lateral_error_mm"
            ),
            "mean_path_length_camera_mm_success": mean(
                successful, "path_length_camera_mm"
            ),
            "mean_reversal_count_success": mean(
                successful, "reversal_count"
            ),
        }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "timestamp": datetime.now().isoformat(),
        "evidence_level": (
            "Matched-seed synthetic NIH/HRA MuJoCo RGB-driven coarse comparison"
        ),
        "privileged_truth_used_for_control": False,
        "seed": reference["seed"],
        "comparison": comparison,
        "evidence_boundary": (
            "This small matched run checks online integration. It is not the "
            "frozen M5 benchmark and does not establish real-image generalization."
        ),
    }
    json_path = output_dir / "m2_coarse_comparison.json"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path = output_dir / "m2_coarse_comparison.md"
    markdown_path.write_text(
        "\n".join(
            [
                "# M2 NIH/HRA 粗对准同条件对比",
                "",
                "| 方法 | 成功率 | 平均步数 | 最终像素误差 | 最终横向误差 (mm) | 路径 (mm) | 折返 |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                *[
                    (
                        f"| {name} | {values['success_rate']:.1%} | "
                        f"{values['mean_steps_success']:.3f} | "
                        f"{values['mean_final_center_error_px_success']:.3f} | "
                        f"{values['mean_final_lateral_error_mm_success']:.3f} | "
                        f"{values['mean_path_length_camera_mm_success']:.3f} | "
                        f"{values['mean_reversal_count_success']:.3f} |"
                    )
                    for name, values in comparison.items()
                ],
                "",
                result["evidence_boundary"],
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")


if __name__ == "__main__":
    main()
