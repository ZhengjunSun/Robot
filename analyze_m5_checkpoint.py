from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_REPORT = (
    ROOT / "output" / "m5_frozen_500" / "m5_frozen_report.json"
)


def classify_episode(episode: dict[str, Any]) -> str:
    if episode["full_flow_success"]:
        return "full_flow_success"
    if not episode["target_acquired"]:
        return "target_acquisition_failure"
    if not episode["fine_success"]:
        return "fine_alignment_failure"
    if (
        episode["insertion_stop_reason"]
        == "visual_clearance_below_threshold"
    ):
        return "unsafe_alignment_rejected_before_insertion"
    return "post_alignment_insertion_failure"


def representative_manifest(
    episodes: list[dict[str, Any]],
    *,
    maximum_per_class_per_stratum: int,
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        key = (
            episode["randomization_stratum"],
            classify_episode(episode),
        )
        buckets[key].append(episode)
    selected: list[dict[str, Any]] = []
    for key in sorted(buckets):
        candidates = sorted(
            buckets[key],
            key=lambda item: (
                item["episode_seed"],
                item["episode"],
            ),
        )
        for episode in candidates[:maximum_per_class_per_stratum]:
            selected.append(
                {
                    "episode": episode["episode"],
                    "episode_seed": episode["episode_seed"],
                    "randomization_stratum": (
                        episode["randomization_stratum"]
                    ),
                    "failure_class": classify_episode(episode),
                    "insertion_stop_reason": (
                        episode["insertion_stop_reason"]
                    ),
                    "randomization": {
                        key: episode[key]
                        for key in (
                            "initial_joint_delta_deg",
                            "trocar_translation_mm",
                            "trocar_rotation_deg_xyz",
                            "camera_fovy_scale",
                            "principal_point_shift_px",
                            "rgb_gain",
                            "rgb_noise_std",
                            "blur_kernel",
                            "trocar_rgb_scale",
                            "sclera_rgb_scale",
                            "light_intensity_scale",
                            "occlusion_probability",
                            "occlusion_fraction",
                        )
                    },
                }
            )
    return selected


def analyze(report: dict[str, Any]) -> dict[str, Any]:
    episodes = report["episode_results"]
    classes = Counter(classify_episode(episode) for episode in episodes)
    aligned = [episode for episode in episodes if episode["fine_success"]]
    unsafe_truth_axis = [
        episode
        for episode in aligned
        if episode["final_evaluation_only_pose_errors"]["axis_error_deg"]
        > 5.0
    ]
    by_stratum: dict[str, dict[str, Any]] = {}
    for stratum in sorted(
        {episode["randomization_stratum"] for episode in episodes}
    ):
        subset = [
            episode
            for episode in episodes
            if episode["randomization_stratum"] == stratum
        ]
        by_stratum[stratum] = {
            "episodes": len(subset),
            "class_counts": dict(
                Counter(classify_episode(episode) for episode in subset)
            ),
            "target_acquisition_rate": sum(
                episode["target_acquired"] for episode in subset
            )
            / len(subset),
            "fine_success_rate": sum(
                episode["fine_success"] for episode in subset
            )
            / len(subset),
            "full_flow_success_rate": sum(
                episode["full_flow_success"] for episode in subset
            )
            / len(subset),
        }
    return {
        "timestamp": datetime.now().astimezone().isoformat(),
        "source_report": {
            "timestamp": report["timestamp"],
            "protocol_sha256": report["protocol_sha256"],
            "git_commit": report["git_commit"],
        },
        "episodes_analyzed": len(episodes),
        "failure_classes": dict(classes),
        "aligned_episode_count": len(aligned),
        "aligned_truth_axis_above_5_deg_count": len(unsafe_truth_axis),
        "aligned_truth_axis_above_5_deg_rate": (
            len(unsafe_truth_axis) / len(aligned) if aligned else None
        ),
        "by_randomization_stratum": by_stratum,
        "m5_1_decision": {
            "triggered": bool(
                report["summary"]["target_acquisition_rate"] >= 0.90
                and (
                    report["summary"]["fine_success_rate"] < 0.80
                    or report["summary"]["full_flow_success_rate"] < 0.80
                )
            ),
            "reason": (
                "Acquisition is adequate while near-field alignment or "
                "insertion authorization is systematically failing."
            ),
            "next_experiment": (
                "Paired replay of identical seeds with temporal near-field "
                "validation, active pose/depth repair, and fail-closed "
                "hold/reobserve behavior."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose a resumable M5 checkpoint without changing it."
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--maximum-per-class-per-stratum",
        type=int,
        default=2,
    )
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    output_dir = args.output_dir or args.report.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnosis = analyze(report)
    manifest = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "source_protocol_sha256": report["protocol_sha256"],
        "selection_contract": (
            "Deterministic lowest-seed samples, capped per failure class "
            "and randomization stratum."
        ),
        "episodes": representative_manifest(
            report["episode_results"],
            maximum_per_class_per_stratum=(
                args.maximum_per_class_per_stratum
            ),
        ),
    }
    diagnosis_path = output_dir / "m5_checkpoint_diagnosis.json"
    manifest_path = output_dir / "m5_1_replay_manifest.json"
    diagnosis_path.write_text(
        json.dumps(diagnosis, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(diagnosis, indent=2, ensure_ascii=False))
    print(f"diagnosis={diagnosis_path}")
    print(f"replay_manifest={manifest_path}")


if __name__ == "__main__":
    main()
