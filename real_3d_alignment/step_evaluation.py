from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .config import PROJECT_ROOT, load_json, write_json
from .control_analysis import error_metrics


def resolve_existing_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    candidate = PROJECT_ROOT / path
    return candidate if candidate.exists() else path


def resolve_run_report(run_report: Path | None = None, run_dir: Path | None = None) -> Path:
    if run_report is not None and run_dir is not None:
        raise ValueError("Use either run_report or run_dir, not both.")
    if run_dir is not None:
        return resolve_existing_path(run_dir) / "run_report.json"
    if run_report is not None:
        return resolve_existing_path(run_report)
    raise ValueError("run_report or run_dir is required.")


def resolve_bridge_report(bridge_report: Path | None = None, bridge_dir: Path | None = None) -> Path | None:
    if bridge_report is not None and bridge_dir is not None:
        raise ValueError("Use either bridge_report or bridge_dir, not both.")
    if bridge_report is not None:
        return resolve_existing_path(bridge_report)
    if bridge_dir is None:
        return None
    bridge_dir = resolve_existing_path(bridge_dir)
    direct = bridge_dir / "real_3d_step_report.json"
    if direct.exists():
        return direct
    candidates = sorted(bridge_dir.glob("real_3d_step_*/real_3d_step_report.json"))
    if not candidates:
        raise FileNotFoundError(f"No real_3d_step_report.json found under {bridge_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def pose_error_from_run(run_report_path: Path, target_distance_mm: float) -> dict[str, Any]:
    run_report_path = resolve_existing_path(run_report_path)
    run_report = load_json(run_report_path)
    pose_report_path = resolve_existing_path(Path(str(run_report["pose_report"])))
    pose_report = load_json(pose_report_path)
    p_mm = np.asarray(pose_report["translation_camera_trocar_m"], dtype=np.float64) * 1000.0
    axis = np.asarray(pose_report["trocar_axis_camera"], dtype=np.float64)
    metrics = error_metrics(p_mm, axis, target_distance_mm)
    metrics.update(
        {
            "run_report": str(run_report_path),
            "run_dir": run_report.get("run_dir"),
            "pose_report": str(pose_report_path),
            "image": pose_report.get("image"),
            "status": run_report.get("status"),
            "quality_gate_status": (run_report.get("quality_gate") or {}).get("status"),
            "quality_gate_score": (run_report.get("quality_gate") or {}).get("score"),
            "pnp_mean_reprojection_error_px": pose_report.get("metrics", {}).get("mean_reprojection_error_px"),
        }
    )
    return metrics


def numeric_delta(post: dict[str, Any], pre: dict[str, Any], keys: list[str]) -> dict[str, float | None]:
    output: dict[str, float | None] = {}
    for key in keys:
        if post.get(key) is None or pre.get(key) is None:
            output[key] = None
        else:
            output[key] = float(post[key]) - float(pre[key])
    return output


def evaluate_step(
    *,
    pre_run_report: Path,
    post_run_report: Path,
    target_distance_mm: float,
    bridge_report: Path | None = None,
) -> dict[str, Any]:
    pre = pose_error_from_run(pre_run_report, target_distance_mm)
    post = pose_error_from_run(post_run_report, target_distance_mm)
    delta = numeric_delta(
        post,
        pre,
        [
            "lateral_error_mm",
            "depth_error_mm",
            "axis_angle_error_deg",
            "axis_equivalent_mm",
            "weighted_3d_error_mm",
            "pnp_mean_reprojection_error_px",
        ],
    )

    bridge: dict[str, Any] | None = None
    prediction: dict[str, Any] | None = None
    if bridge_report is not None:
        bridge_report = resolve_existing_path(bridge_report)
        bridge = load_json(bridge_report)
        predicted = (bridge.get("candidate") or {}).get("predicted_effect") or {}
        prediction = {
            "bridge_report": str(bridge_report),
            "bridge_status": bridge.get("status"),
            "moves_robot": bridge.get("moves_robot"),
            "candidate_name": (bridge.get("candidate") or {}).get("name"),
            "predicted_delta_weighted_3d_error_mm": predicted.get("delta_weighted_3d_error_mm"),
            "observed_delta_weighted_3d_error_mm": delta["weighted_3d_error_mm"],
        }
        if prediction["predicted_delta_weighted_3d_error_mm"] is not None and delta["weighted_3d_error_mm"] is not None:
            prediction["prediction_error_weighted_delta_mm"] = (
                float(delta["weighted_3d_error_mm"])
                - float(prediction["predicted_delta_weighted_3d_error_mm"])
            )

    improved = bool(post["weighted_3d_error_mm"] < pre["weighted_3d_error_mm"])
    return {
        "timestamp": datetime.now().isoformat(),
        "target_distance_mm": target_distance_mm,
        "pre": pre,
        "post": post,
        "delta": delta,
        "improved": improved,
        "prediction": prediction,
    }


def write_step_evaluation_markdown(path: Path, report: dict[str, Any]) -> None:
    pre = report["pre"]
    post = report["post"]
    delta = report["delta"]
    prediction = report.get("prediction") or {}
    lines = [
        "# Real 3D Pre/Post Step Evaluation",
        "",
        f"Generated: `{report['timestamp']}`",
        f"Target distance: `{report['target_distance_mm']:.3f} mm`",
        f"Improved: `{report['improved']}`",
        "",
        "## Error Table",
        "",
        "| Stage | lateral mm | depth mm | axis deg | axis equiv mm | weighted mm | reproj px |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| Pre | {pre['lateral_error_mm']:.3f} | {pre['depth_error_mm']:.3f} | "
            f"{pre['axis_angle_error_deg']:.3f} | {pre['axis_equivalent_mm']:.3f} | "
            f"{pre['weighted_3d_error_mm']:.3f} | {pre['pnp_mean_reprojection_error_px']:.3f} |"
        ),
        (
            f"| Post | {post['lateral_error_mm']:.3f} | {post['depth_error_mm']:.3f} | "
            f"{post['axis_angle_error_deg']:.3f} | {post['axis_equivalent_mm']:.3f} | "
            f"{post['weighted_3d_error_mm']:.3f} | {post['pnp_mean_reprojection_error_px']:.3f} |"
        ),
        (
            f"| Delta | {delta['lateral_error_mm']:.3f} | {delta['depth_error_mm']:.3f} | "
            f"{delta['axis_angle_error_deg']:.3f} | {delta['axis_equivalent_mm']:.3f} | "
            f"{delta['weighted_3d_error_mm']:.3f} | {delta['pnp_mean_reprojection_error_px']:.3f} |"
        ),
        "",
    ]
    if prediction:
        lines += [
            "## Prediction",
            "",
            f"- Bridge report: `{prediction.get('bridge_report')}`",
            f"- Candidate: `{prediction.get('candidate_name')}`",
            f"- Bridge status: `{prediction.get('bridge_status')}`",
            f"- Moves robot: `{prediction.get('moves_robot')}`",
            f"- Predicted weighted delta: `{prediction.get('predicted_delta_weighted_3d_error_mm')}` mm",
            f"- Observed weighted delta: `{prediction.get('observed_delta_weighted_3d_error_mm')}` mm",
            f"- Prediction error: `{prediction.get('prediction_error_weighted_delta_mm')}` mm",
            "",
        ]
    lines += [
        "## Inputs",
        "",
        f"- Pre run: `{pre['run_report']}`",
        f"- Post run: `{post['run_report']}`",
        "",
    ]
    write_json(path.with_suffix(".json"), report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
