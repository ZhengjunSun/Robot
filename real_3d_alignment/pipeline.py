from __future__ import annotations

import math
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .control_analysis import error_metrics, rank_step_candidates
from .config import PROJECT_ROOT, load_config, load_json, resolve_project_path, write_json


SCRIPTS_3D = PROJECT_ROOT / "3d_modeling" / "scripts"
if str(SCRIPTS_3D) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_3D))

import estimate_trocar_pose_from_ring as ring_pose  # noqa: E402
import generate_3d_alignment_dry_run_commands as dry_run  # noqa: E402
from handeye_common import matrix_to_pose_mm_deg, project_transform_to_se3  # noqa: E402
from pose_quality_gate import QualityGateThresholds, evaluate_pose_quality  # noqa: E402


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _clip_vector(vector: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= max_norm or norm <= 1e-12:
        return vector
    return vector * (max_norm / norm)


def _load_observation(observation_dir: Path | None, image: Path | None) -> tuple[Path, dict[str, Any] | None]:
    image_path, metadata = ring_pose.load_observation(observation_dir, image)
    return image_path, metadata


def estimate_pose(
    *,
    image: Path | None,
    observation_dir: Path | None,
    camera_config: Path,
    trocar_config: Path,
    extrinsic_config: Path,
    output_dir: Path,
    detection_method: str,
    use_inner_ring: bool,
    save_debug_images: bool,
) -> dict[str, Any]:
    image_path, metadata = _load_observation(observation_dir, image)
    frame = ring_pose.safe_imread(image_path)
    if frame is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    camera_cfg = load_json(camera_config)
    trocar_cfg = load_json(trocar_config)
    extrinsic = load_json(extrinsic_config)

    outer_ellipse, outer_contour, outer_mask, method_used = ring_pose.detect_ring_with_fallback(
        frame,
        trocar_cfg,
        detection_method=detection_method,
    )
    inner_ellipse = None
    inner_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    inner_detection_error = None
    try:
        inner_ellipse, inner_contour, inner_mask = ring_pose.detect_inner_dark(frame, outer_ellipse, trocar_cfg)
    except RuntimeError as exc:
        inner_detection_error = str(exc)
        if use_inner_ring:
            raise

    object_points, image_points, annotation = ring_pose.build_correspondences(
        trocar_cfg,
        outer_ellipse,
        inner_ellipse,
        use_inner_ring=use_inner_ring,
    )
    T_camera_trocar, metrics = ring_pose.solve_pose(object_points, image_points, camera_cfg)

    report: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "status": "ok",
        "image": str(image_path),
        "observation_dir": None if observation_dir is None else str(observation_dir),
        "camera_config": str(camera_config),
        "trocar_config": str(trocar_config),
        "extrinsic": str(extrinsic_config),
        "outer_ellipse": {k: v for k, v in outer_ellipse.items() if k != "theta_rad"},
        "inner_ellipse": None if inner_ellipse is None else {k: v for k, v in inner_ellipse.items() if k != "theta_rad"},
        "inner_detection_error": inner_detection_error,
        "detection_method_used": method_used,
        "use_inner_ring_for_pnp": use_inner_ring,
        "annotation_points": annotation,
        "T_camera_trocar": T_camera_trocar.tolist(),
        "pose_camera_trocar_mm_deg": matrix_to_pose_mm_deg(T_camera_trocar),
        "translation_camera_trocar_m": T_camera_trocar[:3, 3].tolist(),
        "trocar_axis_camera": T_camera_trocar[:3, 2].tolist(),
        "metrics": metrics,
    }

    if metadata and metadata.get("robot", {}).get("success"):
        T_base_link6 = np.array(metadata["robot"]["T_base_link6"], dtype=np.float64)
        T_link6_camera = project_transform_to_se3(np.array(extrinsic["T_link6_camera"], dtype=np.float64))
        T_base_trocar = T_base_link6 @ T_link6_camera @ T_camera_trocar
        report["T_base_trocar"] = T_base_trocar.tolist()
        report["pose_base_trocar_mm_deg"] = matrix_to_pose_mm_deg(T_base_trocar)
        report["trocar_axis_base"] = T_base_trocar[:3, 2].tolist()
        report["robot"] = metadata["robot"]

    if save_debug_images:
        output_dir.mkdir(parents=True, exist_ok=True)
        overlay = ring_pose.draw_overlay(frame, outer_ellipse, inner_ellipse, object_points, T_camera_trocar, camera_cfg)
        ring_pose.safe_imwrite(output_dir / "pose_overlay.png", overlay)
        ring_pose.safe_imwrite(output_dir / "outer_mask.png", outer_mask)
        ring_pose.safe_imwrite(output_dir / "inner_mask.png", inner_mask)
        report["overlay"] = str(output_dir / "pose_overlay.png")
        report["outer_mask"] = str(output_dir / "outer_mask.png")
        report["inner_mask"] = str(output_dir / "inner_mask.png")

    write_json(output_dir / "real_trocar_ring_annotation.json", {"image": str(image_path), "points": annotation})
    write_json(output_dir / "real_trocar_ring_pose_report.json", report)
    return report


def evaluate_gate(
    report: dict[str, Any],
    thresholds_cfg: dict[str, Any],
    previous_accepted: dict[str, Any] | None = None,
) -> dict[str, Any]:
    thresholds = QualityGateThresholds(**thresholds_cfg)
    return evaluate_pose_quality(report, previous_accepted=previous_accepted, thresholds=thresholds)


def camera_frame_control_suggestion(report: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    pose = np.array(report["pose_camera_trocar_mm_deg"][:3], dtype=np.float64)
    axis = np.array(report["trocar_axis_camera"], dtype=np.float64)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)

    target_distance_mm = float(control["target_distance_mm"])
    current_errors = error_metrics(pose, axis, target_distance_mm)

    raw_translation = np.array(
        [
            float(control["k_xy"]) * pose[0],
            float(control["k_xy"]) * pose[1],
            float(control["k_z"]) * current_errors["depth_error_mm"],
        ],
        dtype=np.float64,
    )
    translation = _clip_vector(raw_translation, float(control["max_translation_step_mm"]))

    raw_rotvec = dry_run.rotation_from_z_to_axis(axis) * float(control["k_rot"])
    max_rot_rad = math.radians(float(control["max_rotation_step_deg"]))
    raw_norm = float(np.linalg.norm(raw_rotvec))
    rotvec = raw_rotvec if raw_norm <= max_rot_rad or raw_norm <= 1e-12 else raw_rotvec * (max_rot_rad / raw_norm)
    candidate_ranking = rank_step_candidates(
        p_mm=pose,
        axis=axis,
        target_distance_mm=target_distance_mm,
        translation_step_mm=translation,
        rotvec_step_rad=rotvec,
    )
    combined = next(
        item for item in candidate_ranking["candidates"] if item["name"] == "combined_translation_rotation"
    )
    warnings = []
    if not combined["improved"]:
        warnings.append("combined_translation_rotation_is_not_predicted_to_reduce_weighted_error")
    if candidate_ranking["best_candidate"] != "combined_translation_rotation":
        warnings.append(f"best_predicted_candidate_is_{candidate_ranking['best_candidate']}")

    return {
        "status": "ok" if combined["improved"] else "needs_review",
        "frame": "camera",
        "target_distance_mm": target_distance_mm,
        "current_errors": current_errors,
        "suggested_step": {
            "translation_step_camera_mm": [float(v) for v in translation],
            "rotation_step_camera_rotvec_deg": [float(math.degrees(v)) for v in rotvec],
            "rotation_step_angle_deg": float(math.degrees(np.linalg.norm(rotvec))),
        },
        "predicted_effect": combined,
        "candidate_ranking": candidate_ranking,
        "warnings": warnings,
        "limits": {
            "max_translation_step_mm": float(control["max_translation_step_mm"]),
            "max_rotation_step_deg": float(control["max_rotation_step_deg"]),
        },
        "note": "Camera-frame suggestion only. Convert through live robot state before executing on Meca500.",
    }


def link6_dry_run_suggestion(
    *,
    pose_report_path: Path,
    extrinsic_config: Path,
    control: dict[str, Any],
) -> dict[str, Any]:
    extrinsic = load_json(extrinsic_config)
    T_link6_camera = np.array(extrinsic["T_link6_camera"], dtype=np.float64)
    return dry_run.build_command(
        report_path=pose_report_path,
        T_link6_camera=T_link6_camera,
        target_distance_mm=float(control["target_distance_mm"]),
        k_xy=float(control["k_xy"]),
        k_z=float(control["k_z"]),
        k_rot=float(control["k_rot"]),
        max_translation_step_mm=float(control["max_translation_step_mm"]),
        max_rotation_step_deg=float(control["max_rotation_step_deg"]),
        max_link6_translation_step_mm=float(control["max_link6_translation_step_mm"]),
        min_link6_z_mm=float(control["min_link6_z_mm"]),
    )


def run_pipeline(
    *,
    config_path: Path,
    image: Path | None = None,
    observation_dir: Path | None = None,
    output_dir: Path | None = None,
    detection_method: str | None = None,
    use_inner_ring: bool | None = None,
    previous_accepted_pose_report: dict[str, Any] | None = None,
    raise_on_failure: bool = False,
) -> dict[str, Any]:
    if image is None and observation_dir is None:
        raise ValueError("Use either image or observation_dir.")

    cfg = load_config(config_path)
    paths = cfg["paths"]
    perception = cfg["perception"]
    control = cfg["control"]

    run_dir = output_dir or (resolve_project_path(paths["output_root"]) / f"run_{_stamp()}")
    pose_dir = run_dir / "pose"
    camera_config = resolve_project_path(paths["camera_config"])
    extrinsic_config = resolve_project_path(paths["extrinsic_config"])
    trocar_config = resolve_project_path(paths["trocar_config"])

    method = detection_method or str(perception["detection_method"])
    inner = bool(perception["use_inner_ring"] if use_inner_ring is None else use_inner_ring)

    write_json(
        run_dir / "run_inputs.json",
        {
            "config": str(config_path),
            "image": None if image is None else str(image),
            "observation_dir": None if observation_dir is None else str(observation_dir),
            "detection_method": method,
            "use_inner_ring": inner,
        },
    )

    try:
        pose_report = estimate_pose(
            image=image,
            observation_dir=observation_dir,
            camera_config=camera_config,
            trocar_config=trocar_config,
            extrinsic_config=extrinsic_config,
            output_dir=pose_dir,
            detection_method=method,
            use_inner_ring=inner,
            save_debug_images=bool(perception.get("save_debug_images", True)),
        )
        pose_report_path = pose_dir / "real_trocar_ring_pose_report.json"

        gate = evaluate_gate(pose_report, cfg["quality_gate"], previous_accepted=previous_accepted_pose_report)
        camera_step = camera_frame_control_suggestion(pose_report, control)
        link6_step = link6_dry_run_suggestion(
            pose_report_path=pose_report_path,
            extrinsic_config=extrinsic_config,
            control=control,
        )

        report = {
            "timestamp": datetime.now().isoformat(),
            "status": "dry_run_complete",
            "run_dir": str(run_dir),
            "pose_report": str(pose_report_path),
            "quality_gate": gate,
            "camera_frame_control": camera_step,
            "link6_dry_run_control": link6_step,
            "execution": {
                "enabled": False,
                "reason": "This unified entrypoint intentionally stops before real robot motion.",
            },
        }
        write_json(run_dir / "quality_gate.json", gate)
        write_json(run_dir / "control_suggestion.json", {"camera_frame": camera_step, "link6_dry_run": link6_step})
        write_json(run_dir / "run_report.json", report)
        return report
    except Exception as exc:
        report = {
            "timestamp": datetime.now().isoformat(),
            "status": "failed",
            "run_dir": str(run_dir),
            "pose_report": None,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            "execution": {
                "enabled": False,
                "reason": "Pipeline failed before any real robot motion could be considered.",
            },
        }
        write_json(run_dir / "run_report.json", report)
        if raise_on_failure:
            raise
        return report
