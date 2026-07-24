from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from real_3d_alignment.geometry_audit import (
    axis_error_after_local_rotation_deg,
    privileged_coaxial_pose_delta,
    rotation_error_deg,
    rotation_matrix_from_local_vector_deg,
    signed_target_axis_tilt_camera_deg,
)
from real_3d_alignment.meca500_visual_env import (
    Meca500VisualAlignmentPlant,
)
from real_3d_alignment.nih_baseline import build_nih_fine_detector
from real_3d_alignment.six_axis_visual_servo import (
    ActiveEllipseOrientationServo,
    ActiveOuterEllipseOrientationServo,
    NihOuterEllipseDetector,
)
from run_mujoco_meca500_full_flow import calibrated_fine_estimate


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "output" / "m5_2_geometry_audit"
TARGET_STANDOFF_MM = 22.0


def truth_geometry(plant: Meca500VisualAlignmentPlant) -> dict[str, Any]:
    rotation = plant.camera_rotation_world()
    camera_axis = -rotation[:, 2]
    tool_axis = plant.tool_insertion_axis_world()
    target_axis = np.asarray(
        plant.data.site_xmat[plant.trocar_site_id],
        dtype=np.float64,
    ).reshape(3, 3)[:, 2]
    return {
        **plant.evaluation_pose_errors(),
        "camera_position_world_m": np.asarray(
            plant.data.site_xpos[plant.camera_site_id],
            dtype=np.float64,
        ).tolist(),
        "target_position_world_m": np.asarray(
            plant.data.site_xpos[plant.trocar_site_id],
            dtype=np.float64,
        ).tolist(),
        "camera_optical_axis_world": camera_axis.tolist(),
        "tool_insertion_axis_world": tool_axis.tolist(),
        "target_axis_world": target_axis.tolist(),
        "camera_tool_axis_dot": float(np.dot(camera_axis, tool_axis)),
        "signed_target_axis_tilt_camera_deg": list(
            signed_target_axis_tilt_camera_deg(rotation, target_axis)
        ),
    }


def solve_privileged_coaxial_reference(
    plant: Meca500VisualAlignmentPlant,
) -> np.ndarray:
    """Create an exact audit reference; never call from online control."""

    plant.reset(plant.ALIGNED_Q_DEG)
    rotation = plant.camera_rotation_world()
    delta = privileged_coaxial_pose_delta(
        camera_position_world=np.asarray(
            plant.data.site_xpos[plant.camera_site_id],
            dtype=np.float64,
        ),
        camera_rotation_world=rotation,
        target_position_world=np.asarray(
            plant.data.site_xpos[plant.trocar_site_id],
            dtype=np.float64,
        ),
        target_axis_world=np.asarray(
            plant.data.site_xmat[plant.trocar_site_id],
            dtype=np.float64,
        ).reshape(3, 3)[:, 2],
        target_standoff_mm=TARGET_STANDOFF_MM,
    )
    joint_delta = plant.camera_pose_joint_delta(
        translation_camera_mm=delta.translation_camera_mm,
        rotation_camera_deg=delta.rotation_camera_deg,
        iterations=80,
    )
    # Offline kinematic probe: exact q is required to audit signs without
    # actuator settling becoming another variable.
    plant.probe_joint_configuration(
        plant.joint_positions_rad() + joint_delta
    )
    return plant.joint_positions_rad()


def center_target_at_current_orientation(
    plant: Meca500VisualAlignmentPlant,
) -> None:
    rotation = plant.camera_rotation_world()
    camera_axis = -rotation[:, 2]
    camera_position = np.asarray(
        plant.data.site_xpos[plant.camera_site_id],
        dtype=np.float64,
    )
    target_position = np.asarray(
        plant.data.site_xpos[plant.trocar_site_id],
        dtype=np.float64,
    )
    desired_position = (
        target_position - camera_axis * TARGET_STANDOFF_MM * 1e-3
    )
    translation_camera_mm = (
        rotation.T @ (desired_position - camera_position) * 1000.0
    )
    delta_q = plant.camera_pose_joint_delta(
        translation_camera_mm=tuple(
            float(value) for value in translation_camera_mm
        ),
        iterations=50,
    )
    plant.probe_joint_configuration(
        plant.joint_positions_rad() + delta_q
    )


def build_detector(image_height: int):
    fine = build_nih_fine_detector(image_height)
    fine.config = replace(
        fine.config,
        hue_ranges=((88, 101),),
        target_standoff_mm=TARGET_STANDOFF_MM,
        maximum_outer_diameter_px=600.0 * image_height / 960.0,
        maximum_outer_center_error_px=260.0 * image_height / 960.0,
        minimum_outer_aspect_ratio=0.40,
    )
    return fine, NihOuterEllipseDetector(fine.config)


def visual_measurement(
    image_rgb: np.ndarray,
    fine_detector,
    outer_detector,
    orientation: ActiveEllipseOrientationServo,
) -> dict[str, Any]:
    fine = fine_detector.detect(image_rgb)
    outer = outer_detector.detect(image_rgb)
    result: dict[str, Any] = {
        "fine_detected": fine is not None,
        "outer_detected": outer is not None,
    }
    if outer is not None:
        result["outer"] = {
            "center_px": list(outer.center_px),
            "major_diameter_px": outer.major_diameter_px,
            "minor_diameter_px": outer.minor_diameter_px,
            "angle_deg": outer.angle_deg,
            "anisotropy": ActiveOuterEllipseOrientationServo.anisotropy(
                outer
            ),
            "feature": ActiveOuterEllipseOrientationServo.ellipse_feature(
                outer
            ).tolist(),
            "estimated_depth_mm": outer.estimated_depth_mm,
        }
    if fine is not None:
        calibrated = calibrated_fine_estimate(
            fine,
            target_standoff_mm=TARGET_STANDOFF_MM,
            aligned_anisotropy_threshold=(
                orientation.config.aligned_anisotropy_threshold
            ),
            target_anisotropy=orientation.config.target_anisotropy,
            aligned_concentricity_ratio_threshold=(
                orientation.config.aligned_concentricity_ratio_threshold
            ),
            require_orientation_convergence=False,
        )
        result["fine"] = {
            "outer_center_px": list(fine.observation.outer_center_px),
            "inner_center_px": list(fine.observation.inner_center_px),
            "outer_major_diameter_px": fine.outer_major_diameter_px,
            "outer_minor_diameter_px": fine.outer_minor_diameter_px,
            "outer_angle_deg": fine.outer_angle_deg,
            "estimated_depth_mm": fine.estimated_depth_mm,
            "raw_axis_error_deg": fine.observation.axis_error_deg,
            "calibrated_axis_error_deg": (
                calibrated.observation.axis_error_deg
            ),
            "standoff_error_mm": (
                calibrated.observation.standoff_error_mm
            ),
            "fit_error_px": fine.ellipse_fit_error_px,
            "quality_gate_pass": fine.observation.quality_gate_pass,
        }
    return result


def save_frame(
    path: Path,
    image_rgb: np.ndarray,
    *,
    label: str,
    truth: dict[str, Any],
    visual: dict[str, Any],
) -> None:
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    lines = (
        label,
        (
            f"truth axis={truth['axis_error_deg']:.3f}deg "
            f"lat={truth['lateral_error_mm']:.3f}mm "
            f"z={truth['axial_distance_mm']:.3f}mm"
        ),
        (
            "visual fine="
            f"{visual.get('fine_detected', False)} "
            f"outer={visual.get('outer_detected', False)}"
        ),
    )
    for index, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (20, 36 + 30 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(path), image)


def monotonic_correlation(
    rows: list[dict[str, Any]],
    x_key: str,
    y_path: tuple[str, ...],
) -> float | None:
    pairs: list[tuple[float, float]] = []
    for row in rows:
        value: Any = row
        for key in y_path:
            if not isinstance(value, dict) or key not in value:
                value = None
                break
            value = value[key]
        if value is not None:
            pairs.append((float(row[x_key]), float(value)))
    if len(pairs) < 2:
        return None
    x, y = np.asarray(pairs, dtype=np.float64).T
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="M5.2-A privileged coordinate and geometry audit."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument(
        "--orientation-grid-deg",
        type=float,
        nargs="+",
        default=(-6.0, 0.0, 6.0),
    )
    parser.add_argument(
        "--depth-offsets-mm",
        type=float,
        nargs="+",
        default=(-4.0, -2.0, 0.0, 2.0, 4.0),
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    plant = Meca500VisualAlignmentPlant(
        image_size_px=(args.width, args.height),
        settle_steps=1,
    )
    fine_detector, outer_detector = build_detector(args.height)
    orientation = ActiveEllipseOrientationServo()
    fixed_correction = (
        ActiveOuterEllipseOrientationServo()
        .config.outer_residual_calibration_rotation_xy_deg
    )
    try:
        plant.reset(plant.HOME_Q_DEG)
        home_truth = truth_geometry(plant)
        plant.reset(plant.SEARCH_Q_DEG)
        search_truth = truth_geometry(plant)
        plant.reset(plant.ALIGNED_Q_DEG)
        named_aligned_truth = truth_geometry(plant)

        reference_q = solve_privileged_coaxial_reference(plant)
        reference_truth = truth_geometry(plant)
        reference_visual = visual_measurement(
            plant.capture_rgb(),
            fine_detector,
            outer_detector,
            orientation,
        )

        orientation_rows: list[dict[str, Any]] = []
        for rotation_x in args.orientation_grid_deg:
            for rotation_y in args.orientation_grid_deg:
                plant.probe_joint_configuration(reference_q)
                reference_rotation = plant.camera_rotation_world().copy()
                requested_rotation = reference_rotation @ (
                    rotation_matrix_from_local_vector_deg(
                        np.asarray(
                            [rotation_x, rotation_y, 0.0],
                            dtype=np.float64,
                        )
                    )
                )
                delta_q = plant.camera_pose_joint_delta(
                    rotation_camera_deg=(
                        float(rotation_x),
                        float(rotation_y),
                        0.0,
                    ),
                    iterations=50,
                )
                plant.probe_joint_configuration(reference_q + delta_q)
                center_target_at_current_orientation(plant)
                truth = truth_geometry(plant)
                achieved_rotation = plant.camera_rotation_world().copy()
                command_rotation_error_deg = rotation_error_deg(
                    achieved_rotation,
                    requested_rotation,
                )
                valid_pose = bool(
                    command_rotation_error_deg <= 0.25
                    and truth["lateral_error_mm"] <= 0.05
                    and abs(
                        truth["axial_distance_mm"] - TARGET_STANDOFF_MM
                    )
                    <= 0.05
                )
                image = plant.capture_rgb()
                visual = visual_measurement(
                    image,
                    fine_detector,
                    outer_detector,
                    orientation,
                )
                ideal_fixed_correction_axis_error_deg = (
                    axis_error_after_local_rotation_deg(
                        camera_rotation_world=achieved_rotation,
                        target_axis_world=np.asarray(
                            plant.data.site_xmat[plant.trocar_site_id],
                            dtype=np.float64,
                        ).reshape(3, 3)[:, 2],
                        rotation_camera_deg=np.asarray(
                            [
                                fixed_correction[0],
                                fixed_correction[1],
                                0.0,
                            ],
                            dtype=np.float64,
                        ),
                    )
                )
                row = {
                    "commanded_camera_rotation_xy_deg": [
                        float(rotation_x),
                        float(rotation_y),
                    ],
                    "command_rotation_error_deg": (
                        command_rotation_error_deg
                    ),
                    "valid_commanded_pose": valid_pose,
                    "truth": truth,
                    "visual": visual,
                    "ideal_fixed_residual_axis_error_deg": (
                        ideal_fixed_correction_axis_error_deg
                    ),
                }
                orientation_rows.append(row)
                save_frame(
                    frames_dir
                    / (
                        f"orientation_x{rotation_x:+.1f}"
                        f"_y{rotation_y:+.1f}.png"
                    ),
                    image,
                    label=(
                        f"camera rotation x={rotation_x:+.1f} "
                        f"y={rotation_y:+.1f} deg"
                    ),
                    truth=truth,
                    visual=visual,
                )

        depth_rows: list[dict[str, Any]] = []
        for depth_offset in args.depth_offsets_mm:
            plant.probe_joint_configuration(reference_q)
            delta_q = plant.camera_pose_joint_delta(
                translation_camera_mm=(
                    0.0,
                    0.0,
                    float(depth_offset),
                ),
                iterations=50,
            )
            plant.probe_joint_configuration(reference_q + delta_q)
            truth = truth_geometry(plant)
            image = plant.capture_rgb()
            visual = visual_measurement(
                image,
                fine_detector,
                outer_detector,
                orientation,
            )
            row = {
                "commanded_camera_z_offset_mm": float(depth_offset),
                "truth": truth,
                "visual": visual,
            }
            depth_rows.append(row)
            save_frame(
                frames_dir / f"depth_z{depth_offset:+.1f}.png",
                image,
                label=f"camera z offset={depth_offset:+.1f} mm",
                truth=truth,
                visual=visual,
            )

        report = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "status": "M5.2-A_privileged_geometry_audit",
            "privileged_truth_used_for_control": False,
            "privileged_truth_use": (
                "offline reference-pose construction and evaluation only"
            ),
            "online_controller_executed": False,
            "image_size_px": [args.width, args.height],
            "target_standoff_mm": TARGET_STANDOFF_MM,
            "named_pose_audit": {
                "HOME_Q_DEG": home_truth,
                "SEARCH_Q_DEG": search_truth,
                "ALIGNED_Q_DEG": named_aligned_truth,
            },
            "privileged_coaxial_reference": {
                "joint_positions_deg": np.rad2deg(reference_q).tolist(),
                "truth": reference_truth,
                "visual": reference_visual,
            },
            "fixed_residual_correction_xy_deg": list(fixed_correction),
            "orientation_grid": orientation_rows,
            "depth_grid": depth_rows,
            "automatic_checks": {
                "camera_tool_axes_coaxial": bool(
                    abs(reference_truth["camera_tool_axis_dot"] - 1.0)
                    <= 1e-8
                ),
                "reference_lateral_below_0_01_mm": bool(
                    reference_truth["lateral_error_mm"] <= 0.01
                ),
                "reference_axis_below_0_01_deg": bool(
                    reference_truth["axis_error_deg"] <= 0.01
                ),
                "reference_standoff_within_0_01_mm": bool(
                    abs(
                        reference_truth["axial_distance_mm"]
                        - TARGET_STANDOFF_MM
                    )
                    <= 0.01
                ),
                "camera_positive_z_increases_true_standoff": (
                    monotonic_correlation(
                        depth_rows,
                        "commanded_camera_z_offset_mm",
                        ("truth", "axial_distance_mm"),
                    )
                ),
                "camera_positive_z_increases_visual_depth": (
                    monotonic_correlation(
                        depth_rows,
                        "commanded_camera_z_offset_mm",
                        ("visual", "fine", "estimated_depth_mm"),
                    )
                ),
                "valid_orientation_grid_poses": int(
                    sum(
                        bool(row["valid_commanded_pose"])
                        for row in orientation_rows
                    )
                ),
                "invalid_orientation_grid_poses": int(
                    sum(
                        not bool(row["valid_commanded_pose"])
                        for row in orientation_rows
                    )
                ),
                "valid_grid_false_aligned_authorizations": int(
                    sum(
                        bool(row["valid_commanded_pose"])
                        and row["truth"]["axis_error_deg"] > 2.0
                        and bool(
                            row["visual"].get("fine", {}).get(
                                "quality_gate_pass",
                                False,
                            )
                        )
                        and float(
                            row["visual"].get("fine", {}).get(
                                "calibrated_axis_error_deg",
                                float("inf"),
                            )
                        )
                        <= 2.0
                        for row in orientation_rows
                    )
                ),
            },
            "limitations": [
                "This audit deliberately uses MuJoCo truth to construct a coordinate reference.",
                "No result from this script may be used as an online control observation.",
                "The grid diagnoses geometry and raster response; it is not a frozen success-rate evaluation.",
            ],
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "m5_2_geometry_audit_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(json.dumps(report["automatic_checks"], indent=2))
        print(f"report={report_path}")
        print(f"frames={frames_dir}")
    finally:
        plant.close()


if __name__ == "__main__":
    main()
