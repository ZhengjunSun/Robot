from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from real_3d_alignment.geometry_audit import rotation_error_deg
from real_3d_alignment.meca500_visual_env import Meca500VisualAlignmentPlant
from real_3d_alignment.multiview_circle_pose import (
    CalibratedCameraView,
    EllipseObservation,
    active_observation_from_multiview_pose,
    estimate_multiview_circle_pose,
)
from real_3d_alignment.nih_baseline import (
    TROCAR_FLANGE_OUTER_RADIUS_MM,
    eye_in_hand_focal_length_px,
)
from real_3d_alignment.staged_alignment import (
    AlignmentThresholds,
    CoarseObservation,
    FineObservation,
    StagedAlignmentGate,
)
from run_m5_2_geometry_audit import (
    TARGET_STANDOFF_MM,
    build_detector,
    center_target_at_current_orientation,
    solve_privileged_coaxial_reference,
)
from run_m5_frozen_batch import (
    RANDOMIZATION_STRATA,
    PerturbedEyeInHandPlant,
    sample_domain,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "output" / "m5_2_multiview_smoke"


def _axis_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    cosine = float(
        np.clip(
            np.dot(
                first / np.linalg.norm(first),
                second / np.linalg.norm(second),
            ),
            -1.0,
            1.0,
        )
    )
    return float(np.rad2deg(np.arccos(cosine)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="M5.2-B three-view circle-plane estimator smoke test."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--tilt-deg", type=float, default=6.0)
    parser.add_argument("--tilt-y-deg", type=float, default=0.0)
    parser.add_argument("--episode", type=int)
    parser.add_argument("--episode-seed", type=int)
    parser.add_argument(
        "--view-offset-xy-mm",
        type=float,
        nargs=2,
        action="append",
        default=None,
        metavar=("X", "Y"),
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
    focal_length_px = eye_in_hand_focal_length_px(args.height)
    principal_point_px = (0.5 * args.width, 0.5 * args.height)
    domain_sample = None
    image_source = plant
    try:
        if (args.episode is None) != (args.episode_seed is None):
            raise ValueError(
                "--episode and --episode-seed must be provided together."
            )
        if args.episode is not None and args.episode_seed is not None:
            rng = np.random.default_rng(args.episode_seed)
            domain_sample = sample_domain(
                rng,
                image_height=args.height,
                stratum=RANDOMIZATION_STRATA[
                    args.episode % len(RANDOMIZATION_STRATA)
                ],
            )
            plant.set_domain_randomization(
                trocar_translation_mm=domain_sample.trocar_translation_mm,
                trocar_rotation_deg_xyz=(
                    domain_sample.trocar_rotation_deg_xyz
                ),
                camera_fovy_scale=domain_sample.camera_fovy_scale,
                trocar_rgb_scale=domain_sample.trocar_rgb_scale,
                sclera_rgb_scale=domain_sample.sclera_rgb_scale,
                light_intensity_scale=domain_sample.light_intensity_scale,
            )
            image_source = PerturbedEyeInHandPlant(
                plant,
                sample=domain_sample,
                rng=rng,
            )
        reference_q = solve_privileged_coaxial_reference(plant)
        plant.probe_joint_configuration(reference_q)
        requested_rotation = plant.camera_rotation_world().copy()
        tilt_delta = plant.camera_pose_joint_delta(
            rotation_camera_deg=(
                float(args.tilt_deg),
                float(args.tilt_y_deg),
                0.0,
            ),
            iterations=80,
        )
        plant.probe_joint_configuration(reference_q + tilt_delta)
        center_target_at_current_orientation(plant)
        tilted_q = plant.joint_positions_rad().copy()
        baseline_camera_position = np.asarray(
            plant.data.site_xpos[plant.camera_site_id],
            dtype=np.float64,
        ).copy()
        baseline_tool_axis = plant.tool_insertion_axis_world()
        achieved_tilt_error = rotation_error_deg(
            plant.camera_rotation_world(),
            requested_rotation,
        )

        view_offsets_xy_mm = args.view_offset_xy_mm or [
            (-1.5, 0.0),
            (0.0, 1.5),
            (1.5, -1.5),
        ]
        views: list[CalibratedCameraView] = []
        rows = []
        for index, (offset_x_mm, offset_y_mm) in enumerate(
            view_offsets_xy_mm
        ):
            plant.probe_joint_configuration(tilted_q)
            translation_delta = plant.camera_pose_joint_delta(
                translation_camera_mm=(
                    float(offset_x_mm),
                    float(offset_y_mm),
                    0.0,
                ),
                iterations=80,
            )
            plant.probe_joint_configuration(tilted_q + translation_delta)
            image = image_source.capture_rgb()
            outer = outer_detector.detect(image)
            if outer is None:
                raise RuntimeError(
                    "Outer ring missing at view offset "
                    f"({offset_x_mm}, {offset_y_mm}) mm."
                )
            view = CalibratedCameraView(
                position_world_m=np.asarray(
                    plant.data.site_xpos[plant.camera_site_id],
                    dtype=np.float64,
                ).copy(),
                rotation_world=plant.camera_rotation_world().copy(),
                ellipse=EllipseObservation(
                    center_px=outer.center_px,
                    major_diameter_px=outer.major_diameter_px,
                    minor_diameter_px=outer.minor_diameter_px,
                    major_angle_deg=outer.angle_deg,
                ),
            )
            views.append(view)
            rows.append(
                {
                    "offset_xy_mm": [
                        float(offset_x_mm),
                        float(offset_y_mm),
                    ],
                    "camera_position_world_m": (
                        view.position_world_m.tolist()
                    ),
                    "camera_rotation_world": (
                        view.rotation_world.tolist()
                    ),
                    "ellipse": asdict(view.ellipse),
                }
            )
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.ellipse(
                image_bgr,
                (
                    tuple(int(round(value)) for value in outer.center_px),
                    (
                        int(round(outer.major_diameter_px)),
                        int(round(outer.minor_diameter_px)),
                    ),
                    float(outer.angle_deg),
                ),
                (0, 255, 255),
                2,
            )
            cv2.imwrite(
                str(
                    frames_dir
                    / (
                        f"view_{index}_x{offset_x_mm:+.1f}"
                        f"_y{offset_y_mm:+.1f}mm.png"
                    )
                ),
                image_bgr,
            )

        first = views[0]
        depth_m = (
            fine_detector.config.focal_length_px
            * fine_detector.config.trocar_outer_radius_mm
            / (
                0.25
                * (
                    first.ellipse.major_diameter_px
                    + first.ellipse.minor_diameter_px
                )
            )
            * 1e-3
        )
        du = first.ellipse.center_px[0] - principal_point_px[0]
        dv = first.ellipse.center_px[1] - principal_point_px[1]
        initial_local = np.asarray(
            [
                du * depth_m / focal_length_px,
                -dv * depth_m / focal_length_px,
                -depth_m,
            ]
        )
        initial_center = (
            first.position_world_m + first.rotation_world @ initial_local
        )
        initial_normal = -first.rotation_world[:, 2]
        estimate = estimate_multiview_circle_pose(
            views,
            radius_m=TROCAR_FLANGE_OUTER_RADIUS_MM * 1e-3,
            focal_length_px=focal_length_px,
            principal_point_px=principal_point_px,
            initial_center_world_m=initial_center,
            initial_normal_world=initial_normal,
        )
        active_observation = active_observation_from_multiview_pose(
            estimate,
            observation_id=1,
            camera_position_world_m=baseline_camera_position,
            tool_axis_world=baseline_tool_axis,
            target_standoff_mm=TARGET_STANDOFF_MM,
            all_views_reachable=True,
            all_rings_detected=True,
        )
        center_ellipse = views[1].ellipse
        smoke_gate = StagedAlignmentGate(
            AlignmentThresholds(
                minimum_coarse_confidence=0.35,
                coarse_to_fine_center_error_px=150.0,
                maximum_optical_outer_error_px=150.0,
                maximum_outer_inner_concentricity_px=2.0,
                maximum_lateral_error_mm=0.20,
                maximum_axis_error_deg=6.0,
                maximum_standoff_error_mm=0.70,
                maximum_reprojection_error_px=0.80,
                required_stable_frames=1,
                require_active_multiview_confirmation=True,
            )
        )
        gate_decision = smoke_gate.update(
            coarse=CoarseObservation(
                image_size_px=(args.width, args.height),
                target_center_px=center_ellipse.center_px,
                confidence=1.0,
            ),
            fine=FineObservation(
                image_center_px=principal_point_px,
                outer_center_px=center_ellipse.center_px,
                inner_center_px=center_ellipse.center_px,
                # Deliberately model the legacy monocular false-aligned
                # observation. The experiment asks whether the new active
                # gate can revoke that otherwise-safe single-frame claim.
                lateral_error_mm=0.05,
                axis_error_deg=0.0,
                standoff_error_mm=0.10,
                reprojection_error_px=0.20,
                quality_gate_pass=True,
            ),
            active_multiview=active_observation,
        )

        truth_center = np.asarray(
            plant.data.site_xpos[plant.trocar_site_id],
            dtype=np.float64,
        )
        truth_normal = np.asarray(
            plant.data.site_xmat[plant.trocar_site_id],
            dtype=np.float64,
        ).reshape(3, 3)[:, 2]
        center_error_mm = float(
            np.linalg.norm(estimate.center_world_m - truth_center) * 1000.0
        )
        normal_error_deg = _axis_error_deg(
            estimate.normal_world,
            truth_normal,
        )
        truth_tool_target_axis_error_deg = _axis_error_deg(
            baseline_tool_axis,
            truth_normal,
        )
        report = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "status": "M5.2-B_multiview_smoke",
            "privileged_truth_used_for_estimation": False,
            "privileged_truth_use": (
                "reference-pose construction and final evaluation only"
            ),
            "view_count": len(views),
            "episode": args.episode,
            "episode_seed": args.episode_seed,
            "domain_sample": (
                None if domain_sample is None else asdict(domain_sample)
            ),
            "requested_tilt_xy_deg": [
                float(args.tilt_deg),
                float(args.tilt_y_deg),
            ],
            "achieved_rotation_from_reference_deg": achieved_tilt_error,
            "views": rows,
            "initial_center_world_m": initial_center.tolist(),
            "initial_normal_world": initial_normal.tolist(),
            "estimate": {
                "center_world_m": estimate.center_world_m.tolist(),
                "normal_world": estimate.normal_world.tolist(),
                "rms_normalized_conic_residual": (
                    estimate.rms_normalized_conic_residual
                ),
                "covariance_condition": estimate.covariance_condition,
                "success": estimate.success,
            },
            "active_multiview_observation": asdict(active_observation),
            "state_machine": {
                "phase": gate_decision.phase.value,
                "insertion_handoff_ready": (
                    gate_decision.insertion_handoff_ready
                ),
                "reasons": list(gate_decision.reasons),
                "metrics": gate_decision.metrics,
            },
            "offline_truth_evaluation": {
                "center_error_mm": center_error_mm,
                "normal_error_deg": normal_error_deg,
                "tool_target_axis_error_deg": (
                    truth_tool_target_axis_error_deg
                ),
            },
            "acceptance_targets": {
                "center_error_mm_max": 0.20,
                "normal_error_deg_max": 2.0,
            },
            "acceptance_pass": bool(
                estimate.success
                and center_error_mm <= 0.20
                and normal_error_deg <= 2.0
            ),
        }
        report_path = output_dir / "m5_2_multiview_smoke_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"report={report_path}")
        print(f"frames={frames_dir}")
    finally:
        plant.close()


if __name__ == "__main__":
    main()
