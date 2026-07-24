from __future__ import annotations

import json
from pathlib import Path

import pytest

from real_3d_alignment.staged_alignment import (
    ActiveMultiviewObservation,
    AlignmentPhase,
    AlignmentThresholds,
    CoarseObservation,
    FineObservation,
    StagedAlignmentGate,
    fine_observation_from_pose_report,
)


def coarse(center: tuple[float, float], confidence: float = 0.9) -> CoarseObservation:
    return CoarseObservation(
        image_size_px=(640, 480),
        target_center_px=center,
        confidence=confidence,
    )


def aligned_fine() -> FineObservation:
    return FineObservation(
        image_center_px=(320.0, 240.0),
        outer_center_px=(320.5, 240.0),
        inner_center_px=(320.2, 240.1),
        lateral_error_mm=0.08,
        axis_error_deg=0.4,
        standoff_error_mm=0.1,
        reprojection_error_px=0.2,
        quality_gate_pass=True,
    )


def aligned_active() -> ActiveMultiviewObservation:
    return ActiveMultiviewObservation(
        observation_id=1,
        view_count=3,
        all_views_reachable=True,
        all_rings_detected=True,
        lateral_error_mm=0.05,
        axis_error_deg=0.8,
        standoff_error_mm=0.1,
        normalized_conic_residual=0.003,
        covariance_condition=1.0e8,
    )


def test_missing_or_low_confidence_target_fails_closed() -> None:
    gate = StagedAlignmentGate()

    missing = gate.update(coarse=None, fine=None)
    low_confidence = gate.update(
        coarse=coarse((320.0, 240.0), confidence=0.1),
        fine=None,
    )

    assert missing.phase is AlignmentPhase.SEARCH
    assert low_confidence.phase is AlignmentPhase.SEARCH
    assert missing.hold_position
    assert not missing.insertion_handoff_ready


def test_far_target_stays_in_coarse_alignment() -> None:
    decision = StagedAlignmentGate().update(
        coarse=coarse((500.0, 400.0)),
        fine=None,
    )

    assert decision.phase is AlignmentPhase.COARSE
    assert not decision.hold_position
    assert not decision.insertion_handoff_ready


def test_fine_alignment_requires_all_geometric_checks() -> None:
    fine = aligned_fine()
    bad_axis = FineObservation(
        **{**fine.__dict__, "axis_error_deg": 2.0}
    )

    decision = StagedAlignmentGate().update(
        coarse=coarse((321.0, 240.0)),
        fine=bad_axis,
    )

    assert decision.phase is AlignmentPhase.FINE
    assert "axis_error_above_threshold" in decision.reasons
    assert not decision.insertion_handoff_ready


def test_insertion_handoff_needs_consecutive_stable_frames() -> None:
    gate = StagedAlignmentGate(
        AlignmentThresholds(required_stable_frames=3)
    )
    observations = [
        gate.update(coarse=coarse((320.0, 240.0)), fine=aligned_fine())
        for _ in range(3)
    ]

    assert [item.phase for item in observations] == [
        AlignmentPhase.FINE,
        AlignmentPhase.FINE,
        AlignmentPhase.ALIGNED,
    ]
    assert not observations[1].insertion_handoff_ready
    assert observations[2].insertion_handoff_ready


def test_active_multiview_phase_is_required_before_alignment() -> None:
    gate = StagedAlignmentGate(
        AlignmentThresholds(
            required_stable_frames=1,
            require_active_multiview_confirmation=True,
        )
    )
    waiting = gate.update(
        coarse=coarse((320.0, 240.0)),
        fine=aligned_fine(),
    )
    aligned = gate.update(
        coarse=coarse((320.0, 240.0)),
        fine=aligned_fine(),
        active_multiview=aligned_active(),
    )
    assert waiting.phase is AlignmentPhase.OBSERVE_ACTIVE
    assert not waiting.insertion_handoff_ready
    assert aligned.phase is AlignmentPhase.ALIGNED
    assert aligned.insertion_handoff_ready


def test_bad_or_incomplete_active_observation_fails_closed() -> None:
    gate = StagedAlignmentGate(
        AlignmentThresholds(
            required_stable_frames=1,
            require_active_multiview_confirmation=True,
        )
    )
    bad = ActiveMultiviewObservation(
        **{
            **aligned_active().__dict__,
            "view_count": 2,
            "all_views_reachable": False,
            "axis_error_deg": 2.5,
            "covariance_condition": float("inf"),
        }
    )
    decision = gate.update(
        coarse=coarse((320.0, 240.0)),
        fine=aligned_fine(),
        active_multiview=bad,
    )
    assert decision.phase is AlignmentPhase.OBSERVE_ACTIVE
    assert not decision.insertion_handoff_ready
    assert "active_view_unreachable" in decision.reasons
    assert "active_view_count_below_threshold" in decision.reasons
    assert "active_axis_error_above_threshold" in decision.reasons
    assert "active_covariance_condition_above_threshold" in decision.reasons


def test_reused_active_observation_cannot_accumulate_stability() -> None:
    gate = StagedAlignmentGate(
        AlignmentThresholds(
            required_stable_frames=2,
            require_active_multiview_confirmation=True,
        )
    )
    first = gate.update(
        coarse=coarse((320.0, 240.0)),
        fine=aligned_fine(),
        active_multiview=aligned_active(),
    )
    stale = gate.update(
        coarse=coarse((320.0, 240.0)),
        fine=aligned_fine(),
        active_multiview=aligned_active(),
    )
    fresh = ActiveMultiviewObservation(
        **{**aligned_active().__dict__, "observation_id": 2}
    )
    after_stale = gate.update(
        coarse=coarse((320.0, 240.0)),
        fine=aligned_fine(),
        active_multiview=fresh,
    )
    assert first.stable_frames == 1
    assert stale.stable_frames == 0
    assert stale.phase is AlignmentPhase.OBSERVE_ACTIVE
    assert stale.reasons == ("active_multiview_observation_not_fresh",)
    assert after_stale.stable_frames == 1
    assert not after_stale.insertion_handoff_ready


def test_failed_frame_resets_stability_counter() -> None:
    gate = StagedAlignmentGate(
        AlignmentThresholds(required_stable_frames=2)
    )
    gate.update(coarse=coarse((320.0, 240.0)), fine=aligned_fine())
    off_center = FineObservation(
        **{**aligned_fine().__dict__, "outer_center_px": (330.0, 240.0)}
    )

    failed = gate.update(
        coarse=coarse((320.0, 240.0)),
        fine=off_center,
    )

    assert failed.stable_frames == 0
    assert failed.phase is AlignmentPhase.FINE


def test_missing_target_revokes_existing_handoff_authorization() -> None:
    gate = StagedAlignmentGate(
        AlignmentThresholds(required_stable_frames=1)
    )
    aligned = gate.update(
        coarse=coarse((320.0, 240.0)), fine=aligned_fine()
    )

    revoked = gate.update(coarse=None, fine=None)

    assert aligned.insertion_handoff_ready
    assert not revoked.insertion_handoff_ready
    assert revoked.phase is AlignmentPhase.SEARCH
    assert revoked.stable_frames == 0


def test_pose_report_adapter_uses_inner_outer_and_axis_geometry() -> None:
    report = {
        "outer_ellipse": {"center": [320.5, 240.0]},
        "inner_ellipse": {"center": [320.0, 240.0]},
        "pose_camera_trocar_mm_deg": [0.06, 0.08, 18.1, 0.0, 0.0, 0.0],
        "trocar_axis_camera": [0.0, 0.0, 1.0],
        "metrics": {"mean_reprojection_error_px": 0.2},
    }

    observation = fine_observation_from_pose_report(
        report,
        image_size_px=(640, 480),
        target_standoff_mm=18.0,
        quality_gate_pass=True,
    )

    assert observation.lateral_error_mm == pytest.approx(0.1)
    assert observation.axis_error_deg == pytest.approx(0.0)
    assert observation.standoff_error_mm == pytest.approx(0.1)


def test_repository_threshold_config_matches_gate_contract() -> None:
    project_root = Path(__file__).resolve().parents[1]
    with (project_root / "config" / "staged_visual_alignment.json").open(
        "r", encoding="utf-8"
    ) as stream:
        config = json.load(stream)

    thresholds = AlignmentThresholds(**config["thresholds"])

    assert thresholds.required_stable_frames == 5
    assert config["handoff"]["fail_closed"] is True
