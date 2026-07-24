from __future__ import annotations

from dataclasses import replace

from real_3d_alignment.staged_alignment import FineObservation
from real_3d_alignment.temporal_validation import (
    FailClosedReobservationGate,
    ReobservationAction,
    TemporalFineObservationValidator,
    TemporalFineValidationConfig,
)


def observation(**changes) -> FineObservation:
    baseline = FineObservation(
        image_center_px=(640.0, 480.0),
        outer_center_px=(640.1, 479.9),
        inner_center_px=(640.0, 480.0),
        lateral_error_mm=0.03,
        axis_error_deg=1.0,
        standoff_error_mm=0.10,
        reprojection_error_px=0.20,
        quality_gate_pass=True,
    )
    return replace(baseline, **changes)


def test_temporal_validator_requires_fresh_stable_window() -> None:
    validator = TemporalFineObservationValidator(
        TemporalFineValidationConfig(
            window_size=3,
            minimum_valid_samples=3,
        )
    )
    assert not validator.update(observation()).ready
    assert not validator.update(observation()).ready
    result = validator.update(observation())
    assert result.ready
    assert result.observation is not None
    assert result.sample_count == 3


def test_temporal_validator_rejects_spread_and_clears_on_missing() -> None:
    validator = TemporalFineObservationValidator(
        TemporalFineValidationConfig(
            window_size=3,
            minimum_valid_samples=3,
            maximum_center_range_px=1.0,
        )
    )
    validator.update(observation())
    validator.update(observation())
    unstable = validator.update(
        observation(outer_center_px=(643.0, 480.0))
    )
    assert not unstable.ready
    assert "temporal_center_unstable" in unstable.reasons
    missing = validator.update(None)
    assert not missing.ready
    assert missing.sample_count == 0
    assert not validator.update(observation()).ready


def test_reobservation_gate_holds_without_authorizing_motion() -> None:
    gate = FailClosedReobservationGate(maximum_reobservation_frames=2)
    first = gate.update(False)
    second = gate.update(False)
    aborted = gate.update(False)
    assert first.action is ReobservationAction.HOLD_AND_REOBSERVE
    assert second.action is ReobservationAction.HOLD_AND_REOBSERVE
    assert aborted.action is ReobservationAction.ABORT
    recovered = gate.update(True)
    assert recovered.action is ReobservationAction.MOVE
    assert recovered.reason == "fresh_visual_authorization_recovered"
