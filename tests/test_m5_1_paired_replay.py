from __future__ import annotations

from run_m5_1_paired_replay import (
    m5_1_temporal_config,
    paired_summary,
)


def episode(index: int, *, fine: bool, success: bool) -> dict:
    return {
        "episode": index,
        "fine_success": fine,
        "full_flow_success": success,
        "insertion_extension_mm": 12.5 if success else 0.0,
        "contact": {"wall_contact_count": 0},
    }


def test_m5_1_temporal_config_scales_pixel_limit() -> None:
    low = m5_1_temporal_config(480)
    high = m5_1_temporal_config(960)
    assert high.maximum_center_range_px == 2 * low.maximum_center_range_px
    assert high.minimum_valid_samples == 3


def test_paired_summary_reports_transitions() -> None:
    baseline = {
        "episode_results": [
            episode(1, fine=True, success=False),
            episode(2, fine=True, success=True),
        ]
    }
    candidate = [
        episode(1, fine=True, success=True),
        episode(2, fine=False, success=False),
    ]
    summary = paired_summary(baseline, candidate)
    assert summary["paired_episodes"] == 2
    assert summary["baseline"]["full_flow_success_rate"] == 0.5
    assert summary["m5_1"]["full_flow_success_rate"] == 0.5
    assert summary["transitions"] == {
        "baseline_fail_to_m5_1_success": 1,
        "baseline_success_to_m5_1_fail": 1,
    }
