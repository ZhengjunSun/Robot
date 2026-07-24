from __future__ import annotations

from validate_m5_frozen_report import (
    EXPECTED_STRATA,
    validate_report,
)


def episode(index: int, stratum: str) -> dict:
    return {
        "episode": index,
        "randomization_stratum": stratum,
    }


def valid_report() -> dict:
    episodes = []
    names = tuple(EXPECTED_STRATA)
    for index in range(500):
        episodes.append(episode(index, names[index % len(names)]))
    return {
        "official_m5_frozen_baseline": True,
        "protocol_sha256": "frozen",
        "privileged_truth_used_for_control": False,
        "runtime_environment": {"python": "test"},
        "run_configuration": {"settle_steps": 100},
        "summary": {
            "total_wall_contact_steps": 0,
            "over_insert_count": 0,
        },
        "episode_results": episodes,
    }


def test_accepts_complete_safe_report() -> None:
    result = validate_report(
        valid_report(),
        current_protocol_sha256="frozen",
    )
    assert result["valid"]
    assert not result["failures"]


def test_rejects_duplicate_and_contact() -> None:
    report = valid_report()
    report["episode_results"][-1]["episode"] = 0
    report["summary"]["total_wall_contact_steps"] = 1
    result = validate_report(
        report,
        current_protocol_sha256="frozen",
    )
    assert not result["valid"]
    assert "unique_episode_indices" in result["failures"]
    assert "complete_episode_index_range" in result["failures"]
    assert "no_wall_contact_steps" in result["failures"]
