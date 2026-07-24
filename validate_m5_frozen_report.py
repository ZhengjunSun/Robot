from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import run_m5_frozen_batch as m5


ROOT = Path(__file__).resolve().parent
DEFAULT_REPORT = (
    ROOT / "output" / "m5_frozen_500" / "m5_frozen_report.json"
)
EXPECTED_STRATA = {
    "geometry": 100,
    "photometric": 100,
    "calibration": 100,
    "occlusion": 100,
    "combined": 100,
}


def validate_report(
    report: dict[str, Any],
    *,
    current_protocol_sha256: str,
) -> dict[str, Any]:
    episodes = report.get("episode_results", [])
    indices = [int(item["episode"]) for item in episodes]
    strata = Counter(
        str(item["randomization_stratum"]) for item in episodes
    )
    checks = {
        "exactly_500_episodes": len(episodes) == 500,
        "official_flag_true": (
            report.get("official_m5_frozen_baseline") is True
        ),
        "unique_episode_indices": len(set(indices)) == len(indices),
        "complete_episode_index_range": sorted(indices) == list(range(500)),
        "five_strata_exactly_100": dict(strata) == EXPECTED_STRATA,
        "protocol_matches_current_frozen_sources": (
            report.get("protocol_sha256") == current_protocol_sha256
        ),
        "controller_does_not_use_privileged_truth": (
            report.get("privileged_truth_used_for_control") is False
        ),
        "runtime_environment_recorded": bool(
            report.get("runtime_environment")
        ),
        "run_configuration_recorded": bool(
            report.get("run_configuration")
        ),
        "no_wall_contact_steps": (
            report.get("summary", {}).get("total_wall_contact_steps") == 0
        ),
        "no_over_insertions": (
            report.get("summary", {}).get("over_insert_count") == 0
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "valid": not failures,
        "checks": checks,
        "failures": failures,
        "observed": {
            "episodes": len(episodes),
            "strata": dict(strata),
            "report_protocol_sha256": report.get("protocol_sha256"),
            "current_protocol_sha256": current_protocol_sha256,
            "official_m5_frozen_baseline": report.get(
                "official_m5_frozen_baseline"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-closed acceptance check for the M5 frozen report."
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Return success while a validly structured batch is incomplete.",
    )
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    result = validate_report(
        report,
        current_protocol_sha256=m5.protocol_fingerprint(),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["valid"]:
        return
    only_incomplete = set(result["failures"]).issubset(
        {
            "exactly_500_episodes",
            "official_flag_true",
            "complete_episode_index_range",
            "five_strata_exactly_100",
        }
    )
    if not (args.allow_incomplete and only_incomplete):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
