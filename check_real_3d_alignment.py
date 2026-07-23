from __future__ import annotations

import argparse
from pathlib import Path

from real_3d_alignment.config import DEFAULT_CONFIG
from real_3d_alignment.preflight import run_preflight, write_preflight_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight checks for the unified real 3D alignment pipeline.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = run_preflight(args.config)
    if args.output is not None:
        write_preflight_report(args.output, report)

    print("real_3d_alignment preflight:", report["status"])
    print("errors:", report["error_count"], "warnings:", report["warning_count"])
    for module in report["sections"]["modules"]:
        if module["status"] != "ok":
            print(f"- module {module['name']}: {module['status']} found={module['found']}")
    for section in ("paths", "entrypoints", "assets", "safety", "control"):
        for item in report["sections"].get(section, []):
            if item.get("status") != "ok":
                print(f"- {section} {item.get('label', '')}: {item.get('message', item.get('path', item))}")
    detector = report["sections"].get("detector_models", {})
    if detector.get("status") != "ok":
        print(f"- detector_models: strategy={detector.get('strategy')} single={detector.get('single_available')} dual={detector.get('dual_available')}")
    evidence = report["sections"].get("evidence_artifacts", {})
    if evidence.get("status") != "ok":
        print(f"- evidence_artifacts: missing_count={evidence.get('missing_count')}")
    if report["sections"]["hardcoded_paths"]["hits"]:
        print("- hardcoded paths:", ", ".join(report["sections"]["hardcoded_paths"]["hits"]))
    if args.output is not None:
        print("report:", args.output)


if __name__ == "__main__":
    main()
