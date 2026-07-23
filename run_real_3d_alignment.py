from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from real_3d_alignment.config import DEFAULT_CONFIG, load_config, resolve_project_path
from real_3d_alignment.observation import capture_live_observation
from real_3d_alignment.pipeline import run_pipeline


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_six_floats(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if len(values) != 6:
        raise argparse.ArgumentTypeError("Expected six comma-separated values.")
    return values


def safe_name(path: Path) -> str:
    name = path.name if path.is_dir() else path.stem
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
    return safe[:80] or "sample"


def iter_observation_dirs(root: Path) -> list[Path]:
    return sorted({path.parent for path in root.rglob("camera_rgb.png")})


def iter_images(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def summarize_record(index: int, source: Path, report: dict[str, Any]) -> dict[str, Any]:
    gate = report.get("quality_gate") or {}
    camera = report.get("camera_frame_control") or {}
    link6 = report.get("link6_dry_run_control") or {}
    errors = link6.get("current_errors") or camera.get("current_errors") or {}
    predicted = camera.get("predicted_effect") or {}
    predicted_after = predicted.get("after") or {}
    ranking = camera.get("candidate_ranking") or {}
    return {
        "index": index,
        "source": str(source),
        "status": report.get("status"),
        "run_dir": report.get("run_dir"),
        "gate_status": gate.get("status"),
        "gate_score": gate.get("score"),
        "gate_reasons": ";".join(gate.get("reasons") or []),
        "lateral_error_mm": errors.get("lateral_error_mm"),
        "depth_error_mm": errors.get("depth_error_mm"),
        "axis_angle_error_deg": errors.get("axis_angle_error_deg"),
        "weighted_3d_error_mm": errors.get("weighted_3d_error_mm"),
        "control_status": camera.get("status"),
        "best_predicted_candidate": ranking.get("best_candidate"),
        "predicted_delta_weighted_3d_error_mm": predicted.get("delta_weighted_3d_error_mm"),
        "predicted_after_weighted_3d_error_mm": predicted_after.get("weighted_3d_error_mm"),
        "control_warnings": ";".join(camera.get("warnings") or []),
        "error_type": (report.get("error") or {}).get("type"),
        "error_message": (report.get("error") or {}).get("message"),
    }


def write_batch_summary(
    batch_dir: Path,
    rows: list[dict[str, Any]],
    reports: list[dict[str, Any]],
    *,
    stateful_gate_enabled: bool,
) -> None:
    batch_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "timestamp": datetime.now().isoformat(),
        "sample_count": len(rows),
        "completed_count": sum(1 for row in rows if row["status"] == "dry_run_complete"),
        "failed_count": sum(1 for row in rows if row["status"] == "failed"),
        "accepted_count": sum(1 for row in rows if row["gate_status"] == "accepted"),
        "stateful_gate_enabled": stateful_gate_enabled,
        "records": rows,
    }
    (batch_dir / "batch_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if rows:
        with (batch_dir / "batch_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    (batch_dir / "batch_reports_index.txt").write_text(
        "\n".join(str(report.get("run_dir")) for report in reports),
        encoding="utf-8",
    )


def run_batch(args: argparse.Namespace, source_kind: str, sources: list[Path]) -> dict[str, Any]:
    cfg = load_config(args.config)
    output_root = resolve_project_path(cfg["paths"]["output_root"])
    batch_dir = args.output_dir or (output_root / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    rows = []
    reports = []
    previous_accepted_pose_report: dict[str, Any] | None = None
    for index, source in enumerate(sources, start=1):
        sample_dir = batch_dir / "samples" / f"{index:03d}_{safe_name(source)}"
        report = run_pipeline(
            config_path=args.config,
            image=source if source_kind == "image" else None,
            observation_dir=source if source_kind == "observation" else None,
            output_dir=sample_dir,
            detection_method=args.detection_method,
            use_inner_ring=args.use_inner_ring if args.use_inner_ring else None,
            previous_accepted_pose_report=previous_accepted_pose_report if args.stateful_gate else None,
        )
        reports.append(report)
        rows.append(summarize_record(index, source, report))
        if args.stateful_gate and (report.get("quality_gate") or {}).get("accepted") and report.get("pose_report"):
            previous_accepted_pose_report = json.loads(Path(report["pose_report"]).read_text(encoding="utf-8"))
        print(f"[{index}/{len(sources)}] {source.name}: {report.get('status')}")
    write_batch_summary(batch_dir, rows, reports, stateful_gate_enabled=bool(args.stateful_gate))
    return {
        "batch_dir": str(batch_dir),
        "sample_count": len(rows),
        "completed_count": sum(1 for row in rows if row["status"] == "dry_run_complete"),
        "failed_count": sum(1 for row in rows if row["status"] == "failed"),
        "accepted_count": sum(1 for row in rows if row["gate_status"] == "accepted"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified dry-run pipeline for real 3D trocar alignment. It does not move the robot."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--image", type=Path, default=None, help="Single RGB image to estimate T_camera_trocar from.")
    parser.add_argument("--observation-dir", type=Path, default=None, help="Observation folder containing camera_rgb.png.")
    parser.add_argument("--capture-live", action="store_true", help="Capture one camera frame and read-only robot state before dry-run.")
    parser.add_argument("--batch-observations", type=Path, default=None, help="Recursively run every folder containing camera_rgb.png.")
    parser.add_argument("--batch-images", type=Path, default=None, help="Recursively run every image file under this folder.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--detection-method", choices=["color", "edge", "hough", "auto"], default=None)
    parser.add_argument("--use-inner-ring", action="store_true")
    parser.add_argument("--stateful-gate", action="store_true", help="In batch mode, compare each pose with the last accepted pose.")
    parser.add_argument("--note", type=str, default="")
    parser.add_argument("--robot-timeout-s", type=float, default=3.0)
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--manual-joints-deg", type=parse_six_floats, default=None)
    parser.add_argument("--manual-cart-pose-mm-deg", type=parse_six_floats, default=None)
    args = parser.parse_args()

    selected_sources = sum(
        [
            args.image is not None,
            args.observation_dir is not None,
            args.capture_live,
            args.batch_observations is not None,
            args.batch_images is not None,
        ]
    )
    if selected_sources != 1:
        parser.error("Choose exactly one input source: --image, --observation-dir, --capture-live, --batch-observations, or --batch-images.")

    if args.batch_observations is not None:
        sources = iter_observation_dirs(args.batch_observations)
        if not sources:
            parser.error(f"No camera_rgb.png observations found under {args.batch_observations}")
        result = run_batch(args, "observation", sources)
        print("batch complete:", result)
        return

    if args.batch_images is not None:
        sources = iter_images(args.batch_images)
        if not sources:
            parser.error(f"No images found under {args.batch_images}")
        result = run_batch(args, "image", sources)
        print("batch complete:", result)
        return

    output_dir = args.output_dir
    observation_dir = args.observation_dir
    if args.capture_live:
        cfg = load_config(args.config)
        output_root = resolve_project_path(cfg["paths"]["output_root"])
        output_dir = output_dir or (output_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        observation_dir = capture_live_observation(
            config_path=args.config,
            output_dir=output_dir / "observation",
            note=args.note,
            robot_timeout_s=args.robot_timeout_s,
            warmup_frames=args.warmup_frames,
            manual_joints_deg=args.manual_joints_deg,
            manual_cart_pose_mm_deg=args.manual_cart_pose_mm_deg,
        )

    report = run_pipeline(
        config_path=args.config,
        image=args.image,
        observation_dir=observation_dir,
        output_dir=output_dir,
        detection_method=args.detection_method,
        use_inner_ring=args.use_inner_ring if args.use_inner_ring else None,
    )

    print("run_dir:", report["run_dir"])
    if report.get("status") != "dry_run_complete":
        error = report.get("error") or {}
        print("real_3d_alignment dry-run failed")
        print("error:", error.get("type"), error.get("message"))
        raise SystemExit(2)
    gate = report["quality_gate"]
    print("real_3d_alignment dry-run complete")
    print("quality_gate:", gate["status"], "score=", gate["score"], "reasons=", gate["reasons"])
    print("pose_report:", report["pose_report"])
    print("control_suggestion:", str(Path(report["run_dir"]) / "control_suggestion.json"))


if __name__ == "__main__":
    main()
