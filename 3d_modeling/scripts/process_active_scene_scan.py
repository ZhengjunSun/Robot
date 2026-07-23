from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from handeye_common import DEFAULT_CONFIG, load_json, pose_mm_deg_to_matrix, write_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "active_scene_scan_processed"
DEFAULT_EXTRINSIC = ROOT / "config" / "camera_extrinsic_colleague_20260612.json"


def find_report(scan_dir: Path) -> Path:
    report = scan_dir / "active_scene_scan_report.json"
    if not report.exists():
        raise FileNotFoundError(f"Active scan report not found: {report}")
    return report


def robot_metadata_from_capture(capture: dict[str, Any], scan_report: dict[str, Any]) -> dict[str, Any]:
    pose = capture.get("pose_after_mm_deg")
    joints = capture.get("joints_after_deg")
    if not pose or len(pose) != 6:
        raise ValueError(f"Capture has no six-value pose_after_mm_deg: {capture.get('name')}")
    T_base_link6 = pose_mm_deg_to_matrix(pose)
    return {
        "success": True,
        "joint_angles_deg": joints,
        "cart_pose_mm_deg": pose,
        "actual_pose_mm_deg": pose,
        "rt_target_joint_angles_deg": joints,
        "rt_target_cart_pose_mm_deg": pose,
        "status": scan_report.get("robot_status_before"),
        "safety_status": scan_report.get("robot_safety_status_before"),
        "T_base_link6": T_base_link6.tolist(),
        "T_base_link6_source": "active_scene_scan_pose_after_xyz_xyz_euler",
        "error": None,
    }


def build_observation(
    capture: dict[str, Any],
    scan_report: dict[str, Any],
    scan_dir: Path,
    output_dir: Path,
    config: dict[str, Any],
    extrinsic: dict[str, Any],
    extrinsic_path: Path,
) -> dict[str, Any]:
    capture_info = capture.get("capture", {})
    image_path = Path(capture_info.get("image", ""))
    if not image_path.is_absolute():
        image_path = scan_dir / image_path
    if not image_path.exists():
        raise FileNotFoundError(f"Capture image not found: {image_path}")

    obs_dir = output_dir / "observations" / f"{capture['index']:02d}_{capture['name']}"
    obs_dir.mkdir(parents=True, exist_ok=True)
    dst_image = obs_dir / "camera_rgb.png"
    shutil.copyfile(image_path, dst_image)

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "moves_robot": True,
        "note": f"active_scene_scan {scan_dir.name} {capture['name']}",
        "image": str(dst_image.relative_to(ROOT).as_posix()),
        "camera_config": config["camera"],
        "extrinsic_config_path": str(extrinsic_path),
        "extrinsic": extrinsic,
        "active_scan": {
            "scan_dir": str(scan_dir),
            "scan_timestamp": scan_report.get("timestamp"),
            "pattern": scan_report.get("parameters", {}).get("pattern"),
            "step_mm": scan_report.get("parameters", {}).get("step_mm"),
            "capture_index": capture.get("index"),
            "capture_name": capture.get("name"),
            "target_pose_mm_deg": capture.get("target_pose_mm_deg"),
            "pose_after_mm_deg": capture.get("pose_after_mm_deg"),
            "joints_after_deg": capture.get("joints_after_deg"),
        },
        "robot": robot_metadata_from_capture(capture, scan_report),
        "next_step": "Run active-scan 6D trocar pose estimation and sim-real consistency checks.",
    }
    write_json(obs_dir / "metadata.json", metadata)
    return {
        "index": capture["index"],
        "name": capture["name"],
        "observation_dir": str(obs_dir),
        "image": str(dst_image),
        "metadata": str(obs_dir / "metadata.json"),
    }


def run_pose_estimation(observation: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    pose_dir = output_dir / "pose_estimates" / Path(observation["observation_dir"]).name
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "estimate_trocar_pose_from_ring.py"),
        "--observation-dir",
        observation["observation_dir"],
        "--output-dir",
        str(pose_dir),
    ]
    result = subprocess.run(cmd, cwd=ROOT.parent, text=True, capture_output=True)
    report_path = pose_dir / "real_trocar_ring_pose_report.json"
    record = {
        "index": observation["index"],
        "name": observation["name"],
        "observation_dir": observation["observation_dir"],
        "pose_output_dir": str(pose_dir),
        "returncode": result.returncode,
        "success": result.returncode == 0 and report_path.exists(),
        "stdout": result.stdout,
        "stderr": result.stderr,
        "pose_report": str(report_path) if report_path.exists() else None,
    }
    if report_path.exists():
        try:
            pose_report = load_json(report_path)
            record["pose_camera_trocar_mm_deg"] = pose_report.get("pose_camera_trocar_mm_deg")
            record["pose_base_trocar_mm_deg"] = pose_report.get("pose_base_trocar_mm_deg")
            record["mean_reprojection_error_px"] = pose_report.get("metrics", {}).get("mean_reprojection_error_px")
            record["translation_camera_trocar_m"] = pose_report.get("translation_camera_trocar_m")
            record["trocar_axis_camera"] = pose_report.get("trocar_axis_camera")
        except Exception as exc:
            record["parse_error"] = f"{type(exc).__name__}: {exc}"
    return record


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Active Scene Scan Processing Report",
        "",
        f"Timestamp: `{report['timestamp']}`",
        f"Scan dir: `{report['scan_dir']}`",
        f"Observations: `{len(report['observations'])}`",
        f"Pose successes: `{sum(1 for item in report['pose_estimates'] if item['success'])}`",
        "",
        "## Pose Estimates",
        "",
        "| # | name | success | camera xyz mm | reproj px | pose report |",
        "|---:|---|---|---|---:|---|",
    ]
    for item in report["pose_estimates"]:
        xyz = ""
        t = item.get("translation_camera_trocar_m")
        if t and len(t) == 3:
            xyz = "[" + ", ".join(f"{float(v) * 1000.0:.3f}" for v in t) + "]"
        reproj = item.get("mean_reprojection_error_px")
        reproj_text = f"{float(reproj):.4f}" if reproj is not None else ""
        lines.append(
            f"| {item['index']} | `{item['name']}` | `{item['success']}` | "
            f"`{xyz}` | {reproj_text} | `{item.get('pose_report')}` |"
        )
    lines += [
        "",
        "## Failed Estimates",
        "",
    ]
    failed = [item for item in report["pose_estimates"] if not item["success"]]
    if not failed:
        lines.append("None.")
    else:
        for item in failed:
            lines.append(f"- `{item['name']}`: {item.get('stderr') or item.get('stdout')}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert an active scene scan into standard observations and estimate poses.")
    parser.add_argument("scan_dir", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--extrinsic", type=Path, default=DEFAULT_EXTRINSIC)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    scan_dir = args.scan_dir.resolve()
    scan_report = load_json(find_report(scan_dir))
    config = load_json(args.config)
    extrinsic = load_json(args.extrinsic)
    output_dir = args.output_dir or (DEFAULT_OUTPUT / scan_dir.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    observations = []
    pose_estimates = []
    for capture in scan_report.get("captures", []):
        observation = build_observation(
            capture=capture,
            scan_report=scan_report,
            scan_dir=scan_dir,
            output_dir=output_dir,
            config=config,
            extrinsic=extrinsic,
            extrinsic_path=args.extrinsic,
        )
        observations.append(observation)
        pose_estimates.append(run_pose_estimation(observation, output_dir))

    report = {
        "timestamp": datetime.now().isoformat(),
        "scan_dir": str(scan_dir),
        "output_dir": str(output_dir),
        "observations": observations,
        "pose_estimates": pose_estimates,
    }
    write_json(output_dir / "active_scene_scan_processing_report.json", report)
    write_markdown(output_dir / "active_scene_scan_processing_report.md", report)

    print("Active scene scan processed:", output_dir)
    print("Observations:", len(observations))
    print("Pose successes:", sum(1 for item in pose_estimates if item["success"]))


if __name__ == "__main__":
    main()
