from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
DEFAULT_DRY_RUN = ROOT / "outputs" / "control_3d" / "dry_run_commands" / "dry_run_commands.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "real_image_shadow_replay"
DEFAULT_YOLO_MODEL = WORKSPACE / "outer_model.pt"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def resolve_path(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    normalized = path_text.replace("\\", "/")
    marker = "3d_modeling/"
    if marker in normalized:
        fragment = normalized[normalized.index(marker) :]
        candidate = WORKSPACE / fragment
        if candidate.exists():
            return candidate
    path = Path(path_text)
    if path.is_absolute() and path.exists():
        return path
    for root in (WORKSPACE, ROOT):
        candidate = root / path
        if candidate.exists():
            return candidate
    return WORKSPACE / path


def run_yolo(
    model: Any,
    image_path: Path,
    conf_threshold: float,
    target_classes: set[str] | None = None,
) -> dict[str, Any]:
    if model is None:
        return {
            "enabled": False,
            "status": "skipped",
            "reason": "YOLO model was not loaded.",
            "detections": [],
        }
    if not image_path.exists():
        return {
            "enabled": True,
            "status": "missing_image",
            "reason": f"Image not found: {image_path}",
            "detections": [],
        }

    try:
        import cv2
        import numpy as np

        image_bytes = np.fromfile(str(image_path), dtype=np.uint8)
        if image_bytes.size < 512:
            prefix = image_path.read_bytes()[:128]
            if prefix.startswith(b"version https://git-lfs.github.com/spec"):
                return {
                    "enabled": True,
                    "status": "git_lfs_pointer_missing",
                    "reason": f"Image content is not present locally; Git LFS pointer found: {image_path}",
                    "detections": [],
                }
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        if image is None:
            return {
                "enabled": True,
                "status": "image_decode_failed",
                "reason": f"Failed to decode image: {image_path}",
                "detections": [],
            }
        results = model(image, verbose=False, conf=conf_threshold)
    except Exception as exc:  # pragma: no cover - depends on optional runtime package
        return {
            "enabled": True,
            "status": "prediction_failed",
            "reason": str(exc),
            "detections": [],
        }
    names = getattr(model, "names", {}) or {}
    detections: list[dict[str, Any]] = []
    if results:
        boxes = getattr(results[0], "boxes", None)
        if boxes is not None:
            for box in boxes:
                xyxy = [float(v) for v in box.xyxy[0].tolist()]
                conf = float(box.conf[0].item()) if getattr(box, "conf", None) is not None else 0.0
                cls_id = int(box.cls[0].item()) if getattr(box, "cls", None) is not None else -1
                detections.append(
                    {
                        "class_id": cls_id,
                        "class_name": str(names.get(cls_id, cls_id)),
                        "confidence": conf,
                        "xyxy": xyxy,
                        "center_xy": [(xyxy[0] + xyxy[2]) * 0.5, (xyxy[1] + xyxy[3]) * 0.5],
                        "width_px": xyxy[2] - xyxy[0],
                        "height_px": xyxy[3] - xyxy[1],
                    }
                )
    detections.sort(key=lambda item: item["confidence"], reverse=True)
    if target_classes:
        target_detections = [
            item
            for item in detections
            if str(item["class_name"]).lower() in target_classes or str(item["class_id"]) in target_classes
        ]
    else:
        target_detections = detections
    best_any = detections[0] if detections else None
    best = target_detections[0] if target_detections else None
    if best and best["confidence"] >= conf_threshold:
        status = "ok"
    elif target_classes and detections:
        status = "target_class_missing_or_low_confidence"
    else:
        status = "low_confidence_or_missing"
    return {
        "enabled": True,
        "status": status,
        "confidence_threshold": conf_threshold,
        "target_classes": sorted(target_classes) if target_classes else [],
        "best_any_detection": best_any,
        "best_detection": best,
        "detection_count": len(detections),
        "target_detection_count": len(target_detections),
        "detections": detections,
    }


def write_detection_overlay(
    image_path: Path,
    detections: list[dict[str, Any]],
    target_classes: set[str],
    output_path: Path,
) -> None:
    try:
        import cv2
        import numpy as np

        image_bytes = np.fromfile(str(image_path), dtype=np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        if image is None:
            return
        for item in detections:
            class_name = str(item.get("class_name"))
            class_id = str(item.get("class_id"))
            is_target = (not target_classes) or class_name.lower() in target_classes or class_id in target_classes
            color = (0, 200, 0) if is_target else (80, 80, 255)
            x1, y1, x2, y2 = [int(round(v)) for v in item.get("xyxy", [0, 0, 0, 0])]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} {float(item.get('confidence', 0.0)):.2f}"
            cv2.putText(image, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ok, encoded = cv2.imencode(".jpg", image)
        if ok:
            encoded.tofile(str(output_path))
    except Exception:
        return


def build_shadow_record(record: dict[str, Any], yolo_result: dict[str, Any], require_yolo: bool) -> dict[str, Any]:
    image_path = resolve_path(record.get("image"))
    checks = record.get("checks") or {}
    geometry_ok = record.get("status") == "ok" and bool(checks.get("all_passed", False))
    yolo_ok = (not require_yolo) or yolo_result.get("status") == "ok"
    decision = "act" if geometry_ok and yolo_ok else "refuse"
    refusal_reasons = []
    if not geometry_ok:
        refusal_reasons.append("geometry_or_dry_run_check_failed")
    if require_yolo and not yolo_ok:
        refusal_reasons.append("yolo_detection_failed_or_low_confidence")
    if image_path is None or not image_path.exists():
        refusal_reasons.append("image_missing")
    return {
        "image": None if image_path is None else str(image_path),
        "pose_report": record.get("report"),
        "dry_run_status": record.get("status"),
        "current_errors": record.get("current_errors"),
        "dry_run_command": record.get("dry_run_command"),
        "dry_run_checks": checks,
        "yolo": yolo_result,
        "shadow_decision": {
            "decision": decision,
            "action_source": "geometry_dry_run_only",
            "residual_policy": "not_evaluated_in_real_image_shadow_v1",
            "would_move_robot": False,
            "refusal_reasons": refusal_reasons,
        },
    }


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    rows = []
    for idx, record in enumerate(records, start=1):
        err = record.get("current_errors") or {}
        yolo = record.get("yolo") or {}
        best = yolo.get("best_detection") or {}
        decision = record.get("shadow_decision") or {}
        rows.append(
            {
                "index": idx,
                "decision": decision.get("decision"),
                "image": record.get("image"),
                "dry_run_status": record.get("dry_run_status"),
                "yolo_status": yolo.get("status"),
                "yolo_confidence": best.get("confidence"),
                "yolo_class": best.get("class_name"),
                "yolo_detection_count": yolo.get("detection_count"),
                "yolo_target_detection_count": yolo.get("target_detection_count"),
                "lateral_error_mm": err.get("lateral_error_mm"),
                "depth_error_mm": err.get("depth_error_mm"),
                "axis_angle_error_deg": err.get("axis_angle_error_deg"),
                "weighted_3d_error_mm": err.get("weighted_3d_error_mm"),
                "refusal_reasons": ";".join(decision.get("refusal_reasons") or []),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# YOLO Real-Image Shadow Replay",
        "",
        f"Generated: `{report['timestamp']}`",
        "",
        "This report uses recorded real images and existing 3D dry-run candidates. It does not send robot commands.",
        "",
        "## Summary",
        "",
        f"- Evidence level: `{report['evidence_level']}`",
        f"- Dry-run input: `{report['dry_run_json']}`",
        f"- YOLO model: `{report['yolo_model']}`",
        f"- Require YOLO for act decision: `{report['require_yolo']}`",
        f"- Target classes: `{', '.join(report['target_classes']) if report['target_classes'] else 'any'}`",
        f"- Samples: `{report['sample_count']}`",
        f"- YOLO detections above threshold: `{report['yolo_ok_count']}`",
        f"- Shadow act decisions: `{report['act_count']}`",
        f"- Shadow refuse decisions: `{report['refuse_count']}`",
        "",
        "## Samples",
        "",
        "| # | decision | yolo | target class | target conf | any class | any conf | lateral mm | depth mm | axis deg | weighted mm | reason |",
        "|---:|---|---|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for idx, record in enumerate(report["records"], start=1):
        err = record.get("current_errors") or {}
        yolo = record.get("yolo") or {}
        best = yolo.get("best_detection") or {}
        best_any = yolo.get("best_any_detection") or {}
        decision = record.get("shadow_decision") or {}
        reasons = "; ".join(decision.get("refusal_reasons") or [])
        lines.append(
            "| {idx} | `{decision}` | `{yolo_status}` | {target_class} | {target_conf} | {any_class} | {any_conf} | {lat} | {depth} | {axis} | {weighted} | {reasons} |".format(
                idx=idx,
                decision=decision.get("decision"),
                yolo_status=yolo.get("status"),
                target_class="" if best.get("class_name") is None else f"`{best.get('class_name')}`",
                target_conf="" if best.get("confidence") is None else f"{best.get('confidence'):.3f}",
                any_class="" if best_any.get("class_name") is None else f"`{best_any.get('class_name')}`",
                any_conf="" if best_any.get("confidence") is None else f"{best_any.get('confidence'):.3f}",
                lat="" if err.get("lateral_error_mm") is None else f"{err.get('lateral_error_mm'):.3f}",
                depth="" if err.get("depth_error_mm") is None else f"{err.get('depth_error_mm'):.3f}",
                axis="" if err.get("axis_angle_error_deg") is None else f"{err.get('axis_angle_error_deg'):.3f}",
                weighted="" if err.get("weighted_3d_error_mm") is None else f"{err.get('weighted_3d_error_mm'):.3f}",
                reasons=reasons,
            )
        )
    lines += [
        "",
        "## Boundary",
        "",
        "- This is real-image replay / shadow-mode evidence only.",
        "- `act` means the offline gates would allow a candidate action; it does not execute motion.",
        "- Residual SAC is not evaluated in this v1 report unless a future real-image policy bridge is added.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO-gated real-image shadow replay from dry-run records without moving the robot.")
    parser.add_argument("--dry-run-json", type=Path, default=DEFAULT_DRY_RUN)
    parser.add_argument("--yolo-model", type=Path, default=DEFAULT_YOLO_MODEL)
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--require-yolo", action="store_true", help="Refuse shadow action if YOLO has no detection above threshold.")
    parser.add_argument(
        "--target-class",
        action="append",
        default=[],
        help="Require a detection of this class name or numeric id. Repeat for multiple acceptable classes.",
    )
    parser.add_argument("--save-overlays", action="store_true", help="Write per-sample YOLO detection overlays.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    dry_run = load_json(args.dry_run_json)
    model = None
    yolo_error = None
    if args.yolo_model:
        try:
            from ultralytics import YOLO

            model = YOLO(str(args.yolo_model))
        except Exception as exc:  # pragma: no cover - depends on optional runtime package
            yolo_error = str(exc)

    target_classes = {str(item).lower() for item in args.target_class if str(item).strip()}
    records = []
    for index, record in enumerate(dry_run.get("records", []), start=1):
        image_path = resolve_path(record.get("image"))
        if yolo_error is not None:
            yolo_result = {
                "enabled": True,
                "status": "model_load_failed",
                "reason": yolo_error,
                "detections": [],
            }
        elif image_path is None:
            yolo_result = {
                "enabled": model is not None,
                "status": "missing_image",
                "reason": "Record has no image path.",
                "detections": [],
            }
        else:
            yolo_result = run_yolo(model, image_path, args.conf_threshold, target_classes=target_classes)
        if args.save_overlays and image_path is not None and yolo_result.get("detections"):
            overlay_path = args.output_dir / "overlays" / f"sample_{index:03d}_yolo_overlay.jpg"
            write_detection_overlay(image_path, yolo_result["detections"], target_classes, overlay_path)
            yolo_result["overlay_path"] = str(overlay_path)
        records.append(build_shadow_record(record, yolo_result, args.require_yolo))

    act_count = sum(1 for item in records if item["shadow_decision"]["decision"] == "act")
    yolo_ok_count = sum(1 for item in records if item.get("yolo", {}).get("status") == "ok")
    report = {
        "timestamp": datetime.now().isoformat(),
        "evidence_level": "E4_real_image_shadow_no_robot_motion",
        "dry_run_json": str(args.dry_run_json),
        "yolo_model": str(args.yolo_model),
        "conf_threshold": args.conf_threshold,
        "target_classes": sorted(target_classes),
        "require_yolo": bool(args.require_yolo),
        "sample_count": len(records),
        "yolo_ok_count": yolo_ok_count,
        "act_count": act_count,
        "refuse_count": len(records) - act_count,
        "records": records,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "shadow_replay_report.json", report)
    write_csv(args.output_dir / "shadow_replay_report.csv", records)
    write_markdown(args.output_dir / "shadow_replay_report.md", report)
    print("YOLO shadow replay written:", args.output_dir)
    print("Samples:", len(records), "YOLO OK:", yolo_ok_count, "Act:", act_count, "Refuse:", len(records) - act_count)


if __name__ == "__main__":
    main()
