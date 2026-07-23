from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return PROJECT_ROOT / value


@dataclass
class Box:
    cls: int
    xyxy: np.ndarray
    conf: float = 1.0

    @property
    def center(self) -> np.ndarray:
        x1, y1, x2, y2 = self.xyxy
        return np.asarray([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float64)

    @property
    def size(self) -> np.ndarray:
        x1, y1, x2, y2 = self.xyxy
        return np.asarray([max(0.0, x2 - x1), max(0.0, y2 - y1)], dtype=np.float64)


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Dataset YAML must be a mapping: {path}")
    return data


def _class_names(data: dict[str, Any], model_names: Any) -> dict[int, str]:
    names = data.get("names", {})
    if isinstance(names, list):
        out = {idx: str(name) for idx, name in enumerate(names)}
    elif isinstance(names, dict):
        out = {int(idx): str(name) for idx, name in names.items()}
    else:
        out = {}
    if isinstance(model_names, dict):
        for idx, name in model_names.items():
            out.setdefault(int(idx), str(name))
    return out


def _resolve_split_dirs(
    dataset_yaml: Path,
    data: dict[str, Any],
    split: str,
    images_dir: str | None,
    labels_dir: str | None,
) -> tuple[Path, Path]:
    if images_dir:
        image_dir = resolve_project_path(images_dir)
    else:
        split_value = data.get(split)
        if not split_value:
            raise ValueError(f"Split '{split}' not found in {dataset_yaml}")
        base = resolve_project_path(data["path"]) if data.get("path") else dataset_yaml.parent
        image_dir = resolve_project_path(split_value) if Path(str(split_value)).is_absolute() else base / str(split_value)
        if not image_dir.exists():
            fallback = PROJECT_ROOT / "images" / split
            if fallback.exists():
                image_dir = fallback

    if labels_dir:
        label_dir = resolve_project_path(labels_dir)
    else:
        if image_dir.parent.name == "images":
            label_dir = image_dir.parent.parent / "labels" / image_dir.name
        else:
            label_dir = image_dir.parent / "labels"
        if label_dir == image_dir or not label_dir.exists():
            fallback = PROJECT_ROOT / "labels" / split
            if fallback.exists():
                label_dir = fallback
    return image_dir, label_dir


def _list_images(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    return sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def _list_labels(label_dir: Path) -> list[Path]:
    if not label_dir.exists():
        return []
    return sorted(path for path in label_dir.iterdir() if path.is_file() and path.suffix.lower() == ".txt")


def _read_labels(path: Path, width: int, height: int) -> list[Box]:
    if not path.exists():
        return []
    boxes: list[Box] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"Invalid YOLO label at {path}:{line_no}: {line}")
        cls = int(float(parts[0]))
        cx, cy, bw, bh = [float(value) for value in parts[1:5]]
        x1 = (cx - bw * 0.5) * width
        y1 = (cy - bh * 0.5) * height
        x2 = (cx + bw * 0.5) * width
        y2 = (cy + bh * 0.5) * height
        boxes.append(Box(cls=cls, xyxy=np.asarray([x1, y1, x2, y2], dtype=np.float64)))
    return boxes


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return 0.0 if union <= 1e-9 else inter / union


def _match_boxes(
    gt_boxes: list[Box],
    pred_boxes: list[Box],
    iou_threshold: float,
    center_threshold_px: float,
) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    matches: list[dict[str, Any]] = []
    unmatched_gt = set(range(len(gt_boxes)))
    unmatched_pred = set(range(len(pred_boxes)))
    pred_order = sorted(range(len(pred_boxes)), key=lambda idx: pred_boxes[idx].conf, reverse=True)

    for pred_idx in pred_order:
        pred = pred_boxes[pred_idx]
        best_gt = None
        best_iou = 0.0
        for gt_idx in unmatched_gt:
            gt = gt_boxes[gt_idx]
            if gt.cls != pred.cls:
                continue
            value = _iou(gt.xyxy, pred.xyxy)
            center_error = float(np.linalg.norm(gt.center - pred.center))
            center_pass = center_threshold_px > 0.0 and center_error <= center_threshold_px
            if value > best_iou or (center_pass and best_gt is None):
                best_iou = value
                best_gt = gt_idx
        if best_gt is not None:
            gt = gt_boxes[best_gt]
            center_error = float(np.linalg.norm(gt.center - pred.center))
            iou_pass = best_iou >= iou_threshold
            center_pass = center_threshold_px > 0.0 and center_error <= center_threshold_px
            if not (iou_pass or center_pass):
                continue
            gt_size = gt.size
            pred_size = pred.size
            size_error = np.abs(gt_size - pred_size)
            rel_size_error = size_error / np.maximum(gt_size, 1.0)
            matches.append(
                {
                    "class_id": int(pred.cls),
                    "iou": float(best_iou),
                    "match_reason": "iou" if iou_pass else "center",
                    "confidence": float(pred.conf),
                    "center_error_px": center_error,
                    "width_error_px": float(size_error[0]),
                    "height_error_px": float(size_error[1]),
                    "relative_width_error": float(rel_size_error[0]),
                    "relative_height_error": float(rel_size_error[1]),
                }
            )
            unmatched_gt.remove(best_gt)
            unmatched_pred.remove(pred_idx)
    return matches, sorted(unmatched_gt), sorted(unmatched_pred)


def _parse_class_map(value: str) -> dict[int, int]:
    if not value:
        return {}
    mapping: dict[int, int] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        src, dst = item.split(":", maxsplit=1)
        mapping[int(src.strip())] = int(dst.strip())
    return mapping


def _predict(
    model: Any,
    image_path: Path,
    imgsz: int,
    conf: float,
    iou: float,
    pred_class_map: dict[int, int],
) -> tuple[list[Box], float]:
    start = time.perf_counter()
    results = model.predict(source=str(image_path), imgsz=imgsz, conf=conf, iou=iou, verbose=False)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    result = results[0]
    boxes: list[Box] = []
    if result.boxes is None:
        return boxes, elapsed_ms
    xyxy = result.boxes.xyxy.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy().astype(int)
    scores = result.boxes.conf.cpu().numpy()
    for cls_id, coords, score in zip(cls, xyxy, scores):
        mapped_cls = pred_class_map.get(int(cls_id), int(cls_id))
        boxes.append(Box(cls=mapped_cls, xyxy=np.asarray(coords, dtype=np.float64), conf=float(score)))
    return boxes, elapsed_ms


def _draw_boxes(image: np.ndarray, gt_boxes: list[Box], pred_boxes: list[Box], class_names: dict[int, str]) -> np.ndarray:
    canvas = image.copy()
    for box in gt_boxes:
        x1, y1, x2, y2 = box.xyxy.astype(int)
        label = f"GT {class_names.get(box.cls, box.cls)}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (30, 180, 30), 2)
        cv2.putText(canvas, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 180, 30), 1)
    for box in pred_boxes:
        x1, y1, x2, y2 = box.xyxy.astype(int)
        label = f"{class_names.get(box.cls, box.cls)} {box.conf:.2f}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (40, 80, 230), 2)
        cv2.putText(canvas, label, (x1, min(canvas.shape[0] - 5, y2 + 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 80, 230), 1)
    return canvas


def _empty_class_stats(class_names: dict[int, str]) -> dict[int, dict[str, Any]]:
    return {
        cls_id: {
            "class_id": cls_id,
            "class_name": class_names.get(cls_id, str(cls_id)),
            "gt": 0,
            "pred": 0,
            "matched": 0,
            "unmatched_gt": 0,
            "unmatched_pred": 0,
            "confidences": [],
            "ious": [],
            "center_errors_px": [],
            "relative_size_errors": [],
        }
        for cls_id in sorted(class_names)
    }


def _finalize_stats(stats: dict[int, dict[str, Any]], image_count: int) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for cls_id, item in stats.items():
        gt = int(item["gt"])
        pred = int(item["pred"])
        matched = int(item["matched"])
        confidences = item["confidences"]
        ious = item["ious"]
        center_errors = item["center_errors_px"]
        rel_sizes = item["relative_size_errors"]
        precision = matched / pred if pred else 0.0
        recall = matched / gt if gt else 0.0
        summary[str(cls_id)] = {
            "class_id": cls_id,
            "class_name": item["class_name"],
            "gt": gt,
            "pred": pred,
            "matched": matched,
            "unmatched_gt": int(item["unmatched_gt"]),
            "unmatched_pred": int(item["unmatched_pred"]),
            "precision": precision,
            "recall": recall,
            "dropout_probability": 1.0 - recall if gt else 0.0,
            "false_positive_per_image": int(item["unmatched_pred"]) / max(1, image_count),
            "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
            "mean_iou": float(np.mean(ious)) if ious else 0.0,
            "mean_center_error_px": float(np.mean(center_errors)) if center_errors else 0.0,
            "std_center_error_px": float(np.std(center_errors)) if center_errors else 0.0,
            "p95_center_error_px": float(np.percentile(center_errors, 95)) if center_errors else 0.0,
            "mean_relative_size_error": float(np.mean(rel_sizes)) if rel_sizes else 0.0,
            "std_relative_size_error": float(np.std(rel_sizes)) if rel_sizes else 0.0,
        }
    return summary


def _percentile(values: list[float], pct: float) -> float:
    return float(np.percentile(values, pct)) if values else 0.0


def _noise_model(
    *,
    audit_summary: dict[str, Any],
    class_summary: dict[str, dict[str, Any]],
    all_matches: list[dict[str, Any]],
    px_to_mm: float,
    axis_deg_per_px: float,
    bbox_scale_to_depth_mm: float,
    noise_classes: list[int],
) -> dict[str, Any]:
    center_errors = [float(item["center_error_px"]) for item in all_matches]
    rel_size_errors = [
        0.5 * (abs(float(item["relative_width_error"])) + abs(float(item["relative_height_error"])))
        for item in all_matches
    ]
    confidences = [float(item["confidence"]) for item in all_matches]
    required_dropout = [
        float(class_summary[str(cls)]["dropout_probability"])
        for cls in noise_classes
        if str(cls) in class_summary and int(class_summary[str(cls)]["gt"]) > 0
    ]
    dropout_probability = max(required_dropout) if required_dropout else 0.0
    center_std = float(np.std(center_errors)) if center_errors else 0.0
    center_p95 = _percentile(center_errors, 95)
    rel_size_std = float(np.std(rel_size_errors)) if rel_size_errors else 0.0
    rel_size_p95 = _percentile(rel_size_errors, 95)
    lateral_std = max(0.02, center_std * px_to_mm)
    depth_std = max(0.05, rel_size_std * bbox_scale_to_depth_mm)
    axis_std = max(0.05, center_std * axis_deg_per_px)
    return {
        "schema_version": 1,
        "source": {
            "audit_summary": audit_summary.get("summary_path", ""),
            "dataset_name": audit_summary["dataset_name"],
            "model_path": audit_summary["model_path"],
            "split": audit_summary["split"],
            "image_count": audit_summary["image_count"],
        },
        "class_dropout_probability": {
            cls_id: float(item["dropout_probability"]) for cls_id, item in class_summary.items()
        },
        "class_false_positive_per_image": {
            cls_id: float(item["false_positive_per_image"]) for cls_id, item in class_summary.items()
        },
        "confidence": {
            "mean": float(np.mean(confidences)) if confidences else 0.5,
            "std": float(np.std(confidences)) if confidences else 0.2,
            "p05": _percentile(confidences, 5) if confidences else 0.1,
            "p50": _percentile(confidences, 50) if confidences else 0.5,
            "p95": _percentile(confidences, 95) if confidences else 0.9,
        },
        "bbox_center_error_px": {
            "mean": float(np.mean(center_errors)) if center_errors else 0.0,
            "std": center_std,
            "p95": center_p95,
        },
        "bbox_relative_size_error": {
            "mean": float(np.mean(rel_size_errors)) if rel_size_errors else 0.0,
            "std": rel_size_std,
            "p95": rel_size_p95,
        },
        "task_space_proxy": {
            "lateral_noise_mm_std": lateral_std,
            "depth_noise_mm_std": depth_std,
            "axis_noise_deg_std": axis_std,
            "dropout_probability": dropout_probability,
            "confidence_mean": float(np.mean(confidences)) if confidences else 0.5,
            "confidence_std": float(np.std(confidences)) if confidences else 0.2,
            "uncertainty_from_center_px_p95": center_p95,
            "px_to_mm": px_to_mm,
            "axis_deg_per_px": axis_deg_per_px,
            "bbox_scale_to_depth_mm": bbox_scale_to_depth_mm,
            "noise_classes": noise_classes,
        },
    }


def audit_dataset(args: argparse.Namespace) -> dict[str, Any]:
    from ultralytics import YOLO
    import ultralytics

    dataset_yaml = resolve_project_path(args.dataset_yaml)
    model_path = resolve_project_path(args.model)
    data = _read_yaml(dataset_yaml)
    model = YOLO(str(model_path))
    pred_class_map = _parse_class_map(args.pred_class_map)
    noise_classes = [int(item.strip()) for item in args.noise_classes.split(",") if item.strip()]
    class_names = _class_names(data, getattr(model, "names", {}))
    image_dir, label_dir = _resolve_split_dirs(dataset_yaml, data, args.split, args.images_dir, args.labels_dir)
    images = _list_images(image_dir)
    labels = _list_labels(label_dir)
    label_stems = {path.stem for path in labels}
    image_stems = {path.stem for path in images}

    output_dir = resolve_project_path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "3d_modeling" / "outputs" / "yolo_detector_audit" / f"{args.dataset_name}_{args.split}_{_now_stamp()}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "visualizations"
    if args.save_visualizations > 0:
        vis_dir.mkdir(parents=True, exist_ok=True)

    consistency = {
        "dataset_yaml": str(dataset_yaml),
        "split": args.split,
        "image_dir": str(image_dir),
        "label_dir": str(label_dir),
        "image_count": len(images),
        "label_count": len(labels),
        "missing_label_files": sorted(str(path.name) for path in images if path.stem not in label_stems),
        "orphan_label_files": sorted(str(path.name) for path in labels if path.stem not in image_stems),
    }

    stats = _empty_class_stats(class_names)
    all_matches: list[dict[str, Any]] = []
    per_image_path = output_dir / "per_image_results.jsonl"
    latencies: list[float] = []
    failure_cases: list[dict[str, Any]] = []

    with per_image_path.open("w", encoding="utf-8") as stream:
        for index, image_path in enumerate(images):
            image = cv2.imread(str(image_path))
            if image is None:
                failure_cases.append({"image": str(image_path), "reason": "image_read_failed"})
                continue
            height, width = image.shape[:2]
            gt_boxes = _read_labels(label_dir / f"{image_path.stem}.txt", width, height)
            pred_boxes, latency_ms = _predict(model, image_path, args.imgsz, args.conf, args.iou, pred_class_map)
            latencies.append(latency_ms)

            for box in gt_boxes:
                stats.setdefault(box.cls, _empty_class_stats({box.cls: class_names.get(box.cls, str(box.cls))})[box.cls])
                stats[box.cls]["gt"] += 1
            for box in pred_boxes:
                stats.setdefault(box.cls, _empty_class_stats({box.cls: class_names.get(box.cls, str(box.cls))})[box.cls])
                stats[box.cls]["pred"] += 1

            matches, unmatched_gt, unmatched_pred = _match_boxes(
                gt_boxes,
                pred_boxes,
                args.match_iou,
                args.match_center_px,
            )
            for match in matches:
                item = stats[match["class_id"]]
                item["matched"] += 1
                item["confidences"].append(match["confidence"])
                item["ious"].append(match["iou"])
                item["center_errors_px"].append(match["center_error_px"])
                item["relative_size_errors"].append(
                    0.5 * (abs(match["relative_width_error"]) + abs(match["relative_height_error"]))
                )
            for gt_idx in unmatched_gt:
                stats[gt_boxes[gt_idx].cls]["unmatched_gt"] += 1
            for pred_idx in unmatched_pred:
                stats[pred_boxes[pred_idx].cls]["unmatched_pred"] += 1

            gt_classes = sorted({box.cls for box in gt_boxes})
            pred_classes = sorted({box.cls for box in pred_boxes})
            if unmatched_gt or (not pred_boxes and gt_boxes):
                failure_cases.append(
                    {
                        "image": str(image_path),
                        "gt_classes": gt_classes,
                        "pred_classes": pred_classes,
                        "unmatched_gt": [gt_boxes[idx].cls for idx in unmatched_gt],
                    }
                )

            record = {
                "image": str(image_path),
                "width": width,
                "height": height,
                "latency_ms": latency_ms,
                "gt_count": len(gt_boxes),
                "pred_count": len(pred_boxes),
                "matches": matches,
                "unmatched_gt_classes": [gt_boxes[idx].cls for idx in unmatched_gt],
                "unmatched_pred_classes": [pred_boxes[idx].cls for idx in unmatched_pred],
                "predictions": [
                    {"class_id": box.cls, "confidence": box.conf, "xyxy": [float(v) for v in box.xyxy]}
                    for box in pred_boxes
                ],
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

            if index < args.save_visualizations:
                canvas = _draw_boxes(image, gt_boxes, pred_boxes, class_names)
                cv2.imwrite(str(vis_dir / f"{image_path.stem}_audit.jpg"), canvas)

    class_summary = _finalize_stats(stats, len(images))
    image_success_inner_outer = 0
    inner_outer_images = 0
    for line in per_image_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        gt = set(record["unmatched_gt_classes"])
        pred_classes = {item["class_id"] for item in record["predictions"]}
        has_gt_inner_outer = any(item["class_id"] in {0, 1} for item in record["predictions"]) or bool({0, 1} & gt)
        if has_gt_inner_outer:
            inner_outer_images += 1
            if 0 in pred_classes and 1 in pred_classes and not ({0, 1} & gt):
                image_success_inner_outer += 1

    summary_path = output_dir / "audit_summary.json"
    summary = {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_name": args.dataset_name,
        "dataset_yaml": str(dataset_yaml),
        "split": args.split,
        "model_path": str(model_path),
        "model_family_note": args.model_family_note,
        "pred_class_map": {str(k): v for k, v in sorted(pred_class_map.items())},
        "ultralytics_version": ultralytics.__version__,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "nms_iou": args.iou,
        "match_iou": args.match_iou,
        "match_center_px": args.match_center_px,
        "class_names": {str(k): v for k, v in sorted(class_names.items())},
        "image_count": len(images),
        "label_count": len(labels),
        "latency_ms": {
            "mean": float(np.mean(latencies)) if latencies else math.nan,
            "std": float(np.std(latencies)) if latencies else math.nan,
            "p95": float(np.percentile(latencies, 95)) if latencies else math.nan,
        },
        "inner_outer_image_success": {
            "eligible_images": inner_outer_images,
            "success_count": image_success_inner_outer,
            "success_rate": image_success_inner_outer / inner_outer_images if inner_outer_images else 0.0,
        },
        "classes": class_summary,
        "failure_case_count": len(failure_cases),
        "failure_cases_preview": failure_cases[:25],
        "summary_path": str(summary_path),
        "per_image_results": str(per_image_path),
    }

    consistency_path = output_dir / "dataset_consistency.json"
    manifest_path = output_dir / "detector_manifest.json"
    noise_path = output_dir / "yolo_noise_model.json"
    summary["consistency_report"] = str(consistency_path)
    summary["noise_model"] = str(noise_path)

    all_matches = []
    for line in per_image_path.read_text(encoding="utf-8").splitlines():
        all_matches.extend(json.loads(line)["matches"])
    noise = _noise_model(
        audit_summary=summary,
        class_summary=class_summary,
        all_matches=all_matches,
        px_to_mm=args.px_to_mm,
        axis_deg_per_px=args.axis_deg_per_px,
        bbox_scale_to_depth_mm=args.bbox_scale_to_depth_mm,
        noise_classes=noise_classes,
    )
    manifest = {
        "dataset_yaml": str(dataset_yaml),
        "dataset_name": args.dataset_name,
        "split": args.split,
        "model_path": str(model_path),
        "model_family_note": args.model_family_note,
        "pred_class_map": {str(k): v for k, v in sorted(pred_class_map.items())},
        "class_names": summary["class_names"],
        "conf": args.conf,
        "nms_iou": args.iou,
        "imgsz": args.imgsz,
        "output_dir": str(output_dir),
    }

    consistency_path.write_text(json.dumps(consistency, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    noise_path.write_text(json.dumps(noise, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a YOLO detector on trocar datasets and fit a perception-noise model.")
    parser.add_argument("--model", default="rz_train_runs/rz_trocar_v11n/weights/best.pt")
    parser.add_argument("--dataset-yaml", default="rz_dataset/data.yaml")
    parser.add_argument("--dataset-name", default="rz_dataset")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--images-dir", default=None)
    parser.add_argument("--labels-dir", default=None)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--match-center-px", type=float, default=0.0, help="Also match same-class boxes when center error is within this many pixels.")
    parser.add_argument("--save-visualizations", type=int, default=12)
    parser.add_argument("--px-to-mm", type=float, default=0.05)
    parser.add_argument("--axis-deg-per-px", type=float, default=0.02)
    parser.add_argument("--bbox-scale-to-depth-mm", type=float, default=5.0)
    parser.add_argument("--model-family-note", default="YOLO11 fine-tuned on project trocar classes")
    parser.add_argument("--pred-class-map", default="", help="Optional prediction class remap, e.g. '0:1,1:0'.")
    parser.add_argument("--noise-classes", default="0,1", help="Class ids used to estimate task-space dropout, e.g. '1' for outer-ring center.")
    args = parser.parse_args()

    summary = audit_dataset(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
