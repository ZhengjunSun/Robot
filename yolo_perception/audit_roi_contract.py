from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class SplitPaths:
    images: Path
    labels: Path
    requested_images: Path
    requested_labels: Path
    fallback_reason: str | None = None


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return data


def class_names(data: dict[str, Any]) -> list[str]:
    names = data.get("names", [])
    if isinstance(names, list):
        return [str(item) for item in names]
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=lambda item: int(item))]
    return []


def resolve_split(dataset_yaml: Path, data: dict[str, Any], split: str) -> SplitPaths:
    base = resolve_project_path(data["path"]) if data.get("path") else dataset_yaml.parent
    split_value = data.get(split)
    if not split_value:
        missing_images = base / "missing_split"
        missing_labels = base / "labels" / split
        return SplitPaths(
            images=missing_images,
            labels=missing_labels,
            requested_images=missing_images,
            requested_labels=missing_labels,
            fallback_reason="missing_split_value",
        )
    image_dir = Path(str(split_value))
    if not image_dir.is_absolute():
        image_dir = base / image_dir
    requested_image_dir = image_dir
    fallback_reason = None
    if not image_dir.exists():
        fallback = PROJECT_ROOT / "images" / split
        if fallback.exists():
            image_dir = fallback
            fallback_reason = "image_dir_fallback_to_workspace_images"
    if image_dir.parent.name == "images":
        label_dir = image_dir.parent.parent / "labels" / image_dir.name
    else:
        label_dir = image_dir.parent / "labels"
    requested_label_dir = label_dir
    if not label_dir.exists():
        fallback_label = PROJECT_ROOT / "labels" / split
        if fallback_label.exists():
            label_dir = fallback_label
            fallback_reason = fallback_reason or "label_dir_fallback_to_workspace_labels"
    return SplitPaths(
        images=image_dir,
        labels=label_dir,
        requested_images=requested_image_dir,
        requested_labels=requested_label_dir,
        fallback_reason=fallback_reason,
    )


def list_files(path: Path, suffixes: set[str]) -> list[Path]:
    if not path.exists():
        return []
    return sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() in suffixes)


def read_label_stats(label_files: list[Path], valid_class_count: int) -> dict[str, Any]:
    class_histogram: dict[str, int] = {}
    invalid_lines: list[dict[str, Any]] = []
    empty_label_files = 0
    for label_path in label_files:
        lines = label_path.read_text(encoding="utf-8").splitlines()
        nonempty = [line for line in lines if line.strip()]
        if not nonempty:
            empty_label_files += 1
        for line_no, line in enumerate(nonempty, start=1):
            parts = line.split()
            if len(parts) < 5:
                invalid_lines.append({"file": str(label_path), "line": line_no, "reason": "too_few_columns"})
                continue
            try:
                class_id = int(float(parts[0]))
                coords = [float(value) for value in parts[1:5]]
            except ValueError:
                invalid_lines.append({"file": str(label_path), "line": line_no, "reason": "parse_error"})
                continue
            class_histogram[str(class_id)] = class_histogram.get(str(class_id), 0) + 1
            if class_id < 0 or class_id >= valid_class_count:
                invalid_lines.append({"file": str(label_path), "line": line_no, "reason": "class_id_out_of_range"})
            cx, cy, width, height = coords
            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < width <= 1.0 and 0.0 < height <= 1.0):
                invalid_lines.append({"file": str(label_path), "line": line_no, "reason": "bbox_out_of_range"})
    return {
        "class_histogram": class_histogram,
        "empty_label_files": empty_label_files,
        "invalid_line_count": len(invalid_lines),
        "invalid_lines_preview": invalid_lines[:20],
    }


def audit_dataset(dataset_key: str, spec: dict[str, Any]) -> dict[str, Any]:
    dataset_yaml = resolve_project_path(spec["dataset_yaml"])
    result: dict[str, Any] = {
        "dataset_key": dataset_key,
        "role": spec.get("role"),
        "dataset_yaml": str(dataset_yaml),
        "expected_schema": spec.get("expected_schema"),
        "expected_names": spec.get("expected_names", []),
        "label_status": spec.get("label_status"),
        "evidence_boundary": spec.get("evidence_boundary"),
        "status": "ok",
        "warnings": [],
    }
    if not dataset_yaml.exists():
        result["status"] = "missing_dataset_yaml"
        return result

    data = read_yaml(dataset_yaml)
    names = class_names(data)
    expected_names = [str(item) for item in spec.get("expected_names", [])]
    if expected_names and names != expected_names:
        result["warnings"].append(
            {
                "type": "class_names_mismatch",
                "expected": expected_names,
                "actual": names,
            }
        )
    if int(data.get("nc", len(names))) != len(names):
        result["warnings"].append(
            {
                "type": "nc_names_count_mismatch",
                "nc": data.get("nc"),
                "name_count": len(names),
            }
        )

    result["class_names"] = names
    result["nc"] = data.get("nc")
    result["splits"] = {}
    resolved_splits: dict[str, SplitPaths] = {}
    for split in ("train", "val"):
        paths = resolve_split(dataset_yaml, data, split)
        resolved_splits[split] = paths
        image_files = list_files(paths.images, IMAGE_SUFFIXES)
        label_files = list_files(paths.labels, {".txt"})
        label_files = [item for item in label_files if item.name.lower() != "classes.txt"]
        label_stems = {path.stem for path in label_files}
        image_stems = {path.stem for path in image_files}
        split_result = {
            "image_dir": str(paths.images),
            "label_dir": str(paths.labels),
            "requested_image_dir": str(paths.requested_images),
            "requested_label_dir": str(paths.requested_labels),
            "fallback_reason": paths.fallback_reason,
            "image_dir_exists": paths.images.exists(),
            "label_dir_exists": paths.labels.exists(),
            "image_count": len(image_files),
            "label_count": len(label_files),
            "missing_label_files": sorted(path.name for path in image_files if path.stem not in label_stems),
            "orphan_label_files": sorted(path.name for path in label_files if path.stem not in image_stems),
            **read_label_stats(label_files, len(names)),
        }
        if not paths.images.exists():
            result["warnings"].append({"type": "missing_image_dir", "split": split, "path": str(paths.images)})
        if paths.fallback_reason:
            result["warnings"].append(
                {
                    "type": "dataset_path_fallback",
                    "split": split,
                    "reason": paths.fallback_reason,
                    "requested_image_dir": str(paths.requested_images),
                    "resolved_image_dir": str(paths.images),
                }
            )
        if image_files and len(split_result["missing_label_files"]) > 0:
            result["warnings"].append(
                {
                    "type": "missing_label_files",
                    "split": split,
                    "count": len(split_result["missing_label_files"]),
                }
            )
        if len(split_result["orphan_label_files"]) > 0:
            result["warnings"].append(
                {
                    "type": "orphan_label_files",
                    "split": split,
                    "count": len(split_result["orphan_label_files"]),
                }
            )
        if split_result["invalid_line_count"] > 0:
            result["warnings"].append(
                {
                    "type": "invalid_label_lines",
                    "split": split,
                    "count": split_result["invalid_line_count"],
                }
            )
        result["splits"][split] = split_result

    train_paths = resolved_splits.get("train")
    val_paths = resolved_splits.get("val")
    if train_paths and val_paths and train_paths.images.resolve() == val_paths.images.resolve():
        result["warnings"].append(
            {
                "type": "train_val_same_image_dir",
                "path": str(train_paths.images),
                "boundary": "same-frame results are not held-out detector evidence",
            }
        )

    if result["warnings"]:
        result["status"] = "warning"
    return result


def audit_models(models: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, spec in models.items():
        path = resolve_project_path(spec["path"])
        out[key] = {
            "path": str(path),
            "role": spec.get("role"),
            "status_note": spec.get("status"),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else None,
        }
    return out


def build_report(contract: dict[str, Any]) -> dict[str, Any]:
    datasets = {
        key: audit_dataset(key, spec)
        for key, spec in (contract.get("datasets") or {}).items()
    }
    models = audit_models(contract.get("models") or {})
    warning_count = sum(len(item.get("warnings", [])) for item in datasets.values())
    missing_model_count = sum(1 for item in models.values() if not item["exists"])
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "contract_name": contract.get("name"),
        "schema_version": contract.get("schema_version"),
        "active_contract": contract.get("active_contract"),
        "dataset_count": len(datasets),
        "dataset_warning_count": warning_count,
        "missing_model_count": missing_model_count,
        "datasets": datasets,
        "models": models,
        "next_steps": contract.get("next_steps", []),
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# YOLO Port ROI Contract Audit",
        "",
        f"Generated: `{report['created_at']}`",
        "",
        "This audit checks the current YOLO perception contract for residual-RL use. It does not train a model.",
        "",
        "## Active Contract",
        "",
        f"- Contract: `{report['contract_name']}`",
        f"- Target classes: `{', '.join(report['active_contract'].get('target_classes', []))}`",
        f"- Confidence threshold: `{report['active_contract'].get('confidence_threshold')}`",
        f"- Dataset warnings: `{report['dataset_warning_count']}`",
        f"- Missing models: `{report['missing_model_count']}`",
        "",
        "## Datasets",
        "",
        "| dataset | role | status | schema | train images/labels | val images/labels | warnings |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for key, item in report["datasets"].items():
        train = item.get("splits", {}).get("train", {})
        val = item.get("splits", {}).get("val", {})
        warning_types = ", ".join(warn.get("type", "warning") for warn in item.get("warnings", []))
        lines.append(
            f"| `{key}` | `{item.get('role')}` | `{item.get('status')}` | `{item.get('expected_schema')}` | "
            f"{train.get('image_count', 0)}/{train.get('label_count', 0)} | "
            f"{val.get('image_count', 0)}/{val.get('label_count', 0)} | {warning_types} |"
        )
    lines += [
        "",
        "## Label Histograms",
        "",
        "| dataset | split | class histogram | missing labels | orphan labels | invalid lines |",
        "|---|---|---|---:|---:|---:|",
    ]
    for key, item in report["datasets"].items():
        for split, split_item in item.get("splits", {}).items():
            lines.append(
                f"| `{key}` | `{split}` | `{json.dumps(split_item.get('class_histogram', {}), ensure_ascii=False)}` | "
                f"{len(split_item.get('missing_label_files', []))} | "
                f"{len(split_item.get('orphan_label_files', []))} | "
                f"{split_item.get('invalid_line_count', 0)} |"
            )
    lines += [
        "",
        "## Models",
        "",
        "| model | role | exists | status note |",
        "|---|---|---:|---|",
    ]
    for key, item in report["models"].items():
        lines.append(
            f"| `{key}` | `{item.get('role')}` | `{item.get('exists')}` | `{item.get('status_note')}` |"
        )
    lines += [
        "",
        "## Boundary",
        "",
        "- The active perception target is a single `trocar`/port ROI, not full `in/out/trocar` semantic detection.",
        "- Detector proposals are not ground truth until accepted or adjusted by manual review.",
        "- Use this audit before fitting a YOLO noise model or launching residual-RL experiments with perception features.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the YOLO port ROI contract and dataset consistency.")
    parser.add_argument("--contract", type=Path, default=PROJECT_ROOT / "config" / "yolo_port_roi_contract.yaml")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "3d_modeling" / "outputs" / "yolo_contract_audit" / "latest",
    )
    args = parser.parse_args()

    contract_path = resolve_project_path(args.contract)
    output_dir = resolve_project_path(args.output_dir)
    contract = read_yaml(contract_path)
    report = build_report(contract)
    report["contract_path"] = str(contract_path)
    report["output_dir"] = str(output_dir)
    write_json(output_dir / "yolo_port_roi_contract_audit.json", report)
    write_markdown(output_dir / "yolo_port_roi_contract_audit.md", report)
    print("YOLO port ROI contract audit written:", output_dir)
    print("Datasets:", report["dataset_count"], "warnings:", report["dataset_warning_count"], "missing models:", report["missing_model_count"])


if __name__ == "__main__":
    main()
