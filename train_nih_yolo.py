from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "output" / "yolo_nih_hra_m2_dataset" / "data.yaml"
DEFAULT_PROJECT = ROOT / "output" / "yolo_nih_hra_m2_training"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and independently test the M2 NIH/HRA trocar detector."
    )
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--name", default="yolo11n_nih_hra_trocar")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is optional. Install it before M2 training: "
            "python -m pip install ultralytics"
        ) from exc

    model = YOLO(args.model)
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        project=str(args.project.resolve()),
        name=args.name,
    )
    best_weights = args.project.resolve() / args.name / "weights" / "best.pt"
    best_model = YOLO(str(best_weights))
    metrics = best_model.val(
        data=str(args.data.resolve()),
        split="test",
        imgsz=args.image_size,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
    )
    report = {
        "timestamp": datetime.now().isoformat(),
        "weights": str(best_weights),
        "split": "test",
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "per_class_map50_95": [float(value) for value in metrics.box.maps],
        "evidence_boundary": (
            "Metrics are from an independent synthetic MuJoCo test split and do "
            "not establish real-image or clinical generalization."
        ),
    }
    report_path = args.project.resolve() / "independent_test_metrics.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Best weights: {best_weights}")
    print(f"Independent test report: {report_path}")


if __name__ == "__main__":
    main()
