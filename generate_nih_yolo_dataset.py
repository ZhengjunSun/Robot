from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from real_3d_alignment.mujoco_visual_env import MujocoCoarseAlignmentPlant
from real_3d_alignment.nih_baseline import NIH_HRA_EYE_SOURCE, NIH_HRA_SCENE


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "output" / "yolo_nih_hra_m2_dataset"


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows, columns = np.nonzero(mask)
    if rows.size == 0:
        return None
    return (
        int(np.min(columns)),
        int(np.min(rows)),
        int(np.max(columns)) + 1,
        int(np.max(rows)) + 1,
    )


def yolo_line(
    bbox: tuple[int, int, int, int],
    *,
    image_width: int,
    image_height: int,
) -> str:
    x1, y1, x2, y2 = bbox
    center_x = 0.5 * (x1 + x2) / image_width
    center_y = 0.5 * (y1 + y2) / image_height
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height
    return (
        f"0 {center_x:.8f} {center_y:.8f} "
        f"{width:.8f} {height:.8f}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate privileged-label NIH/HRA MuJoCo images for M2 YOLO."
    )
    parser.add_argument("--train", type=int, default=240)
    parser.add_argument("--val", type=int, default=60)
    parser.add_argument("--test", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--offset-limit-mm", type=float, default=9.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    rng = np.random.default_rng(args.seed)
    split_counts = {"train": args.train, "val": args.val, "test": args.test}
    plant = MujocoCoarseAlignmentPlant(
        NIH_HRA_SCENE,
        image_size_px=(args.width, args.height),
        initial_camera_xy_mm=(0.0, 0.0),
    )
    records: list[dict] = []
    trocar_material_id = plant.mujoco.mj_name2id(
        plant.model,
        plant.mujoco.mjtObj.mjOBJ_MATERIAL,
        "trocar_mat",
    )
    base_light = np.asarray(plant.model.light_diffuse, dtype=np.float64).copy()
    base_trocar = np.asarray(
        plant.model.mat_rgba[trocar_material_id], dtype=np.float64
    ).copy()

    try:
        for split, count in split_counts.items():
            image_dir = output_dir / "images" / split
            label_dir = output_dir / "labels" / split
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            for index in range(count):
                offset = rng.uniform(
                    -args.offset_limit_mm,
                    args.offset_limit_mm,
                    size=2,
                )
                light_scale = float(rng.uniform(0.72, 1.20))
                saturation_scale = float(rng.uniform(0.78, 1.18))
                plant.model.light_diffuse[:] = np.clip(
                    base_light * light_scale, 0.0, 1.0
                )
                randomized_color = base_trocar.copy()
                randomized_color[1:3] = np.clip(
                    randomized_color[1:3] * saturation_scale,
                    0.0,
                    1.0,
                )
                plant.model.mat_rgba[trocar_material_id] = randomized_color
                plant.reset((float(offset[0]), float(offset[1])))

                image_rgb = plant.capture_rgb()
                mask = plant.capture_trocar_segmentation_mask()
                bbox = mask_bbox(mask)
                if bbox is None:
                    raise RuntimeError(
                        f"Trocar missing from privileged mask at {split}/{index}."
                    )
                stem = f"{split}_{index:05d}"
                image_path = image_dir / f"{stem}.png"
                label_path = label_dir / f"{stem}.txt"
                ok = cv2.imwrite(
                    str(image_path),
                    cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
                )
                if not ok:
                    raise RuntimeError(f"Could not write image: {image_path}")
                label_path.write_text(
                    yolo_line(
                        bbox,
                        image_width=args.width,
                        image_height=args.height,
                    ),
                    encoding="utf-8",
                )
                records.append(
                    {
                        "split": split,
                        "index": index,
                        "image": str(image_path),
                        "label": str(label_path),
                        "camera_offset_mm": [
                            float(offset[0]),
                            float(offset[1]),
                        ],
                        "light_scale": light_scale,
                        "trocar_color_scale": saturation_scale,
                        "bbox_xyxy": list(bbox),
                        "visible_pixels": int(np.sum(mask)),
                    }
                )
    finally:
        plant.close()

    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {output_dir.as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "nc: 1",
                "names:",
                "  0: trocar",
                "",
            ]
        ),
        encoding="utf-8",
    )
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "purpose": "M2 simulated NIH/HRA trocar coarse detector training",
        "anatomy": NIH_HRA_EYE_SOURCE,
        "label_source": "MuJoCo privileged geom segmentation; never a controller input",
        "seed": args.seed,
        "image_size_px": [args.width, args.height],
        "offset_limit_mm": args.offset_limit_mm,
        "split_counts": split_counts,
        "data_yaml": str(data_yaml),
        "records": records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Dataset: {output_dir}")
    print(f"Samples: {len(records)}")
    print(f"Data YAML: {data_yaml}")


if __name__ == "__main__":
    main()
