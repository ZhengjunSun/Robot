from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from handeye_common import DEFAULT_CONFIG, create_charuco_board, load_json, write_json


ROOT = Path(__file__).resolve().parents[1]
TARGET_DIR = ROOT / "calibration" / "targets"


def mm_to_px(mm: float, dpi: int) -> int:
    return int(round(mm / 25.4 * dpi))


def save_png_unicode(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Failed to encode PNG: {path}")
    encoded.tofile(str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a printable ChArUco board.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=TARGET_DIR)
    args = parser.parse_args()

    config = load_json(args.config)
    board = create_charuco_board(config)
    target = config["target"]
    dpi = int(target["print_dpi"])

    board_width_mm = float(target["squares_x"]) * float(target["square_length_m"]) * 1000.0
    board_height_mm = float(target["squares_y"]) * float(target["square_length_m"]) * 1000.0
    board_width_px = mm_to_px(board_width_mm, dpi)
    board_height_px = mm_to_px(board_height_mm, dpi)
    board_img = board.generateImage((board_width_px, board_height_px), marginSize=0, borderBits=1)

    a4_width_px = mm_to_px(297.0, dpi)
    a4_height_px = mm_to_px(210.0, dpi)
    canvas = np.full((a4_height_px, a4_width_px), 255, dtype=np.uint8)
    x0 = (a4_width_px - board_width_px) // 2
    y0 = (a4_height_px - board_height_px) // 2
    canvas[y0 : y0 + board_height_px, x0 : x0 + board_width_px] = board_img

    args.output_dir.mkdir(parents=True, exist_ok=True)
    exact_path = args.output_dir / "charuco_7x5_30mm_exact_300dpi.png"
    a4_path = args.output_dir / "charuco_7x5_30mm_a4_landscape_300dpi.png"
    save_png_unicode(exact_path, board_img)
    save_png_unicode(a4_path, canvas)

    metadata = {
        "target": target,
        "board_width_mm": board_width_mm,
        "board_height_mm": board_height_mm,
        "board_width_px": board_width_px,
        "board_height_px": board_height_px,
        "a4_width_px": a4_width_px,
        "a4_height_px": a4_height_px,
        "exact_board_png": str(exact_path.relative_to(ROOT).as_posix()),
        "a4_landscape_png": str(a4_path.relative_to(ROOT).as_posix()),
        "print_instruction": "Print at 100% scale / actual size. Do not fit-to-page or scale.",
    }
    write_json(args.output_dir / "charuco_7x5_30mm_metadata.json", metadata)
    print("Generated ChArUco board:")
    print(" exact:", exact_path)
    print(" a4:", a4_path)


if __name__ == "__main__":
    main()
