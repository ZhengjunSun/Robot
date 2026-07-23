from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2

from handeye_common import DEFAULT_CONFIG, detect_charuco_pose, load_json


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "preflight"


def safe_imwrite(path: Path, image) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        return False
    encoded.tofile(str(path))
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Live read-only ChArUco detection monitor for hand-eye setup.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--interval-s", type=float, default=0.5)
    parser.add_argument("--save-every", type=int, default=1)
    args = parser.parse_args()

    config = load_json(args.config)
    camera_cfg = config["camera"]
    min_corners = int(config["target"]["min_corners"])
    cap = cv2.VideoCapture(int(camera_cfg["index"]))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(camera_cfg["image_width"]))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(camera_cfg["image_height"]))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_cfg['index']}")

    latest_frame = OUTPUT_DIR / "live_charuco_latest_frame.png"
    latest_overlay = OUTPUT_DIR / "live_charuco_latest_overlay.png"
    started = time.time()
    iteration = 0
    print("Live ChArUco monitor started. This script only reads camera frames.")
    print(f"Need at least {min_corners} ChArUco corners. Latest images are saved under {OUTPUT_DIR}")
    try:
        while time.time() - started <= args.duration_s:
            ok, frame = cap.read()
            if not ok or frame is None:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] camera_read_failed")
                time.sleep(args.interval_s)
                continue

            detection = detect_charuco_pose(frame, config)
            overlay = detection.pop("overlay_bgr", None)
            corners = int(detection.get("detected_charuco_corners", 0))
            markers = int(detection.get("detected_marker_count", 0))
            status = "DETECTED" if detection.get("success") else "not_detected"
            reproj = detection.get("mean_reprojection_error_px")
            reproj_text = f", reproj={reproj:.3f}px" if reproj is not None else ""
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] {status}: "
                f"corners={corners}, markers={markers}{reproj_text}"
            )

            if iteration % max(1, args.save_every) == 0:
                safe_imwrite(latest_frame, frame)
                if overlay is not None:
                    safe_imwrite(latest_overlay, overlay)
            iteration += 1
            time.sleep(args.interval_s)
    finally:
        cap.release()
        print("Live ChArUco monitor finished.")
        print("Latest frame:", latest_frame)
        print("Latest overlay:", latest_overlay)


if __name__ == "__main__":
    main()
