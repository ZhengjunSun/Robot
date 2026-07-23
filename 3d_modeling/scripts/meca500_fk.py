from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "meca500_mdh_initial.json"
OUTPUT_DIR = ROOT / "outputs"


def load_mdh_config(path: Path = CONFIG_PATH) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def mdh_transform(alpha: float, a_m: float, theta: float, d_m: float) -> np.ndarray:
    ca = math.cos(alpha)
    sa = math.sin(alpha)
    ct = math.cos(theta)
    st = math.sin(theta)
    return np.array(
        [
            [ct, -st, 0.0, a_m],
            [ca * st, ca * ct, -sa, -d_m * sa],
            [sa * st, sa * ct, ca, d_m * ca],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def forward_kinematics(joint_angles_rad: np.ndarray, config: dict) -> list[np.ndarray]:
    if len(joint_angles_rad) != 6:
        raise ValueError("Meca500 model expects six joint angles.")

    transforms = [np.eye(4)]
    current = np.eye(4)
    for q, row in zip(joint_angles_rad, config["mdh"]):
        alpha, a_mm, theta_offset, d_mm = row
        current = current @ mdh_transform(
            alpha=float(alpha),
            a_m=float(a_mm) / 1000.0,
            theta=float(theta_offset) + float(q),
            d_m=float(d_mm) / 1000.0,
        )
        transforms.append(current.copy())
    return transforms


def parse_joint_degrees(text: str) -> np.ndarray:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if len(values) != 6:
        raise argparse.ArgumentTypeError("Use six comma-separated joint angles in degrees.")
    return np.deg2rad(np.array(values, dtype=float))


def transform_to_record(index: int, transform: np.ndarray) -> dict:
    return {
        "frame": f"frame_{index}",
        "position_m": transform[:3, 3].round(9).tolist(),
        "rotation_matrix": transform[:3, :3].round(9).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Meca500 initial Modified-DH FK checker.")
    parser.add_argument(
        "--joints-deg",
        type=parse_joint_degrees,
        default=np.zeros(6),
        help="Six comma-separated joint angles in degrees, e.g. 0,-30,60,0,30,0",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "meca500_fk_result.json",
        help="Path to write FK result JSON.",
    )
    args = parser.parse_args()

    config = load_mdh_config()
    transforms = forward_kinematics(args.joints_deg, config)
    records = [transform_to_record(index, transform) for index, transform in enumerate(transforms)]

    result = {
        "model": config["name"],
        "status": config["status"],
        "joint_angles_deg": np.rad2deg(args.joints_deg).round(6).tolist(),
        "frames": records,
        "tool_frame": records[-1],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    tool_position = result["tool_frame"]["position_m"]
    print("FK result written:", args.output)
    print("Tool position [m]:", tool_position)


if __name__ == "__main__":
    main()
