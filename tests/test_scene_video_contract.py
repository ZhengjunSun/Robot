from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from real_3d_alignment.fine_vision import FineRingEstimate
from real_3d_alignment.scene_contract import (
    CAMERA_CENTER_BGR,
    INNER_CENTER_BGR,
    NEEDLE_VISIBLE_LENGTH_MM,
    OUTER_CENTER_BGR,
    REFERENCE_NEEDLE_VISIBLE_LENGTH_MM,
    TROCAR_TILT_DEG,
    VIDEO_CENTER_LABELS,
)
from real_3d_alignment.staged_alignment import FineObservation
from run_mujoco_full_flow import ROOT, annotate


LIGHTWEIGHT_SCENE = (
    ROOT / "3d_modeling" / "mujoco" / "single_arm_trocar_visual_alignment_nih_hra.xml"
)
FULL_ARM_SCENE = (
    ROOT / "3d_modeling" / "mujoco" / "meca500_r4_ophthalmic_visual_execution_scene.xml"
)


def _float_vector(element: ET.Element, attribute: str) -> tuple[float, ...]:
    return tuple(float(value) for value in element.attrib[attribute].split())


def test_trocar_tilt_is_frozen_in_both_scenes() -> None:
    lightweight = ET.parse(LIGHTWEIGHT_SCENE).getroot()
    full_arm = ET.parse(FULL_ARM_SCENE).getroot()

    lightweight_trocar = lightweight.find(".//body[@name='frozen_trocar']")
    full_arm_trocar = full_arm.find(".//body[@name='single_trocar_visual']")

    assert lightweight_trocar is not None
    assert full_arm_trocar is not None
    lightweight_euler = _float_vector(lightweight_trocar, "euler")
    full_arm_euler = _float_vector(full_arm_trocar, "euler")
    assert math.degrees(lightweight_euler[0]) == pytest.approx(TROCAR_TILT_DEG)
    assert math.degrees(full_arm_euler[2]) == pytest.approx(TROCAR_TILT_DEG)


def test_presentation_needle_is_exactly_one_third_reference_length() -> None:
    root = ET.parse(FULL_ARM_SCENE).getroot()
    shaft = root.find(".//geom[@name='instrument_shaft']")
    tip = root.find(".//site[@name='tool_tip']")

    assert shaft is not None
    assert tip is not None
    fromto = _float_vector(shaft, "fromto")
    visible_length_mm = abs(fromto[5] - fromto[2]) * 1000.0
    assert NEEDLE_VISIBLE_LENGTH_MM == pytest.approx(
        REFERENCE_NEEDLE_VISIBLE_LENGTH_MM / 3.0
    )
    assert visible_length_mm == pytest.approx(NEEDLE_VISIBLE_LENGTH_MM)
    assert _float_vector(tip, "pos")[2] * 1000.0 == pytest.approx(48.0)


def test_full_meca500_visual_chain_is_present() -> None:
    root = ET.parse(FULL_ARM_SCENE).getroot()
    geom_names = {
        geom.attrib.get("name")
        for geom in root.findall(".//geom")
    }
    assert {
        f"link_{index}_visual"
        for index in range(7)
    }.issubset(geom_names)


def test_eye_in_hand_annotation_draws_all_three_centers() -> None:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    estimate = FineRingEstimate(
        observation=FineObservation(
            image_center_px=(320.0, 240.0),
            outer_center_px=(300.0, 220.0),
            inner_center_px=(304.0, 222.0),
            lateral_error_mm=0.1,
            axis_error_deg=1.0,
            standoff_error_mm=0.2,
            reprojection_error_px=0.1,
            quality_gate_pass=True,
        ),
        outer_major_diameter_px=40.0,
        outer_minor_diameter_px=38.0,
        inner_major_diameter_px=16.0,
        inner_minor_diameter_px=15.0,
        outer_angle_deg=10.0,
        inner_angle_deg=10.0,
        estimated_depth_mm=45.0,
        ellipse_fit_error_px=0.1,
    )

    annotated = annotate(
        image,
        title="contract test",
        phase="fine_alignment",
        step=1,
        metrics={},
        insertion_extension_mm=0.0,
        reason="test",
        fine_estimate=estimate,
        draw_alignment_centers=True,
    )

    assert tuple(int(value) for value in annotated[240, 320]) == CAMERA_CENTER_BGR
    assert tuple(int(value) for value in annotated[220, 300]) == OUTER_CENTER_BGR
    assert tuple(int(value) for value in annotated[222, 304]) == INNER_CENTER_BGR
    assert VIDEO_CENTER_LABELS == (
        "CAMERA CENTER",
        "OUTER CENTER",
        "INNER CENTER",
    )

