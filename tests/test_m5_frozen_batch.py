from __future__ import annotations

import numpy as np
import pytest

from real_3d_alignment.meca500_visual_env import Meca500VisualAlignmentPlant
from run_m5_frozen_batch import (
    nominal_domain,
    protocol_fingerprint,
    runtime_environment,
    sample_domain,
)


def test_nominal_domain_has_no_hidden_perturbation() -> None:
    sample = nominal_domain()
    assert sample.randomization_stratum == "nominal"
    assert sample.initial_joint_delta_deg == (0.0,) * 6
    assert sample.trocar_translation_mm == (0.0, 0.0, 0.0)
    assert sample.trocar_rotation_deg_xyz == (0.0, 0.0, 0.0)
    assert sample.camera_fovy_scale == 1.0
    assert sample.principal_point_shift_px == (0.0, 0.0)
    assert sample.occlusion_probability == 0.0
    assert sample.light_intensity_scale == 1.0


def test_frozen_domain_sample_is_reproducible() -> None:
    first = sample_domain(np.random.default_rng(20260724), image_height=960)
    second = sample_domain(np.random.default_rng(20260724), image_height=960)
    assert first == second


def test_randomization_strata_isolate_factor_families() -> None:
    geometry = sample_domain(
        np.random.default_rng(11),
        image_height=960,
        stratum="geometry",
    )
    photometric = sample_domain(
        np.random.default_rng(11),
        image_height=960,
        stratum="photometric",
    )
    calibration = sample_domain(
        np.random.default_rng(11),
        image_height=960,
        stratum="calibration",
    )
    occlusion = sample_domain(
        np.random.default_rng(11),
        image_height=960,
        stratum="occlusion",
    )
    nominal = nominal_domain()

    assert geometry.randomization_stratum == "geometry"
    assert geometry.rgb_gain == nominal.rgb_gain
    assert photometric.randomization_stratum == "photometric"
    assert photometric.trocar_translation_mm == nominal.trocar_translation_mm
    assert calibration.randomization_stratum == "calibration"
    assert calibration.rgb_gain == nominal.rgb_gain
    assert occlusion.randomization_stratum == "occlusion"
    assert occlusion.camera_fovy_scale == nominal.camera_fovy_scale


def test_frozen_domain_sample_respects_preregistered_bounds() -> None:
    sample = sample_domain(np.random.default_rng(7), image_height=960)
    assert all(-1.5 <= value <= 1.5 for value in sample.initial_joint_delta_deg)
    assert all(
        low <= value <= high
        for value, low, high in zip(
            sample.trocar_translation_mm,
            (-0.60, -0.60, -0.35),
            (0.60, 0.60, 0.35),
        )
    )
    assert all(
        -2.0 <= value <= 2.0
        for value in sample.trocar_rotation_deg_xyz
    )
    assert 0.985 <= sample.camera_fovy_scale <= 1.015
    assert all(
        -2.0 <= value <= 2.0
        for value in sample.principal_point_shift_px
    )
    assert 0.92 <= sample.rgb_gain <= 1.08
    assert 0.0 <= sample.rgb_noise_std <= 2.0
    assert sample.blur_kernel in {1, 3}
    assert 0.0 <= sample.occlusion_probability <= 0.025
    assert 0.05 <= sample.occlusion_fraction <= 0.12
    assert 0.90 <= sample.light_intensity_scale <= 1.10


def test_frozen_protocol_has_sha256_fingerprint() -> None:
    fingerprint = protocol_fingerprint()
    assert len(fingerprint) == 64
    int(fingerprint, 16)


def test_runtime_environment_records_pixel_pipeline_versions() -> None:
    environment = runtime_environment()
    assert {
        "platform",
        "python",
        "mujoco",
        "opencv",
        "numpy",
        "mujoco_gl",
        "pyopengl_platform",
    }.issubset(environment)


def test_mujoco_light_randomization_is_reversible() -> None:
    pytest.importorskip("mujoco")
    plant = Meca500VisualAlignmentPlant(
        image_size_px=(160, 120),
        settle_steps=1,
    )
    try:
        baseline = plant.model.light_diffuse.copy()
        plant.set_domain_randomization(light_intensity_scale=0.90)
        assert np.allclose(plant.model.light_diffuse, baseline * 0.90)
        plant.reset_domain()
        assert np.allclose(plant.model.light_diffuse, baseline)
    finally:
        plant.close()
