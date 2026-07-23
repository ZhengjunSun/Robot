from __future__ import annotations

from dataclasses import asdict, dataclass
from math import radians, tan
from pathlib import Path
from typing import Any, Iterable

from .config import load_config


@dataclass(frozen=True)
class ClearanceGeometry:
    port_inner_diameter_mm: float
    tool_outer_diameter_mm: float
    effective_wall_length_mm: float

    @property
    def nominal_radial_clearance_mm(self) -> float:
        return 0.5 * (self.port_inner_diameter_mm - self.tool_outer_diameter_mm)

    def validate(self) -> None:
        if self.port_inner_diameter_mm <= 0.0:
            raise ValueError("port_inner_diameter_mm must be positive")
        if self.tool_outer_diameter_mm <= 0.0:
            raise ValueError("tool_outer_diameter_mm must be positive")
        if self.effective_wall_length_mm <= 0.0:
            raise ValueError("effective_wall_length_mm must be positive")
        if self.nominal_radial_clearance_mm <= 0.0:
            raise ValueError("tool must be smaller than the trocar inner diameter")


@dataclass(frozen=True)
class ClearanceSample:
    lateral_error_mm: float
    insertion_depth_mm: float
    axis_error_deg: float = 0.0
    uncertainty_margin_mm: float = 0.0


@dataclass(frozen=True)
class ClearanceSampleResult:
    nominal_radial_clearance_mm: float
    lateral_error_mm: float
    tilt_sweep_mm: float
    uncertainty_margin_mm: float
    geometric_clearance_mm: float
    robust_clearance_mm: float
    insertion_depth_mm: float
    overlaps_port_wall: bool


@dataclass(frozen=True)
class ClearanceEpisodeResult:
    evidence_completeness: str
    trajectory_sample_count: int
    inserted_sample_count: int
    nominal_radial_clearance_mm: float
    terminal_clearance_mm: float
    minimum_wall_clearance_mm: float | None
    minimum_robust_clearance_mm: float | None
    wall_contact_detected: bool
    wall_contact_steps: int | None
    wall_contact_impulse: float | None
    depth_pass: bool
    over_insert_pass: bool
    terminal_clearance_pass: bool
    trajectory_clearance_pass: bool | None
    zero_wall_contact_pass: bool
    joint_limit_pass: bool
    speed_limit_pass: bool
    certified_success: bool
    failure_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["failure_reasons"] = list(self.failure_reasons)
        return result


@dataclass(frozen=True)
class ClearanceContract:
    geometry: ClearanceGeometry
    max_over_insert_mm: float = 0.20
    depth_tolerance_mm: float = 0.20
    contact_force_epsilon_n: float = 1e-9

    @classmethod
    def from_profile(cls, path: str | Path) -> "ClearanceContract":
        profile = load_config(path)
        geometry_cfg = profile.get("stage1_single_arm_geometry", {})
        trocar_cfg = geometry_cfg.get("trocar", {})
        tool_cfg = geometry_cfg.get("injection_needle_proxy", {})
        proxy_cfg = profile.get("mujoco_contact_proxy", {})
        full_depth = proxy_cfg.get("collision_wall_full_depth_mm")
        if full_depth is None:
            full_depth = 2.0 * float(proxy_cfg.get("port_wall_depth_mm", 0.35))
        geometry = ClearanceGeometry(
            port_inner_diameter_mm=float(trocar_cfg["inner_diameter_mm"]),
            tool_outer_diameter_mm=float(tool_cfg["outer_diameter_mm"]),
            effective_wall_length_mm=float(full_depth),
        )
        geometry.validate()
        return cls(
            geometry=geometry,
            max_over_insert_mm=float(proxy_cfg.get("max_over_insert_mm", 0.20)),
            depth_tolerance_mm=float(proxy_cfg.get("success_depth_tolerance_mm", 0.20)),
            contact_force_epsilon_n=float(proxy_cfg.get("contact_force_epsilon_n", 1e-9)),
        )

    def evaluate_sample(self, sample: ClearanceSample) -> ClearanceSampleResult:
        lateral = max(0.0, float(sample.lateral_error_mm))
        insertion = max(0.0, float(sample.insertion_depth_mm))
        uncertainty = max(0.0, float(sample.uncertainty_margin_mm))
        angle_deg = abs(float(sample.axis_error_deg))
        tilt_sweep = self.geometry.effective_wall_length_mm * tan(radians(angle_deg))
        geometric = self.geometry.nominal_radial_clearance_mm - lateral - tilt_sweep
        robust = geometric - uncertainty
        return ClearanceSampleResult(
            nominal_radial_clearance_mm=self.geometry.nominal_radial_clearance_mm,
            lateral_error_mm=lateral,
            tilt_sweep_mm=tilt_sweep,
            uncertainty_margin_mm=uncertainty,
            geometric_clearance_mm=geometric,
            robust_clearance_mm=robust,
            insertion_depth_mm=insertion,
            overlaps_port_wall=insertion > 1e-9,
        )

    def evaluate_episode(
        self,
        samples: Iterable[ClearanceSample],
        *,
        terminal_sample: ClearanceSample,
        target_insert_depth_mm: float,
        max_contact_force_n: float = 0.0,
        wall_contact_steps: int | None = None,
        wall_contact_impulse: float | None = None,
        joint_limit_violation: bool = False,
        speed_limit_violation: bool = False,
        full_trajectory_available: bool = True,
    ) -> ClearanceEpisodeResult:
        sample_results = [self.evaluate_sample(sample) for sample in samples]
        inserted = [item for item in sample_results if item.overlaps_port_wall]
        terminal = self.evaluate_sample(terminal_sample)
        if not inserted and terminal.overlaps_port_wall:
            inserted = [terminal]

        min_clearance = min((item.geometric_clearance_mm for item in inserted), default=None)
        min_robust = min((item.robust_clearance_mm for item in inserted), default=None)
        contact_detected = float(max_contact_force_n) > self.contact_force_epsilon_n
        if wall_contact_steps is not None:
            contact_detected = contact_detected or int(wall_contact_steps) > 0
        if wall_contact_impulse is not None:
            contact_detected = contact_detected or float(wall_contact_impulse) > 0.0

        target = max(0.0, float(target_insert_depth_mm))
        terminal_depth = max(0.0, float(terminal_sample.insertion_depth_mm))
        shortfall = max(0.0, target - terminal_depth)
        over_insert = max(0.0, terminal_depth - target)
        depth_pass = shortfall <= self.depth_tolerance_mm + 1e-12
        over_insert_pass = over_insert <= self.max_over_insert_mm + 1e-12
        terminal_clearance_pass = terminal.robust_clearance_mm > 0.0
        trajectory_clearance_pass = None
        if full_trajectory_available:
            trajectory_clearance_pass = bool(inserted) and min_robust is not None and min_robust > 0.0
        zero_contact_pass = not contact_detected
        joint_pass = not bool(joint_limit_violation)
        speed_pass = not bool(speed_limit_violation)

        failures: list[str] = []
        if not full_trajectory_available:
            failures.append("full_trajectory_unavailable")
        elif not trajectory_clearance_pass:
            failures.append("nonpositive_trajectory_clearance")
        if not terminal_clearance_pass:
            failures.append("nonpositive_terminal_clearance")
        if not zero_contact_pass:
            failures.append("wall_contact_detected")
        if not depth_pass:
            failures.append("insertion_depth_shortfall")
        if not over_insert_pass:
            failures.append("over_insertion")
        if not joint_pass:
            failures.append("joint_limit_violation")
        if not speed_pass:
            failures.append("speed_limit_violation")

        certified = bool(
            full_trajectory_available
            and trajectory_clearance_pass
            and terminal_clearance_pass
            and zero_contact_pass
            and depth_pass
            and over_insert_pass
            and joint_pass
            and speed_pass
        )
        return ClearanceEpisodeResult(
            evidence_completeness="full_trajectory" if full_trajectory_available else "terminal_only",
            trajectory_sample_count=len(sample_results),
            inserted_sample_count=len(inserted),
            nominal_radial_clearance_mm=self.geometry.nominal_radial_clearance_mm,
            terminal_clearance_mm=terminal.robust_clearance_mm,
            minimum_wall_clearance_mm=min_clearance,
            minimum_robust_clearance_mm=min_robust,
            wall_contact_detected=contact_detected,
            wall_contact_steps=wall_contact_steps,
            wall_contact_impulse=wall_contact_impulse,
            depth_pass=depth_pass,
            over_insert_pass=over_insert_pass,
            terminal_clearance_pass=terminal_clearance_pass,
            trajectory_clearance_pass=trajectory_clearance_pass,
            zero_wall_contact_pass=zero_contact_pass,
            joint_limit_pass=joint_pass,
            speed_limit_pass=speed_pass,
            certified_success=certified,
            failure_reasons=tuple(failures),
        )
