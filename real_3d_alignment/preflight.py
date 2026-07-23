from __future__ import annotations

import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, load_config, load_json, resolve_project_path, write_json


REQUIRED_MODULES = [
    "cv2",
    "numpy",
    "scipy",
    "pybullet",
    "PIL",
    "ultralytics",
]

OPTIONAL_HARDWARE_MODULES = [
    "mecademicpy",
]


def _check_module(name: str, required: bool = True) -> dict[str, Any]:
    found = importlib.util.find_spec(name) is not None
    return {
        "name": name,
        "status": "ok" if found else ("error" if required else "warning"),
        "found": found,
        "required": required,
    }


def _check_path(label: str, path: Path) -> dict[str, Any]:
    return {
        "label": label,
        "path": str(path),
        "status": "ok" if path.exists() else "error",
        "exists": path.exists(),
    }


def _check_entrypoints() -> list[dict[str, Any]]:
    entries = [
        ("real_3d_cli", PROJECT_ROOT / "run_real_3d_alignment.py"),
        ("real_3d_step_bridge", PROJECT_ROOT / "execute_real_3d_step.py"),
        ("real_3d_step_compare", PROJECT_ROOT / "compare_real_3d_step.py"),
        ("preflight_cli", PROJECT_ROOT / "check_real_3d_alignment.py"),
        (
            "staged_alignment_config",
            PROJECT_ROOT / "config" / "staged_visual_alignment.json",
        ),
    ]
    return [_check_path(label, path) for label, path in entries]


def _check_assets() -> list[dict[str, Any]]:
    assets = [
        (
            "meca500_visual_scene",
            PROJECT_ROOT
            / "3d_modeling"
            / "mujoco"
            / "meca500_r4_ophthalmic_visual_execution_scene.xml",
        ),
        (
            "trocar_sidecar_scene",
            PROJECT_ROOT
            / "3d_modeling"
            / "mujoco"
            / "single_arm_trocar_sidecar.xml",
        ),
        (
            "trocar_geometry",
            PROJECT_ROOT
            / "3d_modeling"
            / "config"
            / "trocar_model_measured_20260612.json",
        ),
    ]
    return [_check_path(label, path) for label, path in assets]


def _check_detector_models() -> dict[str, Any]:
    config_path = PROJECT_ROOT / "robot_servo_config.json"
    if not config_path.exists():
        return {"status": "warning", "message": "robot_servo_config.json not found", "models": []}
    try:
        config = load_json(config_path)
    except Exception as exc:
        return {"status": "error", "message": f"Cannot parse robot_servo_config.json: {type(exc).__name__}: {exc}", "models": []}

    model_cfg = config.get("models", {})
    strategy = str(model_cfg.get("mode_strategy", "auto")).lower()
    raw_paths = {
        "single": str(model_cfg.get("model", "") or ""),
        "outer": str(model_cfg.get("outer_model", "") or ""),
        "inner": str(model_cfg.get("inner_model", "") or ""),
    }
    model_paths = {label: (PROJECT_ROOT / value if value else None) for label, value in raw_paths.items()}
    checks = [
        {"label": label, "path": "" if path is None else str(path), "exists": False if path is None else path.exists()}
        for label, path in model_paths.items()
    ]
    single_ok = model_paths["single"] is not None and model_paths["single"].exists()
    dual_ok = (
        model_paths["outer"] is not None
        and model_paths["inner"] is not None
        and model_paths["outer"].exists()
        and model_paths["inner"].exists()
    )
    if strategy == "single":
        status = "ok" if single_ok else "warning"
    elif strategy == "dual":
        status = "ok" if dual_ok else "warning"
    elif strategy == "auto":
        status = "ok" if single_ok or dual_ok else "warning"
    else:
        status = "error"
    return {
        "status": status,
        "strategy": strategy,
        "single_available": single_ok,
        "dual_available": dual_ok,
        "models": checks,
    }


def _check_evidence_artifacts() -> dict[str, Any]:
    artifacts = [
        PROJECT_ROOT
        / "docs"
        / "project"
        / "PROJECT_DIRECTION_20260723.md",
        PROJECT_ROOT
        / "docs"
        / "plans"
        / "VISUAL_ALIGNMENT_AND_INSERTION_PLAN_20260723.md",
        PROJECT_ROOT / "tests" / "test_staged_visual_alignment.py",
    ]
    checks = [_check_path(path.name, path) for path in artifacts]
    missing = [item for item in checks if not item["exists"]]
    return {
        "status": "ok" if not missing else "warning",
        "artifacts": checks,
        "missing_count": len(missing),
    }


def _check_safety_config(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    robot_cfg = cfg.get("robot", {})
    checks = []
    if robot_cfg.get("execution_enabled") is True:
        checks.append({"status": "error", "message": "config/real_3d_alignment.json has robot.execution_enabled=true"})
    else:
        checks.append({"status": "ok", "message": "real robot execution is disabled in the unified pipeline"})
    token = str(robot_cfg.get("confirmation_token", ""))
    checks.append({"status": "ok" if token else "warning", "message": "confirmation token is configured" if token else "confirmation token is empty"})
    try:
        max_age = float(robot_cfg.get("max_dry_run_age_s", 0.0))
    except (TypeError, ValueError):
        max_age = 0.0
    checks.append({
        "status": "ok" if max_age > 0.0 else "error",
        "message": "max dry-run age is positive" if max_age > 0.0 else "robot.max_dry_run_age_s must be positive",
    })
    return checks


def _check_hardcoded_paths() -> dict[str, Any]:
    needles = ["E:" + "/桌面/探索版", "E:" + "\\桌面\\探索版"]
    hits: list[str] = []
    skipped_parts = {"没有用的代码", "__pycache__"}
    for path in PROJECT_ROOT.rglob("*.py"):
        if any(part in skipped_parts for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            continue
        if any(needle in text for needle in needles):
            hits.append(str(path.relative_to(PROJECT_ROOT)))
    return {
        "status": "ok" if not hits else "warning",
        "hits": hits,
    }


def _check_gitignore_outputs() -> dict[str, Any]:
    gitignore = PROJECT_ROOT / ".gitignore"
    required_patterns = ["3d_modeling/outputs/", "alignment_sim/outputs/", "*.pt", "*.pth"]
    if not gitignore.exists():
        return {"status": "warning", "missing_patterns": required_patterns, "message": ".gitignore not found"}
    text = gitignore.read_text(encoding="utf-8-sig")
    missing = [pattern for pattern in required_patterns if pattern not in text]
    return {
        "status": "ok" if not missing else "warning",
        "missing_patterns": missing,
    }


def run_preflight(config_path: Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    paths = cfg["paths"]

    path_checks = [
        _check_path("main_config", config_path),
        _check_path("camera_config", resolve_project_path(paths["camera_config"])),
        _check_path("extrinsic_config", resolve_project_path(paths["extrinsic_config"])),
        _check_path("trocar_config", resolve_project_path(paths["trocar_config"])),
    ]

    json_checks = []
    for check in path_checks:
        if not check["exists"]:
            continue
        try:
            load_json(Path(check["path"]))
            json_checks.append({"label": check["label"], "status": "ok"})
        except Exception as exc:
            json_checks.append({"label": check["label"], "status": "error", "error": f"{type(exc).__name__}: {exc}"})

    modules = [_check_module(name, required=True) for name in REQUIRED_MODULES]
    modules += [_check_module(name, required=False) for name in OPTIONAL_HARDWARE_MODULES]

    control = cfg["control"]
    control_checks = []
    if float(control["target_distance_mm"]) <= 0:
        control_checks.append({"status": "error", "message": "target_distance_mm must be positive"})
    if float(control["max_translation_step_mm"]) <= 0:
        control_checks.append({"status": "error", "message": "max_translation_step_mm must be positive"})
    if float(control["max_rotation_step_deg"]) <= 0:
        control_checks.append({"status": "error", "message": "max_rotation_step_deg must be positive"})
    for gain in ("k_xy", "k_z", "k_rot"):
        if float(control[gain]) < 0:
            control_checks.append({"status": "error", "message": f"{gain} must be non-negative"})
    if not control_checks:
        control_checks.append({
            "status": "ok",
            "message": f"control profile {control.get('profile', 'unspecified')} has valid target, gains, and limits",
        })

    sections = {
        "paths": path_checks,
        "json": json_checks,
        "modules": modules,
        "control": control_checks,
        "safety": _check_safety_config(cfg),
        "entrypoints": _check_entrypoints(),
        "assets": _check_assets(),
        "detector_models": _check_detector_models(),
        "evidence_artifacts": _check_evidence_artifacts(),
        "hardcoded_paths": _check_hardcoded_paths(),
        "gitignore": _check_gitignore_outputs(),
    }

    errors = []
    warnings = []
    for value in sections.values():
        items = value if isinstance(value, list) else [value]
        for item in items:
            if item.get("status") == "error":
                errors.append(item)
            elif item.get("status") == "warning":
                warnings.append(item)

    return {
        "timestamp": datetime.now().isoformat(),
        "config": str(config_path),
        "status": "error" if errors else ("warning" if warnings else "ok"),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "sections": sections,
    }


def write_preflight_report(path: Path, report: dict[str, Any]) -> None:
    write_json(path, report)
