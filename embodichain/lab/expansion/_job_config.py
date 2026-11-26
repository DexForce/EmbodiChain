# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""Shared declarations for task-bound expansion and standalone jobs."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import MISSING
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from embodichain.utils import configclass
from embodichain.utils.config_paths import resolve_config_path

from ._task_program_config import _validate_task_program_expansion, load_mapping
from .configured import ExpansionRunCfg, _decode_cohort_job

__all__: list[str] = []

_EXECUTION = {"mode", "batch_size", "seed", "max_steps", "settling_steps"}
_COLLECTION = {"output_dir", "max_attempts", "max_wall_time_s"}
_ROLLING_EXECUTION = {
    "workload_count",
    "cpu_threads",
    "grasp_nms_chunk_size",
    "reuse_idle_rows",
    "nominal_seed",
}
_ROLLING_VALIDATION = {"collision_backend", "collision_device"}
_COHORT_BLOCKS = {
    "execution": _EXECUTION,
    "collection": _COLLECTION
    | {"target_episodes", "retry_interrupted", "export_lerobot", "episode_byte_budget"},
    "scene": {"atlas"},
    "trajectory": {
        "program",
        "motion_source",
        "source_profile",
        "motion_profile",
        "demo_skill_duration_scale",
        "demo_skill_velocity_limits",
        "demo_skill_acceleration_limits",
    },
    "physics": {
        "mass_scale",
        "friction",
        "center_of_mass",
        "joint_stiffness_scale",
        "joint_damping_scale",
    },
    "observation": {
        "capture_media",
        "visual_profile_file",
        "visual_profile",
        "media_buffer_bytes",
    },
    "validation": {"collision_profile", "prototype_validation_profile"},
}
_PATH_FIELDS = {
    "collection": ("output_dir",),
    "scene": ("atlas",),
    "trajectory": ("program", "source_profile", "motion_profile"),
    "observation": ("visual_profile_file",),
    "validation": ("collision_profile", "prototype_validation_profile"),
}


@configclass
class _RollingRunCfg:
    deployment: Path = MISSING
    output_dir: Path = MISSING
    options: dict[str, Any] = MISSING
    seed: int = 0

    @property
    def num_envs(self) -> int:
        settings = self.options["engine"]
        return settings["workload_count"] * settings["batch_size"]


def _has_job_mode(binding: Mapping[str, Any]) -> bool:
    execution = binding.get("execution")
    return isinstance(execution, Mapping) and "mode" in execution


def _closed_mapping(value: object, allowed: set[str], field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    if extra := value.keys() - allowed:
        raise ValueError(f"{field} contains unknown fields: {sorted(map(str, extra))}")
    return dict(value)


def _path(value: object, base_dir: Path, field: str) -> Path:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field} must be a nonempty path")
    return resolve_config_path(value, base_dir=base_dir).resolve()


def _merge_expansion_config(
    base: Mapping[str, Any], patch: Mapping[str, Any]
) -> dict[str, Any]:
    """Deep-merge a referenced expansion config with task-local overrides."""
    merged = deepcopy(dict(base))
    for key, value in patch.items():
        if isinstance(value, Mapping):
            current = merged.get(key, {})
            if not isinstance(current, Mapping):
                raise ValueError(f"expansion.{key} cannot replace a scalar")
            merged[key] = _merge_expansion_config(current, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _resolve_job_paths(binding: Mapping[str, Any], base_dir: Path) -> dict[str, Any]:
    result = deepcopy(dict(binding))
    for section, keys in _PATH_FIELDS.items():
        values = result.get(section)
        if not isinstance(values, dict):
            continue
        for key in keys:
            if values.get(key) is not None:
                values[key] = str(
                    _path(values[key], base_dir, f"expansion.{section}.{key}")
                )
    return result


def _legacy_rolling_declaration(value: dict[str, Any]) -> dict[str, Any]:
    value = _validate_task_program_expansion(value)
    engine = value["engine"]
    scene = {"poses": value["scene"], **value.get("scene_sampling", {})}
    scene.update(
        {key: value[key] for key in ("scheduling", "pose_coverage") if key in value}
    )
    return {
        "execution": {
            "mode": "rolling",
            **{k: engine[k] for k in engine.keys() & (_EXECUTION | _ROLLING_EXECUTION)},
        },
        "collection": {
            k: engine[k]
            for k in engine.keys() & (_COLLECTION | {"episodes_per_operator"})
        },
        "observation": {k: engine[k] for k in ("capture_media", "collection_renderer")},
        "scene": scene,
        "trajectory": value["trajectory"],
        "validation": {
            **value["validation"],
            **{k: engine[k] for k in _ROLLING_VALIDATION},
        },
    }


def _load_expansion_declaration(
    binding: Mapping[str, Any], *, base_dir: Path
) -> tuple[dict[str, Any], Path | None]:
    """Resolve a task or job expansion block and its optional preset."""
    if not isinstance(binding, Mapping):
        raise ValueError("expansion must be a mapping")
    if "config" not in binding:
        return (
            (
                _resolve_job_paths(binding, base_dir)
                if _has_job_mode(binding)
                else deepcopy(dict(binding))
            ),
            None,
        )

    from embodichain.utils.utility import load_config

    config = _path(binding["config"], base_dir, "expansion.config")
    if config.suffix.lower() not in {".yaml", ".yml", ".json"}:
        raise ValueError(f"expansion.config must be a YAML or JSON file: {config}")
    if not config.is_file():
        raise FileNotFoundError(f"expansion.config is not a file: {config}")
    referenced = load_config(config)
    if "expansion" in referenced:
        if set(referenced) != {"expansion"}:
            raise ValueError(
                "expansion config may contain either expansion fields or one outer expansion mapping"
            )
        referenced = referenced["expansion"]
        if not isinstance(referenced, Mapping):
            raise TypeError("expansion config.expansion must be a mapping")
    if "config" in referenced:
        raise ValueError("expansion.config cannot reference another expansion config")
    local = {k: v for k, v in binding.items() if k != "config"}
    if _has_job_mode(referenced) or _has_job_mode(local):
        if "engine" in referenced:
            referenced = _legacy_rolling_declaration(dict(referenced))
        referenced = _resolve_job_paths(referenced, config.parent)
        local = _resolve_job_paths(local, base_dir)
    return _merge_expansion_config(referenced, local), config


def _decode_job(
    deployment: Path, binding: Mapping[str, Any]
) -> ExpansionRunCfg | _RollingRunCfg:
    execution = binding.get("execution")
    mode = execution.get("mode") if isinstance(execution, Mapping) else None
    if mode == "cohort":
        sections = _closed_mapping(binding, set(_COHORT_BLOCKS), "expansion")
        values = {"deployment": str(deployment)}
        for name, fields in _COHORT_BLOCKS.items():
            section = _closed_mapping(
                sections.get(name, {}), fields, f"expansion.{name}"
            )
            if name == "scene" and "atlas" in section:
                section["scene_atlas"] = section.pop("atlas")
            values.update(section)
        values.pop("mode")
        return _decode_cohort_job(values, deployment.parent)
    if mode != "rolling":
        raise ValueError("expansion.execution.mode must be cohort or rolling")

    sections = _closed_mapping(binding, set(_COHORT_BLOCKS) - {"physics"}, "expansion")
    execution = _closed_mapping(
        execution, _EXECUTION | _ROLLING_EXECUTION, "expansion.execution"
    )
    collection = _closed_mapping(
        sections.get("collection", {}),
        _COLLECTION | {"episodes_per_operator"},
        "expansion.collection",
    )
    observation = _closed_mapping(
        sections.get("observation", {}),
        {"capture_media", "collection_renderer"},
        "expansion.observation",
    )
    output = _path(
        collection.pop("output_dir", None),
        deployment.parent,
        "expansion.collection.output_dir",
    )
    if "episodes_per_operator" not in collection:
        raise ValueError(
            "expansion.collection.episodes_per_operator is required for rolling jobs"
        )
    execution.pop("mode")
    seed = execution.pop("seed", 0)
    if type(seed) is not int or seed < 0:
        raise ValueError("expansion.execution.seed must be a nonnegative integer")
    scene, trajectory, validation = (
        sections.get(name) for name in ("scene", "trajectory", "validation")
    )
    if any(not isinstance(value, dict) for value in (scene, trajectory, validation)):
        raise ValueError(
            "rolling expansion requires scene, trajectory and validation mappings"
        )
    scene, validation = dict(scene), dict(validation)
    poses = scene.pop("poses", None)
    feedback = {
        key: scene.pop(key) for key in ("scheduling", "pose_coverage") if key in scene
    }
    checks = {
        key: validation.pop(key) for key in _ROLLING_VALIDATION if key in validation
    }
    options = _validate_task_program_expansion(
        {
            "schema_version": 1,
            "engine": {
                "max_attempts": 32,
                "max_wall_time_s": 3600.0,
                "max_steps": 2000,
                "settling_steps": 5,
                "capture_media": False,
                **execution,
                **collection,
                **observation,
                **checks,
            },
            "scene": poses,
            "trajectory": deepcopy(trajectory),
            "validation": validation,
            **({"scene_sampling": scene} if scene else {}),
            **feedback,
        }
    )
    engine = options["engine"]
    engine["workload_count"] = min(
        engine["workload_count"], len(trajectory["operators"])
    )
    job = _RollingRunCfg(
        deployment=deployment, output_dir=output, options=options, seed=seed
    )
    job.validate()
    return job


def _load_job(path: str | Path) -> ExpansionRunCfg | _RollingRunCfg:
    path = Path(path).expanduser().resolve()
    values = load_mapping(path)
    if "expansion" not in values:
        return _decode_cohort_job(values, path.parent)
    job = _closed_mapping(values, {"deployment", "expansion"}, "expansion job")
    deployment = _path(job.get("deployment"), path.parent, "deployment")
    binding, _ = _load_expansion_declaration(job["expansion"], base_dir=path.parent)
    return _decode_job(deployment, binding)


def _execute_job(
    job: ExpansionRunCfg | _RollingRunCfg,
    operation: Literal["preflight", "run", "resume"],
) -> Mapping[str, Any]:
    if isinstance(job, ExpansionRunCfg):
        from .engine import ExpansionEngine

        engine = ExpansionEngine.from_job(job)
        if operation == "preflight":
            return {
                **engine.preflight(),
                "mode": "cohort",
                "num_envs": job.batch_size,
                "supports_resume": True,
            }
        return getattr(engine, operation)()
    if operation == "resume":
        raise ValueError("Rolling expansion requires a fresh output directory")

    from .task_program_jobs import _collect_task_program_expansion, _load_deployment

    if "real2sim" in load_mapping(job.deployment):
        raise ValueError("Real2Sim source profiles require cohort expansion")
    if operation == "preflight":
        from embodichain.lab.sim.cfg import physics_backend_from_cfg

        environment = _load_deployment(job.deployment)
        if environment.task_program is None:
            raise ValueError("rolling expansion requires a configured Task Program")
        settings = job.options["engine"]
        operators = job.options["trajectory"]["operators"]
        return {
            "mode": "rolling",
            "deployment": str(job.deployment),
            "output_dir": str(job.output_dir),
            "physics_backend": physics_backend_from_cfg(
                environment.sim_cfg.physics_cfg
            ),
            "simulation_device": str(environment.sim_cfg.device),
            "workload_count": settings["workload_count"],
            "batch_size": settings["batch_size"],
            "num_envs": job.num_envs,
            "episodes_per_operator": settings["episodes_per_operator"],
            "target_episodes": len(operators) * settings["episodes_per_operator"],
            "operators": operators,
            "supports_resume": False,
        }

    job.output_dir.mkdir(parents=True)
    digest = hashlib.sha256(
        json.dumps(job.options, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    return _collect_task_program_expansion(
        job.deployment,
        job.options,
        job.output_dir,
        configuration_sha256=digest,
        seed=job.seed,
    )
