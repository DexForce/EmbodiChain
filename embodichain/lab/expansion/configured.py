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

"""Closed file configuration for offline expert expansion jobs."""

from __future__ import annotations

from dataclasses import fields
import hashlib
import json
import math
from pathlib import Path
from collections.abc import Mapping
from embodichain.utils import configclass

__all__ = [
    "ExpansionRunCfg",
    "load_expansion_job",
    "resolve_expansion_job",
    "preflight_expansion_job",
]


@configclass
class ExpansionRunCfg:
    """File-owned deployment and bounded offline collection settings.

    Sources use the deployment's canonical Task Program; Real2Sim additionally
    requires its qualified tabletop source profile. Backend choice remains owned
    by the deployment's physical environment component.
    """

    deployment: str = ""
    output_dir: str = ""
    source: str = "task_program"
    motion_source: str | None = None
    retry_interrupted: bool = True
    source_profile: str | None = None
    collision_profile: str | None = None
    program: str | None = None
    batch_size: int = 8
    target_episodes: int = 8
    max_attempts: int = 32
    max_steps: int = 2000
    max_wall_time_s: float = 3600.0
    seed: int = 0
    episode_byte_budget: int = 64 * 1024 * 1024
    media_buffer_bytes: int = 256 * 1024 * 1024
    capture_media: bool = False
    export_lerobot: bool = False
    visual_profile_file: str | None = None
    visual_profile: str | None = None
    render_profiles: tuple[str, ...] = ()
    observation_independent: bool = False
    motion_profile: str | None = None
    scene_atlas: str | None = None
    settling_steps: int = 5
    mass_scale: dict[str, tuple[float, float]] = {}
    friction: dict[str, tuple[float, float]] = {}
    center_of_mass: dict[str, tuple[tuple[float, float], ...]] = {}
    joint_stiffness_scale: dict[str, tuple[float, float]] = {}
    joint_damping_scale: dict[str, tuple[float, float]] = {}

    def __post_init__(self) -> None:
        for name in ("deployment", "output_dir"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be nonempty")
        for name in (
            "batch_size",
            "target_episodes",
            "max_attempts",
            "max_steps",
            "episode_byte_budget",
            "media_buffer_bytes",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.episode_byte_budget > 120 * 1024 * 1024:
            raise ValueError(
                "episode_byte_budget must leave archive metadata headroom below 128 MiB"
            )
        if (
            type(self.settling_steps) is not int
            or not 0 <= self.settling_steps <= 10000
        ):
            raise ValueError("settling_steps must be an integer in [0, 10000]")
        if (
            isinstance(self.max_wall_time_s, bool)
            or not isinstance(self.max_wall_time_s, (int, float))
            or not math.isfinite(self.max_wall_time_s)
            or self.max_wall_time_s <= 0
        ):
            raise ValueError("max_wall_time_s must be finite and positive")
        if self.max_attempts < self.target_episodes:
            raise ValueError("max_attempts must cover target_episodes")
        if (
            type(self.seed) is not int
            or self.seed < 0
            or type(self.capture_media) is not bool
            or type(self.export_lerobot) is not bool
        ):
            raise ValueError("seed and capture_media have invalid types")
        if self.source not in {"task_program", "real2sim"}:
            raise ValueError("source must be task_program or real2sim")
        if self.motion_source not in {None, "native", "prototype"}:
            raise ValueError("motion_source must be native or prototype")
        if type(self.retry_interrupted) is not bool:
            raise ValueError("retry_interrupted must be bool")
        if type(self.observation_independent) is not bool:
            raise ValueError("observation_independent must be bool")
        if not isinstance(self.render_profiles, (list, tuple)) or any(
            not isinstance(value, str) or not value for value in self.render_profiles
        ):
            raise ValueError("render_profiles must contain profile IDs")
        self.render_profiles = tuple(self.render_profiles)
        if len(set(self.render_profiles)) != len(self.render_profiles):
            raise ValueError("render_profiles must be unique")
        if self.render_profiles and (
            not self.observation_independent or not self.capture_media
        ):
            raise ValueError(
                "state rerendering requires capture_media and explicit observation_independent"
            )
        if (
            self.render_profiles or self.visual_profile is not None
        ) and self.visual_profile_file is None:
            raise ValueError("visual profiles require a registry file")
        for name in (
            "mass_scale",
            "friction",
            "joint_stiffness_scale",
            "joint_damping_scale",
        ):
            values = getattr(self, name)
            if not isinstance(values, Mapping):
                raise ValueError(f"{name} must map body UIDs to ranges")
            normalized = {}
            for uid, bounds in values.items():
                if (
                    not isinstance(uid, str)
                    or not uid
                    or not isinstance(bounds, (list, tuple))
                    or len(bounds) != 2
                ):
                    raise ValueError(f"invalid {name} body range")
                if (
                    any(
                        type(v) not in (int, float) or not math.isfinite(v) or v < 0
                        for v in bounds
                    )
                    or bounds[0] > bounds[1]
                    or (name == "mass_scale" and bounds[0] == 0)
                ):
                    raise ValueError(f"invalid {name} bounds")
                normalized[uid] = tuple(map(float, bounds))
            setattr(self, name, normalized)

        if not isinstance(self.center_of_mass, Mapping):
            raise ValueError("center_of_mass must map body UIDs to three axis ranges")
        for uid, axes in self.center_of_mass.items():
            if (
                not isinstance(uid, str)
                or not uid
                or not isinstance(axes, (list, tuple))
                or len(axes) != 3
                or any(
                    not isinstance(bounds, (list, tuple))
                    or len(bounds) != 2
                    or any(
                        type(v) not in (int, float) or not math.isfinite(v)
                        for v in bounds
                    )
                    or bounds[0] > bounds[1]
                    for bounds in axes
                )
            ):
                raise ValueError(
                    "center_of_mass requires three finite ordered metre ranges per UID"
                )


def resolve_expansion_job(cfg: ExpansionRunCfg) -> ExpansionRunCfg:
    """Resolve motion bindings owned by the selected Task Program deployment.

    Legacy source/profile fields remain accepted. A narrower collection profile
    may select recipes, but cannot replace the deployment's pinned source artifacts.
    """
    from embodichain.utils.utility import load_config
    from embodichain.utils.config_paths import resolve_config_path

    result = ExpansionRunCfg(**cfg.to_dict())
    path = Path(result.deployment).expanduser().resolve()
    declared = load_config(path).get("real2sim", {}).get("profile")
    if declared is not None:
        declared = str(resolve_config_path(declared, base_dir=path.parent).resolve())
    kind = result.motion_source or (
        "prototype"
        if declared or result.source_profile or result.source == "real2sim"
        else "native"
    )
    if kind == "native" and (
        declared or result.source_profile or result.source == "real2sim"
    ):
        raise ValueError("native planning cannot override a pinned prototype source")
    if kind == "prototype":
        from embodichain.lab.task_program.integrations.simulation.real2sim_collection import (
            load_real2sim_tabletop_profile,
        )

        selected = result.source_profile or declared
        if selected is None:
            raise ValueError("prototype motion requires a deployment source profile")
        actual = load_real2sim_tabletop_profile(selected)
        if (
            declared is not None
            and Path(selected).resolve() != Path(declared).resolve()
        ):
            expected = load_real2sim_tabletop_profile(declared)

            def references(profile):
                return {
                    str(call): {
                        name: reference["sha256"] for name, reference in bundle.items()
                    }
                    for call, bundle in profile.source_artifacts.items()
                }

            if references(actual) != references(expected):
                raise ValueError(
                    "collection profile changes deployment-owned source artifacts"
                )
        result.source_profile = str(Path(selected).resolve())
    result.motion_source = kind
    result.source = "real2sim" if kind == "prototype" else "task_program"
    return result


def _semantic_value(value, hashes):
    if isinstance(value, Mapping):
        return {
            str(key): _semantic_value(item, hashes)
            for key, item in value.items()
            if key not in {"output_dir", "save_path"}
        }
    if isinstance(value, (list, tuple)):
        return [_semantic_value(item, hashes) for item in value]
    if isinstance(value, str) and value.startswith("/"):
        path = Path(value)
        if path.is_file():
            if value not in hashes:
                hashes[value] = hashlib.sha256(path.read_bytes()).hexdigest()
            return {"file_sha256": hashes[value]}
    return value


def load_expansion_job(path: str | Path) -> ExpansionRunCfg:
    """Read a closed YAML/JSON job and resolve paths relative to that file."""
    import yaml

    path = Path(path).expanduser().resolve()
    values = yaml.safe_load(path.read_text())
    if not isinstance(values, dict) or set(values) - {
        field.name for field in fields(ExpansionRunCfg)
    }:
        raise ValueError("expansion job contains unknown fields or is not a mapping")
    values = dict(values)
    for key in (
        "deployment",
        "output_dir",
        "source_profile",
        "collision_profile",
        "program",
        "visual_profile_file",
        "motion_profile",
        "scene_atlas",
    ):
        value = values.get(key)
        if value is not None:
            if not isinstance(value, str):
                raise ValueError(f"{key} must be a path string")
            values[key] = str((path.parent / Path(value).expanduser()).resolve())
    return ExpansionRunCfg(**values)


def preflight_expansion_job(cfg: ExpansionRunCfg) -> dict[str, object]:
    """Resolve the existing deployment and source contracts without simulation.

    Args:
        cfg: Explicit expansion job configuration.
    Returns:
        Pinned job identity, physical backend, required capabilities and budgets.
    """
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg
    from embodichain.utils.utility import load_config

    cfg = resolve_expansion_job(cfg)
    cfg.__post_init__()
    path = Path(cfg.deployment)
    environment = config_to_cfg(
        load_config(path), source_path=path, task_program_path_override=cfg.program
    )
    from embodichain.lab.sim.cfg import physics_backend_from_cfg

    backend = physics_backend_from_cfg(environment.sim_cfg.physics_cfg)

    def configured_terms(value):
        if value is None:
            return False
        values = value if isinstance(value, Mapping) else vars(value)
        return any(
            term is not None
            for name, term in values.items()
            if not name.startswith("_")
        )

    if configured_terms(environment.events):
        raise ValueError(
            "native expansion requires explicit scene/physics operators; deployment event functors need a custom host adapter"
        )
    if cfg.visual_profile is not None and not cfg.capture_media:
        raise ValueError("visual_profile requires native media capture")
    if cfg.export_lerobot and configured_terms(environment.observations):
        raise ValueError(
            "native expansion export requires the built-in observation schema; custom observation functors need a sink adapter"
        )
    sensor_cfgs = list(environment.sensor) + list(
        getattr(environment.robot, "sensors", ())
    )
    frame_bytes = 0
    if cfg.capture_media:
        if not sensor_cfgs:
            raise ValueError("capture_media requires a configured native camera")
        for sensor in sensor_cfgs:
            if sensor.sensor_type not in {"Camera", "StereoCamera"}:
                raise ValueError(
                    f"native media collection has no adapter for sensor {sensor.sensor_type!r}"
                )
            fields = sensor.get_data_types()
            widths = {
                "color": 4,
                "rgb": 3,
                "depth": 4,
                "mask": 4,
                "normal": 12,
                "position": 12,
                "disparity": 4,
            }
            frame_bytes += (
                sensor.width
                * sensor.height
                * sum(widths[name.split("_", 1)[0]] for name in fields)
                + 512
            )
        if frame_bytes * cfg.batch_size > cfg.media_buffer_bytes:
            raise ValueError(
                "media_buffer_bytes must hold at least one calibrated frame per environment"
            )
    if environment.task_program is None:
        raise ValueError("expansion deployment requires a Task Program")
    if cfg.source == "task_program" and cfg.collision_profile is None:
        raise ValueError(
            "task_program collection requires an explicit collision_profile"
        )
    if cfg.collision_profile is not None:
        from embodichain.lab.task_program.integrations.feasibility import (
            build_feasibility_host,
        )

        build_feasibility_host(
            load_config(cfg.collision_profile),
            config_directory=Path(cfg.collision_profile).parent,
            output_directory=Path(cfg.output_dir) / "collision",
        )
    if cfg.source == "real2sim":
        from embodichain.lab.task_program.integrations.simulation.real2sim_collection import (
            load_real2sim_tabletop_profile,
        )

        load_real2sim_tabletop_profile(cfg.source_profile)
    if cfg.visual_profile_file is not None:
        from .visual import load_visual_profiles

        profiles = load_visual_profiles(cfg.visual_profile_file)
        for name in (
            *cfg.render_profiles,
            *((cfg.visual_profile,) if cfg.visual_profile else ()),
        ):
            if name not in profiles:
                raise ValueError(f"unknown visual profile {name!r}")
    if cfg.scene_atlas is not None:
        if cfg.source != "task_program":
            raise ValueError(
                "Real2Sim layout changes require a source-qualified layout profile"
            )
        from .scene import JointSceneSource

        JointSceneSource(cfg.scene_atlas)
    if cfg.motion_profile is not None:
        if cfg.source != "task_program":
            raise ValueError("Real2Sim motion recipes are owned by source_profile")
        from embodichain.lab.sim.motion.expansion.profile import load_expansion_profile

        motion = load_expansion_profile(cfg.motion_profile)
        if (
            motion.scene_randomization.enabled
            or motion.visual.enabled
            or motion.trajectory.timing.enabled
        ):
            raise ValueError(
                "motion_profile owns affordance/spatial changes; use scene_atlas, visual profiles and source-specific mapped retiming for other domains"
            )
    payload = cfg.to_dict()
    payload.pop("output_dir")
    resolved = environment.to_dict()
    file_fields = (
        "deployment",
        "program",
        "source_profile",
        "collision_profile",
        "visual_profile_file",
        "motion_profile",
        "scene_atlas",
    )
    files = {
        name: hashlib.sha256(Path(getattr(cfg, name)).read_bytes()).hexdigest()
        for name in file_fields
        if getattr(cfg, name) is not None
    }
    execution_fields = {
        "batch_size",
        "target_episodes",
        "max_attempts",
        "max_steps",
        "max_wall_time_s",
        "episode_byte_budget",
        "media_buffer_bytes",
        "export_lerobot",
        "render_profiles",
        "retry_interrupted",
    }
    semantic = {
        key: value
        for key, value in payload.items()
        if key not in execution_fields and key not in file_fields and key != "source"
    }
    semantic_files = {
        name: checksum
        for name, checksum in files.items()
        if name != "visual_profile_file"
    }
    if cfg.visual_profile:
        semantic["visual_profile_content"] = profiles[cfg.visual_profile]
    semantic_payload = {
        "job": semantic,
        "deployment": _semantic_value(resolved, {}),
        "files": semantic_files,
    }
    # Top-level locator paths and resource budgets do not define physical recipes.
    encoded = json.dumps(semantic_payload, sort_keys=True, default=str, allow_nan=False)
    recipe_revision = hashlib.sha256(encoded.encode()).hexdigest()
    from importlib.metadata import version, PackageNotFoundError
    import platform

    runtime_versions = {"python": platform.python_version()}
    for package in ("embodichain", "dexsim", "newton", "torch"):
        try:
            runtime_versions[package] = version(package)
        except PackageNotFoundError:
            runtime_versions[package] = "uninstalled"
    postprocessing_revision = (
        hashlib.sha256(
            json.dumps(
                {name: profiles[name] for name in cfg.render_profiles},
                sort_keys=True,
                allow_nan=False,
            ).encode()
        ).hexdigest()
        if cfg.render_profiles
        else None
    )
    execution_id = hashlib.sha256(
        json.dumps(
            {
                "resources": {key: payload[key] for key in execution_fields},
                "postprocessing_revision": postprocessing_revision,
                "runtime_versions": runtime_versions,
            },
            sort_keys=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    return {
        "job_id": recipe_revision,
        "semantic_id": recipe_revision,
        "execution_id": execution_id,
        "runtime_versions": runtime_versions,
        "postprocessing_revision": postprocessing_revision,
        "render_profile_snapshots": {
            name: {
                "revision": hashlib.sha256(
                    json.dumps(
                        profiles[name],
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode()
                ).hexdigest(),
                "profile": profiles[name],
            }
            for name in cfg.render_profiles
        },
        "task_runtime": "task_program",
        "motion_source": cfg.motion_source,
        "recipe_revision": recipe_revision,
        "input_files": files,
        "resolved_deployment": json.loads(
            json.dumps(resolved, default=str, allow_nan=False)
        ),
        "source": cfg.source,
        "physics_backend": backend,
        "capabilities": {
            "batch_execution": True,
            "rigid_parameters": ["mass", "inertia", "friction", "center_of_mass"],
            "joint_parameters": ["stiffness", "damping"],
            "single_body_damping": False,
            "media": ["rgb", "depth", "mask", "normal", "position"],
            "state_rerender": "complete-state and observation-independent only",
        },
        "grouping": [
            "one fixed deployment topology",
            "one world visual profile",
            "one backend and control clock",
        ],
        "raw_media_frame_bytes": frame_bytes * cfg.batch_size,
        "batch_size": cfg.batch_size,
        "target_episodes": cfg.target_episodes,
        "max_attempts": cfg.max_attempts,
        "capture_media": cfg.capture_media,
        "pending_numeric_bytes": cfg.batch_size * cfg.episode_byte_budget,
        "media_buffer_bytes": cfg.media_buffer_bytes,
        "deployment": str(path),
        "output_dir": cfg.output_dir,
    }
